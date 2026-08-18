#!/usr/bin/env python3
"""Train source-conditioned learned comparison models.

GNN-only is a one-step realized-cost regressor.  DQN+GNN is trained by offline
sequential fitted Q iteration with a replay buffer, Bellman TD targets, and a
separate target network.  The local DHO-shaped MLP is explicitly a surrogate,
not a reproduction of the interaction-trained DHO protocol cited by the paper;
its checkpoint is therefore ineligible for manuscript tables by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from novanet.baselines import (
    BASELINE_KINDS,
    BASELINE_TRAINING_PROTOCOLS,
    DQN_CANONICAL_DISCOUNT,
    DQN_CANONICAL_TARGET_UPDATE_EPOCHS,
    DQN_CANONICAL_VALIDATION_TRANSITIONS,
    make_baseline_model,
)
from novanet.config import load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)
from train_oaest import move_batch, set_seed


def realized_one_step_scores(batch, config, current_idx=None):
    """Higher-is-better realized scores for the current source--target pairs."""

    snr_db = batch["nominal_snr_db"][:, 0] + (
        10.0 / math.log(10.0)
    ) * batch["residual_target"][:, 0]
    rate_mbps = (
        config.channel.implementation_efficiency
        * config.channel.bandwidth_hz
        * torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))
        / 1e6
    )
    state_cost = (
        -config.planner.alpha
        * rate_mbps
        / config.planner.rate_reference_mbps
        - config.planner.beta
        * batch["ttl_s"][:, 0]
        / config.planner.ttl_reference_s
    )
    batch_index = torch.arange(
        state_cost.shape[0], device=state_cost.device
    )
    current = batch["current_idx"] if current_idx is None else current_idx
    source_ttl = batch["ttl_s"][batch_index, 0, current]
    hof = batch["hof_target"][batch_index, 0, current]
    candidates = state_cost.shape[1]
    switch = 1.0 - F.one_hot(current, candidates).to(state_cost.dtype)
    transition_cost = switch * (
        config.planner.c0
        + config.planner.c1
        * source_ttl[:, None]
        / config.planner.ttl_reference_s
        + config.planner.c2 * hof
    )
    return -(state_cost + transition_cost)


def baseline_loss(model, kind, batch, config):
    if kind == "dqn_gnn":
        raise ValueError(
            "DQN+GNN must use dqn_td_loss with replay and a target network"
        )
    valid = batch["valid_mask"][:, 0]
    loss_sum = valid.new_zeros((), dtype=batch["node_features"].dtype)
    count = 0
    # Counterfactual source enumeration prevents the training distribution from
    # collapsing to the preliminary max-elevation incumbent used to construct a
    # sequence window.
    for source in range(valid.shape[1]):
        current = torch.full_like(batch["current_idx"], source)
        prediction = model(
            batch["node_features"][:, 0],
            batch["spatial_adjacency"][:, 0],
            valid,
            current,
        )
        with torch.no_grad():
            target = realized_one_step_scores(batch, config, current)
        finite = valid & valid[:, source, None] & torch.isfinite(target)
        if torch.any(finite):
            loss_sum = loss_sum + F.smooth_l1_loss(
                prediction[finite], target[finite], reduction="sum"
            )
            count += int(finite.sum())
    if count == 0:
        raise RuntimeError("One-step baseline batch has no finite feasible target")
    return loss_sum / count


@dataclass(frozen=True)
class DQNTransition:
    sample_index: int
    horizon_index: int
    source_index: int
    action_index: int
    reward: float
    next_current: int
    next_freeze: int


def _realized_cost_arrays(sample, config):
    snr_db = np.asarray(sample["nominal_snr_db"], dtype=np.float64) + (
        10.0 / math.log(10.0)
    ) * np.asarray(sample["residual_target"], dtype=np.float64)
    rate_mbps = (
        config.channel.implementation_efficiency
        * config.channel.bandwidth_hz
        * np.log2(1.0 + np.power(10.0, snr_db / 10.0))
        / 1e6
    )
    ttl_s = np.asarray(sample["ttl_s"], dtype=np.float64)
    state_cost = (
        -config.planner.alpha
        * rate_mbps
        / config.planner.rate_reference_mbps
        - config.planner.beta
        * ttl_s
        / config.planner.ttl_reference_s
    )
    horizon, candidates = state_cost.shape
    switch = 1.0 - np.eye(candidates, dtype=np.float64)[None]
    transition_cost = switch * (
        config.planner.c0
        + config.planner.c1
        * ttl_s[:, :, None]
        / config.planner.ttl_reference_s
        + config.planner.c2
        * np.asarray(sample["hof_target"], dtype=np.float64)
    )
    if transition_cost.shape != (horizon, candidates, candidates):
        raise ValueError("Invalid transition target shape in DQN replay source")
    return state_cost, transition_cost


class OfflineDQNReplay:
    """Compact replay over counterfactual source--action transitions.

    The generated sequence arrays remain in the dataset and transitions store
    only indices plus the realized one-step reward.  This avoids copying every
    graph once per source--action pair while retaining full replay coverage.
    """

    def __init__(self, samples, config):
        self.samples = list(samples)
        self.config = config
        transitions: list[DQNTransition] = []
        for sample_index, sample in enumerate(self.samples):
            valid = np.asarray(sample["valid_mask"], dtype=bool)
            state_cost, transition_cost = _realized_cost_arrays(sample, config)
            hof_target = np.asarray(sample["hof_target"], dtype=np.float64)
            hof_mask = np.asarray(sample["hof_mask"], dtype=bool)
            horizon, _candidates = valid.shape
            for h in range(horizon):
                visible = np.flatnonzero(valid[h])
                for source in visible:
                    for action in visible:
                        cost = (
                            state_cost[h, action]
                            + transition_cost[h, source, action]
                        )
                        switch = source != action
                        attempted = switch and bool(
                            hof_mask[h, source, action]
                        )
                        succeeded = attempted and not bool(
                            hof_target[h, source, action] >= 0.5
                        )
                        transitions.append(
                            DQNTransition(
                                sample_index=sample_index,
                                horizon_index=h,
                                source_index=int(source),
                                action_index=int(action),
                                reward=-float(cost),
                                next_current=(
                                    int(action) if succeeded else int(source)
                                ),
                                next_freeze=(
                                    config.handover.freeze_steps
                                    if succeeded
                                    else 0
                                ),
                            )
                        )
        if not transitions:
            raise ValueError("DQN replay contains no feasible transition")
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def _materialize(self, indices, device):
        rows = [self.transitions[int(index)] for index in indices]

        def stack(key, *, next_state=False, dtype=torch.float32):
            values = []
            for row in rows:
                sample = self.samples[row.sample_index]
                h = row.horizon_index
                if next_state and h + 1 < len(sample[key]):
                    h += 1
                values.append(np.asarray(sample[key][h]))
            return torch.as_tensor(
                np.stack(values), dtype=dtype, device=device
            )

        node = stack("node_features")
        adjacency = stack("spatial_adjacency")
        valid = stack("valid_mask", dtype=torch.bool)
        next_node = stack("node_features", next_state=True)
        next_adjacency = stack("spatial_adjacency", next_state=True)
        next_valid = stack("valid_mask", next_state=True, dtype=torch.bool)
        # Each replay slot is treated as an actual sequential decision state,
        # not as NovaNet's causal future placeholder.  Populate its current-link
        # field from the realized observation available at that replay epoch.
        for local, row in enumerate(rows):
            sample = self.samples[row.sample_index]
            h = row.horizon_index
            realized_db = np.asarray(sample["nominal_snr_db"])[h] + (
                10.0 / math.log(10.0)
            ) * np.asarray(sample["residual_target"])[h]
            node[local, :, 5] = torch.as_tensor(
                realized_db / self.config.model.sinr_reference_db,
                dtype=node.dtype,
                device=device,
            )
            next_h = min(h + 1, len(sample["valid_mask"]) - 1)
            next_realized_db = np.asarray(sample["nominal_snr_db"])[next_h] + (
                10.0 / math.log(10.0)
            ) * np.asarray(sample["residual_target"])[next_h]
            next_node[local, :, 5] = torch.as_tensor(
                next_realized_db / self.config.model.sinr_reference_db,
                dtype=next_node.dtype,
                device=device,
            )
        current = torch.tensor(
            [row.source_index for row in rows],
            dtype=torch.long,
            device=device,
        )
        action = torch.tensor(
            [row.action_index for row in rows],
            dtype=torch.long,
            device=device,
        )
        next_current = torch.tensor(
            [row.next_current for row in rows],
            dtype=torch.long,
            device=device,
        )
        reward = torch.tensor(
            [row.reward for row in rows], dtype=torch.float32, device=device
        )
        done = torch.tensor(
            [
                row.horizon_index
                == len(self.samples[row.sample_index]["valid_mask"]) - 1
                for row in rows
            ],
            dtype=torch.bool,
            device=device,
        )
        # A no-coverage next slot has no feasible action and terminates this
        # finite replay segment with zero continuation value.
        done = done | ~next_valid.any(dim=-1)
        next_action_mask = next_valid.clone()
        for local, row in enumerate(rows):
            if done[local]:
                continue
            target = row.next_current
            if row.next_freeze > 0 and bool(next_valid[local, target]):
                next_action_mask[local] = False
                next_action_mask[local, target] = True
        if torch.any(~done & ~next_action_mask.any(dim=-1)):
            raise RuntimeError("Nonterminal DQN transition has no feasible next action")
        return {
            "node": node,
            "adjacency": adjacency,
            "valid": valid,
            "current_idx": current,
            "action": action,
            "reward": reward,
            "next_node": next_node,
            "next_adjacency": next_adjacency,
            "next_valid": next_valid,
            "next_current_idx": next_current,
            "next_action_mask": next_action_mask,
            "done": done,
        }

    def sample(self, batch_size, generator, device):
        indices = torch.randint(
            len(self), (batch_size,), generator=generator
        ).tolist()
        return self._materialize(indices, device)

    def validation_batches(self, batch_size, maximum_transitions, device):
        count = min(len(self), maximum_transitions)
        indices = np.linspace(0, len(self) - 1, num=count, dtype=np.int64)
        for start in range(0, count, batch_size):
            yield self._materialize(indices[start : start + batch_size], device)


def dqn_td_loss(model, target_model, batch, discount):
    q_values = model(
        batch["node"],
        batch["adjacency"],
        batch["valid"],
        batch["current_idx"],
    )
    chosen_q = q_values.gather(1, batch["action"][:, None]).squeeze(1)
    with torch.no_grad():
        online_next = model(
            batch["next_node"],
            batch["next_adjacency"],
            batch["next_valid"],
            batch["next_current_idx"],
        ).masked_fill(~batch["next_action_mask"], -torch.inf)
        next_action = online_next.argmax(dim=-1)
        target_next = target_model(
            batch["next_node"],
            batch["next_adjacency"],
            batch["next_valid"],
            batch["next_current_idx"],
        ).gather(1, next_action[:, None]).squeeze(1)
        td_target = batch["reward"] + discount * target_next * (
            ~batch["done"]
        ).to(target_next.dtype)
    return F.smooth_l1_loss(chosen_q, td_target)


def run_dqn_epoch(
    model,
    target_model,
    replay,
    device,
    *,
    batch_size,
    discount,
    optimizer=None,
    gradient_clip=1.0,
    updates=1,
    generator=None,
    validation_transitions=4096,
):
    training = optimizer is not None
    model.train(training)
    target_model.eval()
    total = 0.0
    observations = 0
    if training:
        if generator is None:
            raise ValueError("Training replay requires a seeded generator")
        batches = (
            replay.sample(batch_size, generator, device) for _ in range(updates)
        )
    else:
        batches = replay.validation_batches(
            batch_size, validation_transitions, device
        )
    for batch in batches:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = dqn_td_loss(model, target_model, batch, discount)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DQN TD loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        size = int(batch["action"].shape[0])
        total += float(loss.detach()) * size
        observations += size
    return total / max(observations, 1)


def run_epoch(model, kind, loader, config, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    for raw in loader:
        batch = move_batch(raw, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss = baseline_loss(model, kind, batch, config)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip
                )
                optimizer.step()
        size = int(batch["current_idx"].shape[0])
        total += float(loss.detach()) * size
        count += size
    return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=sorted(BASELINE_KINDS))
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--dqn-discount",
        type=float,
        default=DQN_CANONICAL_DISCOUNT,
        help="Finite-horizon FQI discount (the paper energy is undiscounted).",
    )
    parser.add_argument(
        "--dqn-updates-per-epoch",
        type=int,
        default=None,
        help="Replay updates per epoch; defaults to one update per source sample batch.",
    )
    parser.add_argument(
        "--dqn-target-update-epochs",
        type=int,
        default=DQN_CANONICAL_TARGET_UPDATE_EPOCHS,
    )
    parser.add_argument(
        "--dqn-validation-transitions",
        type=int,
        default=DQN_CANONICAL_VALIDATION_TRANSITIONS,
    )
    parser.add_argument(
        "--allow-surrogate-dho",
        action="store_true",
        help=(
            "Acknowledge that the local DHO-shaped MLP is a diagnostic "
            "surrogate and cannot populate the cited DHO manuscript row."
        ),
    )
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.dqn_discount <= 1.0:
        parser.error("--dqn-discount must be in [0, 1]")
    if args.dqn_target_update_epochs <= 0:
        parser.error("--dqn-target-update-epochs must be positive")
    if args.dqn_validation_transitions <= 0:
        parser.error("--dqn-validation-transitions must be positive")
    if args.kind == "dho" and not args.allow_surrogate_dho:
        parser.error(
            "The repository DHO model is not the cited interaction-trained "
            "protocol; add --allow-surrogate-dho for a diagnostic surrogate run"
        )

    config = load_config(args.config)
    set_seed(config.experiment.seed)
    provenance = validate_tle_epoch(
        config,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    sample_count = args.samples or config.training.num_samples
    samples = generate_sequence_samples(
        config,
        GenerationOptions(
            num_samples=sample_count,
            allow_stale_tle=args.allow_stale_tle,
        ),
    )
    split = int(0.8 * len(samples))
    if split <= 0 or split >= len(samples):
        raise ValueError("Training requires at least two samples")
    train_samples = samples[:split]
    validation_samples = samples[split:]
    generator = torch.Generator().manual_seed(config.experiment.seed)
    train_loader = None
    validation_loader = None
    train_replay = None
    validation_replay = None
    if args.kind == "dqn_gnn":
        train_replay = OfflineDQNReplay(train_samples, config)
        validation_replay = OfflineDQNReplay(validation_samples, config)
    else:
        train_loader = DataLoader(
            NovaNetSequenceDataset(train_samples),
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            NovaNetSequenceDataset(validation_samples),
            batch_size=config.training.batch_size,
            shuffle=False,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_baseline_model(args.kind, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    target_model = None
    if args.kind == "dqn_gnn":
        target_model = make_baseline_model(args.kind, config).to(device)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        for parameter in target_model.parameters():
            parameter.requires_grad_(False)
    default_name = (
        "dho_surrogate_diagnostic.pt"
        if args.kind == "dho"
        else f"{args.kind}_paper.pt"
    )
    output = Path(args.output or f"checkpoints/{default_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    best = float("inf")
    checkpoint_written = False
    requested_epochs = args.epochs or config.training.epochs
    dqn_updates = args.dqn_updates_per_epoch
    if dqn_updates is None:
        dqn_updates = int(
            math.ceil(len(train_samples) / config.training.batch_size)
        )
    if dqn_updates <= 0:
        parser.error("--dqn-updates-per-epoch must be positive")
    canonical_dqn_updates = int(
        math.ceil(len(train_samples) / config.training.batch_size)
    )
    canonical_dqn_protocol = (
        args.kind != "dqn_gnn"
        or (
            math.isclose(args.dqn_discount, DQN_CANONICAL_DISCOUNT)
            and dqn_updates == canonical_dqn_updates
            and args.dqn_target_update_epochs
            == DQN_CANONICAL_TARGET_UPDATE_EPOCHS
            and args.dqn_validation_transitions
            == DQN_CANONICAL_VALIDATION_TRANSITIONS
        )
    )
    for epoch in range(1, requested_epochs + 1):
        if args.kind == "dqn_gnn":
            assert train_replay is not None
            assert validation_replay is not None
            assert target_model is not None
            train_loss = run_dqn_epoch(
                model,
                target_model,
                train_replay,
                device,
                batch_size=config.training.batch_size,
                discount=args.dqn_discount,
                optimizer=optimizer,
                gradient_clip=config.training.gradient_clip,
                updates=dqn_updates,
                generator=generator,
            )
            validation_loss = run_dqn_epoch(
                model,
                target_model,
                validation_replay,
                device,
                batch_size=config.training.batch_size,
                discount=args.dqn_discount,
                validation_transitions=args.dqn_validation_transitions,
            )
        else:
            assert train_loader is not None
            assert validation_loader is not None
            train_loss = run_epoch(
                model, args.kind, train_loader, config, device, optimizer
            )
            validation_loss = run_epoch(
                model, args.kind, validation_loader, config, device
            )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"kind={args.kind} epoch={epoch:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best:
            best = validation_loss
            source_fidelity = (
                "surrogate_not_cited_protocol"
                if args.kind == "dho"
                else "paper_defined_repository_baseline"
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "baseline_kind": args.kind,
                    "training_protocol": BASELINE_TRAINING_PROTOCOLS[args.kind],
                    "source_fidelity": source_fidelity,
                    # Best-so-far files are not complete runs and therefore
                    # cannot qualify a manuscript-table evaluation.
                    "paper_table_eligible": False,
                    "config": asdict(config),
                    "config_fingerprint": config.fingerprint,
                    "tle_provenance": provenance,
                    "epoch": epoch,
                    "training": {
                        "samples": sample_count,
                        "train_samples": len(train_samples),
                        "validation_samples": len(validation_samples),
                        "epochs_requested": requested_epochs,
                        "epochs_completed": epoch,
                        "training_complete": False,
                        "batch_size": config.training.batch_size,
                        "optimizer": "AdamW",
                        "learning_rate": config.training.learning_rate,
                        "weight_decay": config.training.weight_decay,
                        "seed": config.experiment.seed,
                        "allow_stale_tle": bool(args.allow_stale_tle),
                        "dqn_discount": (
                            args.dqn_discount
                            if args.kind == "dqn_gnn"
                            else None
                        ),
                        "dqn_updates_per_epoch": (
                            dqn_updates if args.kind == "dqn_gnn" else None
                        ),
                        "dqn_target_update_epochs": (
                            args.dqn_target_update_epochs
                            if args.kind == "dqn_gnn"
                            else None
                        ),
                        "dqn_validation_transitions": (
                            args.dqn_validation_transitions
                            if args.kind == "dqn_gnn"
                            else None
                        ),
                        "replay_transitions": (
                            len(train_replay)
                            if train_replay is not None
                            else None
                        ),
                    },
                    "validation": {
                        "loss": validation_loss,
                        "selection": "minimum_validation_loss",
                        "replay_transitions_evaluated": (
                            min(
                                len(validation_replay),
                                args.dqn_validation_transitions,
                            )
                            if validation_replay is not None
                            else None
                        ),
                    },
                },
                output,
            )
            checkpoint_written = True
        if (
            args.kind == "dqn_gnn"
            and epoch % args.dqn_target_update_epochs == 0
        ):
            assert target_model is not None
            target_model.load_state_dict(model.state_dict())
    if not checkpoint_written:
        raise RuntimeError(
            "Training completed without a finite validation checkpoint"
        )
    paper_table_eligible = bool(
        args.kind != "dho"
        and not args.allow_stale_tle
        and sample_count == config.training.num_samples
        and requested_epochs == config.training.epochs
        and canonical_dqn_protocol
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["training"]["epochs_completed"] = requested_epochs
    payload["training"]["training_complete"] = True
    payload["paper_table_eligible"] = paper_table_eligible
    finalized_output = output.with_name(f".{output.name}.finalizing")
    torch.save(payload, finalized_output)
    finalized_output.replace(output)
    history = output.with_suffix(".history.csv")
    with history.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not output.exists():
        raise RuntimeError("Training produced no finite validation checkpoint")
    print(f"saved {output} and {history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
