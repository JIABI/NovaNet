#!/usr/bin/env python3
"""Train NovaNet with the complete finite-horizon objective."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from novanet.config import load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)
from novanet.losses import compute_training_loss
from novanet.model import NovaNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def fit_energy_statistics(model: NovaNet, loader: DataLoader) -> None:
    rates, ttls, angular, masks = [], [], [], []
    cfg = model.config
    for batch in loader:
        snr = batch["snr_target_db"]
        rate = (
            cfg.channel.implementation_efficiency
            * cfg.channel.bandwidth_hz
            * torch.log2(1.0 + torch.pow(10.0, snr / 10.0))
            / 1e6
        )
        rates.append(rate)
        ttls.append(batch["ttl_target_s"])
        angular.append(batch["angular_speed_deg_s"])
        masks.append(batch["valid_mask"])
    model.energy.normalizer.fit(
        torch.cat(rates),
        torch.cat(ttls),
        torch.cat(angular),
        torch.cat(masks),
    )


def run_epoch(
    model: NovaNet,
    loader: DataLoader,
    device: torch.device,
    handover_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    observations = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                batch["node_features"],
                batch["spatial_adjacency"],
                batch["valid_mask"],
                batch["current_idx"],
                batch["angular_speed_deg_s"],
            )
            loss, components = compute_training_loss(
                outputs, batch, model, handover_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        batch_size = int(batch["current_idx"].shape[0])
        observations += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / max(observations, 1) for name, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; results are not paper-reproducible.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    tle_provenance = validate_tle_epoch(
        cfg,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    sample_count = args.samples or cfg.training.num_samples
    samples = generate_sequence_samples(
        cfg,
        GenerationOptions(
            num_samples=sample_count,
            allow_stale_tle=args.allow_stale_tle,
        ),
    )
    split = int(0.8 * len(samples))
    if split == 0 or split == len(samples):
        raise ValueError("Training requires at least two samples")
    train_set = NovaNetSequenceDataset(samples[:split])
    validation_set = NovaNetSequenceDataset(samples[split:])
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.training.num_workers,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NovaNet(cfg)
    fit_energy_statistics(model, train_loader)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    output = Path(args.output or cfg.training.checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = output.with_suffix(".history.csv")
    metadata_path = output.with_suffix(".metadata.json")
    epochs = args.epochs or cfg.training.epochs
    best_validation = float("inf")
    handover_weight = cfg.training.handover_weight_init
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            handover_weight,
            optimizer,
            cfg.training.gradient_clip,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                handover_weight,
                None,
                cfg.training.gradient_clip,
            )
        observed_switch = train_metrics["handover"]
        handover_weight = float(
            np.clip(
                handover_weight
                + cfg.training.dual_step
                * (observed_switch - cfg.training.target_switch_rate),
                0.0,
                cfg.training.handover_weight_max,
            )
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "handover_weight": handover_weight,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        rows.append(row)
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.5f} "
            f"validation={validation_metrics['total']:.5f} "
            f"switch={observed_switch:.4f} dual={handover_weight:.4f}",
            flush=True,
        )
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "config_fingerprint": cfg.fingerprint,
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "handover_weight": handover_weight,
                    "tle_provenance": tle_provenance,
                },
                output,
            )

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(
            {
                "config_fingerprint": cfg.fingerprint,
                "best_validation_loss": best_validation,
                "samples": sample_count,
                "train_samples": len(train_set),
                "validation_samples": len(validation_set),
                "allow_stale_tle": args.allow_stale_tle,
                "tle_provenance": tle_provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
