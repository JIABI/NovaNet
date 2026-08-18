from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from novanet.baselines import (
    BASELINE_KINDS,
    BASELINE_TRAINING_PROTOCOLS,
    DQN_CANONICAL_DISCOUNT,
    DQN_CANONICAL_TARGET_UPDATE_EPOCHS,
    DQN_CANONICAL_VALIDATION_TRANSITIONS,
    LearnedBaselinePolicy,
    make_baseline_model,
)
from novanet.config import load_config
from novanet.dataset import validate_tle_epoch
from novanet.forecast import ForecastSequence
from train_learned_baseline import (
    OfflineDQNReplay,
    baseline_loss,
    dqn_td_loss,
)


def _current_graph_batch():
    config = load_config()
    batch = 2
    candidates = config.experiment.candidate_cap
    node = torch.randn(
        batch,
        candidates,
        config.model.node_feature_dim,
    )
    adjacency = torch.eye(candidates).expand(batch, -1, -1).clone()
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    valid[0, -2:] = False
    valid[1, -1] = False
    current_idx = torch.tensor([0, 1], dtype=torch.long)
    return config, node, adjacency, valid, current_idx


@pytest.mark.parametrize("kind", sorted(BASELINE_KINDS))
def test_learned_baseline_forward_shape_and_valid_encoder_mask(kind):
    config, node, adjacency, valid, current_idx = _current_graph_batch()
    model = make_baseline_model(kind, config).eval()
    with torch.inference_mode():
        score = model(node, adjacency, valid, current_idx)
        if hasattr(model, "encoder"):
            hidden = model.encoder(node, adjacency, valid)
            assert torch.count_nonzero(hidden[~valid]) == 0
    assert score.shape == valid.shape
    assert torch.isfinite(score).all()


@pytest.mark.parametrize("kind", ["gnn_only", "dqn_gnn"])
def test_graph_baseline_scores_depend_on_incumbent_source(kind):
    config, node, adjacency, valid, _current_idx = _current_graph_batch()
    valid[:] = True
    adjacency[:] = 1.0 - torch.eye(adjacency.shape[-1])
    model = make_baseline_model(kind, config).eval()
    head = model.score if kind == "gnn_only" else model.advantage
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Make the score depend only on the explicit stay/source channel.  This
        # turns incumbent sensitivity into a deterministic architectural test.
        head[0].weight[0, -1] = 1.0
        head[2].weight[0, 0] = 1.0
        first = model(node, adjacency, valid, torch.tensor([0, 0]))
        second = model(node, adjacency, valid, torch.tensor([1, 1]))
    assert not torch.equal(first, second)
    assert int(first[0].argmax()) == 0
    assert int(second[0].argmax()) == 1


def _forecast_sequence(config, *, initial_freeze=0):
    horizon = config.planner.horizon_steps
    candidates = config.experiment.candidate_cap
    valid = np.ones((horizon, candidates), dtype=bool)
    valid[0, -2:] = False
    node = np.zeros(
        (horizon, candidates, config.model.node_feature_dim),
        dtype=np.float32,
    )
    adjacency = np.broadcast_to(
        np.eye(candidates, dtype=np.float32),
        (horizon, candidates, candidates),
    ).copy()
    return ForecastSequence(
        node_features=node,
        spatial_adjacency=adjacency,
        valid_mask=valid,
        candidate_ids=np.arange(candidates, dtype=np.int64),
        current_idx=1,
        deterministic_snr_db=np.zeros((horizon, candidates), np.float32),
        ttl_s=np.full((horizon, candidates), 60.0, np.float32),
        initial_freeze=initial_freeze,
    )


def _write_checkpoint(
    path,
    kind,
    config,
    *,
    provenance=None,
    source_fidelity=None,
    paper_table_eligible=True,
    include_qualification=True,
):
    model = make_baseline_model(kind, config)
    payload = {
        "model_state": model.state_dict(),
        "baseline_kind": kind,
        "config_fingerprint": config.fingerprint,
        "tle_provenance": provenance
        or validate_tle_epoch(config, maximum_age_days=float("inf")),
    }
    if include_qualification:
        payload.update(
            {
                "training_protocol": BASELINE_TRAINING_PROTOCOLS[kind],
                "source_fidelity": source_fidelity
                or (
                    "verified_cited_protocol"
                    if kind == "dho"
                    else "paper_defined_repository_baseline"
                ),
                "paper_table_eligible": paper_table_eligible,
                "training": {
                    "samples": config.training.num_samples,
                    "train_samples": int(0.8 * config.training.num_samples),
                    "validation_samples": (
                        config.training.num_samples
                        - int(0.8 * config.training.num_samples)
                    ),
                    "epochs_requested": config.training.epochs,
                    "epochs_completed": config.training.epochs,
                    "training_complete": True,
                    "batch_size": config.training.batch_size,
                    "optimizer": "AdamW",
                    "learning_rate": config.training.learning_rate,
                    "weight_decay": config.training.weight_decay,
                    "seed": config.experiment.seed,
                    "allow_stale_tle": False,
                    "dqn_discount": (
                        DQN_CANONICAL_DISCOUNT if kind == "dqn_gnn" else None
                    ),
                    "dqn_updates_per_epoch": (
                        int(
                            np.ceil(
                                int(0.8 * config.training.num_samples)
                                / config.training.batch_size
                            )
                        )
                        if kind == "dqn_gnn"
                        else None
                    ),
                    "dqn_target_update_epochs": (
                        DQN_CANONICAL_TARGET_UPDATE_EPOCHS
                        if kind == "dqn_gnn"
                        else None
                    ),
                    "dqn_validation_transitions": (
                        DQN_CANONICAL_VALIDATION_TRANSITIONS
                        if kind == "dqn_gnn"
                        else None
                    ),
                    "replay_transitions": 1 if kind == "dqn_gnn" else None,
                },
                "validation": {"loss": 0.1},
                "config": asdict(config),
                "epoch": 1,
            }
        )
    torch.save(payload, path)


@pytest.mark.parametrize("kind", sorted(BASELINE_KINDS))
def test_checkpoint_adapter_masks_invalid_candidates_and_honors_freeze(
    tmp_path, kind
):
    config = load_config()
    checkpoint = tmp_path / f"{kind}.pt"
    _write_checkpoint(checkpoint, kind, config)
    policy = LearnedBaselinePolicy(
        config,
        checkpoint,
        expected_kind=kind,
        device="cpu",
    )

    sequence = _forecast_sequence(config)
    scores = policy.scores(sequence)
    assert scores.shape == sequence.valid_mask[0].shape
    assert np.isneginf(scores[~sequence.valid_mask[0]]).all()
    assert sequence.valid_mask[0, policy.choose(sequence)]

    frozen = _forecast_sequence(config, initial_freeze=1)
    frozen_scores = policy.scores(frozen)
    assert frozen_scores[frozen.current_idx] == 0.0
    assert np.isneginf(
        np.delete(frozen_scores, frozen.current_idx)
    ).all()
    assert policy.choose(frozen) == frozen.current_idx


def test_checkpoint_adapter_rejects_kind_config_and_tle_metadata_mismatches(
    tmp_path,
):
    config = load_config()
    checkpoint = tmp_path / "gnn.pt"
    _write_checkpoint(checkpoint, "gnn_only", config)

    with pytest.raises(ValueError, match="Expected"):
        LearnedBaselinePolicy(
            config,
            checkpoint,
            expected_kind="dho",
            device="cpu",
        )

    changed = replace(
        config,
        planner=replace(config.planner, alpha=config.planner.alpha + 0.1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        LearnedBaselinePolicy(changed, checkpoint, device="cpu")

    provenance = validate_tle_epoch(config, maximum_age_days=float("inf"))
    provenance["selected_names_sha256"] = "not-the-selected-constellation"
    bad_provenance = tmp_path / "bad-provenance.pt"
    _write_checkpoint(
        bad_provenance,
        "gnn_only",
        config,
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="TLE provenance mismatch"):
        LearnedBaselinePolicy(config, bad_provenance, device="cpu")


def test_adapter_rejects_unqualified_dho_and_missing_training_metadata(tmp_path):
    config = load_config()
    surrogate = tmp_path / "dho-surrogate.pt"
    _write_checkpoint(
        surrogate,
        "dho",
        config,
        source_fidelity="surrogate_not_cited_protocol",
        paper_table_eligible=False,
    )
    with pytest.raises(ValueError, match="repository surrogate"):
        LearnedBaselinePolicy(config, surrogate, device="cpu")
    diagnostic = LearnedBaselinePolicy(
        config, surrogate, device="cpu", allow_unqualified=True
    )
    assert diagnostic.kind == "dho"

    incomplete = tmp_path / "incomplete.pt"
    _write_checkpoint(
        incomplete,
        "gnn_only",
        config,
        include_qualification=False,
    )
    with pytest.raises(ValueError, match="qualification metadata"):
        LearnedBaselinePolicy(config, incomplete, device="cpu")


def test_adapter_rejects_noncanonical_dqn_training_as_table_eligible(tmp_path):
    config = load_config()
    checkpoint = tmp_path / "dqn-noncanonical.pt"
    _write_checkpoint(checkpoint, "dqn_gnn", config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training"]["dqn_discount"] = 0.0
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="Abbreviated baseline training"):
        LearnedBaselinePolicy(config, checkpoint, device="cpu")


def test_adapter_rejects_interrupted_baseline_checkpoint(tmp_path):
    config = load_config()
    checkpoint = tmp_path / "partial-gnn.pt"
    _write_checkpoint(
        checkpoint,
        "gnn_only",
        config,
        paper_table_eligible=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["training"]["training_complete"] = False
    payload["training"]["epochs_completed"] = 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="incomplete training"):
        LearnedBaselinePolicy(
            config,
            checkpoint,
            device="cpu",
            allow_unqualified=True,
        )


@pytest.mark.parametrize("kind", ["gnn_only", "dho"])
def test_baseline_training_loss_is_finite_with_padded_candidates(kind):
    config = load_config()
    batch_size = 2
    horizon = config.planner.horizon_steps
    candidates = config.experiment.candidate_cap
    valid = torch.ones(batch_size, horizon, candidates, dtype=torch.bool)
    valid[:, :, -2:] = False
    adjacency = torch.eye(candidates).reshape(1, 1, candidates, candidates)
    batch = {
        "node_features": torch.randn(
            batch_size,
            horizon,
            candidates,
            config.model.node_feature_dim,
        ),
        "spatial_adjacency": adjacency.expand(
            batch_size, horizon, candidates, candidates
        ).clone(),
        "valid_mask": valid,
        "current_idx": torch.tensor([0, 1], dtype=torch.long),
        "initial_freeze": torch.zeros(batch_size, dtype=torch.long),
        "ttl_s": torch.rand(batch_size, horizon, candidates) * 300.0,
        "nominal_snr_db": torch.randn(batch_size, horizon, candidates) + 8.0,
        "residual_target": torch.randn(
            batch_size, horizon, candidates
        ) * 0.05,
        "hof_target": torch.zeros(
            batch_size, horizon, candidates, candidates
        ),
        "hof_mask": torch.zeros(
            batch_size, horizon, candidates, candidates, dtype=torch.bool
        ),
    }
    model = make_baseline_model(kind, config)
    loss = baseline_loss(model, kind, batch, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dqn_uses_sequential_replay_td_target_and_frozen_target_network():
    config = load_config()
    batch_size = 1
    horizon = config.planner.horizon_steps
    candidates = config.experiment.candidate_cap
    valid = torch.ones(batch_size, horizon, candidates, dtype=torch.bool)
    valid[:, :, -2:] = False
    adjacency = torch.ones(batch_size, horizon, candidates, candidates)
    adjacency -= torch.eye(candidates).reshape(1, 1, candidates, candidates)
    source = {
        "node_features": torch.randn(
            batch_size,
            horizon,
            candidates,
            config.model.node_feature_dim,
        ),
        "spatial_adjacency": adjacency,
        "valid_mask": valid,
        "current_idx": torch.tensor([0]),
        "initial_freeze": torch.zeros(batch_size, dtype=torch.long),
        "ttl_s": torch.rand(batch_size, horizon, candidates) * 300.0,
        "nominal_snr_db": torch.randn(batch_size, horizon, candidates) + 8.0,
        "residual_target": torch.randn(batch_size, horizon, candidates) * 0.05,
        "hof_target": torch.zeros(
            batch_size, horizon, candidates, candidates
        ),
        "hof_mask": torch.zeros(
            batch_size,
            horizon,
            candidates,
            candidates,
            dtype=torch.bool,
        ),
    }
    sample = {
        key: value[0].numpy()
        for key, value in source.items()
        if key not in {"current_idx", "initial_freeze"}
    }
    sample["current_idx"] = 0
    sample["initial_freeze"] = 0
    # At h=0, 0->1 succeeds, 0->2 is attempted but fails, and 0->3 is
    # configured without ever entering execution.
    sample["hof_mask"][0, 0, 1] = True
    sample["hof_mask"][0, 0, 2] = True
    sample["hof_target"][0, 0, 2] = 1.0
    replay = OfflineDQNReplay([sample], config)
    assert len(replay) > horizon
    assert any(row.horizon_index > 0 for row in replay.transitions)
    transitions = {
        (row.source_index, row.action_index): row
        for row in replay.transitions
        if row.horizon_index == 0
    }
    assert transitions[(0, 1)].next_current == 1
    assert transitions[(0, 1)].next_freeze == config.handover.freeze_steps
    assert transitions[(0, 2)].next_current == 0
    assert transitions[(0, 2)].next_freeze == 0
    assert transitions[(0, 3)].next_current == 0
    assert transitions[(0, 3)].next_freeze == 0

    online = make_baseline_model("dqn_gnn", config)
    target = make_baseline_model("dqn_gnn", config)
    target.load_state_dict(online.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    replay_batch = replay.sample(
        16, torch.Generator().manual_seed(7), torch.device("cpu")
    )
    loss = dqn_td_loss(online, target, replay_batch, discount=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in online.parameters()
    )
    assert all(parameter.grad is None for parameter in target.parameters())
