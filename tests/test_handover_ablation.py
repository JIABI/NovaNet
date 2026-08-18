from dataclasses import replace

import numpy as np
import torch

from novanet.config import load_config
from novanet.handover import (
    counterfactual_hof_label,
    evaluate_cho_attempt,
    handover_failure_matrix,
)
from novanet.model import NovaNet


def test_hof_labels_follow_ttt_hysteresis_and_execution_grid():
    cfg = load_config()
    handover = replace(
        cfg.handover,
        ttt_s=0.1,
        execution_s=0.15,
        hysteresis_db=3.0,
        failure_outage_fraction=0.2,
    )
    # 0 -> 1 sustains a 5 dB margin and remains above outage; the reverse
    # transition is not attempted and is therefore masked from HOF training.
    labels, mask = handover_failure_matrix(
        np.asarray([0.0, 5.0]),
        np.asarray([0.0, 0.0]),
        handover,
        outage_threshold_db=-5.0,
        event_step_s=0.01,
    )
    assert mask[0, 1] and not mask[1, 0]
    assert not mask[0, 0] and not mask[1, 1]
    assert labels[0, 1] == 0.0
    assert labels[1, 0] == 0.0

    # The target starts usable but drops below the outage threshold during
    # execution, so the pair receives a positive failure label.
    fading, _ = handover_failure_matrix(
        np.asarray([-20.0, 0.0]),
        np.asarray([0.0, -30.0]),
        handover,
        outage_threshold_db=-5.0,
        event_step_s=0.01,
    )
    assert fading[0, 1] == 1.0


def test_transition_cost_matches_eq17_and_has_no_angular_term():
    cfg = load_config()
    full = NovaNet(cfg)

    ttl = torch.full((1, cfg.planner.horizon_steps, 3), 200.0)
    hof = torch.zeros((1, cfg.planner.horizon_steps, 3, 3))
    full_cost, full_components = full.energy.transition_cost(ttl, hof)

    diagonal = full_cost.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal))
    expected = cfg.planner.c0 + cfg.planner.c1 * (
        200.0 / cfg.planner.ttl_reference_s
    )
    assert torch.isclose(full_cost[0, 0, 0, 2], torch.tensor(expected))
    assert "angular_speed" not in full_components


def test_learned_adjacency_gate_has_a_gradient_and_dynadj_bypasses_it():
    cfg = load_config()
    full = NovaNet(cfg)
    prior_only = NovaNet(cfg, ablations=("DynAdj",))
    prior_only.load_state_dict(full.state_dict())
    batch, horizon, candidates = 1, cfg.planner.horizon_steps, 4
    features = torch.randn(
        batch,
        horizon,
        candidates,
        cfg.model.node_feature_dim,
    )
    adjacency = torch.ones(batch, horizon, candidates, candidates)
    valid = torch.ones(batch, horizon, candidates, dtype=torch.bool)

    encoded = full._encode(features, adjacency, valid)
    encoded.square().mean().backward()
    assert full.adjacency_query.weight.grad is not None
    assert full.adjacency_query.weight.grad.abs().sum() > 0

    with torch.no_grad():
        gated = full._encode(features, adjacency, valid)
        ungated = prior_only._encode(features, adjacency, valid)
    assert not torch.allclose(gated, ungated)


def test_hof_fraction_uses_exactly_fifteen_execution_samples():
    cfg = load_config()
    below_seven = lambda time: -1.0 if time < 0.175 else 1.0
    below_eight = lambda time: -1.0 if time < 0.185 else 1.0
    assert not counterfactual_hof_label(
        below_seven,
        cfg.handover,
        cfg.channel.outage_threshold_db,
        event_step_s=cfg.handover.event_step_s,
    )
    assert counterfactual_hof_label(
        below_eight,
        cfg.handover,
        cfg.channel.outage_threshold_db,
        event_step_s=cfg.handover.event_step_s,
    )


def test_cho_timer_can_start_later_in_the_decision_interval():
    cfg = load_config()
    source = lambda _time_s: 0.0
    target = lambda time_s: 0.0 if time_s < 2.0 else 4.0
    outcome = evaluate_cho_attempt(
        source,
        target,
        cfg.handover,
        cfg.channel.outage_threshold_db,
        event_step_s=cfg.handover.event_step_s,
        monitoring_horizon_s=cfg.experiment.decision_interval_s,
    )
    assert outcome.attempted
    assert outcome.success
    assert np.isclose(outcome.execution_start_time_s, 2.1)
    assert np.isclose(outcome.completion_time_s, 2.25)


def test_cho_timer_resets_and_never_triggers_before_next_decision():
    cfg = load_config()
    source = lambda _time_s: 0.0
    target = lambda time_s: 4.0 if int(time_s * 100) % 8 < 4 else 0.0
    outcome = evaluate_cho_attempt(
        source,
        target,
        cfg.handover,
        cfg.channel.outage_threshold_db,
        event_step_s=cfg.handover.event_step_s,
        monitoring_horizon_s=cfg.experiment.decision_interval_s,
    )
    assert not outcome.attempted
    assert outcome.execution_start_time_s is None


def test_execution_start_exactly_at_next_decision_is_not_retained():
    cfg = load_config()
    source = lambda _time_s: 0.0
    target = lambda time_s: 4.0 if time_s >= 29.9 else 0.0
    outcome = evaluate_cho_attempt(
        source,
        target,
        cfg.handover,
        cfg.channel.outage_threshold_db,
        event_step_s=cfg.handover.event_step_s,
        monitoring_horizon_s=30.0,
    )
    assert not outcome.attempted


def test_cho_replay_rejects_nonfinite_link_values():
    cfg = load_config()
    with np.testing.assert_raises_regex(ValueError, "non-finite SNR"):
        evaluate_cho_attempt(
            lambda _time_s: 0.0,
            lambda _time_s: float("nan"),
            cfg.handover,
            cfg.channel.outage_threshold_db,
            monitoring_horizon_s=cfg.experiment.decision_interval_s,
        )
