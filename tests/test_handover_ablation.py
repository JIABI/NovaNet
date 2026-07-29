from dataclasses import replace

import numpy as np
import torch

from novanet.config import load_config
from novanet.handover import handover_failure_matrix
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


def test_transvel_ablation_removes_an_independent_angular_cost():
    cfg = load_config()
    full = NovaNet(cfg)
    without_velocity = NovaNet(cfg, ablations=("TransVel",))
    without_velocity.load_state_dict(full.state_dict())

    ttl = torch.full((1, cfg.planner.horizon_steps, 3), 200.0)
    omega = torch.tensor(
        [[[0.0, 0.2, 0.8]]] * cfg.planner.horizon_steps,
        dtype=torch.float32,
    )
    hof = torch.zeros((1, cfg.planner.horizon_steps, 3, 3))
    full_cost, full_components = full.energy.transition_cost(ttl, omega, hof)
    ablated_cost, _ = without_velocity.energy.transition_cost(ttl, omega, hof)

    diagonal = full_cost.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal))
    assert full_components["angular_speed"][0, 0, 0, 2] > 0
    assert torch.all(full_cost >= ablated_cost)
    assert torch.any(full_cost > ablated_cost)


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
