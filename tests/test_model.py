import torch

from novanet.config import load_config
from novanet.losses import compute_training_loss
from novanet.model import NovaNet


def make_batch():
    cfg = load_config()
    batch, horizon, candidates, features = (
        2,
        cfg.planner.horizon_steps,
        cfg.experiment.candidate_cap,
        cfg.model.node_feature_dim,
    )
    valid = torch.ones(batch, horizon, candidates, dtype=torch.bool)
    eye = torch.eye(candidates).reshape(1, 1, candidates, candidates)
    return cfg, {
        "node_features": torch.randn(batch, horizon, candidates, features),
        "spatial_adjacency": eye.expand(
            batch, horizon, candidates, candidates
        ).clone(),
        "valid_mask": valid,
        "current_idx": torch.tensor([0, 1]),
        "ttl_s": torch.rand(batch, horizon, candidates) * 500.0,
        "nominal_snr_db": torch.randn(batch, horizon, candidates) + 8.0,
        "residual_target": torch.randn(batch, horizon, candidates) * 0.1,
        "residual_mask": valid,
        "hof_target": torch.randint(
            0, 2, (batch, horizon, candidates, candidates)
        ).float(),
        "hof_mask": (
            ~torch.eye(candidates, dtype=torch.bool)
        ).reshape(1, 1, candidates, candidates).expand(
            batch, horizon, candidates, candidates
        ),
    }


def test_hof_head_loss_and_transition_terms_are_closed():
    cfg, batch = make_batch()
    model = NovaNet(cfg)
    outputs = model(
        batch["node_features"],
        batch["spatial_adjacency"],
        batch["valid_mask"],
        batch["current_idx"],
        batch["ttl_s"],
        batch["nominal_snr_db"],
    )
    assert outputs["hof_logits"].shape == batch["hof_target"].shape
    assert torch.allclose(
        outputs["transition_cost"].diagonal(dim1=-2, dim2=-1),
        torch.zeros_like(
            outputs["transition_cost"].diagonal(dim1=-2, dim2=-1)
        ),
    )
    assert "angular_speed" not in outputs["transition_components"]
    assert torch.equal(outputs["ttl_s"], batch["ttl_s"])
    assert not hasattr(model, "ttl_head")
    loss, components = compute_training_loss(outputs, batch, model)
    loss.backward()
    assert components["hof"] > 0
    assert model.hof_head[-1].weight.grad.abs().sum() > 0


def test_current_rate_bypasses_residual_lcb_but_future_rate_uses_it():
    cfg, batch = make_batch()
    model = NovaNet(cfg).eval()
    current_snr = batch["node_features"][:, 0, :, 5] * (
        cfg.model.sinr_reference_db
    )
    nominal = batch["nominal_snr_db"]
    ttl = batch["ttl_s"]
    residual_mu_a = torch.zeros_like(nominal)
    residual_mu_b = residual_mu_a.clone()
    residual_mu_b[:, 0, :] = 4.0
    residual_mu_b[:, 1:, :] = 1.0
    residual_sigma = torch.full_like(nominal, 0.01)

    _, components_a = model.energy.state_cost(
        nominal, residual_mu_a, residual_sigma, ttl, current_snr
    )
    _, components_b = model.energy.state_cost(
        nominal, residual_mu_b, residual_sigma, ttl, current_snr
    )
    assert torch.allclose(
        components_a["lcb_rate_mbps"][:, 0],
        components_b["lcb_rate_mbps"][:, 0],
    )
    assert torch.allclose(
        components_a["lcb_rate_mbps"][:, 0],
        model.energy.rate_from_snr_db(current_snr),
    )
    assert not torch.allclose(
        components_a["lcb_rate_mbps"][:, 1:],
        components_b["lcb_rate_mbps"][:, 1:],
    )


def test_future_encoder_uses_only_geometry_dwell_and_freeze_reaches_dp():
    cfg, batch = make_batch()
    model = NovaNet(cfg).eval()
    changed = batch["node_features"].clone()
    changed[:, 1:, :, 5] += 1000.0
    with torch.no_grad():
        first = model(
            batch["node_features"],
            batch["spatial_adjacency"],
            batch["valid_mask"],
            batch["current_idx"],
            batch["ttl_s"],
            batch["nominal_snr_db"],
            initial_freeze=torch.ones(2, dtype=torch.long),
        )
        second = model(
            changed,
            batch["spatial_adjacency"],
            batch["valid_mask"],
            batch["current_idx"],
            batch["ttl_s"],
            batch["nominal_snr_db"],
            initial_freeze=torch.ones(2, dtype=torch.long),
        )
    assert torch.allclose(first["hidden"], second["hidden"])
    assert torch.all(first["residual_sigma"] > 0.0)
    for row, incumbent in zip(first["q_next"], batch["current_idx"]):
        assert int(row.argmax()) == int(incumbent)
        assert torch.count_nonzero(row) == 1


def test_declared_parameter_count_is_exact():
    cfg, _batch = make_batch()
    model = NovaNet(cfg)
    assert sum(parameter.numel() for parameter in model.parameters()) == 249_603


def test_invisible_forced_recovery_source_keeps_cached_identity_state():
    cfg, batch = make_batch()
    model = NovaNet(cfg).eval()
    batch["valid_mask"][:, 0, 0] = False
    cached = torch.randn(2, cfg.experiment.candidate_cap, cfg.model.hidden_dim)
    with torch.no_grad():
        hidden = model._encode(
            batch["node_features"],
            batch["spatial_adjacency"],
            batch["valid_mask"],
            cached,
        )
    assert torch.allclose(hidden[:, 0, 0], cached[:, 0])


def test_planner_ablation_still_enforces_actual_freeze_on_deployed_costs():
    cfg, batch = make_batch()
    model = NovaNet(cfg, ablations=("Planner",)).eval()
    with torch.no_grad():
        output = model(
            batch["node_features"],
            batch["spatial_adjacency"],
            batch["valid_mask"],
            batch["current_idx"],
            batch["ttl_s"],
            batch["nominal_snr_db"],
            initial_freeze=torch.ones(2, dtype=torch.long),
        )
    for cost, incumbent in zip(
        output["first_step_cost"], batch["current_idx"]
    ):
        index = int(incumbent)
        other = torch.cat((cost[:index], cost[index + 1 :]))
        assert torch.isfinite(cost[index])
        assert torch.isinf(other).all()
