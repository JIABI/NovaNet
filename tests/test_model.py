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
        "angular_speed_deg_s": torch.rand(batch, horizon, candidates),
        "snr_target_db": torch.randn(batch, horizon, candidates) + 8.0,
        "snr_mask": valid,
        "ttl_target_s": torch.rand(batch, horizon, candidates) * 500.0,
        "ttl_mask": valid,
        "hof_target": torch.randint(
            0, 2, (batch, horizon, candidates, candidates)
        ).float(),
        "hof_mask": (
            ~torch.eye(candidates, dtype=torch.bool)
        ).reshape(1, 1, candidates, candidates).expand(
            batch, horizon, candidates, candidates
        ),
        "selection_target": torch.randint(
            0, candidates, (batch, horizon)
        ),
    }


def test_hof_head_loss_and_transition_terms_are_closed():
    cfg, batch = make_batch()
    model = NovaNet(cfg)
    rate = (
        cfg.channel.implementation_efficiency
        * cfg.channel.bandwidth_hz
        * torch.log2(1.0 + 10.0 ** (batch["snr_target_db"] / 10.0))
        / 1e6
    )
    model.energy.normalizer.fit(
        rate,
        batch["ttl_target_s"],
        batch["angular_speed_deg_s"],
        batch["valid_mask"],
    )
    outputs = model(
        batch["node_features"],
        batch["spatial_adjacency"],
        batch["valid_mask"],
        batch["current_idx"],
        batch["angular_speed_deg_s"],
    )
    assert outputs["hof_logits"].shape == batch["hof_target"].shape
    assert torch.allclose(
        outputs["transition_cost"].diagonal(dim1=-2, dim2=-1),
        torch.zeros_like(
            outputs["transition_cost"].diagonal(dim1=-2, dim2=-1)
        ),
    )
    assert "angular_speed" in outputs["transition_components"]
    loss, components = compute_training_loss(outputs, batch, model, 1.0)
    loss.backward()
    assert components["hof"] > 0
    assert model.hof_head[-1].weight.grad.abs().sum() > 0

