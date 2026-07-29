"""Closed training objective including SNR, TTL, HOF, path, and selection."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import NovaNetConfig
from .soft_dp import soft_dynamic_program


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def oracle_teacher_distribution(
    batch: dict[str, torch.Tensor],
    model,
) -> torch.Tensor:
    cfg: NovaNetConfig = model.config
    snr = batch["snr_target_db"]
    valid = batch["valid_mask"]
    rate = (
        cfg.channel.implementation_efficiency
        * cfg.channel.bandwidth_hz
        * torch.log2(1.0 + torch.pow(10.0, snr / 10.0))
        / 1e6
    )
    z_rate = model.energy.normalizer.z_rate(rate)
    z_ttl = model.energy.normalizer.z_ttl(batch["ttl_target_s"])
    state = -cfg.planner.rate_weight * z_rate - cfg.planner.dwell_weight * z_ttl

    batch_size, horizon, candidates = snr.shape
    switch = 1.0 - torch.eye(
        candidates, dtype=snr.dtype, device=snr.device
    )[None, None]
    retained = torch.sigmoid(z_ttl[:, :, :, None])
    angular = batch["angular_speed_deg_s"]
    omega_source = angular[:, :, :, None]
    omega_target = angular[:, :, None, :]
    z_omega = model.energy.normalizer.z_angular_speed(
        (omega_target - omega_source).abs()
    )
    transition = switch * (
        cfg.planner.base_switch_cost
        + cfg.planner.retained_dwell_weight * retained
        + cfg.planner.angular_speed_weight * torch.sigmoid(z_omega)
        + cfg.planner.hof_weight * batch["hof_target"]
    )
    return soft_dynamic_program(
        state,
        transition,
        batch["current_idx"],
        valid,
        temperature=cfg.planner.temperature,
    ).first_action.detach()


def compute_training_loss(
    outputs: dict,
    batch: dict[str, torch.Tensor],
    model,
    handover_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = model.config.training
    variance = torch.exp(outputs["snr_logvar"]).clamp(1e-4, 1e4)
    nll = 0.5 * (
        (batch["snr_target_db"] - outputs["snr_mu"]) ** 2 / variance
        + torch.log(variance)
    )
    loss_nll = masked_mean(nll, batch["snr_mask"])
    loss_ttl = masked_mean(
        F.smooth_l1_loss(
            outputs["ttl_s"],
            batch["ttl_target_s"],
            reduction="none",
            beta=5.0,
        ),
        batch["ttl_mask"],
    )
    loss_hof = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["hof_logits"],
            batch["hof_target"],
            reduction="none",
        ),
        batch["hof_mask"],
    )

    teacher = oracle_teacher_distribution(batch, model)
    predicted = outputs["q_next"].clamp_min(1e-9)
    loss_path = torch.sum(
        teacher * (teacher.clamp_min(1e-9).log() - predicted.log()), dim=-1
    ).mean()

    selection_logits = outputs["selection_logits"]
    batch_size, horizon, candidates = selection_logits.shape
    loss_selection = F.cross_entropy(
        selection_logits.reshape(batch_size * horizon, candidates),
        batch["selection_target"].reshape(batch_size * horizon),
    )
    incumbent_probability = outputs["q_next"].gather(
        1, batch["current_idx"][:, None]
    ).squeeze(1)
    loss_handover = (1.0 - incumbent_probability).mean()
    loss_entropy = (
        -outputs["q_next"]
        * outputs["q_next"].clamp_min(1e-9).log()
    ).sum(dim=-1).mean()

    total = (
        cfg.snr_nll_weight * loss_nll
        + cfg.ttl_weight * loss_ttl
        + cfg.hof_weight * loss_hof
        + cfg.path_weight * loss_path
        + cfg.selection_weight * loss_selection
        + float(handover_weight) * loss_handover
        + cfg.entropy_weight * loss_entropy
    )
    components = {
        "total": total,
        "snr_nll": loss_nll,
        "ttl": loss_ttl,
        "hof": loss_hof,
        "path": loss_path,
        "selection": loss_selection,
        "handover": loss_handover,
        "entropy": loss_entropy,
    }
    return total, components
