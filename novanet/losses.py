"""Residual NLL, pairwise-HOF BCE, path distillation, and L2 regularization."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .config import NovaNetConfig
from .soft_dp import SoftDPResult, soft_dynamic_program


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def oracle_teacher_distribution(
    batch: dict[str, torch.Tensor],
    model,
) -> torch.Tensor:
    result = oracle_teacher_result(batch, model.config)
    return torch.softmax(
        -result.first_cost / model.config.planner.teacher_temperature,
        dim=-1,
    ).detach()


def oracle_teacher_result(
    batch: dict[str, torch.Tensor],
    config: NovaNetConfig,
) -> SoftDPResult:
    """Replay the realized horizon with the same energy and feasibility rules."""

    cfg = config
    snr = batch["nominal_snr_db"] + (
        10.0 / math.log(10.0)
    ) * batch["residual_target"]
    valid = batch["valid_mask"]
    rate = (
        cfg.channel.implementation_efficiency
        * cfg.channel.bandwidth_hz
        * torch.log2(1.0 + torch.pow(10.0, snr / 10.0))
        / 1e6
    )
    normalized_rate = rate / cfg.planner.rate_reference_mbps
    normalized_ttl = batch["ttl_s"] / cfg.planner.ttl_reference_s
    state = (
        -cfg.planner.alpha * normalized_rate
        - cfg.planner.beta * normalized_ttl
    )

    batch_size, horizon, candidates = snr.shape
    switch = 1.0 - torch.eye(
        candidates, dtype=snr.dtype, device=snr.device
    )[None, None]
    retained = normalized_ttl[:, :, :, None]
    transition = switch * (
        cfg.planner.c0
        + cfg.planner.c1 * retained
        + cfg.planner.c2 * batch["hof_target"]
    )
    return soft_dynamic_program(
        state,
        transition,
        batch["current_idx"],
        valid,
        temperature=cfg.planner.teacher_temperature,
        freeze_steps=cfg.handover.freeze_steps,
        initial_freeze=batch.get("initial_freeze"),
        hard=True,
    )


def compute_training_loss(
    outputs: dict,
    batch: dict[str, torch.Tensor],
    model,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = model.config.training
    sigma = outputs["residual_sigma"].clamp_min(1e-4)
    variance = sigma.square()
    nll = 0.5 * (
        (batch["residual_target"] - outputs["residual_mu"]) ** 2 / variance
    ) + torch.log(sigma)
    loss_nll = masked_mean(nll, batch["residual_mask"])
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

    regularized_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("energy.")
        and not name.startswith("planner.")
    ]
    loss_reg = sum(
        (parameter * parameter).sum() for parameter in regularized_parameters
    )

    total = (
        cfg.nll_weight * loss_nll
        + cfg.hof_weight * loss_hof
        + cfg.path_weight * loss_path
        + cfg.weight_decay * loss_reg
    )
    components = {
        "total": total,
        "nll": loss_nll,
        "hof": loss_hof,
        "path": loss_path,
        "reg": loss_reg,
    }
    return total, components
