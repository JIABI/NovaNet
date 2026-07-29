"""Differentiable finite-horizon dynamic programming.

All quantities are costs (lower is better). ``transition_cost[:, t, i, j]``
is the cost of entering state ``j`` at horizon index ``t`` from state ``i``.
Index ``t=0`` therefore represents the transition from the currently serving
satellite to the first planned state.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class SoftDPResult(NamedTuple):
    first_action: torch.Tensor
    cost_to_go: torch.Tensor
    conditional_policy: torch.Tensor


def _validate_inputs(
    state_cost: torch.Tensor,
    transition_cost: torch.Tensor,
    current_idx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[int, int, int]:
    if state_cost.ndim != 3:
        raise ValueError("state_cost must have shape [B,H,K]")
    batch, horizon, candidates = state_cost.shape
    expected_transition = (batch, horizon, candidates, candidates)
    if tuple(transition_cost.shape) != expected_transition:
        raise ValueError(
            f"transition_cost must have shape {expected_transition}, "
            f"got {tuple(transition_cost.shape)}"
        )
    if tuple(valid_mask.shape) != (batch, horizon, candidates):
        raise ValueError("valid_mask must have shape [B,H,K]")
    if tuple(current_idx.shape) != (batch,):
        raise ValueError("current_idx must have shape [B]")
    if torch.any((current_idx < 0) | (current_idx >= candidates)):
        raise ValueError("current_idx contains an out-of-range candidate index")
    if torch.any(valid_mask.sum(dim=-1) == 0):
        raise ValueError("Every batch/horizon element needs at least one valid state")
    return batch, horizon, candidates


def soft_dynamic_program(
    state_cost: torch.Tensor,
    transition_cost: torch.Tensor,
    current_idx: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> SoftDPResult:
    """Solve a batched entropy-regularized finite-horizon cost problem.

    The recursion is

    ``J_t(i) = E_t(i) + softmin_j[T_{t+1}(i,j) + J_{t+1}(j)]``

    with ``J_{H-1}(i)=E_{H-1}(i)``. The returned first-action distribution is
    conditioned on the actual incumbent through ``T_0(current,j)``.
    Invalid candidates receive exactly zero probability.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if valid_mask is None:
        valid_mask = torch.ones_like(state_cost, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=state_cost.device)
    batch, horizon, candidates = _validate_inputs(
        state_cost, transition_cost, current_idx, valid_mask
    )

    tau = torch.as_tensor(
        temperature, dtype=state_cost.dtype, device=state_cost.device
    )
    inf = torch.as_tensor(
        torch.finfo(state_cost.dtype).max / 1e4,
        dtype=state_cost.dtype,
        device=state_cost.device,
    )
    masked_state = torch.where(valid_mask, state_cost, inf)
    cost_to_go = torch.empty_like(masked_state)
    conditional = torch.zeros(
        (batch, horizon, candidates, candidates),
        dtype=state_cost.dtype,
        device=state_cost.device,
    )

    cost_to_go[:, -1, :] = masked_state[:, -1, :]
    for t in range(horizon - 2, -1, -1):
        q_cost = transition_cost[:, t + 1, :, :] + cost_to_go[
            :, t + 1, None, :
        ]
        target_valid = valid_mask[:, t + 1, None, :]
        q_cost = torch.where(target_valid, q_cost, inf)
        logits = -q_cost / tau
        conditional[:, t + 1, :, :] = torch.softmax(logits, dim=-1)
        soft_future = -tau * torch.logsumexp(logits, dim=-1)
        cost_to_go[:, t, :] = torch.where(
            valid_mask[:, t, :],
            masked_state[:, t, :] + soft_future,
            inf,
        )

    batch_index = torch.arange(batch, device=state_cost.device)
    first_cost = (
        transition_cost[batch_index, 0, current_idx, :]
        + cost_to_go[:, 0, :]
    )
    first_logits = torch.where(valid_mask[:, 0, :], -first_cost / tau, -inf)
    first_action = torch.softmax(first_logits, dim=-1)

    all_initial = -(
        transition_cost[:, 0, :, :] + cost_to_go[:, 0, None, :]
    ) / tau
    all_initial = torch.where(valid_mask[:, 0, None, :], all_initial, -inf)
    conditional[:, 0, :, :] = torch.softmax(all_initial, dim=-1)
    return SoftDPResult(first_action, cost_to_go, conditional)


class SoftDP(nn.Module):
    """Module wrapper retaining a strict finite-horizon contract."""

    def __init__(self, horizon: int, temperature: float = 1.0):
        super().__init__()
        if horizon < 2:
            raise ValueError("SoftDP horizon must be at least 2")
        if temperature <= 0.0:
            raise ValueError("SoftDP temperature must be positive")
        self.horizon = int(horizon)
        self.temperature = float(temperature)

    def forward(
        self,
        state_cost: torch.Tensor,
        transition_cost: torch.Tensor,
        current_idx: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        if state_cost.shape[1] != self.horizon:
            raise ValueError(
                f"Configured horizon={self.horizon}, "
                f"but input horizon={state_cost.shape[1]}"
            )
        result = soft_dynamic_program(
            state_cost=state_cost,
            transition_cost=transition_cost,
            current_idx=current_idx,
            valid_mask=valid_mask,
            temperature=self.temperature,
        )
        return result if return_details else result.first_action

