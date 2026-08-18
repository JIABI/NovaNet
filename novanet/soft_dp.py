"""Differentiable finite-horizon dynamic programming.

All quantities are costs (lower is better). ``transition_cost[:, t, i, j]``
is the cost of entering state ``j`` at horizon index ``t`` from state ``i``.
Index ``t=0`` therefore represents the transition from the currently serving
satellite to the first planned state.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


class SoftDPResult(NamedTuple):
    first_action: torch.Tensor
    first_cost: torch.Tensor
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
    if not torch.isfinite(state_cost).all():
        raise ValueError("state_cost must contain only finite values")
    if not torch.isfinite(transition_cost).all():
        raise ValueError("transition_cost must contain only finite values")
    populated = valid_mask.any(dim=-1)
    if torch.any(~populated[:, 0]):
        raise ValueError("Every batch element needs a valid current state")
    # Empty future rows are an early terminal boundary.  Require a suffix so
    # callers cannot accidentally make a path disappear and later reappear.
    seen_empty = (~populated).to(torch.int64).cumsum(dim=-1) > 0
    if torch.any(populated & seen_empty):
        raise ValueError(
            "A no-coverage horizon must be represented as a terminal suffix"
        )
    return batch, horizon, candidates


def soft_dynamic_program(
    state_cost: torch.Tensor,
    transition_cost: torch.Tensor,
    current_idx: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    freeze_steps: int = 0,
    initial_freeze: torch.Tensor | None = None,
    hard: bool = False,
) -> SoftDPResult:
    """Solve the freeze-aware entropy-regularized finite-horizon problem.

    The dynamic-programming state is ``(incumbent, freeze_counter)``.  When
    the counter is positive and the incumbent remains visible, the only
    feasible action is to stay.  A switch resets the virtual counter to
    ``freeze_steps``; a stay decrements it.  If the incumbent is no longer
    visible, the freeze constraint is released so the controller can recover.

    The recursion is

    ``J_h(i,f) = softmin_j[C_h(i,j) + E_h(j) + J_{h+1}(j,f')]``

    with a zero terminal cost.  The returned first-action distribution is
    conditioned on both the actual incumbent and its actual freeze counter.
    Invalid or freeze-infeasible candidates receive exactly zero probability.
    """

    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if freeze_steps < 0:
        raise ValueError("freeze_steps cannot be negative")
    if valid_mask is None:
        valid_mask = torch.ones_like(state_cost, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=state_cost.device)
    batch, horizon, candidates = _validate_inputs(
        state_cost, transition_cost, current_idx, valid_mask
    )
    if initial_freeze is None:
        initial_freeze = torch.zeros(
            batch, dtype=torch.long, device=state_cost.device
        )
    else:
        initial_freeze = initial_freeze.to(
            dtype=torch.long, device=state_cost.device
        )
    if tuple(initial_freeze.shape) != (batch,):
        raise ValueError("initial_freeze must have shape [B]")
    if torch.any((initial_freeze < 0) | (initial_freeze > freeze_steps)):
        raise ValueError(
            "initial_freeze must lie between zero and freeze_steps"
        )

    tau = torch.as_tensor(
        temperature, dtype=state_cost.dtype, device=state_cost.device
    )
    cost_to_go = torch.empty_like(state_cost)
    conditional = torch.zeros(
        (batch, horizon, candidates, candidates),
        dtype=state_cost.dtype,
        device=state_cost.device,
    )

    freeze_states = freeze_steps + 1
    terminal = state_cost.new_zeros((batch, candidates, freeze_states))
    next_value = terminal
    first_q: torch.Tensor | None = None
    identity = torch.eye(
        candidates, dtype=torch.bool, device=state_cost.device
    )[None, :, :]

    for t in range(horizon - 1, -1, -1):
        q_by_freeze: list[torch.Tensor] = []
        policy_by_freeze: list[torch.Tensor] = []
        value_by_freeze: list[torch.Tensor] = []
        active = valid_mask[:, t].any(dim=-1)
        # Supply a finite dummy action only while evaluating a terminal row;
        # its value/policy are overwritten with the zero terminal condition
        # below, so it never becomes a feasible physical action.
        step_valid = valid_mask[:, t].clone()
        step_valid[~active, 0] = True
        source_visible = step_valid[:, :, None]
        target_visible = step_valid[:, None, :]
        for freeze in range(freeze_states):
            stay_next = next_value[:, :, max(freeze - 1, 0)]
            switch_next = next_value[:, :, freeze_steps]
            continuation = torch.where(
                identity,
                stay_next[:, None, :],
                switch_next[:, None, :],
            )
            q_cost = (
                transition_cost[:, t]
                + state_cost[:, t, None, :]
                + continuation
            )
            allowed = target_visible.expand_as(identity.expand(batch, -1, -1))
            if freeze > 0:
                # A positive freeze locks a still-visible source to its
                # diagonal stay action.  An invisible source is explicitly
                # released for forced recovery.
                allowed = allowed & (~source_visible | identity)
            if hard:
                masked_cost = q_cost.masked_fill(~allowed, torch.inf)
                value, action = masked_cost.min(dim=-1)
                policy = torch.nn.functional.one_hot(
                    action,
                    num_classes=candidates,
                ).to(dtype=q_cost.dtype)
                policy = torch.where(
                    allowed,
                    policy,
                    torch.zeros_like(policy),
                )
            else:
                logits = (-q_cost / tau).masked_fill(~allowed, -torch.inf)
                policy = torch.softmax(logits, dim=-1)
                policy = torch.where(allowed, policy, torch.zeros_like(policy))
                value = -tau * torch.logsumexp(logits, dim=-1)
            # The first empty horizon is a zero-cost terminal state.  This
            # also severs any numerically computed continuation beyond it.
            value = torch.where(active[:, None], value, torch.zeros_like(value))
            policy = torch.where(
                active[:, None, None], policy, torch.zeros_like(policy)
            )
            q_cost = torch.where(
                active[:, None, None], q_cost, torch.zeros_like(q_cost)
            )
            q_by_freeze.append(q_cost)
            policy_by_freeze.append(policy)
            value_by_freeze.append(value)

        q_all = torch.stack(q_by_freeze, dim=2)
        policy_all = torch.stack(policy_by_freeze, dim=2)
        next_value = torch.stack(value_by_freeze, dim=-1)
        cost_to_go[:, t, :] = next_value[:, :, 0]
        conditional[:, t, :, :] = policy_all[:, :, 0, :]
        if t == 0:
            first_q = q_all

    if first_q is None:  # pragma: no cover - horizon validation prevents this
        raise RuntimeError("Soft-DP did not evaluate the first horizon step")
    batch_index = torch.arange(batch, device=state_cost.device)
    selected_q = first_q[batch_index, current_idx, initial_freeze, :]
    first_allowed = valid_mask[:, 0, :]
    locked = (initial_freeze > 0) & valid_mask[batch_index, 0, current_idx]
    first_allowed = first_allowed & (
        ~locked[:, None]
        | torch.nn.functional.one_hot(
            current_idx, num_classes=candidates
        ).to(dtype=torch.bool)
    )
    first_cost = selected_q.masked_fill(~first_allowed, torch.inf)
    if hard:
        first_index = first_cost.argmin(dim=-1)
        first_action = torch.nn.functional.one_hot(
            first_index,
            num_classes=candidates,
        ).to(dtype=state_cost.dtype)
    else:
        first_logits = (-selected_q / tau).masked_fill(
            ~first_allowed,
            -torch.inf,
        )
        first_action = torch.softmax(first_logits, dim=-1)
        first_action = torch.where(
            first_allowed, first_action, torch.zeros_like(first_action)
        )
    return SoftDPResult(first_action, first_cost, cost_to_go, conditional)


class SoftDP(nn.Module):
    """Module wrapper retaining a strict finite-horizon contract."""

    def __init__(
        self,
        horizon: int,
        temperature: float = 1.0,
        freeze_steps: int = 0,
    ):
        super().__init__()
        if horizon < 2:
            raise ValueError("SoftDP horizon must be at least 2")
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("SoftDP temperature must be finite and positive")
        if freeze_steps < 0:
            raise ValueError("freeze_steps cannot be negative")
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.freeze_steps = int(freeze_steps)

    def forward(
        self,
        state_cost: torch.Tensor,
        transition_cost: torch.Tensor,
        current_idx: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        initial_freeze: torch.Tensor | None = None,
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
            freeze_steps=self.freeze_steps,
            initial_freeze=initial_freeze,
        )
        return result if return_details else result.first_action
