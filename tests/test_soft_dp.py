import itertools

import numpy as np
import torch

from novanet.soft_dp import soft_dynamic_program


def path_cost(state, transition, current, path):
    total = transition[0, current, path[0]] + state[0, path[0]]
    for time in range(1, len(path)):
        total += transition[time, path[time - 1], path[time]]
        total += state[time, path[time]]
    return total


def test_low_temperature_matches_brute_force_finite_horizon():
    state = np.asarray(
        [
            [0.0, -0.2],
            [2.0, -0.2],
            [2.0, -2.0],
        ],
        dtype=np.float32,
    )
    transition = np.zeros((3, 2, 2), dtype=np.float32)
    transition[:, 0, 1] = 0.7
    transition[:, 1, 0] = 0.7
    all_paths = list(itertools.product(range(2), repeat=3))
    costs = [path_cost(state, transition, 0, path) for path in all_paths]
    optimal_first = all_paths[int(np.argmin(costs))][0]

    result = soft_dynamic_program(
        torch.tensor(state)[None],
        torch.tensor(transition)[None],
        torch.tensor([0]),
        torch.ones((1, 3, 2), dtype=torch.bool),
        temperature=1e-3,
    )
    assert int(result.first_action.argmax(dim=-1)) == optimal_first
    assert optimal_first == 1


def test_future_cost_changes_first_action():
    state = torch.tensor([[[0.0, 0.1], [10.0, -10.0]]])
    transition = torch.zeros((1, 2, 2, 2))
    transition[:, 1, 0, 1] = 20.0
    transition[:, 1, 1, 0] = 20.0
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    result = soft_dynamic_program(
        state, transition, torch.tensor([0]), mask, temperature=0.01
    )
    assert int(result.first_action.argmax()) == 1


def test_invalid_candidate_has_zero_probability_and_gradients_flow():
    state = torch.randn(1, 3, 3, requires_grad=True)
    transition = torch.rand(1, 3, 3, 3, requires_grad=True)
    mask = torch.tensor(
        [[[True, True, False], [True, False, True], [True, True, False]]]
    )
    result = soft_dynamic_program(
        state, transition, torch.tensor([0]), mask, temperature=0.7
    )
    assert result.first_action[0, 2] == 0.0
    result.first_action[0, 0].backward()
    assert state.grad is not None
    assert transition.grad is not None


def test_actual_freeze_locks_visible_incumbent_and_releases_for_recovery():
    state = torch.zeros((1, 3, 2))
    transition = torch.zeros((1, 3, 2, 2))
    valid = torch.ones((1, 3, 2), dtype=torch.bool)
    locked = soft_dynamic_program(
        state,
        transition,
        torch.tensor([0]),
        valid,
        temperature=1.0,
        freeze_steps=1,
        initial_freeze=torch.tensor([1]),
    )
    assert torch.equal(locked.first_action, torch.tensor([[1.0, 0.0]]))
    assert torch.isinf(locked.first_cost[0, 1])

    valid[:, 0, 0] = False
    recovered = soft_dynamic_program(
        state,
        transition,
        torch.tensor([0]),
        valid,
        temperature=1.0,
        freeze_steps=1,
        initial_freeze=torch.tensor([1]),
    )
    assert torch.equal(recovered.first_action, torch.tensor([[0.0, 1.0]]))


def test_clairvoyant_teacher_can_use_hard_path_cost_before_soft_distribution():
    state = torch.tensor([[[0.0, 0.2], [1.0, -1.0]]])
    transition = torch.zeros((1, 2, 2, 2))
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    hard = soft_dynamic_program(
        state,
        transition,
        torch.tensor([0]),
        valid,
        temperature=1.0,
        hard=True,
    )
    soft = soft_dynamic_program(
        state,
        transition,
        torch.tensor([0]),
        valid,
        temperature=1.0,
    )
    # The hard teacher first minimizes realized continuation cost. The
    # deployed soft DP includes entropy inside its Bellman value.
    assert not torch.allclose(hard.first_cost, soft.first_cost)
    assert int(hard.first_action.argmax()) == int(hard.first_cost.argmin())


def test_empty_future_suffix_is_a_zero_cost_early_terminal_state():
    state = torch.tensor([[[0.0, 0.2], [-100.0, -100.0], [-200.0, -200.0]]])
    transition = torch.zeros((1, 3, 2, 2))
    valid = torch.tensor([[[True, True], [False, False], [False, False]]])
    result = soft_dynamic_program(
        state,
        transition,
        torch.tensor([0]),
        valid,
        temperature=0.5,
    )
    one_step = soft_dynamic_program(
        state[:, :1],
        transition[:, :1],
        torch.tensor([0]),
        valid[:, :1],
        temperature=0.5,
    )
    assert torch.allclose(result.first_cost, one_step.first_cost)
    assert torch.equal(result.cost_to_go[:, 1:], torch.zeros_like(result.cost_to_go[:, 1:]))
    assert torch.equal(
        result.conditional_policy[:, 1:],
        torch.zeros_like(result.conditional_policy[:, 1:]),
    )


def test_coverage_cannot_reappear_after_terminal_boundary():
    state = torch.zeros((1, 3, 2))
    transition = torch.zeros((1, 3, 2, 2))
    invalid_gap = torch.tensor(
        [[[True, True], [False, False], [True, False]]]
    )
    with np.testing.assert_raises_regex(ValueError, "terminal suffix"):
        soft_dynamic_program(
            state,
            transition,
            torch.tensor([0]),
            invalid_gap,
        )


def test_soft_dp_rejects_nonfinite_costs_and_temperature():
    state = torch.zeros((1, 2, 2))
    transition = torch.zeros((1, 2, 2, 2))
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    with np.testing.assert_raises_regex(ValueError, "state_cost"):
        soft_dynamic_program(
            state.masked_fill(torch.tensor([[[True, False], [False, False]]]), float("nan")),
            transition,
            torch.tensor([0]),
            valid,
        )
    with np.testing.assert_raises_regex(ValueError, "finite and positive"):
        soft_dynamic_program(
            state,
            transition,
            torch.tensor([0]),
            valid,
            temperature=float("nan"),
        )
