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
