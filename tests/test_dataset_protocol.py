from dataclasses import replace

import pytest

from novanet.config import load_config
from novanet.dataset import GenerationOptions, protocol_window_start_count


def test_generation_seed_can_be_separated_from_paper_config_seed():
    options = GenerationOptions(num_samples=1, seed=12_025)
    assert options.seed == 12_025


def test_complete_training_labels_remain_inside_observation_interval():
    config = load_config("configs/paper.yaml")
    count = protocol_window_start_count(config)
    latest_start_s = (
        (count - 1) * config.experiment.geometry_subsample_s
    )
    latest_label_s = (
        latest_start_s
        + config.planner.horizon_steps
        * config.experiment.decision_interval_s
        + config.handover.execution_s
    )
    assert latest_label_s <= config.experiment.duration_s
    assert (
        latest_label_s + config.experiment.geometry_subsample_s
        > config.experiment.duration_s
    )


def test_too_short_observation_interval_is_rejected():
    config = load_config("configs/paper.yaml")
    shortened = replace(
        config,
        experiment=replace(config.experiment, duration_s=100),
    )
    with pytest.raises(ValueError, match="too short"):
        protocol_window_start_count(shortened)
