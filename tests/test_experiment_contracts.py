from dataclasses import replace
import json
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.blockage import parse_conditions
from experiments.common import (
    add_paired_oracle_gap,
    aggregate_rows,
    evaluation_episode_seed,
    evaluation_rng,
    resolve_evaluation_seed,
    write_protocol,
)
from experiments.density_convergence import (
    candidate_union_statistics,
    convergence_summary,
)
from experiments.ephemeris_aging import parse_ages
from experiments.ephemeris_aging import matched_records
from experiments.ephemeris_aging import require_identical_zero_age_snapshot
from experiments.freeze_sensitivity import parse_windows
from experiments.lcb_variance_sensitivity import (
    parse_floats,
    parse_nonnegative_ints,
)
from experiments.multi_ue import main as multi_ue_main
from novanet.dataset import GenerationOptions
from novanet.config import load_config
from novanet.ephemeris import Ephemeris
from novanet.simulation import Scenario


def test_stress_argument_parsers_reject_or_preserve_protocol_values():
    assert parse_conditions("8:0.10,12:0.20") == [(8.0, 0.1), (12.0, 0.2)]
    assert parse_windows("0,1,2,3") == [0, 1, 2, 3]
    assert parse_ages("0,24,72") == [0, 24, 72]
    assert parse_floats("0,1,2") == [0.0, 1.0, 2.0]
    assert parse_nonnegative_ints("0,1,2") == [0, 1, 2]
    with pytest.raises(ValueError):
        parse_floats("nan")
    with pytest.raises(ValueError):
        parse_nonnegative_ints("0,-1")


def test_scenario_and_generation_options_reject_future_staleness():
    with pytest.raises(ValueError, match="staleness_steps"):
        Scenario(0.0, 0.0, staleness_steps=-1)
    with pytest.raises(ValueError, match="staleness_steps"):
        GenerationOptions(num_samples=1, staleness_steps=-1)


@pytest.mark.parametrize("blocking_cost", ["-1", "nan"])
def test_multi_ue_cli_rejects_invalid_blocking_cost(monkeypatch, blocking_cost):
    monkeypatch.setattr(
        sys,
        "argv",
        ["multi_ue", "--blocking-cost", blocking_cost],
    )
    with pytest.raises(ValueError, match="blocking_cost"):
        multi_ue_main()


def test_convergence_epoch_uses_persistent_one_percent_band():
    frame = pd.DataFrame(
        {
            "density": [60] * 9,
            "candidate_cap": [8] * 9,
            "seed": [2025] * 9,
            "epoch": np.arange(1, 10),
            "validation_loss": [2.0, 1.5, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    result = convergence_summary(frame)[0]
    assert result["converged"] is True
    assert result["convergence_epoch"] == 9


def test_candidate_statistics_use_untruncated_horizon_union():
    base = load_config("configs/paper.yaml")
    cfg = replace(
        base,
        experiment=replace(
            base.experiment,
            duration_s=60,
            decision_interval_s=30,
            geometry_subsample_s=5,
            candidate_cap=2,
            num_satellites=2,
        ),
        planner=replace(base.planner, horizon_steps=2),
    )
    steps = 30
    positions = np.zeros((steps, 2, 3), dtype=float)
    positions[:, 0] = [7_000_000.0, 0.0, 0.0]
    positions[:, 1] = [-7_000_000.0, 0.0, 0.0]
    ephemeris = Ephemeris(
        position_m=positions,
        velocity_m_s=np.zeros_like(positions),
        names=("visible", "hidden"),
        start_utc=cfg.start_utc,
        step_s=5.0,
    )
    result = candidate_union_statistics(
        cfg,
        ephemeris,
        [Scenario(latitude_deg=0.0, longitude_deg=0.0)],
    )
    assert result["evaluation_windows"] > 0
    assert result["mean_visible_union"] == 1.0
    assert result["mean_effective_candidates"] == 1.0
    assert result["cap_activation_percent"] == 0.0


def test_aggregation_reports_pooled_event_ratios():
    rows = [
        {
            "method": "A",
            "user": 0,
            "handovers": 1,
            "handover_failures": 1,
            "hof_percent": 100.0,
            "ping_pong_count": 0,
        },
        {
            "method": "A",
            "user": 1,
            "handovers": 9,
            "handover_failures": 0,
            "hof_percent": 0.0,
            "ping_pong_count": 2,
        },
    ]
    result = aggregate_rows(rows)[0]
    assert result["hof_percent_mean"] == 50.0
    assert result["hof_percent_pooled"] == 10.0
    assert result["ping_pong_percent_pooled"] == 20.0


def test_evaluation_uses_reported_2025_seed_in_a_separate_rng_domain():
    config = load_config()
    seed = resolve_evaluation_seed(config, None)
    assert seed == 2025
    values = evaluation_rng(seed).normal(size=3)
    assert np.array_equal(values, evaluation_rng(seed).normal(size=3))
    assert evaluation_episode_seed(seed, 0) != seed
    assert evaluation_episode_seed(seed, 0) != evaluation_episode_seed(seed, 1)


def test_protocol_records_requested_evaluation_seed(tmp_path, monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        "experiments.common.validate_tle_epoch",
        lambda *_args, **_kwargs: {"tle": "test"},
    )
    path = write_protocol(
        tmp_path / "protocol.json",
        config,
        runner="test",
        evaluation_seed=9090,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evaluation_base_seed"] == 9090
    assert payload["evaluation_rng_domain"] == "EVAL-v1"


def test_aggregate_rows_omits_all_nan_optional_metric_and_rejects_mixed_nan():
    rows = [
        {"method": "x", "value": 1.0, "target_cost_sum": float("nan")},
        {"method": "x", "value": 2.0, "target_cost_sum": float("nan")},
    ]
    summary = aggregate_rows(rows)
    assert summary[0]["value_mean"] == 1.5
    assert "target_cost_sum_mean" not in summary[0]

    rows[1]["target_cost_sum"] = 1.0
    with pytest.raises(ValueError, match="mixes finite and non-finite"):
        aggregate_rows(rows)
    with pytest.raises(ValueError, match="mixes finite and non-finite"):
        aggregate_rows(
            [
                {"method": "x", "throughput": float("nan")},
                {"method": "x", "throughput": float("nan")},
            ]
        )


def test_oracle_gap_uses_each_methods_paired_realized_windows():
    summary = [
        {
            "method": "A",
            "mean_target_cost_mean": 11.0,
            "mean_oracle_target_cost_mean": 10.0,
        },
        {
            "method": "B",
            "mean_target_cost_mean": -8.0,
            "mean_oracle_target_cost_mean": -10.0,
        },
    ]
    add_paired_oracle_gap(summary)
    assert np.isclose(summary[0]["oracle_gap_percent"], 10.0)
    assert np.isclose(summary[1]["oracle_gap_percent"], 20.0)


def test_oracle_gap_pools_unequal_numbers_of_decision_windows():
    rows = [
        {
            "method": "A",
            "target_cost_sum": 2.0,
            "oracle_target_cost_sum": 1.0,
            "paired_cost_windows": 1,
        },
        {
            "method": "A",
            "target_cost_sum": 33.0,
            "oracle_target_cost_sum": 30.0,
            "paired_cost_windows": 3,
        },
    ]
    summary = aggregate_rows(rows)
    add_paired_oracle_gap(summary)
    # Global means are 35/4 and 31/4, hence a 4/31 relative gap.
    assert np.isclose(summary[0]["mean_target_cost_pooled"], 35.0 / 4.0)
    assert np.isclose(
        summary[0]["oracle_gap_percent"], 100.0 * 4.0 / 31.0
    )


def test_ephemeris_aging_matches_norad_ids_in_truth_order(tmp_path):
    truth = [
        ("TRUTH_A", "1 00001 rest-a", "2 truth-a"),
        ("TRUTH_B", "1 00002 rest-b", "2 truth-b"),
    ]
    planning = tmp_path / "planning.tle"
    planning.write_text(
        "NEW_B\n1 00002 prior-b\n2 prior-b\n"
        "NEW_A\n1 00001 prior-a\n2 prior-a\n",
        encoding="utf-8",
    )
    selected = matched_records(planning, truth)
    assert [record[0] for record in selected] == ["TRUTH_A", "TRUTH_B"]
    assert [record[1][2:7] for record in selected] == ["00001", "00002"]
    assert selected[0][1].endswith("prior-a")


def test_zero_age_snapshot_requires_exact_truth_file(tmp_path):
    truth = tmp_path / "truth.tle"
    exact = tmp_path / "exact.tle"
    nearby = tmp_path / "nearby.tle"
    truth.write_text("same snapshot\n", encoding="utf-8")
    exact.write_text("same snapshot\n", encoding="utf-8")
    nearby.write_text("nearby snapshot\n", encoding="utf-8")
    require_identical_zero_age_snapshot(exact, truth)
    with np.testing.assert_raises_regex(ValueError, "byte-identical"):
        require_identical_zero_age_snapshot(nearby, truth)
