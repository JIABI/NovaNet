from datetime import datetime, timezone
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from novanet.config import load_config
from novanet.ephemeris import Ephemeris
from novanet.geometry import UETrajectory
from novanet.forecast import ForecastSequence
from novanet.simulation import (
    Scenario,
    _PendingTransition,
    _epoch_service_segments,
    _realized_first_step_costs,
)


def test_service_stays_on_source_until_actual_cho_completion(monkeypatch):
    config = load_config()
    ephemeris = Ephemeris(
        position_m=np.zeros((3, 2, 3)),
        velocity_m_s=np.zeros((3, 2, 3)),
        names=("SOURCE", "TARGET"),
        start_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
        step_s=5.0,
    )

    def fake_link(
        _config,
        _ephemeris,
        _trajectory,
        _trace,
        satellite_id,
        _time_s,
        _scenario,
    ):
        return (10.0 if satellite_id == 0 else 20.0), 10.0

    monkeypatch.setattr("novanet.simulation._service_link_at_time", fake_link)
    transition = _PendingTransition(
        source_id=0,
        target_id=1,
        execution_start_s=2.10,
        completion_s=2.25,
        success=True,
    )
    rows = _epoch_service_segments(
        config,
        ephemeris,
        UETrajectory(0.0, 0.0),
        object(),
        Scenario(0.0, 0.0),
        0.0,
        5.0,
        0,
        transition,
        [(2.10, 2.25)],
    )
    assert [row[0] for row in rows] == [0.0, 2.25]
    assert [row[2] for row in rows] == [10.0, 20.0]
    assert np.isclose(rows[0][3], 10.0 * (1.0 - 0.15 / 2.25))
    assert rows[1][3] == 20.0


def test_noncausal_reference_uses_realized_horizon_costs(monkeypatch):
    config = load_config()
    config = replace(
        config,
        experiment=replace(config.experiment, decision_interval_s=1),
        planner=replace(config.planner, horizon_steps=2),
    )
    features = np.zeros((2, 2, 6), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    sequence = ForecastSequence(
        node_features=features,
        spatial_adjacency=np.zeros((2, 2, 2), dtype=np.float32),
        valid_mask=valid,
        candidate_ids=np.asarray([0, 1]),
        current_idx=0,
        deterministic_snr_db=np.zeros((2, 2), dtype=np.float32),
        ttl_s=np.asarray([[100.0, 600.0], [70.0, 570.0]], dtype=np.float32),
    )

    def realized_link(*args, **kwargs):
        satellite_id = args[5]
        return -10.0 if satellite_id == 0 else 30.0

    monkeypatch.setattr(
        "novanet.simulation._link_snr_at_offset", realized_link
    )
    costs = _realized_first_step_costs(
        config,
        SimpleNamespace(step_s=1.0),
        UETrajectory(0.0, 0.0),
        object(),
        object(),
        Scenario(0.0, 0.0),
        sequence,
        0,
        0,
    )
    assert costs.shape == (2,)
    assert np.isfinite(costs).all()
    assert costs[1] < costs[0]

    truncated = ForecastSequence(
        node_features=sequence.node_features,
        spatial_adjacency=sequence.spatial_adjacency,
        valid_mask=np.asarray([[True, True], [False, False]]),
        candidate_ids=sequence.candidate_ids,
        current_idx=sequence.current_idx,
        deterministic_snr_db=sequence.deterministic_snr_db,
        ttl_s=sequence.ttl_s,
    )
    truncated_costs = _realized_first_step_costs(
        config,
        SimpleNamespace(step_s=1.0),
        UETrajectory(0.0, 0.0),
        object(),
        object(),
        Scenario(0.0, 0.0),
        truncated,
        0,
        0,
    )
    assert np.isfinite(truncated_costs).all()
