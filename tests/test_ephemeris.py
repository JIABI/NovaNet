from datetime import datetime, timezone

import numpy as np

from novanet.ephemeris import Ephemeris, _teme_state_to_ecef


def test_teme_to_ecef_preserves_position_norm_and_has_physical_velocity():
    position_km = np.asarray([7000.0, 0.0, 0.0])
    velocity_km_s = np.asarray([0.0, 7.5, 0.0])
    position_m, velocity_m_s = _teme_state_to_ecef(
        position_km,
        velocity_km_s,
        2460065.0,
    )
    assert np.isclose(np.linalg.norm(position_m), 7_000_000.0)
    assert 6_000.0 < np.linalg.norm(velocity_m_s) < 8_000.0


def test_ephemeris_interpolates_event_time_instead_of_long_linear_extrapolation():
    ephemeris = Ephemeris(
        position_m=np.asarray([[[0.0, 0.0, 0.0]], [[10.0, 20.0, 30.0]]]),
        velocity_m_s=np.asarray([[[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]]),
        names=("SAT",),
        start_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
        step_s=5.0,
    )
    position, velocity = ephemeris.state_at_time(0, 2.5)
    assert np.allclose(position, [5.0, 10.0, 15.0])
    assert np.allclose(velocity, [2.0, 3.0, 4.0])
