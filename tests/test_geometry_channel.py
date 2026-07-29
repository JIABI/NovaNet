from dataclasses import replace

import numpy as np

from novanet.channel import LinkBudget
from novanet.config import load_config
from novanet.geometry import GeometryState, time_to_leave_seconds


def state(radial_velocity):
    return GeometryState(
        elevation_deg=45.0,
        range_m=700e3,
        range_rate_m_s=radial_velocity,
        elevation_rate_deg_s=0.0,
        angular_speed_deg_s=0.5,
        radial_velocity_m_s=radial_velocity,
        los_unit=np.asarray([1.0, 0.0, 0.0]),
    )


def test_ttl_forward_crossing_is_finite_at_zenith():
    elevation = np.asarray([12.0, 30.0, 60.0, 90.0, 60.0, 20.0, 5.0])
    ttl = time_to_leave_seconds(elevation, 1, 5.0, 10.0)
    assert np.isfinite(ttl)
    assert np.isclose(ttl, (4.0 + 2.0 / 3.0) * 5.0)


def test_doppler_and_tracking_are_explicit():
    cfg = load_config()
    budget = LinkBudget(cfg.channel, seed=1)
    stationary = budget.evaluate(state(0.0), stochastic=False)
    moving = budget.evaluate(state(7000.0), stochastic=False)
    assert stationary.doppler_hz == 0.0
    assert abs(moving.doppler_hz) > 100_000.0
    assert abs(moving.residual_doppler_hz) < abs(moving.doppler_hz)
    assert moving.tracking_loss_db >= 0.0


def test_bandwidth_options_share_eirp_density():
    cfg = load_config()
    budget = LinkBudget(cfg.channel, seed=1)
    narrow = budget.evaluate(state(0.0), bandwidth_hz=20e6)
    wide = budget.evaluate(state(0.0), bandwidth_hz=100e6)
    assert np.isclose(narrow.snr_db, wide.snr_db, atol=1e-9)
    assert np.isclose(wide.rate_bps / narrow.rate_bps, 5.0, rtol=1e-9)

