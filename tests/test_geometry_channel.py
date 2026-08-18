from dataclasses import replace

import numpy as np

from novanet.channel import (
    LinkBudget,
    MeasurementTracker,
    RealizedChannelTrace,
    receive_array_gain_linear,
)
from novanet.config import load_config
from novanet.geometry import (
    GeometryState,
    ecef_local_zenith,
    geodetic_to_ecef,
    sky_dome_adjacency,
    time_to_leave_seconds,
)


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


def test_local_zenith_is_wgs84_geodetic_normal():
    position = geodetic_to_ecef(45.0, 20.0, 1000.0)
    zenith = ecef_local_zenith(position)
    latitude = np.deg2rad(45.0)
    longitude = np.deg2rad(20.0)
    expected = np.asarray(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    assert np.allclose(zenith, expected, atol=1e-10)


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


def test_interference_and_array_gain_enter_sinr_path():
    cfg = load_config()
    budget = LinkBudget(cfg.channel, seed=1)
    reference = budget.evaluate(state(0.0))
    interfered = budget.evaluate(state(0.0), interference_power_w=1e-12)
    gained = budget.evaluate(state(0.0), receive_gain_linear=4.0)
    assert interfered.sinr_db < reference.sinr_db
    assert np.isclose(reference.sinr_db, reference.snr_db)
    assert np.isclose(gained.received_power_dbm - reference.received_power_dbm, 6.0206, atol=1e-3)

    local, gain = receive_array_gain_linear(
        np.asarray([1.0, 0.0, 0.0]),
        np.eye(3),
    )
    assert np.allclose(local, [1.0, 0.0, 0.0])
    assert np.isclose(gain, 1.0)

    callback_budget = LinkBudget(
        cfg.channel,
        seed=1,
        receive_gain_model=lambda _geometry: 4.0,
    )
    callback_gain = callback_budget.evaluate(state(0.0))
    assert np.isclose(
        callback_gain.received_power_dbm - reference.received_power_dbm,
        6.0206,
        atol=1e-3,
    )
    # An explicit per-call gain remains available for controlled overrides.
    overridden = callback_budget.evaluate(state(0.0), receive_gain_linear=1.0)
    assert np.isclose(overridden.received_power_dbm, reference.received_power_dbm)


def test_sky_graph_uses_distinct_neighbors_without_self_messages():
    los = np.eye(3)
    adjacency = sky_dome_adjacency(
        los, np.ones(3, dtype=bool), neighbors=1, temperature=1.0
    )
    assert np.allclose(np.diag(adjacency), 0.0)
    assert np.all((adjacency > 0.0).sum(axis=1) == 1)


def test_realized_channel_trace_is_keyed_not_call_order_dependent():
    cfg = load_config()
    trace = RealizedChannelTrace(cfg.channel, seed=17, event_step_s=0.01)
    first = trace.fading_gain_db(4, 12.34)
    _unrelated = trace.fading_gain_db(9, 77.0)
    second = trace.fading_gain_db(4, 12.34)
    assert first == second
    noise = trace.measurement_noise_db(4, 30.0, 2.0)
    assert isinstance(noise, float)
    assert np.isfinite(noise)
    assert noise == trace.measurement_noise_db(4, 30.0, 2.0)
    assert trace.measurement_noise_db(4, 30.0, 0.0) == 0.0
    unit_noise = trace.measurement_noise_db(4, 30.0, 1.0)
    assert np.isclose(noise, 2.0 * unit_noise)
    with np.testing.assert_raises_regex(ValueError, "finite and nonnegative"):
        trace.measurement_noise_db(4, 30.0, float("nan"))
    assert trace.doppler_estimation_error_hz(
        4, 30.0
    ) == trace.doppler_estimation_error_hz(4, 30.0)

    attenuated = RealizedChannelTrace(
        cfg.channel,
        seed=17,
        event_step_s=0.01,
        receive_gain_model=lambda _geometry: 0.1,
    )
    reference_link = trace.evaluate(state(0.0), 4, 30.0)
    attenuated_link = attenuated.evaluate(state(0.0), 4, 30.0)
    assert np.isclose(
        attenuated_link.sinr_db - reference_link.sinr_db,
        -10.0,
        atol=1e-9,
    )


def test_stale_measurement_is_not_filtered_repeatedly():
    cfg = load_config()
    tracker = MeasurementTracker(cfg.channel)
    tracker.update(4, 10.0, 30.0)
    tracker.update(4, -20.0, 30.0)
    tracker.update(4, -30.0, 0.0)
    assert tracker.records[4].filtered_snr_db == 10.0
    assert tracker.records[4].timestamp_s == 30.0

    tracker.update(4, 12.0, 60.0)
    expected = (
        cfg.channel.measurement_iir_alpha * 12.0
        + (1.0 - cfg.channel.measurement_iir_alpha) * 10.0
    )
    assert tracker.records[4].filtered_snr_db == expected
