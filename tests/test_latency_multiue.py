from dataclasses import replace

import numpy as np
import pytest

from novanet.config import load_config
from novanet.forecast import ForecastSequence
from novanet.latency import latency_summary, simulate_fifo_latency
from novanet.multi_ue import (
    SynchronousAssociationScheduler,
    _capped_proportional_fair_allocation,
    allocate_fixed_associations,
    jain_fairness,
)
from novanet.policies import SkipKPolicy
from experiments.multi_ue import (
    _PendingMultiUETransition,
    _allocate_epoch_service_segments,
    _base_score,
    _result_paths,
)


def test_latency_is_deterministic_and_has_shared_one_ms_component():
    cfg = load_config()
    traffic = replace(cfg.traffic, arrival_rate_packets_s=50.0)
    times = np.asarray([0.0, 20.0])
    rates = np.asarray([60e6, 60e6])
    first = simulate_fifo_latency(traffic, 20.0, times, rates, seed=9)
    second = simulate_fifo_latency(traffic, 20.0, times, rates, seed=9)
    assert np.array_equal(first.total_ms, second.total_ms)
    assert np.all(first.protocol_processing_ms == 1.0)
    no_ho = latency_summary(first)
    with_ho = latency_summary(
        simulate_fifo_latency(
            traffic,
            20.0,
            times,
            rates,
            handover_blackouts=[(10.0, 10.15)],
            seed=9,
        )
    )
    assert with_ho["p99_9_ms"] > no_ho["p99_9_ms"]


def test_blackout_preempts_service_and_finite_fifo_drops_tail():
    cfg = load_config()
    preempt = replace(
        cfg.traffic,
        arrival_rate_packets_s=1.0,
        packet_size_bytes=1_000_000,
        queue_capacity_packets=8,
        fixed_network_delay_ms=0.0,
        protocol_processing_ms=0.0,
    )
    times = np.asarray([0.0, 10.0])
    rates = np.asarray([8e6, 8e6])
    trace = simulate_fifo_latency(
        preempt,
        2.0,
        times,
        rates,
        handover_blackouts=[(0.5, 0.75)],
        seed=3,
    )
    assert np.any(trace.cho_interruption_ms >= 250.0 - 1e-8)

    overload = replace(
        cfg.traffic,
        arrival_rate_packets_s=200.0,
        packet_size_bytes=100_000,
        queue_capacity_packets=2,
        fixed_network_delay_ms=0.0,
        protocol_processing_ms=0.0,
    )
    dropped = simulate_fifo_latency(
        overload,
        1.0,
        np.asarray([0.0, 1.0]),
        np.asarray([1e6, 1e6]),
        seed=4,
    )
    assert dropped.dropped > 0


def test_zero_rate_interval_pauses_and_then_resumes_packet_service():
    cfg = load_config()
    traffic = replace(
        cfg.traffic,
        arrival_rate_packets_s=20.0,
        packet_size_bytes=1500,
        queue_capacity_packets=4096,
        fixed_network_delay_ms=0.0,
        protocol_processing_ms=0.0,
    )
    trace = simulate_fifo_latency(
        traffic,
        1.0,
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 100e6, 100e6]),
        seed=17,
    )
    assert len(trace.total_ms) > 0
    assert np.isfinite(trace.total_ms).all()
    # Packets arriving during the outage wait until 0.5 s and then drain;
    # they must not receive the former 12,000-second clamp-derived service.
    assert trace.total_ms.max() < 1_000.0


def test_latency_rejects_ambiguous_or_nonphysical_rate_traces():
    cfg = load_config()
    traffic = replace(cfg.traffic, arrival_rate_packets_s=1.0)
    with pytest.raises(ValueError, match="start at t=0"):
        simulate_fifo_latency(
            traffic,
            1.0,
            np.asarray([0.1, 1.0]),
            np.asarray([1e6, 1e6]),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_fifo_latency(
            traffic,
            1.0,
            np.asarray([0.0, 1.0]),
            np.asarray([1e6, -1.0]),
        )


def test_synchronous_scheduler_respects_capacity_and_reports_fairness():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=2,
        minimum_admission_rate_mbps=1.0,
    )
    scheduler = SynchronousAssociationScheduler(multi)
    candidates = np.asarray([[1, 2], [1, 2], [1, 2]])
    score = np.asarray([[3.0, 2.0], [3.0, 1.0], [3.0, 2.5]])
    link = np.full_like(score, 100.0)
    result = scheduler.associate(candidates, score, link)
    assert np.sum(result.assigned_satellite == 1) <= 2
    assert result.allocated_rate_mbps.sum() <= 200.0 + 1e-9
    assert 0.0 < jain_fairness(result.allocated_rate_mbps) <= 1.0


def test_multi_ue_skip_state_follows_load_adjusted_target():
    cfg = load_config()
    horizon = cfg.planner.horizon_steps
    features = np.zeros((horizon, 3, 6), dtype=np.float32)
    features[:, :, 0] = np.asarray([0.2, 0.9, 0.8])
    valid = np.ones((horizon, 3), dtype=bool)
    sequence = ForecastSequence(
        node_features=features,
        spatial_adjacency=np.zeros((horizon, 3, 3), dtype=np.float32),
        valid_mask=valid,
        candidate_ids=np.asarray([10, 11, 12]),
        current_idx=0,
        deterministic_snr_db=np.zeros((horizon, 3), dtype=np.float32),
        ttl_s=np.full((horizon, 3), 60.0, dtype=np.float32),
    )
    policy = SkipKPolicy(skip=1)

    first = _base_score(
        cfg,
        "Skip-1-LB",
        sequence,
        None,
        0,
        np.asarray([0.0, 1.0, 0.0]),
        skip_policy=policy,
    )
    assert int(np.argmax(first)) == 0

    second = _base_score(
        cfg,
        "Skip-1-LB",
        sequence,
        None,
        0,
        np.zeros(3),
        skip_policy=policy,
    )
    assert int(np.argmax(second)) == 1


def test_multi_ue_raw_and_summary_paths_cannot_collide(tmp_path):
    rows, summary, protocol = _result_paths(tmp_path / "rows.csv")
    assert rows.name == "rows.csv"
    assert summary.name == "summary.csv"
    assert protocol.name == "protocol.json"
    with pytest.raises(ValueError, match="cannot be named summary.csv"):
        _result_paths(tmp_path / "summary.csv")


def test_multi_ue_scheduler_rejects_nonfinite_real_candidate_inputs():
    cfg = load_config()
    scheduler = SynchronousAssociationScheduler(cfg.multi_ue)
    with pytest.raises(ValueError, match="scores for real candidates"):
        scheduler.associate(
            np.asarray([[1]]),
            np.asarray([[np.nan]]),
            np.asarray([[10.0]]),
        )
    masked = scheduler.associate(
        np.asarray([[1]]),
        np.asarray([[-np.inf]]),
        np.asarray([[10.0]]),
    )
    assert masked.blocked[0]
    with pytest.raises(ValueError, match="fairness rates"):
        jain_fairness(np.asarray([1.0, np.nan]))


def test_two_phase_coordinator_retains_best_and_rejected_user_advances():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=1,
        minimum_admission_rate_mbps=1.0,
    )
    scheduler = SynchronousAssociationScheduler(multi)
    candidates = np.asarray([[1, 2], [1, 2]])
    score = np.asarray([[3.0, 2.0], [4.0, 1.0]])
    link = np.full_like(score, 100.0)
    result = scheduler.associate(candidates, score, link)
    assert np.array_equal(result.assigned_satellite, [2, 1])


def test_pf_reallocates_capacity_left_by_link_limited_user():
    allocated = _capped_proportional_fair_allocation(
        np.asarray([1.0, 1.0]),
        np.asarray([10.0, 100.0]),
        100.0,
    )
    assert np.allclose(allocated, [10.0, 90.0])


def test_incumbent_prefix_uses_same_capacity_limited_pf_scheduler():
    assigned = np.asarray([7, 7, 8])
    candidates = np.asarray([[7, 9], [7, 9], [8, 9]])
    links = np.asarray([[90.0, 0.0], [90.0, 0.0], [80.0, 0.0]])
    allocated = allocate_fixed_associations(
        assigned,
        candidates,
        links,
        np.ones(3),
        capacity_mbps=100.0,
    )
    assert np.allclose(allocated, [50.0, 50.0, 80.0])


def test_pf_recycles_capacity_after_subthreshold_admission_is_removed():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=3,
        minimum_admission_rate_mbps=10.0,
    )
    result = SynchronousAssociationScheduler(multi).associate(
        np.asarray([[7], [7], [7]]),
        np.asarray([[3.0], [2.0], [1.0]]),
        np.asarray([[100.0], [100.0], [100.0]]),
        previous_average_rate_mbps=np.asarray([1.0, 1.0, 100.0]),
    )
    assert np.array_equal(result.assigned_satellite, [7, 7, -1])
    assert np.allclose(result.allocated_rate_mbps, [50.0, 50.0, 0.0])
    assert np.isclose(result.allocated_rate_mbps.sum(), 100.0)


def test_late_cho_uses_source_then_blackout_then_target_pf_capacity():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=32,
        minimum_admission_rate_mbps=1.0,
    )
    transition = _PendingMultiUETransition(
        source_id=1,
        target_id=2,
        execution_start_s=20.0,
        completion_s=25.0,
        success=True,
    )
    rates, load, blocked = _allocate_epoch_service_segments(
        epoch_start_s=0.0,
        epoch_end_s=30.0,
        base_serving_satellite=np.asarray([1, 2]),
        transitions=[transition, None],
        failed_epoch=np.asarray([False, False]),
        candidate_ids=np.asarray([[1, 2], [2, 1]]),
        achievable_rate_mbps=np.full((2, 2), 100.0),
        previous_average_rate_mbps=np.ones(2),
        multi_ue_config=multi,
    )
    # UE 0: 20 s at 100, 5 s blackout, then 5 s at 50.
    # UE 1 keeps all target capacity until the successful completion.
    assert np.allclose(rates, [75.0, 275.0 / 3.0])
    assert np.isclose(load[1], 20.0 / (30.0 * 32.0))
    assert np.isclose(load[2], 35.0 / (30.0 * 32.0))
    assert np.allclose(blocked, 0.0)


def test_failed_transition_does_not_release_capacity_before_outcome():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=32,
        minimum_admission_rate_mbps=1.0,
    )
    transition = _PendingMultiUETransition(
        source_id=1,
        target_id=2,
        execution_start_s=20.0,
        completion_s=25.0,
        success=False,
    )
    rates, _, blocked = _allocate_epoch_service_segments(
        epoch_start_s=0.0,
        epoch_end_s=30.0,
        base_serving_satellite=np.asarray([1, 1]),
        transitions=[transition, None],
        failed_epoch=np.asarray([True, False]),
        candidate_ids=np.asarray([[1, 2], [1, 2]]),
        achievable_rate_mbps=np.full((2, 2), 100.0),
        previous_average_rate_mbps=np.ones(2),
        multi_ue_config=multi,
    )
    # The failed UE is credited zero for the epoch, as specified by the
    # evaluation protocol.  It still holds its causal PF share before the
    # event and after returning to the source; only the 5-s blackout frees
    # capacity for the other UE.
    assert np.allclose(rates, [0.0, (20.0 * 50.0 + 5.0 * 100.0 + 5.0 * 50.0) / 30.0])
    assert np.allclose(blocked, 0.0)


def test_cross_epoch_pending_transition_switches_only_at_completion():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=32,
        minimum_admission_rate_mbps=1.0,
    )
    pending = _PendingMultiUETransition(
        source_id=1,
        target_id=2,
        execution_start_s=29.9,
        completion_s=30.1,
        success=True,
    )
    rates, _, blocked = _allocate_epoch_service_segments(
        epoch_start_s=30.0,
        epoch_end_s=60.0,
        base_serving_satellite=np.asarray([-1, 2]),
        transitions=[pending, None],
        failed_epoch=np.asarray([False, False]),
        candidate_ids=np.asarray([[1, 2], [2, 1]]),
        achievable_rate_mbps=np.full((2, 2), 100.0),
        previous_average_rate_mbps=np.ones(2),
        multi_ue_config=multi,
    )
    assert np.allclose(
        rates,
        [50.0 * 29.9 / 30.0, (100.0 * 0.1 + 50.0 * 29.9) / 30.0],
    )
    assert np.allclose(blocked, 0.0)


def test_segment_allocator_uses_time_varying_realized_link_caps():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=1000.0,
        max_users_per_satellite=32,
        minimum_admission_rate_mbps=1.0,
    )

    def rate_provider(_user, _satellite, time_s):
        return 10.0 if time_s < 15.0 else 30.0

    rates, _, blocked = _allocate_epoch_service_segments(
        epoch_start_s=0.0,
        epoch_end_s=30.0,
        base_serving_satellite=np.asarray([1]),
        transitions=[None],
        failed_epoch=np.asarray([False]),
        candidate_ids=np.asarray([[1]]),
        achievable_rate_mbps=np.asarray([[99.0]]),
        previous_average_rate_mbps=np.ones(1),
        multi_ue_config=multi,
        rate_provider=rate_provider,
        rate_sample_interval_s=15.0,
    )
    assert np.allclose(rates, [20.0])
    assert np.allclose(blocked, 0.0)


def test_pending_target_service_does_not_require_new_ranking_candidate():
    cfg = load_config()
    multi = replace(
        cfg.multi_ue,
        satellite_capacity_mbps=100.0,
        max_users_per_satellite=32,
        minimum_admission_rate_mbps=1.0,
    )
    pending = _PendingMultiUETransition(
        source_id=1,
        target_id=2,
        execution_start_s=29.9,
        completion_s=30.1,
        success=True,
    )

    def rate_provider(_user, satellite, _time_s):
        return 40.0 if satellite == 1 else 80.0

    rates, _, _ = _allocate_epoch_service_segments(
        epoch_start_s=30.0,
        epoch_end_s=60.0,
        base_serving_satellite=np.asarray([-1]),
        transitions=[pending],
        failed_epoch=np.asarray([False]),
        # Target 2 is intentionally absent from this epoch's ranking set.
        candidate_ids=np.asarray([[1, 3]]),
        achievable_rate_mbps=np.asarray([[40.0, 20.0]]),
        previous_average_rate_mbps=np.ones(1),
        multi_ue_config=multi,
        rate_provider=rate_provider,
    )
    assert np.isclose(rates[0], 80.0 * 29.9 / 30.0)
