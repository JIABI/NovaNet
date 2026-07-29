from dataclasses import replace

import numpy as np

from novanet.config import load_config
from novanet.latency import latency_summary, simulate_fifo_latency
from novanet.multi_ue import SynchronousAssociationScheduler, jain_fairness


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
