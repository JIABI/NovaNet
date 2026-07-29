"""Packet-level Poisson/FIFO latency simulation with CHO service blackouts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import TrafficConfig


@dataclass(frozen=True)
class LatencyTrace:
    arrival_time_s: np.ndarray
    serialization_ms: np.ndarray
    queueing_ms: np.ndarray
    cho_interruption_ms: np.ndarray
    protocol_processing_ms: np.ndarray
    fixed_network_ms: np.ndarray
    total_ms: np.ndarray
    dropped: int

    @property
    def transmission_only_ms(self) -> np.ndarray:
        return (
            self.serialization_ms
            + self.queueing_ms
            + self.protocol_processing_ms
        )


def _normalize_blackouts(
    blackouts: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    intervals = sorted(blackouts or [])
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            raise ValueError("Blackout intervals must have positive duration")
        if not merged or start > merged[-1][1]:
            merged.append((float(start), float(end)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], float(end)))
    return merged


def _finish_service(
    start_s: float,
    service_s: float,
    blackouts: list[tuple[float, float]],
) -> float:
    current = start_s
    remaining = service_s
    for blackout_start, blackout_end in blackouts:
        if blackout_end <= current:
            continue
        if blackout_start <= current < blackout_end:
            current = blackout_end
        if current + remaining <= blackout_start:
            return current + remaining
        if current < blackout_start:
            remaining -= blackout_start - current
            current = blackout_end
    return current + remaining


def _available_clock(
    wall_time_s: np.ndarray,
    blackouts: list[tuple[float, float]],
) -> np.ndarray:
    """Map wall time to cumulative time during which the server is available."""

    wall = np.asarray(wall_time_s, dtype=float)
    available = wall.copy()
    for start, end in blackouts:
        available -= np.clip(wall - start, 0.0, end - start)
    return available


def _wall_clock(
    available_time_s: np.ndarray,
    blackouts: list[tuple[float, float]],
    *,
    after_boundary: bool,
) -> np.ndarray:
    """Invert the available-service clock for sorted blackout intervals."""

    work = np.asarray(available_time_s, dtype=float)
    if not blackouts:
        return work.copy()
    durations = np.asarray([end - start for start, end in blackouts], dtype=float)
    cumulative_before = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    virtual_starts = np.asarray(
        [start for start, _end in blackouts], dtype=float
    ) - cumulative_before
    side = "right" if after_boundary else "left"
    count = np.searchsorted(virtual_starts, work, side=side)
    cumulative = np.concatenate(([0.0], np.cumsum(durations)))
    return work + cumulative[count]


def simulate_fifo_latency(
    traffic: TrafficConfig,
    duration_s: float,
    rate_times_s: np.ndarray,
    rates_bps: np.ndarray,
    *,
    handover_blackouts: list[tuple[float, float]] | None = None,
    seed: int = 0,
) -> LatencyTrace:
    """Simulate a work-conserving, drop-tail, single-server FIFO queue."""

    if traffic.arrival_process != "poisson":
        raise ValueError("Only the documented Poisson arrival process is supported")
    if traffic.queue_discipline != "fifo":
        raise ValueError("Only the documented FIFO service rule is supported")
    times = np.asarray(rate_times_s, dtype=float)
    rates = np.asarray(rates_bps, dtype=float)
    if times.ndim != 1 or rates.shape != times.shape or len(times) == 0:
        raise ValueError("rate_times_s and rates_bps must be nonempty 1-D arrays")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("rate_times_s must be strictly increasing")

    rng = np.random.default_rng(seed)
    interarrival = rng.exponential(
        1.0 / traffic.arrival_rate_packets_s,
        int(np.ceil(duration_s * traffic.arrival_rate_packets_s * 1.1)) + 32,
    )
    arrivals = np.cumsum(interarrival)
    while arrivals[-1] < duration_s:
        extra = rng.exponential(
            1.0 / traffic.arrival_rate_packets_s,
            max(32, len(interarrival) // 4),
        )
        arrivals = np.concatenate((arrivals, arrivals[-1] + np.cumsum(extra)))
    arrivals = arrivals[arrivals < duration_s]

    blackouts = _normalize_blackouts(handover_blackouts)
    packet_bits = 8.0 * traffic.packet_size_bytes
    rate_index = np.searchsorted(times, arrivals, side="right") - 1
    rate_index = np.maximum(rate_index, 0)
    packet_service_s = packet_bits / np.maximum(rates[rate_index], 1.0)

    # In the server-available time coordinate, preemptive blackouts disappear
    # and Lindley's FCFS recursion is vectorizable. This is the common fast
    # path for the paper load.
    arrival_work = _available_clock(arrivals, blackouts)
    cumulative_service = np.cumsum(packet_service_s)
    cumulative_before = cumulative_service - packet_service_s
    server_origin = np.maximum.accumulate(arrival_work - cumulative_before)
    departure_work = cumulative_service + server_origin
    service_start_work = departure_work - packet_service_s
    departure_wall = _wall_clock(
        departure_work,
        blackouts,
        after_boundary=False,
    )
    service_start_wall = _wall_clock(
        service_start_work,
        blackouts,
        after_boundary=True,
    )

    completed_before_arrival = np.searchsorted(
        departure_wall,
        arrivals,
        side="right",
    )
    active_before_arrival = (
        np.arange(len(arrivals), dtype=int) - completed_before_arrival
    )

    if active_before_arrival.max(initial=0) < traffic.queue_capacity_packets:
        accepted_arrivals = arrivals
        serialization_ms = 1e3 * packet_service_s
        queueing_ms = 1e3 * np.maximum(0.0, service_start_wall - arrivals)
        interruption_ms = 1e3 * np.maximum(
            0.0,
            departure_wall - service_start_wall - packet_service_s,
        )
        dropped = 0
    else:
        # True finite-buffer drop-tail fallback. ``active_departures`` contains
        # accepted packets that have not completed at the current arrival.
        count_all = len(arrivals)
        accepted_arrivals = np.empty(count_all, dtype=float)
        serialization_ms = np.empty(count_all, dtype=float)
        queueing_ms = np.empty(count_all, dtype=float)
        interruption_ms = np.empty(count_all, dtype=float)
        active_departures: deque[float] = deque()
        last_departure = 0.0
        accepted = 0
        dropped = 0
        for arrival, service_s in zip(arrivals, packet_service_s):
            while active_departures and active_departures[0] <= arrival:
                active_departures.popleft()
            if len(active_departures) >= traffic.queue_capacity_packets:
                dropped += 1
                continue

            service_start = max(float(arrival), last_departure)
            completion = _finish_service(
                service_start,
                float(service_s),
                blackouts,
            )
            accepted_arrivals[accepted] = arrival
            serialization_ms[accepted] = 1e3 * service_s
            queueing_ms[accepted] = 1e3 * (service_start - arrival)
            interruption_ms[accepted] = max(
                0.0,
                1e3 * (completion - service_start - service_s),
            )
            accepted += 1
            last_departure = completion
            active_departures.append(completion)

        accepted_arrivals = accepted_arrivals[:accepted]
        serialization_ms = serialization_ms[:accepted]
        queueing_ms = queueing_ms[:accepted]
        interruption_ms = interruption_ms[:accepted]
    total_ms = (
        queueing_ms
        + serialization_ms
        + interruption_ms
        + traffic.protocol_processing_ms
        + traffic.fixed_network_delay_ms
    )
    count = len(accepted_arrivals)
    return LatencyTrace(
        arrival_time_s=accepted_arrivals,
        serialization_ms=serialization_ms,
        queueing_ms=queueing_ms,
        cho_interruption_ms=interruption_ms,
        protocol_processing_ms=np.full(
            count, traffic.protocol_processing_ms, dtype=float
        ),
        fixed_network_ms=np.full(
            count, traffic.fixed_network_delay_ms, dtype=float
        ),
        total_ms=total_ms,
        dropped=dropped,
    )


def latency_summary(trace: LatencyTrace) -> dict[str, float]:
    if len(trace.total_ms) == 0:
        raise ValueError("Cannot summarize an empty latency trace")
    return {
        "packets": float(len(trace.total_ms)),
        "dropped": float(trace.dropped),
        "transmission_only_mean_ms": float(trace.transmission_only_ms.mean()),
        "p50_ms": float(np.percentile(trace.total_ms, 50.0)),
        "p95_ms": float(np.percentile(trace.total_ms, 95.0)),
        "p99_ms": float(np.percentile(trace.total_ms, 99.0)),
        "p99_9_ms": float(np.percentile(trace.total_ms, 99.9)),
        "exceed_100_percent": float(100.0 * np.mean(trace.total_ms > 100.0)),
    }
