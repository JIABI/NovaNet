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
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("Blackout interval endpoints must be finite")
        if start < 0.0:
            raise ValueError("Blackout intervals cannot start before t=0")
        if end <= start:
            raise ValueError("Blackout intervals must have positive duration")
        if not merged or start > merged[-1][1]:
            merged.append((float(start), float(end)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], float(end)))
    return merged


@dataclass(frozen=True)
class _ServiceCurve:
    """Integral and inverse of a piecewise-constant link service process."""

    boundaries_s: np.ndarray
    rates_bps: np.ndarray
    cumulative_bits: np.ndarray

    def capacity_at(self, wall_time_s: np.ndarray) -> np.ndarray:
        wall = np.asarray(wall_time_s, dtype=float)
        index = np.searchsorted(self.boundaries_s, wall, side="right") - 1
        index = np.clip(index, 0, len(self.boundaries_s) - 1)
        return self.cumulative_bits[index] + self.rates_bps[index] * (
            wall - self.boundaries_s[index]
        )

    def wall_time_at(
        self,
        service_bits: np.ndarray,
        *,
        after_boundary: bool,
    ) -> np.ndarray:
        """Invert cumulative service, choosing before/after zero-rate plateaus."""

        target = np.asarray(service_bits, dtype=float)
        output = np.empty_like(target)
        cumulative = self.cumulative_bits
        boundaries = self.boundaries_s
        rates = self.rates_bps

        if after_boundary:
            lower = np.searchsorted(cumulative, target, side="right") - 1
            lower = np.clip(lower, 0, len(boundaries) - 1)
            rate = rates[lower]
            output.fill(np.inf)
            positive = rate > 0.0
            output[positive] = boundaries[lower[positive]] + np.divide(
                target[positive] - cumulative[lower[positive]],
                rate[positive],
            )
            return output

        upper = np.searchsorted(cumulative, target, side="left")
        exact = (upper < len(cumulative)) & (
            cumulative[np.minimum(upper, len(cumulative) - 1)] == target
        )
        exact_index = np.minimum(upper, len(boundaries) - 1)
        output[exact] = boundaries[exact_index[exact]]
        between = ~exact
        lower = np.clip(upper[between] - 1, 0, len(boundaries) - 1)
        rate = rates[lower]
        output[between] = np.inf
        positive = rate > 0.0
        between_positions = np.flatnonzero(between)
        output[between_positions[positive]] = (
            boundaries[lower[positive]]
            + np.divide(
                target[between_positions[positive]]
                - cumulative[lower[positive]],
                rate[positive],
            )
        )
        return output

    def wall_time_at_scalar(
        self,
        service_bits: float,
        *,
        after_boundary: bool,
    ) -> float:
        """Scalar inverse used by the finite-buffer event fallback."""

        target = float(service_bits)
        cumulative = self.cumulative_bits
        if after_boundary:
            lower = int(np.searchsorted(cumulative, target, side="right") - 1)
            lower = max(0, min(lower, len(cumulative) - 1))
            rate = float(self.rates_bps[lower])
            if rate <= 0.0:
                return float("inf")
            return float(
                self.boundaries_s[lower]
                + (target - cumulative[lower]) / rate
            )

        upper = int(np.searchsorted(cumulative, target, side="left"))
        if upper < len(cumulative) and cumulative[upper] == target:
            return float(self.boundaries_s[upper])
        lower = max(0, min(upper - 1, len(cumulative) - 1))
        rate = float(self.rates_bps[lower])
        if rate <= 0.0:
            return float("inf")
        return float(
            self.boundaries_s[lower]
            + (target - cumulative[lower]) / rate
        )


def _build_service_curve(
    times: np.ndarray,
    rates: np.ndarray,
    blackouts: list[tuple[float, float]],
) -> _ServiceCurve:
    """Build C(t)=integral rate(u) du with CHO intervals set to zero."""

    boundaries = np.unique(
        np.asarray(
            [0.0, *times.tolist(), *(x for interval in blackouts for x in interval)],
            dtype=float,
        )
    )
    boundaries = boundaries[boundaries >= 0.0]
    if len(boundaries) == 0 or boundaries[0] > 0.0:
        boundaries = np.insert(boundaries, 0, 0.0)

    segment_rates = np.empty(len(boundaries), dtype=float)
    for index, start in enumerate(boundaries):
        if index + 1 < len(boundaries):
            probe = 0.5 * (start + boundaries[index + 1])
        else:
            probe = start + 1e-9
        rate_index = int(np.searchsorted(times, probe, side="right") - 1)
        rate_index = max(0, min(rate_index, len(rates) - 1))
        blocked = any(left <= probe < right for left, right in blackouts)
        segment_rates[index] = 0.0 if blocked else max(float(rates[rate_index]), 0.0)

    cumulative = np.zeros(len(boundaries), dtype=float)
    if len(boundaries) > 1:
        cumulative[1:] = np.cumsum(
            segment_rates[:-1] * np.diff(boundaries)
        )
    return _ServiceCurve(boundaries, segment_rates, cumulative)


def _blackout_overlap(
    starts_s: np.ndarray,
    ends_s: np.ndarray,
    blackouts: list[tuple[float, float]],
) -> np.ndarray:
    overlap = np.zeros_like(starts_s, dtype=float)
    for left, right in blackouts:
        overlap += np.maximum(
            0.0,
            np.minimum(ends_s, right) - np.maximum(starts_s, left),
        )
    return overlap


def _blackout_overlap_scalar(
    start_s: float,
    end_s: float,
    blackouts: list[tuple[float, float]],
) -> float:
    return float(
        sum(
            max(0.0, min(end_s, right) - max(start_s, left))
            for left, right in blackouts
        )
    )


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

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if traffic.arrival_process != "poisson":
        raise ValueError("Only the documented Poisson arrival process is supported")
    if traffic.queue_discipline != "fifo":
        raise ValueError("Only the documented FIFO service rule is supported")
    times = np.asarray(rate_times_s, dtype=float)
    rates = np.asarray(rates_bps, dtype=float)
    if times.ndim != 1 or rates.shape != times.shape or len(times) == 0:
        raise ValueError("rate_times_s and rates_bps must be nonempty 1-D arrays")
    if not np.isfinite(times).all() or not np.isfinite(rates).all():
        raise ValueError("rate trace values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("rate_times_s must be strictly increasing")
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-9):
        raise ValueError("rate trace must start at t=0")
    if np.any(rates < 0.0):
        raise ValueError("rates_bps must be nonnegative")

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
    curve = _build_service_curve(times, rates, blackouts)

    # In cumulative-service coordinates every accepted packet requires the
    # same number of bits.  Lindley's FCFS recursion is therefore vectorized
    # even though wall-clock rate varies and may temporarily be zero.
    arrival_work = curve.capacity_at(arrivals)
    cumulative_service = packet_bits * (
        np.arange(len(arrivals), dtype=float) + 1.0
    )
    cumulative_before = cumulative_service - packet_bits
    server_origin = np.maximum.accumulate(arrival_work - cumulative_before)
    departure_work = cumulative_service + server_origin
    service_start_work = departure_work - packet_bits
    departure_wall = curve.wall_time_at(
        departure_work, after_boundary=False
    )
    service_start_wall = curve.wall_time_at(
        service_start_work, after_boundary=True
    )

    completed_before_arrival = np.searchsorted(
        departure_wall,
        arrivals,
        side="right",
    )
    active_before_arrival = (
        np.arange(len(arrivals), dtype=int) - completed_before_arrival
    )

    if (
        np.isfinite(departure_wall).all()
        and active_before_arrival.max(initial=0)
        < traffic.queue_capacity_packets
    ):
        accepted_arrivals = arrivals
        interruption_s = _blackout_overlap(
            service_start_wall, departure_wall, blackouts
        )
        serialization_ms = 1e3 * np.maximum(
            0.0, departure_wall - service_start_wall - interruption_s
        )
        queueing_ms = 1e3 * np.maximum(0.0, service_start_wall - arrivals)
        interruption_ms = 1e3 * interruption_s
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
        accepted = 0
        dropped = 0
        last_departure_work = 0.0
        for arrival, arrival_capacity in zip(arrivals, arrival_work):
            while active_departures and active_departures[0] <= arrival:
                active_departures.popleft()
            if len(active_departures) >= traffic.queue_capacity_packets:
                dropped += 1
                continue

            service_start_capacity = max(
                float(arrival_capacity), last_departure_work
            )
            completion_capacity = service_start_capacity + packet_bits
            service_start = curve.wall_time_at_scalar(
                service_start_capacity,
                after_boundary=True,
            )
            completion = curve.wall_time_at_scalar(
                completion_capacity,
                after_boundary=False,
            )
            if not np.isfinite(completion):
                dropped += 1
                continue
            interruption_s = _blackout_overlap_scalar(
                service_start,
                completion,
                blackouts,
            )
            accepted_arrivals[accepted] = arrival
            serialization_ms[accepted] = 1e3 * max(
                0.0, completion - service_start - interruption_s
            )
            queueing_ms[accepted] = 1e3 * (service_start - arrival)
            interruption_ms[accepted] = 1e3 * interruption_s
            accepted += 1
            last_departure_work = completion_capacity
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
    components = (
        trace.serialization_ms,
        trace.queueing_ms,
        trace.cho_interruption_ms,
        trace.protocol_processing_ms,
        trace.fixed_network_ms,
        trace.total_ms,
    )
    if not all(np.isfinite(component).all() for component in components):
        raise ValueError("Latency trace contains non-finite values")
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
