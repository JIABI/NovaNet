"""Single-UE link/CHO/packet simulation used by all experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .channel import LinkBudget, MeasurementTracker, RealizedChannelTrace
from .config import NovaNetConfig
from .ephemeris import Ephemeris
from .forecast import build_forecast_sequence
from .geometry import GeometryState, UETrajectory, geometry_state
from .handover import counterfactual_hof_label, evaluate_cho_attempt
from .latency import latency_summary, simulate_fifo_latency
from .policies import Policy
from .soft_dp import soft_dynamic_program


@dataclass(frozen=True)
class Scenario:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    measurement_noise_std_db: float = 0.0
    staleness_steps: int = 0
    rain_rate_mm_h: float | None = None
    rain_attenuation_db: float = 0.0
    blockage_loss_db: float = 0.0
    blockage_probability: float = 0.0
    receive_gain_model: Callable[[GeometryState], float] | None = None

    def __post_init__(self) -> None:
        numeric = {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "speed_kmh": self.speed_kmh,
            "heading_deg": self.heading_deg,
            "measurement_noise_std_db": self.measurement_noise_std_db,
            "rain_attenuation_db": self.rain_attenuation_db,
            "blockage_loss_db": self.blockage_loss_db,
            "blockage_probability": self.blockage_probability,
        }
        if not all(np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("Scenario values must be finite")
        if self.rain_rate_mm_h is not None and (
            not np.isfinite(float(self.rain_rate_mm_h))
            or self.rain_rate_mm_h < 0.0
        ):
            raise ValueError("rain_rate_mm_h must be finite and nonnegative")
        if self.altitude_m < 0.0 or self.speed_kmh < 0.0:
            raise ValueError("Scenario altitude and speed must be nonnegative")
        if self.measurement_noise_std_db < 0.0:
            raise ValueError("measurement_noise_std_db must be nonnegative")
        if self.staleness_steps < 0:
            raise ValueError("staleness_steps must be nonnegative")
        if self.rain_attenuation_db < 0.0 or self.blockage_loss_db < 0.0:
            raise ValueError("extra attenuation values must be nonnegative")
        if not 0.0 <= self.blockage_probability <= 1.0:
            raise ValueError("blockage_probability must be in [0,1]")


@dataclass(frozen=True)
class SimulationMetrics:
    method: str
    mean_rate_mbps: float
    effective_throughput_mbps: float
    handovers: int
    handover_failures: int
    hof_percent: float
    outage_percent: float
    cho_hit_percent: float
    transmission_only_mean_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p99_9_latency_ms: float
    exceed_100_percent: float
    ping_pong_count: int
    ping_pong_percent: float
    delivered_packets: int
    dropped_packets: int
    mean_target_cost: float = float("nan")
    mean_oracle_target_cost: float = float("nan")
    target_cost_sum: float = float("nan")
    oracle_target_cost_sum: float = float("nan")
    paired_cost_windows: int = 0


@dataclass(frozen=True)
class _PendingTransition:
    source_id: int
    target_id: int
    execution_start_s: float
    completion_s: float
    success: bool


def _blockage_loss_at(
    channel_trace: RealizedChannelTrace,
    satellite_id: int,
    time_s: float,
    scenario: Scenario,
) -> float:
    return (
        scenario.blockage_loss_db
        if channel_trace.blockage_active(
            satellite_id,
            time_s,
            scenario.blockage_probability,
        )
        else 0.0
    )


def _service_link_at_time(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    trajectory: UETrajectory,
    channel_trace: RealizedChannelTrace,
    satellite_id: int | None,
    time_s: float,
    scenario: Scenario,
) -> tuple[float, float]:
    if satellite_id is None or satellite_id < 0:
        return 0.0, -100.0
    ue_position, ue_velocity = trajectory.state_at(time_s)
    sat_position, sat_velocity = ephemeris.state_at_time(satellite_id, time_s)
    if not (
        np.all(np.isfinite(sat_position))
        and np.all(np.isfinite(sat_velocity))
    ):
        return 0.0, -100.0
    state = geometry_state(
        ue_position,
        ue_velocity,
        sat_position,
        sat_velocity,
    )
    if state.elevation_deg < config.experiment.minimum_elevation_deg:
        return 0.0, -100.0
    link = channel_trace.evaluate(
        state,
        satellite_id,
        time_s,
        rain_rate_mm_h=scenario.rain_rate_mm_h,
        additional_loss_db=(
            scenario.rain_attenuation_db
            + _blockage_loss_at(
                channel_trace,
                satellite_id,
                time_s,
                scenario,
            )
        ),
    )
    return float(link.rate_bps), float(link.sinr_db)


def _epoch_service_segments(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    trajectory: UETrajectory,
    channel_trace: RealizedChannelTrace,
    scenario: Scenario,
    start_s: float,
    end_s: float,
    incumbent_at_start: int | None,
    transition: _PendingTransition | None,
    blackouts: list[tuple[float, float]],
) -> list[tuple[float, float, float, float, bool]]:
    """Return piecewise service rows, splitting at a successful completion."""

    boundaries = list(
        np.arange(start_s, end_s, ephemeris.step_s, dtype=float)
    )
    if not boundaries or not np.isclose(boundaries[0], start_s):
        boundaries.insert(0, float(start_s))
    if (
        transition is not None
        and transition.success
        and start_s < transition.completion_s < end_s
    ):
        boundaries.append(float(transition.completion_s))
    boundaries.append(float(end_s))
    ordered = sorted({round(value, 9) for value in boundaries})
    rows: list[tuple[float, float, float, float, bool]] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        duration = right - left
        if duration <= 0.0:
            continue
        satellite_id = incumbent_at_start
        if (
            transition is not None
            and transition.success
            and left >= transition.completion_s - 1e-9
        ):
            satellite_id = transition.target_id
        rate, sinr = _service_link_at_time(
            config,
            ephemeris,
            trajectory,
            channel_trace,
            satellite_id,
            left,
            scenario,
        )
        blackout_duration = _overlap_seconds(left, right, blackouts)
        service_fraction = 1.0 - min(
            blackout_duration / duration,
            1.0,
        )
        outage = sinr < config.channel.outage_threshold_db
        link_valid_rate = 0.0 if outage else rate
        rows.append(
            (
                left,
                duration,
                rate,
                link_valid_rate * service_fraction,
                outage,
            )
        )
    return rows


def _overlap_seconds(
    start_s: float,
    end_s: float,
    intervals: list[tuple[float, float]],
) -> float:
    return float(
        sum(
            max(0.0, min(end_s, right) - max(start_s, left))
            for left, right in intervals
        )
    )


def _link_snr_at_offset(
    config: NovaNetConfig,
    budget: LinkBudget,
    ephemeris: Ephemeris,
    trajectory: UETrajectory,
    ephemeris_index: int,
    satellite_id: int,
    offset_s: float,
    rain_rate_mm_h: float | None,
    rain_attenuation_db: float,
    blockage_loss_db: float,
    channel_trace: RealizedChannelTrace | None = None,
    blockage_probability: float | None = None,
) -> float:
    base_time_s = ephemeris.time_s(ephemeris_index)
    sample_time_s = base_time_s + offset_s
    ue_position, ue_velocity = trajectory.state_at(sample_time_s)
    sat_position, sat_velocity = ephemeris.state_at_time(
        satellite_id,
        sample_time_s,
    )
    if not (
        np.all(np.isfinite(sat_position))
        and np.all(np.isfinite(sat_velocity))
    ):
        return -100.0
    state = geometry_state(
        ue_position, ue_velocity, sat_position, sat_velocity
    )
    if state.elevation_deg < config.experiment.minimum_elevation_deg:
        return -100.0
    active_blockage_loss_db = float(blockage_loss_db)
    if channel_trace is not None and blockage_probability is not None:
        active_blockage_loss_db = (
            float(blockage_loss_db)
            if channel_trace.blockage_active(
                satellite_id,
                sample_time_s,
                blockage_probability,
            )
            else 0.0
        )
    evaluator = (
        channel_trace.evaluate(
            state,
            satellite_id,
            sample_time_s,
            rain_rate_mm_h=rain_rate_mm_h,
            additional_loss_db=(
                rain_attenuation_db + active_blockage_loss_db
            ),
        )
        if channel_trace is not None
        else budget.evaluate(
            state,
            stochastic=False,
            rain_rate_mm_h=rain_rate_mm_h,
            additional_loss_db=(
                rain_attenuation_db + active_blockage_loss_db
            ),
        )
    )
    return evaluator.sinr_db


def _realized_first_step_costs(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    trajectory: UETrajectory,
    channel_trace: RealizedChannelTrace,
    budget: LinkBudget,
    scenario: Scenario,
    sequence,
    decision_index: int,
    initial_freeze: int,
) -> np.ndarray:
    """Replay the non-causal same-energy teacher for one decision epoch."""

    full_horizon, candidates = sequence.valid_mask.shape
    populated = sequence.valid_mask.any(axis=1)
    if not populated[0]:
        raise RuntimeError("Oracle replay has no feasible current candidate")
    first_empty = np.flatnonzero(~populated)
    # A true no-coverage future epoch is an early terminal state, not an
    # invitation to insert a fabricated satellite.  Truncate the replay at
    # the first such epoch, matching deployment's no-coverage fail-safe.
    horizon = int(first_empty[0]) if len(first_empty) else full_horizon
    replay_valid = sequence.valid_mask[:horizon]
    stride = int(
        round(config.experiment.decision_interval_s / ephemeris.step_s)
    )
    realized_rate_mbps = np.zeros((horizon, candidates), dtype=np.float64)
    realized_hof = np.zeros(
        (horizon, candidates, candidates), dtype=np.float64
    )

    for h in range(horizon):
        event_index = decision_index + h * stride
        cache: dict[tuple[int, int], float] = {}

        def event_sinr(satellite_id: int, offset_s: float) -> float:
            event_bin = int(round(offset_s / config.handover.event_step_s))
            key = (satellite_id, event_bin)
            if key not in cache:
                cache[key] = _link_snr_at_offset(
                    config,
                    budget,
                    ephemeris,
                    trajectory,
                    event_index,
                    satellite_id,
                    event_bin * config.handover.event_step_s,
                    scenario.rain_rate_mm_h,
                    scenario.rain_attenuation_db,
                    scenario.blockage_loss_db,
                    channel_trace,
                    scenario.blockage_probability,
                )
            return cache[key]

        for target, satellite_id in enumerate(sequence.candidate_ids):
            if satellite_id < 0 or not replay_valid[h, target]:
                continue
            sinr_db = event_sinr(int(satellite_id), 0.0)
            realized_rate_mbps[h, target] = (
                config.channel.implementation_efficiency
                * config.channel.bandwidth_hz
                * np.log2(1.0 + 10.0 ** (sinr_db / 10.0))
                / 1e6
            )

        for source, source_id in enumerate(sequence.candidate_ids):
            if source_id < 0:
                continue
            source_trace = (
                lambda offset, sat=int(source_id): event_sinr(sat, offset)
            )
            for target, target_id in enumerate(sequence.candidate_ids):
                if (
                    target == source
                    or target_id < 0
                    or not replay_valid[h, target]
                ):
                    continue
                target_trace = (
                    lambda offset, sat=int(target_id): event_sinr(sat, offset)
                )
                outcome = evaluate_cho_attempt(
                    source_trace,
                    target_trace,
                    config.handover,
                    config.channel.outage_threshold_db,
                    event_step_s=config.handover.event_step_s,
                    monitoring_horizon_s=(
                        config.experiment.decision_interval_s
                    ),
                )
                realized_hof[h, source, target] = float(
                    not outcome.success
                    if outcome.attempted
                    else counterfactual_hof_label(
                        target_trace,
                        config.handover,
                        config.channel.outage_threshold_db,
                        event_step_s=config.handover.event_step_s,
                    )
                )

    normalized_ttl = (
        sequence.ttl_s[:horizon].astype(np.float64)
        / config.planner.ttl_reference_s
    )
    state_cost = -(
        config.planner.alpha
        * realized_rate_mbps
        / config.planner.rate_reference_mbps
        + config.planner.beta * normalized_ttl
    )
    transition_cost = np.zeros_like(realized_hof)
    for source in range(candidates):
        for target in range(candidates):
            if source == target:
                continue
            transition_cost[:, source, target] = (
                config.planner.c0
                + config.planner.c1 * normalized_ttl[:, source]
                + config.planner.c2 * realized_hof[:, source, target]
            )
    result = soft_dynamic_program(
        state_cost=torch.as_tensor(state_cost[None], dtype=torch.float64),
        transition_cost=torch.as_tensor(
            transition_cost[None], dtype=torch.float64
        ),
        current_idx=torch.as_tensor([sequence.current_idx]),
        valid_mask=torch.as_tensor(replay_valid[None]),
        temperature=config.planner.teacher_temperature,
        freeze_steps=config.handover.freeze_steps,
        initial_freeze=torch.as_tensor([initial_freeze]),
        hard=True,
    )
    return result.first_cost[0].detach().cpu().numpy()


def simulate_single_user(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    policy: Policy,
    scenario: Scenario,
    *,
    seed: int | None = None,
    planning_ephemeris: Ephemeris | None = None,
    compute_oracle_cost: bool = False,
) -> SimulationMetrics:
    """Run one held-out episode.

    ``planning_ephemeris`` is optional and is used only for the causal
    geometry/TTL forecast.  Realized CHO and service links always use
    ``ephemeris``.  This dual input is required by the ephemeris-aging test and
    prevents stale priors from replacing the ground-truth evaluation trace.
    """

    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    budget = LinkBudget(
        config.channel,
        seed=config.experiment.seed if seed is None else seed,
        receive_gain_model=scenario.receive_gain_model,
    )
    channel_trace = RealizedChannelTrace(
        config.channel,
        seed=config.experiment.seed if seed is None else seed,
        event_step_s=config.handover.event_step_s,
        receive_gain_model=scenario.receive_gain_model,
    )
    tracker = MeasurementTracker(config.channel)
    trajectory = UETrajectory(
        latitude_deg=scenario.latitude_deg,
        longitude_deg=scenario.longitude_deg,
        altitude_m=scenario.altitude_m,
        speed_m_s=scenario.speed_kmh / 3.6,
        heading_deg=scenario.heading_deg,
    )
    planner_ephemeris = planning_ephemeris or ephemeris
    if (
        planner_ephemeris.num_satellites != ephemeris.num_satellites
        or not np.isclose(planner_ephemeris.step_s, ephemeris.step_s)
        or planner_ephemeris.num_steps < ephemeris.num_steps
    ):
        raise ValueError(
            "planning_ephemeris must share the realized satellite order/time grid"
        )
    stride = int(
        round(config.experiment.decision_interval_s / ephemeris.step_s)
    )
    duration_steps = int(config.experiment.duration_s / ephemeris.step_s)
    final_index = min(
        duration_steps,
        ephemeris.num_steps
        - stride * config.planner.horizon_steps
        - 1,
    )
    decision_indices = range(0, final_index, stride)
    incumbent_id: int | None = None
    freeze_left = 0
    handovers = 0
    failures = 0
    blackouts: list[tuple[float, float]] = []
    rate_times: list[float] = []
    raw_rates: list[float] = []
    link_service_rates: list[float] = []
    effective_rates: list[float] = []
    sample_durations: list[float] = []
    outage_duration_s = 0.0
    total_duration_s = 0.0
    successful_transitions: list[tuple[float, int, int]] = []
    ping_pong_count = 0
    pending_transition: _PendingTransition | None = None
    selected_target_costs: list[float] = []
    paired_oracle_costs: list[float] = []

    def apply_completion(transition: _PendingTransition) -> None:
        nonlocal incumbent_id, freeze_left, ping_pong_count
        if not transition.success:
            return
        if (
            successful_transitions
            and successful_transitions[-1][1] == transition.target_id
            and successful_transitions[-1][2] == transition.source_id
            and transition.completion_s - successful_transitions[-1][0]
            <= config.handover.statistics_window_s
        ):
            ping_pong_count += 1
        successful_transitions.append(
            (
                transition.completion_s,
                transition.source_id,
                transition.target_id,
            )
        )
        incumbent_id = transition.target_id
        freeze_left = config.handover.freeze_steps

    def record_service(
        start_s: float,
        end_s: float,
        incumbent_at_start: int | None,
        transition: _PendingTransition | None,
    ) -> None:
        nonlocal outage_duration_s, total_duration_s
        for time_s, duration_s, raw_rate, effective_rate, outage in (
            _epoch_service_segments(
                config,
                ephemeris,
                trajectory,
                channel_trace,
                scenario,
                start_s,
                end_s,
                incumbent_at_start,
                transition,
                blackouts,
            )
        ):
            rate_times.append(time_s)
            sample_durations.append(duration_s)
            raw_rates.append(raw_rate)
            link_service_rates.append(0.0 if outage else raw_rate)
            effective_rates.append(effective_rate)
            total_duration_s += duration_s
            outage_duration_s += duration_s if outage else 0.0

    for decision_index in decision_indices:
        decision_time_s = ephemeris.time_s(decision_index)
        epoch_end_s = min(
            decision_time_s + config.experiment.decision_interval_s,
            ephemeris.time_s(final_index),
            float(config.experiment.duration_s),
        )
        if epoch_end_s <= decision_time_s:
            continue

        # A CHO that began just before the preceding decision boundary cannot
        # be replaced.  This epoch observes its realized completion and does
        # not configure a second target concurrently.
        if pending_transition is not None:
            if pending_transition.completion_s <= decision_time_s + 1e-9:
                apply_completion(pending_transition)
                pending_transition = None
            else:
                source_at_start = incumbent_id
                record_service(
                    decision_time_s,
                    epoch_end_s,
                    source_at_start,
                    pending_transition,
                )
                if pending_transition.completion_s <= epoch_end_s + 1e-9:
                    apply_completion(pending_transition)
                    pending_transition = None
                continue

        try:
            preliminary = build_forecast_sequence(
                config,
                planner_ephemeris,
                trajectory,
                decision_index,
                tracker,
                incumbent_id=incumbent_id,
                link_budget=budget,
                initial_freeze=freeze_left,
            )
        except RuntimeError:
            # A no-candidate epoch remains on the service timeline.  The
            # current incumbent, if any, is evaluated and normally yields
            # zero service once it has left visibility.
            record_service(
                decision_time_s,
                epoch_end_s,
                incumbent_id,
                None,
            )
            freeze_left = max(freeze_left - 1, 0)
            continue
        if incumbent_id is None:
            incumbent_id = int(
                preliminary.candidate_ids[preliminary.current_idx]
            )

        stale_index = max(
            0,
            decision_index - scenario.staleness_steps * stride,
        )
        report_time = ephemeris.time_s(stale_index)
        for local, satellite_id in enumerate(preliminary.candidate_ids):
            if satellite_id < 0 or not preliminary.valid_mask[0, local]:
                continue
            # A stale report uses the earlier realized link value, not a
            # current sample carrying an artificial old timestamp.
            snr = _link_snr_at_offset(
                config,
                budget,
                ephemeris,
                trajectory,
                stale_index,
                int(satellite_id),
                0.0,
                scenario.rain_rate_mm_h,
                scenario.rain_attenuation_db,
                scenario.blockage_loss_db,
                channel_trace,
                scenario.blockage_probability,
            )
            if snr <= -100.0:
                continue
            measured = snr + channel_trace.measurement_noise_db(
                int(satellite_id),
                report_time,
                scenario.measurement_noise_std_db,
            )
            tracker.update(int(satellite_id), float(measured), report_time)

        sequence = build_forecast_sequence(
            config,
            planner_ephemeris,
            trajectory,
            decision_index,
            tracker,
            incumbent_id=incumbent_id,
            link_budget=budget,
            initial_freeze=freeze_left,
        )
        oracle_costs = None
        is_oracle = bool(getattr(policy, "is_noncausal_oracle", False))
        if compute_oracle_cost or is_oracle:
            oracle_costs = _realized_first_step_costs(
                config,
                ephemeris,
                trajectory,
                channel_trace,
                budget,
                scenario,
                sequence,
                decision_index,
                freeze_left,
            )
        candidate_local = (
            int(np.argmin(oracle_costs))
            if is_oracle
            else policy.choose(sequence)
        )
        target_id = int(sequence.candidate_ids[candidate_local])
        source_local = np.where(sequence.candidate_ids == incumbent_id)[0]
        source_visible = bool(
            len(source_local)
            and sequence.valid_mask[0, int(source_local[0])]
        )
        if freeze_left > 0 and source_visible:
            target_id = int(incumbent_id)
            candidate_local = int(source_local[0])
        if oracle_costs is not None and np.isfinite(oracle_costs[candidate_local]):
            selected_target_costs.append(float(oracle_costs[candidate_local]))
            finite_oracle = oracle_costs[np.isfinite(oracle_costs)]
            if finite_oracle.size:
                paired_oracle_costs.append(float(np.min(finite_oracle)))

        epoch_transition: _PendingTransition | None = None
        source_at_start = incumbent_id
        if target_id >= 0 and target_id != incumbent_id:
            source_id = int(incumbent_id)
            source = lambda offset: _link_snr_at_offset(
                config,
                budget,
                ephemeris,
                trajectory,
                decision_index,
                source_id,
                offset,
                scenario.rain_rate_mm_h,
                scenario.rain_attenuation_db,
                scenario.blockage_loss_db,
                channel_trace,
                scenario.blockage_probability,
            )
            target = lambda offset: _link_snr_at_offset(
                config,
                budget,
                ephemeris,
                trajectory,
                decision_index,
                target_id,
                offset,
                scenario.rain_rate_mm_h,
                scenario.rain_attenuation_db,
                scenario.blockage_loss_db,
                channel_trace,
                scenario.blockage_probability,
            )
            outcome = evaluate_cho_attempt(
                source,
                target,
                config.handover,
                config.channel.outage_threshold_db,
                event_step_s=config.handover.event_step_s,
                monitoring_horizon_s=(
                    config.experiment.decision_interval_s
                ),
            )
            if outcome.attempted:
                handovers += 1
                execution_start_s = (
                    decision_time_s
                    + float(outcome.execution_start_time_s)
                )
                completion_s = (
                    decision_time_s + float(outcome.completion_time_s)
                )
                blackouts.append((execution_start_s, completion_s))
                failures += int(not outcome.success)
                epoch_transition = _PendingTransition(
                    source_id=source_id,
                    target_id=target_id,
                    execution_start_s=execution_start_s,
                    completion_s=completion_s,
                    success=outcome.success,
                )

        record_service(
            decision_time_s,
            epoch_end_s,
            source_at_start,
            epoch_transition,
        )
        if epoch_transition is None:
            freeze_left = max(freeze_left - 1, 0)
        elif epoch_transition.completion_s <= epoch_end_s + 1e-9:
            apply_completion(epoch_transition)
            if not epoch_transition.success:
                freeze_left = max(freeze_left - 1, 0)
        else:
            pending_transition = epoch_transition

    if not rate_times:
        raise RuntimeError("Simulation produced no service samples")
    latency_trace = simulate_fifo_latency(
        config.traffic,
        duration_s=config.experiment.duration_s,
        rate_times_s=np.asarray(rate_times),
        rates_bps=np.asarray(link_service_rates),
        handover_blackouts=blackouts,
        seed=config.experiment.seed if seed is None else seed,
    )
    transmission_trace = simulate_fifo_latency(
        config.traffic,
        duration_s=config.experiment.duration_s,
        rate_times_s=np.asarray(rate_times),
        rates_bps=np.asarray(link_service_rates),
        handover_blackouts=None,
        seed=config.experiment.seed if seed is None else seed,
    )
    latency = latency_summary(latency_trace)
    transmission = latency_summary(transmission_trace)
    hof = 100.0 * failures / max(handovers, 1)
    return SimulationMetrics(
        method=policy.name,
        mean_rate_mbps=float(
            np.average(raw_rates, weights=sample_durations) / 1e6
        ),
        effective_throughput_mbps=float(
            np.average(effective_rates, weights=sample_durations) / 1e6
        ),
        handovers=handovers,
        handover_failures=failures,
        hof_percent=hof,
        outage_percent=(
            100.0 * outage_duration_s / max(total_duration_s, 1e-12)
        ),
        cho_hit_percent=100.0
        * _overlap_seconds(
            0.0,
            float(config.experiment.duration_s),
            blackouts,
        )
        / config.experiment.duration_s,
        transmission_only_mean_ms=transmission["transmission_only_mean_ms"],
        p50_latency_ms=latency["p50_ms"],
        p95_latency_ms=latency["p95_ms"],
        p99_latency_ms=latency["p99_ms"],
        p99_9_latency_ms=latency["p99_9_ms"],
        exceed_100_percent=latency["exceed_100_percent"],
        ping_pong_count=ping_pong_count,
        ping_pong_percent=100.0 * ping_pong_count / max(handovers, 1),
        delivered_packets=int(latency["packets"]),
        dropped_packets=int(latency["dropped"]),
        mean_target_cost=(
            float(np.mean(selected_target_costs))
            if selected_target_costs
            else float("nan")
        ),
        mean_oracle_target_cost=(
            float(np.mean(paired_oracle_costs))
            if paired_oracle_costs
            else float("nan")
        ),
        target_cost_sum=(
            float(np.sum(selected_target_costs))
            if selected_target_costs
            else float("nan")
        ),
        oracle_target_cost_sum=(
            float(np.sum(paired_oracle_costs))
            if paired_oracle_costs
            else float("nan")
        ),
        paired_cost_windows=len(paired_oracle_costs),
    )
