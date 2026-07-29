"""Single-UE link/CHO/packet simulation used by all experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import LinkBudget, MeasurementTracker
from .config import NovaNetConfig
from .ephemeris import Ephemeris
from .forecast import build_forecast_sequence
from .geometry import UETrajectory, geometry_state
from .handover import evaluate_cho_attempt
from .latency import latency_summary, simulate_fifo_latency
from .policies import Policy


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
    p99_9_latency_ms: float
    exceed_100_percent: float
    dropped_packets: int


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
) -> float:
    base_time_s = ephemeris.time_s(ephemeris_index)
    ue_position, ue_velocity = trajectory.state_at(base_time_s + offset_s)
    sat_position = (
        ephemeris.position_m[ephemeris_index, satellite_id]
        + ephemeris.velocity_m_s[ephemeris_index, satellite_id] * offset_s
    )
    sat_velocity = ephemeris.velocity_m_s[ephemeris_index, satellite_id]
    state = geometry_state(
        ue_position, ue_velocity, sat_position, sat_velocity
    )
    return (
        budget.evaluate(
            state,
            stochastic=False,
            rain_rate_mm_h=rain_rate_mm_h,
            additional_loss_db=rain_attenuation_db,
        ).snr_db
        - blockage_loss_db
    )


def simulate_single_user(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    policy: Policy,
    scenario: Scenario,
    *,
    seed: int | None = None,
) -> SimulationMetrics:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    rng = np.random.default_rng(
        config.experiment.seed if seed is None else seed
    )
    budget = LinkBudget(
        config.channel,
        seed=config.experiment.seed if seed is None else seed,
    )
    tracker = MeasurementTracker(config.channel)
    trajectory = UETrajectory(
        latitude_deg=scenario.latitude_deg,
        longitude_deg=scenario.longitude_deg,
        altitude_m=scenario.altitude_m,
        speed_m_s=scenario.speed_kmh / 3.6,
        heading_deg=scenario.heading_deg,
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
    effective_rates: list[float] = []
    outage_samples = 0
    total_samples = 0

    for decision_index in decision_indices:
        try:
            preliminary = build_forecast_sequence(
                config,
                ephemeris,
                trajectory,
                decision_index,
                tracker,
                incumbent_id=incumbent_id,
                link_budget=budget,
            )
        except RuntimeError:
            # No current eligible satellite is a physical coverage outage,
            # not a missing observation. Preserve the epoch on the service
            # timeline with zero rate so queueing and outage metrics include it.
            substeps = int(
                config.experiment.decision_interval_s / ephemeris.step_s
            )
            for offset_index in range(substeps):
                sample_index = decision_index + offset_index
                if sample_index >= final_index:
                    break
                rate_times.append(ephemeris.time_s(sample_index))
                raw_rates.append(0.0)
                effective_rates.append(0.0)
                outage_samples += 1
                total_samples += 1
            continue
        if incumbent_id is None:
            incumbent_id = int(
                preliminary.candidate_ids[preliminary.current_idx]
            )
        decision_time_s = ephemeris.time_s(decision_index)

        blocked = rng.random(len(preliminary.candidate_ids)) < (
            scenario.blockage_probability
        )
        for local, satellite_id in enumerate(preliminary.candidate_ids):
            if satellite_id < 0 or not preliminary.valid_mask[0, local]:
                continue
            # Reports are produced by the impaired physical link. The planner's
            # deterministic future still uses the nominal link budget, and the
            # causal report residual carries rain/blockage information forward.
            snr = _link_snr_at_offset(
                config,
                budget,
                ephemeris,
                trajectory,
                decision_index,
                int(satellite_id),
                0.0,
                scenario.rain_rate_mm_h,
                scenario.rain_attenuation_db,
                scenario.blockage_loss_db if blocked[local] else 0.0,
            )
            measured = snr + rng.normal(
                0.0, scenario.measurement_noise_std_db
            )
            report_time = decision_time_s - (
                scenario.staleness_steps
                * config.experiment.decision_interval_s
            )
            tracker.update(int(satellite_id), float(measured), report_time)

        sequence = build_forecast_sequence(
            config,
            ephemeris,
            trajectory,
            decision_index,
            tracker,
            incumbent_id=incumbent_id,
            link_budget=budget,
        )
        candidate_local = policy.choose(sequence)
        target_id = int(sequence.candidate_ids[candidate_local])
        switched_this_epoch = False

        if freeze_left > 0:
            freeze_left -= 1
        elif target_id >= 0 and target_id != incumbent_id:
            source_id = int(incumbent_id)
            source_local = np.where(sequence.candidate_ids == source_id)[0]
            source_block = (
                scenario.blockage_loss_db
                if len(source_local) and blocked[int(source_local[0])]
                else 0.0
            )
            target_block = (
                scenario.blockage_loss_db
                if blocked[candidate_local]
                else 0.0
            )
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
                source_block,
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
                target_block,
            )
            outcome = evaluate_cho_attempt(
                source,
                target,
                config.handover,
                config.channel.outage_threshold_db,
            )
            if outcome.attempted:
                handovers += 1
                blackout_start = decision_time_s + config.handover.ttt_s
                blackouts.append(
                    (
                        blackout_start,
                        blackout_start + config.handover.execution_s,
                    )
                )
                switched_this_epoch = True
                if outcome.success:
                    incumbent_id = target_id
                    freeze_left = config.handover.freeze_steps
                else:
                    failures += 1

        if incumbent_id is None:
            continue
        substeps = int(
            config.experiment.decision_interval_s / ephemeris.step_s
        )
        for offset_index in range(substeps):
            sample_index = decision_index + offset_index
            if sample_index >= final_index:
                break
            sample_time = ephemeris.time_s(sample_index)
            ue_position, ue_velocity = trajectory.state_at(sample_time)
            sat_position = ephemeris.position_m[sample_index, incumbent_id]
            sat_velocity = ephemeris.velocity_m_s[sample_index, incumbent_id]
            if not (
                np.all(np.isfinite(sat_position))
                and np.all(np.isfinite(sat_velocity))
            ):
                rate, snr = 0.0, -100.0
            else:
                state = geometry_state(
                    ue_position,
                    ue_velocity,
                    sat_position,
                    sat_velocity,
                )
                link = budget.evaluate(
                    state,
                    stochastic=True,
                    rain_rate_mm_h=scenario.rain_rate_mm_h,
                    additional_loss_db=scenario.rain_attenuation_db,
                )
                snr = link.snr_db
                rate = link.rate_bps
            rate_times.append(sample_time)
            raw_rates.append(rate)
            in_blackout = any(start <= sample_time < end for start, end in blackouts)
            effective_rates.append(0.0 if in_blackout else rate)
            outage_samples += int(snr < config.channel.outage_threshold_db)
            total_samples += 1

    if not rate_times:
        raise RuntimeError("Simulation produced no service samples")
    latency_trace = simulate_fifo_latency(
        config.traffic,
        duration_s=config.experiment.duration_s,
        rate_times_s=np.asarray(rate_times),
        rates_bps=np.asarray(raw_rates),
        handover_blackouts=blackouts,
        seed=config.experiment.seed if seed is None else seed,
    )
    transmission_trace = simulate_fifo_latency(
        config.traffic,
        duration_s=config.experiment.duration_s,
        rate_times_s=np.asarray(rate_times),
        rates_bps=np.asarray(raw_rates),
        handover_blackouts=None,
        seed=config.experiment.seed if seed is None else seed,
    )
    latency = latency_summary(latency_trace)
    transmission = latency_summary(transmission_trace)
    hof = 100.0 * failures / max(handovers, 1)
    return SimulationMetrics(
        method=policy.name,
        mean_rate_mbps=float(np.mean(raw_rates) / 1e6),
        effective_throughput_mbps=float(np.mean(effective_rates) / 1e6),
        handovers=handovers,
        handover_failures=failures,
        hof_percent=hof,
        outage_percent=100.0 * outage_samples / max(total_samples, 1),
        cho_hit_percent=100.0
        * sum(end - start for start, end in blackouts)
        / config.experiment.duration_s,
        transmission_only_mean_ms=transmission["transmission_only_mean_ms"],
        p50_latency_ms=latency["p50_ms"],
        p95_latency_ms=latency["p95_ms"],
        p99_9_latency_ms=latency["p99_9_ms"],
        exceed_100_percent=latency["exceed_100_percent"],
        dropped_packets=int(latency["dropped"]),
    )
