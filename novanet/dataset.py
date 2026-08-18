"""Sequence dataset with causal inputs and realized training-only labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import LinkBudget, MeasurementTracker, RealizedChannelTrace
from .config import NovaNetConfig
from .ephemeris import build_ephemeris, orbit_balanced_records, read_tle
from .forecast import build_forecast_sequence
from .geometry import UETrajectory, geometry_state
from .handover import CHOOutcome, counterfactual_hof_label, evaluate_cho_attempt


class StaleTLEError(ValueError):
    pass


def _hof_supervision(
    outcome: CHOOutcome,
    target_trace,
    config: NovaNetConfig,
) -> tuple[float, bool]:
    """Separate all-pair teacher labels from attempted-only HOF supervision."""

    target = float(
        not outcome.success
        if outcome.attempted
        else counterfactual_hof_label(
            target_trace,
            config.handover,
            config.channel.outage_threshold_db,
            event_step_s=config.handover.event_step_s,
        )
    )
    return target, bool(outcome.attempted)


def tle_epoch(line1: str) -> datetime:
    year_2 = int(line1[18:20])
    year = 1900 + year_2 if year_2 >= 57 else 2000 + year_2
    day = float(line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1.0)


def validate_tle_epoch(
    config: NovaNetConfig,
    maximum_age_days: float = 14.0,
) -> dict[str, float | str | int]:
    records = read_tle(config.resolve_tle_path())
    epochs = [
        tle_epoch(line1).timestamp()
        for _name, line1, _line2 in records
    ]
    selected = orbit_balanced_records(
        records,
        config.experiment.num_satellites,
    )
    selected_names = "\n".join(record[0] for record in selected)
    selected_shells = {
        (
            round(float(line2[8:16])),
            round(float(line2[52:63]), 1),
        )
        for _name, _line1, line2 in selected
    }
    median = datetime.fromtimestamp(float(np.median(epochs)), tz=timezone.utc)
    delta_days = abs((config.start_utc - median).total_seconds()) / 86400.0
    report = {
        "tle_path": str(config.resolve_tle_path()),
        "tle_sha256": hashlib.sha256(
            config.resolve_tle_path().read_bytes()
        ).hexdigest(),
        "tle_records": len(epochs),
        "tle_selection": config.experiment.tle_selection,
        "selected_satellites": len(selected),
        "selected_names_sha256": hashlib.sha256(
            selected_names.encode("utf-8")
        ).hexdigest(),
        "selected_shell_count": len(selected_shells),
        "simulation_start_utc": config.start_utc.isoformat(),
        "median_tle_epoch_utc": median.isoformat(),
        "absolute_age_days": delta_days,
    }
    if delta_days > maximum_age_days:
        raise StaleTLEError(
            "The bundled TLE snapshot does not match the paper start time: "
            f"start={report['simulation_start_utc']}, "
            f"median TLE epoch={report['median_tle_epoch_utc']} "
            f"({delta_days:.1f} days apart). Supply the exact historical "
            "TLE snapshot used by the experiment; bypassing this check cannot "
            "produce paper-reproducible orbital results."
        )
    return report


@dataclass(frozen=True)
class GenerationOptions:
    num_samples: int
    seed: int | None = None
    measurement_noise_std_db: float = 0.0
    staleness_steps: int = 0
    ue_speed_kmh: float = 0.0
    ue_altitude_m: float = 0.0
    ue_heading_deg: float | None = None
    initial_freeze_steps: int = 0
    allow_stale_tle: bool = False

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        values = (
            self.measurement_noise_std_db,
            self.ue_speed_kmh,
            self.ue_altitude_m,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Generation options must be finite")
        if self.measurement_noise_std_db < 0.0:
            raise ValueError("measurement_noise_std_db must be nonnegative")
        if self.staleness_steps < 0:
            raise ValueError("staleness_steps must be nonnegative")
        if self.ue_speed_kmh < 0.0 or self.ue_altitude_m < 0.0:
            raise ValueError("UE speed and altitude must be nonnegative")
        if self.ue_heading_deg is not None and not np.isfinite(
            float(self.ue_heading_deg)
        ):
            raise ValueError("ue_heading_deg must be finite")
        if self.initial_freeze_steps < 0:
            raise ValueError("initial_freeze_steps must be nonnegative")


class NovaNetSequenceDataset(Dataset):
    def __init__(self, samples: list[dict[str, np.ndarray | int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        result: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key in {"current_idx", "initial_freeze"}:
                result[key] = torch.tensor(value, dtype=torch.long)
            elif key.endswith("_mask") or key == "valid_mask":
                result[key] = torch.as_tensor(value, dtype=torch.bool)
            else:
                result[key] = torch.as_tensor(value, dtype=torch.float32)
        return result


def protocol_window_start_count(config: NovaNetConfig) -> int:
    """Number of geometry-grid starts whose complete labels fit in T_obs."""

    latest_start_s = (
        config.experiment.duration_s
        - config.planner.horizon_steps
        * config.experiment.decision_interval_s
        - config.handover.execution_s
    )
    count = int(
        np.floor(latest_start_s / config.experiment.geometry_subsample_s)
    ) + 1
    if count <= 0:
        raise ValueError(
            "duration_s is too short for one complete planning/label window"
        )
    return count


def _realized_sinr_at_offset(
    config: NovaNetConfig,
    ephemeris,
    trajectory: UETrajectory,
    channel_trace: RealizedChannelTrace,
    ephemeris_index: int,
    satellite_id: int,
    offset_s: float,
) -> float:
    time_s = ephemeris.time_s(ephemeris_index) + offset_s
    ue_position, ue_velocity = trajectory.state_at(time_s)
    sat_position, sat_velocity = ephemeris.state_at_time(
        satellite_id,
        time_s,
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
    return channel_trace.evaluate(
        state, satellite_id, time_s
    ).sinr_db


def _make_one_sample(
    config: NovaNetConfig,
    ephemeris,
    trajectory: UETrajectory,
    decision_index: int,
    rng: np.random.Generator,
    measurement_noise_std_db: float,
    staleness_steps: int,
    initial_freeze_steps: int,
) -> dict[str, np.ndarray | int]:
    budget = LinkBudget(config.channel, seed=int(rng.integers(0, 2**31 - 1)))
    channel_trace = RealizedChannelTrace(
        config.channel,
        seed=int(rng.integers(0, 2**31 - 1)),
        event_step_s=config.handover.event_step_s,
    )
    tracker = MeasurementTracker(config.channel)

    preliminary = build_forecast_sequence(
        config,
        ephemeris,
        trajectory,
        decision_index,
        tracker,
        link_budget=budget,
    )
    current_time = ephemeris.time_s(decision_index)
    stride = int(
        round(config.experiment.decision_interval_s / ephemeris.step_s)
    )
    stale_index = max(0, decision_index - staleness_steps * stride)
    report_time = ephemeris.time_s(stale_index)
    report_ue_position, report_ue_velocity = trajectory.state_at(report_time)
    for local, satellite_id in enumerate(preliminary.candidate_ids):
        if satellite_id < 0 or not preliminary.valid_mask[0, local]:
            continue
        sat_position = ephemeris.position_m[stale_index, satellite_id]
        sat_velocity = ephemeris.velocity_m_s[stale_index, satellite_id]
        if not (
            np.all(np.isfinite(sat_position))
            and np.all(np.isfinite(sat_velocity))
        ):
            continue
        report_geometry = geometry_state(
            report_ue_position, report_ue_velocity, sat_position, sat_velocity
        )
        if report_geometry.elevation_deg < config.experiment.minimum_elevation_deg:
            continue
        realized = channel_trace.evaluate(
            report_geometry,
            int(satellite_id),
            report_time,
        ).sinr_db
        measured = realized + channel_trace.measurement_noise_db(
            int(satellite_id), report_time, measurement_noise_std_db
        )
        tracker.update(int(satellite_id), float(measured), report_time)

    sequence = build_forecast_sequence(
        config,
        ephemeris,
        trajectory,
        decision_index,
        tracker,
        incumbent_id=int(preliminary.candidate_ids[preliminary.current_idx]),
        link_budget=budget,
        initial_freeze=initial_freeze_steps,
    )
    horizon, candidates = sequence.valid_mask.shape
    realized_snr = np.full((horizon, candidates), -100.0, np.float32)
    for h in range(horizon):
        event_index = decision_index + h * stride
        for local, satellite_id in enumerate(sequence.candidate_ids):
            if satellite_id < 0 or not sequence.valid_mask[h, local]:
                continue
            realized_snr[h, local] = _realized_sinr_at_offset(
                config,
                ephemeris,
                trajectory,
                channel_trace,
                event_index,
                int(satellite_id),
                0.0,
            )

    hof_target = np.zeros((horizon, candidates, candidates), np.float32)
    hof_mask = np.zeros_like(hof_target, dtype=bool)
    for h in range(horizon):
        event_index = decision_index + h * stride
        event_cache: dict[tuple[int, int], float] = {}

        def event_sinr(satellite_id: int, offset_s: float) -> float:
            event_bin = int(
                round(offset_s / config.handover.event_step_s)
            )
            key = (satellite_id, event_bin)
            if key not in event_cache:
                event_cache[key] = _realized_sinr_at_offset(
                    config,
                    ephemeris,
                    trajectory,
                    channel_trace,
                    event_index,
                    satellite_id,
                    event_bin * config.handover.event_step_s,
                )
            return event_cache[key]

        for source in range(candidates):
            source_id = int(sequence.candidate_ids[source])
            if source_id < 0:
                continue
            for target in range(candidates):
                target_id = int(sequence.candidate_ids[target])
                if (
                    source == target
                    or target_id < 0
                    or not sequence.valid_mask[h, target]
                ):
                    continue
                source_trace = (
                    lambda offset, sat=source_id: event_sinr(sat, offset)
                )
                target_trace = (
                    lambda offset, sat=target_id: event_sinr(sat, offset)
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
                # Eq. (51) trains the HOF head only from a valid attempted
                # execution.  The all-pair realized/counterfactual target is
                # still retained for the clairvoyant sequence teacher.
                (
                    hof_target[h, source, target],
                    hof_mask[h, source, target],
                ) = _hof_supervision(
                    outcome,
                    target_trace,
                    config,
                )

    # Eq. (50): the target is the natural-log multiplicative residual in
    # linear SINR. In dB this is ln(10)/10 times the realized-minus-nominal
    # difference. Invalid entries are masked from the NLL.
    residual_target = (
        np.log(10.0)
        / 10.0
        * (realized_snr - sequence.deterministic_snr_db)
    ).astype(np.float32)
    residual_target[~sequence.valid_mask] = 0.0
    return {
        "node_features": sequence.node_features,
        "spatial_adjacency": sequence.spatial_adjacency,
        "valid_mask": sequence.valid_mask,
        "current_idx": sequence.current_idx,
        "initial_freeze": sequence.initial_freeze,
        "ttl_s": sequence.ttl_s,
        "nominal_snr_db": sequence.deterministic_snr_db,
        "residual_target": residual_target,
        "residual_mask": sequence.valid_mask.copy(),
        "hof_target": hof_target,
        "hof_mask": hof_mask,
    }


def generate_sequence_samples(
    config: NovaNetConfig,
    options: GenerationOptions,
) -> list[dict[str, np.ndarray | int]]:
    if not 0 <= options.initial_freeze_steps <= config.handover.freeze_steps:
        raise ValueError(
            "initial_freeze_steps must be between zero and configured freeze_steps"
        )
    if not options.allow_stale_tle:
        validate_tle_epoch(config)
    padding_s = (
        config.planner.horizon_steps * config.experiment.decision_interval_s
        + 900
    )
    ephemeris = build_ephemeris(
        config.resolve_tle_path(),
        config.start_utc,
        duration_s=config.experiment.duration_s + padding_s,
        step_s=config.experiment.geometry_subsample_s,
        limit_satellites=config.experiment.num_satellites,
        selection=config.experiment.tle_selection,
    )
    rng = np.random.default_rng(
        config.experiment.seed if options.seed is None else options.seed
    )
    stride = int(
        round(
            config.experiment.decision_interval_s
            / config.experiment.geometry_subsample_s
        )
    )
    # Training labels may monitor the configured target throughout the last
    # horizon epoch and then replay its execution window.  Select window
    # starts from the documented observation interval only; the padded
    # ephemeris exists for interpolation safety and is not additional data.
    final_start = protocol_window_start_count(config)
    samples: list[dict[str, np.ndarray | int]] = []
    attempts = 0
    max_attempts = max(100, options.num_samples * 30)
    while len(samples) < options.num_samples and attempts < max_attempts:
        attempts += 1
        latitude = rng.uniform(*config.experiment.ue_latitude_deg)
        longitude = rng.uniform(*config.experiment.ue_longitude_deg)
        heading = (
            rng.uniform(0.0, 360.0)
            if options.ue_heading_deg is None
            else options.ue_heading_deg
        )
        trajectory = UETrajectory(
            latitude_deg=float(latitude),
            longitude_deg=float(longitude),
            altitude_m=options.ue_altitude_m,
            speed_m_s=options.ue_speed_kmh / 3.6,
            heading_deg=float(heading),
        )
        decision_index = int(rng.integers(0, final_start))
        try:
            sample = _make_one_sample(
                config,
                ephemeris,
                trajectory,
                decision_index,
                rng,
                options.measurement_noise_std_db,
                options.staleness_steps,
                options.initial_freeze_steps,
            )
        except (RuntimeError, ValueError):
            continue
        samples.append(sample)
    if len(samples) != options.num_samples:
        raise RuntimeError(
            f"Generated {len(samples)}/{options.num_samples} valid samples "
            f"after {attempts} attempts"
        )
    return samples
