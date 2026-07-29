"""Sequence dataset with causal inputs and realized training-only labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel import LinkBudget, MeasurementTracker
from .config import NovaNetConfig
from .ephemeris import build_ephemeris, orbit_balanced_records, read_tle
from .forecast import build_forecast_sequence
from .geometry import UETrajectory
from .handover import handover_failure_matrix


class StaleTLEError(ValueError):
    pass


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
    measurement_noise_std_db: float = 0.0
    staleness_steps: int = 0
    ue_speed_kmh: float = 0.0
    ue_altitude_m: float = 0.0
    ue_heading_deg: float | None = None
    allow_stale_tle: bool = False


class NovaNetSequenceDataset(Dataset):
    def __init__(self, samples: list[dict[str, np.ndarray | int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        result: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key == "current_idx":
                result[key] = torch.tensor(value, dtype=torch.long)
            elif key.endswith("_mask") or key == "valid_mask":
                result[key] = torch.as_tensor(value, dtype=torch.bool)
            elif key == "selection_target":
                result[key] = torch.as_tensor(value, dtype=torch.long)
            else:
                result[key] = torch.as_tensor(value, dtype=torch.float32)
        return result


def _make_one_sample(
    config: NovaNetConfig,
    ephemeris,
    trajectory: UETrajectory,
    decision_index: int,
    rng: np.random.Generator,
    measurement_noise_std_db: float,
    staleness_steps: int,
) -> dict[str, np.ndarray | int]:
    budget = LinkBudget(config.channel, seed=int(rng.integers(0, 2**31 - 1)))
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
    stale_age = staleness_steps * config.experiment.decision_interval_s
    for local, satellite_id in enumerate(preliminary.candidate_ids):
        if satellite_id < 0 or not preliminary.valid_mask[0, local]:
            continue
        realized = preliminary.deterministic_snr_db[0, local] + rng.normal(
            0.0,
            max(config.channel.shadowing_std_db, 0.1),
        )
        measured = realized + rng.normal(0.0, measurement_noise_std_db)
        tracker.update(int(satellite_id), float(measured), current_time - stale_age)

    sequence = build_forecast_sequence(
        config,
        ephemeris,
        trajectory,
        decision_index,
        tracker,
        incumbent_id=int(preliminary.candidate_ids[preliminary.current_idx]),
        link_budget=budget,
    )
    if np.any(~sequence.valid_mask.any(axis=1)):
        raise RuntimeError(
            "Training window contains a no-coverage horizon; there is no "
            "feasible handover label for that horizon"
        )
    horizon, candidates = sequence.valid_mask.shape
    realized_snr = (
        sequence.deterministic_snr_db
        + rng.normal(
            0.0,
            max(config.channel.shadowing_std_db, 0.1),
            size=(horizon, candidates),
        )
    ).astype(np.float32)
    realized_snr[~sequence.valid_mask] = -100.0

    hof_target = np.zeros((horizon, candidates, candidates), np.float32)
    hof_mask = np.zeros_like(hof_target, dtype=bool)
    decision_s = config.experiment.decision_interval_s
    for h in range(horizon):
        if h + 1 < horizon:
            slope = (realized_snr[h + 1] - realized_snr[h]) / decision_s
        elif h > 0:
            slope = (realized_snr[h] - realized_snr[h - 1]) / decision_s
        else:
            slope = np.zeros(candidates, dtype=float)
        labels, pair_mask = handover_failure_matrix(
            realized_snr[h],
            slope,
            config.handover,
            config.channel.outage_threshold_db,
        )
        valid_pair = (
            sequence.valid_mask[h, :, None]
            & sequence.valid_mask[h, None, :]
        )
        hof_target[h] = labels
        hof_mask[h] = pair_mask & valid_pair

    rate = (
        config.channel.implementation_efficiency
        * config.channel.bandwidth_hz
        * np.log2(1.0 + 10.0 ** (realized_snr / 10.0))
        / 1e6
    )
    oracle_score = rate + 0.05 * sequence.ttl_s
    oracle_score[~sequence.valid_mask] = -np.inf
    selection_target = np.argmax(oracle_score, axis=-1).astype(np.int64)
    return {
        "node_features": sequence.node_features,
        "spatial_adjacency": sequence.spatial_adjacency,
        "valid_mask": sequence.valid_mask,
        "current_idx": sequence.current_idx,
        "angular_speed_deg_s": sequence.angular_speed_deg_s,
        "snr_target_db": realized_snr,
        "snr_mask": sequence.valid_mask.copy(),
        "ttl_target_s": sequence.ttl_s,
        "ttl_mask": sequence.valid_mask.copy(),
        "hof_target": hof_target,
        "hof_mask": hof_mask,
        "selection_target": selection_target,
    }


def generate_sequence_samples(
    config: NovaNetConfig,
    options: GenerationOptions,
) -> list[dict[str, np.ndarray | int]]:
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
    rng = np.random.default_rng(config.experiment.seed)
    stride = int(
        round(
            config.experiment.decision_interval_s
            / config.experiment.geometry_subsample_s
        )
    )
    final_start = (
        ephemeris.num_steps
        - stride * config.planner.horizon_steps
        - 1
    )
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
