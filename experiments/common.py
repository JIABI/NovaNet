from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from novanet.config import NovaNetConfig
from novanet.dataset import validate_tle_epoch
from novanet.ephemeris import build_ephemeris


def resolve_evaluation_seed(
    config: NovaNetConfig,
    requested_seed: int | None,
) -> int:
    """Return the manuscript's evaluation base seed.

    Training and evaluation may share a reported base seed while remaining
    disjoint: the helpers below derive a dedicated evaluation RNG domain.
    """

    seed = (
        config.experiment.seed
        if requested_seed is None
        else int(requested_seed)
    )
    if seed < 0:
        raise ValueError("evaluation seed must be nonnegative")
    return seed


def evaluation_rng(base_seed: int) -> np.random.Generator:
    """Return the domain-separated held-out layout generator."""

    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), 0x4556414C, 0])
    )


def evaluation_episode_seed(base_seed: int, index: int) -> int:
    """Derive a stable per-episode channel/traffic seed in the EVAL domain."""

    if index < 0:
        raise ValueError("episode index must be nonnegative")
    return int(
        np.random.SeedSequence(
            [int(base_seed), 0x4556414C, 1, int(index)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def build_paper_ephemeris(
    config: NovaNetConfig,
    *,
    allow_stale_tle: bool = False,
):
    if not allow_stale_tle:
        validate_tle_epoch(config)
    padding_s = (
        config.planner.horizon_steps * config.experiment.decision_interval_s
        + 900
    )
    return build_ephemeris(
        tle_path=config.resolve_tle_path(),
        start_utc=config.start_utc,
        duration_s=config.experiment.duration_s + padding_s,
        step_s=config.experiment.geometry_subsample_s,
        limit_satellites=config.experiment.num_satellites,
        selection=config.experiment.tle_selection,
    )


def write_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_protocol(
    path: str | Path,
    config: NovaNetConfig,
    *,
    runner: str,
    checkpoint: str | Path | None = None,
    evaluation_seed: int | None = None,
    diagnostic: bool = False,
    details: dict | None = None,
) -> Path:
    """Write a common provenance envelope beside generated experiment rows."""

    provenance = validate_tle_epoch(config, maximum_age_days=float("inf"))
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    actual_evaluation_seed = (
        config.experiment.seed
        if evaluation_seed is None
        else int(evaluation_seed)
    )
    if actual_evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative")
    payload = {
        "runner": runner,
        "schema_version": config.schema_version,
        "config_fingerprint": config.fingerprint,
        "evaluation_base_seed": actual_evaluation_seed,
        "evaluation_rng_domain": "EVAL-v1",
        "diagnostic": bool(diagnostic),
        "tle_provenance": provenance,
        "checkpoint": (
            {
                "name": checkpoint_path.name,
                "sha256": artifact_sha256(checkpoint_path),
            }
            if checkpoint_path is not None and checkpoint_path.is_file()
            else None
        ),
        "details": details or {},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def aggregate_rows(rows: list[dict], group_key: str = "method") -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    output: list[dict] = []
    for group, members in grouped.items():
        summary = {group_key: group, "replicates": len(members)}
        for key in members[0]:
            if key in {group_key, "user", "seed", "layout_seed"}:
                continue
            values = [member[key] for member in members]
            if all(isinstance(value, (int, float, np.number)) for value in values):
                numeric = np.asarray(values, dtype=float)
                finite = np.isfinite(numeric)
                if not finite.any() and key in {
                    "mean_target_cost",
                    "mean_oracle_target_cost",
                    "target_cost_sum",
                    "oracle_target_cost_sum",
                }:
                    # Optional metrics such as oracle cost are represented by
                    # NaN when a runner deliberately does not request them.
                    # Do not emit meaningless ``*_mean=nan`` columns.
                    continue
                if not finite.all():
                    raise ValueError(
                        f"Metric {key!r} mixes finite and non-finite values "
                        f"inside group {group!r}"
                    )
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(
                    np.std(values, ddof=1 if len(values) > 1 else 0)
                )
        if all(
            key in member
            for member in members
            for key in ("handovers", "handover_failures")
        ):
            attempts = sum(float(member["handovers"]) for member in members)
            failures = sum(
                float(member["handover_failures"]) for member in members
            )
            summary["hof_percent_pooled"] = (
                100.0 * failures / max(attempts, 1.0)
            )
        if all(
            key in member
            for member in members
            for key in ("handovers", "ping_pong_count")
        ):
            attempts = sum(float(member["handovers"]) for member in members)
            ping_pongs = sum(
                float(member["ping_pong_count"]) for member in members
            )
            summary["ping_pong_percent_pooled"] = (
                100.0 * ping_pongs / max(attempts, 1.0)
            )
        output.append(summary)
    return output


def add_paired_oracle_gap(summary_rows: list[dict]) -> None:
    """Add Eq. (59) from paired realized windows to aggregate rows in place."""

    for row in summary_rows:
        if {
            "target_cost_sum_mean",
            "oracle_target_cost_sum_mean",
            "paired_cost_windows_mean",
        }.issubset(row):
            # aggregate_rows averages per-episode sums and counts. Their
            # ratios are therefore identical to pooling every paired
            # UE--decision window, even when episodes contribute unequal
            # numbers of feasible windows.
            windows = float(row["paired_cost_windows_mean"])
            if not np.isfinite(windows) or windows <= 0.0:
                raise ValueError("Target-cost gap requires paired windows")
            selected_sum = float(row["target_cost_sum_mean"])
            oracle_sum = float(row["oracle_target_cost_sum_mean"])
            selected = selected_sum / windows
            oracle = oracle_sum / windows
            row["mean_target_cost_pooled"] = selected
            row["mean_oracle_target_cost_pooled"] = oracle
        else:
            # Compatibility for pre-schema-5 summary artifacts.
            selected = float(row["mean_target_cost_mean"])
            oracle = float(row["mean_oracle_target_cost_mean"])
        if not (np.isfinite(selected) and np.isfinite(oracle)):
            raise ValueError("Target-cost gap requires finite paired costs")
        row["oracle_gap_percent"] = (
            100.0 * (selected - oracle) / max(abs(oracle), 1e-12)
        )


def metrics_row(metrics, user: int) -> dict:
    row = asdict(metrics)
    row["user"] = user
    return row
