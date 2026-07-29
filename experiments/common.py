from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from novanet.config import NovaNetConfig
from novanet.dataset import validate_tle_epoch
from novanet.ephemeris import build_ephemeris


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


def aggregate_rows(rows: list[dict], group_key: str = "method") -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    output: list[dict] = []
    for group, members in grouped.items():
        summary = {group_key: group, "replicates": len(members)}
        for key in members[0]:
            if key in {group_key, "user"}:
                continue
            values = [member[key] for member in members]
            if all(isinstance(value, (int, float, np.number)) for value in values):
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=0))
        output.append(summary)
    return output


def metrics_row(metrics, user: int) -> dict:
    row = asdict(metrics)
    row["user"] = user
    return row
