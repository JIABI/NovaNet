#!/usr/bin/env python3
"""Evaluate planning with aged TLE priors against fresh realized geometry.

The script deliberately requires real TLE files for every requested age.  It
does not shift timestamps, perturb positions, or synthesize orbital errors to
match the manuscript.  Candidate construction and TTL use ``planning_ephemeris``;
CHO execution and service metrics use the fresh ``truth_ephemeris``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.dataset import tle_epoch
from novanet.ephemeris import (
    build_ephemeris_from_records,
    orbit_balanced_records,
    read_tle,
)
from novanet.policies import NovaNetPolicy
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import (
    aggregate_rows,
    build_paper_ephemeris,
    evaluation_episode_seed,
    evaluation_rng,
    metrics_row,
    resolve_evaluation_seed,
    write_protocol,
    write_rows,
)


def parse_ages(value: str) -> list[int]:
    ages = [int(item) for item in value.split(",") if item.strip()]
    if not ages or any(age < 0 for age in ages):
        raise ValueError("Ages must be nonnegative integer hours")
    return ages


def parse_planning_tles(values: list[str] | None) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values or []:
        try:
            age_text, path_text = value.split("=", maxsplit=1)
            age = int(age_text)
        except ValueError as error:
            raise ValueError(
                "--planning-tle must use AGE_HOURS=/path/to/file.tle"
            ) from error
        path = Path(path_text).expanduser().resolve()
        if age in mapping:
            raise ValueError(f"Duplicate planning TLE for age {age} h")
        if not path.is_file():
            raise FileNotFoundError(path)
        mapping[age] = path
    return mapping


def catalog_id(record: tuple[str, str, str]) -> str:
    return record[1][2:7].strip()


def selected_truth_records(path: Path, satellites: int):
    return orbit_balanced_records(read_tle(path), satellites)


def matched_records(path: Path, truth_records):
    by_catalog = {catalog_id(record): record for record in read_tle(path)}
    catalog_ids = [catalog_id(record) for record in truth_records]
    missing = [identifier for identifier in catalog_ids if identifier not in by_catalog]
    if missing:
        raise ValueError(
            f"Planning TLE {path} is missing {len(missing)} truth-selected "
            f"NORAD IDs, including {missing[:5]}"
        )
    # Preserve the truth snapshot's canonical order and display names while
    # taking the orbital elements from the requested historical snapshot.
    return [
        (truth_record[0], by_catalog[identifier][1], by_catalog[identifier][2])
        for truth_record, identifier in zip(truth_records, catalog_ids)
    ]


def records_median_epoch(selected) -> datetime:
    epochs = [tle_epoch(line1).timestamp() for _, line1, _ in selected]
    return datetime.fromtimestamp(
        float(np.median(epochs)),
        tz=tle_epoch(selected[0][1]).tzinfo,
    )


def require_identical_zero_age_snapshot(
    planning_path: Path,
    truth_path: Path,
) -> None:
    """Require the 0 h planning prior to be the exact truth snapshot."""

    planning_hash = hashlib.sha256(planning_path.read_bytes()).hexdigest()
    truth_hash = hashlib.sha256(truth_path.read_bytes()).hexdigest()
    if planning_hash != truth_hash:
        raise ValueError(
            "The 0 h planning TLE must be byte-identical to the truth TLE; "
            "a merely epoch-near snapshot is not the fresh-prior condition"
        )


def assert_aligned(truth, planning, age_hours: int) -> None:
    if truth.names != planning.names:
        raise ValueError(
            f"{age_hours} h planning TLE selects a different satellite/order; "
            "use matched snapshots with identical object names"
        )
    if truth.position_m.shape != planning.position_m.shape:
        raise ValueError(f"{age_hours} h ephemeris shape differs from truth")
    if truth.start_utc != planning.start_utc or truth.step_s != planning.step_s:
        raise ValueError(f"{age_hours} h ephemeris time grid differs from truth")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--ages-hours", default="0,24,72")
    parser.add_argument(
        "--planning-tle",
        action="append",
        metavar="AGE_HOURS=PATH",
        help="Repeat once for every requested age, including age 0.",
    )
    parser.add_argument(
        "--age-tolerance-hours",
        type=float,
        default=6.0,
        help="Maximum mismatch between requested and median TLE-epoch age.",
    )
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output", default="results/ephemeris_aging/per_user.csv")
    parser.add_argument("--allow-stale-truth-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    requested_ages = parse_ages(args.ages_hours)
    planning_tles = parse_planning_tles(args.planning_tle)
    missing = set(requested_ages) - set(planning_tles)
    extra = set(planning_tles) - set(requested_ages)
    if missing or extra:
        raise ValueError(
            "Planning TLE mapping must exactly cover requested ages: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if args.age_tolerance_hours < 0.0:
        raise ValueError("--age-tolerance-hours must be nonnegative")

    truth = build_paper_ephemeris(
        cfg,
        allow_stale_tle=args.allow_stale_truth_tle,
    )
    truth_path = cfg.resolve_tle_path().resolve()
    truth_records = selected_truth_records(
        truth_path, cfg.experiment.num_satellites
    )
    truth_epoch = records_median_epoch(truth_records)
    planning_by_age = {}
    actual_age_by_requested = {}
    for age_hours in requested_ages:
        path = planning_tles[age_hours]
        if age_hours == 0:
            require_identical_zero_age_snapshot(path, truth_path)
        prior_records = matched_records(path, truth_records)
        prior_epoch = records_median_epoch(prior_records)
        actual_age = (truth_epoch - prior_epoch).total_seconds() / 3600.0
        if abs(actual_age - age_hours) > args.age_tolerance_hours:
            raise ValueError(
                f"Requested age {age_hours} h but selected median epochs differ "
                f"by {actual_age:.2f} h; provide the actual matched snapshot"
            )
        padding_s = (
            cfg.planner.horizon_steps
            * cfg.experiment.decision_interval_s
            + 900
        )
        planning = build_ephemeris_from_records(
            prior_records,
            start_utc=cfg.start_utc,
            duration_s=cfg.experiment.duration_s + padding_s,
            step_s=cfg.experiment.geometry_subsample_s,
        )
        assert_aligned(truth, planning, age_hours)
        planning_by_age[age_hours] = planning
        actual_age_by_requested[age_hours] = actual_age

    policy = NovaNetPolicy(
        cfg,
        checkpoint,
        require_paper_eligible=not args.allow_stale_truth_tle,
    )
    evaluation_seed = resolve_evaluation_seed(cfg, args.evaluation_seed)
    rng = evaluation_rng(evaluation_seed)
    scenarios = [
        Scenario(
            latitude_deg=float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            longitude_deg=float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
            altitude_m=cfg.experiment.ue_altitude_m,
        )
        for _ in range(args.users)
    ]
    rows: list[dict] = []
    for age_hours in requested_ages:
        path = planning_tles[age_hours]
        for user, scenario in enumerate(scenarios):
            metrics = simulate_single_user(
                cfg,
                truth,
                policy,
                scenario,
                seed=evaluation_episode_seed(evaluation_seed, user),
                planning_ephemeris=planning_by_age[age_hours],
            )
            row = metrics_row(metrics, user)
            row.update(
                {
                    "requested_age_hours": age_hours,
                    "actual_median_age_hours": actual_age_by_requested[age_hours],
                    "planning_tle_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    "truth_tle_sha256": hashlib.sha256(
                        truth_path.read_bytes()
                    ).hexdigest(),
                    "condition": f"{age_hours}h",
                }
            )
            rows.append(row)

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    protocol_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        cfg,
        runner="ephemeris_aging",
        checkpoint=checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_truth_tle,
        details={
            "evaluation_seed": evaluation_seed,
            "truth_tle_sha256": hashlib.sha256(
                truth_path.read_bytes()
            ).hexdigest(),
            "requested_ages_hours": requested_ages,
            "actual_median_ages_hours": actual_age_by_requested,
            "planning_tle_sha256": {
                str(age): hashlib.sha256(path.read_bytes()).hexdigest()
                for age, path in planning_tles.items()
            },
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
