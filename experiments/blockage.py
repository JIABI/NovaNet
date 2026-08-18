#!/usr/bin/env python3
"""Zero-shot partial-outdoor-blockage evaluation.

Each ``LOSS_DB:OCCUPANCY`` condition uses the same UE layouts and random seeds
for every policy.  The simulator applies the sampled blockage state to the
measurement, CHO execution, and service-link paths; this script never replaces
missing observations with manuscript table values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.baselines import LearnedBaselinePolicy
from novanet.config import load_config
from novanet.policies import MaxElevationPolicy, NovaNetPolicy, RateDwellPolicy
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import (
    aggregate_rows,
    artifact_sha256,
    build_paper_ephemeris,
    evaluation_episode_seed,
    evaluation_rng,
    metrics_row,
    resolve_evaluation_seed,
    write_protocol,
    write_rows,
)


def parse_conditions(value: str) -> list[tuple[float, float]]:
    conditions: list[tuple[float, float]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            loss_text, occupancy_text = item.split(":", maxsplit=1)
            loss_db = float(loss_text)
            occupancy = float(occupancy_text)
        except ValueError as error:
            raise ValueError(
                "Blockage conditions must use LOSS_DB:OCCUPANCY, for "
                "example 8:0.10,12:0.20"
            ) from error
        if loss_db < 0.0 or not 0.0 <= occupancy <= 1.0:
            raise ValueError("Loss must be nonnegative and occupancy in [0, 1]")
        conditions.append((loss_db, occupancy))
    if not conditions:
        raise ValueError("At least one blockage condition is required")
    return conditions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument(
        "--conditions",
        default="8:0.10,12:0.20",
        help="Comma-separated LOSS_DB:OCCUPANCY pairs.",
    )
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output", default="results/blockage/per_user.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not args.baselines_only and not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    policies = [
        MaxElevationPolicy(),
        RateDwellPolicy(
            dwell_weight=0.5,
            switch_penalty=0.2,
            rate_reference_mbps=cfg.planner.rate_reference_mbps,
            ttl_reference_s=cfg.planner.ttl_reference_s,
            bandwidth_hz=cfg.channel.bandwidth_hz,
            implementation_efficiency=cfg.channel.implementation_efficiency,
        ),
    ]
    if not args.baselines_only:
        if args.dqn_checkpoint:
            policies.append(
                LearnedBaselinePolicy(
                    cfg, args.dqn_checkpoint, expected_kind="dqn_gnn"
                )
            )
        policies.append(
            NovaNetPolicy(
                cfg,
                checkpoint,
                require_paper_eligible=not args.allow_stale_tle,
            )
        )

    evaluation_seed = resolve_evaluation_seed(cfg, args.evaluation_seed)
    rng = evaluation_rng(evaluation_seed)
    locations = [
        (
            float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
        )
        for _ in range(args.users)
    ]
    rows: list[dict] = []
    for loss_db, occupancy in parse_conditions(args.conditions):
        for user, (latitude, longitude) in enumerate(locations):
            scenario = Scenario(
                latitude_deg=latitude,
                longitude_deg=longitude,
                altitude_m=cfg.experiment.ue_altitude_m,
                blockage_loss_db=loss_db,
                blockage_probability=occupancy,
            )
            for policy in policies:
                metrics = simulate_single_user(
                    cfg,
                    ephemeris,
                    policy,
                    scenario,
                    seed=evaluation_episode_seed(evaluation_seed, user),
                )
                row = metrics_row(metrics, user)
                row.update(
                    {
                        "blockage_loss_db": loss_db,
                        "blockage_occupancy": occupancy,
                        "condition": (
                            f"{metrics.method}|{loss_db:g}dB|"
                            f"{100.0 * occupancy:g}pct"
                        ),
                    }
                )
                rows.append(row)

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    protocol_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        cfg,
        runner="blockage",
        checkpoint=None if args.baselines_only else checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_tle,
        details={
            "evaluation_seed": evaluation_seed,
            "conditions": parse_conditions(args.conditions),
            "nominal_checkpoint_reused_without_retraining": True,
            "learned_baseline_checkpoints": (
                {
                    "dqn_gnn": {
                        "name": Path(args.dqn_checkpoint).name,
                        "sha256": artifact_sha256(args.dqn_checkpoint),
                    }
                }
                if args.dqn_checkpoint and not args.baselines_only
                else {}
            ),
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
