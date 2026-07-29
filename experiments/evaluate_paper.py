#!/usr/bin/env python3
"""Run the unified clear-sky evaluation from one canonical configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.policies import (
    MaxElevationPolicy,
    MaxServeTimePolicy,
    NovaNetPolicy,
    RateDwellPolicy,
)
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import (
    aggregate_rows,
    build_paper_ephemeris,
    metrics_row,
    write_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--output", default="results/evaluation/per_user.csv")
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; results cannot be called paper-reproducible.",
    )
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    policies = [
        MaxElevationPolicy(),
        MaxServeTimePolicy(),
        RateDwellPolicy(),
    ]
    if not args.baselines_only:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing trained checkpoint {checkpoint}. The repository "
                "does not substitute a random model for the paper model."
            )
        policies.append(NovaNetPolicy(cfg, checkpoint))

    rng = np.random.default_rng(cfg.experiment.seed)
    rows: list[dict] = []
    scenarios = [
        Scenario(
            latitude_deg=float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            longitude_deg=float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
            altitude_m=cfg.experiment.ue_altitude_m,
            heading_deg=float(rng.uniform(0.0, 360.0)),
        )
        for _ in range(args.users)
    ]
    for user, scenario in enumerate(scenarios):
        for policy in policies:
            metrics = simulate_single_user(
                cfg,
                ephemeris,
                policy,
                scenario,
                seed=cfg.experiment.seed + user,
            )
            row = metrics_row(metrics, user)
            rows.append(row)
            print(
                f"user={user:03d} method={metrics.method:16s} "
                f"rate={metrics.mean_rate_mbps:7.2f} "
                f"HO={metrics.handovers:3d} HOF={metrics.hof_percent:6.2f}%",
                flush=True,
            )

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    write_rows(summary_path, aggregate_rows(rows))
    print(f"wrote {args.output} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
