#!/usr/bin/env python3
"""Controlled excess-attenuation sweep used by the manuscript rain study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.policies import MaxElevationPolicy, NovaNetPolicy, RateDwellPolicy
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
    parser.add_argument("--attenuation-db", default="0,5,10")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--output", default="results/rain/per_user.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    ephemeris = build_paper_ephemeris(
        cfg,
        allow_stale_tle=args.allow_stale_tle,
    )
    policies = (
        MaxElevationPolicy(),
        RateDwellPolicy(),
        NovaNetPolicy(cfg, checkpoint),
    )
    attenuation_values = [
        float(value) for value in args.attenuation_db.split(",")
    ]
    rng = np.random.default_rng(cfg.experiment.seed)
    locations = [
        (
            float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
        )
        for _ in range(args.users)
    ]
    rows: list[dict] = []
    for attenuation in attenuation_values:
        for user, (latitude, longitude) in enumerate(locations):
            scenario = Scenario(
                latitude_deg=latitude,
                longitude_deg=longitude,
                rain_rate_mm_h=cfg.channel.rain_rate_mm_h,
                rain_attenuation_db=attenuation,
            )
            for policy in policies:
                metrics = simulate_single_user(
                    cfg,
                    ephemeris,
                    policy,
                    scenario,
                    seed=cfg.experiment.seed + user,
                )
                row = metrics_row(metrics, user)
                row["rain_attenuation_db"] = attenuation
                row["condition"] = f"{metrics.method}|{attenuation:g}dB"
                rows.append(row)
    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    print(f"wrote {args.output} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
