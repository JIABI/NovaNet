#!/usr/bin/env python3
"""Aerial-vehicle experiment with explicit altitude, heading, and tracking."""

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

from experiments.common import build_paper_ephemeris, metrics_row, write_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--speed-kmh", type=float, default=300.0)
    parser.add_argument("--altitude-m", type=float, default=1000.0)
    parser.add_argument("--headings-deg", default="0,90,180,270")
    parser.add_argument("--users-per-heading", type=int, default=15)
    parser.add_argument("--output", default="results/aerial/doppler.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    policies = [
        MaxElevationPolicy(),
        RateDwellPolicy(),
        NovaNetPolicy(cfg, args.checkpoint),
    ]
    headings = [float(value) for value in args.headings_deg.split(",")]
    rng = np.random.default_rng(cfg.experiment.seed)
    locations = [
        (
            float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
        )
        for _ in range(args.users_per_heading)
    ]
    rows: list[dict] = []
    for heading in headings:
        for user, (latitude, longitude) in enumerate(locations):
            scenario = Scenario(
                latitude_deg=latitude,
                longitude_deg=longitude,
                altitude_m=args.altitude_m,
                speed_kmh=args.speed_kmh,
                heading_deg=heading,
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
                row.update(
                    {
                        "altitude_m": args.altitude_m,
                        "speed_kmh": args.speed_kmh,
                        "heading_deg": heading,
                        "carrier_hz": cfg.channel.carrier_hz,
                        "tracking_efficiency": (
                            cfg.channel.doppler_tracking_efficiency
                        ),
                        "doppler_error_std_hz": (
                            cfg.channel.doppler_estimation_std_hz
                        ),
                        "coherent_integration_s": (
                            cfg.channel.coherent_integration_s
                        ),
                    }
                )
                rows.append(row)
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
