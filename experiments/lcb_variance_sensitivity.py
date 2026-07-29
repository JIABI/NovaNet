#!/usr/bin/env python3
"""Sensitivity to estimation variance at fixed, validation-selected kappa."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.policies import NovaNetPolicy
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import build_paper_ephemeris, metrics_row, write_rows


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--measurement-std-db", default="0,1,2,3,4")
    parser.add_argument("--staleness-steps", default="0,1,2")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--output", default="results/lcb/variance_sensitivity.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    lcb = NovaNetPolicy(cfg, args.checkpoint)
    no_lcb = NovaNetPolicy(cfg, args.checkpoint)
    no_lcb.name = "NovaNet-no-LCB"
    no_lcb.model.energy.lcb_kappa = 0.0

    rng = np.random.default_rng(cfg.experiment.seed)
    locations = [
        (
            float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
        )
        for _ in range(args.users)
    ]
    rows: list[dict] = []
    for sigma in parse_floats(args.measurement_std_db):
        for stale in [int(value) for value in args.staleness_steps.split(",")]:
            for user, (latitude, longitude) in enumerate(locations):
                scenario = Scenario(
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    measurement_noise_std_db=sigma,
                    staleness_steps=stale,
                )
                for policy in (no_lcb, lcb):
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
                            "measurement_std_db": sigma,
                            "staleness_steps": stale,
                            "fixed_kappa": policy.model.energy.lcb_kappa,
                        }
                    )
                    rows.append(row)
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
