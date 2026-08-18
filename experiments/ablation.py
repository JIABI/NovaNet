#!/usr/bin/env python3
"""Run the manuscript ablations through explicit model switches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.model import VALID_ABLATIONS
from novanet.policies import NovaNetPolicy
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import (
    add_paired_oracle_gap,
    aggregate_rows,
    build_paper_ephemeris,
    evaluation_episode_seed,
    evaluation_rng,
    metrics_row,
    resolve_evaluation_seed,
    write_protocol,
    write_rows,
)


DEFAULT_VARIANTS = (
    "Full",
    "OrbitPrior",
    "DynAdj",
    "Temporal",
    "Planner",
    "UncLCB",
    "TransTTL",
    "TransHOF",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output", default="results/ablation/per_user.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint}; ablations require the trained paper checkpoint"
        )
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = set(variants) - (set(VALID_ABLATIONS) | {"Full"})
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    ephemeris = build_paper_ephemeris(
        cfg,
        allow_stale_tle=args.allow_stale_tle,
    )
    evaluation_seed = resolve_evaluation_seed(cfg, args.evaluation_seed)
    rng = evaluation_rng(evaluation_seed)
    scenarios = [
        Scenario(
            latitude_deg=float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            longitude_deg=float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
            altitude_m=cfg.experiment.ue_altitude_m,
            heading_deg=float(rng.uniform(0.0, 360.0)),
        )
        for _ in range(args.users)
    ]

    rows: list[dict] = []
    for variant in variants:
        ablations = () if variant == "Full" else (variant,)
        policy = NovaNetPolicy(
            cfg,
            checkpoint,
            ablations=ablations,
            require_paper_eligible=not args.allow_stale_tle,
        )
        policy.name = "Full" if variant == "Full" else f"--{variant}"
        for user, scenario in enumerate(scenarios):
            metrics = simulate_single_user(
                cfg,
                ephemeris,
                policy,
                scenario,
                seed=evaluation_episode_seed(evaluation_seed, user),
                compute_oracle_cost=True,
            )
            row = metrics_row(metrics, user)
            row["ablation"] = variant
            row["evaluation_seed"] = evaluation_seed
            row["intervention"] = "inference_time_component_bypass"
            rows.append(row)
            print(
                f"variant={variant:12s} user={user:03d} "
                f"rate={metrics.mean_rate_mbps:.2f} "
                f"HO={metrics.handovers} HOF={metrics.hof_percent:.2f}%",
                flush=True,
            )

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    summary = aggregate_rows(rows, group_key="ablation")
    add_paired_oracle_gap(summary)
    write_rows(summary_path, summary)
    protocol_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        cfg,
        runner="ablation",
        checkpoint=checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_tle,
        details={
            "intervention": "inference_time_component_bypass",
            "retrained_per_variant": False,
            "training_seed": cfg.experiment.seed,
            "oracle_reference": (
                "paired non-causal same-energy replay for every selected "
                "decision window"
            ),
            "orbit_prior_scope": (
                "zeros the five geometry/dwell encoder channels; "
                "candidate construction, geometric adjacency, direct TTL, "
                "and nominal physical SINR remain available"
            ),
            "variants": variants,
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
