#!/usr/bin/env python3
"""Evaluate NovaNet with virtual/executed freeze windows W=0,1,2,3."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
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


def parse_windows(value: str) -> list[int]:
    windows = [int(item) for item in value.split(",") if item.strip()]
    if not windows or any(window < 0 for window in windows):
        raise ValueError("Freeze windows must be nonnegative integers")
    return windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--freeze-steps", default="0,1,2,3")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output", default="results/freeze/per_user.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    ephemeris = build_paper_ephemeris(
        base_cfg, allow_stale_tle=args.allow_stale_tle
    )
    evaluation_seed = resolve_evaluation_seed(
        base_cfg, args.evaluation_seed
    )
    rng = evaluation_rng(evaluation_seed)
    scenarios = [
        Scenario(
            latitude_deg=float(
                rng.uniform(*base_cfg.experiment.ue_latitude_deg)
            ),
            longitude_deg=float(
                rng.uniform(*base_cfg.experiment.ue_longitude_deg)
            ),
            altitude_m=base_cfg.experiment.ue_altitude_m,
        )
        for _ in range(args.users)
    ]

    rows: list[dict] = []
    for freeze_steps in parse_windows(args.freeze_steps):
        cfg = replace(
            base_cfg,
            handover=replace(base_cfg.handover, freeze_steps=freeze_steps),
        )
        # The checkpoint is held fixed. Only the explicit planner/execution
        # freeze constraint changes at evaluation time.
        policy = NovaNetPolicy(
            cfg,
            checkpoint,
            allowed_config_overrides=("handover.freeze_steps",),
            require_paper_eligible=not args.allow_stale_tle,
        )
        policy.name = f"NovaNet-W{freeze_steps}"
        for user, scenario in enumerate(scenarios):
            metrics = simulate_single_user(
                cfg,
                ephemeris,
                policy,
                scenario,
                seed=evaluation_episode_seed(evaluation_seed, user),
            )
            if not hasattr(metrics, "ping_pong_percent"):
                raise RuntimeError(
                    "The simulator must expose event-derived "
                    "ping_pong_percent before freeze sensitivity can be run"
                )
            row = metrics_row(metrics, user)
            row.update(
                {
                    "freeze_steps": freeze_steps,
                    "freeze_duration_s": (
                        freeze_steps
                        * cfg.experiment.decision_interval_s
                    ),
                    "condition": f"W={freeze_steps}",
                }
            )
            rows.append(row)

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    metadata_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        base_cfg,
        runner="freeze_sensitivity",
        checkpoint=checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_tle,
        details={
            "checkpoint_held_fixed": True,
            "changed_parameter": "handover.freeze_steps",
            "freeze_steps": parse_windows(args.freeze_steps),
            "decision_interval_s": base_cfg.experiment.decision_interval_s,
            "normalized_reward_equivalent_score": (
                "not emitted: exact reproduction requires the original "
                "validation-objective artifact and its normalization"
            ),
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
