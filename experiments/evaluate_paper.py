#!/usr/bin/env python3
"""Run the unified clear-sky evaluation from one canonical configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.baselines import LearnedBaselinePolicy
from novanet.config import load_config
from novanet.policies import (
    DwellAwarePolicy,
    MaxElevationPolicy,
    MaxServeTimePolicy,
    NovaNetPolicy,
    OfflineOraclePolicy,
    PeriodicHOPolicy,
    RateDwellPolicy,
    SkipKPolicy,
)
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import (
    add_paired_oracle_gap,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--gnn-checkpoint", default=None)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--dho-checkpoint", default=None)
    parser.add_argument(
        "--allow-surrogate-baselines",
        action="store_true",
        help=(
            "Diagnostic only: permit checkpoints explicitly marked ineligible "
            "for manuscript tables, such as the local DHO surrogate."
        ),
    )
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help=(
            "Evaluation base seed. Defaults to the configured paper seed; "
            "layout/channel streams are separated in the EVAL-v1 domain."
        ),
    )
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
        PeriodicHOPolicy(period_steps=16),
        SkipKPolicy(skip=1),
        SkipKPolicy(skip=2),
        DwellAwarePolicy(
            improvement_threshold=0.10,
            ttl_reference_s=cfg.planner.ttl_reference_s,
        ),
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
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing trained checkpoint {checkpoint}. The repository "
                "does not substitute a random model for the paper model."
            )
        policies.append(
            NovaNetPolicy(
                cfg,
                checkpoint,
                require_paper_eligible=not args.allow_stale_tle,
            )
        )
    learned_paths = {
        "gnn_only": args.gnn_checkpoint,
        "dqn_gnn": args.dqn_checkpoint,
        "dho": args.dho_checkpoint,
    }
    for kind, path in learned_paths.items():
        if path:
            policies.append(
                LearnedBaselinePolicy(
                    cfg,
                    path,
                    expected_kind=kind,
                    allow_unqualified=args.allow_surrogate_baselines,
                )
            )
    policies.append(OfflineOraclePolicy())

    evaluation_seed = resolve_evaluation_seed(cfg, args.evaluation_seed)
    rng = evaluation_rng(evaluation_seed)
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
        user_rows: list[dict] = []
        for policy in policies:
            metrics = simulate_single_user(
                cfg,
                ephemeris,
                policy,
                scenario,
                seed=evaluation_episode_seed(evaluation_seed, user),
                compute_oracle_cost=True,
            )
            row = metrics_row(metrics, user)
            user_rows.append(row)
            print(
                f"user={user:03d} method={metrics.method:16s} "
                f"rate={metrics.mean_rate_mbps:7.2f} "
                f"HO={metrics.handovers:3d} HOF={metrics.hof_percent:6.2f}%",
                flush=True,
            )
        rows.extend(user_rows)

    write_rows(args.output, rows)
    summary_path = Path(args.output).with_name("summary.csv")
    summary = aggregate_rows(rows)
    add_paired_oracle_gap(summary)
    write_rows(summary_path, summary)
    supplied = {
        "NovaNet": None if args.baselines_only else args.checkpoint,
        "GNN-only": args.gnn_checkpoint,
        "DQN+GNN": args.dqn_checkpoint,
        "DHO": args.dho_checkpoint,
    }
    checkpoint_artifacts = {
        method: (
            {
                "name": Path(path).name,
                "sha256": artifact_sha256(path),
            }
            if path and Path(path).is_file()
            else None
        )
        for method, path in supplied.items()
    }
    protocol_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        cfg,
        runner="evaluate_paper",
        checkpoint=(
            None if args.baselines_only else Path(args.checkpoint)
        ),
        evaluation_seed=evaluation_seed,
        diagnostic=(
            args.allow_stale_tle or args.allow_surrogate_baselines
        ),
        details={
            "executed_methods": [policy.name for policy in policies],
            "checkpoint_artifacts": checkpoint_artifacts,
            "users": int(args.users),
            "scenario_generation": (
                "EVAL-v1 domain-separated layouts and per-user channel/"
                "traffic streams"
            ),
            "offline_oracle": (
                "non-causal same-energy replay on each method's paired "
                "held-out decision windows"
            ),
            "diagnostic_stale_tle": bool(args.allow_stale_tle),
            "diagnostic_surrogate_baselines": bool(
                args.allow_surrogate_baselines
            ),
            "training_seed": cfg.experiment.seed,
            "held_out_from_training_rng": True,
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
