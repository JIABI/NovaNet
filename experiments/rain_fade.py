#!/usr/bin/env python3
"""Controlled excess-attenuation sweep used by the manuscript rain study."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.baselines import LearnedBaselinePolicy
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
    parser.add_argument("--attenuation-db", default="0,5,10")
    parser.add_argument(
        "--bandwidth-mhz",
        default="20,100",
        help="Comma-separated bandwidths. Reliability is computed at each "
        "requested bandwidth; use 20,100 for the manuscript rate panels.",
    )
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--output", default="results/rain/per_user.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run deterministic baselines without requiring a checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not args.baselines_only and not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    attenuation_values = [
        float(value) for value in args.attenuation_db.split(",")
    ]
    bandwidth_values_hz = [
        float(value) * 1e6 for value in args.bandwidth_mhz.split(",")
    ]
    if not attenuation_values:
        raise ValueError("At least one attenuation value is required")
    if not bandwidth_values_hz or any(value <= 0 for value in bandwidth_values_hz):
        raise ValueError("Bandwidths must be positive")
    configured = set(cfg.channel.bandwidth_options_hz)
    unsupported = [value for value in bandwidth_values_hz if value not in configured]
    if unsupported:
        raise ValueError(
            "Requested bandwidth is outside channel.bandwidth_options_hz: "
            f"{[value / 1e6 for value in unsupported]} MHz"
        )
    ephemeris = build_paper_ephemeris(
        cfg,
        allow_stale_tle=args.allow_stale_tle,
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
    for bandwidth_hz in bandwidth_values_hz:
        run_cfg = replace(
            cfg,
            channel=replace(cfg.channel, bandwidth_hz=bandwidth_hz),
        )
        policies = [
            MaxElevationPolicy(),
            MaxServeTimePolicy(),
            RateDwellPolicy(
                dwell_weight=0.5,
                switch_penalty=0.2,
                rate_reference_mbps=run_cfg.planner.rate_reference_mbps,
                ttl_reference_s=run_cfg.planner.ttl_reference_s,
                bandwidth_hz=run_cfg.channel.bandwidth_hz,
                implementation_efficiency=(
                    run_cfg.channel.implementation_efficiency
                ),
            ),
        ]
        if not args.baselines_only:
            # The manuscript evaluates one nominally trained checkpoint at
            # both bandwidths. The architecture and preprocessing are
            # unchanged, so this intentional test-time config change is
            # recorded in every output row rather than hidden.
            for kind, path in (
                ("gnn_only", args.gnn_checkpoint),
                ("dqn_gnn", args.dqn_checkpoint),
            ):
                if path:
                    policies.append(
                        LearnedBaselinePolicy(
                            cfg,
                            path,
                            expected_kind=kind,
                        )
                    )
            policies.append(
                NovaNetPolicy(
                    run_cfg,
                    checkpoint,
                    allowed_config_overrides=(
                        ("channel.bandwidth_hz",)
                        if bandwidth_hz != cfg.channel.bandwidth_hz
                        else ()
                    ),
                    require_paper_eligible=not args.allow_stale_tle,
                )
            )
        for attenuation in attenuation_values:
            for user, (latitude, longitude) in enumerate(locations):
                scenario = Scenario(
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    rain_rate_mm_h=run_cfg.channel.rain_rate_mm_h,
                    rain_attenuation_db=attenuation,
                )
                for policy in policies:
                    metrics = simulate_single_user(
                        run_cfg,
                        ephemeris,
                        policy,
                        scenario,
                        seed=evaluation_episode_seed(evaluation_seed, user),
                    )
                    row = metrics_row(metrics, user)
                    row.update(
                        {
                            "bandwidth_mhz": bandwidth_hz / 1e6,
                            "rain_attenuation_db": attenuation,
                            "checkpoint_training_bandwidth_mhz": (
                                cfg.channel.bandwidth_hz / 1e6
                            ),
                            "condition": (
                                f"{metrics.method}|{bandwidth_hz / 1e6:g}MHz|"
                                f"{attenuation:g}dB"
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
        runner="rain_fade",
        checkpoint=None if args.baselines_only else checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_tle,
        details={
            "evaluation_seed": evaluation_seed,
            "bandwidth_hz": bandwidth_values_hz,
            "attenuation_db": attenuation_values,
            "nominal_checkpoint_reused_without_retraining": True,
            "learned_baseline_checkpoints": {
                kind: {
                    "name": Path(path).name,
                    "sha256": artifact_sha256(path),
                }
                for kind, path in (
                    ("gnn_only", args.gnn_checkpoint),
                    ("dqn_gnn", args.dqn_checkpoint),
                )
                if path and not args.baselines_only
            },
        },
    )
    print(f"wrote {args.output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
