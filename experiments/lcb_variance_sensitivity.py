#!/usr/bin/env python3
"""Report-error sensitivity at fixed, validation-selected kappa.

Two explicit protocols prevent the nominal report-corruption table from being
silently conflated with the reviewer-requested high-mobility joint sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.policies import NovaNetPolicy, RateDwellPolicy
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


def parse_floats(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed or not np.isfinite(np.asarray(parsed, dtype=float)).all():
        raise ValueError("Expected a nonempty list of finite numbers")
    return parsed


def parse_nonnegative_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError("Expected nonempty nonnegative integer steps")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--measurement-std-db", default="0,1,2,3,4")
    parser.add_argument("--staleness-steps", default="0,1,2")
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--protocol-mode",
        choices=("nominal-ground", "joint-high-mobility"),
        default="nominal-ground",
        help=(
            "nominal-ground reproduces the separate report-noise/staleness "
            "protocol; joint-high-mobility runs the requested 300-km/h "
            "variance x LCB on/off experiment."
        ),
    )
    parser.add_argument("--speed-kmh", type=float, default=None)
    parser.add_argument("--altitude-m", type=float, default=None)
    parser.add_argument("--headings-deg", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint}")
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    lcb = NovaNetPolicy(
        cfg,
        checkpoint,
        require_paper_eligible=not args.allow_stale_tle,
    )
    no_lcb = NovaNetPolicy(
        cfg,
        checkpoint,
        require_paper_eligible=not args.allow_stale_tle,
    )
    no_lcb.name = "NovaNet-no-LCB"
    no_lcb.model.energy.lcb_kappa = 0.0
    heuristic = RateDwellPolicy(
        dwell_weight=0.5,
        switch_penalty=0.2,
        rate_reference_mbps=cfg.planner.rate_reference_mbps,
        ttl_reference_s=cfg.planner.ttl_reference_s,
        bandwidth_hz=cfg.channel.bandwidth_hz,
        implementation_efficiency=cfg.channel.implementation_efficiency,
    )
    heuristic.name = "Rate-Dwell"

    evaluation_seed = resolve_evaluation_seed(cfg, args.evaluation_seed)
    rng = evaluation_rng(evaluation_seed)
    locations = [
        (
            float(rng.uniform(*cfg.experiment.ue_latitude_deg)),
            float(rng.uniform(*cfg.experiment.ue_longitude_deg)),
        )
        for _ in range(args.users)
    ]
    if args.protocol_mode == "nominal-ground":
        speed_kmh = 0.0 if args.speed_kmh is None else args.speed_kmh
        altitude_m = (
            cfg.experiment.ue_altitude_m
            if args.altitude_m is None
            else args.altitude_m
        )
        headings_text = "0" if args.headings_deg is None else args.headings_deg
        default_output = "results/lcb/nominal_report_sensitivity.csv"
    else:
        speed_kmh = 300.0 if args.speed_kmh is None else args.speed_kmh
        altitude_m = 1000.0 if args.altitude_m is None else args.altitude_m
        headings_text = (
            "0,90,180,270"
            if args.headings_deg is None
            else args.headings_deg
        )
        default_output = "results/lcb/joint_high_mobility_variance.csv"
    output = args.output or default_output

    rows: list[dict] = []
    headings = parse_floats(headings_text)
    if not headings:
        raise ValueError("At least one mobility heading is required")
    if speed_kmh < 0.0 or altitude_m < 0.0:
        raise ValueError("Mobility speed and altitude must be nonnegative")
    measurement_std = parse_floats(args.measurement_std_db)
    if any(value < 0.0 for value in measurement_std):
        raise ValueError("Measurement standard deviations must be nonnegative")
    staleness_steps = parse_nonnegative_ints(args.staleness_steps)
    for sigma in measurement_std:
        for stale in staleness_steps:
            for user, (latitude, longitude) in enumerate(locations):
                scenario = Scenario(
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                    altitude_m=altitude_m,
                    speed_kmh=speed_kmh,
                    heading_deg=headings[user % len(headings)],
                    measurement_noise_std_db=sigma,
                    staleness_steps=stale,
                )
                for policy in (heuristic, no_lcb, lcb):
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
                            "measurement_std_db": sigma,
                            "staleness_steps": stale,
                            "protocol_mode": args.protocol_mode,
                            "speed_kmh": speed_kmh,
                            "altitude_m": altitude_m,
                            "heading_deg": headings[user % len(headings)],
                            "fixed_kappa": (
                                policy.model.energy.lcb_kappa
                                if isinstance(policy, NovaNetPolicy)
                                else ""
                            ),
                            "condition": (
                                f"{metrics.method}|sigma={sigma:g}|"
                                f"stale={stale}|speed={speed_kmh:g}"
                            ),
                        }
                    )
                    rows.append(row)
    write_rows(output, rows)
    summary_path = Path(output).with_name(
        f"{Path(output).stem}_summary.csv"
    )
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    protocol_path = write_protocol(
        Path(output).with_name(f"{Path(output).stem}_protocol.json"),
        cfg,
        runner="lcb_variance_sensitivity",
        checkpoint=checkpoint,
        evaluation_seed=evaluation_seed,
        diagnostic=args.allow_stale_tle,
        details={
            "evaluation_seed": evaluation_seed,
            "protocol_mode": args.protocol_mode,
            "measurement_std_db": measurement_std,
            "staleness_steps": staleness_steps,
            "speed_kmh": speed_kmh,
            "altitude_m": altitude_m,
            "headings_deg": headings,
            "fixed_lcb_kappa": cfg.planner.lcb_kappa,
            "nominal_checkpoint_reused_without_retraining": True,
        },
    )
    print(f"wrote {output}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
