#!/usr/bin/env python3
"""Load-aware multi-UE experiment with fully specified network mechanics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.channel import LinkBudget, MeasurementTracker
from novanet.config import load_config
from novanet.forecast import build_forecast_sequence
from novanet.geometry import UETrajectory
from novanet.handover import handover_failure_matrix
from novanet.multi_ue import SynchronousAssociationScheduler, jain_fairness
from novanet.policies import NovaNetPolicy

from experiments.common import build_paper_ephemeris, write_rows


METHODS = (
    "Max-Elevation-LB",
    "Max-ServeTime-LB",
    "Skip-1-LB",
    "Rate-Dwell-LB",
    "NovaNet-LB",
)


def _users_in_disk(config, count: int, rng: np.random.Generator):
    center = UETrajectory(
        config.multi_ue.region_center_lat_deg,
        config.multi_ue.region_center_lon_deg,
    )
    users = []
    for _ in range(count):
        radius_m = (
            config.multi_ue.region_radius_km
            * 1000.0
            * np.sqrt(rng.random())
        )
        heading = rng.uniform(0.0, 360.0)
        mover = replace(
            center,
            speed_m_s=radius_m,
            heading_deg=float(heading),
        )
        latitude, longitude, altitude = mover.geodetic_at(1.0)
        users.append(UETrajectory(latitude, longitude, altitude))
    return users


def _base_score(
    config,
    method: str,
    sequence,
    novanet: NovaNetPolicy,
    user_context: int,
    candidate_load: np.ndarray,
) -> np.ndarray:
    valid = sequence.valid_mask[0]
    if method == "Max-Elevation-LB":
        score = sequence.node_features[0, :, 0] * 90.0
    elif method == "Max-ServeTime-LB":
        score = sequence.ttl_s[0]
    elif method == "Skip-1-LB":
        elevation = sequence.node_features[0, :, 0] * 90.0
        order = np.argsort(-elevation)
        score = np.full_like(elevation, -1e6)
        score[order[::2]] = elevation[order[::2]]
    elif method == "Rate-Dwell-LB":
        score = (
            sequence.deterministic_snr_db[0]
            + 0.05 * sequence.ttl_s[0]
        )
    elif method == "NovaNet-LB":
        load_forecast = np.broadcast_to(
            candidate_load[None, :],
            sequence.valid_mask.shape,
        ).copy()
        score = novanet.scores(
            sequence,
            context=user_context,
            load=load_forecast,
        )
    else:
        raise ValueError(method)
    score = np.asarray(score, dtype=float)
    score[~valid] = -np.inf
    if method != "NovaNet-LB":
        finite = score[valid]
        scale = max(float(finite.std()), 1e-6)
        score[valid] = (finite - float(finite.mean())) / scale
        score[valid] -= (
            config.planner.load_weight * candidate_load[valid]
        )
    return score


def run_method(
    config,
    ephemeris,
    method: str,
    user_count: int,
    novanet: NovaNetPolicy,
    max_epochs: int | None,
) -> dict:
    rng = np.random.default_rng(config.experiment.seed + user_count)
    trajectories = _users_in_disk(config, user_count, rng)
    trackers = [MeasurementTracker(config.channel) for _ in trajectories]
    budgets = [
        LinkBudget(config.channel, seed=config.experiment.seed + index)
        for index in range(user_count)
    ]
    scheduler = SynchronousAssociationScheduler(config.multi_ue)
    previous_load: dict[int, float] = {}
    incumbent = np.full(user_count, -1, dtype=int)
    average_rate = np.ones(user_count, dtype=float)
    rate_accumulator = np.zeros(user_count, dtype=float)
    blocking_events = 0
    handovers = 0
    failures = 0
    novanet.reset()
    stride = int(
        config.experiment.decision_interval_s / ephemeris.step_s
    )
    epoch_count = int(
        config.experiment.duration_s
        / config.experiment.decision_interval_s
    )
    if max_epochs is not None:
        epoch_count = min(epoch_count, max_epochs)

    for epoch in range(epoch_count):
        decision_index = epoch * stride
        candidate_ids = np.full(
            (user_count, config.experiment.candidate_cap), -1, dtype=int
        )
        scores = np.full_like(candidate_ids, -np.inf, dtype=float)
        rates = np.zeros_like(scores, dtype=float)
        sequences = []
        hof_events: list[tuple[np.ndarray, np.ndarray] | None] = []
        for user, (trajectory, tracker, budget) in enumerate(
            zip(trajectories, trackers, budgets)
        ):
            preliminary = build_forecast_sequence(
                config,
                ephemeris,
                trajectory,
                decision_index,
                tracker,
                incumbent_id=None if incumbent[user] < 0 else int(incumbent[user]),
                link_budget=budget,
            )
            report_time = ephemeris.time_s(decision_index)
            for local, satellite_id in enumerate(preliminary.candidate_ids):
                if satellite_id < 0 or not preliminary.valid_mask[0, local]:
                    continue
                tracker.update(
                    int(satellite_id),
                    float(preliminary.deterministic_snr_db[0, local]),
                    report_time,
                )
            sequence = build_forecast_sequence(
                config,
                ephemeris,
                trajectory,
                decision_index,
                tracker,
                incumbent_id=None if incumbent[user] < 0 else int(incumbent[user]),
                link_budget=budget,
            )
            sequences.append(sequence)
            candidate_ids[user] = sequence.candidate_ids
            load_penalty = np.asarray(
                [
                    previous_load.get(int(satellite), 0.0)
                    if satellite >= 0
                    else 1.0
                    for satellite in sequence.candidate_ids
                ]
            )
            scores[user] = _base_score(
                config,
                method,
                sequence,
                novanet,
                user,
                load_penalty,
            )
            event = None
            if incumbent[user] >= 0:
                source_local = np.where(
                    sequence.candidate_ids == incumbent[user]
                )[0]
                if len(source_local):
                    current = sequence.deterministic_snr_db[0]
                    slope = (
                        sequence.deterministic_snr_db[1] - current
                    ) / config.experiment.decision_interval_s
                    labels, trigger_mask = handover_failure_matrix(
                        current,
                        slope,
                        config.handover,
                        config.channel.outage_threshold_db,
                    )
                    source_index = int(source_local[0])
                    for local in range(config.experiment.candidate_cap):
                        if (
                            local != source_index
                            and not trigger_mask[source_index, local]
                        ):
                            scores[user, local] = -np.inf
                    event = (labels, trigger_mask)
            hof_events.append(event)
            snr = sequence.deterministic_snr_db[0]
            rates[user] = (
                config.channel.implementation_efficiency
                * config.channel.bandwidth_hz
                * np.log2(1.0 + 10.0 ** (snr / 10.0))
                / 1e6
            )
            rates[user, ~sequence.valid_mask[0]] = 0.0

        association = scheduler.associate(
            candidate_ids, scores, rates, average_rate
        )
        blocking_events += int(association.blocked.sum())
        for user, target in enumerate(association.assigned_satellite):
            if target < 0:
                continue
            transition_succeeded = True
            if incumbent[user] >= 0 and incumbent[user] != target:
                sequence = sequences[user]
                source_local = np.where(
                    sequence.candidate_ids == incumbent[user]
                )[0]
                target_local = np.where(
                    sequence.candidate_ids == target
                )[0]
                event = hof_events[user]
                if len(source_local) and len(target_local) and event is not None:
                    labels, trigger_mask = event
                    source_index = int(source_local[0])
                    target_index = int(target_local[0])
                    if trigger_mask[source_index, target_index]:
                        handovers += 1
                        transition_succeeded = not bool(
                            labels[source_index, target_index]
                        )
                        failures += int(not transition_succeeded)
                    else:
                        transition_succeeded = False
                else:
                    transition_succeeded = False
            if transition_succeeded:
                incumbent[user] = target
                rate_accumulator[user] += association.allocated_rate_mbps[user]
                average_rate[user] = (
                    0.9 * average_rate[user]
                    + 0.1 * association.allocated_rate_mbps[user]
                )
        active_satellites, active_counts = np.unique(
            incumbent[incumbent >= 0],
            return_counts=True,
        )
        previous_load = {
            int(satellite): float(count)
            / config.multi_ue.max_users_per_satellite
            for satellite, count in zip(active_satellites, active_counts)
        }

    # Time-average throughput includes blocked and failed-transition epochs as
    # zero service rather than conditioning on admission.
    per_user_rate = rate_accumulator / max(epoch_count, 1)
    return {
        "users": user_count,
        "method": method,
        "mean_throughput_mbps": float(per_user_rate.mean()),
        "p05_throughput_mbps": float(np.percentile(per_user_rate, 5.0)),
        "jain_index": jain_fairness(per_user_rate),
        "blocking_percent": 100.0
        * blocking_events
        / (user_count * epoch_count),
        "handovers_per_ue": handovers / user_count,
        "hof_percent": 100.0 * failures / max(handovers, 1),
        "satellite_capacity_mbps": config.multi_ue.satellite_capacity_mbps,
        "max_users_per_satellite": (
            config.multi_ue.max_users_per_satellite
        ),
        "minimum_admission_rate_mbps": (
            config.multi_ue.minimum_admission_rate_mbps
        ),
        "scheduler": config.multi_ue.scheduler,
        "association_update": config.multi_ue.association_update,
        "region_radius_km": config.multi_ue.region_radius_km,
        "epochs": epoch_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--users", default="50,100,200")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output", default="results/multi_ue/summary.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    novanet = NovaNetPolicy(cfg, args.checkpoint)
    rows = []
    for users in [int(value) for value in args.users.split(",")]:
        for method in METHODS:
            row = run_method(
                cfg,
                ephemeris,
                method,
                users,
                novanet,
                args.max_epochs,
            )
            rows.append(row)
            print(
                f"M={users:3d} {method:20s} "
                f"mean={row['mean_throughput_mbps']:.2f} "
                f"p05={row['p05_throughput_mbps']:.2f}",
                flush=True,
            )
    write_rows(args.output, rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
