#!/usr/bin/env python3
"""Load-aware multi-UE experiment with fully specified network mechanics."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.channel import LinkBudget, MeasurementTracker, RealizedChannelTrace
from novanet.config import load_config
from novanet.forecast import build_forecast_sequence
from novanet.geometry import UETrajectory, geometry_state
from novanet.handover import evaluate_cho_attempt
from novanet.multi_ue import (
    SynchronousAssociationScheduler,
    allocate_fixed_associations,
    jain_fairness,
)
from novanet.policies import NovaNetPolicy, SkipKPolicy

from experiments.common import (
    aggregate_rows,
    build_paper_ephemeris,
    write_protocol,
    write_rows,
)


METHODS = (
    "Max-Elevation-LB",
    "Max-ServeTime-LB",
    "Skip-1-LB",
    "Rate-Dwell-LB",
    "NovaNet-LB",
)


@dataclass(frozen=True)
class _PendingMultiUETransition:
    source_id: int
    target_id: int
    execution_start_s: float
    completion_s: float
    success: bool


def _allocate_epoch_service_segments(
    *,
    epoch_start_s: float,
    epoch_end_s: float,
    base_serving_satellite: np.ndarray,
    transitions: list[_PendingMultiUETransition | None],
    failed_epoch: np.ndarray,
    candidate_ids: np.ndarray,
    achievable_rate_mbps: np.ndarray,
    previous_average_rate_mbps: np.ndarray,
    multi_ue_config,
    coordinator_blocked: np.ndarray | None = None,
    rate_provider: Callable[[int, int, float], float] | None = None,
    rate_sample_interval_s: float | None = None,
) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    """Allocate service on every within-epoch association interval.

    A configured CHO does not move data-plane service to its target.  The UE
    remains on the source before execution, receives no service during the
    execution blackout, and moves to the target only after a successful
    completion.  Every event boundary therefore triggers a fresh PF allocation
    over the actual serving associations, not the configured targets.
    """

    start = float(epoch_start_s)
    end = float(epoch_end_s)
    if end <= start:
        raise ValueError("epoch_end_s must be after epoch_start_s")
    base = np.asarray(base_serving_satellite, dtype=int)
    failed = np.asarray(failed_epoch, dtype=bool)
    candidates = np.asarray(candidate_ids, dtype=int)
    rates = np.asarray(achievable_rate_mbps, dtype=float)
    previous = np.asarray(previous_average_rate_mbps, dtype=float)
    users = len(base)
    if len(transitions) != users:
        raise ValueError("transitions must match the user axis")
    if failed.shape != (users,) or previous.shape != (users,):
        raise ValueError("failed_epoch and previous rates must match users")
    if candidates.shape != rates.shape or candidates.shape[0] != users:
        raise ValueError("candidate matrices must match the user axis")
    base_blocked = (
        np.zeros(users, dtype=bool)
        if coordinator_blocked is None
        else np.asarray(coordinator_blocked, dtype=bool)
    )
    if base_blocked.shape != (users,):
        raise ValueError("coordinator_blocked must match users")

    boundaries = {start, end}
    if rate_sample_interval_s is not None:
        if rate_sample_interval_s <= 0.0:
            raise ValueError("rate_sample_interval_s must be positive")
        boundaries.update(
            float(value)
            for value in np.arange(
                start + rate_sample_interval_s,
                end,
                rate_sample_interval_s,
            )
        )
    for transition in transitions:
        if transition is None:
            continue
        if start < transition.execution_start_s < end:
            boundaries.add(float(transition.execution_start_s))
        if start < transition.completion_s < end:
            boundaries.add(float(transition.completion_s))

    duration = end - start
    epoch_rate = np.zeros(users, dtype=float)
    blocked_fraction = np.zeros(users, dtype=float)
    load_time: dict[int, float] = {}
    ordered = sorted(boundaries)
    for left, right in zip(ordered[:-1], ordered[1:]):
        segment_duration = right - left
        if segment_duration <= 1e-12:
            continue
        serving = base.copy()
        for user, transition in enumerate(transitions):
            if transition is None:
                continue
            if left < transition.execution_start_s - 1e-9:
                serving[user] = transition.source_id
            elif left < transition.completion_s - 1e-9:
                serving[user] = -1
            else:
                serving[user] = (
                    transition.target_id
                    if transition.success
                    else transition.source_id
                )

        segment_candidates = candidates.copy()
        # A transition carried from the preceding decision may complete onto
        # an identity absent from the new ranking candidate matrix.  Service
        # accounting is independent of ranking: retain the actual serving ID
        # in a temporary lookup row rather than dropping its rate to zero.
        for user, satellite in enumerate(serving):
            if satellite < 0 or np.any(segment_candidates[user] == satellite):
                continue
            empty = np.flatnonzero(segment_candidates[user] < 0)
            slot = int(empty[0]) if len(empty) else 0
            segment_candidates[user, slot] = satellite

        segment_rates = rates.copy()
        if rate_provider is not None:
            segment_rates = np.zeros_like(rates)
            for user in range(users):
                for local, satellite in enumerate(segment_candidates[user]):
                    if satellite >= 0:
                        segment_rates[user, local] = max(
                            float(rate_provider(user, int(satellite), left)),
                            0.0,
                        )
        allocated = allocate_fixed_associations(
            serving,
            segment_candidates,
            segment_rates,
            previous,
            multi_ue_config.satellite_capacity_mbps,
            minimum_admission_rate_mbps=(
                multi_ue_config.minimum_admission_rate_mbps
            ),
            max_users_per_satellite=(
                multi_ue_config.max_users_per_satellite
            ),
        )
        # The evaluation convention assigns a failed-transition UE zero
        # credited throughput for that epoch.  Keep it in the segmentwise PF
        # allocation, however, so knowing the later HOF outcome cannot release
        # its earlier capacity to other UEs.  This also implements the stated
        # no-post-failure-rescheduling rule.
        credited = allocated.copy()
        credited[failed] = 0.0
        epoch_rate += credited * (segment_duration / duration)

        capacity_blocked = (
            (serving >= 0)
            & (allocated < multi_ue_config.minimum_admission_rate_mbps - 1e-12)
        )
        segment_blocked = base_blocked | capacity_blocked
        # CHO execution and failed transitions are reliability events, not
        # capacity blocking; neither is added merely because serving=-1.
        segment_blocked &= ~failed
        blocked_fraction += segment_blocked * (segment_duration / duration)

        admitted = allocated >= (
            multi_ue_config.minimum_admission_rate_mbps - 1e-12
        )
        admitted &= allocated > 1e-12
        for satellite in sorted(set(serving[admitted])):
            if satellite < 0:
                continue
            count = int(np.sum(admitted & (serving == satellite)))
            load_time[int(satellite)] = load_time.get(int(satellite), 0.0) + (
                segment_duration
                * count
                / multi_ue_config.max_users_per_satellite
            )

    average_load = {
        satellite: occupied_time / duration
        for satellite, occupied_time in load_time.items()
    }
    return epoch_rate, average_load, blocked_fraction


def _realized_sinr_at_offset(
    config,
    ephemeris,
    trajectory: UETrajectory,
    trace: RealizedChannelTrace,
    decision_index: int,
    satellite_id: int,
    offset_s: float,
) -> float:
    base_time_s = ephemeris.time_s(decision_index)
    sample_time_s = base_time_s + offset_s
    ue_position, ue_velocity = trajectory.state_at(sample_time_s)
    satellite_position, satellite_velocity = ephemeris.state_at_time(
        satellite_id,
        sample_time_s,
    )
    if not (
        np.all(np.isfinite(satellite_position))
        and np.all(np.isfinite(satellite_velocity))
    ):
        return -100.0
    state = geometry_state(
        ue_position,
        ue_velocity,
        satellite_position,
        satellite_velocity,
    )
    if state.elevation_deg < config.experiment.minimum_elevation_deg:
        return -100.0
    return trace.evaluate(
        state,
        satellite_id,
        sample_time_s,
    ).sinr_db


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
    skip_policy: SkipKPolicy | None = None,
) -> np.ndarray:
    valid = sequence.valid_mask[0]
    if method == "Max-Elevation-LB":
        # Per-step, dimensionless preference on the same order-one scale as
        # the planner utility.  Do not z-score each UE independently: doing so
        # destroys cross-UE comparability at the shared capacity constraint.
        score = sequence.node_features[0, :, 0]
    elif method == "Max-ServeTime-LB":
        score = (
            sequence.ttl_s[0] / config.planner.ttl_reference_s
        )
    elif method == "Skip-1-LB":
        elevation = sequence.node_features[0, :, 0]
        if skip_policy is None:
            raise ValueError("Skip-1-LB requires a per-UE chronological state")
        # Apply the common load term before advancing the chronological
        # Skip-k state.  Otherwise the policy can record a raw-elevation
        # target that is not the target preferred by the load-adjusted rule.
        score = elevation.astype(float) - (
            config.planner.load_weight * candidate_load
        )
        adjusted_features = sequence.node_features.copy()
        adjusted_features[0, :, 0] = score
        selected = skip_policy.choose(
            replace(sequence, node_features=adjusted_features)
        )
        finite = score[valid]
        score[selected] = float(finite.max()) + 1e-6
    elif method == "Rate-Dwell-LB":
        measured_sinr_db = (
            sequence.node_features[0, :, 5]
            * config.model.sinr_reference_db
        )
        snr_linear = np.power(
            10.0, measured_sinr_db / 10.0
        )
        rate_mbps = (
            config.channel.implementation_efficiency
            * config.channel.bandwidth_hz
            * np.log2(1.0 + snr_linear)
            / 1e6
        )
        score = (
            rate_mbps / config.planner.rate_reference_mbps
            + 0.5
            * sequence.ttl_s[0]
            / config.planner.ttl_reference_s
            - 0.2
            * (np.arange(len(rate_mbps)) != sequence.current_idx)
        )
    elif method == "NovaNet-LB":
        load_forecast = np.broadcast_to(
            candidate_load[None, :],
            sequence.valid_mask.shape,
        ).copy()
        # NovaNet returns the negative first-step-conditioned H-step
        # cost-to-go from Eq. (45).  Keep that exact scale: the coordinator's
        # dummy cost c_blk is selected against the same horizon cost, so a
        # silent division by H would change the blocking decision.
        score = novanet.scores(
            sequence,
            context=user_context,
            load=load_forecast,
        )
    else:
        raise ValueError(method)
    score = np.asarray(score, dtype=float)
    score[~valid] = -np.inf
    if method not in {"NovaNet-LB", "Skip-1-LB"}:
        score[valid] -= (
            config.planner.load_weight * candidate_load[valid]
        )
    return score


def _result_paths(output: str | Path) -> tuple[Path, Path, Path]:
    rows_path = Path(output)
    summary_path = rows_path.with_name("summary.csv")
    protocol_path = rows_path.with_name("protocol.json")
    if rows_path.resolve() == summary_path.resolve():
        raise ValueError(
            "--output is the raw-row file and cannot be named summary.csv; "
            "use rows.csv or another distinct filename"
        )
    return rows_path, summary_path, protocol_path


def run_method(
    config,
    ephemeris,
    method: str,
    user_count: int,
    novanet: NovaNetPolicy,
    max_epochs: int | None,
    layout_seed: int,
) -> dict:
    rng = np.random.default_rng(layout_seed + user_count)
    trajectories = _users_in_disk(config, user_count, rng)
    trackers = [MeasurementTracker(config.channel) for _ in trajectories]
    budgets = [
        LinkBudget(config.channel, seed=layout_seed + index)
        for index in range(user_count)
    ]
    traces = [
        RealizedChannelTrace(
            config.channel,
            seed=layout_seed + index,
            ue_key=index,
            event_step_s=config.handover.event_step_s,
        )
        for index in range(user_count)
    ]
    scheduler = SynchronousAssociationScheduler(config.multi_ue)
    previous_load: dict[int, float] = {}
    incumbent = np.full(user_count, -1, dtype=int)
    freeze_left = np.zeros(user_count, dtype=int)
    pending_transitions: list[_PendingMultiUETransition | None] = [
        None
    ] * user_count
    average_rate = np.zeros(user_count, dtype=float)
    rate_accumulator = np.zeros(user_count, dtype=float)
    blocking_events = 0
    handovers = 0
    failures = 0
    failed_transition_epochs = 0
    novanet.reset()
    skip_policies = (
        [SkipKPolicy(skip=1) for _ in trajectories]
        if method == "Skip-1-LB"
        else [None] * user_count
    )
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
        decision_time_s = epoch * config.experiment.decision_interval_s
        epoch_end_s = decision_time_s + config.experiment.decision_interval_s
        for user, pending in enumerate(pending_transitions):
            if pending is None or pending.completion_s > decision_time_s + 1e-9:
                continue
            if pending.success:
                incumbent[user] = pending.target_id
                freeze_left[user] = config.handover.freeze_steps
            else:
                incumbent[user] = pending.source_id
                freeze_left[user] = max(freeze_left[user] - 1, 0)
            pending_transitions[user] = None
        candidate_ids = np.full(
            (user_count, config.experiment.candidate_cap), -1, dtype=int
        )
        scores = np.full_like(candidate_ids, -np.inf, dtype=float)
        rates = np.zeros_like(scores, dtype=float)
        sequences = [None] * user_count
        for user, (trajectory, tracker, budget, trace) in enumerate(
            zip(trajectories, trackers, budgets, traces)
        ):
            try:
                preliminary = build_forecast_sequence(
                    config,
                    ephemeris,
                    trajectory,
                    decision_index,
                    tracker,
                    incumbent_id=(
                        None
                        if incumbent[user] < 0
                        else int(incumbent[user])
                    ),
                    link_budget=budget,
                    initial_freeze=int(freeze_left[user]),
                )
            except RuntimeError:
                continue
            report_time = ephemeris.time_s(decision_index)
            for local, satellite_id in enumerate(preliminary.candidate_ids):
                if satellite_id < 0 or not preliminary.valid_mask[0, local]:
                    continue
                realized_sinr = _realized_sinr_at_offset(
                    config,
                    ephemeris,
                    trajectory,
                    trace,
                    decision_index,
                    int(satellite_id),
                    0.0,
                )
                if realized_sinr <= -100.0:
                    continue
                tracker.update(
                    int(satellite_id),
                    float(realized_sinr),
                    report_time,
                )
            sequence = build_forecast_sequence(
                config,
                ephemeris,
                trajectory,
                decision_index,
                tracker,
                incumbent_id=(
                    None if incumbent[user] < 0 else int(incumbent[user])
                ),
                link_budget=budget,
                initial_freeze=int(freeze_left[user]),
            )
            sequences[user] = sequence
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
                skip_policy=skip_policies[user],
            )
            pending = pending_transitions[user]
            if pending is not None:
                # An execution already under way cannot be replaced by the
                # current association round.  Excluding it here also avoids
                # reserving target capacity before realized completion.
                scores[user] = -np.inf
            source_local = np.where(
                sequence.candidate_ids == incumbent[user]
            )[0]
            source_visible = bool(
                len(source_local)
                and sequence.valid_mask[0, int(source_local[0])]
            )
            if (
                pending is None
                and freeze_left[user] > 0
                and source_visible
            ):
                source_index = int(source_local[0])
                scores[user] = -np.inf
                scores[user, source_index] = 0.0
            snr = (
                sequence.node_features[0, :, 5]
                * config.model.sinr_reference_db
            )
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
        pending_at_start = np.asarray(
            [transition is not None for transition in pending_transitions],
            dtype=bool,
        )
        coordinator_blocked = association.blocked & ~pending_at_start
        base_serving = association.assigned_satellite.copy()
        epoch_transitions = list(pending_transitions)
        failed_epoch = np.zeros(user_count, dtype=bool)
        for user, target in enumerate(association.assigned_satellite):
            pending = pending_transitions[user]
            if pending is not None:
                # Its exact source/blackout/post-completion service is handled
                # with the other UEs on the common event-boundary timeline.
                base_serving[user] = -1
                continue
            if target < 0:
                freeze_left[user] = max(freeze_left[user] - 1, 0)
                continue
            if incumbent[user] < 0 or incumbent[user] == target:
                incumbent[user] = target
                base_serving[user] = target
                freeze_left[user] = max(freeze_left[user] - 1, 0)
                continue

            sequence = sequences[user]
            if sequence is None:
                base_serving[user] = incumbent[user]
                freeze_left[user] = max(freeze_left[user] - 1, 0)
                continue
            source_id = int(incumbent[user])
            target_id = int(target)
            source = lambda offset: _realized_sinr_at_offset(
                config,
                ephemeris,
                trajectories[user],
                traces[user],
                decision_index,
                source_id,
                offset,
            )
            target_link = lambda offset: _realized_sinr_at_offset(
                config,
                ephemeris,
                trajectories[user],
                traces[user],
                decision_index,
                target_id,
                offset,
            )
            outcome = evaluate_cho_attempt(
                source,
                target_link,
                config.handover,
                config.channel.outage_threshold_db,
                event_step_s=config.handover.event_step_s,
                monitoring_horizon_s=config.experiment.decision_interval_s,
            )
            if not outcome.attempted:
                # A configured target that never enters execution does not
                # interrupt incumbent service.
                base_serving[user] = source_id
                freeze_left[user] = max(freeze_left[user] - 1, 0)
                continue

            handovers += 1
            transition = _PendingMultiUETransition(
                source_id=source_id,
                target_id=target_id,
                execution_start_s=(
                    decision_time_s
                    + float(outcome.execution_start_time_s)
                ),
                completion_s=(
                    decision_time_s + float(outcome.completion_time_s)
                ),
                success=bool(outcome.success),
            )
            epoch_transitions[user] = transition
            base_serving[user] = source_id
            if transition.success:
                if transition.completion_s <= epoch_end_s + 1e-9:
                    incumbent[user] = target_id
                    freeze_left[user] = config.handover.freeze_steps
                else:
                    pending_transitions[user] = transition
            else:
                failures += 1
                failed_transition_epochs += 1
                failed_epoch[user] = True
                if transition.completion_s > epoch_end_s + 1e-9:
                    pending_transitions[user] = transition
                else:
                    incumbent[user] = source_id
                    freeze_left[user] = max(freeze_left[user] - 1, 0)

        # Apply completions of transitions carried into this epoch only after
        # their segment plan has been retained in epoch_transitions.
        for user in np.flatnonzero(pending_at_start):
            pending = epoch_transitions[int(user)]
            if pending is None or pending.completion_s > epoch_end_s + 1e-9:
                continue
            if pending.success:
                incumbent[user] = pending.target_id
                freeze_left[user] = config.handover.freeze_steps
            else:
                incumbent[user] = pending.source_id
                freeze_left[user] = max(freeze_left[user] - 1, 0)
            pending_transitions[user] = None

        def realized_rate_provider(
            user: int,
            satellite_id: int,
            sample_time_s: float,
        ) -> float:
            sinr_db = _realized_sinr_at_offset(
                config,
                ephemeris,
                trajectories[user],
                traces[user],
                decision_index,
                satellite_id,
                sample_time_s - decision_time_s,
            )
            if sinr_db < config.channel.outage_threshold_db:
                return 0.0
            return float(
                config.channel.implementation_efficiency
                * config.channel.bandwidth_hz
                * np.log2(1.0 + 10.0 ** (sinr_db / 10.0))
                / 1e6
            )

        epoch_rate, realized_load, blocked_fraction = (
            _allocate_epoch_service_segments(
                epoch_start_s=decision_time_s,
                epoch_end_s=epoch_end_s,
                base_serving_satellite=base_serving,
                transitions=epoch_transitions,
                failed_epoch=failed_epoch,
                candidate_ids=candidate_ids,
                achievable_rate_mbps=rates,
                previous_average_rate_mbps=average_rate,
                multi_ue_config=config.multi_ue,
                coordinator_blocked=coordinator_blocked,
                rate_provider=realized_rate_provider,
                rate_sample_interval_s=(
                    config.experiment.geometry_subsample_s
                ),
            )
        )
        rate_accumulator += epoch_rate
        blocking_events += float(np.sum(blocked_fraction))
        # PF history is the exact running mean of prior epoch allocations;
        # no undisclosed EWMA coefficient is introduced.
        average_rate = (
            epoch * average_rate + epoch_rate
        ) / float(epoch + 1)
        previous_load = realized_load

    # Time-average throughput includes blocked and failed-transition epochs as
    # zero service rather than conditioning on admission.
    per_user_rate = rate_accumulator / max(epoch_count, 1)
    return {
        "users": user_count,
        "layout_seed": layout_seed,
        "condition": f"{user_count}|{method}",
        "method": method,
        "mean_throughput_mbps": float(per_user_rate.mean()),
        "p05_throughput_mbps": float(np.percentile(per_user_rate, 5.0)),
        "jain_index": jain_fairness(per_user_rate),
        "blocking_percent": 100.0
        * blocking_events
        / (user_count * epoch_count),
        "failed_transition_percent": 100.0
        * failed_transition_epochs
        / (user_count * epoch_count),
        "handovers_per_ue": handovers / user_count,
        "hof_percent": 100.0 * failures / max(handovers, 1),
        "freeze_steps": config.handover.freeze_steps,
        "execution_s": config.handover.execution_s,
        "pre_completion_service": "segmentwise_active_association_pf",
        "cross_epoch_execution": "pending_transition_carried",
        "satellite_capacity_mbps": config.multi_ue.satellite_capacity_mbps,
        "max_users_per_satellite": (
            config.multi_ue.max_users_per_satellite
        ),
        "minimum_admission_rate_mbps": (
            config.multi_ue.minimum_admission_rate_mbps
        ),
        "scheduler": config.multi_ue.scheduler,
        "pf_history": "running_mean_allocated_throughput",
        "association_update": config.multi_ue.association_update,
        "region_radius_km": config.multi_ue.region_radius_km,
        "blocking_cost": (
            config.multi_ue.blocking_cost
            if config.multi_ue.blocking_cost is not None
            else ""
        ),
        "epochs": epoch_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument("--users", default="50,100,200")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--layout-seeds",
        default="12025,12026,12027",
        help="Comma-separated held-out layout/channel seeds.",
    )
    parser.add_argument("--output", default="results/multi_ue/rows.csv")
    parser.add_argument("--allow-stale-tle", action="store_true")
    parser.add_argument(
        "--blocking-cost",
        type=float,
        default=None,
        help="Validation-selected c_blk for the unlimited dummy option.",
    )
    parser.add_argument(
        "--diagnostic-no-blocking-cost",
        action="store_true",
        help="Allow dummy blocking only after real candidates are exhausted.",
    )
    args = parser.parse_args()
    rows_path, summary_path, protocol_output = _result_paths(args.output)

    checkpoint_cfg = load_config(args.config)
    cfg = checkpoint_cfg
    if args.blocking_cost is not None:
        cfg = replace(
            cfg,
            multi_ue=replace(
                cfg.multi_ue,
                blocking_cost=args.blocking_cost,
            ),
        )
        cfg.validate()
    if (
        cfg.multi_ue.blocking_cost is None
        and not args.diagnostic_no_blocking_cost
    ):
        raise ValueError(
            "The manuscript defines c_blk but does not report its value. "
            "Supply --blocking-cost from the validation artifact. Use "
            "--diagnostic-no-blocking-cost only for software checks."
        )
    ephemeris = build_paper_ephemeris(
        cfg, allow_stale_tle=args.allow_stale_tle
    )
    novanet = NovaNetPolicy(
        checkpoint_cfg,
        args.checkpoint,
        require_paper_eligible=not args.allow_stale_tle,
    )
    rows = []
    layout_seeds = [
        int(value) for value in args.layout_seeds.split(",") if value.strip()
    ]
    if not layout_seeds:
        raise ValueError("At least one held-out layout seed is required")
    if cfg.experiment.seed in layout_seeds:
        raise ValueError(
            "Held-out layout seeds must not reuse the training seed"
        )
    for users in [int(value) for value in args.users.split(",")]:
        for layout_seed in layout_seeds:
            for method in METHODS:
                row = run_method(
                    cfg,
                    ephemeris,
                    method,
                    users,
                    novanet,
                    args.max_epochs,
                    layout_seed,
                )
                rows.append(row)
                print(
                    f"M={users:3d} seed={layout_seed} {method:20s} "
                    f"mean={row['mean_throughput_mbps']:.2f} "
                    f"p05={row['p05_throughput_mbps']:.2f}",
                    flush=True,
                )
    write_rows(rows_path, rows)
    write_rows(summary_path, aggregate_rows(rows, group_key="condition"))
    protocol_path = write_protocol(
        protocol_output,
        cfg,
        runner="multi_ue",
        checkpoint=args.checkpoint,
        diagnostic=(
            args.allow_stale_tle or args.diagnostic_no_blocking_cost
        ),
        details={
            "layout_seeds": layout_seeds,
            "user_counts": [
                int(value) for value in args.users.split(",")
            ],
            "blocking_cost": cfg.multi_ue.blocking_cost,
            "rate_sampling_interval_s": (
                cfg.experiment.geometry_subsample_s
            ),
            "pf_history": "running_mean_allocated_throughput",
        },
    )
    print(f"wrote {rows_path}, {summary_path}, and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
