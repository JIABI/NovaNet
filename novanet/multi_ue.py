"""Synchronous two-phase association and proportional-fair resource sharing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MultiUEConfig


@dataclass(frozen=True)
class AssociationResult:
    assigned_satellite: np.ndarray
    allocated_rate_mbps: np.ndarray
    blocked: np.ndarray
    satellite_load: dict[int, float]


class SynchronousAssociationScheduler:
    """All UEs rank on the same prior-load snapshot, then commit together."""

    def __init__(self, config: MultiUEConfig):
        if config.scheduler != "proportional_fair":
            raise ValueError("Only proportional_fair allocation is implemented")
        if config.association_update != "synchronous_two_phase":
            raise ValueError("Only synchronous_two_phase updates are implemented")
        self.config = config

    def associate(
        self,
        candidate_ids: np.ndarray,
        ranking_score: np.ndarray,
        achievable_rate_mbps: np.ndarray,
        previous_average_rate_mbps: np.ndarray | None = None,
    ) -> AssociationResult:
        candidate_ids = np.asarray(candidate_ids, dtype=int)
        score = np.asarray(ranking_score, dtype=float)
        link_rate = np.asarray(achievable_rate_mbps, dtype=float)
        if candidate_ids.shape != score.shape or score.shape != link_rate.shape:
            raise ValueError("candidate_ids, score, and link_rate shapes must match")
        users, candidates = score.shape
        if users == 0 or candidates == 0:
            raise ValueError("association arrays must have nonzero dimensions")
        real = candidate_ids >= 0
        if np.any(real & (np.isnan(score) | np.isposinf(score))):
            raise ValueError(
                "scores for real candidates cannot be NaN or positive infinity"
            )
        if np.any(real & (~np.isfinite(link_rate) | (link_rate < 0.0))):
            raise ValueError(
                "link rates for real candidates must be finite and nonnegative"
            )
        if self.config.blocking_cost is not None and not np.isfinite(
            float(self.config.blocking_cost)
        ):
            raise ValueError("blocking_cost must be finite when configured")
        if previous_average_rate_mbps is not None:
            raw_previous = np.asarray(previous_average_rate_mbps, dtype=float)
            if np.any(~np.isfinite(raw_previous)) or np.any(raw_previous < 0.0):
                raise ValueError(
                    "previous_average_rate_mbps must be finite and nonnegative"
                )
        previous = (
            np.ones(users, dtype=float)
            if previous_average_rate_mbps is None
            else np.maximum(raw_previous, 1e-3)
        )
        if previous.shape != (users,):
            raise ValueError("previous_average_rate_mbps must have shape [M]")
        dummy_score = (
            -float(self.config.blocking_cost)
            if self.config.blocking_cost is not None
            else -np.inf
        )

        # Phase 1: every UE forms its ordered list from the same prior-load
        # snapshot.  Ineligible links are omitted before coordination.
        eligible = (
            (candidate_ids >= 0)
            & np.isfinite(score)
            & (link_rate >= self.config.minimum_admission_rate_mbps)
        )
        order = np.argsort(-score, axis=1, kind="stable")
        preferences: list[list[int]] = []
        local_by_satellite: list[dict[int, int]] = []
        for user in range(users):
            preference: list[int] = []
            lookup: dict[int, int] = {}
            for local in order[user]:
                if not eligible[user, local]:
                    continue
                if score[user, local] <= dummy_score:
                    continue
                satellite = int(candidate_ids[user, local])
                if satellite in lookup:
                    continue
                lookup[satellite] = int(local)
                preference.append(satellite)
            preferences.append(preference)
            local_by_satellite.append(lookup)

        # Phase 2: synchronous application/retention rounds.  Each satellite
        # retains the lowest submitted planner costs, equivalently the highest
        # supplied ranking scores.  Rejected UEs advance to their next target;
        # exhausted lists select the unlimited dummy blocking option.
        retained: dict[int, list[int]] = {}
        next_choice = np.zeros(users, dtype=int)
        pending = set(range(users))
        while pending:
            applications: dict[int, list[int]] = {}
            exhausted: set[int] = set()
            for user in sorted(pending):
                if next_choice[user] >= len(preferences[user]):
                    exhausted.add(user)
                    continue
                satellite = preferences[user][next_choice[user]]
                applications.setdefault(satellite, []).append(user)
            pending = set()
            for satellite, applicants in applications.items():
                pool = sorted(set(retained.get(satellite, []) + applicants))
                ranked = sorted(
                    pool,
                    key=lambda user: (
                        -score[
                            user,
                            local_by_satellite[user][satellite],
                        ],
                        user,
                    ),
                )
                kept = ranked[: self.config.max_users_per_satellite]
                rejected = ranked[self.config.max_users_per_satellite :]
                retained[satellite] = kept
                for user in rejected:
                    next_choice[user] += 1
                    pending.add(user)
            # UEs with no remaining real target are represented by the dummy
            # option and stay unassigned; no special-case capacity is needed.
            pending.difference_update(exhausted)

        assigned = np.full(users, -1, dtype=int)
        groups: dict[int, list[int]] = {}
        for satellite, members in retained.items():
            groups[satellite] = list(members)
            assigned[np.asarray(members, dtype=int)] = satellite

        allocated = np.zeros(users, dtype=float)
        for satellite, members in groups.items():
            member_array = np.asarray(members, dtype=int)
            weights = 1.0 / previous[member_array]
            caps = np.asarray(
                [
                    link_rate[
                        user,
                        local_by_satellite[user][satellite],
                    ]
                    for user in member_array
                ],
                dtype=float,
            )
            priorities = np.asarray(
                [
                    score[user, local_by_satellite[user][satellite]]
                    for user in member_array
                ],
                dtype=float,
            )
            shares, admitted = _admission_aware_pf_allocation(
                weights,
                caps,
                self.config.satellite_capacity_mbps,
                self.config.minimum_admission_rate_mbps,
                priorities,
            )
            for user, share, is_admitted in zip(
                member_array, shares, admitted
            ):
                if not is_admitted:
                    assigned[user] = -1
                    continue
                local_matches = np.where(candidate_ids[user] == satellite)[0]
                local = int(local_matches[0])
                allocated[user] = min(float(share), float(link_rate[user, local]))

        blocked = assigned < 0
        assigned[blocked] = -1
        allocated[blocked] = 0.0
        load = {
            satellite: float(np.sum(assigned == satellite))
            / self.config.max_users_per_satellite
            for satellite in sorted(set(assigned[assigned >= 0]))
        }
        return AssociationResult(assigned, allocated, blocked, load)


def _capped_proportional_fair_allocation(
    weights: np.ndarray,
    link_caps_mbps: np.ndarray,
    capacity_mbps: float,
) -> np.ndarray:
    """Weighted proportional-fair allocation with per-link upper bounds."""

    weights = np.asarray(weights, dtype=float)
    caps = np.maximum(np.asarray(link_caps_mbps, dtype=float), 0.0)
    if weights.shape != caps.shape or weights.ndim != 1:
        raise ValueError("weights and link caps must be matching vectors")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("PF weights must be finite and positive")
    if np.any(~np.isfinite(caps)):
        raise ValueError("PF link caps must be finite")
    if not np.isfinite(capacity_mbps) or capacity_mbps < 0.0:
        raise ValueError("PF capacity must be finite and nonnegative")
    allocation = np.zeros_like(caps)
    active = np.ones(len(caps), dtype=bool)
    remaining = max(float(capacity_mbps), 0.0)
    while np.any(active) and remaining > 1e-12:
        active_weights = weights[active]
        proposal = remaining * active_weights / active_weights.sum()
        active_indices = np.flatnonzero(active)
        saturated = proposal >= caps[active] - allocation[active] - 1e-12
        if not np.any(saturated):
            allocation[active] += proposal
            remaining = 0.0
            break
        for index in active_indices[saturated]:
            increment = max(caps[index] - allocation[index], 0.0)
            allocation[index] += increment
            remaining -= increment
            active[index] = False
    return allocation


def _admission_aware_pf_allocation(
    weights: np.ndarray,
    link_caps_mbps: np.ndarray,
    capacity_mbps: float,
    minimum_admission_rate_mbps: float,
    admission_priority: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate PF shares, removing infeasible admissions one at a time.

    A one-shot PF allocation can leave one UE below the configured admission
    floor while other admitted UEs still share capacity with it.  This helper
    removes the lowest-priority UE among those below the floor and reruns PF
    until every retained UE is feasible.  The rerun recycles all released
    capacity; it also avoids the over-rejection caused by dropping every
    sub-threshold UE simultaneously.
    """

    weights = np.asarray(weights, dtype=float)
    caps = np.maximum(np.asarray(link_caps_mbps, dtype=float), 0.0)
    if weights.shape != caps.shape or weights.ndim != 1:
        raise ValueError("weights and link caps must be matching vectors")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("PF weights must be finite and positive")
    minimum = float(minimum_admission_rate_mbps)
    if minimum < 0.0:
        raise ValueError("minimum admission rate must be nonnegative")
    priority = (
        weights.copy()
        if admission_priority is None
        else np.asarray(admission_priority, dtype=float)
    )
    if priority.shape != weights.shape:
        raise ValueError("admission priority must match the PF vectors")
    if np.any(~np.isfinite(caps)) or np.any(~np.isfinite(priority)):
        raise ValueError("PF caps and admission priorities must be finite")
    if not np.isfinite(capacity_mbps) or capacity_mbps < 0.0:
        raise ValueError("PF capacity must be finite and nonnegative")

    active = caps >= minimum - 1e-12
    allocation = np.zeros_like(caps)
    while np.any(active):
        active_indices = np.flatnonzero(active)
        trial = _capped_proportional_fair_allocation(
            weights[active],
            caps[active],
            capacity_mbps,
        )
        below = active_indices[trial < minimum - 1e-12]
        if not len(below):
            allocation[active_indices] = trial
            break

        # Keep deterministic, higher-priority admissions when congestion
        # forces a choice.  For equal priorities, the lower UE index wins.
        rejected = min(
            (int(index) for index in below),
            key=lambda index: (float(priority[index]), -index),
        )
        active[rejected] = False

    return allocation, active


def allocate_fixed_associations(
    assigned_satellite: np.ndarray,
    candidate_ids: np.ndarray,
    achievable_rate_mbps: np.ndarray,
    previous_average_rate_mbps: np.ndarray,
    capacity_mbps: float,
    *,
    minimum_admission_rate_mbps: float = 0.0,
    max_users_per_satellite: int | None = None,
) -> np.ndarray:
    """Apply the common PF scheduler to a fixed serving association.

    This is used for the incumbent-service prefix before a configured CHO
    actually enters and completes execution.  It prevents that prefix from
    being either dropped or double-counted outside the satellite budget.
    """

    assigned = np.asarray(assigned_satellite, dtype=int)
    candidates = np.asarray(candidate_ids, dtype=int)
    link_rate = np.asarray(achievable_rate_mbps, dtype=float)
    previous = np.maximum(
        np.asarray(previous_average_rate_mbps, dtype=float), 1e-3
    )
    if candidates.shape != link_rate.shape:
        raise ValueError("candidate_ids and achievable_rate_mbps must match")
    if assigned.shape != (candidates.shape[0],) or previous.shape != assigned.shape:
        raise ValueError("assignment and previous rates must match the user axis")
    real = candidates >= 0
    if np.any(real & (~np.isfinite(link_rate) | (link_rate < 0.0))):
        raise ValueError(
            "link rates for real candidates must be finite and nonnegative"
        )
    raw_previous = np.asarray(previous_average_rate_mbps, dtype=float)
    if np.any(~np.isfinite(raw_previous)) or np.any(raw_previous < 0.0):
        raise ValueError(
            "previous_average_rate_mbps must be finite and nonnegative"
        )
    if not np.isfinite(capacity_mbps) or capacity_mbps < 0.0:
        raise ValueError("capacity_mbps must be finite and nonnegative")

    output = np.zeros(len(assigned), dtype=float)
    for satellite in sorted(set(int(value) for value in assigned if value >= 0)):
        members = []
        caps = []
        for user in np.flatnonzero(assigned == satellite):
            local = np.flatnonzero(candidates[user] == satellite)
            if not len(local):
                continue
            members.append(int(user))
            caps.append(float(link_rate[user, int(local[0])]))
        if not members:
            continue
        member_array = np.asarray(members, dtype=int)
        member_caps = np.asarray(caps, dtype=float)
        weights = 1.0 / previous[member_array]
        eligible = member_caps >= minimum_admission_rate_mbps - 1e-12
        if max_users_per_satellite is not None:
            if max_users_per_satellite <= 0:
                raise ValueError("max_users_per_satellite must be positive")
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices) > max_users_per_satellite:
                ordered = sorted(
                    (int(index) for index in eligible_indices),
                    key=lambda index: (
                        -float(weights[index]),
                        int(member_array[index]),
                    ),
                )
                eligible[:] = False
                eligible[np.asarray(ordered[:max_users_per_satellite])] = True
        shares = np.zeros(len(member_array), dtype=float)
        if np.any(eligible):
            local_shares, admitted = _admission_aware_pf_allocation(
                weights[eligible],
                member_caps[eligible],
                capacity_mbps,
                minimum_admission_rate_mbps,
                weights[eligible],
            )
            eligible_indices = np.flatnonzero(eligible)
            shares[eligible_indices[admitted]] = local_shares[admitted]
        output[member_array] = shares
    return output


def jain_fairness(rates: np.ndarray) -> float:
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("fairness rates must be a finite nonnegative vector")
    denominator = len(values) * float(np.square(values).sum())
    return 0.0 if denominator <= 0.0 else float(values.sum() ** 2 / denominator)
