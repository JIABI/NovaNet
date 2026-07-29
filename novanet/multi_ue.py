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
        previous = (
            np.ones(users, dtype=float)
            if previous_average_rate_mbps is None
            else np.maximum(np.asarray(previous_average_rate_mbps, dtype=float), 1e-3)
        )
        if previous.shape != (users,):
            raise ValueError("previous_average_rate_mbps must have shape [M]")

        # Phase 1: every UE ranks from the same previous-epoch load snapshot.
        order = np.argsort(-score, axis=1)
        regret = np.take_along_axis(score, order[:, :1], axis=1).squeeze(1)
        if candidates > 1:
            regret -= np.take_along_axis(score, order[:, 1:2], axis=1).squeeze(1)
        ue_commit_order = np.argsort(-regret, kind="stable")

        # Phase 2: deterministic batch commit under per-satellite admission caps.
        assigned = np.full(users, -1, dtype=int)
        groups: dict[int, list[int]] = {}
        for user in ue_commit_order:
            for local in order[user]:
                satellite = int(candidate_ids[user, local])
                if satellite < 0 or link_rate[user, local] <= 0.0:
                    continue
                members = groups.setdefault(satellite, [])
                if len(members) < self.config.max_users_per_satellite:
                    assigned[user] = satellite
                    members.append(int(user))
                    break

        allocated = np.zeros(users, dtype=float)
        for satellite, members in groups.items():
            member_array = np.asarray(members, dtype=int)
            weights = 1.0 / previous[member_array]
            shares = (
                self.config.satellite_capacity_mbps
                * weights
                / weights.sum()
            )
            for user, share in zip(member_array, shares):
                local_matches = np.where(candidate_ids[user] == satellite)[0]
                local = int(local_matches[0])
                allocated[user] = min(float(share), float(link_rate[user, local]))

        blocked = (
            (assigned < 0)
            | (allocated < self.config.minimum_admission_rate_mbps)
        )
        assigned[blocked] = -1
        allocated[blocked] = 0.0
        load = {
            satellite: float(np.sum(assigned == satellite))
            / self.config.max_users_per_satellite
            for satellite in groups
        }
        return AssociationResult(assigned, allocated, blocked, load)


def jain_fairness(rates: np.ndarray) -> float:
    values = np.asarray(rates, dtype=float)
    denominator = len(values) * float(np.square(values).sum())
    return 0.0 if denominator <= 0.0 else float(values.sum() ** 2 / denominator)
