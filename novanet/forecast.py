"""Causal construction of the future energy/candidate sequence used by Soft-DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import LinkBudget, MeasurementTracker
from .config import NovaNetConfig
from .ephemeris import Ephemeris
from .geometry import (
    UETrajectory,
    geometry_state,
    sky_dome_adjacency,
    time_to_leave_seconds,
)


def _finite_difference_at(
    values: np.ndarray,
    index: int,
    step_s: float,
) -> float:
    """Finite-difference a propagated geometry trace at one SGP4 epoch."""

    left = index - 1
    right = index + 1
    if left >= 0 and right < len(values):
        if np.isfinite(values[left]) and np.isfinite(values[right]):
            return float((values[right] - values[left]) / (2.0 * step_s))
    if right < len(values) and np.isfinite(values[index]) and np.isfinite(values[right]):
        return float((values[right] - values[index]) / step_s)
    if left >= 0 and np.isfinite(values[left]) and np.isfinite(values[index]):
        return float((values[index] - values[left]) / step_s)
    return 0.0


@dataclass(frozen=True)
class ForecastSequence:
    node_features: np.ndarray
    spatial_adjacency: np.ndarray
    valid_mask: np.ndarray
    candidate_ids: np.ndarray
    current_idx: int
    deterministic_snr_db: np.ndarray
    ttl_s: np.ndarray
    initial_freeze: int = 0


def _visible_ranked(
    elevations: np.ndarray, mask_deg: float, cap: int
) -> list[int]:
    valid = np.flatnonzero(np.isfinite(elevations) & (elevations >= mask_deg))
    order = valid[np.argsort(elevations[valid])[::-1]]
    return [int(value) for value in order[:cap]]


def _aligned_candidate_union(
    elevations_by_horizon: np.ndarray,
    mask_deg: float,
    cap: int,
    incumbent_id: int | None,
) -> np.ndarray:
    score: dict[int, float] = {}
    ranked_by_horizon: list[list[int]] = []
    for elevations in elevations_by_horizon:
        ranked = _visible_ranked(elevations, mask_deg, cap)
        ranked_by_horizon.append(ranked)
        for satellite_id in ranked:
            score[satellite_id] = max(
                score.get(satellite_id, -np.inf), elevations[satellite_id]
            )
    ordered = sorted(score, key=lambda sat: (-score[sat], sat))
    chosen: list[int] = []
    if incumbent_id is not None:
        chosen.append(int(incumbent_id))

    # A max-over-horizon truncation can otherwise retain eight high peaks that
    # are all absent in one forecast slot. Reserve one representative for
    # every uncovered slot before filling the remaining capacity by score.
    for ranked in ranked_by_horizon:
        if ranked and not any(satellite in chosen for satellite in ranked):
            chosen.append(ranked[0])
    for satellite in ordered:
        if satellite not in chosen:
            chosen.append(satellite)
        if len(chosen) == cap:
            break
    chosen = chosen[:cap]
    return np.asarray(chosen + [-1] * (cap - len(chosen)), dtype=np.int64)


def build_forecast_sequence(
    config: NovaNetConfig,
    ephemeris: Ephemeris,
    trajectory: UETrajectory,
    decision_index: int,
    measurement_tracker: MeasurementTracker,
    incumbent_id: int | None = None,
    link_budget: LinkBudget | None = None,
    initial_freeze: int = 0,
) -> ForecastSequence:
    """Construct the exact H-step input used at deployment.

    Future geometry and TTL are known from TLE/SGP4.  The sixth node feature is
    a decision-time measurement and is populated only at ``h=0``; future
    realized SINR is never inserted into the graph input.  Deterministic future
    SINR is returned separately for the residual/uncertainty head.
    """

    cfg = config
    budget = link_budget or LinkBudget(cfg.channel, seed=cfg.experiment.seed)
    stride = int(round(cfg.experiment.decision_interval_s / ephemeris.step_s))
    if not np.isclose(stride * ephemeris.step_s, cfg.experiment.decision_interval_s):
        raise ValueError("Ephemeris step must divide the decision interval")
    horizon_indices = decision_index + stride * np.arange(cfg.planner.horizon_steps)
    if horizon_indices[-1] >= ephemeris.num_steps:
        raise IndexError("Ephemeris does not cover the full planning horizon")

    num_satellites = ephemeris.num_satellites
    elevations = np.full(
        (cfg.planner.horizon_steps, num_satellites), np.nan, dtype=float
    )
    geometries: list[list] = []
    for horizon, ephemeris_index in enumerate(horizon_indices):
        elapsed_s = ephemeris.time_s(int(ephemeris_index))
        ue_position, ue_velocity = trajectory.state_at(elapsed_s)
        row = []
        for satellite_id in range(num_satellites):
            sat_position = ephemeris.position_m[ephemeris_index, satellite_id]
            sat_velocity = ephemeris.velocity_m_s[ephemeris_index, satellite_id]
            if np.all(np.isfinite(sat_position)) and np.all(
                np.isfinite(sat_velocity)
            ):
                state = geometry_state(
                    ue_position,
                    ue_velocity,
                    sat_position,
                    sat_velocity,
                )
                elevations[horizon, satellite_id] = state.elevation_deg
            else:
                state = None
            row.append(state)
        geometries.append(row)

    candidate_ids = _aligned_candidate_union(
        elevations,
        cfg.experiment.minimum_elevation_deg,
        cfg.experiment.candidate_cap,
        incumbent_id,
    )
    if np.all(candidate_ids < 0):
        raise RuntimeError("No satellite is visible in the planning horizon")

    current_visible = [
        candidate
        for candidate in candidate_ids
        if candidate >= 0
        and elevations[0, candidate] >= cfg.experiment.minimum_elevation_deg
    ]
    if not current_visible:
        raise RuntimeError("No valid incumbent or current visible candidate")
    if incumbent_id is not None and incumbent_id in candidate_ids:
        current_idx = int(np.where(candidate_ids == incumbent_id)[0][0])
    else:
        best = max(current_visible, key=lambda sat: elevations[0, sat])
        current_idx = int(np.where(candidate_ids == best)[0][0])

    horizon = cfg.planner.horizon_steps
    candidates = cfg.experiment.candidate_cap
    features = np.zeros((horizon, candidates, cfg.model.node_feature_dim), np.float32)
    adjacency = np.zeros((horizon, candidates, candidates), np.float32)
    valid_mask = np.zeros((horizon, candidates), dtype=bool)
    deterministic_snr = np.full((horizon, candidates), -100.0, np.float32)
    ttl_s = np.zeros((horizon, candidates), np.float32)

    current_time_s = ephemeris.time_s(decision_index)
    for local, satellite_id in enumerate(candidate_ids):
        if satellite_id < 0:
            continue
        elevation_trace = np.empty(ephemeris.num_steps, dtype=float)
        elevation_trace.fill(np.nan)
        for ephemeris_index in range(max(decision_index - 1, 0), ephemeris.num_steps):
            ue_position, ue_velocity = trajectory.state_at(
                ephemeris.time_s(ephemeris_index)
            )
            sat_position = ephemeris.position_m[ephemeris_index, satellite_id]
            sat_velocity = ephemeris.velocity_m_s[ephemeris_index, satellite_id]
            if np.all(np.isfinite(sat_position)) and np.all(
                np.isfinite(sat_velocity)
            ):
                elevation_trace[ephemeris_index] = geometry_state(
                    ue_position,
                    ue_velocity,
                    sat_position,
                    sat_velocity,
                ).elevation_deg

        now_geometry = geometries[0][satellite_id]
        if now_geometry is None:
            # A future candidate may be absent at the current propagation
            # epoch. Its causal prior starts from its first finite horizon.
            first_valid_horizon = next(
                h
                for h in range(horizon)
                if geometries[h][satellite_id] is not None
            )
            now_geometry = geometries[first_valid_horizon][satellite_id]
        deterministic_now = budget.evaluate(now_geometry, stochastic=False).sinr_db
        measured_now, _age_s, _available = measurement_tracker.current_fields(
            int(satellite_id), current_time_s, deterministic_now
        )
        for h, ephemeris_index in enumerate(horizon_indices):
            state = geometries[h][satellite_id]
            if state is None:
                continue
            visible = state.elevation_deg >= cfg.experiment.minimum_elevation_deg
            valid_mask[h, local] = visible
            link = budget.evaluate(state, stochastic=False)
            deterministic_snr[h, local] = link.sinr_db
            ttl_s[h, local] = time_to_leave_seconds(
                elevation_trace,
                int(ephemeris_index),
                ephemeris.step_s,
                cfg.experiment.minimum_elevation_deg,
            )
            features[h, local] = np.asarray(
                [
                    state.elevation_deg
                    / cfg.model.elevation_reference_deg,
                    _finite_difference_at(
                        elevation_trace,
                        int(ephemeris_index),
                        ephemeris.step_s,
                    )
                    / cfg.model.elevation_rate_reference_deg_s,
                    state.range_m / cfg.model.range_reference_m,
                    state.range_rate_m_s
                    / cfg.model.range_rate_reference_m_s,
                    ttl_s[h, local] / cfg.planner.ttl_reference_s,
                    (
                        measured_now / cfg.model.sinr_reference_db
                        if h == 0
                        else 0.0
                    ),
                ],
                dtype=np.float32,
            )

    for h in range(horizon):
        # A planning path cannot jump across a true no-coverage epoch.  Treat
        # the first empty horizon slot as an early terminal boundary and mask
        # every later slot.  Soft-DP consumes this terminal suffix directly;
        # deployment must not replace the learned planner with an unrelated
        # heuristic merely because coverage ends inside the forecast window.
        if not valid_mask[h].any():
            valid_mask[h:] = False
            break

    for h in range(horizon):
        los = np.zeros((candidates, 3), dtype=float)
        for local, satellite_id in enumerate(candidate_ids):
            if satellite_id >= 0:
                state = geometries[h][satellite_id]
                if state is not None:
                    los[local] = state.los_unit
        adjacency[h] = sky_dome_adjacency(
            los,
            valid_mask[h],
            cfg.model.graph_neighbors,
            cfg.model.adjacency_temperature,
        )
    return ForecastSequence(
        node_features=features,
        spatial_adjacency=adjacency,
        valid_mask=valid_mask,
        candidate_ids=candidate_ids,
        current_idx=current_idx,
        deterministic_snr_db=deterministic_snr,
        ttl_s=ttl_s,
        initial_freeze=int(initial_freeze),
    )
