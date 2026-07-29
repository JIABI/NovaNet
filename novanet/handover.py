"""Event-level CHO timing and reproducible handover-failure labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import HandoverConfig


@dataclass(frozen=True)
class CHOOutcome:
    attempted: bool
    success: bool
    failure_reason: str | None
    completion_time_s: float | None
    interruption_s: float


def evaluate_cho_attempt(
    source_snr_db: Callable[[float], float],
    target_snr_db: Callable[[float], float],
    handover: HandoverConfig,
    outage_threshold_db: float,
    event_step_s: float = 0.01,
) -> CHOOutcome:
    """Apply TTT, hysteresis, and execution checks at event-level resolution."""

    trigger_times = np.arange(
        0.0, handover.ttt_s + 0.5 * event_step_s, event_step_s
    )
    condition = [
        target_snr_db(float(time))
        - source_snr_db(float(time))
        >= handover.hysteresis_db
        and target_snr_db(float(time)) >= outage_threshold_db
        for time in trigger_times
    ]
    if not all(condition):
        return CHOOutcome(False, False, "ttt_or_hysteresis_not_sustained", None, 0.0)

    execution_start = handover.ttt_s
    execution_end = handover.ttt_s + handover.execution_s
    execution_times = np.arange(
        execution_start,
        execution_end + 0.5 * event_step_s,
        event_step_s,
    )
    target_outage = np.asarray(
        [
            target_snr_db(float(time)) < outage_threshold_db
            for time in execution_times
        ],
        dtype=float,
    )
    if target_outage.mean() > handover.failure_outage_fraction:
        return CHOOutcome(
            True,
            False,
            "target_outage_during_execution",
            execution_end,
            handover.execution_s,
        )
    return CHOOutcome(
        True,
        True,
        None,
        execution_end,
        handover.execution_s,
    )


def handover_failure_matrix(
    snr_now_db: np.ndarray,
    snr_slope_db_s: np.ndarray,
    handover: HandoverConfig,
    outage_threshold_db: float,
    event_step_s: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Event-level pair labels and masks for the HOF head.

    The mask excludes stay transitions and pairs that never sustain the CHO
    trigger, because HOF is conditional on an attempted handover. An attempted
    pair is labeled failed when the target is in outage for too much of the
    execution window. This uses the same 10 ms event grid and failure rule as
    :func:`evaluate_cho_attempt`.
    """

    current = np.asarray(snr_now_db, dtype=float)
    slope = np.asarray(snr_slope_db_s, dtype=float)
    if current.shape != slope.shape or current.ndim != 1:
        raise ValueError("snr_now_db and snr_slope_db_s must be matching vectors")
    candidates = len(current)
    labels = np.zeros((candidates, candidates), dtype=np.float32)
    mask = np.ones_like(labels, dtype=bool)
    np.fill_diagonal(mask, False)
    trigger_times = np.arange(
        0.0,
        handover.ttt_s + 0.5 * event_step_s,
        event_step_s,
    )
    execution_times = np.arange(
        handover.ttt_s,
        handover.ttt_s + handover.execution_s + 0.5 * event_step_s,
        event_step_s,
    )
    for source_index in range(candidates):
        for target_index in range(candidates):
            if source_index == target_index:
                continue
            source_trigger = (
                current[source_index]
                + slope[source_index] * trigger_times
            )
            target_trigger = (
                current[target_index]
                + slope[target_index] * trigger_times
            )
            sustained = bool(
                np.all(
                    (target_trigger - source_trigger >= handover.hysteresis_db)
                    & (target_trigger >= outage_threshold_db)
                )
            )
            target_execution = (
                current[target_index]
                + slope[target_index] * execution_times
            )
            completion_ok = bool(
                np.mean(target_execution < outage_threshold_db)
                <= handover.failure_outage_fraction
            )
            if not sustained:
                mask[source_index, target_index] = False
                continue
            labels[source_index, target_index] = float(not completion_ok)
    return labels, mask


def dimensionless_transition_cost(
    switched: np.ndarray,
    retained_dwell_normalized: np.ndarray,
    angular_speed_normalized: np.ndarray,
    hof_probability: np.ndarray,
    *,
    base_weight: float,
    dwell_weight: float,
    angular_speed_weight: float,
    hof_weight: float,
) -> np.ndarray:
    """Reference NumPy implementation used by the simulator and tests."""

    indicator = np.asarray(switched, dtype=float)
    return indicator * (
        base_weight
        + dwell_weight * retained_dwell_normalized
        + angular_speed_weight * angular_speed_normalized
        + hof_weight * hof_probability
    )
