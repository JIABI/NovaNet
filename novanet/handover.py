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
    execution_start_time_s: float | None = None


def evaluate_cho_attempt(
    source_snr_db: Callable[[float], float],
    target_snr_db: Callable[[float], float],
    handover: HandoverConfig,
    outage_threshold_db: float,
    event_step_s: float = 0.01,
    monitoring_horizon_s: float | None = None,
) -> CHOOutcome:
    """Apply the stored-target CHO rule over one decision interval.

    The timer may start at the first event-grid instant within the monitoring
    horizon for which the target conditions become true.  A violation resets
    it.  Once one full TTT interval has elapsed, execution proceeds even when
    its completion extends beyond the next decision epoch.
    """

    if event_step_s <= 0.0:
        raise ValueError("event_step_s must be positive")
    explicit_decision_boundary = monitoring_horizon_s is not None
    monitoring_horizon = (
        handover.ttt_s
        if not explicit_decision_boundary
        else float(monitoring_horizon_s)
    )
    if monitoring_horizon < handover.ttt_s:
        raise ValueError("monitoring_horizon_s must be at least the TTT")
    # An explicit monitoring horizon is the next decision boundary.  The
    # manuscript retains a target only when execution starts *before* that
    # boundary, so the endpoint itself is excluded.  The legacy short form
    # without a boundary still includes the TTT endpoint.
    trigger_times = np.arange(
        0.0,
        (
            monitoring_horizon
            if explicit_decision_boundary
            else monitoring_horizon + 0.5 * event_step_s
        ),
        event_step_s,
    )
    required_samples = int(round(handover.ttt_s / event_step_s)) + 1
    consecutive = 0
    execution_start: float | None = None
    for time in trigger_times:
        instant = float(time)
        target_snr = float(target_snr_db(instant))
        source_snr = float(source_snr_db(instant))
        if not np.isfinite(target_snr) or not np.isfinite(source_snr):
            raise ValueError("CHO monitoring received a non-finite SNR")
        condition = (
            target_snr - source_snr >= handover.hysteresis_db
            and target_snr >= outage_threshold_db
        )
        consecutive = consecutive + 1 if condition else 0
        if consecutive >= required_samples:
            execution_start = instant
            break
    if execution_start is None:
        return CHOOutcome(False, False, "ttt_or_hysteresis_not_sustained", None, 0.0)

    execution_end = execution_start + handover.execution_s
    failed = counterfactual_hof_label(
        target_snr_db,
        handover,
        outage_threshold_db,
        event_step_s=event_step_s,
        execution_start_s=execution_start,
    )
    if failed:
        return CHOOutcome(
            True,
            False,
            "target_outage_during_execution",
            execution_end,
            handover.execution_s,
            execution_start,
        )
    return CHOOutcome(
        True,
        True,
        None,
        execution_end,
        handover.execution_s,
        execution_start,
    )


def counterfactual_hof_label(
    target_snr_db: Callable[[float], float],
    handover: HandoverConfig,
    outage_threshold_db: float,
    event_step_s: float = 0.01,
    execution_start_s: float | None = None,
) -> bool:
    """Replay the execution window even when the CHO trigger was not met."""

    execution_start = (
        handover.ttt_s
        if execution_start_s is None
        else float(execution_start_s)
    )
    execution_samples = int(round(handover.execution_s / event_step_s))
    execution_times = execution_start + event_step_s * np.arange(
        1,
        execution_samples + 1,
        dtype=float,
    )
    target_values = np.asarray(
        [float(target_snr_db(float(time))) for time in execution_times],
        dtype=float,
    )
    if not np.isfinite(target_values).all():
        raise ValueError("CHO execution replay received a non-finite SNR")
    target_outage = (target_values < outage_threshold_db).astype(float)
    return bool(target_outage.mean() > handover.failure_outage_fraction)


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
    if not np.isfinite(current).all() or not np.isfinite(slope).all():
        raise ValueError("SNR values and slopes must be finite")
    candidates = len(current)
    labels = np.zeros((candidates, candidates), dtype=np.float32)
    mask = np.ones_like(labels, dtype=bool)
    np.fill_diagonal(mask, False)
    trigger_times = np.arange(
        0.0,
        handover.ttt_s + 0.5 * event_step_s,
        event_step_s,
    )
    execution_samples = int(round(handover.execution_s / event_step_s))
    execution_times = handover.ttt_s + event_step_s * np.arange(
        1,
        execution_samples + 1,
        dtype=float,
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
    hof_probability: np.ndarray,
    *,
    base_weight: float,
    dwell_weight: float,
    hof_weight: float,
) -> np.ndarray:
    """Reference NumPy implementation used by the simulator and tests."""

    indicator = np.asarray(switched, dtype=float)
    return indicator * (
        base_weight
        + dwell_weight * retained_dwell_normalized
        + hof_weight * hof_probability
    )
