#!/usr/bin/env python3
"""Select heuristic hyperparameters on paired held-out validation episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.common import build_paper_ephemeris, metrics_row, write_rows
from novanet.config import NovaNetConfig, load_config
from novanet.dataset import validate_tle_epoch
from novanet.policies import (
    DwellAwarePolicy,
    PeriodicHOPolicy,
    RateDwellPolicy,
)
from novanet.simulation import Scenario, simulate_single_user


PARAMETER_FIELDS = (
    "period_steps",
    "improvement_threshold",
    "rate_weight",
    "dwell_weight",
    "switch_penalty",
)
SUMMARY_METRICS = (
    "mean_target_cost",
    "mean_rate_mbps",
    "effective_throughput_mbps",
    "handovers",
    "hof_percent",
    "outage_percent",
    "ping_pong_percent",
    "p99_9_latency_ms",
)
FAMILY_ORDER = ("periodic_ho", "dwell_aware", "rate_dwell")


@dataclass(frozen=True)
class HeuristicSetting:
    setting_id: str
    family: str
    grid_index: int
    period_steps: int | None = None
    improvement_threshold: float | None = None
    rate_weight: float | None = None
    dwell_weight: float | None = None
    switch_penalty: float | None = None


@dataclass(frozen=True)
class ValidationEpisode:
    validation_seed: int
    user: int
    episode_seed: int
    scenario: Scenario


def parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(
            "integer grid values must be positive"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return values


def parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values or any(
        not math.isfinite(item) or item < 0.0 for item in values
    ):
        raise argparse.ArgumentTypeError(
            "floating-point grid values must be finite and nonnegative"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return values


def parse_seed_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated validation seeds"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "validation seeds must be nonnegative"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("validation seeds must be unique")
    return values


def resolve_validation_seeds(
    config: NovaNetConfig,
    requested: str | None,
    *,
    reserved_test_seed: int | None = None,
) -> tuple[int, ...]:
    """Return validation seeds disjoint from training and the reserved test RNG."""

    training_seed = int(config.experiment.seed)
    test_seed = (
        training_seed if reserved_test_seed is None else int(reserved_test_seed)
    )
    seeds = (
        (
            training_seed + 1_000,
            training_seed + 2_000,
            training_seed + 3_000,
        )
        if requested is None
        else parse_seed_grid(requested)
    )
    if training_seed in seeds:
        raise ValueError("validation seeds must not include the training seed")
    if test_seed in seeds:
        raise ValueError(
            "validation seeds must not include the reserved held-out test seed"
        )
    return tuple(seeds)


def _number_token(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_settings(
    periods: Iterable[int],
    improvement_thresholds: Iterable[float],
    dwell_weights: Iterable[float],
    switch_penalties: Iterable[float],
) -> list[HeuristicSetting]:
    """Construct the three declared validation grids without result values."""

    periods = tuple(int(value) for value in periods)
    thresholds = tuple(float(value) for value in improvement_thresholds)
    dwell = tuple(float(value) for value in dwell_weights)
    switch = tuple(float(value) for value in switch_penalties)
    if not periods or any(value < 1 for value in periods):
        raise ValueError("period grid must contain positive integers")
    for name, values in (
        ("improvement threshold", thresholds),
        ("dwell weight", dwell),
        ("switch penalty", switch),
    ):
        if not values or any(
            not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError(f"{name} grid must be finite and nonnegative")
    if any(
        len(set(values)) != len(values)
        for values in (periods, thresholds, dwell, switch)
    ):
        raise ValueError("parameter grids must not contain duplicates")

    settings: list[HeuristicSetting] = []
    for period in periods:
        settings.append(
            HeuristicSetting(
                setting_id=f"Periodic-HO_period-{period}",
                family="periodic_ho",
                grid_index=len(settings),
                period_steps=period,
            )
        )
    for threshold in thresholds:
        settings.append(
            HeuristicSetting(
                setting_id=(
                    "Dwell-Aware_threshold-"
                    + _number_token(threshold)
                ),
                family="dwell_aware",
                grid_index=len(settings),
                improvement_threshold=threshold,
            )
        )
    for dwell_weight in dwell:
        for switch_penalty in switch:
            settings.append(
                HeuristicSetting(
                    setting_id=(
                        "Rate-Dwell_dwell-"
                        f"{_number_token(dwell_weight)}_switch-"
                        f"{_number_token(switch_penalty)}"
                    ),
                    family="rate_dwell",
                    grid_index=len(settings),
                    rate_weight=1.0,
                    dwell_weight=dwell_weight,
                    switch_penalty=switch_penalty,
                )
            )
    return settings


def policy_for_setting(
    setting: HeuristicSetting,
    config: NovaNetConfig,
):
    if setting.family == "periodic_ho":
        policy = PeriodicHOPolicy(period_steps=int(setting.period_steps))
    elif setting.family == "dwell_aware":
        policy = DwellAwarePolicy(
            improvement_threshold=float(setting.improvement_threshold),
            ttl_reference_s=config.planner.ttl_reference_s,
        )
    elif setting.family == "rate_dwell":
        policy = RateDwellPolicy(
            rate_weight=float(setting.rate_weight),
            dwell_weight=float(setting.dwell_weight),
            switch_penalty=float(setting.switch_penalty),
            rate_reference_mbps=config.planner.rate_reference_mbps,
            ttl_reference_s=config.planner.ttl_reference_s,
            sinr_reference_db=config.model.sinr_reference_db,
            bandwidth_hz=config.channel.bandwidth_hz,
            implementation_efficiency=config.channel.implementation_efficiency,
        )
    else:
        raise ValueError(f"Unknown heuristic family {setting.family!r}")
    policy.name = setting.setting_id
    return policy


def paired_episode_seed(validation_seed: int, user: int) -> int:
    """Derive a stable common-random-number seed for one validation episode."""

    return int(
        np.random.SeedSequence(
            [int(validation_seed), int(user), 0x4854554E]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def make_validation_episodes(
    config: NovaNetConfig,
    validation_seeds: Iterable[int],
    users_per_seed: int,
) -> list[ValidationEpisode]:
    if users_per_seed < 1:
        raise ValueError("users_per_seed must be positive")
    episodes: list[ValidationEpisode] = []
    for validation_seed in validation_seeds:
        rng = np.random.default_rng(validation_seed)
        for user in range(users_per_seed):
            episodes.append(
                ValidationEpisode(
                    validation_seed=int(validation_seed),
                    user=user,
                    episode_seed=paired_episode_seed(validation_seed, user),
                    scenario=Scenario(
                        latitude_deg=float(
                            rng.uniform(*config.experiment.ue_latitude_deg)
                        ),
                        longitude_deg=float(
                            rng.uniform(*config.experiment.ue_longitude_deg)
                        ),
                        altitude_m=config.experiment.ue_altitude_m,
                        heading_deg=float(rng.uniform(0.0, 360.0)),
                    ),
                )
            )
    return episodes


def evaluate_settings(
    config: NovaNetConfig,
    ephemeris,
    settings: list[HeuristicSetting],
    episodes: list[ValidationEpisode],
    *,
    simulator: Callable | None = None,
    progress: bool = False,
) -> list[dict]:
    """Evaluate every setting on the same validation episodes and channel RNGs."""

    run_simulation = simulate_single_user if simulator is None else simulator
    rows: list[dict] = []
    for episode in episodes:
        for setting in settings:
            policy = policy_for_setting(setting, config)
            metrics = run_simulation(
                config,
                ephemeris,
                policy,
                episode.scenario,
                seed=episode.episode_seed,
                compute_oracle_cost=True,
            )
            if not math.isfinite(float(metrics.mean_target_cost)):
                raise RuntimeError(
                    "simulate_single_user did not return a finite "
                    "mean_target_cost with compute_oracle_cost=True"
                )
            row = metrics_row(metrics, episode.user)
            row.update(asdict(setting))
            row.update(
                {
                    "validation_seed": episode.validation_seed,
                    "episode_seed": episode.episode_seed,
                    "latitude_deg": episode.scenario.latitude_deg,
                    "longitude_deg": episode.scenario.longitude_deg,
                    "heading_deg": episode.scenario.heading_deg,
                }
            )
            rows.append(row)
            if progress:
                print(
                    f"setting={setting.setting_id} "
                    f"seed={episode.validation_seed} user={episode.user:03d} "
                    f"target_cost={metrics.mean_target_cost:.6f}",
                    flush=True,
                )
    return rows


def summarize_settings(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("No validation rows to summarize")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["setting_id"]), []).append(row)
    paired_keys: set[tuple[int, int, int]] | None = None
    for setting_id, members in grouped.items():
        keys = {
            (
                int(member["validation_seed"]),
                int(member["user"]),
                int(member["episode_seed"]),
            )
            for member in members
        }
        if len(keys) != len(members):
            raise ValueError(f"Duplicate validation episode for {setting_id}")
        if paired_keys is None:
            paired_keys = keys
        elif keys != paired_keys:
            raise ValueError(
                "Every heuristic setting must use the same paired validation "
                "episodes"
            )
    summaries: list[dict] = []
    for setting_id, members in grouped.items():
        first = members[0]
        summary: dict = {
            "setting_id": setting_id,
            "family": first["family"],
            "grid_index": int(first["grid_index"]),
            "replicates": len(members),
            "validation_seed_count": len(
                {int(member["validation_seed"]) for member in members}
            ),
        }
        for field in PARAMETER_FIELDS:
            summary[field] = first[field]
        for metric in SUMMARY_METRICS:
            values = np.asarray([member[metric] for member in members], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite validation metric {metric}")
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(
                values.std(ddof=1 if values.size > 1 else 0)
            )

        seed_means = []
        for validation_seed in sorted(
            {int(member["validation_seed"]) for member in members}
        ):
            values = [
                float(member["mean_target_cost"])
                for member in members
                if int(member["validation_seed"]) == validation_seed
            ]
            seed_means.append(float(np.mean(values)))
        summary["seed_mean_target_cost_std"] = float(
            np.std(seed_means, ddof=1 if len(seed_means) > 1 else 0)
        )
        attempts = sum(float(member["handovers"]) for member in members)
        failures = sum(
            float(member["handover_failures"]) for member in members
        )
        summary["hof_percent_pooled"] = (
            100.0 * failures / attempts if attempts > 0.0 else 0.0
        )
        summaries.append(summary)

    for family in FAMILY_ORDER:
        family_rows = [row for row in summaries if row["family"] == family]
        if not family_rows:
            raise ValueError(f"No summary rows for family {family}")
        ordered = sorted(
            family_rows,
            key=lambda row: (
                float(row["mean_target_cost_mean"]),
                int(row["grid_index"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["objective_rank"] = rank
            row["selected"] = rank == 1
    return sorted(summaries, key=lambda row: int(row["grid_index"]))


def selected_settings(summaries: list[dict]) -> list[dict]:
    selected = [dict(row) for row in summaries if bool(row.get("selected"))]
    by_family = {str(row["family"]): row for row in selected}
    if set(by_family) != set(FAMILY_ORDER):
        raise ValueError("Exactly one selected setting is required per family")
    return [by_family[family] for family in FAMILY_ORDER]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select Periodic-HO, Dwell-Aware, and Rate-Dwell parameters "
            "using only paired held-out validation episodes."
        )
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--periods",
        type=parse_int_grid,
        default=parse_int_grid("8,16,24"),
    )
    parser.add_argument(
        "--dwell-thresholds",
        type=parse_float_grid,
        default=parse_float_grid("0.05,0.10,0.15"),
    )
    parser.add_argument(
        "--rate-dwell-dwell-weights",
        type=parse_float_grid,
        default=parse_float_grid("0.25,0.50,0.75"),
    )
    parser.add_argument(
        "--rate-dwell-switch-penalties",
        type=parse_float_grid,
        default=parse_float_grid("0.10,0.20,0.30"),
    )
    parser.add_argument(
        "--validation-seeds",
        default=None,
        help=(
            "Comma-separated held-out seeds. Default: training seed plus "
            "1000, 2000, and 3000."
        ),
    )
    parser.add_argument(
        "--reserved-test-seed",
        type=int,
        default=None,
        help=(
            "Evaluation base seed protected from tuning; default is the "
            "configured paper seed (its RNG stream is domain-separated)."
        ),
    )
    parser.add_argument("--users-per-seed", type=int, default=20)
    parser.add_argument(
        "--output-dir", default="results/heuristic_tuning"
    )
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; selected values cannot be called paper values.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.users_per_seed < 1:
        raise ValueError("users_per_seed must be positive")
    reserved_test_seed = (
        config.experiment.seed
        if args.reserved_test_seed is None
        else args.reserved_test_seed
    )
    if reserved_test_seed < 0:
        raise ValueError("reserved_test_seed must be nonnegative")
    validation_seeds = resolve_validation_seeds(
        config,
        args.validation_seeds,
        reserved_test_seed=reserved_test_seed,
    )
    settings = make_settings(
        args.periods,
        args.dwell_thresholds,
        args.rate_dwell_dwell_weights,
        args.rate_dwell_switch_penalties,
    )
    episodes = make_validation_episodes(
        config, validation_seeds, args.users_per_seed
    )
    tle_provenance = validate_tle_epoch(
        config,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    ephemeris = build_paper_ephemeris(
        config, allow_stale_tle=args.allow_stale_tle
    )
    raw_rows = evaluate_settings(
        config,
        ephemeris,
        settings,
        episodes,
        progress=True,
    )
    summaries = summarize_settings(raw_rows)
    selected = selected_settings(summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "raw.csv", raw_rows)
    write_rows(output_dir / "summary.csv", summaries)
    write_rows(output_dir / "selected.csv", selected)
    protocol = {
        "purpose": "validation-only heuristic parameter selection",
        "objective": {
            "field": "mean_target_cost",
            "aggregation": "arithmetic mean over all paired validation episodes",
            "direction": "minimize",
            "source": (
                "simulate_single_user(compute_oracle_cost=True), using the "
                "same realized finite-horizon target energy as the offline "
                "first-step replay"
            ),
            "tie_break": "earlier declared grid order",
        },
        "test_set_used_for_selection": False,
        "training_seed": config.experiment.seed,
        "validation_seeds": list(validation_seeds),
        "reserved_test_seed": reserved_test_seed,
        "reserved_test_rng_domain": "EVAL-v1",
        "users_per_seed": args.users_per_seed,
        "validation_episode_count": len(episodes),
        "setting_count": len(settings),
        "simulation_count": len(raw_rows),
        "paired_scenarios_across_all_settings": True,
        "paired_channel_seed_across_all_settings": True,
        "episode_seed_derivation": "SeedSequence(validation_seed,user,0x4854554E)",
        "config_path": str(args.config),
        "canonical_config_fingerprint": config.fingerprint,
        "target_cost_planner_configuration": asdict(config.planner),
        "tle_provenance": tle_provenance,
        "diagnostic_stale_tle": bool(args.allow_stale_tle),
        "grids": {
            "period_steps": list(args.periods),
            "improvement_threshold": list(args.dwell_thresholds),
            "rate_weight": 1.0,
            "dwell_weight": list(args.rate_dwell_dwell_weights),
            "switch_penalty": list(args.rate_dwell_switch_penalties),
        },
        "settings": [asdict(setting) for setting in settings],
        "selected_setting_ids": [row["setting_id"] for row in selected],
        "outputs": ["raw.csv", "summary.csv", "selected.csv", "protocol.json"],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {output_dir / 'raw.csv'}, {output_dir / 'summary.csv'}, "
        f"{output_dir / 'selected.csv'}, and {output_dir / 'protocol.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
