#!/usr/bin/env python3
"""Validation-only sensitivity and Pareto analysis for planner weights.

The learned checkpoint is always loaded against the unmodified canonical
configuration.  Only the fixed, non-learned planner coefficients are changed
after that strict checkpoint/TLE/architecture check has succeeded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from novanet.config import NovaNetConfig, load_config
from novanet.dataset import validate_tle_epoch
from novanet.policies import NovaNetPolicy
from novanet.simulation import Scenario, simulate_single_user

from experiments.common import build_paper_ephemeris, metrics_row, write_rows


# lambda_u is the manuscript name of the LCB coefficient stored as lcb_kappa.
WEIGHT_TO_CONFIG = {
    "alpha": "alpha",
    "beta": "beta",
    "c0": "c0",
    "c1": "c1",
    "c2": "c2",
    "lambda_u": "lcb_kappa",
}
CORE_WEIGHTS = ("alpha", "beta", "c0", "c1", "c2")
PARETO_MAXIMIZE = ("effective_throughput_mbps_mean",)
PARETO_MINIMIZE = (
    "handovers_mean",
    "hof_percent_pooled",
    "outage_percent_mean",
)


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one multiplier is required")
    if any(not np.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("Multipliers must be finite and nonnegative")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one validation seed is required")
    if any(seed < 0 for seed in values):
        raise ValueError("Validation seeds must be nonnegative")
    if len(set(values)) != len(values):
        raise ValueError("Validation seeds must be unique")
    return values


def _base_weights(config: NovaNetConfig) -> dict[str, float]:
    return {
        manuscript_name: float(
            getattr(config.planner, config_name)
        )
        for manuscript_name, config_name in WEIGHT_TO_CONFIG.items()
    }


def _setting_id(
    varied_weight: str,
    multiplier: float,
    ordinal: int,
) -> str:
    token = f"{multiplier:g}".replace("-", "m").replace(".", "p")
    return f"{ordinal:03d}_{varied_weight}_x{token}"


def make_weight_settings(
    config: NovaNetConfig,
    multipliers: Iterable[float],
    *,
    design: str = "one-at-a-time",
    include_lambda_u: bool = False,
) -> list[dict[str, float | str]]:
    """Return deterministic settings without inserting any result values."""

    factors = tuple(float(value) for value in multipliers)
    if not factors:
        raise ValueError("At least one multiplier is required")
    if any(not np.isfinite(value) or value < 0.0 for value in factors):
        raise ValueError("Multipliers must be finite and nonnegative")
    active = CORE_WEIGHTS + (("lambda_u",) if include_lambda_u else ())
    base = _base_weights(config)
    settings: list[dict[str, float | str]] = []

    def append(
        weights: dict[str, float],
        varied_weight: str,
        multiplier: float | str,
    ) -> None:
        signature = tuple(round(float(weights[name]), 14) for name in active)
        if any(
            tuple(round(float(row[name]), 14) for name in active) == signature
            for row in settings
        ):
            return
        ordinal = len(settings)
        setting_id = (
            "000_nominal"
            if varied_weight == "nominal"
            else _setting_id(varied_weight, float(multiplier), ordinal)
        )
        row: dict[str, float | str] = {
            "setting_id": setting_id,
            "varied_weight": varied_weight,
            "multiplier": multiplier,
        }
        row.update(weights)
        settings.append(row)

    append(dict(base), "nominal", "nominal")
    if design == "one-at-a-time":
        for name in active:
            for factor in factors:
                if np.isclose(factor, 1.0):
                    continue
                weights = dict(base)
                weights[name] *= factor
                append(weights, name, factor)
    elif design == "cartesian":
        for factor_tuple in product(factors, repeat=len(active)):
            weights = dict(base)
            for name, factor in zip(active, factor_tuple):
                weights[name] *= factor
            # The full factor tuple is recorded in the setting ID while the
            # exact coefficient values remain separate machine-readable fields.
            signature = "_".join(
                f"{name}x{factor:g}" for name, factor in zip(active, factor_tuple)
            )
            append(weights, signature, 1.0)
    else:
        raise ValueError("design must be 'one-at-a-time' or 'cartesian'")
    return settings


def _config_differences(
    reference: object,
    candidate: object,
    prefix: str = "",
) -> set[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences: set[str] = set()
        for key in reference.keys() | candidate.keys():
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                differences.add(child)
            else:
                differences.update(
                    _config_differences(reference[key], candidate[key], child)
                )
        return differences
    return set() if reference == candidate else {prefix}


def config_with_runtime_weights(
    base_config: NovaNetConfig,
    setting: dict[str, float | str],
) -> NovaNetConfig:
    """Create a runtime config and reject any non-weight difference."""

    metadata_keys = {"setting_id", "varied_weight", "multiplier"}
    allowed_keys = set(WEIGHT_TO_CONFIG) | metadata_keys
    unknown = set(setting) - allowed_keys
    missing = set(WEIGHT_TO_CONFIG) - set(setting)
    if unknown or missing:
        raise ValueError(
            "Weight setting must contain only the declared planner weights "
            f"and metadata; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    planner_updates = {
        config_name: float(setting[manuscript_name])
        for manuscript_name, config_name in WEIGHT_TO_CONFIG.items()
    }
    runtime = replace(
        base_config,
        planner=replace(base_config.planner, **planner_updates),
    )
    runtime.validate()
    differences = _config_differences(asdict(base_config), asdict(runtime))
    allowed = {f"planner.{name}" for name in WEIGHT_TO_CONFIG.values()}
    disallowed = differences - allowed
    if disallowed:
        raise ValueError(
            "Sensitivity run attempted non-weight configuration changes: "
            f"{sorted(disallowed)}"
        )
    return runtime


def load_policy_for_setting(
    base_config: NovaNetConfig,
    checkpoint: str | Path,
    setting: dict[str, float | str],
    *,
    device: str | None = None,
    require_paper_eligible: bool = True,
) -> tuple[NovaNetPolicy, NovaNetConfig]:
    """Strictly load the checkpoint, then install only fixed planner weights."""

    # No mismatch escape hatch is used here.  This checks the complete config
    # fingerprint, strict state-dict shape, and TLE/subset provenance first.
    policy = NovaNetPolicy(
        base_config,
        checkpoint,
        allow_config_mismatch=False,
        require_paper_eligible=require_paper_eligible,
        device=device,
    )
    runtime = config_with_runtime_weights(base_config, setting)
    policy.config = runtime
    policy.model.config = runtime
    energy = policy.model.energy
    energy.alpha = runtime.planner.alpha
    energy.beta = runtime.planner.beta
    energy.c0 = runtime.planner.c0
    energy.c1 = runtime.planner.c1
    energy.c2 = runtime.planner.c2
    energy.lcb_kappa = runtime.planner.lcb_kappa
    policy.name = str(setting["setting_id"])
    return policy, runtime


def validation_scenarios(
    config: NovaNetConfig,
    validation_seeds: Iterable[int],
    users_per_seed: int,
) -> list[tuple[int, int, Scenario]]:
    if users_per_seed < 1:
        raise ValueError("users-per-seed must be positive")
    rows: list[tuple[int, int, Scenario]] = []
    for validation_seed in validation_seeds:
        rng = np.random.default_rng(validation_seed)
        for user in range(users_per_seed):
            rows.append(
                (
                    validation_seed,
                    user,
                    Scenario(
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
    return rows


def episode_seed(validation_seed: int, user: int) -> int:
    """Derive a paired, collision-resistant episode seed from two indices."""

    return int(
        np.random.SeedSequence(
            [validation_seed, user, 0x4E4F5641]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def summarize_settings(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["setting_id"]), []).append(row)
    summaries: list[dict] = []
    metric_names = (
        "mean_rate_mbps",
        "effective_throughput_mbps",
        "handovers",
        "hof_percent",
        "outage_percent",
        "ping_pong_percent",
    )
    for setting_id, members in grouped.items():
        first = members[0]
        summary: dict[str, float | int | str | bool] = {
            "setting_id": setting_id,
            "varied_weight": first["varied_weight"],
            "multiplier": first["multiplier"],
            "replicates": len(members),
        }
        for name in WEIGHT_TO_CONFIG:
            summary[name] = float(first[name])
        for metric in metric_names:
            values = np.asarray([row[metric] for row in members], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(
                values.std(ddof=1 if len(values) > 1 else 0)
            )
        attempts = sum(float(row["handovers"]) for row in members)
        failures = sum(float(row["handover_failures"]) for row in members)
        summary["hof_percent_pooled"] = (
            100.0 * failures / attempts if attempts > 0.0 else 0.0
        )
        summaries.append(summary)
    return summaries


def pareto_mask(
    rows: list[dict],
    *,
    maximize: Iterable[str] = PARETO_MAXIMIZE,
    minimize: Iterable[str] = PARETO_MINIMIZE,
) -> list[bool]:
    """Return the exact multiobjective non-dominance mask."""

    maximize = tuple(maximize)
    minimize = tuple(minimize)
    mask: list[bool] = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            no_worse = all(float(other[key]) >= float(row[key]) for key in maximize)
            no_worse &= all(float(other[key]) <= float(row[key]) for key in minimize)
            strictly_better = any(
                float(other[key]) > float(row[key]) for key in maximize
            ) or any(float(other[key]) < float(row[key]) for key in minimize)
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _checkpoint_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-only sensitivity and Pareto analysis for "
            "alpha,beta,c0,c1,c2 and optionally lambda_u."
        )
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument(
        "--design",
        choices=("one-at-a-time", "cartesian"),
        default="one-at-a-time",
        help="OAT is the tractable default; Cartesian must be requested explicitly.",
    )
    parser.add_argument(
        "--multipliers",
        default="0.5,1.0,1.5",
        help="Nonnegative factors applied to nominal coefficients.",
    )
    parser.add_argument(
        "--include-lambda-u",
        action="store_true",
        help="Also vary the LCB uncertainty coefficient lambda_u (lcb_kappa).",
    )
    parser.add_argument(
        "--validation-seeds",
        default="3025,4025,5025",
        help="Independent scenario seeds; the training seed is rejected.",
    )
    parser.add_argument("--users-per-seed", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/weight_pareto",
    )
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; outputs are labeled non-reproducible.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing trained checkpoint {checkpoint}; a random model is not "
            "a valid weight-sensitivity reference."
        )
    validation_seeds = parse_int_list(args.validation_seeds)
    if config.experiment.seed in validation_seeds:
        raise ValueError(
            "Validation seeds must not include the configured training seed "
            f"{config.experiment.seed}."
        )
    settings = make_weight_settings(
        config,
        parse_float_list(args.multipliers),
        design=args.design,
        include_lambda_u=args.include_lambda_u,
    )
    scenarios = validation_scenarios(
        config,
        validation_seeds,
        args.users_per_seed,
    )
    ephemeris = build_paper_ephemeris(
        config,
        allow_stale_tle=args.allow_stale_tle,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    for setting in settings:
        policy, runtime_config = load_policy_for_setting(
            config,
            checkpoint,
            setting,
            device=args.device,
            require_paper_eligible=not args.allow_stale_tle,
        )
        for validation_seed, user, scenario in scenarios:
            channel_seed = episode_seed(validation_seed, user)
            metrics = simulate_single_user(
                runtime_config,
                ephemeris,
                policy,
                scenario,
                seed=channel_seed,
            )
            row = metrics_row(metrics, user)
            row.update(
                {
                    "setting_id": setting["setting_id"],
                    "varied_weight": setting["varied_weight"],
                    "multiplier": setting["multiplier"],
                    "validation_seed": validation_seed,
                    "episode_seed": channel_seed,
                    "latitude_deg": scenario.latitude_deg,
                    "longitude_deg": scenario.longitude_deg,
                    "heading_deg": scenario.heading_deg,
                }
            )
            for name in WEIGHT_TO_CONFIG:
                row[name] = float(setting[name])
            raw_rows.append(row)
            print(
                f"setting={setting['setting_id']} seed={validation_seed} "
                f"user={user:03d} throughput={metrics.effective_throughput_mbps:.3f} "
                f"HO={metrics.handovers} HOF={metrics.hof_percent:.3f}%",
                flush=True,
            )

    summaries = summarize_settings(raw_rows)
    nondominated = pareto_mask(summaries)
    for row, keep in zip(summaries, nondominated):
        row["pareto_optimal"] = keep
    pareto_rows = [row for row in summaries if row["pareto_optimal"]]
    write_rows(output_dir / "raw.csv", raw_rows)
    write_rows(output_dir / "summary.csv", summaries)
    write_rows(output_dir / "pareto.csv", pareto_rows)

    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    current_tle = validate_tle_epoch(
        config,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    protocol = {
        "purpose": "validation-only planner-weight sensitivity",
        "test_set_used_for_selection": False,
        "config_path": str(args.config),
        "canonical_config_fingerprint": config.fingerprint,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "checkpoint_config_fingerprint": checkpoint_payload.get(
            "config_fingerprint"
        ),
        "checkpoint_tle_provenance": checkpoint_payload.get("tle_provenance"),
        "current_tle_provenance": current_tle,
        "checkpoint_loading": (
            "strict canonical fingerprint, strict state_dict, and exact "
            "TLE/subset provenance; no config-mismatch bypass"
        ),
        "runtime_overrides": [
            f"planner.{name}" for name in WEIGHT_TO_CONFIG.values()
        ],
        "learned_parameters_changed": False,
        "design": args.design,
        "multipliers": list(parse_float_list(args.multipliers)),
        "include_lambda_u": bool(args.include_lambda_u),
        "training_seed": config.experiment.seed,
        "validation_seeds": list(validation_seeds),
        "users_per_seed": args.users_per_seed,
        "paired_scenarios_across_settings": True,
        "pareto_maximize": list(PARETO_MAXIMIZE),
        "pareto_minimize": list(PARETO_MINIMIZE),
        "diagnostic_stale_tle": bool(args.allow_stale_tle),
        "settings": settings,
        "outputs": ["raw.csv", "summary.csv", "pareto.csv"],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(
        f"wrote {output_dir / 'raw.csv'}, {output_dir / 'summary.csv'}, "
        f"{output_dir / 'pareto.csv'}, and {output_dir / 'protocol.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
