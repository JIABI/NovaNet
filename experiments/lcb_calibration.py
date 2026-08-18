#!/usr/bin/env python3
"""Recompute the Appendix-A/Liu9 uncertainty-calibration artifacts.

By default this runner regenerates the canonical 3,600-sequence stream and
uses its final 20%, exactly matching ``train_oaest.py``.  An explicitly named
independent diagnostic mode remains available for fast software smoke tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.common import write_rows
from novanet.config import NovaNetConfig, load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)
from novanet.policies import NovaNetPolicy


DB_PER_LOG_RESIDUAL = 10.0 / math.log(10.0)


def parse_kappas(value: str) -> list[float]:
    """Parse a nonempty, comma-separated grid of nonnegative LCB weights."""

    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "kappas must be comma-separated numbers"
        ) from error
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "kappas must contain finite, nonnegative values"
        )
    return values


def canonical_validation_bounds(total_samples: int) -> tuple[int, int]:
    """Return the exact 80/20 positional split used by training."""

    if total_samples < 2:
        raise ValueError("canonical stream requires at least two samples")
    split = int(0.8 * total_samples)
    if split == 0 or split == total_samples:
        raise ValueError("canonical stream does not contain a validation split")
    return split, total_samples


def calibration_bins(
    predicted_sigma_db: np.ndarray,
    prediction_error_db: np.ndarray,
    bins: int,
) -> list[dict[str, float | int]]:
    """Return equal-count bins ordered by predicted standard deviation.

    Empirical RMSE is computed around the predicted mean, i.e. from
    ``realized_residual - residual_mu``.  The signed gap follows the Liu9
    caption: empirical RMSE minus mean predicted standard deviation.
    """

    sigma = np.asarray(predicted_sigma_db, dtype=float).reshape(-1)
    error = np.asarray(prediction_error_db, dtype=float).reshape(-1)
    if sigma.shape != error.shape or sigma.size == 0:
        raise ValueError("sigma and error must be nonempty arrays of equal size")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(error)):
        raise ValueError("calibration inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("predicted standard deviations must be positive")

    order = np.argsort(sigma, kind="stable")
    groups = np.array_split(order, min(bins, sigma.size))
    rows: list[dict[str, float | int]] = []
    for index, group in enumerate(groups, start=1):
        group_sigma = sigma[group]
        group_error = error[group]
        mean_sigma = float(np.mean(group_sigma))
        empirical_rmse = float(np.sqrt(np.mean(np.square(group_error))))
        rows.append(
            {
                "bin": index,
                "count": int(group.size),
                "sigma_min_db": float(np.min(group_sigma)),
                "sigma_max_db": float(np.max(group_sigma)),
                "mean_predicted_sigma_db": mean_sigma,
                "empirical_rmse_db": empirical_rmse,
                "signed_gap_db": empirical_rmse - mean_sigma,
            }
        )
    return rows


def coverage_curve(
    realized_residual: np.ndarray,
    residual_mu: np.ndarray,
    residual_sigma: np.ndarray,
    kappas: list[float],
) -> list[dict[str, float | int]]:
    """Compute ``P[y >= mu - kappa sigma]`` and Gaussian ``Phi(kappa)``."""

    target = np.asarray(realized_residual, dtype=float).reshape(-1)
    mean = np.asarray(residual_mu, dtype=float).reshape(-1)
    sigma = np.asarray(residual_sigma, dtype=float).reshape(-1)
    if target.shape != mean.shape or target.shape != sigma.shape or target.size == 0:
        raise ValueError("target, mean, and sigma must be nonempty and equal-sized")
    if not all(
        np.all(np.isfinite(array)) for array in (target, mean, sigma)
    ):
        raise ValueError("coverage inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("predicted standard deviations must be positive")

    rows: list[dict[str, float | int]] = []
    count = int(target.size)
    for kappa in kappas:
        if not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappas must be finite and nonnegative")
        lower_bound = mean - kappa * sigma
        covered = target >= lower_bound
        empirical = float(np.mean(covered))
        reference = 0.5 * (1.0 + math.erf(kappa / math.sqrt(2.0)))
        rows.append(
            {
                "kappa": float(kappa),
                "count": count,
                "covered_count": int(np.count_nonzero(covered)),
                "violation_count": int(count - np.count_nonzero(covered)),
                "empirical_coverage": empirical,
                "gaussian_phi": reference,
                "coverage_gap": empirical - reference,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_predictions(
    model: torch.nn.Module,
    samples: list[dict[str, np.ndarray | int]],
    config: NovaNetConfig,
    device: torch.device,
) -> list[dict[str, float | int]]:
    dataset = NovaNetSequenceDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=min(config.training.batch_size, len(dataset)),
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model.eval()
    rows: list[dict[str, float | int]] = []
    sample_offset = 0
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
            outputs = model(
                batch["node_features"],
                batch["spatial_adjacency"],
                batch["valid_mask"],
                batch["current_idx"],
                batch["ttl_s"],
                batch["nominal_snr_db"],
                initial_freeze=batch.get("initial_freeze"),
            )
            mask = batch["residual_mask"] & batch["valid_mask"]
            # The deployed LCB is used only for future horizons; h=0 consumes
            # the current measured SINR directly and is not a calibration row.
            mask[:, 0, :] = False
            indices = torch.nonzero(mask, as_tuple=False).cpu().numpy()
            residual_mu = outputs["residual_mu"]
            residual_sigma = outputs["residual_sigma"]
            target = batch["residual_target"]
            nominal = batch["nominal_snr_db"]
            mu_values = residual_mu[mask].detach().cpu().numpy()
            sigma_values = residual_sigma[mask].detach().cpu().numpy()
            target_values = target[mask].detach().cpu().numpy()
            nominal_values = nominal[mask].detach().cpu().numpy()
            selected_kappa = config.planner.lcb_kappa
            for position, (batch_index, horizon_index, candidate_index) in enumerate(
                indices
            ):
                mu = float(mu_values[position])
                sigma = float(sigma_values[position])
                realized = float(target_values[position])
                nominal_db = float(nominal_values[position])
                mu_db = DB_PER_LOG_RESIDUAL * mu
                sigma_db = DB_PER_LOG_RESIDUAL * sigma
                realized_db = DB_PER_LOG_RESIDUAL * realized
                lower_db = mu_db - selected_kappa * sigma_db
                rows.append(
                    {
                        "sample": sample_offset + batch_index,
                        "horizon_step": horizon_index,
                        "candidate_index": candidate_index,
                        "nominal_sinr_db": nominal_db,
                        "residual_mu": mu,
                        "residual_sigma": sigma,
                        "realized_residual_target": realized,
                        "prediction_error": realized - mu,
                        "residual_mu_db": mu_db,
                        "predicted_sigma_db": sigma_db,
                        "realized_residual_target_db": realized_db,
                        "prediction_error_db": realized_db - mu_db,
                        "predicted_mean_sinr_db": nominal_db + mu_db,
                        "realized_sinr_db": nominal_db + realized_db,
                        "selected_kappa": selected_kappa,
                        "selected_lcb_residual_db": lower_db,
                        "covered_by_selected_lcb": int(realized_db >= lower_db),
                    }
                )
            sample_offset += int(batch["current_idx"].shape[0])
    if not rows:
        raise RuntimeError("No valid future residual labels were generated")
    return rows


def _checkpoint_metadata(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "config_fingerprint": payload.get("config_fingerprint"),
        "epoch": payload.get("epoch"),
        "validation": payload.get("validation"),
        "tle_provenance": payload.get("tle_provenance"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute Appendix-A/Liu9 LCB calibration artifacts."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/novanet_paper.pt")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Validation sequences. Canonical mode requires the exact final "
            "20%% count (720 for the paper); independent mode may be smaller."
        ),
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of equal-count sigma bins."
    )
    parser.add_argument(
        "--kappas",
        type=parse_kappas,
        default=parse_kappas("0,0.5,1,1.5,2,2.5,3"),
        help="Comma-separated nonnegative LCB coefficients.",
    )
    parser.add_argument("--output-dir", default="results/lcb/calibration")
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help=(
            "Seed used only by --split-source independent-diagnostic; "
            "default is canonical seed + 1."
        ),
    )
    parser.add_argument(
        "--split-source",
        choices=("canonical-validation", "independent-diagnostic"),
        default="canonical-validation",
        help=(
            "Use the canonical training stream's final 20%% (default), or an "
            "explicitly non-paper independent stream for diagnostics."
        ),
    )
    parser.add_argument(
        "--measurement-std-db",
        type=float,
        default=None,
        help="Validation measurement noise; default is the canonical nominal value.",
    )
    parser.add_argument("--staleness-steps", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; outputs cannot be labeled paper-reproducible.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint {checkpoint_path}")
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    if args.bins < 1:
        raise ValueError("bins must be positive")
    if args.staleness_steps < 0:
        raise ValueError("staleness_steps cannot be negative")

    config = load_config(config_path)
    if config.channel.exogenous_interference_power_w != 0.0:
        raise ValueError(
            "The Appendix-A/Liu9 zero-interference protocol requires "
            "exogenous_interference_power_w=0"
        )
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tle_provenance = validate_tle_epoch(
        config,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    # NovaNetPolicy performs strict fingerprint, TLE-file, selection, and
    # selected-satellite-subset checks before exposing the model.
    policy = NovaNetPolicy(
        config,
        checkpoint_path,
        device=device,
        require_paper_eligible=not args.allow_stale_tle,
    )
    checkpoint_metadata = _checkpoint_metadata(checkpoint_path, device)

    measurement_std_db = (
        config.channel.nominal_measurement_std_db
        if args.measurement_std_db is None
        else args.measurement_std_db
    )
    if not math.isfinite(measurement_std_db) or measurement_std_db < 0.0:
        raise ValueError("measurement_std_db must be finite and nonnegative")
    canonical_start, canonical_stop = canonical_validation_bounds(
        config.training.num_samples
    )
    canonical_count = canonical_stop - canonical_start
    if args.split_source == "canonical-validation":
        if args.validation_seed is not None:
            raise ValueError(
                "--validation-seed is not used by canonical-validation; "
                "remove it or select independent-diagnostic"
            )
        if args.samples is not None and args.samples != canonical_count:
            raise ValueError(
                "canonical-validation must use the complete final 20% "
                f"partition ({canonical_count} sequences)"
            )
        if (
            measurement_std_db != config.channel.nominal_measurement_std_db
            or args.staleness_steps != 0
        ):
            raise ValueError(
                "canonical-validation reproduces the nominal training split; "
                "use independent-diagnostic for shifted noise or staleness"
            )
        canonical_stream = generate_sequence_samples(
            config,
            GenerationOptions(
                num_samples=config.training.num_samples,
                measurement_noise_std_db=measurement_std_db,
                staleness_steps=0,
                allow_stale_tle=args.allow_stale_tle,
            ),
        )
        samples = canonical_stream[canonical_start:canonical_stop]
        sample_count = len(samples)
        validation_seed = config.experiment.seed
        generation_config_fingerprint = config.fingerprint
        split_kind = "canonical_stream_final_20_percent"
    else:
        validation_seed = (
            config.experiment.seed + 1
            if args.validation_seed is None
            else args.validation_seed
        )
        if validation_seed < 0:
            raise ValueError("validation_seed must be nonnegative")
        if validation_seed == config.experiment.seed:
            raise ValueError(
                "independent validation_seed must differ from the canonical seed"
            )
        sample_count = canonical_count if args.samples is None else args.samples
        validation_config = replace(
            config,
            experiment=replace(config.experiment, seed=validation_seed),
        )
        samples = generate_sequence_samples(
            validation_config,
            GenerationOptions(
                num_samples=sample_count,
                measurement_noise_std_db=measurement_std_db,
                staleness_steps=args.staleness_steps,
                allow_stale_tle=args.allow_stale_tle,
            ),
        )
        generation_config_fingerprint = validation_config.fingerprint
        split_kind = "independent_seed_diagnostic"
    raw_rows = _collect_predictions(policy.model, samples, config, device)

    predicted_sigma_db = np.asarray(
        [row["predicted_sigma_db"] for row in raw_rows], dtype=float
    )
    prediction_error_db = np.asarray(
        [row["prediction_error_db"] for row in raw_rows], dtype=float
    )
    targets = np.asarray(
        [row["realized_residual_target"] for row in raw_rows], dtype=float
    )
    means = np.asarray([row["residual_mu"] for row in raw_rows], dtype=float)
    sigmas = np.asarray([row["residual_sigma"] for row in raw_rows], dtype=float)
    binned_rows = calibration_bins(
        predicted_sigma_db, prediction_error_db, args.bins
    )
    coverage_rows = coverage_curve(targets, means, sigmas, args.kappas)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.csv"
    binned_path = output_dir / "binned.csv"
    coverage_path = output_dir / "coverage.csv"
    protocol_path = output_dir / "protocol.json"
    write_rows(raw_path, raw_rows)
    write_rows(binned_path, binned_rows)
    write_rows(coverage_path, coverage_rows)

    protocol = {
        "runner": "experiments.lcb_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": bool(
            args.allow_stale_tle
            or args.split_source == "independent-diagnostic"
        ),
        "config_path": str(config_path.resolve()),
        "canonical_config_fingerprint": config.fingerprint,
        "canonical_config": asdict(config),
        "checkpoint": checkpoint_metadata,
        "runtime_tle_provenance": tle_provenance,
        "validation_split": {
            "kind": split_kind,
            "split_source": args.split_source,
            "canonical_training_seed": config.experiment.seed,
            "validation_seed": validation_seed,
            "generation_config_fingerprint": generation_config_fingerprint,
            "canonical_stream_sequences": config.training.num_samples,
            "canonical_partition_start": (
                canonical_start
                if args.split_source == "canonical-validation"
                else None
            ),
            "canonical_partition_stop": (
                canonical_stop
                if args.split_source == "canonical-validation"
                else None
            ),
            "sequences": sample_count,
            "future_valid_rows": len(raw_rows),
            "included_horizon_steps": list(
                range(1, config.planner.horizon_steps)
            ),
            "measurement_std_db": measurement_std_db,
            "staleness_steps": args.staleness_steps,
            "exogenous_interference_power_w": (
                config.channel.exogenous_interference_power_w
            ),
        },
        "calibration": {
            "residual_unit": "natural-log multiplicative SINR residual",
            "plotted_unit": "dB",
            "db_per_log_residual": DB_PER_LOG_RESIDUAL,
            "binning": "equal_count_sorted_by_predicted_sigma_db",
            "requested_bins": args.bins,
            "realized_bins": len(binned_rows),
            "signed_gap": "empirical_rmse_db - mean_predicted_sigma_db",
            "coverage_event": "realized_residual >= residual_mu - kappa * residual_sigma",
            "kappas": args.kappas,
            "selected_kappa": config.planner.lcb_kappa,
            "gaussian_reference": "Phi(kappa)",
        },
        "outputs": {
            "raw_csv": str(raw_path.resolve()),
            "binned_csv": str(binned_path.resolve()),
            "coverage_csv": str(coverage_path.resolve()),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(
        f"wrote {raw_path}, {binned_path}, {coverage_path}, and {protocol_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
