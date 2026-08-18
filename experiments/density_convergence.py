#!/usr/bin/env python3
"""Factorial density/candidate-cap convergence experiment.

Unlike the previous top-8-only paragraph, this records an epoch-by-epoch
validation curve for every requested constellation density and candidate cap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from novanet.config import load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)
from novanet.geometry import UETrajectory, ecef_local_zenith
from novanet.model import NovaNet
from novanet.policies import NovaNetPolicy
from novanet.simulation import Scenario, simulate_single_user
from train_oaest import run_epoch, set_seed, supervision_counts

from experiments.common import (
    artifact_sha256,
    build_paper_ephemeris,
    evaluation_episode_seed,
    write_protocol,
    write_rows,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def evaluation_layouts(config, count: int, seed: int) -> list[Scenario]:
    """Return paired held-out layouts shared by all caps at one seed."""

    rng = np.random.default_rng(seed)
    return [
        Scenario(
            latitude_deg=float(rng.uniform(*config.experiment.ue_latitude_deg)),
            longitude_deg=float(rng.uniform(*config.experiment.ue_longitude_deg)),
            altitude_m=config.experiment.ue_altitude_m,
        )
        for _ in range(count)
    ]


def candidate_union_statistics(
    config,
    ephemeris,
    layouts: list[Scenario],
) -> dict[str, float | int]:
    """Measure the untruncated H-step visible union over test windows.

    This is computed directly from the full selected constellation before the
    candidate cap is applied.  It therefore supports both ``K_eff`` and the
    cap-activation frequency reported in Appendix B.
    """

    stride = int(
        round(config.experiment.decision_interval_s / ephemeris.step_s)
    )
    duration_steps = int(config.experiment.duration_s / ephemeris.step_s)
    final_index = min(
        duration_steps,
        ephemeris.num_steps - stride * config.planner.horizon_steps - 1,
    )
    union_sizes: list[int] = []
    for layout in layouts:
        trajectory = UETrajectory(
            layout.latitude_deg,
            layout.longitude_deg,
            layout.altitude_m,
            speed_m_s=layout.speed_kmh / 3.6,
            heading_deg=layout.heading_deg,
        )
        for decision_index in range(0, final_index, stride):
            visible_union = np.zeros(ephemeris.num_satellites, dtype=bool)
            for horizon in range(config.planner.horizon_steps):
                index = decision_index + horizon * stride
                ue_position, _ = trajectory.state_at(ephemeris.time_s(index))
                satellite_position = ephemeris.position_m[index]
                relative = satellite_position - ue_position[None, :]
                ranges = np.linalg.norm(relative, axis=1)
                finite = np.all(np.isfinite(relative), axis=1) & (ranges > 0.0)
                elevation = np.full(ephemeris.num_satellites, -90.0)
                up = ecef_local_zenith(ue_position)
                elevation[finite] = np.rad2deg(
                    np.arcsin(
                        np.clip(
                            (relative[finite] @ up) / ranges[finite],
                            -1.0,
                            1.0,
                        )
                    )
                )
                visible_union |= (
                    finite
                    & (
                        elevation
                        >= config.experiment.minimum_elevation_deg
                    )
                )
            union_sizes.append(int(visible_union.sum()))
    sizes = np.asarray(union_sizes, dtype=float)
    cap = config.experiment.candidate_cap
    return {
        "evaluation_windows": len(union_sizes),
        "mean_visible_union": float(sizes.mean()),
        "mean_effective_candidates": float(np.minimum(sizes, cap).mean()),
        "cap_activation_percent": float(100.0 * np.mean(sizes > cap)),
    }


def evaluate_checkpoint(
    config,
    ephemeris,
    checkpoint: Path,
    layouts: list[Scenario],
    *,
    require_paper_eligible: bool,
) -> dict[str, float]:
    policy = NovaNetPolicy(
        config,
        checkpoint,
        require_paper_eligible=require_paper_eligible,
    )
    metrics = [
        simulate_single_user(
            config,
            ephemeris,
            policy,
            layout,
            seed=evaluation_episode_seed(config.experiment.seed, user),
        )
        for user, layout in enumerate(layouts)
    ]
    handovers = sum(item.handovers for item in metrics)
    failures = sum(item.handover_failures for item in metrics)
    return {
        "effective_throughput_mbps": float(
            np.mean([item.effective_throughput_mbps for item in metrics])
        ),
        "mean_rate_mbps": float(
            np.mean([item.mean_rate_mbps for item in metrics])
        ),
        "hof_percent": 100.0 * failures / max(handovers, 1),
        "outage_percent": float(
            np.mean([item.outage_percent for item in metrics])
        ),
        "handovers_per_2400s": float(
            np.mean([item.handovers for item in metrics])
        ),
    }


def summarize_cells(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    keys = ["density", "candidate_cap"]
    excluded = set(keys + ["seed"])
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    for (density, cap), group in frame.groupby(keys):
        row: dict[str, float | int] = {
            "density": int(density),
            "candidate_cap": int(cap),
            "seeds": int(group["seed"].nunique()),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(
                group[column].std(ddof=1 if len(group) > 1 else 0)
            )
        rows.append(row)
    return rows


def convergence_summary(frame: pd.DataFrame) -> list[dict]:
    """Return the paper's five-epoch, one-percent convergence statistic."""

    rows: list[dict] = []
    keys = ["density", "candidate_cap", "seed"]
    for (density, cap, seed), group in frame.groupby(keys):
        ordered = group.sort_values("epoch")
        moving = ordered["validation_loss"].rolling(
            window=5,
            min_periods=5,
        ).mean()
        valid = moving.dropna()
        best = float(valid.min()) if len(valid) else float("nan")
        tolerance = 0.01 * max(abs(best), 1e-8)
        converged_epoch: int | None = None
        valid_values = valid.to_numpy()
        valid_epochs = ordered.loc[valid.index, "epoch"].to_numpy()
        for index, epoch in enumerate(valid_epochs):
            if (valid_values[index:] <= best + tolerance).all():
                converged_epoch = int(epoch)
                break
        rows.append(
            {
                "density": int(density),
                "candidate_cap": int(cap),
                "seed": int(seed),
                "best_five_epoch_validation_loss": best,
                "converged": converged_epoch is not None,
                "convergence_epoch": (
                    converged_epoch
                    if converged_epoch is not None
                    else float("nan")
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--densities", default="60,120,240")
    parser.add_argument("--candidate-caps", default="8,16,32")
    parser.add_argument("--seeds", default="2025,2026,2027")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Training sequences per cell (paper config value if omitted).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-users", type=int, default=60)
    parser.add_argument(
        "--checkpoint-dir",
        default="results/density/checkpoints",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Write convergence curves only; not sufficient for Appendix B.",
    )
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; do not report these results as paper runs.",
    )
    parser.add_argument("--output", default="results/density/convergence.csv")
    args = parser.parse_args()

    base = load_config(args.config)
    densities = parse_ints(args.densities)
    caps = parse_ints(args.candidate_caps)
    seeds = parse_ints(args.seeds)
    if not densities or not caps or not seeds:
        raise ValueError("Densities, candidate caps, and seeds cannot be empty")
    if args.epochs < 1 or (args.samples is not None and args.samples < 2):
        raise ValueError("Training requires positive epochs and at least 2 samples")
    if not args.skip_evaluation and args.test_users < 1:
        raise ValueError("--test-users must be positive")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cell_rows: list[dict] = []
    ephemeris_cache: dict[int, object] = {}
    layout_cache: dict[tuple[int, int], list[Scenario]] = {}
    union_cache: dict[tuple[int, int, int], dict[str, float | int]] = {}
    cell_protocols: list[dict] = []
    for density in densities:
        for cap in caps:
            if cap > density:
                continue
            for seed in seeds:
                cfg = replace(
                    base,
                    experiment=replace(
                        base.experiment,
                        seed=seed,
                        num_satellites=density,
                        candidate_cap=cap,
                    ),
                )
                set_seed(seed)
                sample_count = args.samples or cfg.training.num_samples
                samples = generate_sequence_samples(
                    cfg,
                    GenerationOptions(
                        num_samples=sample_count,
                        allow_stale_tle=args.allow_stale_tle,
                    ),
                )
                split = int(0.8 * len(samples))
                train_supervision = supervision_counts(samples[:split])
                validation_supervision = supervision_counts(samples[split:])
                train_loader = DataLoader(
                    NovaNetSequenceDataset(samples[:split]),
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                validation_loader = DataLoader(
                    NovaNetSequenceDataset(samples[split:]),
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                )
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                model = NovaNet(cfg)
                model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.training.learning_rate,
                    weight_decay=0.0,
                )
                checkpoint = checkpoint_dir / (
                    f"novanet_N{density}_K{cap}_seed{seed}.pt"
                )
                provenance = validate_tle_epoch(
                    cfg,
                    maximum_age_days=(
                        float("inf") if args.allow_stale_tle else 14.0
                    ),
                )
                paper_table_eligible = bool(
                    not args.allow_stale_tle
                    and sample_count == cfg.training.num_samples
                    and args.epochs == cfg.training.epochs
                )
                if paper_table_eligible and (
                    min(train_supervision.values()) <= 0
                    or min(validation_supervision.values()) <= 0
                ):
                    raise RuntimeError(
                        "A density cell cannot be manuscript-grade without "
                        "residual and HOF labels in both data partitions"
                    )
                best_validation = float("inf")
                checkpoint_written = False
                for epoch in range(1, args.epochs + 1):
                    train = run_epoch(
                        model,
                        train_loader,
                        device,
                        optimizer,
                        cfg.training.gradient_clip,
                    )
                    validation = run_epoch(
                        model,
                        validation_loader,
                        device,
                        None,
                        cfg.training.gradient_clip,
                    )
                    rows.append(
                        {
                            "density": density,
                            "candidate_cap": cap,
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": train["total"],
                            "validation_loss": validation["total"],
                            "validation_hof_loss": validation["hof"],
                            "validation_path_loss": validation["path"],
                        }
                    )
                    if validation["total"] < best_validation:
                        best_validation = validation["total"]
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "config": asdict(cfg),
                                "config_fingerprint": cfg.fingerprint,
                                "checkpoint_kind": "novanet",
                                "training_protocol": (
                                    "novanet_h6_softdp_sequence_v2"
                                ),
                                # Best-so-far files remain diagnostic until the
                                # complete per-cell epoch loop returns.
                                "paper_table_eligible": False,
                                "training": {
                                    "samples": sample_count,
                                    "train_samples": split,
                                    "validation_samples": len(samples) - split,
                                    "epochs_requested": args.epochs,
                                    "epochs_completed": epoch,
                                    "training_complete": False,
                                    "batch_size": cfg.training.batch_size,
                                    "optimizer": "AdamW",
                                    "learning_rate": cfg.training.learning_rate,
                                    "weight_decay": 0.0,
                                    "training_seed": seed,
                                    "allow_stale_tle": bool(args.allow_stale_tle),
                                    "train_residual_labels": train_supervision[
                                        "residual_labels"
                                    ],
                                    "validation_residual_labels": (
                                        validation_supervision["residual_labels"]
                                    ),
                                    "train_hof_labels": train_supervision[
                                        "hof_labels"
                                    ],
                                    "validation_hof_labels": (
                                        validation_supervision["hof_labels"]
                                    ),
                                },
                                "epoch": epoch,
                                "validation": validation,
                                "tle_provenance": provenance,
                                "experiment_cell": {
                                    "density": density,
                                    "candidate_cap": cap,
                                    "seed": seed,
                                    "samples": sample_count,
                                },
                            },
                            checkpoint,
                        )
                        checkpoint_written = True
                    print(
                        f"N={density:3d} K={cap:2d} seed={seed} "
                        f"epoch={epoch:03d} val={validation['total']:.5f}",
                        flush=True,
                    )
                if not checkpoint_written:
                    raise RuntimeError(
                        "Density-cell training completed without a finite "
                        "validation checkpoint"
                    )
                payload = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                payload["training"]["epochs_completed"] = args.epochs
                payload["training"]["training_complete"] = True
                payload["paper_table_eligible"] = paper_table_eligible
                finalized_checkpoint = checkpoint.with_name(
                    f".{checkpoint.name}.finalizing"
                )
                torch.save(payload, finalized_checkpoint)
                finalized_checkpoint.replace(checkpoint)
                cell_protocols.append(
                    {
                        "density": density,
                        "candidate_cap": cap,
                        "training_seed": seed,
                        "config_fingerprint": cfg.fingerprint,
                        "tle_provenance": provenance,
                        "checkpoint_name": checkpoint.name,
                        "checkpoint_sha256": artifact_sha256(checkpoint),
                        "paper_table_eligible": paper_table_eligible,
                    }
                )
                if args.skip_evaluation:
                    continue
                if density not in ephemeris_cache:
                    ephemeris_cache[density] = build_paper_ephemeris(
                        cfg,
                        allow_stale_tle=args.allow_stale_tle,
                    )
                ephemeris = ephemeris_cache[density]
                layout_key = (density, seed)
                if layout_key not in layout_cache:
                    layout_cache[layout_key] = evaluation_layouts(
                        cfg,
                        args.test_users,
                        seed + 200_000,
                    )
                layouts = layout_cache[layout_key]
                union_key = (density, cap, seed)
                union_cache[union_key] = candidate_union_statistics(
                    cfg,
                    ephemeris,
                    layouts,
                )
                evaluation = evaluate_checkpoint(
                    cfg,
                    ephemeris,
                    checkpoint,
                    layouts,
                    require_paper_eligible=not args.allow_stale_tle,
                )
                cell_row: dict[str, float | int | str] = {
                    "density": density,
                    "candidate_cap": cap,
                    "seed": seed,
                    "training_samples": sample_count,
                    "training_epochs": args.epochs,
                    "best_validation_loss": best_validation,
                    "checkpoint": str(checkpoint),
                }
                cell_row.update(union_cache[union_key])
                cell_row.update(evaluation)
                cell_rows.append(cell_row)
    write_rows(args.output, rows)

    frame = pd.DataFrame(rows)
    summary_path = Path(args.output).with_name(
        f"{Path(args.output).stem}_summary.csv"
    )
    convergence_rows = convergence_summary(frame)
    write_rows(summary_path, convergence_rows)
    if not args.skip_evaluation:
        convergence_by_cell = {
            (
                int(row["density"]),
                int(row["candidate_cap"]),
                int(row["seed"]),
            ): row
            for row in convergence_rows
        }
        for row in cell_rows:
            key = (
                int(row["density"]),
                int(row["candidate_cap"]),
                int(row["seed"]),
            )
            convergence = convergence_by_cell[key]
            row["converged"] = convergence["converged"]
            row["convergence_epoch"] = convergence["convergence_epoch"]
        cell_path = Path(args.output).with_name("cell_results.csv")
        write_rows(cell_path, cell_rows)
        cell_summary_path = Path(args.output).with_name("cell_summary.csv")
        write_rows(
            cell_summary_path,
            summarize_cells(pd.DataFrame(cell_rows)),
        )
    figure, axis = plt.subplots(figsize=(7.16, 3.4), constrained_layout=True)
    for (density, cap), group in frame.groupby(["density", "candidate_cap"]):
        curve = group.groupby("epoch")["validation_loss"].agg(["mean", "std"])
        epochs = curve.index.to_numpy(dtype=float)
        mean = curve["mean"].to_numpy(dtype=float)
        std = curve["std"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(epochs, mean, label=f"N={density}, K={cap}")
        axis.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.15,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation objective")
    axis.legend(ncol=3, frameon=False, fontsize=7)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure_path = Path(args.output).with_suffix(".pdf")
    figure.savefig(figure_path)
    outputs = [str(args.output), str(summary_path), str(figure_path)]
    if not args.skip_evaluation:
        outputs.extend([str(cell_path), str(cell_summary_path)])
    protocol_path = write_protocol(
        Path(args.output).with_name("protocol.json"),
        base,
        runner="density_convergence",
        diagnostic=args.allow_stale_tle,
        details={
            "densities": densities,
            "candidate_caps": caps,
            "training_seeds": seeds,
            "samples_per_cell": args.samples or base.training.num_samples,
            "epochs": args.epochs,
            "test_users": 0 if args.skip_evaluation else args.test_users,
            "nested_tle_selection": base.experiment.tle_selection,
            "checkpoint_directory": checkpoint_dir.name,
            "cells": cell_protocols,
        },
    )
    outputs.append(str(protocol_path))
    print(f"wrote {', '.join(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
