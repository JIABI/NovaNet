#!/usr/bin/env python3
"""Factorial density/candidate-cap convergence experiment.

Unlike the previous top-8-only paragraph, this records an epoch-by-epoch
validation curve for every requested constellation density and candidate cap.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
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
)
from novanet.model import NovaNet
from train_oaest import fit_energy_statistics, run_epoch, set_seed

from experiments.common import write_rows


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


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
                    converged_epoch if converged_epoch is not None else ""
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
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; do not report these results as paper runs.",
    )
    parser.add_argument("--output", default="results/density/convergence.csv")
    args = parser.parse_args()

    base = load_config(args.config)
    rows: list[dict] = []
    for density in parse_ints(args.densities):
        for cap in parse_ints(args.candidate_caps):
            if cap > density:
                continue
            for seed in parse_ints(args.seeds):
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
                samples = generate_sequence_samples(
                    cfg,
                    GenerationOptions(
                        num_samples=args.samples,
                        allow_stale_tle=args.allow_stale_tle,
                    ),
                )
                split = int(0.8 * len(samples))
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
                fit_energy_statistics(model, train_loader)
                model.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                )
                dual = cfg.training.handover_weight_init
                for epoch in range(1, args.epochs + 1):
                    train = run_epoch(
                        model,
                        train_loader,
                        device,
                        dual,
                        optimizer,
                        cfg.training.gradient_clip,
                    )
                    validation = run_epoch(
                        model,
                        validation_loader,
                        device,
                        dual,
                        None,
                        cfg.training.gradient_clip,
                    )
                    dual = min(
                        cfg.training.handover_weight_max,
                        max(
                            0.0,
                            dual
                            + cfg.training.dual_step
                            * (
                                train["handover"]
                                - cfg.training.target_switch_rate
                            ),
                        ),
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
                            "expected_switch_probability": validation["handover"],
                        }
                    )
                    print(
                        f"N={density:3d} K={cap:2d} seed={seed} "
                        f"epoch={epoch:03d} val={validation['total']:.5f}",
                        flush=True,
                    )
    write_rows(args.output, rows)

    frame = pd.DataFrame(rows)
    summary_path = Path(args.output).with_name(
        f"{Path(args.output).stem}_summary.csv"
    )
    write_rows(summary_path, convergence_summary(frame))
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
    print(f"wrote {args.output}, {summary_path}, and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
