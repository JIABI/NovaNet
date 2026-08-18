#!/usr/bin/env python3
"""Train NovaNet with the complete finite-horizon objective."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from novanet.config import load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)
from novanet.losses import compute_training_loss
from novanet.model import NovaNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def supervision_counts(samples: list[dict]) -> dict[str, int]:
    """Count labels that contribute to the two masked supervised heads."""

    return {
        "residual_labels": int(
            sum(np.count_nonzero(sample["residual_mask"]) for sample in samples)
        ),
        "hof_labels": int(
            sum(np.count_nonzero(sample["hof_mask"]) for sample in samples)
        ),
    }


def run_epoch(
    model: NovaNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    observations = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                batch["node_features"],
                batch["spatial_adjacency"],
                batch["valid_mask"],
                batch["current_idx"],
                batch["ttl_s"],
                batch["nominal_snr_db"],
                initial_freeze=batch.get("initial_freeze"),
            )
            loss, components = compute_training_loss(outputs, batch, model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        batch_size = int(batch["current_idx"].shape[0])
        observations += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / max(observations, 1) for name, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--allow-stale-tle",
        action="store_true",
        help="Diagnostic only; results are not paper-reproducible.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)
    tle_provenance = validate_tle_epoch(
        cfg,
        maximum_age_days=float("inf") if args.allow_stale_tle else 14.0,
    )
    sample_count = args.samples or cfg.training.num_samples
    samples = generate_sequence_samples(
        cfg,
        GenerationOptions(
            num_samples=sample_count,
            allow_stale_tle=args.allow_stale_tle,
        ),
    )
    split = int(0.8 * len(samples))
    if split == 0 or split == len(samples):
        raise ValueError("Training requires at least two samples")
    train_set = NovaNetSequenceDataset(samples[:split])
    validation_set = NovaNetSequenceDataset(samples[split:])
    train_supervision = supervision_counts(samples[:split])
    validation_supervision = supervision_counts(samples[split:])
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.training.num_workers,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NovaNet(cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        # The manuscript objective includes L2 explicitly in the loss.
        weight_decay=0.0,
    )

    output = Path(args.output or cfg.training.checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path = output.with_suffix(".history.csv")
    metadata_path = output.with_suffix(".metadata.json")
    epochs = args.epochs or cfg.training.epochs
    paper_table_eligible = bool(
        not args.allow_stale_tle
        and sample_count == cfg.training.num_samples
        and epochs == cfg.training.epochs
    )
    if paper_table_eligible and (
        min(train_supervision.values()) <= 0
        or min(validation_supervision.values()) <= 0
    ):
        raise RuntimeError(
            "A manuscript-grade split requires nonempty residual and HOF "
            "supervision in both training and validation partitions"
        )
    best_validation = float("inf")
    checkpoint_written = False
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            cfg.training.gradient_clip,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                None,
                cfg.training.gradient_clip,
            )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        rows.append(row)
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.5f} "
            f"validation={validation_metrics['total']:.5f}",
            flush=True,
        )
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "config_fingerprint": cfg.fingerprint,
                    "checkpoint_kind": "novanet",
                    "training_protocol": "novanet_h6_softdp_sequence_v2",
                    # A best-so-far file is not a completed training artifact.
                    # It is promoted only after the full epoch loop returns.
                    "paper_table_eligible": False,
                    "training": {
                        "samples": sample_count,
                        "train_samples": len(train_set),
                        "validation_samples": len(validation_set),
                        "epochs_requested": epochs,
                        "epochs_completed": epoch,
                        "training_complete": False,
                        "batch_size": cfg.training.batch_size,
                        "optimizer": "AdamW",
                        "learning_rate": cfg.training.learning_rate,
                        "weight_decay": 0.0,
                        "training_seed": cfg.experiment.seed,
                        "allow_stale_tle": bool(args.allow_stale_tle),
                        "train_residual_labels": train_supervision[
                            "residual_labels"
                        ],
                        "validation_residual_labels": validation_supervision[
                            "residual_labels"
                        ],
                        "train_hof_labels": train_supervision["hof_labels"],
                        "validation_hof_labels": validation_supervision[
                            "hof_labels"
                        ],
                    },
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "tle_provenance": tle_provenance,
                },
                output,
            )
            checkpoint_written = True

    if not checkpoint_written:
        raise RuntimeError(
            "Training completed without a finite validation checkpoint"
        )
    # Promote the selected best epoch only after every requested epoch has
    # completed.  Atomic replacement prevents an interrupted final write from
    # exposing a manuscript-qualified checkpoint.
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["training"]["epochs_completed"] = epochs
    payload["training"]["training_complete"] = True
    payload["paper_table_eligible"] = paper_table_eligible
    finalized_output = output.with_name(f".{output.name}.finalizing")
    torch.save(payload, finalized_output)
    finalized_output.replace(output)

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(
            {
                "config_fingerprint": cfg.fingerprint,
                "best_validation_loss": best_validation,
                "training_protocol": "novanet_h6_softdp_sequence_v2",
                "paper_table_eligible": paper_table_eligible,
                "epochs_requested": epochs,
                "epochs_completed": epochs,
                "training_complete": True,
                "samples": sample_count,
                "train_samples": len(train_set),
                "validation_samples": len(validation_set),
                "train_supervision": train_supervision,
                "validation_supervision": validation_supervision,
                "allow_stale_tle": args.allow_stale_tle,
                "tle_provenance": tle_provenance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
