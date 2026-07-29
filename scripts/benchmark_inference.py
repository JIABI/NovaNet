#!/usr/bin/env python3
"""Benchmark the neural encoder and Soft-DP with a reproducible protocol."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from novanet.config import load_config
from novanet.model import NovaNet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--candidate-caps", default="8,16,32")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="results/benchmark/inference.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    base = load_config(args.config)
    rows = []
    for candidates in [
        int(value) for value in args.candidate_caps.split(",")
    ]:
        cfg = replace(
            base,
            experiment=replace(
                base.experiment,
                candidate_cap=candidates,
                num_satellites=max(
                    candidates,
                    base.experiment.num_satellites,
                ),
            ),
        )
        model = NovaNet(cfg).eval()
        horizon = cfg.planner.horizon_steps
        features = cfg.model.node_feature_dim
        node = torch.randn(1, horizon, candidates, features)
        adjacency = torch.ones(1, horizon, candidates, candidates)
        valid = torch.ones(1, horizon, candidates, dtype=torch.bool)
        incumbent = torch.zeros(1, dtype=torch.long)
        angular = torch.rand(1, horizon, candidates)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(node, adjacency, valid, incumbent, angular)
            samples_ms = []
            for _ in range(args.repetitions):
                start = time.perf_counter_ns()
                model(node, adjacency, valid, incumbent, angular)
                samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        ordered = sorted(samples_ms)
        rows.append(
            {
                "candidate_cap": candidates,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "mean_ms": statistics.fmean(samples_ms),
                "median_ms": statistics.median(samples_ms),
                "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
            }
        )

    payload = {
        "scope": "neural encoder, learned adjacency, heads, and Soft-DP; "
        "ephemeris/feature construction excluded",
        "threads": args.threads,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_fingerprint": base.fingerprint,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
