#!/usr/bin/env python3
"""Recompute packet latency from explicit rate and CHO-event traces.

For each ``METHOD.csv`` in ``--trace-dir``, required columns are
``time_s,rate_bps``. An optional ``METHOD.blackouts.csv`` contains
``start_s,end_s``. The packet source, buffer, FIFO rule, fixed network
component, and shared 1 ms protocol component come only from paper.yaml.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from novanet.config import load_config
from novanet.latency import latency_summary, simulate_fifo_latency

from experiments.common import artifact_sha256, write_rows, write_protocol


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"time_s", "rate_bps"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return (
        np.asarray([float(row["time_s"]) for row in rows]),
        np.asarray([float(row["rate_bps"]) for row in rows]),
    )


def read_blackouts(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"start_s", "end_s"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return [(float(row["start_s"]), float(row["end_s"])) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", default="results/latency/summary.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    directory = Path(args.trace_dir)
    trace_files = sorted(
        path
        for path in directory.glob("*.csv")
        if not path.name.endswith(".blackouts.csv")
    )
    if not trace_files:
        raise FileNotFoundError(f"No method CSV traces in {directory}")
    rows = []
    input_artifacts: list[dict] = []
    for path in trace_files:
        method = path.stem
        times, rates = read_trace(path)
        blackout_path = directory / f"{method}.blackouts.csv"
        blackouts = read_blackouts(blackout_path)
        trace = simulate_fifo_latency(
            cfg.traffic,
            cfg.experiment.duration_s,
            times,
            rates,
            handover_blackouts=blackouts,
            seed=cfg.experiment.seed,
        )
        counterfactual = simulate_fifo_latency(
            cfg.traffic,
            cfg.experiment.duration_s,
            times,
            rates,
            handover_blackouts=None,
            seed=cfg.experiment.seed,
        )
        summary = latency_summary(trace)
        summary["method"] = method
        summary["transmission_only_mean_ms"] = latency_summary(
            counterfactual
        )["transmission_only_mean_ms"]
        summary.update(
            {
                "arrival_process": cfg.traffic.arrival_process,
                "arrival_rate_packets_s": cfg.traffic.arrival_rate_packets_s,
                "packet_size_bytes": cfg.traffic.packet_size_bytes,
                "queue_discipline": cfg.traffic.queue_discipline,
                "queue_capacity_packets": cfg.traffic.queue_capacity_packets,
                "fixed_network_delay_ms": cfg.traffic.fixed_network_delay_ms,
                "protocol_processing_ms": (
                    cfg.traffic.protocol_processing_ms
                ),
            }
        )
        rows.append(summary)
        input_artifacts.append(
            {
                "method": method,
                "rate_trace": {
                    "name": path.name,
                    "sha256": artifact_sha256(path),
                    "rows": int(len(times)),
                    "sample_and_hold_through_s": float(
                        cfg.experiment.duration_s
                    ),
                },
                "blackout_trace": (
                    {
                        "name": blackout_path.name,
                        "sha256": artifact_sha256(blackout_path),
                        "intervals": int(len(blackouts)),
                    }
                    if blackout_path.is_file()
                    else None
                ),
            }
        )
    write_rows(args.output, rows)
    protocol_path = Path(args.output).with_name(
        f"{Path(args.output).stem}_protocol.json"
    )
    write_protocol(
        protocol_path,
        cfg,
        runner="experiments.latency_from_traces",
        details={
            "trace_directory": directory.name,
            "input_artifacts": input_artifacts,
            "rate_interpolation": "left-continuous sample-and-hold",
            "packet_service": (
                "each accepted packet requires packet_size_bytes*8 bits "
                "under the piecewise-constant service process"
            ),
        },
    )
    print(f"wrote {args.output} and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
