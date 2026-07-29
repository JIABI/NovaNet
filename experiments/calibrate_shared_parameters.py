#!/usr/bin/env python3
"""Diagnose which *shared* parameters can be inferred from published results.

This script never fits method-specific parameters. It verifies the common
1 ms protocol component in Fig. 7 and tests whether a single M/M/1 arrival
rate can explain all five queueing/serialization means from the published
method rates. A large residual is reported as non-identifiability rather than
being hidden with per-method tuning.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


PAPER_RATE_MBPS = {
    "Max-Elevation": 68.39,
    "Max-ServeTime": 65.70,
    "GNN-only": 58.70,
    "DQN+GNN": 60.00,
    "NovaNet (Ours)": 62.20,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-data", default="results/reference/liu7.csv")
    parser.add_argument("--packet-bytes", type=int, default=1500)
    args = parser.parse_args()

    with Path(args.figure_data).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    processing = np.asarray(
        [float(row["shared_protocol_processing_ms"]) for row in rows]
    )
    reported = np.asarray(
        [float(row["reported_transmission_only_ms"]) for row in rows]
    )
    queue_serial = np.asarray(
        [float(row["queue_and_serialization_ms"]) for row in rows]
    )
    if not np.allclose(processing, processing[0]) or not np.allclose(
        queue_serial + processing, reported
    ):
        raise ValueError("Fig. 7 decomposition is not shared or does not sum")

    service_rate_pps = np.asarray(
        [
            PAPER_RATE_MBPS[row["method"]] * 1e6 / (8.0 * args.packet_bytes)
            for row in rows
        ]
    )

    def residual(parameter):
        arrival_rate, common_offset_ms = parameter
        if arrival_rate >= service_rate_pps.min():
            return np.full_like(queue_serial, 1e6)
        mm1_ms = 1e3 / (service_rate_pps - arrival_rate)
        return mm1_ms + common_offset_ms - queue_serial

    fit = least_squares(
        residual,
        x0=np.asarray([0.5 * service_rate_pps.min(), 0.0]),
        bounds=(
            np.asarray([0.0, 0.0]),
            np.asarray([0.999 * service_rate_pps.min(), 20.0]),
        ),
    )
    rmse = float(np.sqrt(np.mean(np.square(fit.fun))))
    print(f"shared protocol processing = {processing[0]:.3f} ms")
    print(f"best common Poisson arrival = {fit.x[0]:.3f} packets/s")
    print(f"best common residual offset = {fit.x[1]:.3f} ms")
    print(f"cross-method delay RMSE = {rmse:.3f} ms")
    if rmse > 0.5:
        print(
            "RESULT: the five means do not identify one common M/M/1 queue. "
            "Packet traces or the exact scheduler/service process are required; "
            "do not introduce method-specific arrival rates to force a match."
        )
        return 2
    print("RESULT: a shared queue parameterization is numerically compatible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

