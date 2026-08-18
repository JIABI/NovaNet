#!/usr/bin/env python3
"""Fail fast when code/config/reference artifacts drift from the manuscript."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from novanet.config import load_config
from novanet.dataset import StaleTLEError, validate_tle_epoch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check manuscript parameters and orbital-data provenance."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--max-tle-age-days", type=float, default=14.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.schema_version == 5
    assert cfg.experiment.seed == 2025
    assert cfg.experiment.minimum_elevation_deg == 10.0
    assert cfg.experiment.candidate_cap == 8
    assert cfg.experiment.duration_s == 2400
    assert cfg.experiment.decision_interval_s == 30
    assert cfg.experiment.geometry_subsample_s == 5
    assert cfg.handover.ttt_s == 0.1
    assert cfg.handover.execution_s == 0.15
    assert cfg.handover.freeze_steps == 1
    assert cfg.traffic.packet_size_bytes == 1500
    assert cfg.traffic.arrival_process == "poisson"
    assert cfg.traffic.arrival_rate_packets_s == 500.0
    assert cfg.traffic.queue_discipline == "fifo"
    assert cfg.traffic.queue_capacity_packets == 4096
    assert cfg.traffic.fixed_network_delay_ms == 33.5
    assert cfg.traffic.protocol_processing_ms == 1.0
    assert cfg.channel.bandwidth_options_hz == (20e6, 100e6)
    assert cfg.channel.nominal_measurement_std_db == 0.0
    assert cfg.channel.doppler_tracking_efficiency == 0.995
    assert cfg.channel.doppler_estimation_std_hz == 25.0
    assert cfg.channel.coherent_integration_s == 1e-4
    assert cfg.planner.horizon_steps == 6
    assert cfg.model.node_feature_dim == 6
    assert cfg.model.transition_feature_dim == 5
    assert cfg.planner.rate_reference_mbps == 50.0
    assert cfg.planner.ttl_reference_s == 600.0
    assert (
        cfg.planner.alpha,
        cfg.planner.beta,
        cfg.planner.c0,
        cfg.planner.c1,
        cfg.planner.c2,
    ) == (1.0, 0.35, 1.0, 0.5, 1.5)
    assert cfg.multi_ue.satellite_capacity_mbps == 1200.0
    assert cfg.multi_ue.max_users_per_satellite == 32
    assert cfg.multi_ue.scheduler == "proportional_fair"
    assert cfg.multi_ue.association_update == "synchronous_two_phase"

    reference = REPO_ROOT / "results/reference/liu7.csv"
    with reference.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = {
        row["method"]: float(row["reported_transmission_only_ms"])
        for row in rows
    }
    assert values["Max-Elevation"] == 3.982
    assert values["NovaNet (Ours)"] == 4.521
    for row in rows:
        assert abs(
            float(row["queue_and_serialization_ms"])
            + float(row["shared_protocol_processing_ms"])
            - float(row["reported_transmission_only_ms"])
        ) < 1e-12

    print(f"configuration fingerprint: {cfg.fingerprint}")
    print("paper parameters and Liu7 reference: PASS")
    try:
        report = validate_tle_epoch(cfg, maximum_age_days=args.max_tle_age_days)
    except StaleTLEError as error:
        print(f"TLE provenance: BLOCKED\n{error}")
        return 2
    print(f"TLE provenance: PASS {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
