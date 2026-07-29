#!/usr/bin/env python3
"""Plot the best deterministic link SNR using the canonical paper model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from novanet.channel import LinkBudget
from novanet.config import load_config
from novanet.ephemeris import build_ephemeris
from novanet.geometry import UETrajectory, geometry_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--tle", default=None)
    parser.add_argument("--ue-lat", type=float, default=51.5)
    parser.add_argument("--ue-lon", type=float, default=0.0)
    parser.add_argument("--output", default="results/diagnostics/best_snr.pdf")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tle is not None:
        cfg = replace(
            cfg,
            experiment=replace(cfg.experiment, tle_path=args.tle),
        )
    ephemeris = build_ephemeris(
        cfg.resolve_tle_path(),
        cfg.start_utc,
        cfg.experiment.duration_s,
        cfg.experiment.geometry_subsample_s,
        cfg.experiment.num_satellites,
    )
    trajectory = UETrajectory(args.ue_lat, args.ue_lon)
    budget = LinkBudget(cfg.channel, cfg.experiment.seed)
    best_snr = np.full(ephemeris.num_steps, np.nan, dtype=float)
    for time_index in range(ephemeris.num_steps):
        ue_position, ue_velocity = trajectory.state_at(
            ephemeris.time_s(time_index)
        )
        values = []
        for satellite in range(ephemeris.num_satellites):
            state = geometry_state(
                ue_position,
                ue_velocity,
                ephemeris.position_m[time_index, satellite],
                ephemeris.velocity_m_s[time_index, satellite],
            )
            if state.elevation_deg >= cfg.experiment.minimum_elevation_deg:
                values.append(budget.evaluate(state, stochastic=False).snr_db)
        if values:
            best_snr[time_index] = max(values)

    figure, axis = plt.subplots(figsize=(7.16, 3.2), constrained_layout=True)
    axis.plot(
        np.arange(ephemeris.num_steps) * ephemeris.step_s,
        best_snr,
        linewidth=1.2,
    )
    axis.axhline(
        cfg.channel.outage_threshold_db,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Outage threshold",
    )
    axis.set(xlabel="Time (s)", ylabel="Best visible-link SNR (dB)")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
