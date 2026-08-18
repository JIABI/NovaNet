"""Deterministic TLE/SGP4 ephemeris with ECEF position and velocity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday


EARTH_ROTATION_RAD_S = 7.2921150e-5
MIN_LEO_RADIUS_KM = 6_350.0
MAX_LEO_RADIUS_KM = 9_000.0
MAX_LEO_SPEED_KM_S = 12.0


@dataclass(frozen=True)
class Ephemeris:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    names: tuple[str, ...]
    start_utc: datetime
    step_s: float

    @property
    def num_steps(self) -> int:
        return int(self.position_m.shape[0])

    @property
    def num_satellites(self) -> int:
        return int(self.position_m.shape[1])

    def time_s(self, index: int) -> float:
        return float(index) * self.step_s

    def state_at_time(
        self,
        satellite_id: int,
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate a propagated state on the stored time grid."""

        if not 0 <= satellite_id < self.num_satellites:
            raise IndexError("satellite_id outside ephemeris")
        coordinate = float(time_s) / self.step_s
        if coordinate < 0.0 or coordinate > self.num_steps - 1:
            nan = np.full(3, np.nan, dtype=float)
            return nan, nan.copy()
        left = int(np.floor(coordinate))
        right = min(left + 1, self.num_steps - 1)
        fraction = coordinate - left
        left_position = self.position_m[left, satellite_id]
        left_velocity = self.velocity_m_s[left, satellite_id]
        right_position = self.position_m[right, satellite_id]
        right_velocity = self.velocity_m_s[right, satellite_id]
        if not (
            np.all(np.isfinite(left_position))
            and np.all(np.isfinite(left_velocity))
            and np.all(np.isfinite(right_position))
            and np.all(np.isfinite(right_velocity))
        ):
            nan = np.full(3, np.nan, dtype=float)
            return nan, nan.copy()
        position = (1.0 - fraction) * left_position + fraction * right_position
        velocity = (1.0 - fraction) * left_velocity + fraction * right_velocity
        return position, velocity


def read_tle(path: str | Path) -> list[tuple[str, str, str]]:
    with Path(path).open(encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]
    records: list[tuple[str, str, str]] = []
    cursor = 0
    while cursor < len(lines):
        if (
            not lines[cursor].startswith("1 ")
            and cursor + 2 < len(lines)
            and lines[cursor + 1].startswith("1 ")
            and lines[cursor + 2].startswith("2 ")
        ):
            name = lines[cursor].removeprefix("0 ").strip()
            records.append((name, lines[cursor + 1], lines[cursor + 2]))
            cursor += 3
        elif (
            lines[cursor].startswith("1 ")
            and cursor + 1 < len(lines)
            and lines[cursor + 1].startswith("2 ")
        ):
            records.append(
                (f"SAT_{len(records)}", lines[cursor], lines[cursor + 1])
            )
            cursor += 2
        else:
            cursor += 1
    if not records:
        raise ValueError(f"No valid TLE records found in {path}")
    return records


def orbit_balanced_records(
    records: list[tuple[str, str, str]],
    limit: int,
) -> list[tuple[str, str, str]]:
    """Select a nested, shell-stratified RAAN/mean-anomaly TLE subset."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if limit > len(records):
        raise ValueError(
            f"Requested {limit} satellites from only {len(records)} TLE records"
        )
    shell_keys = [
        (
            round(float(line2[8:16])),
            round(float(line2[52:63]), 1),
        )
        for _name, _line1, line2 in records
    ]
    shell_counts = Counter(shell_keys)
    shell_records = {
        shell: [
            record
            for record, key in zip(records, shell_keys)
            if key == shell
        ]
        for shell in sorted(shell_counts)
    }

    def farthest_order(
        members: list[tuple[str, str, str]],
        count: int,
    ) -> list[tuple[str, str, str]]:
        if count == 0:
            return []
        angles = np.deg2rad(
            np.asarray(
                [
                    [float(line2[17:25]), float(line2[43:51])]
                    for _name, _line1, line2 in members
                ],
                dtype=float,
            )
        )
        selected = [0]
        available = np.ones(len(members), dtype=bool)
        available[0] = False
        minimum_distance = np.full(len(members), np.inf, dtype=float)
        while len(selected) < min(count, len(members)):
            delta = np.abs(angles - angles[selected[-1]])
            delta = np.minimum(delta, 2.0 * np.pi - delta)
            minimum_distance = np.minimum(
                minimum_distance,
                np.square(delta).sum(axis=1),
            )
            minimum_distance[~available] = -np.inf
            next_index = int(np.argmax(minimum_distance))
            selected.append(next_index)
            available[next_index] = False
        return [members[index] for index in selected]

    shell_orders = {
        shell: farthest_order(members, min(limit, len(members)))
        for shell, members in shell_records.items()
    }
    used = Counter()
    selected_records: list[tuple[str, str, str]] = []
    total = len(records)
    while len(selected_records) < limit:
        next_position = len(selected_records) + 1
        available_shells = [
            shell
            for shell in sorted(shell_orders)
            if used[shell] < len(shell_orders[shell])
        ]
        shell = max(
            available_shells,
            key=lambda key: (
                shell_counts[key] * next_position / total - used[key],
                -key[0],
                -key[1],
            ),
        )
        selected_records.append(shell_orders[shell][used[shell]])
        used[shell] += 1
    return selected_records


def _gmst_rad(julian_date: float) -> float:
    centuries = (julian_date - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * centuries
        + 0.093104 * centuries**2
        - 6.2e-6 * centuries**3
    )
    return float((seconds * np.pi / 180.0 / 240.0) % (2.0 * np.pi))


def _teme_state_to_ecef(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    julian_date: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = _gmst_rad(julian_date)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    position_ecef_km = rotation @ position_km
    velocity_rotating_km_s = rotation @ velocity_km_s
    omega = np.asarray([0.0, 0.0, EARTH_ROTATION_RAD_S])
    velocity_ecef_km_s = (
        velocity_rotating_km_s - np.cross(omega, position_ecef_km)
    )
    return position_ecef_km * 1000.0, velocity_ecef_km_s * 1000.0


def build_ephemeris(
    tle_path: str | Path,
    start_utc: datetime,
    duration_s: int,
    step_s: int,
    limit_satellites: int | None = None,
    selection: str = "shell_stratified_orbit_balanced_nested",
) -> Ephemeris:
    records = read_tle(tle_path)
    if limit_satellites is not None:
        if selection != "shell_stratified_orbit_balanced_nested":
            raise ValueError(f"Unsupported TLE selection strategy: {selection}")
        records = orbit_balanced_records(records, limit_satellites)
    return build_ephemeris_from_records(
        records,
        start_utc=start_utc,
        duration_s=duration_s,
        step_s=step_s,
    )


def build_ephemeris_from_records(
    records: list[tuple[str, str, str]],
    *,
    start_utc: datetime,
    duration_s: int,
    step_s: int,
) -> Ephemeris:
    """Propagate an explicitly ordered set of TLE records."""

    if not records:
        raise ValueError("At least one TLE record is required")
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    satellites = [
        (name, Satrec.twoline2rv(line1, line2))
        for name, line1, line2 in records
    ]
    num_steps = int(duration_s // step_s) + 1
    position = np.full((num_steps, len(satellites), 3), np.nan, dtype=np.float64)
    velocity = np.full_like(position, np.nan)

    for time_index in range(num_steps):
        timestamp = start_utc + timedelta(seconds=time_index * step_s)
        jd, fraction = jday(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second + timestamp.microsecond * 1e-6,
        )
        full_jd = jd + fraction
        for sat_index, (_name, satellite) in enumerate(satellites):
            error, teme_position, teme_velocity = satellite.sgp4(jd, fraction)
            if error != 0:
                continue
            teme_position_array = np.asarray(teme_position, dtype=float)
            teme_velocity_array = np.asarray(teme_velocity, dtype=float)
            orbital_radius_km = float(np.linalg.norm(teme_position_array))
            orbital_speed_km_s = float(np.linalg.norm(teme_velocity_array))
            # SGP4 can return error=0 while an extremely stale/decayed TLE
            # extrapolates to a nonphysical state.  Do not let those states
            # enter visibility, feature normalization, or link-budget code.
            if not (
                MIN_LEO_RADIUS_KM <= orbital_radius_km <= MAX_LEO_RADIUS_KM
                and 0.0 < orbital_speed_km_s <= MAX_LEO_SPEED_KM_S
            ):
                continue
            ecef_position, ecef_velocity = _teme_state_to_ecef(
                teme_position_array,
                teme_velocity_array,
                full_jd,
            )
            position[time_index, sat_index] = ecef_position
            velocity[time_index, sat_index] = ecef_velocity
    return Ephemeris(
        position_m=position,
        velocity_m_s=velocity,
        names=tuple(name for name, _ in satellites),
        start_utc=start_utc,
        step_s=float(step_s),
    )
