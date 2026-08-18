"""WGS-84 UE motion and user-centric satellite geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
MEAN_EARTH_RADIUS_M = 6_371_008.8


def geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0
) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    sin_latitude = np.sin(latitude)
    prime_vertical = WGS84_A_M / np.sqrt(
        1.0 - WGS84_E2 * sin_latitude**2
    )
    return np.asarray(
        [
            (prime_vertical + altitude_m)
            * np.cos(latitude)
            * np.cos(longitude),
            (prime_vertical + altitude_m)
            * np.cos(latitude)
            * np.sin(longitude),
            (
                prime_vertical * (1.0 - WGS84_E2)
                + altitude_m
            )
            * sin_latitude,
        ],
        dtype=np.float64,
    )


def ecef_local_zenith(position_m: np.ndarray) -> np.ndarray:
    """Return the WGS-84 geodetic surface normal at an ECEF position."""

    x, y, z = np.asarray(position_m, dtype=float)
    horizontal = float(np.hypot(x, y))
    if not np.isfinite(horizontal + abs(z)) or horizontal + abs(z) <= 0.0:
        raise ValueError("Invalid ECEF position for local zenith")
    longitude = float(np.arctan2(y, x))
    second_eccentricity_sq = (
        (WGS84_A_M**2 - WGS84_B_M**2) / WGS84_B_M**2
    )
    theta = float(
        np.arctan2(z * WGS84_A_M, horizontal * WGS84_B_M)
    )
    latitude = float(
        np.arctan2(
            z
            + second_eccentricity_sq
            * WGS84_B_M
            * np.sin(theta) ** 3,
            horizontal
            - WGS84_E2
            * WGS84_A_M
            * np.cos(theta) ** 3,
        )
    )
    return np.asarray(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class UETrajectory:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    speed_m_s: float = 0.0
    heading_deg: float = 0.0

    def geodetic_at(self, elapsed_s: float) -> tuple[float, float, float]:
        if self.speed_m_s == 0.0 or elapsed_s == 0.0:
            return self.latitude_deg, self.longitude_deg, self.altitude_m
        angular_distance = self.speed_m_s * elapsed_s / MEAN_EARTH_RADIUS_M
        heading = np.deg2rad(self.heading_deg)
        latitude_0 = np.deg2rad(self.latitude_deg)
        longitude_0 = np.deg2rad(self.longitude_deg)
        latitude = np.arcsin(
            np.sin(latitude_0) * np.cos(angular_distance)
            + np.cos(latitude_0)
            * np.sin(angular_distance)
            * np.cos(heading)
        )
        longitude = longitude_0 + np.arctan2(
            np.sin(heading)
            * np.sin(angular_distance)
            * np.cos(latitude_0),
            np.cos(angular_distance)
            - np.sin(latitude_0) * np.sin(latitude),
        )
        longitude = (longitude + np.pi) % (2.0 * np.pi) - np.pi
        return np.rad2deg(latitude), np.rad2deg(longitude), self.altitude_m

    def state_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        latitude, longitude, altitude = self.geodetic_at(elapsed_s)
        position = geodetic_to_ecef(latitude, longitude, altitude)
        if self.speed_m_s == 0.0:
            return position, np.zeros(3, dtype=np.float64)
        delta_s = 0.1
        before = geodetic_to_ecef(*self.geodetic_at(elapsed_s - delta_s))
        after = geodetic_to_ecef(*self.geodetic_at(elapsed_s + delta_s))
        return position, (after - before) / (2.0 * delta_s)


@dataclass(frozen=True)
class GeometryState:
    elevation_deg: float
    range_m: float
    range_rate_m_s: float
    elevation_rate_deg_s: float
    angular_speed_deg_s: float
    radial_velocity_m_s: float
    los_unit: np.ndarray


def geometry_state(
    ue_position_m: np.ndarray,
    ue_velocity_m_s: np.ndarray,
    sat_position_m: np.ndarray,
    sat_velocity_m_s: np.ndarray,
) -> GeometryState:
    los = sat_position_m - ue_position_m
    distance = float(np.linalg.norm(los))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("Invalid UE-satellite range")
    los_unit = los / distance
    up = ecef_local_zenith(ue_position_m)
    sine_elevation = float(np.clip(np.dot(los_unit, up), -1.0, 1.0))
    elevation_rad = np.arcsin(sine_elevation)

    relative_velocity = sat_velocity_m_s - ue_velocity_m_s
    radial_velocity = float(np.dot(relative_velocity, los_unit))
    transverse_velocity = relative_velocity - radial_velocity * los_unit
    angular_speed_rad_s = float(np.linalg.norm(transverse_velocity) / distance)

    if np.linalg.norm(ue_velocity_m_s) <= 0.0:
        up_rate = np.zeros(3, dtype=float)
    else:
        derivative_step_s = 0.05
        up_before = ecef_local_zenith(
            ue_position_m - derivative_step_s * ue_velocity_m_s
        )
        up_after = ecef_local_zenith(
            ue_position_m + derivative_step_s * ue_velocity_m_s
        )
        up_rate = (up_after - up_before) / (2.0 * derivative_step_s)
    los_rate = transverse_velocity / distance
    sine_rate = float(np.dot(los_rate, up) + np.dot(los_unit, up_rate))
    cosine_elevation = max(float(np.cos(elevation_rad)), 1e-8)
    elevation_rate_rad_s = sine_rate / cosine_elevation
    return GeometryState(
        elevation_deg=float(np.rad2deg(elevation_rad)),
        range_m=distance,
        range_rate_m_s=radial_velocity,
        elevation_rate_deg_s=float(np.rad2deg(elevation_rate_rad_s)),
        angular_speed_deg_s=float(np.rad2deg(angular_speed_rad_s)),
        radial_velocity_m_s=radial_velocity,
        los_unit=los_unit,
    )


def time_to_leave_seconds(
    elevations_deg: np.ndarray,
    start_index: int,
    step_s: float,
    elevation_mask_deg: float,
) -> float:
    """Forward-search the next visibility-boundary crossing.

    This handles both the rising and setting branches and remains finite near
    closest approach, unlike ``(psi_b-psi)/abs(dot_psi)``.
    """

    values = np.asarray(elevations_deg, dtype=float)
    if start_index < 0 or start_index >= len(values):
        raise IndexError("start_index outside elevation trace")
    if not np.isfinite(values[start_index]) or values[start_index] < elevation_mask_deg:
        return 0.0
    for index in range(start_index + 1, len(values)):
        current = values[index]
        if not np.isfinite(current):
            return float((index - start_index - 1) * step_s)
        if current < elevation_mask_deg:
            previous = values[index - 1]
            denominator = previous - current
            fraction = (
                0.0
                if abs(denominator) < 1e-12
                else (previous - elevation_mask_deg) / denominator
            )
            fraction = float(np.clip(fraction, 0.0, 1.0))
            return ((index - 1 - start_index) + fraction) * step_s
    return float((len(values) - 1 - start_index) * step_s)


def sky_dome_adjacency(
    los_units: np.ndarray,
    valid_mask: np.ndarray,
    neighbors: int,
    temperature: float,
) -> np.ndarray:
    los_units = np.asarray(los_units, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    candidates = len(valid)
    adjacency = np.zeros((candidates, candidates), dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    for source in valid_indices:
        cosine = np.clip(los_units @ los_units[source], -1.0, 1.0)
        angular = np.arccos(cosine)
        order = [
            int(index)
            for index in np.argsort(angular)
            if valid[index] and int(index) != int(source)
        ][: min(neighbors, max(len(valid_indices) - 1, 0))]
        for target in order:
            adjacency[source, target] = np.exp(
                -float(angular[target]) / max(temperature, 1e-6)
            )
        total = float(adjacency[source].sum())
        if total > 0.0:
            adjacency[source] /= total
    return adjacency
