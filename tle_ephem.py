"""Compatibility wrapper around the canonical position-and-velocity ephemeris."""

from __future__ import annotations

from datetime import datetime, timezone

from novanet.ephemeris import Ephemeris, build_ephemeris, read_tle


load_tle_file = read_tle


def build_ephemeris_from_tle(
    tle_path: str,
    start_utc: datetime | None = None,
    duration_s: int = 2400,
    dt_s: int = 5,
    limit_sats: int | None = None,
):
    """Return the historical tuple while using the canonical propagator."""

    start = start_utc or datetime.now(timezone.utc).replace(microsecond=0)
    ephemeris: Ephemeris = build_ephemeris(
        tle_path,
        start,
        duration_s,
        dt_s,
        limit_sats,
    )
    return (
        ephemeris.position_m,
        list(ephemeris.names),
        ephemeris.start_utc,
        ephemeris.step_s,
    )


__all__ = [
    "Ephemeris",
    "build_ephemeris",
    "build_ephemeris_from_tle",
    "load_tle_file",
    "read_tle",
]
