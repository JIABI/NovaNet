import numpy as np

from datetime import datetime, timezone, timedelta

from typing import List, Tuple

from sgp4.api import Satrec, jday


# --- 读取 TLE ---

def load_tle_file(tle_path: str) -> List[Tuple[str, str, str]]:
    sats = []

    with open(tle_path, "r", encoding="utf-8") as f:

        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0

    while i < len(lines):

        if lines[i].startswith("0 ") and i + 2 < len(lines) and lines[i + 1].startswith("1 "):

            # 格式: 0 Name, 1 L1, 1 L2

            name = lines[i][2:].strip()

            l1 = lines[i + 1];
            l2 = lines[i + 2];
            i += 3

            sats.append((name, l1, l2))

        elif lines[i].startswith("1 ") and i + 1 < len(lines):

            name = f"SAT_{i}"

            l1 = lines[i];
            l2 = lines[i + 1];
            i += 2

            sats.append((name, l1, l2))

        else:

            i += 1

    return sats


# --- 计算 GMST (Greenwich Mean Sidereal Time) ---

def gstime_from_jd(jd_ut1: float) -> float:
    """Return Greenwich Sidereal Time (radians) from Julian date UT1."""

    tut1 = (jd_ut1 - 2451545.0) / 36525.0

    gmst = 67310.54841 \
 \
           + (876600.0 * 3600 + 8640184.812866) * tut1 \
 \
           + 0.093104 * (tut1 ** 2) \
 \
           - 6.2e-6 * (tut1 ** 3)
    gmst = (gmst * np.pi / 180.0) / 240.0  # to radians
    gmst = gmst % (2 * np.pi)
    return gmst


# --- TEME -> ECEF (绕 z 轴旋转 GMST) ---

def teme_to_ecef(r_teme_km: np.ndarray, jd_ut1: float) -> np.ndarray:
    theta = gstime_from_jd(jd_ut1)

    c, s = np.cos(theta), np.sin(theta)

    R3 = np.array([[c, s, 0.0],

                   [-s, c, 0.0],

                   [0.0, 0.0, 1.0]])

    r_ecef_km = r_teme_km @ R3.T

    return r_ecef_km


# --- 构建星历矩阵 [T, N, 3] ---

def build_ephemeris_from_tle(

        tle_path: str,

        start_utc: datetime = None,

        duration_s: int = 2400,

        dt_s: int = 1,

        limit_sats: int = None

):
    sats = load_tle_file(tle_path)

    if limit_sats is not None:
        sats = sats[:limit_sats]

    if start_utc is None:
        start_utc = datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0)

    recs = [(name, Satrec.twoline2rv(l1, l2)) for (name, l1, l2) in sats]

    names = [n for n, _ in recs]

    N = len(recs)

    T = duration_s // dt_s + 1

    ephem = np.full((T, N, 3), np.nan, dtype=float)

    for ti in range(T):

        t = start_utc + timedelta(seconds=ti * dt_s)

        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)

        jd_full = jd + fr

        for si, (name, sat) in enumerate(recs):

            e, r_teme_km, _v = sat.sgp4(jd, fr)

            if e != 0:
                continue

            r_ecef_km = teme_to_ecef(np.asarray(r_teme_km)[None, :], jd_full)[0]

            ephem[ti, si, :] = r_ecef_km * 1000.0  # to meters

    return ephem, names, start_utc, dt_s



