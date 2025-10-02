import argparse, random

import numpy as np

import matplotlib.pyplot as plt

from config import (SAT_TX_POWER_DBM, BANDWIDTH_HZ, CARRIER_HZ,

                    ELEV_MIN_DEG, SMALL_SCALE_FADING_DB, ATTEN_DB_PER_KM)

from tle_ephem import build_ephemeris_from_tle

from utils_geo import ecef_to_geodetic_spherical

EARTH_RADIUS_M = 6_371_000.0

HOM_DB = 3.0  # 你也可 from config import HOM_DB

DT_S = 1  # 可 from config


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg);
    lon = np.radians(lon_deg)

    r = EARTH_RADIUS_M + alt_m

    x = r * np.cos(lat) * np.cos(lon)

    y = r * np.cos(lat) * np.sin(lon)

    z = r * np.sin(lat)

    return np.array([x, y, z], float)


def noise_power_dbm(bw_hz): return -173.0 + 10 * np.log10(bw_hz)


def fspl_db(d_km, f_hz):
    if d_km <= 0: d_km = 0.001

    f_mhz = f_hz / 1e6

    return 20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44


def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km


def eirp_dbm(tx, txg): return tx + txg


def elevation_angle(ue, sat):
    ue_n = ue / np.linalg.norm(ue)

    los = sat - ue

    los_n = los / np.linalg.norm(los)

    cos_z = np.clip(np.dot(los_n, ue_n), -1.0, 1.0)

    z = np.degrees(np.arccos(cos_z))

    return 90.0 - z


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tle", required=True)

    ap.add_argument("--ue_lat", type=float, default=None)

    ap.add_argument("--ue_lon", type=float, default=None)

    args = ap.parse_args()

    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(args.tle, duration_s=2400, dt_s=1)

    T, N, _ = ephem.shape

    # 随机或指定一个 UE 位置

    if args.ue_lat is None or args.ue_lon is None:

        ue_lat = random.uniform(-60, 60)

        ue_lon = random.uniform(-180, 180)

    else:

        ue_lat, ue_lon = args.ue_lat, args.ue_lon

    ue = geodetic_to_ecef(ue_lat, ue_lon, 0.0)

    eirp = eirp_dbm(SAT_TX_POWER_DBM, 30.0)  # 30 dBi

    noise = noise_power_dbm(BANDWIDTH_HZ)

    best_snr = []

    best_sat = []

    for ti in range(T):

        sats_t = ephem[ti]  # [N,3]

        elevs = np.array([elevation_angle(ue, s) for s in sats_t])

        mask = elevs >= ELEV_MIN_DEG

        if not np.any(mask):
            best_snr.append(np.nan)

            best_sat.append(-1)

            continue

        d_km = np.linalg.norm(sats_t[mask] - ue, axis=1) / 1000.0

        pl = np.array([pathloss_db(d, CARRIER_HZ) for d in d_km])

        snr = eirp - pl - noise

        idx = np.argmax(snr)

        best_snr.append(snr[idx])

        # 记录真实卫星索引

        sat_indices = np.where(mask)[0]

        best_sat.append(int(sat_indices[idx]))

    # 检测切换：HOM=3 dB，TTT=0 s

    ho_times = []

    for ti in range(T - 1):

        if np.isnan(best_snr[ti]) or np.isnan(best_snr[ti + 1]): continue

        # 如果下一秒“新最佳卫星”的 SNR 比当前高 >= 3 dB，就认为触发切换

        if best_sat[ti] != best_sat[ti + 1] and (best_snr[ti + 1] - best_snr[ti] >= HOM_DB):
            ho_times.append(ti + 1)

    # 画图

    t_axis = np.arange(T) * dt_s

    plt.figure(figsize=(10, 4))

    plt.plot(t_axis, best_snr, label="Best SNR (UE at lat={:.2f}, lon={:.2f})".format(ue_lat, ue_lon))

    for ht in ho_times:
        plt.axvline(ht, linestyle="--", alpha=0.4)

    plt.title("Best SNR over time & detected handovers (HOM=3 dB, TTT=0 s)")

    plt.xlabel("Time (s)")

    plt.ylabel("SNR (dB)")

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

