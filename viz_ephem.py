import argparse

import numpy as np

import matplotlib.pyplot as plt

from tle_ephem import build_ephemeris_from_tle

from utils_geo import ecef_to_geodetic_spherical


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tle", required=True, help="Path to TLE file (e.g., starlink.tle)")

    ap.add_argument("--limit_sats", type=int, default=5, help="Number of satellites to plot ground tracks for")

    args = ap.parse_args()

    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(args.tle, duration_s=2400, dt_s=1,
                                                             limit_sats=args.limit_sats)

    T, N, _ = ephem.shape

    print(f"Ephemeris: T={T} steps, N={N} satellites, start={start_utc.isoformat()}")

    plt.figure(figsize=(10, 5))

    # 画空白“世界地图”背景（简单经纬度坐标轴）

    plt.title("Ground Tracks (first {} sats)".format(N))

    plt.xlabel("Longitude (deg)")

    plt.ylabel("Latitude (deg)")

    plt.xlim([-180, 180])

    plt.ylim([-90, 90])

    plt.grid(True, alpha=0.3)

    for si in range(N):
        lat, lon, _alt = ecef_to_geodetic_spherical(ephem[:, si, :])

        # 处理经度跳变

        lon_unwrapped = np.unwrap(np.radians(lon))

        lon_unwrapped = np.degrees(lon_unwrapped)

        plt.plot(lon_unwrapped, lat, label=names[si], linewidth=1)

    plt.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

