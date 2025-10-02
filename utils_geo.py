import numpy as np

EARTH_RADIUS_M = 6_371_000.0  # spherical Earth (足够用于可视化)


def ecef_to_geodetic_spherical(xyz_m: np.ndarray):
    """

    输入: [..., 3] 的 ECEF 米坐标

    输出: (lat_deg, lon_deg, alt_m) 的数组

    注意: 为可视化与快速验证，采用球地模型；若需更高精度可改用 WGS84 椭球 + pymap3d。

    """

    x, y, z = xyz_m[..., 0], xyz_m[..., 1], xyz_m[..., 2]

    r_xy = np.hypot(x, y)

    lon = np.arctan2(y, x)

    hyp = np.hypot(r_xy, z)

    lat = np.arctan2(z, r_xy + 1e-12)

    alt = hyp - EARTH_RADIUS_M

    return np.degrees(lat), np.degrees(lon), alt

