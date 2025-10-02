import csv, numpy as np

from config import SIM_DURATION_S, DT_S


def _autokey(hdr, key):
    for k in hdr:

        if k and str(k).lower().startswith(key):
            return k

    return None


def load_stk_ephemeris(csv_path):
    """

    读取 STK 导出的星历 CSV，返回：

    - ephem: [T_valid, N_sat, 3] 的 ECEF 轨迹（米）

    - sat_names: 卫星名称列表（与 ephem 的第二维对齐）

    - valid_t: 原始时间索引中有效的时间片（去除了缺数据的时刻）

    期望 CSV 列：Object Name, Time(sec), X(m), Y(m), Z(m)，坐标系 ITRF(Earth Fixed)。

    """

    rows = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:

        reader = csv.DictReader(f)

        hdr = reader.fieldnames

        # 列名自动匹配（尽量宽松）

        name_k = None

        for k in hdr:

            if 'object' in k.lower() and 'name' in k.lower():
                name_k = k

        time_k = _autokey(hdr, 'time')

        x_k = _autokey(hdr, 'x');
        y_k = _autokey(hdr, 'y');
        z_k = _autokey(hdr, 'z')

        assert all([name_k, time_k, x_k, y_k, z_k]), f"CSV missing needed columns. Found: {hdr}"

        for r in reader:

            try:

                t = float(r[time_k])  # 要求 Time 为秒

            except:

                continue

            rows.append((r[name_k], t, float(r[x_k]), float(r[y_k]), float(r[z_k])))

    sat_names = sorted(list({r[0] for r in rows}))

    idx = {n: i for i, n in enumerate(sat_names)}

    T = int(SIM_DURATION_S // DT_S) + 1

    N = len(sat_names)

    ephem = np.full((T, N, 3), np.nan, dtype=float)

    for name, t, x, y, z in rows:

        ti = int(round(t / DT_S))

        if 0 <= ti < T:
            ephem[ti, idx[name], :] = (x, y, z)

    # 仅保留完整的时间片

    mask = np.isfinite(ephem).all(axis=(1, 2))

    valid_t = np.where(mask)[0]

    if valid_t.size == 0:
        raise ValueError("No complete time slices in CSV. Check STK export (all satellites every second).")

    ephem = ephem[valid_t]

    return ephem, sat_names, valid_t

