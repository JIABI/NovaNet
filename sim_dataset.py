import math
import random
import numpy as np

from typing import List, Dict, Tuple

from config import (
    # 链路/物理
    SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI, BANDWIDTH_HZ, CARRIER_HZ,
    NOISE_PSD_DBM_HZ, SMALL_SCALE_FADING_DB, ATTEN_DB_PER_KM,
    EARTH_RADIUS_M, ELEV_MIN_DEG,

    # 数据规模/采样
    NUM_SAMPLES, TOP_K, RNG_SEED, SIM_DURATION_S, DT_S, DELTA,

    # 模型输入维度
    F_SAT, F_UE,

    # 切换/判据
    HOM_DB, EXTRA_MARGIN_DB, TAU_TTL_STEPS,
)

# 星历构建（TLE）
from tle_ephem import build_ephemeris_from_tle


# ========== 基础工具 ==========
def eirp_dbm(tx_dbm: float, tx_gain_dbi: float) -> float:
    return tx_dbm + tx_gain_dbi


def noise_power_dbm(bw_hz: float) -> float:
    # N0(dBm/Hz) + 10log10(BW)
    return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)


def fspl_db(d_km: float, f_hz: float) -> float:
    # FSPL(dB) = 20log10(d[km]) + 20log10(f[MHz]) + 32.44
    if d_km <= 0:
        d_km = 0.001
    f_mhz = f_hz / 1e6
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44


def pathloss_db(d_km: float, f_hz: float) -> float:
    # 基础FSPL + 小尺度起伏 + 气体/云雨等简化衰减(与距离成正比)
    return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = EARTH_RADIUS_M + alt_m
    return np.array([
        r * math.cos(lat) * math.cos(lon),
        r * math.cos(lat) * math.sin(lon),
        r * math.sin(lat)
    ], dtype=float)


def elevation_angle(ue_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    ue_n = ue_ecef / np.linalg.norm(ue_ecef)
    los = sat_ecef - ue_ecef
    los_n = los / np.linalg.norm(los)
    cosz = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    z = math.degrees(math.acos(cosz))
    return 90.0 - z


# ========== TTL（可见剩余时长，单位：步） ==========
def time_to_leave_steps(
        ue_ecef: np.ndarray,
        sat_ecef_all: np.ndarray,  # [T, N, 3]
        global_id: int,
        start_t_idx: int,
        max_look_steps: int = 1800
) -> int:
    T = sat_ecef_all.shape[0]
    end = min(T - 1, start_t_idx + max_look_steps)
    dur = 0
    for tt in range(start_t_idx, end):
        if elevation_angle(ue_ecef, sat_ecef_all[tt, global_id]) < ELEV_MIN_DEG:
            break
        dur += 1
    return dur


# ========== 样本构建（路线B 标签齐全版） ==========
def build_sample(
        rng: random.Random,
        sat_ecef_all: np.ndarray,  # [T, N, 3]
        t_idx: int,  # 当前时刻索引，需满足 1 <= t_idx < T - delta_idx
        delta_idx: int  # 未来步长
) -> Dict:
    """
    返回：
      ue_feat:   [1, F_UE]
      sat_feats: [K, F_SAT]  (F_SAT=6: [elev, d_elev_dt, range, d_range_dt, pathloss, potential])
      edge_attr: [K, F_SAT + 3] (拼上方向余弦)
      edge_index:[2, K]  (星图：UE->K个卫星的星形图)
      label_cls: ()   未来Δ步真正最佳卫星在候选内的下标
      label_snr: [K]  未来Δ步各候选的 SNR 标签
      ho_event:  ()   简易切换标签（是否达到 HOM）
      ttl:       [K]  每个候选的剩余可见时隙（从 t_idx 起）
      y_stay:    ()   是否应当保留当前连接（1=保留，0=值得切）
      label_curr:()   当前最佳（按 pathloss 最小）
    """
    assert isinstance(t_idx, (int, np.integer)), "t_idx 必须是整数"
    assert isinstance(delta_idx, (int, np.integer)), "delta_idx 必须是整数"

    T, N = sat_ecef_all.shape[:2]
    assert 1 <= t_idx < T - delta_idx, "t_idx 越界：需要 1 <= t_idx 且 t_idx + delta_idx < T"

    sat_prev = sat_ecef_all[t_idx - 1]  # [N,3]
    sat_t = sat_ecef_all[t_idx]  # [N,3]
    sat_fut = sat_ecef_all[t_idx + delta_idx]  # [N,3]

    # 1) 随机 UE 位置（纬度-60~60，模拟中低纬活动区域；可按需改）
    lat, lon = rng.uniform(-60, 60), rng.uniform(-180, 180)
    ue = geodetic_to_ecef(lat, lon, 0.0)

    # 2) 候选 Top-K（按当前仰角），若不足则用最高仰角补齐到K
    elevs_t = np.array([elevation_angle(ue, s) for s in sat_t])
    cand = np.where(elevs_t >= ELEV_MIN_DEG)[0].tolist()
    if len(cand) < TOP_K:
        order_all = np.argsort(elevs_t)[::-1]
        extra = [i for i in order_all if i not in cand]
        cand = (cand + extra)[:TOP_K]
    else:
        cand = sorted(cand, key=lambda i: elevs_t[i], reverse=True)[:TOP_K]
    sel = np.array(cand, dtype=np.int64)  # 候选全球索引 [K]

    # 3) 未来Δ步 SNR 标签
    eirp = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)

    d_km_future = np.linalg.norm(sat_fut - ue, axis=1) / 1000.0
    pl_future = np.array([pathloss_db(d, CARRIER_HZ) for d in d_km_future])
    snr_future = eirp - pl_future - noise  # [N]

    best_future_global = int(np.argmax(snr_future))
    # 保证未来最佳在候选内
    if best_future_global not in sel:
        sel[-1] = best_future_global

    # 未来最佳在候选内的下标
    label_cls = int(np.where(sel == best_future_global)[0][0])
    y_snr = snr_future[sel].astype(np.float32)  # [K]

    # 4) 构造当前输入特征 F_SAT
    elevs_prev = np.array([elevation_angle(ue, s) for s in sat_prev])
    sel_elev_t = elevs_t[sel]
    sel_elev_prev = elevs_prev[sel]
    d_elev_dt = (sel_elev_t - sel_elev_prev) / DT_S

    rng_prev = np.linalg.norm(sat_prev[sel] - ue, axis=1) / 1000.0
    rng_now = np.linalg.norm(sat_t[sel] - ue, axis=1) / 1000.0
    d_rng_dt = (rng_now - rng_prev) / DT_S
    pl_now = np.array([pathloss_db(d, CARRIER_HZ) for d in rng_now])
    pot = sel_elev_t / 90.0

    if F_SAT >= 6:
        sat_feats = np.stack([sel_elev_t, d_elev_dt, rng_now, d_rng_dt, pl_now, pot], axis=1).astype(np.float32)
        pl_col = 4
    else:
        # 兼容 F_SAT=4 的旧设置
        sat_feats = np.stack([sel_elev_t, rng_now, pl_now, pot], axis=1).astype(np.float32)
        pl_col = 2

    # 方向余弦作为 edge_attr 的几何信息
    dirs = (sat_t[sel] - ue)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    edge_attr = np.concatenate([sat_feats, dirs.astype(np.float32)], axis=1)  # [K, F_SAT+3]

    ue_feat = np.zeros((1, F_UE), np.float32)  # 预留UE特征位

    # 5) 当前最佳（用 pathloss 最小）
    cur_idx = int(np.argmin(sat_feats[:, pl_col]))
    label_curr = cur_idx

    # 6) TTL 标签：每个候选的剩余可见步数
    ttl = []
    for gid in sel.tolist():
        ttl_k = time_to_leave_steps(ue, sat_ecef_all, gid, t_idx)
        ttl.append(ttl_k)
    ttl = np.array(ttl, dtype=np.float32)  # [K]

    # 7) y_stay 标签（应当保留当前=1）：
    # 真实 margin = 未来最佳的 y_snr[label_cls] - 当前的 SNR_now(cur_idx)
    snr_now_all = eirp - pl_now - noise
    snr_now_cur = float(snr_now_all[cur_idx])
    snr_fut_best = float(y_snr[label_cls])
    margin_true = snr_fut_best - snr_now_cur

    ttl_gain = float(ttl[label_cls] - ttl[cur_idx])
    y_stay = 1 if (margin_true < (HOM_DB + EXTRA_MARGIN_DB) or ttl_gain < TAU_TTL_STEPS) else 0

    # 8) 简易 ho_event（与早期版本保持风格）
    ho_event = 1 if margin_true >= HOM_DB else 0

    return dict(
        ue_feat=ue_feat,  # [1,F_UE]
        sat_feats=sat_feats,  # [K,F_SAT]
        edge_attr=edge_attr,  # [K,F_SAT+3]
        edge_index=np.stack([  # 星形图（UE->sat）
            np.zeros(TOP_K, dtype= np.int64),
            np.arange(TOP_K, dtype=np.int64)
        ], axis=0),
        label_cls=label_cls,  # ()
        label_snr=y_snr,  # [K]
        ho_event=ho_event,  # ()
        ttl=ttl,  # [K]
        y_stay=y_stay,  # ()
        label_curr=label_curr,  # ()
    )


# ========== 从 TLE 生成数据集 ==========
def generate_dataset_from_tle(
        tle_path: str,
        num_samples: int = NUM_SAMPLES,
        seed: int = RNG_SEED,
        limit_sats: int = None
) -> List[Dict]:
    """
    1) 用 TLE 构建星历 ephem: [T, N, 3]
    2) 随机采样 t_idx，构造样本（含 TTL/y_stay/label_curr）
    """
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        tle_path, duration_s=SIM_DURATION_S, dt_s=DT_S, limit_sats=limit_sats
    )
    T = ephem.shape[0]
    delta = int(DELTA)
    assert delta >= 1, "DELTA 必须为正整数"

    rng = random.Random(seed)

    data: List[Dict] = []
    # 需要 t_idx-1 有效、且 t_idx+delta < T
    for _ in range(num_samples):
        t_idx = rng.randrange(1, T - delta)  # 保证 1 <= t_idx < T - delta
        sample = build_sample(rng, ephem, t_idx, delta)
        data.append(sample)

    return data





