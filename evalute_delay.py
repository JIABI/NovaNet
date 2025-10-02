import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

import config as CFG
from tle_ephem import build_ephemeris_from_tle
from model import PCGNN


# ---------- 从 config 读参数（若缺失给默认） ----------
def C(name, default):
    return getattr(CFG, name, default)

# 基本/星历 & 评估规模
TLE_PATH      = C('TLE_PATH', 'starlink.tle')
SIM_DURATION_S= C('SIM_DURATION_S', 3600)
DT_S          = C('DT_S', 10)
DELTA         = int(C('DELTA', 30))
TOP_K         = C('TOP_K', 15)
ELEV_MIN_DEG  = C('ELEV_MIN_DEG', 10.0)
LIMIT_SATS    = C('LIMIT_SATS', 200)

# 模型维度
F_UE          = C('F_UE', 4)
F_SAT         = C('F_SAT', 6)
F_EDGE        = C('F_EDGE', 9)
HIDDEN        = C('HIDDEN', 128)
GNN_LAYERS    = C('GNN_LAYERS', 2)
GRAPH_TOPK    = C('GRAPH_TOPK', 8)
ADJ_TAU       = C('ADJ_TAU', 0.2)

# 物理链路
SAT_TX_POWER_DBM   = C('SAT_TX_POWER_DBM', 40.0)
SAT_ANT_GAIN_DBI   = C('SAT_ANT_GAIN_DBI', 30.0)
BANDWIDTH_HZ       = C('BANDWIDTH_HZ', 10e6)
CARRIER_HZ         = C('CARRIER_HZ', 11.9e9)
NOISE_PSD_DBM_HZ   = C('NOISE_PSD_DBM_HZ', -173.0)
SMALL_SCALE_FADING_DB = C('SMALL_SCALE_FADING_DB', 20.0)
ATTEN_DB_PER_KM    = C('ATTEN_DB_PER_KM', 0.05)
EARTH_RADIUS_M     = C('EARTH_RADIUS_M', 6371000.0)
HOM_DB             = C('HOM_DB', 3.0)
TTT_SEC            = C('TTT_SEC', 0.0)

# 路线B新增
EXTRA_MARGIN_DB    = C('EXTRA_MARGIN_DB', 1.0)     # 额外 SNR 冗余
TAU_TTL_STEPS      = int(C('TAU_TTL_STEPS', 2))    # TTL 增益门槛（单位=步）

# Delay 模型（可放到 config；这里给默认值）
LIGHT_SPEED_M_S    = C('LIGHT_SPEED_M_S', 299792458.0)
HO_SIGNALING_MS    = C('HO_SIGNALING_MS', 60.0)
HO_SYNC_MS         = C('HO_SYNC_MS', 10.0)
HO_BACKOFF_MS      = C('HO_BACKOFF_MS', 5.0)
HO_DELAY_MS        = C('HO_DELAY_MS', HO_SIGNALING_MS + HO_SYNC_MS + HO_BACKOFF_MS)

PKT_SIZE_BITS      = C('PKT_SIZE_BITS', 12*1024*8)  # 12KB
PHY_EFFICIENCY     = C('PHY_EFFICIENCY', 0.75)
MIN_DATA_RATE_BPS  = C('MIN_DATA_RATE_BPS', 1e5)
PROC_DELAY_MS      = C('PROC_DELAY_MS', 0.3)
QUEUE_DELAY_MS     = C('QUEUE_DELAY_MS', 0.5)


# ---------- 基础链路/几何工具 ----------
def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi

def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)

def fspl_db(d_km, f_hz):
    if d_km <= 0: d_km = 0.001
    f_mhz = f_hz / 1e6
    return 20.0*math.log10(d_km) + 20.0*math.log10(f_mhz) + 32.44

def pathloss_db(d_km, f_hz):
    return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km

def geodetic_to_ecef(lat, lon, alt_m=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    r = EARTH_RADIUS_M + alt_m
    return np.array([r*np.cos(lat)*np.cos(lon),
                     r*np.cos(lat)*np.sin(lon),
                     r*np.sin(lat)], float)

def elevation_angle(ue, sat):
    ue_n  = ue / np.linalg.norm(ue)
    los   = sat - ue
    los_n = los / np.linalg.norm(los)
    cosz  = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    return 90.0 - math.degrees(math.acos(cosz))


# ---------- TTL 工具 ----------
def time_to_leave_steps(ue_ecef, ephem, global_id, t_idx, max_look=1800):
    T = ephem.shape[0]
    end = min(T-1, t_idx + max_look)
    dur = 0
    for tt in range(t_idx, end):
        if elevation_angle(ue_ecef, ephem[tt, global_id]) < ELEV_MIN_DEG:
            break
        dur += 1
    return dur


# ---------- 构帧：强制把当前卫星放入候选 ----------
def build_frame_for_fixed_ue(ue_ecef, sat_prev, sat_t, cur_global_id=None):
    elevs_t = np.array([elevation_angle(ue_ecef, s) for s in sat_t])
    cand = np.where(elevs_t >= ELEV_MIN_DEG)[0].tolist()
    if len(cand) < TOP_K:
        order_all = np.argsort(elevs_t)[::-1]
        extra = [i for i in order_all if i not in cand]
        cand = (cand + extra)[:TOP_K]
    else:
        cand = sorted(cand, key=lambda i: elevs_t[i], reverse=True)[:TOP_K]

    if cur_global_id is not None and elevs_t[cur_global_id] >= ELEV_MIN_DEG:
        if cur_global_id not in cand:
            cand[-1] = cur_global_id

    sel = np.array(cand, dtype=np.int64)

    elevs_prev = np.array([elevation_angle(ue_ecef, s) for s in sat_prev])
    sel_elev_t = elevs_t[sel]
    sel_elev_prev = elevs_prev[sel]
    d_elev_dt = (sel_elev_t - sel_elev_prev) / DT_S

    rng_prev = np.linalg.norm(sat_prev[sel] - ue_ecef, axis=1) / 1000.0
    rng_now  = np.linalg.norm(sat_t[sel]   - ue_ecef, axis=1) / 1000.0
    d_rng_dt = (rng_now - rng_prev) / DT_S
    pl_now   = np.array([pathloss_db(d, CARRIER_HZ) for d in rng_now])
    pot      = sel_elev_t / 90.0

    if F_SAT >= 6:
        sat_feats = np.stack([sel_elev_t, d_elev_dt, rng_now, d_rng_dt, pl_now, pot], axis=1).astype(np.float32)
    else:
        sat_feats = np.stack([sel_elev_t, rng_now, pl_now, pot], axis=1).astype(np.float32)

    dirs = (sat_t[sel] - ue_ecef)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    edge_attr = np.concatenate([sat_feats, dirs.astype(np.float32)], axis=1)

    ue_feat = np.zeros((1, F_UE), np.float32)
    return ue_feat, sat_feats, edge_attr, sel


# ---------- 三种策略的“下一步选择” ----------
def choose_by_model(model, ue_feat, sat_feats, edge_attr):
    dev = next(model.parameters()).device
    with torch.no_grad():
        ue = torch.tensor(ue_feat, dtype=torch.float32, device=dev).unsqueeze(0)
        sa = torch.tensor(sat_feats, dtype=torch.float32, device=dev).unsqueeze(0)
        ed = torch.tensor(edge_attr, dtype=torch.float32, device=dev).unsqueeze(0)
        # 期望 model 返回: psnr, plog, pho, p_stay, p_ttl
        _, plog, _, p_stay, _ = model(ue, sa, ed)
        idx = int(torch.argmax(plog, dim=1).item())
        return idx, float(p_stay.item())

def choose_by_max_elevation(sat_feats):
    return int(np.argmax(sat_feats[:, 0]))

def choose_by_max_serve_time(ue_ecef, sel_global, ephem, t_idx):
    durations = [time_to_leave_steps(ue_ecef, ephem, gid, t_idx) for gid in sel_global]
    return int(np.argmax(durations))


# ---------- Handover 频次仿真 ----------
def simulate_ho_frequency(model, ephem, n_users=50, span_s=1800, step_s=30, stay_gate=0.5):
    eirp  = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)
    PL_COL = 4 if F_SAT >= 6 else 2

    T = ephem.shape[0]
    span = span_s
    step = step_s

    TTT_SLOTS = max(1, int(TTT_SEC / step)) if TTT_SEC > 0 else 1
    MIN_DWELL_SLOTS = 5

    freq_model, freq_elev, freq_serv = [], [], []

    for _ in range(n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(1, T - span - max(DELTA, 1))

        ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_m, _pstay = choose_by_model(model, ue_f, sf_m, ea_m); cur_gid_m = sel_m[idx_m]

        ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_e = choose_by_max_elevation(sf_e); cur_gid_e = sel_e[idx_e]

        ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t0); cur_gid_s = sel_s[idx_s]

        cnt_m = cnt_e = cnt_s = 0
        better_m = better_e = better_s = 0
        dwell_m = dwell_e = dwell_s = 1

        for t in range(t0 + step, t0 + span, step):
            ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_m)
            ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_e)
            ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_s)

            # ===== PCGNN =====
            nxt_idx_m, p_stay = choose_by_model(model, ue_f, sf_m, ea_m)
            nxt_gid_m = sel_m[nxt_idx_m]
            snr_all_m = eirp - sf_m[:, PL_COL] - noise

            cur_pos_m = np.where(sel_m == cur_gid_m)[0]
            if len(cur_pos_m) == 0:
                cur_gid_m = nxt_gid_m
                cnt_m += 1
                dwell_m = 1
                better_m = 0
            else:
                cur_pos_m = int(cur_pos_m[0])
                margin_m = float(snr_all_m[nxt_idx_m] - snr_all_m[cur_pos_m])

                ttl_cur = time_to_leave_steps(ue, ephem, cur_gid_m, t)
                ttl_nxt = time_to_leave_steps(ue, ephem, nxt_gid_m, t)
                ttl_gain = ttl_nxt - ttl_cur

                if p_stay >= stay_gate:
                    better_m = 0
                    dwell_m += 1
                else:
                    if margin_m >= (HOM_DB + EXTRA_MARGIN_DB): better_m += 1
                    else: better_m = 0

                    if nxt_gid_m == cur_gid_m:
                        dwell_m += 1
                    else:
                        if (better_m >= TTT_SLOTS and dwell_m >= MIN_DWELL_SLOTS and ttl_gain >= TAU_TTL_STEPS):
                            cnt_m += 1
                            cur_gid_m = nxt_gid_m
                            dwell_m = 1
                            better_m = 0
                        else:
                            dwell_m += 1

            # ===== Max-Elevation =====
            nxt_idx_e = choose_by_max_elevation(sf_e)
            nxt_gid_e = sel_e[nxt_idx_e]
            snr_all_e = eirp - sf_e[:, PL_COL] - noise

            cur_pos_e = np.where(sel_e == cur_gid_e)[0]
            if len(cur_pos_e) == 0:
                cur_gid_e = nxt_gid_e
                cnt_e += 1
                dwell_e = 1
                better_e = 0
            else:
                cur_pos_e = int(cur_pos_e[0])
                margin_e = float(snr_all_e[nxt_idx_e] - snr_all_e[cur_pos_e])

                if margin_e >= (HOM_DB + EXTRA_MARGIN_DB): better_e += 1
                else: better_e = 0

                if nxt_gid_e == cur_gid_e:
                    dwell_e += 1
                else:
                    if (better_e >= TTT_SLOTS and dwell_e >= MIN_DWELL_SLOTS):
                        cnt_e += 1
                        cur_gid_e = nxt_gid_e
                        dwell_e = 1
                        better_e = 0
                    else:
                        dwell_e += 1

            # ===== Max-serveTime =====
            nxt_idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t)
            nxt_gid_s = sel_s[nxt_idx_s]
            snr_all_s = eirp - sf_s[:, PL_COL] - noise

            cur_pos_s = np.where(sel_s == cur_gid_s)[0]
            if len(cur_pos_s) == 0:
                cur_gid_s = nxt_gid_s
                cnt_s += 1
                dwell_s = 1
                better_s = 0
            else:
                cur_pos_s = int(cur_pos_s[0])
                margin_s = float(snr_all_s[nxt_idx_s] - snr_all_s[cur_pos_s])

                if margin_s >= (HOM_DB + EXTRA_MARGIN_DB): better_s += 1
                else: better_s = 0

                if nxt_gid_s == cur_gid_s:
                    dwell_s += 1
                else:
                    if (better_s >= TTT_SLOTS and dwell_s >= MIN_DWELL_SLOTS):
                        cnt_s += 1
                        cur_gid_s = nxt_gid_s
                        dwell_s = 1
                        better_s = 0
                    else:
                        dwell_s += 1

        freq_model.append(cnt_m)
        freq_elev.append(cnt_e)
        freq_serv.append(cnt_s)

    return np.array(freq_model), np.array(freq_elev), np.array(freq_serv)


# ---------- Delay 辅助 ----------
def snr_db_from_pathloss_db(pl_db, eirp_dbm, noise_dbm):
    return eirp_dbm - pl_db - noise_dbm

def rate_bps_from_snr_db(snr_db, bandwidth_hz, phy_eff=0.75, min_bps=1e5):
    snr_lin = np.power(10.0, snr_db / 10.0)
    cap = bandwidth_hz * np.log2(1.0 + snr_lin)
    return max(phy_eff * cap, min_bps)

def tx_delay_ms_from_rate(rate_bps, pkt_bits, proc_ms=0.3, queue_ms=0.5):
    return pkt_bits / rate_bps * 1000.0 + proc_ms + queue_ms

def prop_delay_ms(range_km):
    return (range_km * 1000.0) / LIGHT_SPEED_M_S * 1000.0


# ---------- 频次+时延联合仿真 ----------
def simulate_delays(model, ephem, n_users=50, span_s=1800, step_s=30, stay_gate=0.5,
                    include_prop_delay=False):
    eirp  = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)
    PL_COL = 4 if F_SAT >= 6 else 2
    RG_COL = 2 if F_SAT >= 6 else 1

    T = ephem.shape[0]
    span = span_s
    step = step_s

    TTT_SLOTS = max(1, int(TTT_SEC / step)) if TTT_SEC > 0 else 1
    MIN_DWELL_SLOTS = 5

    fm, fe, fs = [], [], []
    dm, de, ds = [], [], []
    tm, te, ts = [], [], []

    for _ in range(n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(1, T - span - max(DELTA, 1))

        ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_m, _pstay = choose_by_model(model, ue_f, sf_m, ea_m); cur_gid_m = sel_m[idx_m]

        ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_e = choose_by_max_elevation(sf_e); cur_gid_e = sel_e[idx_e]

        ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t0); cur_gid_s = sel_s[idx_s]

        cnt_m = cnt_e = cnt_s = 0
        dly_m = dly_e = dly_s = 0.0
        tx_sum_m = tx_sum_e = tx_sum_s = 0.0
        tx_cnt = 0

        better_m = better_e = better_s = 0
        dwell_m = dwell_e = dwell_s = 1

        for t in range(t0 + step, t0 + span, step):
            ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_m)
            ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_e)
            ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_s)

            def cur_tx_delay_ms(sf, sel, cur_gid):
                pos = np.where(sel == cur_gid)[0]
                if len(pos) == 0:
                    pos = int(np.argmax(sf[:, 0]))
                else:
                    pos = int(pos[0])
                pl = float(sf[pos, PL_COL])
                snr_db = snr_db_from_pathloss_db(pl, eirp, noise)
                rate = rate_bps_from_snr_db(snr_db, BANDWIDTH_HZ, PHY_EFFICIENCY, MIN_DATA_RATE_BPS)
                tx_ms = tx_delay_ms_from_rate(rate, PKT_SIZE_BITS, PROC_DELAY_MS, QUEUE_DELAY_MS)
                if include_prop_delay:
                    tx_ms += prop_delay_ms(float(sf[pos, RG_COL]))
                return tx_ms

            tx_sum_m += cur_tx_delay_ms(sf_m, sel_m, cur_gid_m)
            tx_sum_e += cur_tx_delay_ms(sf_e, sel_e, cur_gid_e)
            tx_sum_s += cur_tx_delay_ms(sf_s, sel_s, cur_gid_s)
            tx_cnt += 1

            # ===== PCGNN 切换逻辑 =====
            nxt_idx_m, p_stay = choose_by_model(model, ue_f, sf_m, ea_m)
            nxt_gid_m = sel_m[nxt_idx_m]
            snr_all_m = eirp - sf_m[:, PL_COL] - noise

            cur_pos_m = np.where(sel_m == cur_gid_m)[0]
            if len(cur_pos_m) == 0:
                cur_gid_m = nxt_gid_m
                cnt_m += 1
                dly_m += HO_DELAY_MS
                dwell_m = 1
                better_m = 0
            else:
                cur_pos_m = int(cur_pos_m[0])
                margin_m = float(snr_all_m[nxt_idx_m] - snr_all_m[cur_pos_m])

                ttl_cur = time_to_leave_steps(ue, ephem, cur_gid_m, t)
                ttl_nxt = time_to_leave_steps(ue, ephem, nxt_gid_m, t)
                ttl_gain = ttl_nxt - ttl_cur

                if p_stay >= stay_gate:
                    better_m = 0
                    dwell_m += 1
                else:
                    if margin_m >= (HOM_DB + EXTRA_MARGIN_DB): better_m += 1
                    else: better_m = 0

                    if nxt_gid_m == cur_gid_m:
                        dwell_m += 1
                    else:
                        if (better_m >= max(1, int(TTT_SEC/step_s)) and dwell_m >= 5 and ttl_gain >= TAU_TTL_STEPS):
                            cnt_m += 1
                            dly_m += HO_DELAY_MS
                            cur_gid_m = nxt_gid_m
                            dwell_m = 1
                            better_m = 0
                        else:
                            dwell_m += 1

            # ===== Max-Elevation =====
            nxt_idx_e = choose_by_max_elevation(sf_e)
            nxt_gid_e = sel_e[nxt_idx_e]
            snr_all_e = eirp - sf_e[:, PL_COL] - noise

            cur_pos_e = np.where(sel_e == cur_gid_e)[0]
            if len(cur_pos_e) == 0:
                cur_gid_e = nxt_gid_e
                cnt_e += 1; dly_e += HO_DELAY_MS
                dwell_e = 1; better_e = 0
            else:
                cur_pos_e = int(cur_pos_e[0])
                margin_e = float(snr_all_e[nxt_idx_e] - snr_all_e[cur_pos_e])
                if margin_e >= (HOM_DB + EXTRA_MARGIN_DB): better_e += 1
                else: better_e = 0
                if nxt_gid_e == cur_gid_e: dwell_e += 1
                else:
                    if (better_e >= TTT_SLOTS and dwell_e >= 5):
                        cnt_e += 1; dly_e += HO_DELAY_MS
                        cur_gid_e = nxt_gid_e
                        dwell_e = 1; better_e = 0
                    else:
                        dwell_e += 1

            # ===== Max-serveTime =====
            nxt_idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t)
            nxt_gid_s = sel_s[nxt_idx_s]
            snr_all_s = eirp - sf_s[:, PL_COL] - noise

            cur_pos_s = np.where(sel_s == cur_gid_s)[0]
            if len(cur_pos_s) == 0:
                cur_gid_s = nxt_gid_s
                cnt_s += 1; dly_s += HO_DELAY_MS
                dwell_s = 1; better_s = 0
            else:
                cur_pos_s = int(cur_pos_s[0])
                margin_s = float(snr_all_s[nxt_idx_s] - snr_all_s[cur_pos_s])
                if margin_s >= (HOM_DB + EXTRA_MARGIN_DB): better_s += 1
                else: better_s = 0
                if nxt_gid_s == cur_gid_s: dwell_s += 1
                else:
                    if (better_s >= TTT_SLOTS and dwell_s >= 5):
                        cnt_s += 1; dly_s += HO_DELAY_MS
                        cur_gid_s = nxt_gid_s
                        dwell_s = 1; better_s = 0
                    else:
                        dwell_s += 1

        fm.append(cnt_m); fe.append(cnt_e); fs.append(cnt_s)
        dm.append(dly_m); de.append(dly_e); ds.append(dly_s)
        tm.append(tx_sum_m / max(1, tx_cnt))
        te.append(tx_sum_e / max(1, tx_cnt))
        ts.append(tx_sum_s / max(1, tx_cnt))

    return {
        'ho_freq':  (np.array(fm), np.array(fe), np.array(fs)),
        'ho_delay': (np.array(dm), np.array(de), np.array(ds)),
        'tx_delay': (np.array(tm), np.array(te), np.array(ts)),
    }


# ---------- 画图 ----------
def plot_ho_frequency(freq_model, freq_elev, freq_serv, out_png="ho_frequency.png"):
    df = pd.DataFrame({
        "Strategy": (["PCGNN"] * len(freq_model) + ["Max-Elevation"] * len(freq_elev) + ["Max-serveTime"] * len(freq_serv)),
        "HO_Freq": np.concatenate([freq_model, freq_elev, freq_serv])
    })
    strategies = ["PCGNN", "Max-Elevation", "Max-serveTime"]
    data  = [df[df["Strategy"] == s]["HO_Freq"].values for s in strategies]
    means = [np.mean(x) for x in data]
    stds  = [np.std(x)  for x in data]

    plt.figure(figsize=(8, 6))
    xs = np.arange(len(strategies))
    plt.bar(xs, means, yerr=stds, alpha=0.6, capsize=6)
    for i, arr in enumerate(data):
        jitter = (np.random.rand(len(arr)) - 0.5) * 0.15
        plt.plot(np.full_like(arr, xs[i]) + jitter, arr, 'o', alpha=0.5)
    for i, m in enumerate(means):
        plt.text(xs[i], m + 0.2, f"{m:.1f}", ha='center')
    plt.xticks(xs, strategies)
    plt.ylabel("Handover Frequency")
    plt.title("Handover Frequency Comparison")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f">> Saved plot to {out_png}")
    plt.close()

def plot_bar_with_values(means, labels, title, ylabel, out_png):
    xs = np.arange(len(labels))
    plt.figure(figsize=(6.4, 4.0))
    bars = plt.bar(xs, means, alpha=0.8)
    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.5g}", ha='center', va='bottom', fontsize=9)
    plt.xticks(xs, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f">> Saved {out_png}")
    plt.close()


# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--span_s", type=int, default=30*60)  # 30 分钟窗口
    ap.add_argument("--step_s", type=int, default=30)     # 30 秒/步
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--stay_gate", type=float, default=0.5)   # p_stay 闸门
    ap.add_argument("--include_prop_delay", action='store_true')  # 是否计入传播时延
    args = ap.parse_args()

    # 1) 星历
    print(">> Building ephemeris ...", flush=True)
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        TLE_PATH,
        duration_s=max(SIM_DURATION_S, args.span_s + 600),
        dt_s=DT_S,
        limit_sats=args.limit_sats
    )
    print(f">> ephem: T={ephem.shape[0]}, Nsat={ephem.shape[1]}", flush=True)

    # 2) 模型
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN(
        f_ue=F_UE, f_sat=F_SAT, f_edge=F_EDGE,
        hidden=HIDDEN, gnn_layers=GNN_LAYERS,
        graph_topk=GRAPH_TOPK, adj_tau=ADJ_TAU
    ).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd)
    model.eval()
    print(f">> Loaded checkpoint: {args.ckpt}", flush=True)

    # 3) Handover 频次（可选：若只要时延，可跳过）
    fm, fe, fs = simulate_ho_frequency(
        model, ephem, n_users=args.n_users,
        span_s=args.span_s, step_s=args.step_s, stay_gate=args.stay_gate
    )
    print(f"[HO Freq] PCGNN mean={fm.mean():.2f}, Max-Elev mean={fe.mean():.2f}, Max-serveTime mean={fs.mean():.2f}")
    plot_ho_frequency(fm, fe, fs, out_png="ho_frequency.png")

    # 4) 切换总时延 + 传输时延
    res = simulate_delays(
        model, ephem,
        n_users=args.n_users, span_s=args.span_s, step_s=args.step_s,
        stay_gate=args.stay_gate, include_prop_delay=args.include_prop_delay
    )

    dm, de, ds = res['ho_delay']
    tm, te, ts = res['tx_delay']

    mean_dm, mean_de, mean_ds = dm.mean(), de.mean(), ds.mean()
    mean_tm, mean_te, mean_ts = tm.mean(), te.mean(), ts.mean()

    print(f"[HO Delay] PCGNN={mean_dm:.2f} ms | Max-Elev={mean_de:.2f} ms | Max-serveTime={mean_ds:.2f} ms")
    print(f"[TX Delay] PCGNN={mean_tm:.3f} ms | Max-Elev={mean_te:.3f} ms | Max-serveTime={mean_ts:.3f} ms")

    plot_bar_with_values(
        [mean_dm, mean_de, mean_ds],
        ["PCGNN", "Max-Elevation", "Max-serveTime"],
        "Total Delay of Handover Processes",
        "Total Delay of Handover Processes (ms)",
        out_png="bar_total_ho_delay.png"
    )
    plot_bar_with_values(
        [mean_tm, mean_te, mean_ts],
        ["PCGNN", "Max-Elevation", "Max-serveTime"],
        "Transmission Delay",
        "Transmission Delay (ms)",
        out_png="bar_tx_delay.png"
    )


if __name__ == "__main__":
    main()