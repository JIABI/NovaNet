# evaluate_oaest.py

# -*- coding: utf-8 -*-

import math, argparse, numpy as np, torch
import importlib

CFG = importlib.import_module('config')


def _get(name, default): return getattr(CFG, name, default)


# 参数
DT_S = _get('DT_S', 30)
TOP_K = _get('TOP_K', 8)
ELEV_MIN_DEG = _get('ELEV_MIN_DEG', 10.0)
LIMIT_SATS = _get('LIMIT_SATS', 32)
FREEZE_S = _get('FREEZE_S', 90.0)
HYS_MARGIN = _get('HYS_MARGIN', 0.5)
SAT_TX_POWER_DBM = _get('SAT_TX_POWER_DBM', 20.0)
SAT_ANT_GAIN_DBI = _get('SAT_ANT_GAIN_DBI', 30.0)
BANDWIDTH_HZ = _get('BANDWIDTH_HZ', 20e6)
NOISE_PSD_DBM_HZ = _get('NOISE_PSD_DBM_HZ', -174.0)
SMALL_SCALE_FADING_DB = _get('SMALL_SCALE_FADING_DB', 1.5)
CARRIER_HZ = _get('CARRIER_HZ', 400e9)
EARTH_RADIUS_M = _get('EARTH_RADIUS_M', 6371000.0)

from tle_ephem import build_ephemeris_from_tle
from orbit_feats import load_orbit_elements, orbit_prior_vector, build_spatial_affinity, build_temporal_affinity
from model_oaest import PCGNN_OAEST


# --- 基础函数 ---
def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi


def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)


def fspl_db(d_km, f_hz):
    """Free-space path loss: d in km, f in Hz"""
    f_mhz = f_hz / 1e6
    return 32.45 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(f_mhz)


def pathloss_db(d_km, f_hz):
    return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB


def elevation_angle(ue_ecef, sat_ecef):
    ue_n = ue_ecef / np.linalg.norm(ue_ecef)
    los = sat_ecef - ue_ecef
    los_n = los / np.linalg.norm(los)
    cosz = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    return 90.0 - math.degrees(math.acos(cosz))


def geodetic_to_ecef(lat, lon, alt_m=0.0):
    R = EARTH_RADIUS_M + alt_m
    clat, slat = math.cos(math.radians(lat)), math.sin(math.radians(lat))
    clon, slon = math.cos(math.radians(lon)), math.sin(math.radians(lon))
    x = R * clat * clon
    y = R * clat * slon
    z = R * slat
    return np.array([x, y, z], dtype=np.float64)


# --- Frame 构建 ---
def build_frame(ue, sat_prev, sat_t):
    elevs = np.array([elevation_angle(ue, s) for s in sat_t])
    cand = np.where(elevs >= ELEV_MIN_DEG)[0].tolist()
    if len(cand) < TOP_K:
        order_all = np.argsort(elevs)[::-1]
        extra = [i for i in order_all if i not in cand]
        cand = (cand + extra)[:TOP_K]
    else:
        cand = sorted(cand, key=lambda i: elevs[i], reverse=True)[:TOP_K]
    sel = np.array(cand, dtype=int)
    feat = []
    for gid in sel:
        elev = elevation_angle(ue, sat_t[gid])
        elev_prev = elevation_angle(ue, sat_prev[gid])
        d_elev = (elev - elev_prev) / max(1.0, DT_S)
        d = np.linalg.norm(sat_t[gid] - ue) / 1000.0
        d_prev = np.linalg.norm(sat_prev[gid] - ue) / 1000.0
        dd = (d - d_prev) / max(1e-6, DT_S)
        pl = pathloss_db(d, CARRIER_HZ)
        pot = elev / 90.0
        feat.append([elev, d_elev, d, dd, pl, pot])
    sat_feats = np.array(feat, dtype=np.float32)
    return sel, sat_feats


# --- SNR + 吞吐 ---
def compute_snr_and_rate(ue, sat_pos, verbose=False):
    d_km = np.linalg.norm(sat_pos - ue) / 1000.0
    pl = pathloss_db(d_km, CARRIER_HZ)
    eirp = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise_dbm = noise_power_dbm(BANDWIDTH_HZ)
    rx_gain = 20.0  # UE天线增益
    pr_dbm = eirp + rx_gain - pl
    snr_db = pr_dbm - noise_dbm
    snr_lin = 10 ** (snr_db / 10.0)
    eta = 0.8
    rate = eta * BANDWIDTH_HZ * math.log2(1 + snr_lin)
    if verbose:
        print(
            f"d={d_km:.1f} km, PL={pl:.1f} dB, Pr={pr_dbm:.1f} dBm, Noise={noise_dbm:.1f} dBm, SNR={snr_db:.1f} dB, R={rate / 1e6:.2f} Mbps")
    return snr_db, rate


# --- 通用策略 + 吞吐 ---
def run_strategy_with_throughput(strategy_name, model, dev, ephem, orbit_elems, ue, t0, span_s, step_s):
    rates, ho_count = [], 0
    sel, st = build_frame(ue, ephem[t0 - 1], ephem[t0])
    cur_gid = sel[np.argmax(st[:, 0])]

    for t in range(t0, t0 + span_s, step_s):
        # --- 吞吐计算 ---
        sat_pos = ephem[t, cur_gid]
        snr_db, rate = compute_snr_and_rate(ue, sat_pos)
        rates.append(rate if snr_db > -5 else 0.0)

        # --- 策略更新 ---
        if strategy_name == "Max-Elev":
            sel, st = build_frame(ue, ephem[t - 1], ephem[t])
            best = int(np.argmax(st[:, 0]))
            nxt_gid = sel[best]
            if nxt_gid != cur_gid:
                ho_count += 1
                cur_gid = nxt_gid

        elif strategy_name == "Max-ServeTime":
            sel, st = build_frame(ue, ephem[t - 1], ephem[t])
            ttl = np.array([DT_S for _ in sel], dtype=np.float32)  # placeholder TTL
            best = int(np.argmax(ttl))
            nxt_gid = sel[best]
            if nxt_gid != cur_gid:
                ho_count += 1
                cur_gid = nxt_gid

        elif strategy_name == "OA-EST":
            sel, st = build_frame(ue, ephem[t - 1], ephem[t])
            orbit_prior = np.array([orbit_prior_vector(orbit_elems.get(f"SAT_{int(g)}", None)) for g in sel],
                                   dtype=np.float32)
            A_sp = build_spatial_affinity(st, orbit_prior).astype(np.float32)
            A_tm = build_temporal_affinity(st.shape[0], gate=getattr(CFG, 'TIME_EDGE_GATE_INIT', 0.7)).astype(
                np.float32)

            if cur_gid in sel:
                cur_local = int(np.where(sel == cur_gid)[0][0])
            else:
                cur_local = int(np.argmax(st[:, 0]))
                cur_gid = sel[cur_local]

            ue_t = torch.zeros((1, 1, _get('F_UE', 1)), dtype=torch.float32, device=dev)
            st_t = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)
            sp_t = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)
            Asp_t = torch.tensor(A_sp, dtype=torch.float32, device=dev).unsqueeze(0)
            Atm_t = torch.tensor(A_tm, dtype=torch.float32, device=dev).unsqueeze(0)

            with torch.no_grad():
                outs = model(ue_t, st_t, sp_t, Asp_t, Atm_t, torch.tensor([cur_local], device=dev))
                E = outs['energy'][0]

            best_idx = int(torch.argmin(E).item())
            if best_idx != cur_local:
                if E[best_idx].item() + HYS_MARGIN < E[cur_local].item():
                    ho_count += 1
                    cur_local = best_idx
                    cur_gid = sel[cur_local]
        else:
            raise ValueError("Unknown strategy")

    avg_rate_mbps = np.mean(rates) / 1e6
    return ho_count, avg_rate_mbps


# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/oaest_best.pt")
    ap.add_argument("--n_users", type=int, default=10)
    ap.add_argument("--span_s", type=int, default=1800)
    ap.add_argument("--step_s", type=int, default=30)
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    args = ap.parse_args()

    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        CFG.TLE_PATH, duration_s=max(3 * 3600, args.span_s + 600),
        dt_s=DT_S, limit_sats=args.limit_sats
    )
    orbit_elems = load_orbit_elements(CFG.TLE_PATH)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN_OAEST(f_ue=_get('F_UE', 1), f_sat=_get('F_SAT', 6), f_edge=_get('F_EDGE', 0),
                        hidden=_get('HIDDEN', 128), gnn_layers=_get('GNN_LAYERS', 2)).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd);
    model.eval()

    results = {"OA-EST": [], "Max-Elev": [], "Max-ServeTime": []}
    T = ephem.shape[0]

    for _ in range(args.n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(2, T - args.span_s - 2)

        for strat in results.keys():
            ho, thr = run_strategy_with_throughput(strat, model, dev, ephem, orbit_elems, ue, t0, args.span_s,
                                                   args.step_s)
            results[strat].append((ho, thr))

    # 输出结果
    for strat, vals in results.items():
        hos = [v[0] for v in vals];
        ths = [v[1] for v in vals]
        print(f"{strat:>15s} | HO mean={np.mean(hos):.2f}, Thrpt mean={np.mean(ths):.2f} Mbps")


if __name__ == "__main__":
    main()