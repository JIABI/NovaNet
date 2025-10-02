# evaluate.py —— PCGNN 评估：A) 保留当前卫星进候选 + B) TTT/驻留 + SNR冗余 + TTL增益 + p_stay闸门

# evaluate.py
# PCGNN 评估：handover 频次 + 切换总时延 + 传输时延
# 路线B：A) 强制保留当前卫星进候选；B) TTT + 最小驻留 + SNR冗余 + TTL增益 + p_stay 闸门

# evaluate_hys_dp.py — dynamic hysteresis + TTT + freeze + short-horizon DP

# evaluate_hys_dp.py — dynamic hysteresis + TTT + freeze + short-horizon DP

import argparse, math, numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from config import (
    # 基本/星历
    TLE_PATH, SIM_DURATION_S, DT_S, DELTA, TOP_K, ELEV_MIN_DEG,
    # 模型维度
    F_UE, F_SAT, F_EDGE, HIDDEN, GNN_LAYERS, GRAPH_TOPK, ADJ_TAU,
    # 物理链路
    SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI, BANDWIDTH_HZ, CARRIER_HZ,
    NOISE_PSD_DBM_HZ, SMALL_SCALE_FADING_DB, ATTEN_DB_PER_KM,
    EARTH_RADIUS_M, HOM_DB, TTT_SEC,
    # 评估规模
    LIMIT_SATS,
    # 训练/评估共享
    EXTRA_MARGIN_DB, TAU_TTL_STEPS,
    # Anti-ping-pong (inference)
    FREEZE_S, DELTA0_DB, ALPHA_DB_PER_DEG_S, BETA_DB, DP_HORIZON_STEPS, DP_KAPPA
)
from tle_ephem import build_ephemeris_from_tle
from model import PCGNN

# ---------- Link helpers ----------
def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi
def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)
def fspl_db(d_km, f_hz): return 32.45 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(f_hz/1e6)
def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km

# ---------- Geometry ----------
def elevation_angle(ue_ecef, sat_ecef):
    ue_n = ue_ecef / np.linalg.norm(ue_ecef)
    los  = sat_ecef - ue_ecef
    los_n= los / np.linalg.norm(los)
    cosz = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    return 90.0 - math.degrees(math.acos(cosz))

def time_to_leave_steps(ue_ecef, ephem, global_id, t_idx, max_look=1800):
    T = ephem.shape[0]
    end = min(T-1, t_idx + max_look)
    steps = 0
    for tt in range(t_idx, end):
        if elevation_angle(ue_ecef, ephem[tt, global_id]) < ELEV_MIN_DEG:
            break
        steps += 1
    return steps

# ---------- Frame builder (force current sat included) ----------
def build_frame_for_fixed_ue(ue_ecef, sat_prev, sat_t, cur_global_id=None):
    elevs = np.array([elevation_angle(ue_ecef, s) for s in sat_t])
    cand = np.where(elevs >= ELEV_MIN_DEG)[0].tolist()
    if len(cand) < TOP_K:
        order_all = np.argsort(elevs)[::-1]
        extra = [i for i in order_all if i not in cand]
        cand = (cand + extra)[:TOP_K]
    else:
        cand = sorted(cand, key=lambda i: elevs[i], reverse=True)[:TOP_K]

    # force include current
    if cur_global_id is not None and cur_global_id not in cand:
        cand[-1] = cur_global_id

    sel = np.array(cand, dtype=int)              # [K]
    # build sat feature per-candidate: [elev, d_elev, range_km, d_range, pathloss, potential]
    feat = []
    for gid in sel:
        elev = elevation_angle(ue_ecef, sat_t[gid])
        elev_prev = elevation_angle(ue_ecef, sat_prev[gid])
        d_elev = (elev - elev_prev) / max(1.0, DT_S)
        d = np.linalg.norm(sat_t[gid] - ue_ecef) / 1000.0  # km
        d_prev = np.linalg.norm(sat_prev[gid] - ue_ecef) / 1000.0
        dd = (d - d_prev) / max(1e-6, DT_S)
        pl = pathloss_db(d, CARRIER_HZ)
        pot = elev / 90.0
        feat.append([elev, d_elev, d, dd, pl, pot])
    sat_feats = np.array(feat, dtype=np.float32)  # [K,Fs]
    ue_feat  = np.zeros((1, F_UE), dtype=np.float32)  # 这里按你数据集保持 1xF_UE
    edge     = np.zeros((TOP_K, F_EDGE), dtype=np.float32)  # 若有边特征可按需替换
    return ue_feat, sat_feats, edge, sel

def choose_by_max_serve_time(ue_ecef, sel, ephem, t_idx):
    ttls = [time_to_leave_steps(ue_ecef, ephem, int(g), t_idx) for g in sel]
    return int(np.argmax(ttls))

# ---------- Plot ----------
def plot_ho_frequency(pc_counts, elev_counts, serv_counts, out_png="ho_frequency_plus.png"):
    x = ["PCGNN+", "Max-Elev", "Max-ServeTime"]
    y = [pc_counts.mean(), elev_counts.mean(), serv_counts.mean()]
    std = [pc_counts.std(), elev_counts.std(), serv_counts.std()]
    plt.figure(figsize=(5,4))
    bars = plt.bar(x, y, yerr=std, capsize=4)
    plt.ylabel("Handover count per user / span")
    plt.title("HO frequency comparison")
    for b, v in zip(bars, y):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------- DP helper ----------
def viterbi_dp_gain_path(snr_mat, kappa):
    """
    snr_mat: [H, K] smoothed 'reward' per step; kappa: switch cost.
    return: start_choice(int), path(list of int)
    """
    H, K = snr_mat.shape
    V  = np.full((H, K), -1e9, dtype=float)
    Ptr= np.full((H, K), -1, dtype=int)
    V[0] = snr_mat[0]
    for t in range(1, H):
        for j in range(K):
            stay = V[t-1, j] + snr_mat[t, j]
            cand = V[t-1] + snr_mat[t, j] - kappa
            i = int(np.argmax(cand))
            if cand[i] >= stay:
                V[t, j] = cand[i]; Ptr[t, j] = i
            else:
                V[t, j] = stay;    Ptr[t, j] = j
    j = int(np.argmax(V[-1])); path = [j]
    for t in range(H-1, 0, -1):
        j = Ptr[t, j]; path.append(j)
    path.reverse()
    return int(path[0]), path

# ---------- Simulator (PCGNN+) ----------
def simulate_ho_frequency_plus(model, ephem, n_users=50, span_s=1800, step_s=30,
                               stay_gate=0.5, freeze_s=FREEZE_S, ttt_s=None,
                               delta0_db=DELTA0_DB, alpha_db_per_deg_s=ALPHA_DB_PER_DEG_S, beta_db=BETA_DB,
                               dp_horizon_steps=DP_HORIZON_STEPS, dp_kappa=DP_KAPPA):
    eirp  = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)
    PL_COL = 4 if F_SAT >= 6 else 2

    TTT = TTT_SEC if ttt_s is None else ttt_s
    TTT_SLOTS = max(1, int(TTT / step_s)) if (TTT and TTT > 0) else 1
    MIN_DWELL_SLOTS = 3
    FREEZE_SLOTS = max(0, int(freeze_s / step_s))

    def dyn_hys(delev_tar, delev_cur, ttl_cur_steps):
        inv_ttl = 1.0 / max(ttl_cur_steps, 1e-6)
        return delta0_db + alpha_db_per_deg_s * abs(delev_tar - delev_cur) + beta_db * inv_ttl

    pc_counts, elev_counts, serv_counts = [], [], []
    T = ephem.shape[0]
    for _ in range(n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        # quick ECEF on a sphere: (for eval only)
        R = EARTH_RADIUS_M
        ue = np.array([
            R * math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
            R * math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
            R * math.sin(math.radians(lat))
        ], dtype=np.float64)

        t0 = np.random.randint(1, T - span_s - max(DELTA, 1))
        # init (PCGNN+)
        ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t0-1], ephem[t0], None)
        dev = next(model.parameters()).device
        with torch.no_grad():
            ue_t = torch.tensor(ue_f, dtype=torch.float32, device=dev).unsqueeze(0)
            sa_t = torch.tensor(sf_m, dtype=torch.float32, device=dev).unsqueeze(0)
            ed_t = torch.tensor(ea_m, dtype=torch.float32, device=dev).unsqueeze(0)
            psnr_t, plog_t, pho_t, p_stay_t, p_ttl_t = model(ue_t, sa_t, ed_t)
            idx0 = int(torch.argmax(plog_t, dim=1).item())
        cur_gid_m = sel_m[idx0]; dwell_m=1; better_m=0; freeze_left=0; cnt_m=0

        # baselines init
        ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t0-1], ephem[t0], None)
        cur_gid_e = sel_e[int(np.argmax(sf_e[:,0]))]
        ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t0-1], ephem[t0], None)
        cur_gid_s = sel_s[choose_by_max_serve_time(ue, sel_s, ephem, t0)]
        dwell_e=dwell_s=1; better_e=better_s=0; cnt_e=cnt_s=0

        for t in range(t0 + step_s, t0 + span_s, step_s):
            # build frames
            ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t-1], ephem[t], cur_gid_m)
            ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t-1], ephem[t], cur_gid_e)
            ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t-1], ephem[t], cur_gid_s)

            # ---- PCGNN+ ----
            with torch.no_grad():
                ue_t = torch.tensor(ue_f, dtype=torch.float32, device=dev).unsqueeze(0)
                sa_t = torch.tensor(sf_m, dtype=torch.float32, device=dev).unsqueeze(0)
                ed_t = torch.tensor(ea_m, dtype=torch.float32, device=dev).unsqueeze(0)
                psnr_t, plog_t, pho_t, p_stay_t, p_ttl_t = model(ue_t, sa_t, ed_t)
                psnr_pred = psnr_t.squeeze(0).cpu().numpy()  # [K]
                p_stay = float(p_stay_t.item())
                nxt_idx_m = int(torch.argmax(plog_t, dim=1).item())
                nxt_gid_m = sel_m[nxt_idx_m]

            snr_all_m = eirp - sf_m[:, PL_COL] - noise
            cur_pos = np.where(sel_m == cur_gid_m)[0]
            if len(cur_pos) == 0:
                # current not in candidate: force switch
                cur_gid_m = nxt_gid_m; cnt_m += 1; dwell_m=1; better_m=0; freeze_left=FREEZE_SLOTS
            else:
                cur_pos = int(cur_pos[0])
                margin = float(snr_all_m[nxt_idx_m] - snr_all_m[cur_pos])
                ttl_cur = time_to_leave_steps(ue, ephem, cur_gid_m, t)
                ttl_tar = time_to_leave_steps(ue, ephem, nxt_gid_m, t)
                ttl_gain = ttl_tar - ttl_cur
                delev_cur = float(sf_m[cur_pos,1]) if sf_m.shape[1] > 1 else 0.0
                delev_tar = float(sf_m[nxt_idx_m,1]) if sf_m.shape[1] > 1 else 0.0

                if (p_stay >= stay_gate) and (ttl_cur >= max(1, DELTA)):
                    better_m = 0; dwell_m += 1
                    if freeze_left > 0: freeze_left -= 1
                else:
                    delta_hys = dyn_hys(delev_tar, delev_cur, ttl_cur)
                    if margin >= (HOM_DB + delta_hys):
                        better_m += 1
                    else:
                        better_m = 0

                    # short-horizon DP (optional)
                    dp_choice = nxt_idx_m
                    if dp_horizon_steps and dp_horizon_steps > 0:
                        sm = np.copy(psnr_pred)
                        sm = 0.6 * sm + 0.4 * (sm.mean())   # simple smooth
                        snr_mat = np.tile(sm[None, :], (int(dp_horizon_steps), 1))
                        dp_choice, _ = viterbi_dp_gain_path(snr_mat, kappa=dp_kappa)

                    if freeze_left > 0:
                        dwell_m += 1; freeze_left -= 1
                    else:
                        target_idx = int(dp_choice)
                        target_gid = sel_m[target_idx]
                        if (better_m >= TTT_SLOTS and dwell_m >= 3 and ttl_gain >= 0):
                            if target_gid != cur_gid_m:
                                cnt_m += 1; cur_gid_m = target_gid; dwell_m=1; better_m=0; freeze_left=FREEZE_SLOTS
                            else:
                                dwell_m += 1
                        else:
                            dwell_m += 1

            # ---- Baseline: Max-Elev ----
            nxt_idx_e = int(np.argmax(sf_e[:,0])); nxt_gid_e = sel_e[nxt_idx_e]
            snr_all_e = eirp - sf_e[:, PL_COL] - noise
            cur_pos_e = np.where(sel_e == cur_gid_e)[0]
            if len(cur_pos_e) == 0:
                cur_gid_e = nxt_gid_e; cnt_e += 1; dwell_e=1; better_e=0
            else:
                cur_pos_e = int(cur_pos_e[0])
                margin_e = float(snr_all_e[nxt_idx_e] - snr_all_e[cur_pos_e])
                if margin_e >= (HOM_DB + EXTRA_MARGIN_DB):
                    better_e += 1
                else:
                    better_e = 0
                if nxt_gid_e == cur_gid_e:
                    dwell_e += 1
                else:
                    if (better_e >= TTT_SLOTS and dwell_e >= 3):
                        cnt_e += 1; cur_gid_e = nxt_gid_e; dwell_e=1; better_e=0
                    else:
                        dwell_e += 1

            # ---- Baseline: Max-ServeTime ----
            nxt_idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t)
            nxt_gid_s = sel_s[nxt_idx_s]
            snr_all_s = eirp - sf_s[:, PL_COL] - noise
            cur_pos_s = np.where(sel_s == cur_gid_s)[0]
            if len(cur_pos_s) == 0:
                cur_gid_s = nxt_gid_s; cnt_s += 1; dwell_s=1; better_s=0
            else:
                cur_pos_s = int(cur_pos_s[0])
                margin_s = float(snr_all_s[nxt_idx_s] - snr_all_s[cur_pos_s])
                if margin_s >= (HOM_DB + EXTRA_MARGIN_DB):
                    better_s += 1
                else:
                    better_s = 0
                if nxt_gid_s == cur_gid_s:
                    dwell_s += 1
                else:
                    if (better_s >= TTT_SLOTS and dwell_s >= 3):
                        cnt_s += 1; cur_gid_s = nxt_gid_s; dwell_s=1; better_s=0
                    else:
                        dwell_s += 1

        pc_counts.append(cnt_m); elev_counts.append(cnt_e); serv_counts.append(cnt_s)

    return np.array(pc_counts), np.array(elev_counts), np.array(serv_counts)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--span_s", type=int, default=30*60)
    ap.add_argument("--step_s", type=int, default=30)
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--stay_gate", type=float, default=0.5)
    # allow override but default to config
    ap.add_argument("--freeze_s", type=float, default=FREEZE_S)
    ap.add_argument("--delta0_db", type=float, default=DELTA0_DB)
    ap.add_argument("--alpha_db_per_deg_s", type=float, default=ALPHA_DB_PER_DEG_S)
    ap.add_argument("--beta_db", type=float, default=BETA_DB)
    ap.add_argument("--ttt_s", type=float, default=None)
    ap.add_argument("--dp_horizon_steps", type=int, default=DP_HORIZON_STEPS)
    ap.add_argument("--dp_kappa", type=float, default=DP_KAPPA)
    args = ap.parse_args()

    print(">> Building ephemeris ...", flush=True)
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        TLE_PATH, duration_s=max(SIM_DURATION_S, args.span_s + 600), dt_s=DT_S, limit_sats=args.limit_sats
    )
    print(f">> ephem: T={ephem.shape[0]}, Nsat={ephem.shape[1]}", flush=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN(f_ue=F_UE, f_sat=F_SAT, f_edge=F_EDGE,
                  hidden=HIDDEN, gnn_layers=GNN_LAYERS,
                  graph_topk=GRAPH_TOPK, adj_tau=ADJ_TAU).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd); model.eval()

    fm, fe, fs = simulate_ho_frequency_plus(
        model, ephem,
        n_users=args.n_users, span_s=args.span_s, step_s=args.step_s, stay_gate=args.stay_gate,
        freeze_s=args.freeze_s, ttt_s=args.ttt_s,
        delta0_db=args.delta0_db, alpha_db_per_deg_s=args.alpha_db_per_deg_s, beta_db=args.beta_db,
        dp_horizon_steps=args.dp_horizon_steps, dp_kappa=args.dp_kappa
    )
    print(f"PCGNN+        mean={fm.mean():.2f}  std={fm.std():.2f}")
    print(f"Max-Elevation mean={fe.mean():.2f}  std={fe.std():.2f}")
    print(f"Max-ServeTime mean={fs.mean():.2f}  std={fs.std():.2f}")
    plot_ho_frequency(fm, fe, fs, out_png="ho_frequency_plus.png")

if __name__ == "__main__":
    main()
