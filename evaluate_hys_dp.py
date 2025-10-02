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
    # 路线B新增的评估/训练一致参数
    EXTRA_MARGIN_DB, TAU_TTL_STEPS,
)
from tle_ephem import build_ephemeris_from_tle
from model import PCGNN


# ===== 工具函数（与训练保持一致） =====
def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi
def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10 * math.log10(bw_hz)

def fspl_db(d_km, f_hz):
    if d_km <= 0: d_km = 0.001
    f_mhz = f_hz / 1e6
    return 20 * math.log10(d_km) + 20 * math.log10(f_mhz) + 32.44

def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km

def geodetic_to_ecef(lat, lon, alt_m=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    r = EARTH_RADIUS_M + alt_m
    return np.array([r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)], float)

def elevation_angle(ue, sat):
    ue_n  = ue / np.linalg.norm(ue)
    los   = sat - ue
    los_n = los / np.linalg.norm(los)
    cosz  = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    return 90.0 - math.degrees(math.acos(cosz))


# ===== TTL（Time-To-Leave）工具 =====
def time_to_leave_steps(ue_ecef, ephem, global_id, t_idx, max_look=1800):
    """从 t_idx 开始向前看，直到仰角跌破阈值，返回剩余可见时隙数（单位=步）。"""
    T = ephem.shape[0]
    end = min(T-1, t_idx + max_look)
    dur = 0
    for tt in range(t_idx, end):
        if elevation_angle(ue_ecef, ephem[tt, global_id]) < ELEV_MIN_DEG:
            break
        dur += 1
    return dur


# ===== 为固定 UE 构造一帧（不窥视未来），并强制保留当前服务卫星 =====
def build_frame_for_fixed_ue(ue_ecef, sat_prev, sat_t, cur_global_id=None):
    elevs_t = np.array([elevation_angle(ue_ecef, s) for s in sat_t])
    cand = np.where(elevs_t >= ELEV_MIN_DEG)[0].tolist()
    if len(cand) < TOP_K:
        order_all = np.argsort(elevs_t)[::-1]
        extra = [i for i in order_all if i not in cand]
        cand = (cand + extra)[:TOP_K]
    else:
        cand = sorted(cand, key=lambda i: elevs_t[i], reverse=True)[:TOP_K]

    # ★ 若当前服务卫星仍可见，则强制放入候选，避免被动切换
    if cur_global_id is not None and elevs_t[cur_global_id] >= ELEV_MIN_DEG:
        if cur_global_id not in cand:
            cand[-1] = cur_global_id

    sel = np.array(cand, dtype=np.int64)

    elevs_prev     = np.array([elevation_angle(ue_ecef, s) for s in sat_prev])
    sel_elev_t     = elevs_t[sel]
    sel_elev_prev  = elevs_prev[sel]
    d_elev_dt      = (sel_elev_t - sel_elev_prev) / DT_S

    rng_prev = np.linalg.norm(sat_prev[sel] - ue_ecef, axis=1) / 1000.0
    rng_now  = np.linalg.norm(sat_t[sel]   - ue_ecef, axis=1) / 1000.0
    d_rng_dt = (rng_now - rng_prev) / DT_S
    pl_now   = np.array([pathloss_db(d, CARRIER_HZ) for d in rng_now])
    pot      = sel_elev_t / 90.0

    if F_SAT >= 6:
        sat_feats = np.stack([sel_elev_t, d_elev_dt, rng_now, d_rng_dt, pl_now, pot], axis=1).astype(np.float32)
    else:
        # 兼容 F_SAT=4 的旧配置
        sat_feats = np.stack([sel_elev_t, rng_now, pl_now, pot], axis=1).astype(np.float32)

    # edge_attr = sat_feats + direction
    dirs = (sat_t[sel] - ue_ecef)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    edge_attr = np.concatenate([sat_feats, dirs.astype(np.float32)], axis=1)

    ue_feat = np.zeros((1, F_UE), np.float32)
    return ue_feat, sat_feats, edge_attr, sel


# ===== 三种策略的“下一步选择” =====
def choose_by_model(model, ue_feat, sat_feats, edge_attr):
    """返回：(best_idx_in_sel, p_stay)"""
    dev = next(model.parameters()).device
    with torch.no_grad():
        ue = torch.tensor(ue_feat, dtype=torch.float32, device=dev).unsqueeze(0)
        sa = torch.tensor(sat_feats, dtype=torch.float32, device=dev).unsqueeze(0)
        ed = torch.tensor(edge_attr, dtype=torch.float32, device=dev).unsqueeze(0)
        _, plog, _, p_stay, _ = model(ue, sa, ed)
        idx = int(torch.argmax(plog, dim=1).item())
        return idx, float(p_stay.item())

def choose_by_max_elevation(sat_feats): return int(np.argmax(sat_feats[:, 0]))

def choose_by_max_serve_time(ue_ecef, sel_global, ephem, t_idx):
    durations = [time_to_leave_steps(ue_ecef, ephem, gid, t_idx) for gid in sel_global]
    return int(np.argmax(durations))


# ===== 统计 HO 频率（核心仿真，含 A+B+SNR冗余+TTL增益+p_stay） =====
def simulate_ho_frequency(model, ephem, n_users=50, span_s=1800, step_s=30, stay_gate=0.5):
    """
    对 n_users 个随机 UE，时间窗口 span_s，每 step_s 决策一次。
    返回三种策略的 HO 次数数组（PCGNN / Max-Elevation / Max-serveTime）。
    """
    eirp  = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)
    PL_COL = 4 if F_SAT >= 6 else 2

    T = ephem.shape[0]
    span = span_s
    step = step_s

    # 将秒级 TTT 映射为时隙数
    TTT_SLOTS = max(1, int(TTT_SEC / step)) if TTT_SEC > 0 else 1
    MIN_DWELL_SLOTS = 5  # 稍保守一点（可按需改成 3~5）

    freq_model, freq_elev, freq_serv = [], [], []

    for _ in range(n_users):
        # 随机 UE 经纬度
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)

        # 随机起点，保证有 t-1 与足够 span
        t0 = np.random.randint(1, T - span - max(DELTA, 1))

        # 初始化三种策略各自的当前服务卫星
        ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_m, _pstay = choose_by_model(model, ue_f, sf_m, ea_m); cur_gid_m = sel_m[idx_m]

        ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_e = choose_by_max_elevation(sf_e);                    cur_gid_e = sel_e[idx_e]

        ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t0);   cur_gid_s = sel_s[idx_s]

        cnt_m = cnt_e = cnt_s = 0
        better_m = better_e = better_s = 0
        dwell_m = dwell_e = dwell_s = 1

        # 时间推进
        for t in range(t0 + step, t0 + span, step):
            # 各策略独立构帧（确保“当前卫星”在候选里）
            ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_m)
            ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_e)
            ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_s)

            # ========== PCGNN ==========
            nxt_idx_m, p_stay = choose_by_model(model, ue_f, sf_m, ea_m)
            nxt_gid_m = sel_m[nxt_idx_m]
            snr_all_m = eirp - sf_m[:, PL_COL] - noise

            cur_pos_m = np.where(sel_m == cur_gid_m)[0]
            if len(cur_pos_m) == 0:
                # 当前已不可见 → 被动切换
                cur_gid_m = nxt_gid_m
                cnt_m += 1
                dwell_m = 1
                better_m = 0
            else:
                cur_pos_m = int(cur_pos_m[0])
                margin_m = float(snr_all_m[nxt_idx_m] - snr_all_m[cur_pos_m])

                # TTL 增益（步数）
                ttl_cur = time_to_leave_steps(ue, ephem, cur_gid_m, t)
                ttl_nxt = time_to_leave_steps(ue, ephem, nxt_gid_m, t)
                ttl_gain = ttl_nxt - ttl_cur

                # p_stay 闸门：若模型认为应当保留，则直接不切
                if p_stay >= stay_gate:
                    better_m = 0
                    dwell_m += 1
                else:
                    # 满足 SNR 冗余才累计 TTT
                    if margin_m >= (HOM_DB + EXTRA_MARGIN_DB):
                        better_m += 1
                    else:
                        better_m = 0

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

            # ========== Max-Elevation ==========
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

                if margin_e >= (HOM_DB + EXTRA_MARGIN_DB):
                    better_e += 1
                else:
                    better_e = 0

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

            # ========== Max-serveTime ==========
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

                if margin_s >= (HOM_DB + EXTRA_MARGIN_DB):
                    better_s += 1
                else:
                    better_s = 0

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


# ===== 画图（均值±方差 + 抖点） =====
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
    plt.show()


# ===== 主函数 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--span_s", type=int, default=30*60)  # 30 分钟
    ap.add_argument("--step_s", type=int, default=30)     # 30 秒/步（与常见设置一致）
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--stay_gate", type=float, default=0.5)  # p_stay 阈值
    args = ap.parse_args()

    # 1) 星历
    print(">> Building ephemeris ...", flush=True)
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        TLE_PATH, duration_s=max(SIM_DURATION_S, args.span_s + 600), dt_s=DT_S, limit_sats=args.limit_sats
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

    # 3) 统计 HO 频率
    fm, fe, fs = simulate_ho_frequency(
        model, ephem,
        n_users=args.n_users,
        span_s=args.span_s,
        step_s=args.step_s,
        stay_gate=args.stay_gate
    )
    print(f"PCGNN:         mean={fm.mean():.2f}  std={fm.std():.2f}")
    print(f"Max-Elevation: mean={fe.mean():.2f}  std={fe.std():.2f}")
    print(f"Max-serveTime: mean={fs.mean():.2f}  std={fs.std():.2f}")

    # 4) 画图
    plot_ho_frequency(fm, fe, fs, out_png="ho_frequency.png")


if __name__ == "__main__":
    main()


# ===== Enhanced inference: dynamic hysteresis + TTT + Freeze + short-horizon DP =====
def _quantile(arr, q):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim==0: return float(arr)
    return float(np.quantile(arr, q))

def _ema(seq, alpha=0.5):
    out = []
    m = None
    for x in seq:
        m = x if m is None else alpha * x + (1-alpha) * m
        out.append(m)
    return np.array(out, dtype=float)

def _hold_over_next(diff_series, thresh, steps, hold_ratio=0.8):
    # diff_series: array length H, check proportion above thresh over next steps
    if len(diff_series) < steps: steps = len(diff_series)
    if steps<=0: return 0.0
    ok = np.sum(diff_series[:steps] > thresh)
    return ok / steps

def viterbi_dp_gain_path(snr_mat, kappa):
    """
    snr_mat is shape [H, K]: predicted (smoothed) SNR for each future step and candidate K (sel ordering).
    kappa: switching cost in 'dB equivalent' (penalize when changing satellite index between steps).
    return best_start_choice (int), best_path_indices length H (list)
    """
    H, K = snr_mat.shape
    # DP arrays
    V = np.full((H, K), -1e9, dtype=float)
    Ptr = np.full((H, K), -1, dtype=int)
    V[0] = snr_mat[0]
    Ptr[0] = -1
    for t in range(1, H):
        for j in range(K):
            # stay
            stay_scores = V[t-1, j] + snr_mat[t, j]
            # switch from i != j
            from_i = V[t-1] + snr_mat[t, j] - kappa
            best_i = int(np.argmax(from_i))
            if from_i[best_i] >= stay_scores:
                V[t, j] = from_i[best_i]
                Ptr[t, j] = best_i
            else:
                V[t, j] = stay_scores
                Ptr[t, j] = j
    # backtrack
    j_star = int(np.argmax(V[-1]))
    path = [j_star]
    for t in range(H-1, 0, -1):
        j_star = Ptr[t, j_star]
        path.append(j_star)
    path = list(reversed(path))
    return int(path[0]), path

def simulate_ho_frequency_plus(model, ephem, n_users=50, span_s=1800, step_s=30, stay_gate=0.5,
                               freeze_s=20.0, ttt_s=None, delta0_db=3.0, alpha_db_per_deg_s=0.5, beta_db=2.0,
                               dp_horizon_steps=6, dp_kappa=2.5):
    """
    Improved HO simulation using dynamic hysteresis + TTT + Freeze + DP.
    dp_horizon_steps: lookahead steps for DP (H). H=0 disables DP.
    dp_kappa: switch cost in 'dB reward' used by DP.
    """
    eirp  = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    noise = noise_power_dbm(BANDWIDTH_HZ)
    PL_COL = 4 if F_SAT >= 6 else 2

    T = ephem.shape[0]
    span = span_s
    step = step_s

    TTT_SLOTS = max(1, int((TTT_SEC if ttt_s is None else ttt_s) / step)) if (TTT_SEC if ttt_s is None else ttt_s) > 0 else 1
    MIN_DWELL_SLOTS = 3
    FREEZE_SLOTS = max(0, int(freeze_s / step))

    def dynamic_hys(elev_rate_tar, elev_rate_cur, ttl_cur_steps):
        # ttl in steps, convert to seconds-equivalent inverse scaling
        inv_ttl = (1.0 / max(ttl_cur_steps, 1e-6))
        return delta0_db + alpha_db_per_deg_s * abs(elev_rate_tar - elev_rate_cur) + beta_db * inv_ttl

    pc_counts, elev_counts, serv_counts = [], [], []

    for _ in range(n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(1, T - span - max(DELTA, 1))

        # init states
        ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        dev = next(model.parameters()).device
        with torch.no_grad():
            ue_t = torch.tensor(ue_f, dtype=torch.float32, device=dev).unsqueeze(0)
            sa_t = torch.tensor(sf_m, dtype=torch.float32, device=dev).unsqueeze(0)
            ed_t = torch.tensor(ea_m, dtype=torch.float32, device=dev).unsqueeze(0)
            psnr_t, plog_t, pho_t, p_stay_t, p_ttl_t = model(ue_t, sa_t, ed_t)
            idx_m = int(torch.argmax(plog_t, dim=1).item())
        cur_gid_m = sel_m[idx_m]
        dwell_m = 1
        better_m = 0
        freeze_left = 0
        cnt_m = 0

        # baselines
        ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        cur_gid_e = sel_e[int(np.argmax(sf_e[:,0]))]
        ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t0 - 1], ephem[t0], None)
        cur_gid_s = sel_s[choose_by_max_serve_time(ue, sel_s, ephem, t0)]
        dwell_e = dwell_s = 1
        better_e = better_s = 0
        cnt_e = cnt_s = 0

        for t in range(t0 + step, t0 + span, step):
            # Build frames per strategy
            ue_f, sf_m, ea_m, sel_m = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_m)
            ue_f, sf_e, ea_e, sel_e = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_e)
            ue_f, sf_s, ea_s, sel_s = build_frame_for_fixed_ue(ue, ephem[t - 1], ephem[t], cur_gid_s)

            # ===== PCGNN+ (ours improved) =====
            dev = next(model.parameters()).device
            with torch.no_grad():
                ue_t = torch.tensor(ue_f, dtype=torch.float32, device=dev).unsqueeze(0)
                sa_t = torch.tensor(sf_m, dtype=torch.float32, device=dev).unsqueeze(0)
                ed_t = torch.tensor(ea_m, dtype=torch.float32, device=dev).unsqueeze(0)
                psnr_t, plog_t, pho_t, p_stay_t, p_ttl_t = model(ue_t, sa_t, ed_t)
                psnr_pred = psnr_t.squeeze(0).cpu().numpy()  # [K]
                p_stay = float(p_stay_t.item())
                plog_np = plog_t.squeeze(0).cpu().numpy()
                nxt_idx_m = int(np.argmax(plog_np))
                nxt_gid_m = sel_m[nxt_idx_m]

            snr_all_m = eirp - sf_m[:, PL_COL] - noise
            cur_pos_m = np.where(sel_m == cur_gid_m)[0]
            if len(cur_pos_m) == 0:
                cur_gid_m = nxt_gid_m
                cnt_m += 1
                dwell_m = 1
                better_m = 0
                freeze_left = FREEZE_SLOTS
            else:
                cur_pos_m = int(cur_pos_m[0])
                margin_m = float(snr_all_m[nxt_idx_m] - snr_all_m[cur_pos_m])

                # compute ttl gain and elev rates for dynamic hysteresis
                ttl_cur = time_to_leave_steps(ue, ephem, cur_gid_m, t)
                ttl_tar = time_to_leave_steps(ue, ephem, nxt_gid_m, t)
                ttl_gain = ttl_tar - ttl_cur
                # elevation rate approx: use feature columns if available (elev, d_elev/dt at indices 0,1)
                delev_cur = float(sf_m[cur_pos_m,1]) if sf_m.shape[1] > 1 else 0.0
                delev_tar = float(sf_m[nxt_idx_m,1]) if sf_m.shape[1] > 1 else 0.0

                # If stay head is confident and ttl is decent, stay
                if (p_stay >= stay_gate) and (ttl_cur >= max(1, DELTA)):
                    better_m = 0
                    dwell_m += 1
                    if freeze_left>0: freeze_left -= 1
                else:
                    # dynamic hysteresis
                    delta_hys = dynamic_hys(delev_tar, delev_cur, ttl_cur)
                    # count TTT if margin above hysteresis + base HOM_DB
                    if margin_m >= (HOM_DB + delta_hys):
                        better_m += 1
                    else:
                        better_m = 0

                    # DP lookahead (optional)
                    do_dp = (dp_horizon_steps is not None) and (dp_horizon_steps > 0)
                    dp_choice = nxt_idx_m
                    if do_dp:
                        # approximate future SNRs: use current psnr prediction as base and persistence
                        snr0 = psnr_pred
                        sm = _ema(snr0, alpha=0.6)
                        snr_mat = np.tile(sm[None,:], (int(dp_horizon_steps), 1))
                        dp_choice, _ = viterbi_dp_gain_path(snr_mat, kappa=dp_kappa)
                    # apply freeze timer
                    if freeze_left > 0:
                        dwell_m += 1
                        freeze_left -= 1
                    else:
                        target_idx = dp_choice
                        target_gid = sel_m[target_idx]
                        if (better_m >= TTT_SLOTS and dwell_m >= MIN_DWELL_SLOTS and ttl_gain >= 0):
                            if target_gid != cur_gid_m:
                                cnt_m += 1
                                cur_gid_m = target_gid
                                dwell_m = 1
                                better_m = 0
                                freeze_left = FREEZE_SLOTS
                            else:
                                dwell_m += 1
                        else:
                            dwell_m += 1

            # ===== Baselines (unchanged) =====
            # Max-Elevation
            nxt_idx_e = int(np.argmax(sf_e[:,0])); nxt_gid_e = sel_e[nxt_idx_e]
            snr_all_e = eirp - sf_e[:, PL_COL] - noise
            cur_pos_e = np.where(sel_e == cur_gid_e)[0]
            if len(cur_pos_e) == 0:
                cur_gid_e = nxt_gid_e; cnt_e += 1; dwell_e = 1; better_e = 0
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
                    if (better_e >= TTT_SLOTS and dwell_e >= MIN_DWELL_SLOTS):
                        cnt_e += 1; cur_gid_e = nxt_gid_e; dwell_e = 1; better_e = 0
                    else:
                        dwell_e += 1

            # Max-ServeTime
            nxt_idx_s = choose_by_max_serve_time(ue, sel_s, ephem, t)
            nxt_gid_s = sel_s[nxt_idx_s]
            snr_all_s = eirp - sf_s[:, PL_COL] - noise
            cur_pos_s = np.where(sel_s == cur_gid_s)[0]
            if len(cur_pos_s) == 0:
                cur_gid_s = nxt_gid_s; cnt_s += 1; dwell_s = 1; better_s = 0
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
                    if (better_s >= TTT_SLOTS and dwell_s >= MIN_DWELL_SLOTS):
                        cnt_s += 1; cur_gid_s = nxt_gid_s; dwell_s = 1; better_s = 0
                    else:
                        dwell_s += 1

        pc_counts.append(cnt_m); elev_counts.append(cnt_e); serv_counts.append(cnt_s)

    return np.array(pc_counts), np.array(elev_counts), np.array(serv_counts)


def main_plus():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--span_s", type=int, default=30*60)
    ap.add_argument("--step_s", type=int, default=30)
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--stay_gate", type=float, default=0.5)
    # new controls
    ap.add_argument("--freeze_s", type=float, default=20.0)
    ap.add_argument("--delta0_db", type=float, default=3.0)
    ap.add_argument("--alpha_db_per_deg_s", type=float, default=0.5)
    ap.add_argument("--beta_db", type=float, default=2.0)
    ap.add_argument("--ttt_s", type=float, default=None)
    ap.add_argument("--dp_horizon_steps", type=int, default=6)
    ap.add_argument("--dp_kappa", type=float, default=2.5)
    args = ap.parse_args()

    # 1) Ephemeris
    print(">> Building ephemeris ...", flush=True)
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        TLE_PATH, duration_s=max(SIM_DURATION_S, args.span_s + 600), dt_s=DT_S, limit_sats=args.limit_sats
    )
    print(f">> ephem: T={ephem.shape[0]}, Nsat={ephem.shape[1]}", flush=True)

    # 2) Model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from model import PCGNN
    model = PCGNN(F_UE, F_SAT, F_EDGE, hidden=HIDDEN, layers=GNN_LAYERS, graph_topk=GRAPH_TOPK, adj_tau=ADJ_TAU).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd); model.eval()

    # 3) Run improved simulation
    fm, fe, fs = simulate_ho_frequency_plus(
        model, ephem,
        n_users=args.n_users, span_s=args.span_s, step_s=args.step_s, stay_gate=args.stay_gate,
        freeze_s=args.freeze_s, ttt_s=args.ttt_s, delta0_db=args.delta0_db,
        alpha_db_per_deg_s=args.alpha_db_per_deg_s, beta_db=args.beta_db,
        dp_horizon_steps=args.dp_horizon_steps, dp_kappa=args.dp_kappa
    )
    print(f"PCGNN+        mean={fm.mean():.2f}  std={fm.std():.2f}")
    print(f"Max-Elevation mean={fe.mean():.2f}  std={fe.std():.2f}")
    print(f"Max-serveTime mean={fs.mean():.2f}  std={fs.std():.2f}")

    plot_ho_frequency(fm, fe, fs, out_png="ho_frequency_plus.png")
