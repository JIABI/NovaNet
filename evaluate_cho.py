# evaluate_cho.py — CHO failure & average-rate comparison for 2400 s
import math, argparse, numpy as np, torch
import matplotlib.pyplot as plt
import importlib
CFG = importlib.import_module('config')

def _get(name, default): return getattr(CFG, name, default)

# ===== 基本参数（可在 config.py 覆盖）=====
DT_S         = _get('DT_S', 30)
TOP_K        = _get('TOP_K', 8)
ELEV_MIN_DEG = _get('ELEV_MIN_DEG', 10.0)
LIMIT_SATS   = _get('LIMIT_SATS', 32)

# CHO 参数
CHO_TARGET_M_LIST = _get('CHO_TARGET_M_LIST', [1, 2, 3])
CHO_TTT_STEPS     = _get('CHO_TTT_STEPS', 2)     # 触发需持续的步数
CHO_EXEC_STEPS    = _get('CHO_EXEC_STEPS', 1)    # 执行时延（步）；执行期吞吐=0
CHO_HYS_DB        = _get('CHO_HYS_DB', 1.5)      # 触发门限
CHO_MIN_SNR_DB    = _get('CHO_MIN_SNR_DB', -5.0) # 目标最低 SNR

# 吞吐计算
BANDWIDTH_HZ = _get('BANDWIDTH_HZ', 20e6)
EFFICIENCY   = _get('EFFICIENCY', 0.75)          # 实现效率（<=1），考虑编码/协议损耗
MIN_SNR_DB   = _get('MIN_SNR_DB', -10.0)         # 低于此门限吞吐置0

PLOT_OUT_HOF   = _get('CHO_PLOT_OUT_HOF',   'cho_hof_compare.png')
PLOT_OUT_RATE  = _get('CHO_PLOT_OUT_RATE',  'cho_rate_compare.png')

# 物理参数（与 evaluate_oaest.py 保持一致）
SAT_TX_POWER_DBM   = _get('SAT_TX_POWER_DBM', 20.0)
SAT_ANT_GAIN_DBI   = _get('SAT_ANT_GAIN_DBI', 30.0)
NOISE_PSD_DBM_HZ   = _get('NOISE_PSD_DBM_HZ', -174.0)
ATTEN_DB_PER_KM    = _get('ATTEN_DB_PER_KM', 0.0)
SMALL_SCALE_FADING_DB = _get('SMALL_SCALE_FADING_DB', 1.5)
CARRIER_HZ         = _get('CARRIER_HZ', 12e9)
EARTH_RADIUS_M     = _get('EARTH_RADIUS_M', 6371000.0)

from tle_ephem import build_ephemeris_from_tle
from orbit_feats import load_orbit_elements, orbit_prior_vector, build_spatial_affinity, build_temporal_affinity
from model_oaest import PCGNN_OAEST

# ============== 常用函数 ==============
def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi
def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0*math.log10(bw_hz)
EIRP = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
NOISE = noise_power_dbm(BANDWIDTH_HZ)

def fspl_db(d_km, f_hz): return 32.45 + 20.0*math.log10(max(d_km,1e-6)) + 20.0*math.log10(f_hz/1e6)
def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM*d_km

def elevation_angle(ue_ecef, sat_ecef):
    ue_n = ue_ecef / np.linalg.norm(ue_ecef)
    los  = sat_ecef - ue_ecef
    los_n= los / np.linalg.norm(los)
    cosz = float(np.clip(np.dot(los_n, ue_n), -1.0, 1.0))
    return 90.0 - math.degrees(math.acos(cosz))

def geodetic_to_ecef(lat, lon, alt_m=0.0):
    R = EARTH_RADIUS_M + alt_m
    clat, slat = math.cos(math.radians(lat)), math.sin(math.radians(lat))
    clon, slon = math.cos(math.radians(lon)), math.sin(math.radians(lon))
    x = R * clat * clon; y = R * clat * slon; z = R * slat
    return np.array([x,y,z], dtype=np.float64)

def time_to_leave_steps(ue_ecef, ephem, gid, t_idx, elev_thr=ELEV_MIN_DEG, max_look=3600):
    T = ephem.shape[0]; end = min(T-1, t_idx + max_look)
    steps = 0
    for tt in range(t_idx, end):
        if elevation_angle(ue_ecef, ephem[tt, gid]) < elev_thr: break
        steps += 1
    return steps

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

def snr_from_gid(ue, ephem, t, gid):
    d = np.linalg.norm(ephem[t, int(gid)] - ue)/1000.0
    pl = pathloss_db(d, CARRIER_HZ)
    return float(EIRP - pl - NOISE)

def snr_from_feats(ue, ephem, t, gids):
    return np.array([snr_from_gid(ue, ephem, t, g) for g in gids], dtype=np.float32)

def rate_mbps_from_snr_db(snr_db):
    if snr_db < MIN_SNR_DB: return 0.0
    snr_lin = 10.0**(snr_db/10.0)
    bits_per_s = EFFICIENCY * BANDWIDTH_HZ * math.log2(1.0 + snr_lin)
    return bits_per_s / 1e6

# ============== 目标集合生成：策略实现 ==============
def target_set_oaest(model, dev, ue, ephem, orbit_elems, t, sel, st, M):
    orbit_prior = np.array([orbit_prior_vector(orbit_elems.get(f"SAT_{int(g)}", None)) for g in sel], dtype=np.float32)
    A_sp = build_spatial_affinity(st, orbit_prior).astype(np.float32)
    A_tm = build_temporal_affinity(st.shape[0], gate=getattr(CFG, 'TIME_EDGE_GATE_INIT', 0.7)).astype(np.float32)
    cur_local = int(np.argmax(st[:,0]))  # 仅用于能量评估的参考
    ue_t  = torch.zeros((1,1,_get('F_UE',1)), dtype=torch.float32, device=dev)
    st_t  = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)
    sp_t  = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)
    Asp_t = torch.tensor(A_sp, dtype=torch.float32, device=dev).unsqueeze(0)
    Atm_t = torch.tensor(A_tm, dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        E = model(ue_t, st_t, sp_t, Asp_t, Atm_t, torch.tensor([cur_local], device=dev))['energy'][0].cpu().numpy()
    idx = np.argsort(E)[:M]
    return sel[idx], idx  # 返回全局ID与局部索引

def target_set_max_elev(sel, st, M):
    idx = np.argsort(-st[:,0])[:M]
    return sel[idx], idx

def target_set_max_servetime(ue, ephem, t, sel, M):
    ttl = np.array([time_to_leave_steps(ue, ephem, int(g), t) for g in sel], dtype=np.float32)
    idx = np.argsort(-ttl)[:M]
    return sel[idx], idx

# ======== 预留：你后续把这三个函数改成真实实现即可 ========
def target_set_gnn(sel, st, M):
    raise NotImplementedError("TODO: implement pure-GNN target set selection")

def target_set_pointcloud(sel, st, M):
    raise NotImplementedError("TODO: implement PointCloud-based target set selection")

def target_set_dqn_gnn(sel, st, M):
    raise NotImplementedError("TODO: implement DQN+GNN target set selection")

# ============== 单用户：按某策略统计 2400s 的 HOF & 平均吞吐 ==============
def run_user_cho(ephem, orbit_elems, ue, t0, span_s, step_s, strategy, M, model=None, dev=None):
    # 初始 serving
    sel, st = build_frame(ue, ephem[t0-1], ephem[t0])
    cur_local = int(np.argmax(st[:,0])); cur_gid = sel[cur_local]
    T_end = t0 + span_s

    hof = 0; attempts = 0
    ttt_cnt = 0
    pending_target_gid = None
    exec_left = 0

    # 速率累计
    rate_sum = 0.0; n_steps = 0

    t = t0
    while t < T_end:
        sel, st = build_frame(ue, ephem[t-1], ephem[t])
        # 若当前不在候选里，重选为 elevation 最大
        if cur_gid in sel:
            cur_local = int(np.where(sel == cur_gid)[0][0])
        else:
            cur_local = int(np.argmax(st[:,0])); cur_gid = sel[cur_local]

        # 计算 SNR & 速率
        snr_all = snr_from_feats(ue, ephem, t, sel)  # [K]
        snr_cur = float(snr_all[cur_local])

        # （1）执行窗口：暂不传输
        if exec_left > 0:
            exec_left -= 1
            rate_sum += 0.0
            n_steps  += 1
            if exec_left == 0:
                # 到执行结束时刻检查目标是否仍满足
                if pending_target_gid in sel:
                    j = int(np.where(sel == pending_target_gid)[0][0])
                    if snr_all[j] >= CHO_MIN_SNR_DB:
                        # 成功：切换
                        cur_gid = pending_target_gid
                        cur_local = j
                    else:
                        hof += 1
                else:
                    hof += 1
                pending_target_gid = None
            t += step_s
            continue

        # （2）生成 CHO 目标集合 S
        if strategy == "OA-EST":
            tgts_gid, tgts_idx = target_set_oaest(model, dev, ue, ephem, orbit_elems, t, sel, st, M)
        elif strategy == "Maximum-Elevation":
            tgts_gid, tgts_idx = target_set_max_elev(sel, st, M)
        elif strategy == "Maximum-serveTime":
            tgts_gid, tgts_idx = target_set_max_servetime(ue, ephem, t, sel, M)
        elif strategy == "GNN":
            tgts_gid, tgts_idx = target_set_gnn(sel, st, M)             # TODO
        elif strategy == "PointCloud":
            tgts_gid, tgts_idx = target_set_pointcloud(sel, st, M)      # TODO
        elif strategy == "DQN+GNN":
            tgts_gid, tgts_idx = target_set_dqn_gnn(sel, st, M)         # TODO
        else:
            raise ValueError("unknown strategy")

        # （3）CHO 触发检测（取 margin 最大的目标）
        margins = snr_all[tgts_idx] - snr_cur
        ok = (snr_all[tgts_idx] >= CHO_MIN_SNR_DB) & (margins >= CHO_HYS_DB)
        if np.any(ok):
            best_local = tgts_idx[np.argmax(margins)]
            ttt_cnt += 1
            if ttt_cnt >= CHO_TTT_STEPS:
                attempts += 1
                pending_target_gid = sel[best_local]
                exec_left = CHO_EXEC_STEPS
                ttt_cnt = 0
        else:
            ttt_cnt = 0

        # （4）当前步的吞吐（未进入执行窗口的正常传输）
        rate_sum += rate_mbps_from_snr_db(snr_cur)
        n_steps  += 1

        t += step_s

    avg_rate_mbps = rate_sum / max(1, n_steps)
    return hof, attempts, avg_rate_mbps

# ============== 主流程：多用户 & 画图 ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/oaest_best.pt")
    ap.add_argument("--n_users", type=int, default=60)
    ap.add_argument("--span_s", type=int, default=2400)  # 2400 s
    ap.add_argument("--step_s", type=int, default=30)
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--out_hof",  default=PLOT_OUT_HOF)
    ap.add_argument("--out_rate", default=PLOT_OUT_RATE)
    args = ap.parse_args()

    # 星历
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(
        CFG.TLE_PATH, duration_s=max(3*3600, args.span_s+600),
        dt_s=DT_S, limit_sats=args.limit_sats)
    orbit_elems = load_orbit_elements(CFG.TLE_PATH)

    # 模型
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN_OAEST(f_ue=_get('F_UE',1), f_sat=_get('F_SAT',6), f_edge=_get('F_EDGE',0),
                        hidden=_get('HIDDEN',128), gnn_layers=_get('GNN_LAYERS',2)).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd); model.eval()

    # 策略集合（含占位）
    strategies = ["OA-EST", "Maximum-Elevation", "Maximum-serveTime",
                  "GNN", "PointCloud", "DQN+GNN"]  # 后三项为占位，未实现会跳过

    results_hof  = {M: {} for M in CHO_TARGET_M_LIST}
    results_rate = {M: {} for M in CHO_TARGET_M_LIST}

    T = ephem.shape[0]
    for _ in range(args.n_users):
        lat, lon = np.random.uniform(-60,60), np.random.uniform(-180,180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(2, T-args.span_s-2)

        for M in CHO_TARGET_M_LIST:
            for s in strategies:
                try:
                    hof, attempts, avg_rate = run_user_cho(
                        ephem, orbit_elems, ue, t0, args.span_s, args.step_s,
                        s, M, model=model, dev=dev)
                except NotImplementedError:
                    # 占位策略：跳过
                    continue
                results_hof[M].setdefault(s, []).append(hof)
                results_rate[M].setdefault(s, []).append(avg_rate)

    # 打印统计
    for M in CHO_TARGET_M_LIST:
        print(f"\n--- CHO Target Set Size M={M} ---")
        for s in strategies:
            if s not in results_hof[M]: continue
            arr_h = np.array(results_hof[M][s], dtype=np.float32)
            arr_r = np.array(results_rate[M][s], dtype=np.float32)
            print(f"{s:>18s} | HOF mean={arr_h.mean():.2f}, std={arr_h.std():.2f} | "
                  f"Rate mean={arr_r.mean():.2f} Mbps, std={arr_r.std():.2f}")

    # ---------- 画图：HOF ----------
    groups = CHO_TARGET_M_LIST
    width = 0.13
    fig, ax = plt.subplots(figsize=(10.5,4.6))
    x = np.arange(len(groups))
    palette = {
        "OA-EST":"#e74c3c", "Maximum-Elevation":"#27ae60", "Maximum-serveTime":"#8e44ad",
        "GNN":"#2c3e50", "PointCloud":"#95a5a6", "DQN+GNN":"#f39c12"
    }
    order = [s for s in strategies if any(s in results_hof[M] for M in groups)]
    for i, s in enumerate(order):
        means = [np.mean(results_hof[M][s]) for M in groups]
        stds  = [np.std(results_hof[M][s])  for M in groups]
        ax.bar(x + (i-(len(order)-1)/2)*width, means, width=width, yerr=stds, capsize=4,
               label=s, color=palette.get(s, None), alpha=0.92, edgecolor='black', linewidth=0.8)
        # jitter scatter
        rng = np.random.RandomState(0)
        for j, M in enumerate(groups):
            arr = np.array(results_hof[M][s])
            jitter = (rng.rand(len(arr))-0.5)*(width*0.9)
            ax.scatter(np.full_like(arr, x[j] + (i-(len(order)-1)/2)*width)+jitter, arr,
                       s=20, alpha=0.85, edgecolors='white', linewidths=0.3)
        # annotate mean
        for j, m in enumerate(means):
            ax.text(x[j] + (i-(len(order)-1)/2)*width, m - 0.05*max(1.0, ax.get_ylim()[1]),
                    f"{m:.1f}", ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={M}" for M in groups])
    ax.set_ylabel("HO Failures (2400 s)")
    ax.set_xlabel("CHO Target Set Size")
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(args.out_hof, dpi=300)
    plt.close()

    # ---------- 画图：Average Rate ----------
    fig, ax = plt.subplots(figsize=(10.5,4.6))
    order = [s for s in strategies if any(s in results_rate[M] for M in groups)]
    for i, s in enumerate(order):
        means = [np.mean(results_rate[M][s]) for M in groups]
        stds  = [np.std(results_rate[M][s])  for M in groups]
        ax.bar(x + (i-(len(order)-1)/2)*width, means, width=width, yerr=stds, capsize=4,
               label=s, color=palette.get(s, None), alpha=0.92, edgecolor='black', linewidth=0.8)
        rng = np.random.RandomState(1)
        for j, M in enumerate(groups):
            arr = np.array(results_rate[M][s])
            jitter = (rng.rand(len(arr))-0.5)*(width*0.9)
            ax.scatter(np.full_like(arr, x[j] + (i-(len(order)-1)/2)*width)+jitter, arr,
                       s=20, alpha=0.85, edgecolors='white', linewidths=0.3)
        for j, m in enumerate(means):
            ax.text(x[j] + (i-(len(order)-1)/2)*width, m - 0.05*max(1.0, ax.get_ylim()[1]),
                    f"{m:.1f}", ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={M}" for M in groups])
    ax.set_ylabel("Average Rate (Mbps)")
    ax.set_xlabel("CHO Target Set Size")
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(args.out_rate, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

