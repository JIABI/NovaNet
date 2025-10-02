
# evaluate_oaest.py — Energy-based evaluation with OA-EST-PCGNN
import math, argparse, numpy as np, torch

import matplotlib.pyplot as plt

import importlib

CFG = importlib.import_module('config')


def _get(name, default): return getattr(CFG, name, default)


DT_S = _get('DT_S', 30)

TOP_K = _get('TOP_K', 8)

ELEV_MIN_DEG = _get('ELEV_MIN_DEG', 10.0)

LIMIT_SATS = _get('LIMIT_SATS', 32)

DELTA = _get('DELTA', 30)

FREEZE_S = _get('FREEZE_S', 90.0)  # 秒，保证 > step_s

HYS_MARGIN = _get('HYS_MARGIN', 0.5)  # 能量滞回门槛

SAT_TX_POWER_DBM = _get('SAT_TX_POWER_DBM', 20.0)

SAT_ANT_GAIN_DBI = _get('SAT_ANT_GAIN_DBI', 30.0)

BANDWIDTH_HZ = _get('BANDWIDTH_HZ', 20e6)

NOISE_PSD_DBM_HZ = _get('NOISE_PSD_DBM_HZ', -174.0)

ATTEN_DB_PER_KM = _get('ATTEN_DB_PER_KM', 0.0)

SMALL_SCALE_FADING_DB = _get('SMALL_SCALE_FADING_DB', 1.5)

CARRIER_HZ = _get('CARRIER_HZ', 12e9)

EARTH_RADIUS_M = _get('EARTH_RADIUS_M', 6371000.0)

from tle_ephem import build_ephemeris_from_tle

from orbit_feats import load_orbit_elements, orbit_prior_vector, build_spatial_affinity, build_temporal_affinity

from model_oaest import PCGNN_OAEST


def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi


def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)


def fspl_db(d_km, f_hz): return 32.45 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(f_hz / 1e6)


def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km


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


def time_to_leave_steps(ue_ecef, ephem, global_id, t_idx, elev_thr=ELEV_MIN_DEG, max_look=3600):
    T = ephem.shape[0]

    end = min(T - 1, t_idx + max_look)

    steps = 0

    for tt in range(t_idx, end):

        if elevation_angle(ue_ecef, ephem[tt, global_id]) < elev_thr:
            break

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


def run_strategy_oaest(model, dev, ephem, orbit_elems, ue, t0, span_s, step_s):
    FREEZE_SLOTS = max(0, int(FREEZE_S / step_s))

    cnt = 0

    sel, st = build_frame(ue, ephem[t0 - 1], ephem[t0])

    # affinities

    orbit_prior = np.array([orbit_prior_vector(orbit_elems.get(f"SAT_{int(g)}", None)) for g in sel], dtype=np.float32)

    A_sp = build_spatial_affinity(st, orbit_prior).astype(np.float32)

    A_tm = build_temporal_affinity(st.shape[0], gate=getattr(CFG, 'TIME_EDGE_GATE_INIT', 0.7)).astype(np.float32)

    # init

    cur_local = int(np.argmax(st[:, 0]))

    cur_gid = sel[cur_local]

    freeze_left = 0

    for t in range(t0, t0 + span_s, step_s):

        # frame at t

        sel, st = build_frame(ue, ephem[t - 1], ephem[t])

        orbit_prior = np.array([orbit_prior_vector(orbit_elems.get(f"SAT_{int(g)}", None)) for g in sel],
                               dtype=np.float32)

        A_sp = build_spatial_affinity(st, orbit_prior).astype(np.float32)

        A_tm = build_temporal_affinity(st.shape[0], gate=getattr(CFG, 'TIME_EDGE_GATE_INIT', 0.7)).astype(np.float32)

        # realign current index

        if cur_gid in sel:

            cur_local = int(np.where(sel == cur_gid)[0][0])

        else:

            cur_local = int(np.argmax(st[:, 0]))

            cur_gid = sel[cur_local]

        # forward

        ue_t = torch.zeros((1, 1, _get('F_UE', 1)), dtype=torch.float32, device=dev)

        st_t = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)

        sp_t = torch.tensor(st, dtype=torch.float32, device=dev).unsqueeze(0)

        Asp_t = torch.tensor(A_sp, dtype=torch.float32, device=dev).unsqueeze(0)

        Atm_t = torch.tensor(A_tm, dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():

            outs = model(ue_t, st_t, sp_t, Asp_t, Atm_t, torch.tensor([cur_local], device=dev))

            E = outs['energy'][0]  # [K]

        # decision with freeze + hysteresis

        if freeze_left > 0:
            freeze_left -= 1

            continue  # stay during freeze

        best_idx = int(torch.argmin(E).item())

        if best_idx != cur_local:

            if E[best_idx].item() + HYS_MARGIN < E[cur_local].item():
                cnt += 1

                cur_local = best_idx

                cur_gid = sel[cur_local]

                freeze_left = FREEZE_SLOTS

        # else stay

    return cnt


def run_strategy_max_elev(ephem, ue, t0, span_s, step_s):
    cnt = 0

    sel, st = build_frame(ue, ephem[t0 - 1], ephem[t0])

    cur_local = int(np.argmax(st[:, 0]));
    cur_gid = sel[cur_local]

    for t in range(t0, t0 + span_s, step_s):

        sel, st = build_frame(ue, ephem[t - 1], ephem[t])

        best = int(np.argmax(st[:, 0]))

        nxt_gid = sel[best]

        if nxt_gid != cur_gid:
            cnt += 1

            cur_gid = nxt_gid

    return cnt


def run_strategy_max_servetime(ephem, ue, t0, span_s, step_s):
    cnt = 0

    sel, st = build_frame(ue, ephem[t0 - 1], ephem[t0])

    # compute TTL for current frame candidates

    ttl = np.array([time_to_leave_steps(ue, ephem, int(gid), t0) for gid in sel], dtype=np.float32)

    cur_local = int(np.argmax(ttl));
    cur_gid = sel[cur_local]

    for t in range(t0, t0 + span_s, step_s):

        sel, st = build_frame(ue, ephem[t - 1], ephem[t])

        ttl = np.array([time_to_leave_steps(ue, ephem, int(gid), t) for gid in sel], dtype=np.float32)

        best = int(np.argmax(ttl))

        nxt_gid = sel[best]

        if nxt_gid != cur_gid:
            cnt += 1

            cur_gid = nxt_gid

    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/oaest_best.pt")
    ap.add_argument("--n_users", type=int, default=60)
    ap.add_argument("--span_s", type=int, default=1800)
    ap.add_argument("--step_s", type=int, default=30)
    ap.add_argument("--limit_sats", type=int, default=LIMIT_SATS)
    ap.add_argument("--out", default="oaest_compare_ho.png")
    args = ap.parse_args()

    # Ephemeris
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(CFG.TLE_PATH, duration_s=max(3 * 3600, args.span_s + 600),
                                                             dt_s=DT_S, limit_sats=args.limit_sats)
    orbit_elems = load_orbit_elements(CFG.TLE_PATH)

    # Model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN_OAEST(f_ue=_get('F_UE', 1), f_sat=_get('F_SAT', 6), f_edge=_get('F_EDGE', 0),
                        hidden=_get('HIDDEN', 128), gnn_layers=_get('GNN_LAYERS', 2)).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd);
    model.eval()

    # Run all users
    counts = {"OA-EST": [], "Maximum-Elevation": [], "Maximum-serveTime": []}
    T = ephem.shape[0]
    for _ in range(args.n_users):
        lat, lon = np.random.uniform(-60, 60), np.random.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = np.random.randint(2, T - args.span_s - 2)

        # OA-EST
        cnt_oa = run_strategy_oaest(model, dev, ephem, orbit_elems, ue, t0, args.span_s, args.step_s)
        counts["OA-EST"].append(cnt_oa)
        # Max-Elev
        cnt_me = run_strategy_max_elev(ephem, ue, t0, args.span_s, args.step_s)
        counts["Maximum-Elevation"].append(cnt_me)
        # Max-ServeTime
        cnt_ms = run_strategy_max_servetime(ephem, ue, t0, args.span_s, args.step_s)
        counts["Maximum-serveTime"].append(cnt_ms)

    # Print stats
    for k, v in counts.items():
        arr = np.array(v, dtype=np.float32)
        print(f"{k:>18s} | HO mean={arr.mean():.2f}, std={arr.std():.2f}")

    # ---- Plot like your figure ----
    labels = list(counts.keys())
    data = [np.array(counts[k], dtype=np.float32) for k in labels]
    means = [x.mean() for x in data]
    stds = [x.std() for x in data]

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.55

    # bars + errorbars
    bars = plt.bar(x, means, width=width, yerr=stds, capsize=6, alpha=0.85,
                   edgecolor='black', linewidth=1.0)

    # jittered scatter of per-user counts
    rng = np.random.RandomState(0)
    for i, arr in enumerate(data):
        jitter = (rng.rand(len(arr)) - 0.5) * (width * 0.6)
        plt.scatter(np.full_like(arr, x[i]) + jitter, arr, s=28, alpha=0.85, edgecolors='none')

    # annotate mean value inside bars
    for rect, m in zip(bars, means):
        plt.text(rect.get_x() + rect.get_width() / 2.0, m - (0.08 * plt.ylim()[1]),
                 f"{m:.1f}", ha='center', va='top', fontsize=12, color='white', fontweight='bold')

    plt.ylabel("Handover Frequency")
    plt.xticks(x, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.title("Handover Frequency Comparison")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()