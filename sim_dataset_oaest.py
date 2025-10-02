
import math, random
import numpy as np
import torch
import importlib
CFG = importlib.import_module('config')

def _get(name, default):
    return getattr(CFG, name, default)

# pull needed config with fallbacks
TLE_PATH     = _get('TLE_PATH', 'tle.txt')
NUM_SAMPLES  = _get('NUM_SAMPLES', 2000)
LIMIT_SATS   = _get('LIMIT_SATS', 32)
DT_S         = _get('DT_S', 30)
TOP_K        = _get('TOP_K', 8)
ELEV_MIN_DEG = _get('ELEV_MIN_DEG', 10.0)
DELTA        = _get('DELTA', 30)

SAT_TX_POWER_DBM   = _get('SAT_TX_POWER_DBM', 20.0)
SAT_ANT_GAIN_DBI   = _get('SAT_ANT_GAIN_DBI', 30.0)
BANDWIDTH_HZ       = _get('BANDWIDTH_HZ', 20e6)
NOISE_PSD_DBM_HZ   = _get('NOISE_PSD_DBM_HZ', -174.0)
SMALL_SCALE_FADING_DB = _get('SMALL_SCALE_FADING_DB', 1.5)
ATTEN_DB_PER_KM    = _get('ATTEN_DB_PER_KM', 0.0)
CARRIER_HZ         = _get('CARRIER_HZ', 12e9)
EARTH_RADIUS_M     = _get('EARTH_RADIUS_M', 6371000.0)

from tle_ephem import build_ephemeris_from_tle
from orbit_feats import load_orbit_elements, orbit_prior_vector, build_spatial_affinity, build_temporal_affinity

def eirp_dbm(tx_dbm, tx_gain_dbi): return tx_dbm + tx_gain_dbi
def noise_power_dbm(bw_hz): return NOISE_PSD_DBM_HZ + 10.0 * math.log10(bw_hz)
def fspl_db(d_km, f_hz): return 32.45 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(f_hz/1e6)
def pathloss_db(d_km, f_hz): return fspl_db(d_km, f_hz) + SMALL_SCALE_FADING_DB + ATTEN_DB_PER_KM * d_km

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
    x = R * clat * clon
    y = R * clat * slon
    z = R * slat
    return np.array([x,y,z], dtype=np.float64)

def time_to_leave_steps(ue_ecef, ephem, global_id, t_idx, max_look=1800, elev_thr=ELEV_MIN_DEG):
    T = ephem.shape[0]
    end = min(T-1, t_idx + max_look)
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
    sel = np.array(cand, dtype=int)  # [K]
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
    sat_feats = np.array(feat, dtype=np.float32)  # [K,Fs]
    return sel, sat_feats

def generate_dataset_oaest(tle_path=TLE_PATH, num_samples=NUM_SAMPLES, limit_sats=LIMIT_SATS, seed=42):
    rng = np.random.RandomState(seed)
    ephem, names, start_utc, dt_s = build_ephemeris_from_tle(tle_path, duration_s=3*3600, dt_s=DT_S, limit_sats=limit_sats)
    orbit_elems = load_orbit_elements(tle_path)
    out = []
    EIRP = eirp_dbm(SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI)
    NOISE= noise_power_dbm(BANDWIDTH_HZ)

    T = ephem.shape[0]
    for _ in range(num_samples):
        lat, lon = rng.uniform(-60, 60), rng.uniform(-180, 180)
        ue = geodetic_to_ecef(lat, lon)
        t0 = rng.randint(1, T-DELTA-2)

        sel_t,  sat_t  = build_frame(ue, ephem[t0-1], ephem[t0])
        sel_tm, sat_tm = build_frame(ue, ephem[t0-2], ephem[t0-1])  # previous frame (t-Δt)
        # align prev selection order to current by global id
        order = {gid:i for i,gid in enumerate(sel_t)}
        sat_prev = np.zeros_like(sat_t)
        for j, gid in enumerate(sel_tm):
            if gid in order:
                sat_prev[order[int(gid)]] = sat_tm[j]
        # orbit prior per candidate (best-effort by index name)
        orbit_prior = []
        for gid in sel_t:
            key = f"SAT_{int(gid)}"
            elem = orbit_elems.get(key, None)
            orbit_prior.append( orbit_prior_vector(elem) )
        orbit_prior = np.array(orbit_prior, dtype=np.float32)  # [K,6]

        # spatial/temporal affinities
        A_sp = build_spatial_affinity(sat_t, orbit_prior)
        A_tm = build_temporal_affinity(sat_t.shape[0], gate=getattr(CFG, 'TIME_EDGE_GATE_INIT', 0.7))

        # TTL at time t for each candidate
        ttl = np.array([time_to_leave_steps(ue, ephem, int(g), t0) for g in sel_t], dtype=np.float32)

        # current connection index (greedy elevation as proxy)
        cur_idx = int(np.argmax(sat_t[:,0]))

        # teacher next distribution using true next SNR
        snr_next = []
        for gid in sel_t:
            d = np.linalg.norm(ephem[t0+DELTA, int(gid)] - ue)/1000.0
            pl = pathloss_db(d, CARRIER_HZ)
            snr = EIRP - pl - NOISE
            snr_next.append(snr)
        snr_next = np.array(snr_next, dtype=np.float32)  # [K]
        # energy terms for teacher
        mu = snr_next
        ttl_cur = max(1.0, ttl[cur_idx])
        delev = sat_t[:,1]
        delta_de = np.abs(delev - delev[cur_idx])
        not_cur = np.ones_like(mu); not_cur[cur_idx]=0.0
        alpha = getattr(CFG, 'E_ALPHA_SNR', 1.0)
        beta  = getattr(CFG, 'E_BETA_TTL', 1.0)
        gamma = getattr(CFG, 'E_GAMMA_SWITCH', 2.5)
        E_teacher = -alpha*mu + beta * (1.0/(1.0+ttl_cur)) + gamma * not_cur * np.log1p(np.exp(1.0*delta_de))
        tau = getattr(CFG, 'DP_TEMPERATURE', 1.0)
        q_teacher = np.exp(-E_teacher/tau); q_teacher = q_teacher / (q_teacher.sum() + 1e-9)

        logits = -E_teacher/max(1e-6, tau)
        logits = logits - logits.max()
        q_teacher = np.exp(logits)
        den = q_teacher.sum()
        if not np.isfinite(den) or den <= 0:
            q_teacher = np.ones_like(q_teacher, dtype=np.float32)/float(len(q_teacher))
        else:
            q_teacher = (q_teacher/ den).astype(np.float32)
        sample = dict(
            ue_feat = np.zeros((1, getattr(CFG, 'F_UE', 1)), dtype=np.float32),
            sat_feats_t = sat_t,
            sat_feats_prev = sat_prev,
            orbit_prior = orbit_prior,
            A_sp = A_sp.astype(np.float32),
            A_tm = A_tm.astype(np.float32),
            cur_idx = cur_idx,
            label_snr = snr_next,
            ttl = ttl,
            teacher_next_dist = q_teacher.astype(np.float32),
        )
        out.append(sample)
    return out
