
import numpy as np

def parse_tle_file(tle_path):
    """
    Returns list of (name, L1, L2). Name line may or may not exist; handle both.
    """
    triples = []
    with open(tle_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('1 ') and i+1 < len(lines):
            name = f"SAT_{len(triples)}"
            l1 = lines[i]; l2 = lines[i+1]; i += 2
        elif i+2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            name = lines[i]; l1 = lines[i+1]; l2 = lines[i+2]; i += 3
        else:
            i += 1; continue
        triples.append((name, l1, l2))
    return triples

def tle_orbit_elements(line2):
    """
    Extract key orbital elements from TLE line 2 by column.
    Returns: (incl_deg, raan_deg, ecc, argp_deg, mean_anom_deg, mean_motion_revperday)
    """
    incl = float(line2[8:16])
    raan = float(line2[17:25])
    ecc  = float('.' + line2[26:33].strip())
    argp = float(line2[34:42])
    manom= float(line2[43:51])
    n    = float(line2[52:63])
    return incl, raan, ecc, argp, manom, n

def load_orbit_elements(tle_path, names_filter=None):
    elems = {}
    for name, l1, l2 in parse_tle_file(tle_path):
        key = name.strip()
        incl, raan, ecc, argp, manom, n = tle_orbit_elements(l2)
        elems[key] = dict(incl=incl, raan=raan, ecc=ecc, argp=argp, manom=manom, mm=n)
    if names_filter is not None:
        out = []
        for i, nm in enumerate(names_filter):
            out.append(elems.get(nm, None))
        return out
    return elems

def orbit_prior_vector(elem):
    """
    Normalize and pack orbital elements to a compact prior vector.
    """
    if elem is None:
        return np.zeros((6,), dtype=np.float32)
    incl = elem['incl'] / 180.0         # 0..1
    raan = elem['raan'] / 360.0         # 0..1
    ecc  = float(elem['ecc'])            # already ~1e-3
    argp = elem['argp'] / 360.0
    man  = elem['manom'] / 360.0
    mm   = elem['mm'] / 20.0             # rev/day, scale ~[0,1]
    return np.array([incl, raan, ecc, argp, man, mm], dtype=np.float32)

def build_spatial_affinity(sat_feats_t, orbit_priors, geom_w=1.0, orbit_w=1.0):
    """
    sat_feats_t: [K, Fs] with columns [elev, d_elev, range_km, d_range, pathloss, potential]
    orbit_priors: [K, 6]
    Returns A_spatial: [K,K] row-normalized affinities.
    """
    K = sat_feats_t.shape[0]
    A = np.zeros((K, K), dtype=np.float32)
    # geometric distance in feature space (elev, range, d_elev)
    G = sat_feats_t[:, [0,1,2]].astype(np.float32)
    # normalize
    Gn = (G - G.mean(0, keepdims=True)) / (G.std(0, keepdims=True) + 1e-6)
    On = (orbit_priors - orbit_priors.mean(0, keepdims=True)) / (orbit_priors.std(0, keepdims=True) + 1e-6 + 1e-9)
    for i in range(K):
        for j in range(K):
            dg = np.linalg.norm(Gn[i] - Gn[j])
            do = np.linalg.norm(On[i] - On[j])
            A[i, j] = np.exp(-(geom_w * dg + orbit_w * do))
    # row normalize
    A = A / (A.sum(1, keepdims=True) + 1e-6)
    return A

def build_temporal_affinity(K, gate=0.7):
    A = np.zeros((K, K), dtype=np.float32)
    np.fill_diagonal(A, float(gate))
    return A
