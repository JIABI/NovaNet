from dataclasses import replace

import numpy as np

from novanet.config import load_config
from novanet.forecast import ForecastSequence
from novanet.policies import (
    DwellAwarePolicy,
    PeriodicHOPolicy,
    RateDwellPolicy,
    SkipKPolicy,
)


def sequence(
    *,
    elevation=(20.0, 40.0),
    ttl=(100.0, 500.0),
    sinr=(10.0, 12.0),
    current=0,
):
    cfg = load_config()
    candidates = len(elevation)
    if len(ttl) != candidates or len(sinr) != candidates:
        raise ValueError("elevation, ttl, and sinr must have equal length")
    features = np.zeros(
        (cfg.planner.horizon_steps, candidates, 6), np.float32
    )
    features[:, :, 0] = np.asarray(elevation) / 90.0
    features[:, :, 4] = np.asarray(ttl) / 600.0
    features[0, :, 5] = np.asarray(sinr) / 30.0
    valid = np.ones((cfg.planner.horizon_steps, candidates), dtype=bool)
    return ForecastSequence(
        node_features=features,
        spatial_adjacency=np.zeros(
            (cfg.planner.horizon_steps, candidates, candidates)
        ),
        valid_mask=valid,
        candidate_ids=np.arange(10, 10 + candidates),
        current_idx=current,
        deterministic_snr_db=np.tile(sinr, (cfg.planner.horizon_steps, 1)),
        ttl_s=np.tile(ttl, (cfg.planner.horizon_steps, 1)),
    )


def test_handcrafted_baselines_follow_documented_gates():
    cfg = load_config()
    seq = sequence()
    assert DwellAwarePolicy().choose(seq) == 1
    assert RateDwellPolicy.from_config(cfg).choose(seq) == 1

    periodic = PeriodicHOPolicy(period_steps=2)
    assert periodic.choose(seq) == 1
    assert periodic.choose(seq) == 0

    skip = SkipKPolicy(skip=1)
    assert skip.choose(seq) == 0
    # Satellite 11 remains the same chronological target and is skipped for
    # its complete dominance interval, not merely delayed by one epoch.
    assert skip.choose(seq) == 0
    next_distinct = sequence(
        elevation=(20.0, 30.0, 50.0),
        ttl=(100.0, 300.0, 500.0),
        sinr=(10.0, 11.0, 12.0),
    )
    assert skip.choose(next_distinct) == 2

    skip_two = SkipKPolicy(skip=2)
    assert skip_two.choose(seq) == 0
    assert skip_two.choose(next_distinct) == 0
    third_distinct = sequence(
        elevation=(20.0, 30.0, 40.0, 60.0),
        ttl=(100.0, 200.0, 300.0, 600.0),
        sinr=(10.0, 11.0, 12.0, 13.0),
    )
    assert skip_two.choose(third_distinct) == 3
