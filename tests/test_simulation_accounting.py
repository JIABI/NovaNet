import numpy as np

from novanet.simulation import _overlap_seconds


def test_blackout_overlap_is_integrated_over_rate_interval():
    intervals = [(0.1, 0.25), (4.9, 5.1)]
    assert np.isclose(_overlap_seconds(0.0, 5.0, intervals), 0.25)
    assert np.isclose(_overlap_seconds(5.0, 10.0, intervals), 0.1)
