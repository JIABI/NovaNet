import argparse
import math

import numpy as np
import pytest

from experiments.lcb_calibration import (
    calibration_bins,
    canonical_validation_bounds,
    coverage_curve,
    parse_kappas,
)


def test_canonical_calibration_uses_training_stream_final_twenty_percent():
    assert canonical_validation_bounds(3600) == (2880, 3600)
    assert canonical_validation_bounds(7) == (5, 7)


def test_calibration_bins_report_rmse_and_signed_gap():
    rows = calibration_bins(
        predicted_sigma_db=np.asarray([1.0, 2.0, 3.0, 4.0]),
        prediction_error_db=np.asarray([1.0, -1.0, 2.0, -2.0]),
        bins=2,
    )
    assert [row["count"] for row in rows] == [2, 2]
    assert rows[0]["mean_predicted_sigma_db"] == pytest.approx(1.5)
    assert rows[0]["empirical_rmse_db"] == pytest.approx(1.0)
    assert rows[0]["signed_gap_db"] == pytest.approx(-0.5)
    assert rows[1]["mean_predicted_sigma_db"] == pytest.approx(3.5)
    assert rows[1]["empirical_rmse_db"] == pytest.approx(2.0)
    assert rows[1]["signed_gap_db"] == pytest.approx(-1.5)


def test_one_sided_lcb_coverage_matches_event_and_gaussian_reference():
    rows = coverage_curve(
        realized_residual=np.asarray([-1.0, 0.0, 1.0]),
        residual_mu=np.zeros(3),
        residual_sigma=np.ones(3),
        kappas=[0.0, 1.0],
    )
    assert rows[0]["covered_count"] == 2
    assert rows[0]["empirical_coverage"] == pytest.approx(2.0 / 3.0)
    assert rows[0]["gaussian_phi"] == pytest.approx(0.5)
    assert rows[1]["covered_count"] == 3
    assert rows[1]["empirical_coverage"] == pytest.approx(1.0)
    assert rows[1]["gaussian_phi"] == pytest.approx(
        0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    )


def test_kappa_parser_rejects_negative_or_empty_grids():
    assert parse_kappas("0, 1.5, 3") == [0.0, 1.5, 3.0]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_kappas("")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_kappas("0,-1")


def test_calibration_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="positive"):
        calibration_bins(np.asarray([0.0]), np.asarray([0.0]), bins=1)
