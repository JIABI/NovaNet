import pytest

from novanet.config import load_config
from novanet.dataset import StaleTLEError, validate_tle_epoch


def test_repository_does_not_silently_use_date_mismatched_tle():
    with pytest.raises(StaleTLEError, match="does not match the paper start time"):
        validate_tle_epoch(load_config())

