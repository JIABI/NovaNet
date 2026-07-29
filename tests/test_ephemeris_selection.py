from novanet.config import load_config
from novanet.ephemeris import orbit_balanced_records, read_tle


def test_orbit_balanced_selection_is_deterministic_and_nested():
    cfg = load_config()
    records = read_tle(cfg.resolve_tle_path())
    selected_60 = orbit_balanced_records(records, 60)
    selected_120 = orbit_balanced_records(records, 120)

    names_60 = [record[0] for record in selected_60]
    names_120 = [record[0] for record in selected_120]
    assert names_120[:60] == names_60
    assert len(set(names_120)) == 120
    shells = {
        (
            round(float(record[2][8:16])),
            round(float(record[2][52:63]), 1),
        )
        for record in selected_120
    }
    assert len(shells) > 1
