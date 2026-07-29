from novanet.config import load_config


def test_paper_configuration_is_single_source_of_truth():
    cfg = load_config()
    assert cfg.experiment.minimum_elevation_deg == 10.0
    assert (
        cfg.experiment.tle_selection
        == "shell_stratified_orbit_balanced_nested"
    )
    assert cfg.experiment.candidate_cap == 8
    assert cfg.experiment.duration_s == 2400
    assert (
        cfg.experiment.decision_interval_s,
        cfg.experiment.geometry_subsample_s,
    ) == (30, 5)
    assert (cfg.handover.ttt_s, cfg.handover.execution_s) == (0.1, 0.15)
    assert cfg.handover.freeze_steps == 1
    assert cfg.traffic.packet_size_bytes == 1500
    assert cfg.traffic.protocol_processing_ms == 1.0
    assert cfg.channel.bandwidth_options_hz == (20e6, 100e6)
    assert cfg.planner.horizon_steps == 6


def test_legacy_constants_match_canonical_config():
    import config

    assert config.ELEV_MIN_DEG == 10.0
    assert config.TOP_K == 8
    assert config.SIM_DURATION_S == 2400
    assert config.TTT_SEC == 0.1
    assert config.FREEZE_S == 30
    assert config.PKT_SIZE_BITS == 12000
    assert config.HO_DELAY_MS == 150.0
    assert config.BANDWIDTH_OPTIONS_HZ == (20e6, 100e6)
