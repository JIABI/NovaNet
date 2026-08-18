from dataclasses import replace

import pytest

from novanet.config import load_config


def test_paper_configuration_is_single_source_of_truth():
    cfg = load_config()
    assert cfg.schema_version == 5
    assert cfg.experiment.seed == 2025
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
    assert cfg.channel.ue_antenna_elements == 1
    assert cfg.channel.exogenous_interference_power_w == 0.0
    assert cfg.channel.nominal_measurement_std_db == 0.0
    assert cfg.handover.event_step_s == 0.01
    assert cfg.multi_ue.blocking_cost is None
    assert cfg.planner.horizon_steps == 6
    assert (
        cfg.planner.temperature,
        cfg.planner.policy_temperature,
        cfg.planner.teacher_temperature,
    ) == (1.0, 1.0, 1.0)
    assert cfg.model.node_feature_dim == 6
    assert cfg.model.transition_feature_dim == 5
    assert cfg.planner.rate_reference_mbps == 50.0
    assert cfg.planner.ttl_reference_s == 600.0
    assert (
        cfg.planner.alpha,
        cfg.planner.beta,
        cfg.planner.c0,
        cfg.planner.c1,
        cfg.planner.c2,
    ) == (1.0, 0.35, 1.0, 0.5, 1.5)


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


def test_config_rejects_nonfinite_hof_and_invalid_multi_ue_limits():
    cfg = load_config()
    with pytest.raises(ValueError, match="failure_outage_fraction"):
        replace(
            cfg,
            handover=replace(
                cfg.handover,
                failure_outage_fraction=float("nan"),
            ),
        ).validate()
    with pytest.raises(ValueError, match="failure_outage_fraction"):
        replace(
            cfg,
            handover=replace(cfg.handover, failure_outage_fraction=1.0),
        ).validate()
    with pytest.raises(ValueError, match="minimum admission rate"):
        replace(
            cfg,
            multi_ue=replace(
                cfg.multi_ue,
                minimum_admission_rate_mbps=float("nan"),
            ),
        ).validate()
