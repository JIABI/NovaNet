from dataclasses import asdict, replace

import pytest
import torch

from experiments.weight_pareto import (
    CORE_WEIGHTS,
    config_with_runtime_weights,
    episode_seed,
    load_policy_for_setting,
    make_weight_settings,
    pareto_mask,
    summarize_settings,
)
from novanet.config import load_config
from novanet.dataset import validate_tle_epoch
from novanet.model import NovaNet
from novanet.policies import NovaNetPolicy


def _checkpoint(path, config, *, fingerprint=None, provenance=None):
    torch.save(
        {
            "model_state": NovaNet(config).state_dict(),
            "config": asdict(config),
            "config_fingerprint": fingerprint or config.fingerprint,
            "checkpoint_kind": "novanet",
            "training_protocol": "novanet_h6_softdp_sequence_v2",
            "paper_table_eligible": True,
            "training": {
                "samples": config.training.num_samples,
                "train_samples": int(0.8 * config.training.num_samples),
                "validation_samples": (
                    config.training.num_samples
                    - int(0.8 * config.training.num_samples)
                ),
                "epochs_requested": config.training.epochs,
                "epochs_completed": config.training.epochs,
                "training_complete": True,
                "batch_size": config.training.batch_size,
                "optimizer": "AdamW",
                "learning_rate": config.training.learning_rate,
                "weight_decay": 0.0,
                "training_seed": config.experiment.seed,
                "allow_stale_tle": False,
                "train_residual_labels": 1,
                "validation_residual_labels": 1,
                "train_hof_labels": 1,
                "validation_hof_labels": 1,
            },
            "epoch": 1,
            "validation": {"total": 0.1},
            "tle_provenance": provenance
            or validate_tle_epoch(
                config, maximum_age_days=float("inf")
            ),
        },
        path,
    )


def test_oat_design_changes_exactly_one_core_weight_from_nominal():
    config = load_config("configs/paper.yaml")
    settings = make_weight_settings(
        config,
        (0.5, 1.0, 1.5),
        design="one-at-a-time",
    )
    assert len(settings) == 1 + 2 * len(CORE_WEIGHTS)
    nominal = settings[0]
    assert nominal["setting_id"] == "000_nominal"
    for setting in settings[1:]:
        changed = [
            name
            for name in CORE_WEIGHTS
            if setting[name] != nominal[name]
        ]
        assert changed == [setting["varied_weight"]]
        assert setting["lambda_u"] == nominal["lambda_u"]
    seeds = {
        episode_seed(validation_seed, user)
        for validation_seed in (3025, 4025, 5025)
        for user in range(20)
    }
    assert len(seeds) == 60


def test_runtime_override_whitelist_rejects_any_extra_config_drift(tmp_path):
    config = load_config("configs/paper.yaml")
    checkpoint = tmp_path / "valid.pt"
    _checkpoint(checkpoint, config)
    bandwidth_only = replace(
        config,
        channel=replace(config.channel, bandwidth_hz=100e6),
    )
    NovaNetPolicy(
        bandwidth_only,
        checkpoint,
        device="cpu",
        allowed_config_overrides=("channel.bandwidth_hz",),
    )

    extra_drift = replace(
        bandwidth_only,
        handover=replace(config.handover, freeze_steps=0),
    )
    with pytest.raises(ValueError, match="outside.*whitelist"):
        NovaNetPolicy(
            extra_drift,
            checkpoint,
            device="cpu",
            allowed_config_overrides=("channel.bandwidth_hz",),
        )


def test_runtime_override_is_limited_to_declared_planner_weights(tmp_path):
    config = load_config("configs/paper.yaml")
    setting = make_weight_settings(config, (0.5,))[1]
    runtime = config_with_runtime_weights(config, setting)
    assert runtime.planner.alpha == config.planner.alpha * 0.5
    assert runtime.experiment == config.experiment
    assert runtime.channel == config.channel

    invalid = dict(setting)
    invalid["duration_s"] = 60
    with pytest.raises(ValueError, match="unknown=.*duration_s"):
        config_with_runtime_weights(config, invalid)

    checkpoint = tmp_path / "valid.pt"
    _checkpoint(checkpoint, config)
    policy, runtime = load_policy_for_setting(
        config,
        checkpoint,
        setting,
        device="cpu",
    )
    assert policy.model.energy.alpha == runtime.planner.alpha
    assert policy.model.energy.beta == runtime.planner.beta


def test_strict_checkpoint_fingerprint_cannot_be_bypassed(tmp_path):
    config = load_config("configs/paper.yaml")
    checkpoint = tmp_path / "wrong-fingerprint.pt"
    _checkpoint(checkpoint, config, fingerprint="not-the-canonical-config")
    setting = make_weight_settings(config, (0.5,))[0]
    with pytest.raises(ValueError, match="does not match selected config"):
        load_policy_for_setting(
            config,
            checkpoint,
            setting,
            device="cpu",
        )

    provenance = validate_tle_epoch(config, maximum_age_days=float("inf"))
    provenance["selected_names_sha256"] = "wrong-selected-subset"
    checkpoint = tmp_path / "wrong-provenance.pt"
    _checkpoint(checkpoint, config, provenance=provenance)
    with pytest.raises(ValueError, match="does not match the current"):
        load_policy_for_setting(
            config,
            checkpoint,
            setting,
            device="cpu",
        )


def test_novanet_policy_rejects_interrupted_checkpoint(tmp_path):
    config = load_config("configs/paper.yaml")
    checkpoint = tmp_path / "partial.pt"
    _checkpoint(checkpoint, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["paper_table_eligible"] = False
    payload["training"]["training_complete"] = False
    payload["training"]["epochs_completed"] = 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="incomplete training"):
        NovaNetPolicy(
            config,
            checkpoint,
            device="cpu",
            require_paper_eligible=False,
        )

def test_pareto_front_and_summary_are_computed_from_raw_rows():
    summary_rows = [
        {
            "effective_throughput_mbps_mean": 60.0,
            "handovers_mean": 4.0,
            "hof_percent_pooled": 1.0,
            "outage_percent_mean": 2.0,
        },
        {
            "effective_throughput_mbps_mean": 61.0,
            "handovers_mean": 3.0,
            "hof_percent_pooled": 0.8,
            "outage_percent_mean": 1.9,
        },
        {
            "effective_throughput_mbps_mean": 62.0,
            "handovers_mean": 5.0,
            "hof_percent_pooled": 0.7,
            "outage_percent_mean": 1.8,
        },
    ]
    assert pareto_mask(summary_rows) == [False, True, True]

    config = load_config("configs/paper.yaml")
    setting = make_weight_settings(config, (1.0,))[0]
    raw = []
    for handovers, failures, throughput in ((1, 1, 60.0), (9, 0, 62.0)):
        row = {
            "setting_id": setting["setting_id"],
            "varied_weight": setting["varied_weight"],
            "multiplier": setting["multiplier"],
            "mean_rate_mbps": throughput + 1.0,
            "effective_throughput_mbps": throughput,
            "handovers": handovers,
            "handover_failures": failures,
            "hof_percent": 100.0 * failures / handovers,
            "outage_percent": 1.0,
            "ping_pong_percent": 0.0,
        }
        row.update({name: setting[name] for name in setting if name in {
            "alpha", "beta", "c0", "c1", "c2", "lambda_u"
        }})
        raw.append(row)
    summary = summarize_settings(raw)[0]
    assert summary["replicates"] == 2
    assert summary["effective_throughput_mbps_mean"] == 61.0
    assert summary["hof_percent_pooled"] == 10.0
