import argparse
from pathlib import Path

import pytest
import torch

import novanet.config as config_module
from novanet.config import load_config
from scripts.benchmark_inference import (
    benchmark_case,
    parse_candidate_caps,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_candidate_cap_parser_matches_paper_benchmark_grid():
    assert parse_candidate_caps("8,16,32") == [8, 16, 32]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_candidate_caps("8,8")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_candidate_caps("1")


def test_tiny_benchmark_uses_current_six_dimensional_model():
    row = benchmark_case(
        load_config(),
        candidates=2,
        warmup=0,
        repetitions=1,
        threads=1,
        requested_device="cpu",
        seed=2025,
    )
    assert row["node_feature_dim"] == 6
    assert row["future_recurrent_feature_dim"] == 5
    assert row["parameters"] == 249_603
    assert row["trainable_parameters"] == 249_603
    assert row["timing"]["mean_ms"] > 0.0
    assert row["memory"]["model_parameter_bytes"] == 249_603 * 4
    assert row["memory"]["logical_input_tensor_bytes"] > 0
    assert row["memory"]["worker_process_peak_rss_scope"].startswith(
        "isolated worker high-water RSS"
    )
    assert torch.get_num_threads() == 1


def test_default_config_falls_back_to_wheel_data_directory(tmp_path):
    source_root = tmp_path / "missing-source"
    installed_root = tmp_path / "share" / "novanet"
    installed_config = installed_root / "configs" / "paper.yaml"
    installed_config.parent.mkdir(parents=True)
    installed_config.write_text("schema_version: 5\n", encoding="utf-8")
    assert config_module._select_default_config_path(
        source_root, installed_root
    ) == installed_config


def test_relative_tle_uses_selected_packaged_data_root(tmp_path, monkeypatch):
    packaged_root = tmp_path / "share" / "novanet"
    packaged_root.mkdir(parents=True)
    packaged_tle = packaged_root / "starlink.tle"
    packaged_tle.write_text("packaged TLE sentinel\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path / "missing-source")
    monkeypatch.setattr(config_module, "DEFAULT_DATA_ROOT", packaged_root)
    assert load_config().resolve_tle_path() == packaged_tle


def test_wheel_data_files_are_declared_in_pyproject():
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"share/novanet/configs" = ["configs/paper.yaml"]' in text
    assert '"share/novanet" = ["starlink.tle"]' in text
