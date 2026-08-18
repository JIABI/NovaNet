import argparse

import pytest

from experiments.tune_heuristics import (
    evaluate_settings,
    make_settings,
    make_validation_episodes,
    policy_for_setting,
    resolve_validation_seeds,
    selected_settings,
    summarize_settings,
)
from novanet.config import load_config
from novanet.policies import (
    DwellAwarePolicy,
    PeriodicHOPolicy,
    RateDwellPolicy,
)
from novanet.simulation import SimulationMetrics


def _metrics(method: str, target_cost: float) -> SimulationMetrics:
    return SimulationMetrics(
        method=method,
        mean_rate_mbps=60.0,
        effective_throughput_mbps=59.5,
        handovers=4,
        handover_failures=1,
        hof_percent=25.0,
        outage_percent=1.0,
        cho_hit_percent=90.0,
        transmission_only_mean_ms=4.0,
        p50_latency_ms=40.0,
        p95_latency_ms=50.0,
        p99_latency_ms=55.0,
        p99_9_latency_ms=60.0,
        exceed_100_percent=0.0,
        ping_pong_count=1,
        ping_pong_percent=25.0,
        delivered_packets=100,
        dropped_packets=0,
        mean_target_cost=target_cost,
    )


def test_grid_builds_three_real_heuristic_families():
    config = load_config()
    settings = make_settings(
        periods=(8, 16),
        improvement_thresholds=(0.05, 0.10),
        dwell_weights=(0.25, 0.50),
        switch_penalties=(0.10, 0.20),
    )
    assert len(settings) == 2 + 2 + 2 * 2
    assert len({setting.setting_id for setting in settings}) == len(settings)
    assert isinstance(policy_for_setting(settings[0], config), PeriodicHOPolicy)
    assert isinstance(policy_for_setting(settings[2], config), DwellAwarePolicy)
    assert isinstance(policy_for_setting(settings[4], config), RateDwellPolicy)


def test_validation_seeds_exclude_training_and_reserved_test_rng():
    config = load_config()
    seeds = resolve_validation_seeds(config, None)
    assert len(seeds) == 3
    assert config.experiment.seed not in seeds
    with pytest.raises(ValueError, match="training"):
        resolve_validation_seeds(config, str(config.experiment.seed))
    with pytest.raises(ValueError, match="reserved"):
        resolve_validation_seeds(config, "9999", reserved_test_seed=9999)


def test_validation_evaluation_is_paired_and_uses_oracle_target_cost():
    config = load_config()
    settings = make_settings(
        periods=(8, 16),
        improvement_thresholds=(0.10, 0.20),
        dwell_weights=(0.50, 0.75),
        switch_penalties=(0.20,),
    )
    episodes = make_validation_episodes(config, (3025,), users_per_seed=2)
    calls = []

    def fake_simulator(
        _config,
        _ephemeris,
        policy,
        _scenario,
        *,
        seed,
        compute_oracle_cost,
    ):
        calls.append((policy.name, seed, compute_oracle_cost))
        if isinstance(policy, PeriodicHOPolicy):
            cost = 1.0 + abs(policy.period_steps - 16)
        elif isinstance(policy, DwellAwarePolicy):
            cost = policy.improvement_threshold
        else:
            cost = policy.dwell_weight + policy.switch_penalty
        return _metrics(policy.name, cost)

    raw = evaluate_settings(
        config,
        object(),
        settings,
        episodes,
        simulator=fake_simulator,
    )
    assert len(raw) == len(settings) * len(episodes)
    assert all(compute_oracle for _name, _seed, compute_oracle in calls)
    for episode in episodes:
        paired_seeds = {
            seed
            for _name, seed, _oracle in calls
            if seed == episode.episode_seed
        }
        assert paired_seeds == {episode.episode_seed}
        assert sum(seed == episode.episode_seed for _, seed, _ in calls) == len(
            settings
        )

    summaries = summarize_settings(raw)
    chosen = {row["family"]: row for row in selected_settings(summaries)}
    assert chosen["periodic_ho"]["period_steps"] == 16
    assert chosen["dwell_aware"]["improvement_threshold"] == pytest.approx(0.10)
    assert chosen["rate_dwell"]["dwell_weight"] == pytest.approx(0.50)
    assert all(row["objective_rank"] == 1 for row in chosen.values())
    with pytest.raises(ValueError, match="same paired validation"):
        summarize_settings(raw[:-1])


def test_invalid_grid_parser_reaches_argparse_error_via_cli_type():
    from experiments.tune_heuristics import parse_float_grid, parse_int_grid

    with pytest.raises(argparse.ArgumentTypeError):
        parse_int_grid("0,8")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_float_grid("0.1,nan")
