"""Compatibility entry point for sequence-dataset generation."""

from novanet.config import load_config
from novanet.dataset import (
    GenerationOptions,
    NovaNetSequenceDataset,
    generate_sequence_samples,
    validate_tle_epoch,
)


def generate_dataset_oaest(
    tle_path=None,
    num_samples=None,
    limit_sats=None,
    seed=None,
    allow_stale_tle=False,
):
    cfg = load_config()
    if tle_path not in (None, cfg.experiment.tle_path, str(cfg.resolve_tle_path())):
        raise ValueError("Use --config to select a different TLE path")
    if limit_sats not in (None, cfg.experiment.num_satellites):
        raise ValueError("Use --config to select a different satellite count")
    if seed not in (None, cfg.experiment.seed):
        raise ValueError("Use --config to select a different seed")
    return generate_sequence_samples(
        cfg,
        GenerationOptions(
            num_samples=num_samples or cfg.training.num_samples,
            allow_stale_tle=allow_stale_tle,
        ),
    )


__all__ = [
    "GenerationOptions",
    "NovaNetSequenceDataset",
    "generate_dataset_oaest",
    "generate_sequence_samples",
    "validate_tle_epoch",
]
