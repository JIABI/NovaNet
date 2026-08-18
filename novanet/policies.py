"""Candidate-ranking policies sharing the same forecast information budget."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch

from .config import NovaNetConfig
from .dataset import validate_tle_epoch
from .forecast import ForecastSequence
from .model import NovaNet


def _config_difference_paths(
    reference: object,
    runtime: object,
    prefix: str = "",
) -> set[str]:
    """Return dotted leaf paths that differ between two config mappings."""

    if isinstance(reference, dict) and isinstance(runtime, dict):
        paths: set[str] = set()
        for key in set(reference) | set(runtime):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in runtime:
                paths.add(path)
            else:
                paths.update(
                    _config_difference_paths(
                        reference[key], runtime[key], path
                    )
                )
        return paths
    return set() if reference == runtime else {prefix}


class Policy(Protocol):
    name: str

    def choose(self, sequence: ForecastSequence) -> int:
        """Return a local candidate index."""


@dataclass
class OfflineOraclePolicy:
    """Marker for the non-causal same-energy reference used in evaluation."""

    name: str = "Offline oracle"
    is_noncausal_oracle: bool = True

    def choose(self, sequence: ForecastSequence) -> int:
        raise RuntimeError(
            "OfflineOraclePolicy requires realized future link/CHO context "
            "from simulate_single_user"
        )


@dataclass
class MaxElevationPolicy:
    name: str = "Max-Elevation"

    def choose(self, sequence: ForecastSequence) -> int:
        score = sequence.node_features[0, :, 0].astype(float)
        score[~sequence.valid_mask[0]] = -np.inf
        return int(np.argmax(score))


@dataclass
class MaxServeTimePolicy:
    name: str = "Max-ServeTime"

    def choose(self, sequence: ForecastSequence) -> int:
        score = sequence.ttl_s[0].astype(float)
        score[~sequence.valid_mask[0]] = -np.inf
        return int(np.argmax(score))


@dataclass
class RateDwellPolicy:
    rate_weight: float = 1.0
    dwell_weight: float = 0.5
    switch_penalty: float = 0.2
    rate_reference_mbps: float = 50.0
    ttl_reference_s: float = 600.0
    sinr_reference_db: float = 30.0
    bandwidth_hz: float = 20e6
    implementation_efficiency: float = 0.75
    name: str = "Rate-Dwell"

    @classmethod
    def from_config(cls, config: NovaNetConfig) -> "RateDwellPolicy":
        return cls(
            rate_weight=1.0,
            dwell_weight=0.5,
            switch_penalty=0.2,
            rate_reference_mbps=config.planner.rate_reference_mbps,
            ttl_reference_s=config.planner.ttl_reference_s,
            sinr_reference_db=config.model.sinr_reference_db,
            bandwidth_hz=config.channel.bandwidth_hz,
            implementation_efficiency=config.channel.implementation_efficiency,
        )

    def choose(self, sequence: ForecastSequence) -> int:
        measured_sinr_db = (
            sequence.node_features[0, :, 5] * self.sinr_reference_db
        )
        snr_linear = np.power(10.0, measured_sinr_db / 10.0)
        rate_mbps = (
            self.implementation_efficiency
            * self.bandwidth_hz
            * np.log2(1.0 + snr_linear)
            / 1e6
        )
        score = (
            self.rate_weight * rate_mbps / self.rate_reference_mbps
            + self.dwell_weight
            * sequence.ttl_s[0]
            / self.ttl_reference_s
        )
        score = score.astype(float)
        score -= self.switch_penalty * (
            np.arange(len(score)) != sequence.current_idx
        )
        score[~sequence.valid_mask[0]] = -np.inf
        return int(np.argmax(score))


@dataclass
class DwellAwarePolicy:
    """Max-TTL rule with the manuscript's normalized improvement gate."""

    improvement_threshold: float = 0.10
    ttl_reference_s: float = 600.0
    name: str = "Dwell-Aware"

    def choose(self, sequence: ForecastSequence) -> int:
        valid = sequence.valid_mask[0]
        target = int(np.argmax(np.where(valid, sequence.ttl_s[0], -np.inf)))
        if not valid[sequence.current_idx]:
            return target
        improvement = (
            sequence.ttl_s[0, target] - sequence.ttl_s[0, sequence.current_idx]
        ) / self.ttl_reference_s
        return target if improvement > self.improvement_threshold else sequence.current_idx


@dataclass
class PeriodicHOPolicy:
    """Apply Max-Elevation once every configured number of decisions."""

    period_steps: int = 16
    name: str = "Periodic-HO"
    _epoch: int = 0

    def reset(self) -> None:
        self._epoch = 0

    def choose(self, sequence: ForecastSequence) -> int:
        eligible = self._epoch % self.period_steps == 0
        self._epoch += 1
        if sequence.valid_mask[0, sequence.current_idx] and not eligible:
            return sequence.current_idx
        return MaxElevationPolicy().choose(sequence)


@dataclass
class SkipKPolicy:
    """Omit ``k`` distinct targets in the chronological Max-Elevation sequence.

    A skipped target remains skipped for its whole consecutive dominance
    interval.  Recommending the same satellite at the next decision therefore
    does not turn a Skip-1 rule into a one-epoch delay.  The next distinct
    Max-Elevation target advances the sequence counter.
    """

    skip: int = 1
    name: str = "Max-Elevation-Skip-k"
    _last_proposal: int | None = None
    _remaining: int = 0
    _last_proposal_was_skipped: bool = False

    def __post_init__(self) -> None:
        if self.skip < 1:
            raise ValueError("skip must be at least one")
        self.name = f"Max-Elevation-Skip-{self.skip}"
        self.reset()

    def reset(self) -> None:
        self._last_proposal = None
        self._remaining = self.skip
        self._last_proposal_was_skipped = False

    def choose(self, sequence: ForecastSequence) -> int:
        proposal = MaxElevationPolicy().choose(sequence)
        if proposal == sequence.current_idx or not sequence.valid_mask[
            0, sequence.current_idx
        ]:
            return proposal
        satellite = int(sequence.candidate_ids[proposal])
        if satellite != self._last_proposal:
            self._last_proposal = satellite
            if self._remaining > 0:
                self._remaining -= 1
                self._last_proposal_was_skipped = True
                return sequence.current_idx
            self._remaining = self.skip
            self._last_proposal_was_skipped = False
            return proposal
        if not self._last_proposal_was_skipped:
            # Retry an accepted target until the serving state catches up;
            # CHO qualification or a capacity rejection must not silently
            # convert the next call into a new skip event.
            return proposal
        # The same skipped target remains omitted.  Only a new distinct
        # target can consume the next skip/selection slot.  For k>1, seeing
        # it repeatedly still counts as one omitted target, not many.
        return sequence.current_idx


class NovaNetPolicy:
    name = "NovaNet"

    def __init__(
        self,
        config: NovaNetConfig,
        checkpoint: str | Path,
        *,
        allow_config_mismatch: bool = False,
        allowed_config_overrides: Iterable[str] = (),
        require_paper_eligible: bool = True,
        device: str | torch.device | None = None,
        ablations: Iterable[str] = (),
    ):
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        fingerprint = payload.get("config_fingerprint")
        if fingerprint != config.fingerprint:
            allowed = set(allowed_config_overrides)
            if allowed:
                checkpoint_config = payload.get("config")
                if not isinstance(checkpoint_config, dict):
                    raise ValueError(
                        "Checkpoint must store its full configuration before "
                        "a scoped runtime override can be validated"
                    )
                differences = _config_difference_paths(
                    checkpoint_config, asdict(config)
                )
                disallowed = differences - allowed
                if disallowed:
                    raise ValueError(
                        "Checkpoint/runtime config differs outside the "
                        f"declared override whitelist: {sorted(disallowed)}"
                    )
                if not differences:
                    raise ValueError(
                        "Checkpoint fingerprint differs although the stored "
                        "configuration has no declared runtime difference"
                    )
            elif not allow_config_mismatch:
                raise ValueError(
                    f"Checkpoint config {fingerprint!r} does not match selected "
                    f"config {config.fingerprint!r}"
                )
        training = payload.get("training")
        validation = payload.get("validation")
        required_training = (
            payload.get("checkpoint_kind") == "novanet"
            and payload.get("training_protocol")
            == "novanet_h6_softdp_sequence_v2"
            and isinstance(training, dict)
            and isinstance(validation, dict)
            and isinstance(payload.get("epoch"), int)
        )
        if not required_training:
            raise ValueError(
                "Checkpoint lacks the actual NovaNet training protocol, "
                "sample/epoch, or validation metadata"
            )
        training_fields = {
            "samples",
            "train_samples",
            "validation_samples",
            "epochs_requested",
            "epochs_completed",
            "training_complete",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "training_seed",
            "allow_stale_tle",
            "train_residual_labels",
            "validation_residual_labels",
            "train_hof_labels",
            "validation_hof_labels",
        }
        missing_training = training_fields - set(training)
        if missing_training:
            raise ValueError(
                "NovaNet checkpoint training metadata is incomplete: "
                f"{sorted(missing_training)}"
            )
        validation_total = validation.get("total")
        if not isinstance(validation_total, (int, float)) or not math.isfinite(
            float(validation_total)
        ):
            raise ValueError(
                "NovaNet checkpoint requires a finite validation total loss"
            )
        if (
            not bool(training["training_complete"])
            or int(training["epochs_completed"])
            != int(training["epochs_requested"])
        ):
            raise ValueError(
                "NovaNet checkpoint records incomplete training and cannot be loaded"
            )
        split = int(0.8 * config.training.num_samples)
        full_training = (
            int(training["samples"]) == config.training.num_samples
            and int(training["train_samples"]) == split
            and int(training["validation_samples"])
            == config.training.num_samples - split
            and int(training["epochs_requested"]) == config.training.epochs
            and int(training["epochs_completed"]) == config.training.epochs
            and int(training["batch_size"]) == config.training.batch_size
            and str(training["optimizer"]) == "AdamW"
            and math.isclose(
                float(training["learning_rate"]),
                config.training.learning_rate,
            )
            and math.isclose(float(training["weight_decay"]), 0.0)
            and int(training["training_seed"]) == config.experiment.seed
            and not bool(training["allow_stale_tle"])
            and int(training["train_residual_labels"]) > 0
            and int(training["validation_residual_labels"]) > 0
            and int(training["train_hof_labels"]) > 0
            and int(training["validation_hof_labels"]) > 0
        )
        self.paper_table_eligible = bool(
            payload.get("paper_table_eligible", False)
        )
        if self.paper_table_eligible and not full_training:
            raise ValueError(
                "Abbreviated, stale-TLE, or noncanonical NovaNet training "
                "cannot be marked paper_table_eligible"
            )
        if require_paper_eligible and not self.paper_table_eligible:
            raise ValueError(
                "Checkpoint is diagnostic or abbreviated and is not eligible "
                "for manuscript-table evaluation"
            )
        checkpoint_provenance = payload.get("tle_provenance", {})
        current_provenance = validate_tle_epoch(
            config,
            maximum_age_days=float("inf"),
        )
        required_provenance = (
            "tle_sha256",
            "tle_selection",
            "selected_satellites",
            "selected_names_sha256",
        )
        if any(
            checkpoint_provenance.get(key) is None
            for key in required_provenance
        ):
            raise ValueError(
                "Checkpoint does not record complete TLE/subset provenance; "
                "it cannot be used as the reproducible paper checkpoint"
            )
        if any(
            checkpoint_provenance[key] != current_provenance[key]
            for key in required_provenance
        ):
            raise ValueError(
                "Checkpoint TLE file or selected satellite subset does not "
                "match the current paper configuration"
            )
        self.model = NovaNet(config, ablations=ablations).to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()
        self.hidden_by_context: dict[object, dict[int, torch.Tensor]] = {}

    def reset(self) -> None:
        self.hidden_by_context.clear()

    def choose(self, sequence: ForecastSequence) -> int:
        return int(np.argmin(self.costs(sequence)))

    def scores(
        self,
        sequence: ForecastSequence,
        *,
        context: object = "default",
        load: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return higher-is-better scores equal to negative planner costs."""

        return -self.costs(sequence, context=context, load=load)

    def costs(
        self,
        sequence: ForecastSequence,
        *,
        context: object = "default",
        load: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the planner's comparable lower-is-better first-step costs."""

        def tensor(value, dtype=None):
            return torch.as_tensor(
                value,
                dtype=dtype or torch.float32,
                device=self.device,
            ).unsqueeze(0)

        hidden_by_satellite = self.hidden_by_context.setdefault(context, {})
        initial_hidden = torch.zeros(
            1,
            len(sequence.candidate_ids),
            self.config.model.hidden_dim,
            device=self.device,
        )
        for local, satellite_id in enumerate(sequence.candidate_ids):
            cached = hidden_by_satellite.get(int(satellite_id))
            if satellite_id >= 0 and cached is not None:
                initial_hidden[0, local] = cached

        load_tensor = None
        if load is not None:
            load_array = np.asarray(load, dtype=np.float32)
            expected = sequence.valid_mask.shape
            if load_array.shape != expected:
                raise ValueError(f"load must have shape {expected}")
            load_tensor = tensor(load_array)

        with torch.no_grad():
            output = self.model(
                tensor(sequence.node_features),
                tensor(sequence.spatial_adjacency),
                tensor(sequence.valid_mask, torch.bool),
                torch.tensor(
                    [sequence.current_idx], dtype=torch.long, device=self.device
                ),
                tensor(sequence.ttl_s),
                tensor(sequence.deterministic_snr_db),
                load=load_tensor,
                initial_hidden=initial_hidden,
                initial_freeze=torch.tensor(
                    [sequence.initial_freeze],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        current_hidden = output["hidden"][0, 0].detach()
        next_cache: dict[int, torch.Tensor] = {}
        for local, satellite_id in enumerate(sequence.candidate_ids):
            if satellite_id >= 0 and sequence.valid_mask[0, local]:
                next_cache[int(satellite_id)] = (
                    current_hidden[local].clone()
                )
        # Eq. (29) aligns only identities visible at consecutive decision
        # epochs.  A departed identity may be used once as the forced-
        # recovery source above, but it must not re-enter with an older cache.
        self.hidden_by_context[context] = next_cache
        return output["first_step_cost"][0].detach().cpu().numpy()
