"""Candidate-ranking policies sharing the same forecast information budget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch

from .config import NovaNetConfig
from .dataset import validate_tle_epoch
from .forecast import ForecastSequence
from .model import NovaNet


class Policy(Protocol):
    name: str

    def choose(self, sequence: ForecastSequence) -> int:
        """Return a local candidate index."""


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
    dwell_weight: float = 0.05
    switch_penalty: float = 2.0
    name: str = "Rate-Dwell"

    def choose(self, sequence: ForecastSequence) -> int:
        score = (
            self.rate_weight * sequence.deterministic_snr_db[0]
            + self.dwell_weight * sequence.ttl_s[0]
        )
        score = score.astype(float)
        score -= self.switch_penalty * (
            np.arange(len(score)) != sequence.current_idx
        )
        score[~sequence.valid_mask[0]] = -np.inf
        return int(np.argmax(score))


class NovaNetPolicy:
    name = "NovaNet"

    def __init__(
        self,
        config: NovaNetConfig,
        checkpoint: str | Path,
        *,
        allow_config_mismatch: bool = False,
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
        if fingerprint != config.fingerprint and not allow_config_mismatch:
            raise ValueError(
                f"Checkpoint config {fingerprint!r} does not match selected "
                f"config {config.fingerprint!r}"
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
        return int(np.argmax(self.scores(sequence)))

    def scores(
        self,
        sequence: ForecastSequence,
        *,
        context: object = "default",
        load: np.ndarray | None = None,
    ) -> np.ndarray:
        if np.any(~sequence.valid_mask.any(axis=1)):
            valid = sequence.valid_mask[0]
            if not np.any(valid):
                raise RuntimeError(
                    "No current candidate is eligible for the deployment "
                    "fail-safe"
                )
            score = (
                sequence.deterministic_snr_db[0].astype(float)
                + 0.05 * sequence.ttl_s[0].astype(float)
            )
            finite = score[valid]
            scale = max(float(finite.std()), 1e-6)
            score[valid] = (finite - float(finite.mean())) / scale
            score[valid] -= self.config.planner.base_switch_cost * (
                np.flatnonzero(valid) != sequence.current_idx
            )
            if load is not None:
                load_array = np.asarray(load, dtype=np.float32)
                if load_array.shape != sequence.valid_mask.shape:
                    raise ValueError(
                        f"load must have shape {sequence.valid_mask.shape}"
                    )
                score[valid] -= (
                    self.config.planner.load_weight
                    * load_array[0, valid]
                )
            score[~valid] = -np.inf
            return score

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
                tensor(sequence.angular_speed_deg_s),
                load=load_tensor,
                initial_hidden=initial_hidden,
            )
        current_hidden = output["hidden"][0, 0].detach()
        for local, satellite_id in enumerate(sequence.candidate_ids):
            if satellite_id >= 0 and sequence.valid_mask[0, local]:
                hidden_by_satellite[int(satellite_id)] = (
                    current_hidden[local].clone()
                )
        return output["q_next"][0].detach().cpu().numpy()
