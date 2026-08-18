"""Explicit learned baselines used by the comparison protocol.

These models are intentionally separate from :class:`novanet.model.NovaNet`.
Evaluation never obtains a DQN/DHO/GNN-only row by renaming a NovaNet
ablation.  Each baseline requires its own checkpoint, configuration
fingerprint, TLE provenance, and training metadata.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import NovaNetConfig
from .dataset import validate_tle_epoch
from .forecast import ForecastSequence
from .model import ResidualGraphLayer


BASELINE_KINDS = frozenset({"gnn_only", "dqn_gnn", "dho"})
BASELINE_NAMES = {
    "gnn_only": "GNN-only",
    "dqn_gnn": "DQN+GNN",
    "dho": "DHO",
}

# These identifiers are part of the checkpoint contract.  In particular, a
# feed-forward DHO-shaped surrogate must never be silently presented as the
# interaction-trained DHO protocol cited by the manuscript.
BASELINE_TRAINING_PROTOCOLS = {
    "gnn_only": "source_conditioned_one_step_cost_regression_v1",
    "dqn_gnn": "offline_sequential_double_dqn_fqi_v1",
    "dho": "source_conditioned_dho_surrogate_v1",
}
DQN_CANONICAL_DISCOUNT = 1.0
DQN_CANONICAL_TARGET_UPDATE_EPOCHS = 5
DQN_CANONICAL_VALIDATION_TRANSITIONS = 4096


def _source_target_context(
    hidden: torch.Tensor,
    current_idx: torch.Tensor,
) -> torch.Tensor:
    """Return candidate features conditioned on the serving satellite.

    A handover score is a source--target quantity.  Supplying only a candidate
    embedding makes the score invariant to the incumbent and therefore cannot
    represent the stay/switch cost used by the common CHO protocol.
    """

    batch, candidates, _width = hidden.shape
    if tuple(current_idx.shape) != (batch,):
        raise ValueError(f"current_idx must have shape {(batch,)}")
    if torch.any((current_idx < 0) | (current_idx >= candidates)):
        raise ValueError("current_idx contains an out-of-range candidate index")
    batch_index = torch.arange(batch, device=hidden.device)
    incumbent = hidden[batch_index, current_idx]
    incumbent = incumbent[:, None, :].expand(-1, candidates, -1)
    stay = F.one_hot(current_idx, num_classes=candidates).to(hidden.dtype)
    return torch.cat(
        (hidden, incumbent, (hidden - incumbent).abs(), stay[..., None]),
        dim=-1,
    )


class CurrentGraphEncoder(nn.Module):
    """Current-epoch graph encoder without recurrence or planning."""

    def __init__(self, config: NovaNetConfig):
        super().__init__()
        hidden = config.model.hidden_dim
        self.input = nn.Sequential(
            nn.Linear(config.model.node_feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        width = max(8, hidden // 4)
        self.query = nn.Linear(hidden, width, bias=False)
        self.key = nn.Linear(hidden, width, bias=False)
        self.layers = nn.ModuleList(
            ResidualGraphLayer(hidden)
            for _ in range(config.model.gnn_layers)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency_prior: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input(node_features)
        hidden = hidden * valid_mask[..., None].to(hidden.dtype)
        prior = adjacency_prior.clamp_min(0.0)
        pair_valid = (
            valid_mask[:, :, None]
            & valid_mask[:, None, :]
            & (prior > 0.0)
        )
        query = self.query(hidden)
        key = self.key(hidden)
        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / math.sqrt(query.shape[-1])
        logits = logits + torch.log(prior.clamp_min(1e-8))
        logits = logits.masked_fill(
            ~pair_valid, -torch.finfo(logits.dtype).max / 1e4
        )
        adjacency = torch.softmax(logits, dim=-1)
        adjacency = torch.where(pair_valid, adjacency, torch.zeros_like(adjacency))
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return hidden * valid_mask[..., None].to(hidden.dtype)


class GNNOnlySelector(nn.Module):
    def __init__(self, config: NovaNetConfig):
        super().__init__()
        self.encoder = CurrentGraphEncoder(config)
        hidden = config.model.hidden_dim
        pair_width = 3 * hidden + 1
        self.score = nn.Sequential(
            nn.Linear(pair_width, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, node, adjacency, valid, current_idx):
        hidden = self.encoder(node, adjacency, valid)
        pair = _source_target_context(hidden, current_idx)
        return self.score(pair).squeeze(-1)


class DQNGNNSelector(nn.Module):
    """Dueling one-step Q network on the current constellation graph."""

    def __init__(self, config: NovaNetConfig):
        super().__init__()
        self.encoder = CurrentGraphEncoder(config)
        hidden = config.model.hidden_dim
        pair_width = 3 * hidden + 1
        self.value = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(pair_width, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, node, adjacency, valid, current_idx):
        hidden = self.encoder(node, adjacency, valid)
        pair = _source_target_context(hidden, current_idx)
        weights = valid[..., None].to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        incumbent = hidden[batch_index, current_idx]
        value = self.value(torch.cat((pooled, incumbent), dim=-1))
        advantage = self.advantage(pair).squeeze(-1)
        mean_advantage = (
            (advantage * valid.to(advantage.dtype)).sum(dim=1, keepdim=True)
            / valid.sum(dim=1, keepdim=True).clamp_min(1)
        )
        return value + advantage - mean_advantage


class DHOSelector(nn.Module):
    """One-step handover network using candidate and incumbent context."""

    def __init__(self, config: NovaNetConfig):
        super().__init__()
        hidden = config.model.hidden_dim
        width = 2 * config.model.node_feature_dim + 1
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node, adjacency, valid, current_idx):
        del adjacency
        batch, candidates, _ = node.shape
        batch_index = torch.arange(batch, device=node.device)
        incumbent = node[batch_index, current_idx]
        incumbent = incumbent[:, None, :].expand(-1, candidates, -1)
        stay = F.one_hot(current_idx, num_classes=candidates).to(node.dtype)
        inputs = torch.cat((node, incumbent, stay[..., None]), dim=-1)
        return self.network(inputs).squeeze(-1)


def make_baseline_model(kind: str, config: NovaNetConfig) -> nn.Module:
    if kind == "gnn_only":
        return GNNOnlySelector(config)
    if kind == "dqn_gnn":
        return DQNGNNSelector(config)
    if kind == "dho":
        return DHOSelector(config)
    raise ValueError(f"Unknown learned baseline kind: {kind!r}")


class LearnedBaselinePolicy:
    """Checkpoint-validated deployment adapter for one learned baseline."""

    def __init__(
        self,
        config: NovaNetConfig,
        checkpoint: str | Path,
        *,
        expected_kind: str | None = None,
        device: str | torch.device | None = None,
        allow_config_mismatch: bool = False,
        allow_unqualified: bool = False,
    ):
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        kind = str(payload.get("baseline_kind", ""))
        if kind not in BASELINE_KINDS:
            raise ValueError("Checkpoint has no supported baseline_kind metadata")
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"Expected {expected_kind!r}, checkpoint contains {kind!r}")
        required_metadata = (
            "training_protocol",
            "paper_table_eligible",
            "source_fidelity",
            "training",
            "validation",
            "config",
            "epoch",
        )
        missing_metadata = [
            key for key in required_metadata if key not in payload
        ]
        if missing_metadata:
            raise ValueError(
                "Baseline checkpoint lacks qualification metadata: "
                f"{missing_metadata}"
            )
        for key in ("training", "validation", "config"):
            if not isinstance(payload[key], dict) or not payload[key]:
                raise ValueError(
                    f"Baseline checkpoint metadata {key!r} must be a nonempty mapping"
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
            "seed",
            "allow_stale_tle",
            "dqn_discount",
            "dqn_updates_per_epoch",
            "dqn_target_update_epochs",
            "dqn_validation_transitions",
            "replay_transitions",
        }
        missing_training = training_fields - set(payload["training"])
        if missing_training:
            raise ValueError(
                "Baseline checkpoint training metadata is incomplete: "
                f"{sorted(missing_training)}"
            )
        validation_loss = payload["validation"].get("loss")
        if not isinstance(validation_loss, (int, float)) or not math.isfinite(
            float(validation_loss)
        ):
            raise ValueError(
                "Baseline checkpoint requires a finite validation loss"
            )
        if (
            bool(payload["training"]["allow_stale_tle"])
            and bool(payload["paper_table_eligible"])
        ):
            raise ValueError(
                "A stale-TLE diagnostic checkpoint cannot be table eligible"
            )
        if (
            not bool(payload["training"]["training_complete"])
            or int(payload["training"]["epochs_completed"])
            != int(payload["training"]["epochs_requested"])
        ):
            raise ValueError(
                "Baseline checkpoint records incomplete training and cannot be loaded"
            )
        full_training = (
            int(payload["training"]["samples"])
            == config.training.num_samples
            and int(payload["training"]["train_samples"])
            == int(0.8 * config.training.num_samples)
            and int(payload["training"]["validation_samples"])
            == config.training.num_samples
            - int(0.8 * config.training.num_samples)
            and int(payload["training"]["epochs_requested"])
            == config.training.epochs
            and int(payload["training"]["epochs_completed"])
            == config.training.epochs
            and int(payload["training"]["batch_size"])
            == config.training.batch_size
            and str(payload["training"]["optimizer"]) == "AdamW"
            and math.isclose(
                float(payload["training"]["learning_rate"]),
                config.training.learning_rate,
            )
            and math.isclose(
                float(payload["training"]["weight_decay"]),
                config.training.weight_decay,
            )
            and int(payload["training"]["seed"])
            == config.experiment.seed
            and not bool(payload["training"]["allow_stale_tle"])
        )
        if kind == "dqn_gnn":
            expected_updates = math.ceil(
                int(0.8 * config.training.num_samples)
                / config.training.batch_size
            )
            dqn_full = (
                math.isclose(
                    float(payload["training"]["dqn_discount"]),
                    DQN_CANONICAL_DISCOUNT,
                )
                and int(payload["training"]["dqn_updates_per_epoch"])
                == expected_updates
                and int(payload["training"]["dqn_target_update_epochs"])
                == DQN_CANONICAL_TARGET_UPDATE_EPOCHS
                and int(payload["training"]["dqn_validation_transitions"])
                == DQN_CANONICAL_VALIDATION_TRANSITIONS
                and int(payload["training"]["replay_transitions"]) > 0
            )
            full_training = full_training and dqn_full
        if bool(payload["paper_table_eligible"]) and not full_training:
            raise ValueError(
                "Abbreviated baseline training cannot be marked "
                "paper_table_eligible"
            )
        expected_protocol = BASELINE_TRAINING_PROTOCOLS.get(kind)
        protocol = str(payload["training_protocol"])
        if (
            kind in {"gnn_only", "dqn_gnn"}
            and protocol != expected_protocol
            and payload.get("source_fidelity") != "verified_cited_protocol"
        ):
            raise ValueError(
                f"Unsupported {kind} training protocol {protocol!r}; expected "
                f"{expected_protocol!r} or verified_cited_protocol metadata"
            )
        fidelity = str(payload["source_fidelity"])
        if kind == "dho" and fidelity != "verified_cited_protocol":
            if not allow_unqualified:
                raise ValueError(
                    "This DHO checkpoint is a repository surrogate, not a "
                    "verified reproduction of the cited interaction-trained "
                    "DHO protocol; pass allow_unqualified=True only for a "
                    "diagnostic run"
                )
        if (
            not allow_unqualified
            and fidelity
            not in {"paper_defined_repository_baseline", "verified_cited_protocol"}
        ):
            raise ValueError(
                f"Unqualified baseline source_fidelity={fidelity!r}"
            )
        if not bool(payload["paper_table_eligible"]) and not allow_unqualified:
            raise ValueError(
                "Baseline checkpoint is not qualified for a manuscript table"
            )
        if (
            payload.get("config_fingerprint") != config.fingerprint
            and not allow_config_mismatch
        ):
            raise ValueError("Baseline checkpoint/config fingerprint mismatch")
        current = validate_tle_epoch(config, maximum_age_days=float("inf"))
        provenance = payload.get("tle_provenance", {})
        for key in (
            "tle_sha256",
            "tle_selection",
            "selected_satellites",
            "selected_names_sha256",
        ):
            if provenance.get(key) != current.get(key):
                raise ValueError(f"Baseline checkpoint TLE provenance mismatch: {key}")
        self.kind = kind
        self.name = BASELINE_NAMES[kind]
        self.model = make_baseline_model(kind, config).to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()

    def reset(self) -> None:
        return None

    def scores(self, sequence: ForecastSequence) -> np.ndarray:
        valid = sequence.valid_mask[0]
        if sequence.initial_freeze > 0 and valid[sequence.current_idx]:
            scores = np.full(len(valid), -np.inf, dtype=float)
            scores[sequence.current_idx] = 0.0
            return scores
        with torch.inference_mode():
            node = torch.as_tensor(
                sequence.node_features[0], dtype=torch.float32, device=self.device
            )[None]
            adjacency = torch.as_tensor(
                sequence.spatial_adjacency[0],
                dtype=torch.float32,
                device=self.device,
            )[None]
            mask = torch.as_tensor(valid, dtype=torch.bool, device=self.device)[None]
            incumbent = torch.tensor(
                [sequence.current_idx], dtype=torch.long, device=self.device
            )
            score = self.model(node, adjacency, mask, incumbent)[0]
            score = score.masked_fill(~mask[0], -torch.inf)
        return score.cpu().numpy()

    def choose(self, sequence: ForecastSequence) -> int:
        return int(np.argmax(self.scores(sequence)))
