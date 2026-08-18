"""Spatio-temporal encoder, calibrated risk heads, and finite-horizon planner."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .config import NovaNetConfig, load_config
from .soft_dp import SoftDP


VALID_ABLATIONS = frozenset(
    {
        "OrbitPrior",
        "DynAdj",
        "Temporal",
        "Planner",
        "UncLCB",
        "TransTTL",
        "TransHOF",
    }
)


class ResidualGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(adjacency, self.message(hidden))
        return self.norm(hidden + F.gelu(self.update(message)))


class EnergyHead(nn.Module):
    """Fixed-reference implementation of manuscript Eqs. (15)--(17)."""

    def __init__(
        self,
        cfg: NovaNetConfig,
        ablations: frozenset[str] = frozenset(),
    ):
        super().__init__()
        planner = cfg.planner
        self.ablations = ablations
        self.bandwidth_hz = float(cfg.channel.bandwidth_hz)
        self.efficiency = float(cfg.channel.implementation_efficiency)
        self.lcb_kappa = float(planner.lcb_kappa)
        self.rate_reference_mbps = float(planner.rate_reference_mbps)
        self.ttl_reference_s = float(planner.ttl_reference_s)
        self.alpha = float(planner.alpha)
        self.beta = float(planner.beta)
        self.c0 = float(planner.c0)
        self.c1 = float(planner.c1)
        self.c2 = float(planner.c2)
        self.load_weight = float(planner.load_weight)

    def lcb_rate_mbps(
        self,
        nominal_snr_db: torch.Tensor,
        residual_mu: torch.Tensor,
        residual_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Map the log-SINR residual LCB of Eqs. (37)--(40) to Mbps."""

        kappa = 0.0 if "UncLCB" in self.ablations else self.lcb_kappa
        residual_lcb = residual_mu - kappa * residual_sigma
        lcb_db = nominal_snr_db + (10.0 / math.log(10.0)) * residual_lcb
        return self.rate_from_snr_db(lcb_db)

    def rate_from_snr_db(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured physical-rate mapping for an SNR in dB."""

        snr_linear = torch.pow(10.0, snr_db / 10.0)
        return (
            self.efficiency
            * self.bandwidth_hz
            * torch.log2(1.0 + snr_linear)
            / 1e6
        )

    def state_cost(
        self,
        nominal_snr_db: torch.Tensor,
        residual_mu: torch.Tensor,
        residual_sigma: torch.Tensor,
        ttl_s: torch.Tensor,
        current_snr_db: torch.Tensor,
        load: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rate_mbps = self.lcb_rate_mbps(
            nominal_snr_db, residual_mu, residual_sigma
        )
        # Manuscript Eq. (44): h=0 consumes the current measured link quality;
        # residual forecasting and its LCB are used only for h >= 1.
        rate_mbps = rate_mbps.clone()
        rate_mbps[:, 0, :] = self.rate_from_snr_db(current_snr_db)
        normalized_rate = rate_mbps / self.rate_reference_mbps
        normalized_ttl = ttl_s / self.ttl_reference_s
        if load is None:
            load = torch.zeros_like(rate_mbps)
        cost = (
            -self.alpha * normalized_rate
            - self.beta * normalized_ttl
            + self.load_weight * load
        )
        return cost, {
            "lcb_rate_mbps": rate_mbps,
            "normalized_rate": normalized_rate,
            "normalized_ttl": normalized_ttl,
            "load": load,
        }

    def transition_cost(
        self,
        ttl_s: torch.Tensor,
        hof_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return [B,H,K,K] cost; every diagonal stay transition is exactly 0."""

        batch, horizon, candidates = ttl_s.shape
        # T_h(i,j) is evaluated at decision time h: source i is the incumbent
        # carried into that time and target j is the candidate entered there.
        ttl_source = ttl_s[:, :, :, None].expand(
            batch, horizon, candidates, candidates
        )
        normalized_source_ttl = ttl_source / self.ttl_reference_s
        switch = (
            1.0
            - torch.eye(
                candidates,
                dtype=ttl_s.dtype,
                device=ttl_s.device,
            )[None, None, :, :]
        )
        retained_weight = 0.0 if "TransTTL" in self.ablations else self.c1
        hof_weight = 0.0 if "TransHOF" in self.ablations else self.c2
        cost = switch * (
            self.c0
            + retained_weight * normalized_source_ttl
            + hof_weight * hof_probability
        )
        return cost, {
            "switch_indicator": switch,
            "base": switch * self.c0,
            "normalized_source_ttl": switch * normalized_source_ttl,
            "hof_probability": switch * hof_probability,
        }


class NovaNet(nn.Module):
    """Sequence model whose forward pass is the paper's deployed planner."""

    def __init__(
        self,
        config: NovaNetConfig | None = None,
        ablations: Iterable[str] = (),
    ):
        super().__init__()
        self.config = config or load_config()
        self.ablations = frozenset(ablations)
        unknown = self.ablations - VALID_ABLATIONS
        if unknown:
            raise ValueError(f"Unknown ablations: {sorted(unknown)}")
        model_cfg = self.config.model
        planner_cfg = self.config.planner
        hidden = model_cfg.hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(model_cfg.node_feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.graph_layers = nn.ModuleList(
            ResidualGraphLayer(hidden) for _ in range(model_cfg.gnn_layers)
        )
        attention_width = max(8, hidden // 4)
        self.adjacency_query = nn.Linear(hidden, attention_width, bias=False)
        self.adjacency_key = nn.Linear(hidden, attention_width, bias=False)
        self.temporal_gru = nn.GRUCell(hidden, hidden)

        def scalar_head(positive: bool = False) -> nn.Module:
            layers: list[nn.Module] = [
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            ]
            if positive:
                layers.append(nn.Softplus())
            return nn.Sequential(*layers)

        self.residual_mean_head = scalar_head()
        self.residual_logvar_head = scalar_head()

        pair_dim = 3 * hidden + model_cfg.transition_feature_dim
        self.hof_head = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.energy = EnergyHead(self.config, self.ablations)
        self.planner = SoftDP(
            horizon=planner_cfg.horizon_steps,
            temperature=planner_cfg.temperature,
            freeze_steps=self.config.handover.freeze_steps,
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def _encode(
        self,
        node_features: torch.Tensor,
        spatial_adjacency: torch.Tensor,
        valid_mask: torch.Tensor,
        initial_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, horizon, candidates, _ = node_features.shape
        # The current graph consumes the complete six-dimensional measured
        # feature vector.  Future states are not separate observed graphs:
        # they contain only the five SGP4/TTL features and are rolled through
        # the same identity-aligned GRU, as specified in Sec. III-B.
        current = self.node_encoder(node_features[:, 0])
        current = current * valid_mask[:, 0, :, None].to(current.dtype)

        prior = spatial_adjacency[:, 0].clamp_min(0.0)
        pair_valid = (
            valid_mask[:, 0, :, None]
            & valid_mask[:, 0, None, :]
            & (prior > 0.0)
        )
        if "DynAdj" in self.ablations:
            adjacency = prior
        else:
            query = self.adjacency_query(current)
            key = self.adjacency_key(current)
            logits = torch.matmul(query, key.transpose(-1, -2))
            logits = logits / math.sqrt(query.shape[-1])
            logits = logits + torch.log(prior.clamp_min(1e-8))
            logits = logits.masked_fill(
                ~pair_valid,
                -torch.finfo(logits.dtype).max / 1e4,
            )
            adjacency = torch.softmax(logits, dim=-1)
        adjacency = adjacency * pair_valid.to(adjacency.dtype)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        spatial = current
        for layer in self.graph_layers:
            spatial = layer(spatial, adjacency)

        first_linear = self.node_encoder[0]
        future = F.linear(
            node_features[:, 1:, :, :5],
            first_linear.weight[:, :5],
            first_linear.bias,
        )
        future = self.node_encoder[1](future)
        future = self.node_encoder[2](future)
        future = future * valid_mask[:, 1:, :, None].to(future.dtype)

        if initial_hidden is None:
            recurrent = torch.zeros(
                batch,
                candidates,
                spatial.shape[-1],
                dtype=spatial.dtype,
                device=spatial.device,
            )
        else:
            expected = (batch, candidates, spatial.shape[-1])
            if tuple(initial_hidden.shape) != expected:
                raise ValueError(f"initial_hidden must have shape {expected}")
            recurrent = initial_hidden.to(
                dtype=spatial.dtype,
                device=spatial.device,
            )
        recurrent = recurrent.reshape(batch * candidates, -1)
        outputs = []
        for t in range(horizon):
            encoded_t = (
                spatial if t == 0 else future[:, t - 1]
            ).reshape(batch * candidates, -1)
            if "Temporal" in self.ablations:
                updated = encoded_t
            else:
                updated = self.temporal_gru(encoded_t, recurrent)
            mask_t = valid_mask[:, t].reshape(batch * candidates, 1)
            # An invisible incumbent keeps its last identity-cached state for
            # forced-recovery pair scoring at this step.  It is not carried as
            # the GRU predecessor for a future reappearance, which preserves
            # the zero-birth rule of the horizon rollout.
            visible_output = torch.where(mask_t, updated, recurrent)
            outputs.append(visible_output.reshape(batch, candidates, -1))
            recurrent = torch.where(
                mask_t, visible_output, torch.zeros_like(visible_output)
            )
        return torch.stack(outputs, dim=1)

    def _transition_features(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, candidates, width = hidden.shape
        source = hidden[:, :, :, None, :].expand(
            batch, horizon, candidates, candidates, width
        )
        target = hidden[:, :, None, :, :].expand(
            batch, horizon, candidates, candidates, width
        )

        shape = (batch, horizon, candidates, candidates)
        reference = hidden.new_zeros(shape)
        ttt = torch.full_like(reference, self.config.handover.ttt_s)
        execution = torch.full_like(reference, self.config.handover.execution_s)
        hysteresis = torch.full_like(
            reference, self.config.handover.hysteresis_db
        )
        threshold = torch.full_like(
            reference, self.config.channel.outage_threshold_db
        )
        failure_fraction = torch.full_like(
            reference, self.config.handover.failure_outage_fraction
        )
        context = torch.stack(
            (ttt, execution, hysteresis, threshold, failure_fraction), dim=-1
        )
        pair = torch.cat((source, target, (target - source).abs(), context), dim=-1)
        return pair

    def forward(
        self,
        node_features: torch.Tensor,
        spatial_adjacency: torch.Tensor,
        valid_mask: torch.Tensor,
        current_idx: torch.Tensor,
        ttl_s: torch.Tensor,
        nominal_snr_db: torch.Tensor,
        load: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
        initial_freeze: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Plan from an explicit future candidate/feature sequence.

        Args:
            node_features: ``[B,H,K,6]`` features from manuscript Eq. (24).
            spatial_adjacency: ``[B,H,K,K]`` sky-dome adjacency.
            valid_mask: ``[B,H,K]`` visibility/candidate mask.
            current_idx: ``[B]`` incumbent index in the aligned candidate union.
            ttl_s: ``[B,H,K]`` TTL propagated directly by TLE/SGP4.
            nominal_snr_db: ``[B,H,K]`` deterministic nominal link SINR.
            load: optional normalized ``[B,H,K]`` load forecast.
            initial_hidden: optional incumbent-history state ``[B,K,d]``
                aligned by global satellite identifier.
            initial_freeze: optional actual CHO freeze counter ``[B]``.
        """

        if node_features.ndim != 4:
            raise ValueError("node_features must have shape [B,H,K,F]")
        batch, horizon, candidates, feature_dim = node_features.shape
        expected = self.config
        if horizon != expected.planner.horizon_steps:
            raise ValueError(
                f"Expected H={expected.planner.horizon_steps}, got H={horizon}"
            )
        if candidates > expected.experiment.candidate_cap:
            raise ValueError(
                f"K={candidates} exceeds configured cap "
                f"{expected.experiment.candidate_cap}"
            )
        if feature_dim != expected.model.node_feature_dim:
            raise ValueError(
                f"Expected F={expected.model.node_feature_dim}, got F={feature_dim}"
            )
        if tuple(valid_mask.shape) != (batch, horizon, candidates):
            raise ValueError("valid_mask shape does not match node_features")
        if tuple(ttl_s.shape) != (batch, horizon, candidates):
            raise ValueError("ttl_s shape does not match node_features")
        if tuple(nominal_snr_db.shape) != (batch, horizon, candidates):
            raise ValueError("nominal_snr_db shape does not match node_features")

        current_snr_db = (
            node_features[:, 0, :, 5] * self.config.model.sinr_reference_db
        )

        if "OrbitPrior" in self.ablations:
            node_features = node_features.clone()
            node_features[..., :5] = 0.0
        hidden = self._encode(
            node_features,
            spatial_adjacency,
            valid_mask,
            initial_hidden,
        )
        residual_mu = self.residual_mean_head(hidden).squeeze(-1)
        residual_scale_raw = self.residual_logvar_head(hidden).squeeze(-1)
        residual_sigma = F.softplus(residual_scale_raw) + 1e-4
        residual_logvar = 2.0 * torch.log(residual_sigma)

        pair = self._transition_features(hidden)
        hof_logits = self.hof_head(pair).squeeze(-1)
        hof_probability = torch.sigmoid(hof_logits)
        diagonal = torch.eye(
            candidates, dtype=torch.bool, device=hof_probability.device
        )[None, None]
        hof_probability = hof_probability.masked_fill(diagonal, 0.0)

        state_cost, state_components = self.energy.state_cost(
            nominal_snr_db,
            residual_mu,
            residual_sigma,
            ttl_s,
            current_snr_db,
            load,
        )
        transition_cost, transition_components = self.energy.transition_cost(
            ttl_s, hof_probability
        )
        if "Planner" in self.ablations:
            batch_index = torch.arange(batch, device=state_cost.device)
            first_cost = (
                transition_cost[batch_index, 0, current_idx, :]
                + state_cost[:, 0, :]
            )
            first_allowed = valid_mask[:, 0, :].clone()
            if initial_freeze is not None:
                freeze = initial_freeze.to(
                    device=first_cost.device, dtype=torch.long
                )
                locked = (freeze > 0) & valid_mask[
                    batch_index, 0, current_idx
                ]
                stay = F.one_hot(
                    current_idx, num_classes=candidates
                ).to(dtype=torch.bool)
                first_allowed = first_allowed & (
                    ~locked[:, None] | stay
                )
            first_logits = (
                -first_cost / self.config.planner.temperature
            ).masked_fill(~first_allowed, -torch.inf)
            first_action = torch.softmax(first_logits, dim=-1)
            cost_to_go = state_cost
            conditional_policy = torch.zeros_like(transition_cost)
            first_step_cost = first_cost.masked_fill(
                ~first_allowed, torch.inf
            )
        else:
            planner_result = self.planner(
                state_cost,
                transition_cost,
                current_idx,
                valid_mask,
                initial_freeze=initial_freeze,
                return_details=True,
            )
            first_action = planner_result.first_action
            cost_to_go = planner_result.cost_to_go
            conditional_policy = planner_result.conditional_policy
            first_step_cost = planner_result.first_cost
            first_action = torch.softmax(
                -first_step_cost / self.config.planner.policy_temperature,
                dim=-1,
            )
        return {
            "hidden": hidden,
            "residual_mu": residual_mu,
            "residual_scale_raw": residual_scale_raw,
            "residual_sigma": residual_sigma,
            "residual_logvar": residual_logvar,
            "ttl_s": ttl_s,
            "hof_logits": hof_logits,
            "hof_probability": hof_probability,
            "state_cost": state_cost,
            "transition_cost": transition_cost,
            "q_next": first_action,
            "first_step_cost": first_step_cost,
            "cost_to_go": cost_to_go,
            "conditional_policy": conditional_policy,
            "state_components": state_components,
            "transition_components": transition_components,
        }
