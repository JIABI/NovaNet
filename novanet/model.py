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
        "TransVel",
        "TransHOF",
    }
)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class FrozenEnergyNormalizer(nn.Module):
    """Training-split statistics stored in, and restored from, checkpoints."""

    def __init__(self):
        super().__init__()
        self.register_buffer("rate_mean", torch.tensor(50.0))
        self.register_buffer("rate_std", torch.tensor(20.0))
        self.register_buffer("ttl_mean", torch.tensor(180.0))
        self.register_buffer("ttl_std", torch.tensor(120.0))
        self.register_buffer("angular_speed_mean", torch.tensor(0.05))
        self.register_buffer("angular_speed_std", torch.tensor(0.05))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        rate_mbps: torch.Tensor,
        ttl_s: torch.Tensor,
        angular_speed_deg_s: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if valid_mask is None:
            valid_mask = torch.ones_like(rate_mbps, dtype=torch.bool)
        valid_mask = valid_mask.bool()
        if valid_mask.sum() < 2:
            raise ValueError("At least two valid observations are needed to fit stats")

        def assign(name: str, values: torch.Tensor) -> None:
            selected = values[valid_mask].detach().float()
            mean = selected.mean()
            std = selected.std(unbiased=False).clamp_min(1e-4)
            getattr(self, f"{name}_mean").copy_(mean)
            getattr(self, f"{name}_std").copy_(std)

        assign("rate", rate_mbps)
        assign("ttl", ttl_s)
        assign("angular_speed", angular_speed_deg_s.abs())
        self.is_fitted.fill_(True)

    def z_rate(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.rate_mean) / self.rate_std.clamp_min(1e-4)

    def z_ttl(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.ttl_mean) / self.ttl_std.clamp_min(1e-4)

    def z_angular_speed(self, value: torch.Tensor) -> torch.Tensor:
        return (
            value.abs() - self.angular_speed_mean
        ) / self.angular_speed_std.clamp_min(1e-4)


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
    """Dimensionless state and transition costs with explicit switch gating."""

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
        self.normalizer = FrozenEnergyNormalizer()

        self.raw_rate = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.rate_weight))
        )
        self.raw_dwell = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.dwell_weight))
        )
        self.raw_base = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.base_switch_cost))
        )
        self.raw_retained_dwell = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.retained_dwell_weight))
        )
        self.raw_velocity = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.angular_speed_weight))
        )
        self.raw_hof = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.hof_weight))
        )
        # The multi-UE extension is evaluated zero-shot, so its common load
        # coefficient is a frozen validation-selected control knob rather than
        # a parameter silently trained on all-zero single-UE loads.
        self.raw_load = nn.Parameter(
            torch.tensor(_inverse_softplus(planner.load_weight)),
            requires_grad=False,
        )

    @staticmethod
    def positive(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def lcb_rate_mbps(
        self, snr_mu_db: torch.Tensor, snr_logvar_db2: torch.Tensor
    ) -> torch.Tensor:
        sigma_db = torch.exp(0.5 * snr_logvar_db2.clamp(-12.0, 12.0))
        kappa = 0.0 if "UncLCB" in self.ablations else self.lcb_kappa
        lcb_db = snr_mu_db - kappa * sigma_db
        snr_linear = torch.pow(10.0, lcb_db / 10.0)
        return (
            self.efficiency
            * self.bandwidth_hz
            * torch.log2(1.0 + snr_linear)
            / 1e6
        )

    def state_cost(
        self,
        snr_mu_db: torch.Tensor,
        snr_logvar_db2: torch.Tensor,
        ttl_s: torch.Tensor,
        load: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rate_mbps = self.lcb_rate_mbps(snr_mu_db, snr_logvar_db2)
        z_rate = self.normalizer.z_rate(rate_mbps)
        z_ttl = self.normalizer.z_ttl(ttl_s)
        if load is None:
            load = torch.zeros_like(rate_mbps)
        cost = (
            -self.positive(self.raw_rate) * z_rate
            - self.positive(self.raw_dwell) * z_ttl
            + self.positive(self.raw_load) * load
        )
        return cost, {
            "lcb_rate_mbps": rate_mbps,
            "z_rate": z_rate,
            "z_ttl": z_ttl,
            "load": load,
        }

    def transition_cost(
        self,
        ttl_s: torch.Tensor,
        angular_speed_deg_s: torch.Tensor,
        hof_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return [B,H,K,K] cost; every diagonal stay transition is exactly 0."""

        batch, horizon, candidates = ttl_s.shape
        # T_h(i,j) is evaluated at decision time h: source i is the incumbent
        # carried into that time and target j is the candidate entered there.
        ttl_source = ttl_s[:, :, :, None].expand(
            batch, horizon, candidates, candidates
        )
        omega_source = angular_speed_deg_s[:, :, :, None]
        omega_target = angular_speed_deg_s[:, :, None, :]
        relative_omega = (omega_target - omega_source).abs()

        retained_dwell = torch.sigmoid(self.normalizer.z_ttl(ttl_source))
        angular_term = torch.sigmoid(
            self.normalizer.z_angular_speed(relative_omega)
        )
        switch = (
            1.0
            - torch.eye(
                candidates,
                dtype=ttl_s.dtype,
                device=ttl_s.device,
            )[None, None, :, :]
        )
        base = self.positive(self.raw_base)
        retained_weight = (
            torch.zeros_like(self.raw_retained_dwell)
            if "TransTTL" in self.ablations
            else self.positive(self.raw_retained_dwell)
        )
        velocity_weight = (
            torch.zeros_like(self.raw_velocity)
            if "TransVel" in self.ablations
            else self.positive(self.raw_velocity)
        )
        hof_weight = (
            torch.zeros_like(self.raw_hof)
            if "TransHOF" in self.ablations
            else self.positive(self.raw_hof)
        )
        cost = switch * (
            base
            + retained_weight * retained_dwell
            + velocity_weight * angular_term
            + hof_weight * hof_probability
        )
        return cost, {
            "switch_indicator": switch,
            "base": switch * base,
            "retained_dwell": switch * retained_dwell,
            "angular_speed": switch * angular_term,
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

        self.snr_mean_head = scalar_head()
        self.snr_logvar_head = scalar_head()
        self.ttl_head = scalar_head(positive=True)
        self.selection_head = scalar_head()

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
        hidden = self.node_encoder(node_features)
        valid_float = valid_mask[..., None].to(hidden.dtype)
        hidden = hidden * valid_float

        prior = spatial_adjacency.clamp_min(0.0)
        pair_valid = (
            valid_mask[:, :, :, None]
            & valid_mask[:, :, None, :]
            & (prior > 0.0)
        )
        if "DynAdj" in self.ablations:
            adjacency = prior
        else:
            query = self.adjacency_query(hidden)
            key = self.adjacency_key(hidden)
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

        flat_hidden = hidden.reshape(batch * horizon, candidates, -1)
        flat_adj = adjacency.reshape(batch * horizon, candidates, candidates)
        for layer in self.graph_layers:
            flat_hidden = layer(flat_hidden, flat_adj)
        spatial = flat_hidden.reshape(batch, horizon, candidates, -1)

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
            current = spatial[:, t].reshape(batch * candidates, -1)
            if "Temporal" in self.ablations:
                recurrent = current
            else:
                recurrent = self.temporal_gru(current, recurrent)
            mask_t = valid_mask[:, t].reshape(batch * candidates, 1)
            recurrent = torch.where(mask_t, recurrent, torch.zeros_like(recurrent))
            outputs.append(recurrent.reshape(batch, candidates, -1))
        return torch.stack(outputs, dim=1)

    def _transition_features(
        self,
        hidden: torch.Tensor,
        angular_speed_deg_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, candidates, width = hidden.shape
        source = hidden[:, :, :, None, :].expand(
            batch, horizon, candidates, candidates, width
        )
        target = hidden[:, :, None, :, :].expand(
            batch, horizon, candidates, candidates, width
        )

        relative_omega = (
            angular_speed_deg_s[:, :, None, :]
            - angular_speed_deg_s[:, :, :, None]
        ).abs()
        ttt = torch.full_like(relative_omega, self.config.handover.ttt_s)
        hysteresis = torch.full_like(
            relative_omega, self.config.handover.hysteresis_db
        )
        context = torch.stack((relative_omega, ttt, hysteresis), dim=-1)
        pair = torch.cat((source, target, (target - source).abs(), context), dim=-1)
        return pair, relative_omega

    def forward(
        self,
        node_features: torch.Tensor,
        spatial_adjacency: torch.Tensor,
        valid_mask: torch.Tensor,
        current_idx: torch.Tensor,
        angular_speed_deg_s: torch.Tensor,
        load: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Plan from an explicit future candidate/feature sequence.

        Args:
            node_features: ``[B,H,K,9]`` causal feature forecasts.
            spatial_adjacency: ``[B,H,K,K]`` sky-dome adjacency.
            valid_mask: ``[B,H,K]`` visibility/candidate mask.
            current_idx: ``[B]`` incumbent index in the aligned candidate union.
            angular_speed_deg_s: ``[B,H,K]`` apparent angular speed.
            load: optional normalized ``[B,H,K]`` load forecast.
            initial_hidden: optional incumbent-history state ``[B,K,d]``
                aligned by global satellite identifier.
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
        if tuple(angular_speed_deg_s.shape) != (batch, horizon, candidates):
            raise ValueError("angular_speed_deg_s shape does not match node_features")

        if "OrbitPrior" in self.ablations:
            node_features = node_features.clone()
            node_features[..., :6] = 0.0
        hidden = self._encode(
            node_features,
            spatial_adjacency,
            valid_mask,
            initial_hidden,
        )
        snr_mu = self.snr_mean_head(hidden).squeeze(-1)
        snr_logvar = self.snr_logvar_head(hidden).squeeze(-1).clamp(-10.0, 8.0)
        ttl_s = self.ttl_head(hidden).squeeze(-1)
        selection_logits = self.selection_head(hidden).squeeze(-1)

        pair, relative_omega = self._transition_features(
            hidden, angular_speed_deg_s
        )
        hof_logits = self.hof_head(pair).squeeze(-1)
        hof_probability = torch.sigmoid(hof_logits)

        state_cost, state_components = self.energy.state_cost(
            snr_mu, snr_logvar, ttl_s, load
        )
        transition_cost, transition_components = self.energy.transition_cost(
            ttl_s, angular_speed_deg_s, hof_probability
        )
        if "Planner" in self.ablations:
            batch_index = torch.arange(batch, device=state_cost.device)
            first_cost = (
                transition_cost[batch_index, 0, current_idx, :]
                + state_cost[:, 0, :]
            )
            first_logits = (-first_cost / self.config.planner.temperature).masked_fill(
                ~valid_mask[:, 0, :],
                -torch.finfo(first_cost.dtype).max / 1e4,
            )
            first_action = torch.softmax(first_logits, dim=-1)
            cost_to_go = state_cost
            conditional_policy = torch.zeros_like(transition_cost)
        else:
            planner_result = self.planner(
                state_cost,
                transition_cost,
                current_idx,
                valid_mask,
                return_details=True,
            )
            first_action = planner_result.first_action
            cost_to_go = planner_result.cost_to_go
            conditional_policy = planner_result.conditional_policy
        masked_selection_logits = selection_logits.masked_fill(
            ~valid_mask, -torch.finfo(selection_logits.dtype).max / 1e4
        )

        return {
            "hidden": hidden,
            "snr_mu": snr_mu,
            "snr_logvar": snr_logvar,
            "ttl_s": ttl_s,
            "selection_logits": masked_selection_logits,
            "hof_logits": hof_logits,
            "hof_probability": hof_probability,
            "state_cost": state_cost,
            "transition_cost": transition_cost,
            "q_next": first_action,
            "cost_to_go": cost_to_go,
            "conditional_policy": conditional_policy,
            "state_components": state_components,
            "transition_components": transition_components,
            "relative_angular_speed": relative_omega,
        }
