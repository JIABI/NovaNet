"""Typed, validated configuration for all NovaNet pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "paper.yaml"
T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    tle_path: str
    tle_selection: str
    start_utc: str
    duration_s: int
    decision_interval_s: int
    geometry_subsample_s: int
    minimum_elevation_deg: float
    candidate_cap: int
    num_satellites: int
    ue_latitude_deg: tuple[float, float]
    ue_longitude_deg: tuple[float, float]
    ue_altitude_m: float


@dataclass(frozen=True)
class ChannelConfig:
    carrier_hz: float
    bandwidth_hz: float
    bandwidth_options_hz: tuple[float, ...]
    eirp_density_dbw_mhz: float
    ue_antenna_gain_dbi: float
    system_noise_temperature_k: float
    implementation_efficiency: float
    outage_threshold_db: float
    minimum_rate_bps: float
    gaseous_zenith_loss_db: float
    rain_rate_mm_h: float
    rain_specific_attenuation_db_km: float
    rician_k_db: float
    shadowing_std_db: float
    measurement_iir_alpha: float
    measurement_interval_s: int
    nominal_measurement_std_db: float
    doppler_tracking_efficiency: float
    doppler_estimation_std_hz: float
    coherent_integration_s: float

    @property
    def total_eirp_dbm(self) -> float:
        bandwidth_mhz = self.bandwidth_hz / 1e6
        return (
            self.eirp_density_dbw_mhz
            + 10.0 * __import__("math").log10(bandwidth_mhz)
            + 30.0
        )

    @property
    def noise_psd_dbm_hz(self) -> float:
        # kT in dBm/Hz, evaluated at the configured system temperature.
        import math

        boltzmann = 1.380649e-23
        return 10.0 * math.log10(
            boltzmann * self.system_noise_temperature_k * 1000.0
        )


@dataclass(frozen=True)
class HandoverConfig:
    ttt_s: float
    execution_s: float
    statistics_window_s: int
    hysteresis_db: float
    freeze_steps: int
    failure_outage_fraction: float


@dataclass(frozen=True)
class PlannerConfig:
    horizon_steps: int
    temperature: float
    lcb_kappa: float
    rate_weight: float
    dwell_weight: float
    base_switch_cost: float
    retained_dwell_weight: float
    angular_speed_weight: float
    hof_weight: float
    load_weight: float


@dataclass(frozen=True)
class ModelConfig:
    ue_feature_dim: int
    node_feature_dim: int
    transition_feature_dim: int
    hidden_dim: int
    gnn_layers: int
    graph_neighbors: int
    adjacency_temperature: float


@dataclass(frozen=True)
class TrainingConfig:
    num_samples: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    use_amp: bool
    snr_nll_weight: float
    ttl_weight: float
    hof_weight: float
    path_weight: float
    selection_weight: float
    entropy_weight: float
    handover_weight_init: float
    handover_weight_max: float
    dual_step: float
    target_switch_rate: float
    checkpoint_path: str


@dataclass(frozen=True)
class TrafficConfig:
    packet_size_bytes: int
    arrival_process: str
    arrival_rate_packets_s: float
    queue_discipline: str
    queue_capacity_packets: int
    fixed_network_delay_ms: float
    protocol_processing_ms: float


@dataclass(frozen=True)
class MultiUEConfig:
    satellite_capacity_mbps: float
    max_users_per_satellite: int
    minimum_admission_rate_mbps: float
    scheduler: str
    association_update: str
    region_center_lat_deg: float
    region_center_lon_deg: float
    region_radius_km: float


@dataclass(frozen=True)
class NovaNetConfig:
    schema_version: int
    experiment: ExperimentConfig
    channel: ChannelConfig
    handover: HandoverConfig
    planner: PlannerConfig
    model: ModelConfig
    training: TrainingConfig
    traffic: TrafficConfig
    multi_ue: MultiUEConfig

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def start_utc(self) -> datetime:
        return datetime.fromisoformat(self.experiment.start_utc)

    def resolve_tle_path(self) -> Path:
        path = Path(self.experiment.tle_path)
        return path if path.is_absolute() else REPO_ROOT / path

    def validate(self) -> None:
        exp, ch, ho, plan = (
            self.experiment,
            self.channel,
            self.handover,
            self.planner,
        )
        if self.schema_version != 2:
            raise ValueError("Unsupported config schema; expected schema_version=2")
        if not 0.0 <= exp.minimum_elevation_deg < 90.0:
            raise ValueError("minimum_elevation_deg must be in [0, 90)")
        if exp.candidate_cap < 2 or exp.candidate_cap > exp.num_satellites:
            raise ValueError("candidate_cap must be in [2, num_satellites]")
        if exp.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if exp.tle_selection != "shell_stratified_orbit_balanced_nested":
            raise ValueError(
                "The paper protocol requires "
                "shell_stratified_orbit_balanced_nested TLE selection"
            )
        if (
            exp.decision_interval_s <= 0
            or exp.geometry_subsample_s <= 0
            or exp.decision_interval_s % exp.geometry_subsample_s
        ):
            raise ValueError(
                "positive geometry_subsample_s must divide decision_interval_s"
            )
        if ho.ttt_s <= 0.0 or ho.execution_s <= 0.0:
            raise ValueError("CHO TTT and execution duration must be positive")
        if ho.freeze_steps < 0:
            raise ValueError("freeze_steps cannot be negative")
        if not ch.bandwidth_options_hz or any(
            bandwidth <= 0.0 for bandwidth in ch.bandwidth_options_hz
        ):
            raise ValueError("bandwidth_options_hz must contain positive values")
        if not 0.0 < ch.implementation_efficiency <= 1.0:
            raise ValueError("implementation_efficiency must be in (0, 1]")
        if not 0.0 <= ch.measurement_iir_alpha <= 1.0:
            raise ValueError("measurement_iir_alpha must be in [0, 1]")
        if plan.horizon_steps < 2:
            raise ValueError("Finite-horizon DP requires horizon_steps >= 2")
        if plan.temperature <= 0.0:
            raise ValueError("Soft-DP temperature must be positive")
        if self.traffic.arrival_process != "poisson":
            raise ValueError("The reproducible paper latency model uses Poisson arrivals")
        if self.traffic.queue_discipline != "fifo":
            raise ValueError("The reproducible paper latency model uses a FIFO queue")
        if self.traffic.arrival_rate_packets_s <= 0.0:
            raise ValueError("arrival_rate_packets_s must be positive")
        if self.traffic.packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")
        if self.traffic.queue_capacity_packets < 1:
            raise ValueError("queue_capacity_packets must be positive")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    names = {field.name for field in fields(cls)}
    unknown = set(values) - names
    missing = names - set(values)
    if unknown or missing:
        raise ValueError(
            f"{cls.__name__}: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    normalized = dict(values)
    for name, value in tuple(normalized.items()):
        if isinstance(value, list):
            normalized[name] = tuple(value)
    return cls(**normalized)


def load_config(path: str | Path | None = None) -> NovaNetConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    cfg = NovaNetConfig(
        schema_version=int(raw["schema_version"]),
        experiment=_construct(ExperimentConfig, raw["experiment"]),
        channel=_construct(ChannelConfig, raw["channel"]),
        handover=_construct(HandoverConfig, raw["handover"]),
        planner=_construct(PlannerConfig, raw["planner"]),
        model=_construct(ModelConfig, raw["model"]),
        training=_construct(TrainingConfig, raw["training"]),
        traffic=_construct(TrafficConfig, raw["traffic"]),
        multi_ue=_construct(MultiUEConfig, raw["multi_ue"]),
    )
    cfg.validate()
    return cfg
