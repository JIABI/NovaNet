"""Auditable Ku-band link budget, fading, measurements, and Doppler tracking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np

from .config import ChannelConfig
from .geometry import GeometryState


LIGHT_SPEED_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True)
class LinkState:
    fspl_db: float
    gas_loss_db: float
    rain_loss_db: float
    fading_gain_db: float
    doppler_hz: float
    residual_doppler_hz: float
    tracking_loss_db: float
    additional_loss_db: float
    receive_gain_db: float
    received_power_dbm: float
    noise_power_dbm: float
    interference_power_dbm: float
    snr_db: float
    sinr_db: float
    rate_bps: float


def receive_array_gain_linear(
    los_ecef: np.ndarray,
    orientation_local_to_ecef: np.ndarray,
    *,
    element_gain_linear: float = 1.0,
    array_response: np.ndarray | None = None,
    combiner: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate the manuscript's orientation/array-gain interface.

    ``orientation_local_to_ecef`` maps a UE-local vector into ECEF, so the
    direction seen by the terminal is ``Q.T @ los_ecef``.  The default is the
    paper's main-experiment single isotropic element.  For an M-element array,
    callers supply its complex response and a unit-norm combiner.
    """

    los = np.asarray(los_ecef, dtype=float)
    orientation = np.asarray(orientation_local_to_ecef, dtype=float)
    if los.shape != (3,) or orientation.shape != (3, 3):
        raise ValueError("los_ecef must be [3] and orientation must be [3,3]")
    norm = float(np.linalg.norm(los))
    if norm <= 0.0:
        raise ValueError("los_ecef must have non-zero length")
    local_los = orientation.T @ (los / norm)
    if element_gain_linear < 0.0:
        raise ValueError("element_gain_linear cannot be negative")
    response = (
        np.ones(1, dtype=np.complex128)
        if array_response is None
        else np.asarray(array_response, dtype=np.complex128).reshape(-1)
    )
    weights = (
        np.ones(response.size, dtype=np.complex128) / np.sqrt(response.size)
        if combiner is None
        else np.asarray(combiner, dtype=np.complex128).reshape(-1)
    )
    if response.size != weights.size:
        raise ValueError("array_response and combiner must have the same length")
    weight_norm = float(np.linalg.norm(weights))
    if weight_norm <= 0.0:
        raise ValueError("combiner must have non-zero norm")
    weights = weights / weight_norm
    gain = float(element_gain_linear * abs(np.vdot(weights, response)) ** 2)
    return local_los, gain


def free_space_path_loss_db(range_m: float, carrier_hz: float) -> float:
    return float(
        20.0 * np.log10(4.0 * np.pi * range_m * carrier_hz / LIGHT_SPEED_M_S)
    )


def noise_power_dbm(bandwidth_hz: float, temperature_k: float) -> float:
    watts = BOLTZMANN_J_K * temperature_k * bandwidth_hz
    return float(10.0 * np.log10(watts * 1000.0))


def atmospheric_losses_db(
    elevation_deg: float,
    config: ChannelConfig,
    rain_rate_mm_h: float | None = None,
) -> tuple[float, float]:
    sine = max(
        float(np.sin(np.deg2rad(max(elevation_deg, 1.0)))),
        1e-6,
    )
    gas = config.gaseous_zenith_loss_db / sine
    rain_rate = config.rain_rate_mm_h if rain_rate_mm_h is None else rain_rate_mm_h
    rain_height_km = 5.0
    rain_path_km = rain_height_km / sine
    rain = (
        config.rain_specific_attenuation_db_km
        * rain_path_km
        * (max(rain_rate, 0.0) / 8.0) ** 0.9
    )
    return float(gas), float(rain)


class LinkBudget:
    def __init__(
        self,
        config: ChannelConfig,
        seed: int = 0,
        receive_gain_model: Callable[[GeometryState], float] | None = None,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.receive_gain_model = receive_gain_model

    def _fading_gain_db(self, stochastic: bool) -> float:
        if not stochastic:
            return 0.0
        k_linear = 10.0 ** (self.config.rician_k_db / 10.0)
        los = np.sqrt(k_linear / (k_linear + 1.0))
        scatter_scale = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        coefficient = los + scatter_scale * (
            self.rng.normal() + 1j * self.rng.normal()
        )
        rician_gain = 10.0 * np.log10(max(abs(coefficient) ** 2, 1e-12))
        shadowing = self.rng.normal(0.0, self.config.shadowing_std_db)
        return float(rician_gain + shadowing)

    def fading_gain_from_rng(self, rng: np.random.Generator) -> float:
        """Draw the configured Rician-plus-shadowing gain from ``rng``."""

        k_linear = 10.0 ** (self.config.rician_k_db / 10.0)
        los = np.sqrt(k_linear / (k_linear + 1.0))
        scatter_scale = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        coefficient = los + scatter_scale * (
            rng.normal() + 1j * rng.normal()
        )
        rician_gain = 10.0 * np.log10(max(abs(coefficient) ** 2, 1e-12))
        shadowing = rng.normal(0.0, self.config.shadowing_std_db)
        return float(rician_gain + shadowing)

    def evaluate(
        self,
        geometry: GeometryState,
        *,
        bandwidth_hz: float | None = None,
        rain_rate_mm_h: float | None = None,
        stochastic: bool = False,
        doppler_estimation_error_hz: float | None = None,
        additional_loss_db: float = 0.0,
        receive_gain_linear: float | None = None,
        interference_power_w: float | None = None,
        fading_gain_db: float | None = None,
    ) -> LinkState:
        bandwidth = (
            self.config.bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        )
        fspl = free_space_path_loss_db(geometry.range_m, self.config.carrier_hz)
        gas, rain = atmospheric_losses_db(
            geometry.elevation_deg, self.config, rain_rate_mm_h
        )
        fading = (
            float(fading_gain_db)
            if fading_gain_db is not None
            else self._fading_gain_db(stochastic)
        )

        doppler = (
            -geometry.radial_velocity_m_s
            / LIGHT_SPEED_M_S
            * self.config.carrier_hz
        )
        if doppler_estimation_error_hz is None:
            error = (
                self.rng.normal(0.0, self.config.doppler_estimation_std_hz)
                if stochastic
                else 0.0
            )
        else:
            error = float(doppler_estimation_error_hz)
        residual_doppler = (
            (1.0 - self.config.doppler_tracking_efficiency) * doppler + error
        )
        coherent_gain = np.sinc(
            residual_doppler * self.config.coherent_integration_s
        ) ** 2
        tracking_loss = float(
            -10.0 * np.log10(max(float(coherent_gain), 1e-3))
        )

        bandwidth_mhz = bandwidth / 1e6
        total_eirp_dbm = (
            self.config.eirp_density_dbw_mhz
            + 10.0 * np.log10(bandwidth_mhz)
            + 30.0
        )
        gain_linear = (
            float(self.receive_gain_model(geometry))
            if receive_gain_linear is None and self.receive_gain_model is not None
            else (1.0 if receive_gain_linear is None else float(receive_gain_linear))
        )
        if not np.isfinite(gain_linear) or gain_linear < 0.0:
            raise ValueError("receive gain must be finite and nonnegative")
        interference_w = (
            self.config.exogenous_interference_power_w
            if interference_power_w is None
            else float(interference_power_w)
        )
        if interference_w < 0.0:
            raise ValueError("interference_power_w cannot be negative")
        receive_gain_db = 10.0 * np.log10(max(gain_linear, 1e-12))
        received = (
            total_eirp_dbm
            + self.config.ue_antenna_gain_dbi
            + receive_gain_db
            - fspl
            - gas
            - rain
            + fading
            - tracking_loss
            - float(additional_loss_db)
        )
        noise = noise_power_dbm(
            bandwidth, self.config.system_noise_temperature_k
        )
        snr_db = received - noise
        noise_w = 10.0 ** ((noise - 30.0) / 10.0)
        denominator_w = noise_w + interference_w
        received_w = 10.0 ** ((received - 30.0) / 10.0)
        sinr_linear = received_w / max(denominator_w, 1e-30)
        sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-30))
        interference_dbm = (
            10.0 * np.log10(interference_w * 1000.0)
            if interference_w > 0.0
            else -np.inf
        )
        rate = (
            self.config.implementation_efficiency
            * bandwidth
            * np.log2(1.0 + sinr_linear)
        )
        # The physical rate in Eq. (10) is defined at every finite SINR.
        # Link validity is applied separately by the effective-throughput and
        # packet-service logic, so the threshold must not alter this value.
        rate = max(float(rate), 0.0)
        return LinkState(
            fspl_db=fspl,
            gas_loss_db=gas,
            rain_loss_db=rain,
            fading_gain_db=fading,
            doppler_hz=float(doppler),
            residual_doppler_hz=float(residual_doppler),
            tracking_loss_db=tracking_loss,
            additional_loss_db=float(additional_loss_db),
            receive_gain_db=float(receive_gain_db),
            received_power_dbm=float(received),
            noise_power_dbm=float(noise),
            interference_power_dbm=float(interference_dbm),
            snr_db=float(snr_db),
            sinr_db=float(sinr_db),
            rate_bps=rate,
        )


class RealizedChannelTrace:
    """Order-independent seeded channel samples shared by all event consumers.

    A keyed sample makes repeated queries for the same UE/satellite/time point
    identical, irrespective of whether the query comes from a measurement,
    CHO replay, service simulation, or a counterfactual training label.  This
    removes call-order drift between those code paths.
    """

    def __init__(
        self,
        config: ChannelConfig,
        *,
        seed: int,
        ue_key: int = 0,
        event_step_s: float = 0.01,
        receive_gain_model: Callable[[GeometryState], float] | None = None,
    ):
        if event_step_s <= 0.0:
            raise ValueError("event_step_s must be positive")
        self.config = config
        self.seed = int(seed)
        self.ue_key = int(ue_key)
        self.event_step_s = float(event_step_s)
        self.budget = LinkBudget(
            config,
            seed=seed,
            receive_gain_model=receive_gain_model,
        )

    def _rng(self, satellite_id: int, time_bin: int, stream: str):
        payload = (
            f"{self.seed}|{self.ue_key}|{int(satellite_id)}|"
            f"{int(time_bin)}|{stream}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def fading_gain_db(self, satellite_id: int, time_s: float) -> float:
        fast_bin = int(np.floor(time_s / self.event_step_s + 1e-9))
        slow_bin = int(
            np.floor(time_s / max(self.config.measurement_interval_s, 1.0))
        )
        fast = self._rng(satellite_id, fast_bin, "rician")
        k_linear = 10.0 ** (self.config.rician_k_db / 10.0)
        los = np.sqrt(k_linear / (k_linear + 1.0))
        scatter_scale = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        coefficient = los + scatter_scale * (
            fast.normal() + 1j * fast.normal()
        )
        rician = 10.0 * np.log10(max(abs(coefficient) ** 2, 1e-12))
        shadow = self._rng(satellite_id, slow_bin, "shadow").normal(
            0.0, self.config.shadowing_std_db
        )
        return float(rician + shadow)

    def measurement_noise_db(
        self, satellite_id: int, time_s: float, standard_deviation_db: float
    ) -> float:
        if not np.isfinite(standard_deviation_db) or standard_deviation_db < 0.0:
            raise ValueError(
                "standard_deviation_db must be finite and nonnegative"
            )
        if not np.isfinite(time_s):
            raise ValueError("measurement time must be finite")
        time_bin = int(
            np.floor(time_s / max(self.config.measurement_interval_s, 1.0))
        )
        return float(
            self._rng(satellite_id, time_bin, "measurement").normal(
                0.0, standard_deviation_db
            )
        )

    def doppler_estimation_error_hz(
        self, satellite_id: int, time_s: float
    ) -> float:
        time_bin = int(np.floor(time_s / self.event_step_s + 1e-9))
        return float(
            self._rng(satellite_id, time_bin, "doppler_error").normal(
                0.0, self.config.doppler_estimation_std_hz
            )
        )

    def blockage_active(
        self, satellite_id: int, time_s: float, probability: float
    ) -> bool:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("blockage probability must be in [0,1]")
        time_bin = int(
            np.floor(time_s / max(self.config.measurement_interval_s, 1.0))
        )
        return bool(
            self._rng(satellite_id, time_bin, "blockage").random() < probability
        )

    def evaluate(
        self,
        geometry: GeometryState,
        satellite_id: int,
        time_s: float,
        **kwargs,
    ) -> LinkState:
        kwargs.setdefault(
            "doppler_estimation_error_hz",
            self.doppler_estimation_error_hz(satellite_id, time_s),
        )
        return self.budget.evaluate(
            geometry,
            stochastic=False,
            fading_gain_db=self.fading_gain_db(satellite_id, time_s),
            **kwargs,
        )


@dataclass
class MeasurementRecord:
    filtered_snr_db: float
    timestamp_s: float


class MeasurementTracker:
    """IIR-filtered reports plus causal forecast with horizon-growing variance."""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.records: dict[int, MeasurementRecord] = {}

    def update(self, satellite_id: int, snr_db: float, timestamp_s: float) -> None:
        previous = self.records.get(satellite_id)
        # A delayed report may be presented to several decision epochs.  It
        # is one physical observation, so do not feed the same (or an older)
        # timestamp through the IIR filter more than once.
        if previous is not None and timestamp_s <= previous.timestamp_s + 1e-12:
            return
        alpha = self.config.measurement_iir_alpha
        filtered = (
            float(snr_db)
            if previous is None
            else alpha * float(snr_db)
            + (1.0 - alpha) * previous.filtered_snr_db
        )
        self.records[satellite_id] = MeasurementRecord(filtered, timestamp_s)

    def current_fields(
        self,
        satellite_id: int,
        timestamp_s: float,
        deterministic_snr_db: float,
    ) -> tuple[float, float, float]:
        record = self.records.get(satellite_id)
        if record is None:
            return float(deterministic_snr_db), 0.0, 0.0
        age_s = max(0.0, timestamp_s - record.timestamp_s)
        return record.filtered_snr_db, age_s, 1.0

    def forecast(
        self,
        satellite_id: int,
        current_time_s: float,
        future_time_s: float,
        deterministic_now_db: float,
        deterministic_future_db: float,
    ) -> tuple[float, float, float, float]:
        report, age_s, available = self.current_fields(
            satellite_id, current_time_s, deterministic_now_db
        )
        horizon_s = max(0.0, future_time_s - current_time_s)
        decay_s = 2.0 * self.config.measurement_interval_s
        residual = report - deterministic_now_db
        predicted = deterministic_future_db + residual * np.exp(-horizon_s / decay_s)
        effective_age = age_s + horizon_s
        variance = self.config.nominal_measurement_std_db**2 * (
            1.0
            + effective_age / max(self.config.measurement_interval_s, 1.0)
            + (1.0 - available)
        )
        return float(predicted), float(np.sqrt(variance)), float(age_s), available
