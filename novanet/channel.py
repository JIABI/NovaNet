"""Auditable Ku-band link budget, fading, measurements, and Doppler tracking."""

from __future__ import annotations

from dataclasses import dataclass

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
    received_power_dbm: float
    snr_db: float
    rate_bps: float


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
    def __init__(self, config: ChannelConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)

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

    def evaluate(
        self,
        geometry: GeometryState,
        *,
        bandwidth_hz: float | None = None,
        rain_rate_mm_h: float | None = None,
        stochastic: bool = False,
        doppler_estimation_error_hz: float | None = None,
        additional_loss_db: float = 0.0,
    ) -> LinkState:
        bandwidth = (
            self.config.bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        )
        fspl = free_space_path_loss_db(geometry.range_m, self.config.carrier_hz)
        gas, rain = atmospheric_losses_db(
            geometry.elevation_deg, self.config, rain_rate_mm_h
        )
        fading = self._fading_gain_db(stochastic)

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
        received = (
            total_eirp_dbm
            + self.config.ue_antenna_gain_dbi
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
        snr_linear = 10.0 ** (snr_db / 10.0)
        rate = (
            self.config.implementation_efficiency
            * bandwidth
            * np.log2(1.0 + snr_linear)
        )
        if snr_db < self.config.outage_threshold_db:
            rate = 0.0
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
            received_power_dbm=float(received),
            snr_db=float(snr_db),
            rate_bps=rate,
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
