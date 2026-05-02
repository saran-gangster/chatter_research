from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ChatterLabel = Literal["stable", "transition", "slight", "severe", "unknown"]
ProductivityMode = Literal["feed", "mrr"]


@dataclass(frozen=True)
class ModalParams:
    """Low-order x/y tool-tip dynamics."""

    mass_x_kg: float = 0.8
    mass_y_kg: float = 0.8
    stiffness_x_n_m: float = 1.55e7
    stiffness_y_n_m: float = 1.25e7
    damping_x_n_s_m: float = 210.0
    damping_y_n_s_m: float = 190.0

    def __post_init__(self) -> None:
        values = (
            self.mass_x_kg,
            self.mass_y_kg,
            self.stiffness_x_n_m,
            self.stiffness_y_n_m,
            self.damping_x_n_s_m,
            self.damping_y_n_s_m,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Modal parameters must be positive")

    @property
    def mass(self) -> NDArray[np.float64]:
        return np.array([self.mass_x_kg, self.mass_y_kg], dtype=float)

    @property
    def stiffness(self) -> NDArray[np.float64]:
        return np.array([self.stiffness_x_n_m, self.stiffness_y_n_m], dtype=float)

    @property
    def damping(self) -> NDArray[np.float64]:
        return np.array([self.damping_x_n_s_m, self.damping_y_n_s_m], dtype=float)

    @property
    def natural_frequency_hz(self) -> float:
        omega = np.sqrt(self.stiffness / self.mass)
        return float(np.mean(omega / (2.0 * np.pi)))

    @property
    def damping_ratio(self) -> float:
        ratio = self.damping / (2.0 * np.sqrt(self.stiffness * self.mass))
        return float(np.mean(ratio))


@dataclass(frozen=True)
class ToolConfig:
    diameter_m: float = 0.010
    flute_count: int = 4
    overhang_m: float = 0.040
    runout_m: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.diameter_m <= 0:
            raise ValueError("Tool diameter must be positive")
        if self.flute_count < 1:
            raise ValueError("Tool must have at least one flute")
        if self.overhang_m <= 0:
            raise ValueError("Tool overhang must be positive")
        if self.runout_m < 0:
            raise ValueError("Tool runout cannot be negative")


@dataclass(frozen=True)
class CutConfig:
    spindle_rpm: float = 9000.0
    feed_per_tooth_m: float = 45.0e-6
    axial_depth_m: float = 0.0006
    radial_depth_m: float = 0.004
    cutting_coeff_t_n_m2: float = 7.0e8
    cutting_coeff_r_n_m2: float = 2.1e8
    immersion_start_rad: float = 0.0
    immersion_end_rad: float = np.pi

    def __post_init__(self) -> None:
        if self.spindle_rpm <= 0:
            raise ValueError("Spindle speed must be positive")
        if self.feed_per_tooth_m <= 0:
            raise ValueError("Feed per tooth must be positive")
        if self.axial_depth_m <= 0 or self.radial_depth_m <= 0:
            raise ValueError("Depths of cut must be positive")
        if self.cutting_coeff_t_n_m2 <= 0 or self.cutting_coeff_r_n_m2 <= 0:
            raise ValueError("Cutting coefficients must be positive")
        if self.immersion_end_rad <= self.immersion_start_rad:
            raise ValueError("Immersion end must exceed immersion start")


@dataclass(frozen=True)
class SimulationConfig:
    duration_s: float = 0.25
    sample_rate_hz: float = 5000.0
    sensor_noise_std: float = 0.25
    drift_per_s: float = 0.0
    axial_depth_start_scale: float = 1.0
    axial_depth_ramp_per_s: float = 0.0
    cutting_coeff_start_scale: float = 1.0
    random_seed: int | None = 7
    initial_displacement_m: tuple[float, float] = (1.0e-7, -1.0e-7)
    initial_velocity_m_s: tuple[float, float] = (0.0, 0.0)
    displacement_limit_m: float = 4.0e-3

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("Duration must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")
        if self.sensor_noise_std < 0:
            raise ValueError("Sensor noise cannot be negative")
        if self.axial_depth_start_scale <= 0:
            raise ValueError("Initial axial-depth scale must be positive")
        if self.cutting_coeff_start_scale <= 0:
            raise ValueError("Initial cutting-coefficient scale must be positive")
        if self.axial_depth_start_scale + self.axial_depth_ramp_per_s * self.duration_s <= 0:
            raise ValueError("Axial-depth ramp must stay positive for the simulation duration")
        if self.cutting_coeff_start_scale + self.drift_per_s * self.duration_s <= 0:
            raise ValueError("Cutting-coefficient drift must stay positive for the simulation duration")
        if self.displacement_limit_m <= 0:
            raise ValueError("Displacement limit must be positive")


@dataclass(frozen=True)
class SimulationResult:
    time_s: NDArray[np.float64]
    displacement_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    acceleration_m_s2: NDArray[np.float64]
    cutting_force_n: NDArray[np.float64]
    sensor_signal: NDArray[np.float64]
    modal: ModalParams
    tool: ToolConfig
    cut: CutConfig
    axial_depth_m: NDArray[np.float64] | None = None
    cutting_coeff_t_n_m2: NDArray[np.float64] | None = None
    cutting_coeff_r_n_m2: NDArray[np.float64] | None = None
    spindle_rpm: NDArray[np.float64] | None = None
    feed_per_tooth_m: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class SignalFeatures:
    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    tooth_frequency_hz: float
    tooth_band_energy: float
    chatter_band_energy: float
    non_tooth_harmonic_ratio: float
    dominant_frequency_hz: float
    spectral_entropy: float
    sample_rate_hz: float


@dataclass(frozen=True)
class RiskEstimate:
    risk_chatter_now: float
    risk_chatter_horizon: float
    margin_physics: float
    margin_signal: float
    label: ChatterLabel
    uncertainty: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ActionProposal:
    feed_override: float
    spindle_override: float


@dataclass(frozen=True)
class MachineState:
    controller_mode: str = "auto"
    in_cut: bool = True
    sensor_healthy: bool = True
    last_feed_override: float = 1.0
    last_spindle_override: float = 1.0


@dataclass(frozen=True)
class ShieldConfig:
    min_feed_override: float = 0.80
    max_feed_override: float = 1.05
    min_spindle_override: float = 0.90
    max_spindle_override: float = 1.10
    max_feed_delta: float = 0.05
    max_spindle_delta: float = 0.05
    max_uncertainty: float = 0.65
    uncertainty_mode: str = "reject"
    allowed_controller_modes: tuple[str, ...] = ("auto", "mdi")
    allow_unknown_state: bool = False

    def __post_init__(self) -> None:
        if self.uncertainty_mode not in {"reject", "hold", "advisory"}:
            raise ValueError("uncertainty_mode must be 'reject', 'hold', or 'advisory'")


@dataclass(frozen=True)
class ShieldResult:
    feed_override: float
    spindle_override: float
    accepted: bool
    rejected: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RewardConfig:
    productivity_mode: ProductivityMode = "feed"
    productivity_weight: float = 1.0
    risk_now_weight: float = 3.0
    risk_horizon_weight: float = 1.5
    severe_penalty: float = 2.0
    smoothness_weight: float = 0.5
    rejection_penalty: float = 0.75
    clip_penalty: float = 0.0
    rate_limit_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.productivity_mode not in {"feed", "mrr"}:
            raise ValueError("productivity_mode must be 'feed' or 'mrr'")
        if self.productivity_weight < 0:
            raise ValueError("productivity_weight cannot be negative")
        if self.risk_now_weight < 0 or self.risk_horizon_weight < 0:
            raise ValueError("risk weights cannot be negative")
        if self.severe_penalty < 0:
            raise ValueError("severe_penalty cannot be negative")
        if self.smoothness_weight < 0:
            raise ValueError("smoothness_weight cannot be negative")
        if self.rejection_penalty < 0:
            raise ValueError("rejection_penalty cannot be negative")
        if self.clip_penalty < 0:
            raise ValueError("clip_penalty cannot be negative")
        if self.rate_limit_penalty < 0:
            raise ValueError("rate_limit_penalty cannot be negative")
