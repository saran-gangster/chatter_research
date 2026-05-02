from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chatter_twin.models import CutConfig, ModalParams, ToolConfig


@dataclass(frozen=True)
class StabilityEstimate:
    """Compact frequency-domain stability estimate for a milling cut."""

    critical_axial_depth_m: float
    signed_margin: float
    limiting_frequency_hz: float
    tooth_frequency_hz: float
    regenerative_factor: float
    dynamic_compliance_m_n: float
    directional_factor: float


def estimate_stability(modal: ModalParams, tool: ToolConfig, cut: CutConfig) -> StabilityEstimate:
    """Estimate chatter margin from modal FRFs and regenerative phase.

    This is still an MVP model, but it is intentionally closer to milling
    stability logic than the initial scalar surrogate: it scans plausible
    chatter frequencies around the modal frequencies, combines the tool-tip
    receptance with a regenerative delay factor, and returns the limiting
    axial depth for the current spindle speed.
    """

    tooth_frequency = cut.spindle_rpm * tool.flute_count / 60.0
    frequencies = _candidate_chatter_frequencies(modal, tooth_frequency)
    compliance = directional_dynamic_compliance(modal, cut, frequencies)
    regen = regenerative_phase_factor(frequencies, tooth_frequency)
    instability_index = compliance * regen
    limiting_idx = int(np.argmax(instability_index))

    direction = immersion_directional_factor(cut, tool)
    max_index = float(instability_index[limiting_idx])
    # Empirical calibration constant for the MVP simulator scale. The
    # surrounding terms keep the lobe shape tied to FRF and regenerative phase.
    calibration = 0.50
    critical = calibration / max(cut.cutting_coeff_t_n_m2 * direction * max_index, 1.0e-18)
    critical = max(1.0e-5, critical)
    margin = (critical - cut.axial_depth_m) / critical

    return StabilityEstimate(
        critical_axial_depth_m=float(critical),
        signed_margin=float(margin),
        limiting_frequency_hz=float(frequencies[limiting_idx]),
        tooth_frequency_hz=float(tooth_frequency),
        regenerative_factor=float(regen[limiting_idx]),
        dynamic_compliance_m_n=float(compliance[limiting_idx]),
        directional_factor=float(direction),
    )


def critical_axial_depth_m(modal: ModalParams, tool: ToolConfig, cut: CutConfig) -> float:
    """Return critical axial depth from the FRF/regenerative stability estimate."""

    return estimate_stability(modal, tool, cut).critical_axial_depth_m


def signed_stability_margin(modal: ModalParams, tool: ToolConfig, cut: CutConfig) -> float:
    """Positive means stable by the estimate, negative means predicted unstable."""

    return estimate_stability(modal, tool, cut).signed_margin


def modal_receptance(modal: ModalParams, frequency_hz: float | NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return x/y displacement receptance for the low-order modal model."""

    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    omega = 2.0 * np.pi * frequency
    mass = modal.mass[:, np.newaxis]
    damping = modal.damping[:, np.newaxis]
    stiffness = modal.stiffness[:, np.newaxis]
    denominator = stiffness - mass * omega[np.newaxis, :] ** 2 + 1j * damping * omega[np.newaxis, :]
    response = 1.0 / denominator
    if np.ndim(frequency_hz) == 0:
        return response[:, 0]
    return response


def directional_dynamic_compliance(
    modal: ModalParams,
    cut: CutConfig,
    frequency_hz: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an immersion-weighted scalar dynamic compliance."""

    response = modal_receptance(modal, frequency_hz)
    if response.ndim == 1:
        response = response[:, np.newaxis]
    radial_ratio = cut.cutting_coeff_r_n_m2 / cut.cutting_coeff_t_n_m2
    direction = _average_force_direction(cut, radial_ratio)
    projected = np.abs(direction[:, np.newaxis] * response)
    compliance = np.sum(projected, axis=0)
    if np.ndim(frequency_hz) == 0:
        return np.array(float(compliance[0]))
    return compliance


def regenerative_phase_factor(
    frequency_hz: float | NDArray[np.float64],
    tooth_frequency_hz: float,
) -> NDArray[np.float64]:
    """Return a bounded regenerative delay factor for chatter frequencies."""

    frequency = np.asarray(frequency_hz, dtype=float)
    phase = 2.0 * np.pi * frequency / max(tooth_frequency_hz, 1.0e-12)
    # The delay contribution is strongest away from integer tooth-period phase.
    factor = 0.20 + 2.0 * np.abs(np.sin(0.5 * phase))
    return np.clip(factor, 0.20, 2.20)


def immersion_directional_factor(cut: CutConfig, tool: ToolConfig) -> float:
    """Return a scalar cutting-direction multiplier from immersion geometry."""

    radial_ratio = min(max(cut.radial_depth_m / tool.diameter_m, 0.02), 1.0)
    arc = _immersion_arc(cut.immersion_start_rad, cut.immersion_end_rad)
    arc_ratio = min(max(arc / (2.0 * np.pi), 0.02), 1.0)
    force_ratio = 1.0 + 0.25 * cut.cutting_coeff_r_n_m2 / cut.cutting_coeff_t_n_m2
    return float((0.50 + radial_ratio) * (0.65 + arc_ratio) * force_ratio)


def _candidate_chatter_frequencies(modal: ModalParams, tooth_frequency_hz: float) -> NDArray[np.float64]:
    wx = np.sqrt(modal.stiffness_x_n_m / modal.mass_x_kg) / (2.0 * np.pi)
    wy = np.sqrt(modal.stiffness_y_n_m / modal.mass_y_kg) / (2.0 * np.pi)
    center = 0.5 * (wx + wy)
    lower = max(5.0, 0.65 * min(wx, wy), 0.60 * tooth_frequency_hz)
    upper = max(lower + 1.0, 1.35 * max(wx, wy), 1.40 * tooth_frequency_hz)
    return np.linspace(lower, upper, 240)


def _average_force_direction(cut: CutConfig, radial_ratio: float) -> NDArray[np.float64]:
    phis = _engaged_angles(cut.immersion_start_rad, cut.immersion_end_rad)
    normal_x = np.sin(phis)
    normal_y = np.cos(phis)
    tangent_x = -np.cos(phis)
    tangent_y = np.sin(phis)
    force_x = tangent_x + radial_ratio * normal_x
    force_y = tangent_y + radial_ratio * normal_y
    direction = np.array([np.mean(np.abs(force_x)), np.mean(np.abs(force_y))], dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 1.0e-12:
        return np.array([0.5, 0.5], dtype=float)
    return direction / norm


def _engaged_angles(start: float, end: float, count: int = 96) -> NDArray[np.float64]:
    arc = _immersion_arc(start, end)
    return (start + np.linspace(0.0, arc, count)) % (2.0 * np.pi)


def _immersion_arc(start: float, end: float) -> float:
    start = start % (2.0 * np.pi)
    end = end % (2.0 * np.pi)
    if end > start:
        return end - start
    return 2.0 * np.pi - start + end
