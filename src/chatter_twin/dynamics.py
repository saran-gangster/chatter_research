from __future__ import annotations

from dataclasses import replace

import numpy as np

from chatter_twin.models import CutConfig, ModalParams, SimulationConfig, SimulationResult, ToolConfig


def simulate_milling(
    modal: ModalParams,
    tool: ToolConfig,
    cut: CutConfig,
    config: SimulationConfig | None = None,
    feed_override_profile: np.ndarray | None = None,
    spindle_override_profile: np.ndarray | None = None,
) -> SimulationResult:
    """Run a small fixed-step 2-DOF regenerative milling simulation."""

    config = config or SimulationConfig()
    rng = np.random.default_rng(config.random_seed)
    dt = 1.0 / config.sample_rate_hz
    n_steps = max(2, int(round(config.duration_s * config.sample_rate_hz)))
    time_s = np.arange(n_steps, dtype=float) * dt

    displacement = np.zeros((n_steps, 2), dtype=float)
    velocity = np.zeros((n_steps, 2), dtype=float)
    acceleration = np.zeros((n_steps, 2), dtype=float)
    force = np.zeros((n_steps, 2), dtype=float)

    displacement[0] = np.asarray(config.initial_displacement_m, dtype=float)
    velocity[0] = np.asarray(config.initial_velocity_m_s, dtype=float)

    tooth_period_s = 60.0 / (tool.flute_count * cut.spindle_rpm)
    delay_steps = max(1, int(round(tooth_period_s / dt)))
    omega = 2.0 * np.pi * cut.spindle_rpm / 60.0
    flute_offsets = 2.0 * np.pi * np.arange(tool.flute_count) / tool.flute_count
    runout_offsets = tool.runout_m * np.cos(flute_offsets)

    axial_depth_scale = np.maximum(
        config.axial_depth_start_scale + config.axial_depth_ramp_per_s * time_s,
        1.0e-9,
    )
    cutting_coeff_scale = np.maximum(
        config.cutting_coeff_start_scale + config.drift_per_s * time_s,
        1.0e-9,
    )
    axial_depth_profile = cut.axial_depth_m * axial_depth_scale
    cutting_coeff_t_profile = cut.cutting_coeff_t_n_m2 * cutting_coeff_scale
    cutting_coeff_r_profile = cut.cutting_coeff_r_n_m2 * cutting_coeff_scale
    feed_override = _profile_or_ones(feed_override_profile, n_steps, "feed_override_profile")
    spindle_override = _profile_or_ones(spindle_override_profile, n_steps, "spindle_override_profile")
    feed_profile = cut.feed_per_tooth_m * feed_override
    spindle_profile = cut.spindle_rpm * spindle_override

    mass = modal.mass
    damping = modal.damping
    stiffness = modal.stiffness
    spindle_angle_rad = np.zeros(n_steps, dtype=float)

    for idx in range(1, n_steps):
        active_spindle_rpm = float(spindle_profile[idx])
        spindle_angle_rad[idx] = spindle_angle_rad[idx - 1] + 2.0 * np.pi * active_spindle_rpm / 60.0 * dt
        tooth_period_s = 60.0 / (tool.flute_count * active_spindle_rpm)
        delay_steps = max(1, int(round(tooth_period_s / dt)))
        active_cut = replace(
            cut,
            spindle_rpm=active_spindle_rpm,
            feed_per_tooth_m=float(feed_profile[idx]),
            axial_depth_m=float(axial_depth_profile[idx]),
            cutting_coeff_t_n_m2=float(cutting_coeff_t_profile[idx]),
            cutting_coeff_r_n_m2=float(cutting_coeff_r_profile[idx]),
        )

        delayed_idx = max(0, idx - delay_steps)
        x_now = displacement[idx - 1]
        x_delay = displacement[delayed_idx]
        f_cut = _cutting_force(spindle_angle_rad[idx], x_now, x_delay, active_cut, tool, runout_offsets)
        f_cut = np.clip(f_cut, -5_000.0, 5_000.0)
        force[idx] = f_cut

        acceleration[idx] = (f_cut - damping * velocity[idx - 1] - stiffness * x_now) / mass
        velocity[idx] = velocity[idx - 1] + acceleration[idx] * dt
        displacement[idx] = x_now + velocity[idx] * dt

        if np.any(np.abs(displacement[idx]) > config.displacement_limit_m):
            displacement[idx] = np.clip(
                displacement[idx],
                -config.displacement_limit_m,
                config.displacement_limit_m,
            )
            velocity[idx] *= 0.2

    sensor_signal = acceleration.copy()
    if config.sensor_noise_std:
        sensor_signal += rng.normal(0.0, config.sensor_noise_std, size=sensor_signal.shape)

    return SimulationResult(
        time_s=time_s,
        displacement_m=displacement,
        velocity_m_s=velocity,
        acceleration_m_s2=acceleration,
        cutting_force_n=force,
        sensor_signal=sensor_signal,
        modal=modal,
        tool=tool,
        cut=cut,
        axial_depth_m=axial_depth_profile,
        cutting_coeff_t_n_m2=cutting_coeff_t_profile,
        cutting_coeff_r_n_m2=cutting_coeff_r_profile,
        spindle_rpm=spindle_profile,
        feed_per_tooth_m=feed_profile,
    )


def _cutting_force(
    spindle_angle_rad: float,
    x_now: np.ndarray,
    x_delay: np.ndarray,
    cut: CutConfig,
    tool: ToolConfig,
    runout_offsets: np.ndarray,
) -> np.ndarray:
    total = np.zeros(2, dtype=float)
    regen = x_now - x_delay

    for flute_idx in range(tool.flute_count):
        phi = (spindle_angle_rad + 2.0 * np.pi * flute_idx / tool.flute_count) % (2.0 * np.pi)
        if not _is_engaged(phi, cut.immersion_start_rad, cut.immersion_end_rad):
            continue

        normal = np.array([np.sin(phi), np.cos(phi)], dtype=float)
        tangent = np.array([-np.cos(phi), np.sin(phi)], dtype=float)
        chip_m = cut.feed_per_tooth_m * max(np.sin(phi), 0.0)
        chip_m += float(normal @ regen) + float(runout_offsets[flute_idx])
        if chip_m <= 0:
            continue

        tangential = cut.cutting_coeff_t_n_m2 * cut.axial_depth_m * chip_m
        radial = cut.cutting_coeff_r_n_m2 * cut.axial_depth_m * chip_m
        total += tangential * tangent + radial * normal

    return total


def _profile_or_ones(profile: np.ndarray | None, n_steps: int, name: str) -> np.ndarray:
    if profile is None:
        return np.ones(n_steps, dtype=float)
    values = np.asarray(profile, dtype=float)
    if values.shape != (n_steps,):
        raise ValueError(f"{name} must have shape ({n_steps},)")
    if np.any(values <= 0):
        raise ValueError(f"{name} values must be positive")
    return values


def _is_engaged(phi: float, start: float, end: float) -> bool:
    start = start % (2.0 * np.pi)
    end = end % (2.0 * np.pi)
    if start < end:
        return start <= phi <= end
    return phi >= start or phi <= end
