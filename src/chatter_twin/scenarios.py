from __future__ import annotations

from dataclasses import replace

from chatter_twin.models import CutConfig, ModalParams, SimulationConfig, ToolConfig


def make_scenario(name: str) -> tuple[ModalParams, ToolConfig, CutConfig, SimulationConfig]:
    modal = ModalParams()
    tool = ToolConfig()
    base_cut = CutConfig()
    config = SimulationConfig()

    match name:
        case "stable":
            cut = replace(base_cut, axial_depth_m=0.00025, spindle_rpm=9200.0)
            sim = replace(config, sensor_noise_std=0.15, random_seed=11)
        case "near_boundary":
            cut = replace(base_cut, axial_depth_m=0.00085, spindle_rpm=9000.0)
            sim = replace(config, sensor_noise_std=0.20, random_seed=13)
        case "unstable":
            cut = replace(base_cut, axial_depth_m=0.00180, spindle_rpm=8800.0)
            sim = replace(config, sensor_noise_std=0.25, drift_per_s=0.10, random_seed=17)
        case "onset":
            cut = replace(base_cut, axial_depth_m=0.00082, spindle_rpm=8950.0)
            sim = replace(
                config,
                duration_s=0.90,
                sensor_noise_std=0.20,
                axial_depth_start_scale=0.25,
                axial_depth_ramp_per_s=1.60,
                drift_per_s=0.20,
                random_seed=19,
            )
        case _:
            raise ValueError(f"Unknown scenario {name!r}")

    return modal, tool, cut, sim
