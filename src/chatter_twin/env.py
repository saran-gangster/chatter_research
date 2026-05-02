from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chatter_twin.calibration import MarginCalibration, apply_margin_calibration, calibrated_margin_uncertainty
from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.models import (
    ActionProposal,
    CutConfig,
    MachineState,
    ModalParams,
    RewardConfig,
    RiskEstimate,
    ShieldConfig,
    ShieldResult,
    SimulationConfig,
    ToolConfig,
)
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.shield import apply_safety_shield
from chatter_twin.stability import signed_stability_margin


class ChatterSuppressEnv(gym.Env):
    """Gymnasium environment for shielded feed/spindle override experiments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        modal: ModalParams | None = None,
        tool: ToolConfig | None = None,
        cut: CutConfig | None = None,
        sim_config: SimulationConfig | None = None,
        shield_config: ShieldConfig | None = None,
        reward_config: RewardConfig | None = None,
        margin_calibration: MarginCalibration | None = None,
        max_steps: int = 40,
        decision_interval_s: float = 0.25,
    ) -> None:
        super().__init__()
        self.modal = modal or ModalParams()
        self.tool = tool or ToolConfig()
        self.base_cut = cut or CutConfig()
        self.sim_config = sim_config or SimulationConfig(duration_s=decision_interval_s)
        self.shield_config = shield_config or ShieldConfig()
        self.reward_config = reward_config or RewardConfig()
        self.margin_calibration = margin_calibration
        self.max_steps = max_steps
        self.decision_interval_s = decision_interval_s
        self.action_space = spaces.Box(
            low=np.array(
                [self.shield_config.min_feed_override, self.shield_config.min_spindle_override],
                dtype=np.float32,
            ),
            high=np.array(
                [self.shield_config.max_feed_override, self.shield_config.max_spindle_override],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        self._step_idx = 0
        self._machine_state = MachineState()
        self._last_obs = np.zeros(12, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._machine_state = MachineState()
        self._last_obs, info = self._evaluate_window(1.0, 1.0)
        return self._last_obs, info

    def step(self, action):
        proposal = ActionProposal(feed_override=float(action[0]), spindle_override=float(action[1]))
        obs_raw, info = self._evaluate_window(
            self._machine_state.last_feed_override,
            self._machine_state.last_spindle_override,
        )
        shielded = apply_safety_shield(proposal, info["risk"], self._machine_state, self.shield_config)
        obs, info = self._evaluate_window(shielded.feed_override, shielded.spindle_override)
        info["shield"] = shielded

        risk = info["risk"]
        reward, reward_terms = compute_control_reward(
            risk=risk,
            shielded=shielded,
            previous_state=self._machine_state,
            config=self.reward_config,
        )
        info["reward_terms"] = reward_terms

        self._machine_state = MachineState(
            last_feed_override=shielded.feed_override,
            last_spindle_override=shielded.spindle_override,
        )
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.max_steps
        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    def _evaluate_window(self, feed_override: float, spindle_override: float):
        cut = replace(
            self.base_cut,
            feed_per_tooth_m=self.base_cut.feed_per_tooth_m * feed_override,
            spindle_rpm=self.base_cut.spindle_rpm * spindle_override,
        )
        sim_config = replace(
            self.sim_config,
            duration_s=self.decision_interval_s,
            random_seed=(self.sim_config.random_seed or 0) + self._step_idx,
        )
        result = simulate_milling(self.modal, self.tool, cut, sim_config)
        features = extract_signal_features(
            result.sensor_signal,
            sim_config.sample_rate_hz,
            cut.spindle_rpm,
            self.tool.flute_count,
            self.modal.natural_frequency_hz,
        )
        raw_margin = signed_stability_margin(self.modal, self.tool, cut)
        margin = apply_margin_calibration(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
        risk = estimate_chatter_risk(features, margin)
        calibration_uncertainty = calibrated_margin_uncertainty(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
        if calibration_uncertainty is not None and calibration_uncertainty > risk.uncertainty:
            risk = replace(
                risk,
                uncertainty=calibration_uncertainty,
                reason_codes=tuple(dict.fromkeys((*risk.reason_codes, "calibration_uncertainty"))),
            )
        obs = np.array(
            [
                feed_override,
                spindle_override,
                cut.axial_depth_m,
                cut.radial_depth_m,
                risk.risk_chatter_now,
                risk.risk_chatter_horizon,
                risk.margin_physics,
                risk.margin_signal,
                risk.uncertainty,
                features.rms,
                features.chatter_band_energy,
                features.non_tooth_harmonic_ratio,
            ],
            dtype=np.float32,
        )
        return obs, {"features": features, "risk": risk, "cut": cut}


def compute_control_reward(
    *,
    risk: RiskEstimate,
    shielded: ShieldResult,
    previous_state: MachineState,
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    config = config or RewardConfig()
    productivity_raw = shielded.feed_override
    if config.productivity_mode == "mrr":
        productivity_raw = shielded.feed_override * shielded.spindle_override
    productivity = config.productivity_weight * productivity_raw
    risk_now_penalty = config.risk_now_weight * risk.risk_chatter_now
    risk_horizon_penalty = config.risk_horizon_weight * risk.risk_chatter_horizon
    severe_penalty = config.severe_penalty if risk.label == "severe" else 0.0
    smoothness_raw = abs(shielded.feed_override - previous_state.last_feed_override)
    smoothness_raw += abs(shielded.spindle_override - previous_state.last_spindle_override)
    smoothness_penalty = config.smoothness_weight * smoothness_raw
    rejection_penalty = config.rejection_penalty if shielded.rejected else 0.0
    clip_count = sum(reason in {"feed_clipped", "spindle_clipped"} for reason in shielded.reasons)
    rate_limit_count = sum(reason in {"feed_rate_limited", "spindle_rate_limited"} for reason in shielded.reasons)
    clip_penalty = config.clip_penalty * clip_count
    rate_limit_penalty = config.rate_limit_penalty * rate_limit_count
    reward = (
        productivity
        - risk_now_penalty
        - risk_horizon_penalty
        - severe_penalty
        - smoothness_penalty
        - rejection_penalty
        - clip_penalty
        - rate_limit_penalty
    )
    return float(reward), {
        "productivity_raw": float(productivity_raw),
        "productivity": float(productivity),
        "risk_now_penalty": float(risk_now_penalty),
        "risk_horizon_penalty": float(risk_horizon_penalty),
        "severe_penalty": float(severe_penalty),
        "smoothness_raw": float(smoothness_raw),
        "smoothness_penalty": float(smoothness_penalty),
        "rejection_penalty": float(rejection_penalty),
        "shield_clip_count": float(clip_count),
        "shield_rate_limit_count": float(rate_limit_count),
        "clip_penalty": float(clip_penalty),
        "rate_limit_penalty": float(rate_limit_penalty),
    }
