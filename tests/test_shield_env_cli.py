from pathlib import Path

import numpy as np

from chatter_twin.calibration import MarginCalibration
from chatter_twin.cli import main
from chatter_twin.controllers import (
    CounterfactualRiskController,
    HybridRiskMarginController,
    LookaheadMpcController,
    RuleBasedController,
    StabilityLobeController,
    make_controller,
)
from chatter_twin.env import ChatterSuppressEnv, compute_control_reward
from chatter_twin.models import ActionProposal, MachineState, RewardConfig, RiskEstimate, ShieldConfig, ShieldResult
from chatter_twin.scenarios import make_scenario
from chatter_twin.shield import apply_safety_shield
from chatter_twin.stability import signed_stability_margin


def test_shield_clips_and_rate_limits_out_of_range_action():
    risk = RiskEstimate(0.2, 0.2, 0.5, 0.5, "stable", 0.2, ("nominal",))
    state = MachineState(last_feed_override=1.0, last_spindle_override=1.0)
    result = apply_safety_shield(ActionProposal(2.0, 0.2), risk, state)
    assert result.accepted
    assert not result.rejected
    assert result.feed_override == 1.05
    assert result.spindle_override == 0.95
    assert "feed_clipped" in result.reasons
    assert "spindle_clipped" in result.reasons


def test_shield_rejects_high_uncertainty():
    risk = RiskEstimate(0.5, 0.5, 0.0, 0.0, "transition", 0.9, ("near_stability_boundary",))
    state = MachineState(last_feed_override=0.95, last_spindle_override=1.02)
    result = apply_safety_shield(ActionProposal(1.0, 1.0), risk, state)
    assert result.rejected
    assert not result.accepted
    assert result.feed_override == 0.95
    assert "high_uncertainty" in result.reasons


def test_shield_holds_high_uncertainty_when_configured():
    risk = RiskEstimate(0.5, 0.5, 0.0, 0.0, "transition", 0.9, ("near_stability_boundary",))
    state = MachineState(last_feed_override=0.95, last_spindle_override=1.02)
    result = apply_safety_shield(
        ActionProposal(1.03, 1.04),
        risk,
        state,
        ShieldConfig(uncertainty_mode="hold"),
    )
    assert result.accepted
    assert not result.rejected
    assert result.feed_override == 0.95
    assert result.spindle_override == 1.02
    assert "high_uncertainty" in result.reasons
    assert "uncertainty_hold" in result.reasons


def test_shield_rejects_hard_failure_even_in_uncertainty_hold_mode():
    risk = RiskEstimate(0.5, 0.5, 0.0, 0.0, "transition", 0.9, ("near_stability_boundary",))
    state = MachineState(sensor_healthy=False, last_feed_override=0.95, last_spindle_override=1.02)
    result = apply_safety_shield(
        ActionProposal(1.03, 1.04),
        risk,
        state,
        ShieldConfig(uncertainty_mode="hold"),
    )
    assert result.rejected
    assert not result.accepted
    assert "sensor_unhealthy" in result.reasons


def test_env_reset_step_api_returns_finite_values():
    env = ChatterSuppressEnv(max_steps=2)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert terminated is False
    assert "risk" in info


def test_reward_config_switches_productivity_proxy():
    risk = RiskEstimate(0.1, 0.2, 0.4, 0.3, "stable", 0.2, ("nominal",))
    state = MachineState(last_feed_override=1.0, last_spindle_override=1.0)
    shielded = ShieldResult(feed_override=0.9, spindle_override=1.1, accepted=True, rejected=False)

    feed_reward, feed_terms = compute_control_reward(
        risk=risk,
        shielded=shielded,
        previous_state=state,
        config=RewardConfig(productivity_mode="feed", productivity_weight=1.0),
    )
    mrr_reward, mrr_terms = compute_control_reward(
        risk=risk,
        shielded=shielded,
        previous_state=state,
        config=RewardConfig(productivity_mode="mrr", productivity_weight=1.0),
    )

    assert feed_terms["productivity_raw"] == 0.9
    assert mrr_terms["productivity_raw"] == 0.9900000000000001
    assert mrr_reward > feed_reward


def test_reward_config_penalizes_shield_adjustments():
    risk = RiskEstimate(0.1, 0.2, 0.4, 0.3, "stable", 0.2, ("nominal",))
    state = MachineState(last_feed_override=1.0, last_spindle_override=1.0)
    clean = ShieldResult(feed_override=1.0, spindle_override=1.0, accepted=True, rejected=False)
    adjusted = ShieldResult(
        feed_override=1.0,
        spindle_override=1.0,
        accepted=True,
        rejected=False,
        reasons=("feed_clipped", "spindle_rate_limited"),
    )

    clean_reward, _ = compute_control_reward(
        risk=risk,
        shielded=clean,
        previous_state=state,
        config=RewardConfig(clip_penalty=0.25, rate_limit_penalty=0.5),
    )
    adjusted_reward, terms = compute_control_reward(
        risk=risk,
        shielded=adjusted,
        previous_state=state,
        config=RewardConfig(clip_penalty=0.25, rate_limit_penalty=0.5),
    )

    assert terms["shield_clip_count"] == 1.0
    assert terms["shield_rate_limit_count"] == 1.0
    assert adjusted_reward == clean_reward - 0.75


def test_rule_controller_respects_minimum_economic_feed():
    controller = RuleBasedController(min_economic_feed=0.82)
    risk = RiskEstimate(0.9, 0.95, -1.0, -0.8, "severe", 0.3, ("negative_physics_margin",))
    state = MachineState(last_feed_override=0.83, last_spindle_override=1.0)
    action = controller.propose(risk, state)
    assert action.feed_override >= 0.82


def test_sld_controller_improves_or_preserves_physics_margin():
    modal, tool, cut, _ = make_scenario("near_boundary")
    controller = StabilityLobeController(modal=modal, tool=tool, base_cut=cut)
    risk = RiskEstimate(0.7, 0.8, -0.5, -0.3, "slight", 0.2, ("negative_physics_margin",))
    action = controller.propose(risk, MachineState())
    base_margin = signed_stability_margin(modal, tool, cut)
    adjusted_cut = type(cut)(
        spindle_rpm=cut.spindle_rpm * action.spindle_override,
        feed_per_tooth_m=cut.feed_per_tooth_m,
        axial_depth_m=cut.axial_depth_m,
        radial_depth_m=cut.radial_depth_m,
        cutting_coeff_t_n_m2=cut.cutting_coeff_t_n_m2,
        cutting_coeff_r_n_m2=cut.cutting_coeff_r_n_m2,
        immersion_start_rad=cut.immersion_start_rad,
        immersion_end_rad=cut.immersion_end_rad,
    )
    assert signed_stability_margin(modal, tool, adjusted_cut) >= base_margin
    assert 0.90 <= action.spindle_override <= 1.10


def test_make_controller_supports_sld():
    modal, tool, cut, _ = make_scenario("stable")
    assert make_controller("sld", modal=modal, tool=tool, cut=cut)


def test_mpc_controller_returns_shield_compatible_action():
    modal, tool, cut, _ = make_scenario("unstable")
    controller = LookaheadMpcController(modal=modal, tool=tool, base_cut=cut)
    risk = RiskEstimate(0.85, 0.9, -2.0, -0.5, "severe", 0.2, ("negative_physics_margin",))
    action = controller.propose(risk, MachineState())
    assert 0.80 <= action.feed_override <= 1.05
    assert 0.90 <= action.spindle_override <= 1.10
    assert action.feed_override <= 1.0


def test_make_controller_supports_mpc():
    modal, tool, cut, _ = make_scenario("stable")
    assert make_controller("mpc", modal=modal, tool=tool, cut=cut)


def test_counterfactual_controller_returns_shield_compatible_action():
    modal, tool, cut, sim_config = make_scenario("unstable")
    controller = CounterfactualRiskController(modal=modal, tool=tool, base_cut=cut, sim_config=sim_config)
    risk = RiskEstimate(0.60, 0.65, -2.0, -0.5, "slight", 0.2, ("negative_physics_margin",))
    action = controller.propose(risk, MachineState())
    assert 0.80 <= action.feed_override <= 1.05
    assert 0.90 <= action.spindle_override <= 1.10
    assert action.feed_override == 1.0


def test_make_controller_supports_counterfactual():
    modal, tool, cut, sim_config = make_scenario("stable")
    assert make_controller("cf", modal=modal, tool=tool, cut=cut, sim_config=sim_config)


def test_hybrid_controller_returns_shield_compatible_action():
    modal, tool, cut, sim_config = make_scenario("unstable")
    controller = HybridRiskMarginController(modal=modal, tool=tool, base_cut=cut, sim_config=sim_config)
    risk = RiskEstimate(0.85, 0.9, -2.0, -0.5, "severe", 0.2, ("negative_physics_margin",))
    action = controller.propose(risk, MachineState())
    assert 0.80 <= action.feed_override <= 1.05
    assert 0.90 <= action.spindle_override <= 1.10


def test_uncertainty_aware_controllers_hold_current_action_when_candidates_are_too_uncertain():
    modal, tool, cut, sim_config = make_scenario("unstable")
    calibration = MarginCalibration(
        intercept=0.0,
        slope=0.0,
        train_brier_score=0.35,
        holdout_brier_score=0.35,
    )
    risk = RiskEstimate(0.60, 0.65, -2.0, -0.5, "slight", 0.2, ("negative_physics_margin",))
    state = MachineState(last_feed_override=0.97, last_spindle_override=1.02)

    controllers = [
        StabilityLobeController(modal=modal, tool=tool, base_cut=cut, margin_calibration=calibration),
        LookaheadMpcController(modal=modal, tool=tool, base_cut=cut, margin_calibration=calibration),
        CounterfactualRiskController(modal=modal, tool=tool, base_cut=cut, sim_config=sim_config, margin_calibration=calibration),
        HybridRiskMarginController(modal=modal, tool=tool, base_cut=cut, sim_config=sim_config, margin_calibration=calibration),
    ]

    for controller in controllers:
        action = controller.propose(risk, state)
        assert action.feed_override == state.last_feed_override
        assert action.spindle_override == state.last_spindle_override


def test_make_controller_supports_hybrid():
    modal, tool, cut, sim_config = make_scenario("stable")
    assert make_controller("hybrid", modal=modal, tool=tool, cut=cut, sim_config=sim_config)


def test_cli_simulate_writes_npz(tmp_path: Path):
    out = tmp_path / "stable.npz"
    status = main(["simulate", "--scenario", "stable", "--duration", "0.1", "--out", str(out)])
    assert status == 0
    data = np.load(out)
    assert "sensor_signal" in data


def test_cli_env_smoke_runs():
    assert main(["env-smoke"]) == 0
