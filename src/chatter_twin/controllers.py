from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from chatter_twin.calibration import MarginCalibration, apply_margin_calibration, calibrated_margin_uncertainty
from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.models import ActionProposal, CutConfig, MachineState, ModalParams, RiskEstimate, ToolConfig
from chatter_twin.models import SimulationConfig
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.shield import ShieldConfig
from chatter_twin.stability import estimate_stability, signed_stability_margin


class Controller:
    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        raise NotImplementedError


@dataclass
class FixedController(Controller):
    feed_override: float = 1.0
    spindle_override: float = 1.0

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        return ActionProposal(self.feed_override, self.spindle_override)


@dataclass
class RuleBasedController(Controller):
    min_economic_feed: float = 0.82
    feed_step: float = 0.04
    spindle_step: float = 0.04

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        feed = state.last_feed_override
        spindle = state.last_spindle_override
        if risk.risk_chatter_now >= 0.70:
            feed = max(self.min_economic_feed, feed - self.feed_step)
            spindle = min(1.10, spindle + self.spindle_step)
        elif risk.risk_chatter_now >= 0.45:
            spindle = min(1.08, spindle + 0.5 * self.spindle_step)
        else:
            feed = min(1.0, feed + 0.5 * self.feed_step)
            spindle = spindle + 0.25 * (1.0 - spindle)
        return ActionProposal(feed_override=feed, spindle_override=spindle)


@dataclass
class StabilityLobeController(Controller):
    """Physics-margin baseline that schedules spindle override before cutting feed."""

    modal: ModalParams
    tool: ToolConfig
    base_cut: CutConfig
    shield_config: ShieldConfig = ShieldConfig()
    margin_calibration: MarginCalibration | None = None
    min_economic_feed: float = 0.90
    high_risk_threshold: float = 0.80
    feed_step: float = 0.03
    candidate_count: int = 17
    candidate_uncertainty_weight: float = 0.80
    avoid_high_uncertainty_candidates: bool = True
    margin_uncertainty: float = 0.22

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        spindle = self._best_spindle_override(state.last_spindle_override)
        feed = state.last_feed_override
        if risk.risk_chatter_now >= self.high_risk_threshold:
            feed = max(self.min_economic_feed, feed - self.feed_step)
        elif risk.risk_chatter_now < 0.35:
            feed = min(1.0, feed + 0.5 * self.feed_step)
        return ActionProposal(feed_override=feed, spindle_override=spindle)

    def _best_spindle_override(self, current_override: float) -> float:
        lower = max(self.shield_config.min_spindle_override, current_override - self.shield_config.max_spindle_delta)
        upper = min(self.shield_config.max_spindle_override, current_override + self.shield_config.max_spindle_delta)
        candidates = np.linspace(
            lower,
            upper,
            self.candidate_count,
        )
        best_override = current_override
        best_score = float("-inf")
        for candidate in candidates:
            cut = replace(self.base_cut, spindle_rpm=self.base_cut.spindle_rpm * float(candidate))
            raw_margin = signed_stability_margin(self.modal, self.tool, cut)
            margin = apply_margin_calibration(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
            uncertainty = calibrated_margin_uncertainty(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
            if uncertainty is None:
                uncertainty = _risk_uncertainty_from_margin(margin, self.margin_uncertainty)
            else:
                uncertainty = max(uncertainty, _risk_uncertainty_from_margin(margin, self.margin_uncertainty))
            if (
                self.avoid_high_uncertainty_candidates
                and uncertainty > self.shield_config.max_uncertainty
                and float(candidate) != current_override
            ):
                continue
            closeness_penalty = 0.05 * abs(float(candidate) - current_override)
            productivity_penalty = 0.01 * abs(float(candidate) - 1.0)
            score = margin - self.candidate_uncertainty_weight * uncertainty - closeness_penalty - productivity_penalty
            if score > best_score:
                best_score = score
                best_override = float(candidate)
        return best_override


@dataclass
class LookaheadMpcController(Controller):
    """Grid-search lookahead baseline over safe feed and spindle overrides."""

    modal: ModalParams
    tool: ToolConfig
    base_cut: CutConfig
    shield_config: ShieldConfig = ShieldConfig()
    margin_calibration: MarginCalibration | None = None
    feed_candidates: int = 7
    spindle_candidates: int = 9
    risk_weight: float = 2.8
    productivity_weight: float = 1.0
    smoothness_weight: float = 0.45
    uncertainty_weight: float = 0.35
    candidate_uncertainty_weight: float = 0.90
    high_uncertainty_penalty: float = 3.0
    avoid_high_uncertainty_candidates: bool = True
    high_risk_feed_increase_weight: float = 4.0
    margin_uncertainty: float = 0.22

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        best_action = ActionProposal(state.last_feed_override, state.last_spindle_override)
        best_score = float("-inf")
        for feed in self._feed_grid(state.last_feed_override):
            for spindle in self._spindle_grid(state.last_spindle_override):
                score = self._score(float(feed), float(spindle), risk, state)
                if score > best_score:
                    best_score = score
                    best_action = ActionProposal(feed_override=float(feed), spindle_override=float(spindle))
        return best_action

    def _score(self, feed: float, spindle: float, risk: RiskEstimate, state: MachineState) -> float:
        cut = replace(
            self.base_cut,
            feed_per_tooth_m=self.base_cut.feed_per_tooth_m * feed,
            spindle_rpm=self.base_cut.spindle_rpm * spindle,
        )
        raw_margin = signed_stability_margin(self.modal, self.tool, cut)
        margin = apply_margin_calibration(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
        physics_risk = _sigmoid(-margin / self.margin_uncertainty)
        candidate_uncertainty = calibrated_margin_uncertainty(raw_margin, self.margin_calibration, self.modal, self.tool, cut)
        if candidate_uncertainty is None:
            candidate_uncertainty = risk.uncertainty
        candidate_uncertainty = max(candidate_uncertainty, _risk_uncertainty_from_margin(margin, self.margin_uncertainty))
        if (
            self.avoid_high_uncertainty_candidates
            and candidate_uncertainty > self.shield_config.max_uncertainty
            and (feed != state.last_feed_override or spindle != state.last_spindle_override)
        ):
            return float("-inf")
        load_relief = 0.25 * max(0.0, 1.0 - feed)
        expected_risk = min(1.0, max(0.0, 0.65 * physics_risk + 0.35 * risk.risk_chatter_now - load_relief))
        productivity = feed * spindle
        smoothness = abs(feed - state.last_feed_override) + abs(spindle - state.last_spindle_override)
        uncertainty_penalty = self.uncertainty_weight * risk.uncertainty
        future_uncertainty_penalty = self.candidate_uncertainty_weight * candidate_uncertainty
        if candidate_uncertainty > self.shield_config.max_uncertainty:
            future_uncertainty_penalty += self.high_uncertainty_penalty * (candidate_uncertainty - self.shield_config.max_uncertainty)
        severe_penalty = 0.75 if margin < -1.0 else 0.0
        feed_increase_penalty = 0.0
        if risk.risk_chatter_now > 0.70 and feed > state.last_feed_override:
            feed_increase_penalty = self.high_risk_feed_increase_weight * (feed - state.last_feed_override)
        return (
            self.productivity_weight * productivity
            - self.risk_weight * expected_risk
            - self.smoothness_weight * smoothness
            - uncertainty_penalty
            - future_uncertainty_penalty
            - severe_penalty
            - feed_increase_penalty
        )

    def _feed_grid(self, current: float) -> np.ndarray:
        lower = max(self.shield_config.min_feed_override, current - self.shield_config.max_feed_delta)
        upper = min(self.shield_config.max_feed_override, current + self.shield_config.max_feed_delta)
        return np.linspace(lower, upper, self.feed_candidates)

    def _spindle_grid(self, current: float) -> np.ndarray:
        lower = max(self.shield_config.min_spindle_override, current - self.shield_config.max_spindle_delta)
        upper = min(self.shield_config.max_spindle_override, current + self.shield_config.max_spindle_delta)
        return np.linspace(lower, upper, self.spindle_candidates)


@dataclass
class CounterfactualRiskController(Controller):
    """Receding-horizon baseline that chooses actions by local simulated risk."""

    modal: ModalParams
    tool: ToolConfig
    base_cut: CutConfig
    sim_config: SimulationConfig = SimulationConfig()
    shield_config: ShieldConfig = ShieldConfig()
    margin_calibration: MarginCalibration | None = None
    feed_values: tuple[float, ...] = (1.0,)
    spindle_values: tuple[float, ...] = (0.96, 1.0, 1.04, 1.08)
    intervention_threshold: float = 0.45
    min_risk_reduction: float = 0.005
    risk_weight: float = 1.0
    productivity_weight: float = 0.02
    move_penalty: float = 0.04
    uncertainty_weight: float = 0.45
    high_uncertainty_penalty: float = 3.0
    planning_noise_std: float = 0.0
    planning_duration_s: float = 0.12

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        current = ActionProposal(state.last_feed_override, state.last_spindle_override)
        if risk.risk_chatter_now < self.intervention_threshold:
            return self._recover_toward_nominal(state)

        baseline_risk = self._predict_risk(current.feed_override, current.spindle_override)
        best_action = current
        best_reduction = 0.0
        best_score = self._score(0.0, baseline_risk.uncertainty, current.feed_override, current.spindle_override, state)
        for feed in self._feed_grid(state):
            for spindle in self._spindle_grid(state):
                candidate_risk = self._predict_risk(float(feed), float(spindle))
                if candidate_risk.uncertainty > self.shield_config.max_uncertainty and (
                    float(feed) != current.feed_override or float(spindle) != current.spindle_override
                ):
                    continue
                reduction = baseline_risk.risk_chatter_now - candidate_risk.risk_chatter_now
                score = self._score(reduction, candidate_risk.uncertainty, float(feed), float(spindle), state)
                if score > best_score:
                    best_action = ActionProposal(feed_override=float(feed), spindle_override=float(spindle))
                    best_reduction = reduction
                    best_score = score

        if best_reduction < self.min_risk_reduction:
            return current
        return best_action

    def _predict_risk(self, feed: float, spindle: float) -> RiskEstimate:
        cut = replace(
            self.base_cut,
            feed_per_tooth_m=self.base_cut.feed_per_tooth_m * feed,
            spindle_rpm=self.base_cut.spindle_rpm * spindle,
        )
        config = replace(
            self.sim_config,
            duration_s=self.planning_duration_s,
            sensor_noise_std=self.planning_noise_std,
        )
        result = simulate_milling(self.modal, self.tool, cut, config)
        features = extract_signal_features(
            result.sensor_signal,
            config.sample_rate_hz,
            cut.spindle_rpm,
            self.tool.flute_count,
            self.modal.natural_frequency_hz,
        )
        stability = estimate_stability(self.modal, self.tool, cut)
        margin = apply_margin_calibration(stability.signed_margin, self.margin_calibration, self.modal, self.tool, cut)
        risk = estimate_chatter_risk(features, margin)
        uncertainty = calibrated_margin_uncertainty(stability.signed_margin, self.margin_calibration, self.modal, self.tool, cut)
        if uncertainty is not None and uncertainty > risk.uncertainty:
            risk = replace(
                risk,
                uncertainty=uncertainty,
                reason_codes=tuple(dict.fromkeys((*risk.reason_codes, "calibration_uncertainty"))),
            )
        return risk

    def _score(self, reduction: float, uncertainty: float, feed: float, spindle: float, state: MachineState) -> float:
        productivity = feed * spindle
        move = abs(feed - state.last_feed_override) + abs(spindle - state.last_spindle_override)
        high_uncertainty = max(0.0, uncertainty - self.shield_config.max_uncertainty)
        return (
            self.risk_weight * reduction
            + self.productivity_weight * productivity
            - self.move_penalty * move
            - self.uncertainty_weight * uncertainty
            - self.high_uncertainty_penalty * high_uncertainty
        )

    def _feed_grid(self, state: MachineState) -> tuple[float, ...]:
        return tuple(
            value
            for value in sorted({1.0, *self.feed_values})
            if self.shield_config.min_feed_override <= value <= self.shield_config.max_feed_override
            and abs(value - state.last_feed_override) <= self.shield_config.max_feed_delta + 1.0e-12
        )

    def _spindle_grid(self, state: MachineState) -> tuple[float, ...]:
        return tuple(
            value
            for value in sorted({1.0, *self.spindle_values})
            if self.shield_config.min_spindle_override <= value <= self.shield_config.max_spindle_override
            and abs(value - state.last_spindle_override) <= self.shield_config.max_spindle_delta + 1.0e-12
        )

    def _recover_toward_nominal(self, state: MachineState) -> ActionProposal:
        feed_delta = np.clip(1.0 - state.last_feed_override, -0.5 * self.shield_config.max_feed_delta, 0.5 * self.shield_config.max_feed_delta)
        spindle_delta = np.clip(1.0 - state.last_spindle_override, -0.5 * self.shield_config.max_spindle_delta, 0.5 * self.shield_config.max_spindle_delta)
        return ActionProposal(
            feed_override=float(state.last_feed_override + feed_delta),
            spindle_override=float(state.last_spindle_override + spindle_delta),
        )


@dataclass
class HybridRiskMarginController(CounterfactualRiskController):
    """Hybrid baseline combining simulated risk reduction and stability margin."""

    margin_weight: float = 0.18
    min_margin_improvement: float = 0.05
    max_predicted_risk_worsening: float = 0.002

    def propose(self, risk: RiskEstimate, state: MachineState) -> ActionProposal:
        current = ActionProposal(state.last_feed_override, state.last_spindle_override)
        if risk.risk_chatter_now < self.intervention_threshold and risk.margin_physics > 0.1:
            return self._recover_toward_nominal(state)

        baseline_risk = self._predict_risk(current.feed_override, current.spindle_override)
        baseline_margin = self._predict_margin(current.feed_override, current.spindle_override)
        best_action = current
        best_reduction = 0.0
        best_margin_improvement = 0.0
        best_score = self._hybrid_score(0.0, baseline_risk.uncertainty, 0.0, current.feed_override, current.spindle_override, state)
        for feed in self._feed_grid(state):
            for spindle in self._spindle_grid(state):
                candidate_risk = self._predict_risk(float(feed), float(spindle))
                if candidate_risk.uncertainty > self.shield_config.max_uncertainty and (
                    float(feed) != current.feed_override or float(spindle) != current.spindle_override
                ):
                    continue
                candidate_margin = self._predict_margin(float(feed), float(spindle))
                reduction = baseline_risk.risk_chatter_now - candidate_risk.risk_chatter_now
                if reduction < -self.max_predicted_risk_worsening:
                    continue
                margin_improvement = candidate_margin - baseline_margin
                score = self._hybrid_score(reduction, candidate_risk.uncertainty, margin_improvement, float(feed), float(spindle), state)
                if score > best_score:
                    best_action = ActionProposal(feed_override=float(feed), spindle_override=float(spindle))
                    best_reduction = reduction
                    best_margin_improvement = margin_improvement
                    best_score = score

        if best_reduction < self.min_risk_reduction and best_margin_improvement < self.min_margin_improvement:
            return current
        return best_action

    def _predict_margin(self, feed: float, spindle: float) -> float:
        cut = replace(
            self.base_cut,
            feed_per_tooth_m=self.base_cut.feed_per_tooth_m * feed,
            spindle_rpm=self.base_cut.spindle_rpm * spindle,
        )
        return apply_margin_calibration(signed_stability_margin(self.modal, self.tool, cut), self.margin_calibration, self.modal, self.tool, cut)

    def _hybrid_score(
        self,
        reduction: float,
        uncertainty: float,
        margin_improvement: float,
        feed: float,
        spindle: float,
        state: MachineState,
    ) -> float:
        base_score = self._score(reduction, uncertainty, feed, spindle, state)
        return base_score + self.margin_weight * margin_improvement


def make_controller(
    name: str,
    *,
    modal: ModalParams | None = None,
    tool: ToolConfig | None = None,
    cut: CutConfig | None = None,
    sim_config: SimulationConfig | None = None,
    margin_calibration: MarginCalibration | None = None,
) -> Controller:
    match name:
        case "fixed":
            return FixedController()
        case "rule":
            return RuleBasedController()
        case "sld":
            return StabilityLobeController(
                modal=modal or ModalParams(),
                tool=tool or ToolConfig(),
                base_cut=cut or CutConfig(),
                margin_calibration=margin_calibration,
            )
        case "mpc":
            return LookaheadMpcController(
                modal=modal or ModalParams(),
                tool=tool or ToolConfig(),
                base_cut=cut or CutConfig(),
                margin_calibration=margin_calibration,
            )
        case "cf":
            return CounterfactualRiskController(
                modal=modal or ModalParams(),
                tool=tool or ToolConfig(),
                base_cut=cut or CutConfig(),
                sim_config=sim_config or SimulationConfig(),
                margin_calibration=margin_calibration,
            )
        case "hybrid":
            return HybridRiskMarginController(
                modal=modal or ModalParams(),
                tool=tool or ToolConfig(),
                base_cut=cut or CutConfig(),
                sim_config=sim_config or SimulationConfig(),
                margin_calibration=margin_calibration,
            )
        case _:
            raise ValueError(f"Unknown controller {name!r}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _risk_uncertainty_from_margin(margin: float, margin_uncertainty: float) -> float:
    margin_uncertainty = max(margin_uncertainty, 1.0e-6)
    return float(min(0.95, 0.12 + 0.55 * np.exp(-abs(margin) / margin_uncertainty)))
