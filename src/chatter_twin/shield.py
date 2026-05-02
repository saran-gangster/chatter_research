from __future__ import annotations

from chatter_twin.models import ActionProposal, MachineState, RiskEstimate, ShieldConfig, ShieldResult


def apply_safety_shield(
    action: ActionProposal,
    risk: RiskEstimate,
    state: MachineState | None = None,
    config: ShieldConfig | None = None,
) -> ShieldResult:
    """Clip or reject a controller action before any machine-facing use."""

    state = state or MachineState()
    config = config or ShieldConfig()
    reasons: list[str] = []

    if state.controller_mode not in config.allowed_controller_modes:
        reasons.append("controller_mode_blocked")
    if not state.sensor_healthy:
        reasons.append("sensor_unhealthy")
    if not state.in_cut:
        reasons.append("not_in_cut")
    if risk.label == "unknown" and not config.allow_unknown_state:
        reasons.append("unknown_chatter_state")
    high_uncertainty = risk.uncertainty > config.max_uncertainty
    if high_uncertainty:
        reasons.append("high_uncertainty")

    if _has_hard_rejection(reasons) or (high_uncertainty and config.uncertainty_mode == "reject"):
        return ShieldResult(
            feed_override=state.last_feed_override,
            spindle_override=state.last_spindle_override,
            accepted=False,
            rejected=True,
            reasons=tuple(reasons),
        )

    if high_uncertainty and config.uncertainty_mode == "hold":
        reasons.append("uncertainty_hold")
        return ShieldResult(
            feed_override=state.last_feed_override,
            spindle_override=state.last_spindle_override,
            accepted=True,
            rejected=False,
            reasons=tuple(reasons),
        )

    feed = _clip(action.feed_override, config.min_feed_override, config.max_feed_override)
    spindle = _clip(action.spindle_override, config.min_spindle_override, config.max_spindle_override)
    if feed != action.feed_override:
        reasons.append("feed_clipped")
    if spindle != action.spindle_override:
        reasons.append("spindle_clipped")

    feed_limited = _rate_limit(feed, state.last_feed_override, config.max_feed_delta)
    spindle_limited = _rate_limit(spindle, state.last_spindle_override, config.max_spindle_delta)
    if feed_limited != feed:
        reasons.append("feed_rate_limited")
    if spindle_limited != spindle:
        reasons.append("spindle_rate_limited")

    return ShieldResult(
        feed_override=feed_limited,
        spindle_override=spindle_limited,
        accepted=True,
        rejected=False,
        reasons=tuple(reasons),
    )


def _has_hard_rejection(reasons: list[str]) -> bool:
    hard_rejections = {
        "controller_mode_blocked",
        "sensor_unhealthy",
        "not_in_cut",
        "unknown_chatter_state",
    }
    return any(reason in hard_rejections for reason in reasons)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    return _clip(value, previous - max_delta, previous + max_delta)
