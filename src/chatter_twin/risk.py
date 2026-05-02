from __future__ import annotations

import math

from chatter_twin.models import ChatterLabel, RiskEstimate, SignalFeatures


def estimate_chatter_risk(
    features: SignalFeatures,
    margin_physics: float,
    *,
    margin_uncertainty: float = 0.20,
    sensor_healthy: bool = True,
    in_cut: bool = True,
) -> RiskEstimate:
    """Hybrid physics/signal risk estimate with conservative labels."""

    if not sensor_healthy or not in_cut:
        reason = "sensor_unhealthy" if not sensor_healthy else "not_in_cut"
        return RiskEstimate(
            risk_chatter_now=0.5,
            risk_chatter_horizon=0.5,
            margin_physics=margin_physics,
            margin_signal=0.0,
            label="unknown",
            uncertainty=0.95,
            reason_codes=(reason,),
        )

    margin_uncertainty = max(margin_uncertainty, 1.0e-6)
    physics_score = _sigmoid(-margin_physics / margin_uncertainty)
    chatter_ratio = features.chatter_band_energy / max(features.tooth_band_energy, 1.0e-18)
    ratio_score = _sigmoid((math.log10(chatter_ratio + 1.0e-12) + 0.4) * 2.0)
    crest_score = _sigmoid((features.crest_factor - 4.0) / 1.5)
    entropy_score = _sigmoid((features.spectral_entropy - 0.55) * 6.0)
    signal_score = min(1.0, 0.55 * ratio_score + 0.25 * crest_score + 0.20 * entropy_score)

    risk_now = min(1.0, max(0.0, 0.55 * physics_score + 0.45 * signal_score))
    risk_horizon = min(1.0, max(risk_now, 0.75 * physics_score + 0.25 * signal_score))
    uncertainty = min(0.95, 0.12 + 0.55 * math.exp(-abs(margin_physics) / margin_uncertainty))
    margin_signal = 1.0 - 2.0 * signal_score
    label = _label_from_risk(risk_now)
    reasons = _reason_codes(features, margin_physics, signal_score, physics_score)

    return RiskEstimate(
        risk_chatter_now=float(risk_now),
        risk_chatter_horizon=float(risk_horizon),
        margin_physics=float(margin_physics),
        margin_signal=float(margin_signal),
        label=label,
        uncertainty=float(uncertainty),
        reason_codes=reasons,
    )


def _label_from_risk(risk: float) -> ChatterLabel:
    if risk < 0.30:
        return "stable"
    if risk < 0.55:
        return "transition"
    if risk < 0.80:
        return "slight"
    return "severe"


def _reason_codes(
    features: SignalFeatures,
    margin: float,
    signal_score: float,
    physics_score: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if margin < 0:
        reasons.append("negative_physics_margin")
    if signal_score > 0.55:
        reasons.append("chatter_band_energy")
    if physics_score > 0.70:
        reasons.append("near_stability_boundary")
    if features.crest_factor > 5.0:
        reasons.append("impulsive_vibration")
    if not reasons:
        reasons.append("nominal")
    return tuple(reasons)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
