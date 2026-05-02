import numpy as np

from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.scenarios import make_scenario
from chatter_twin.stability import signed_stability_margin


def test_simulator_returns_finite_expected_shapes():
    modal, tool, cut, config = make_scenario("stable")
    result = simulate_milling(modal, tool, cut, config)
    assert result.displacement_m.shape == (result.time_s.size, 2)
    assert result.velocity_m_s.shape == (result.time_s.size, 2)
    assert result.acceleration_m_s2.shape == (result.time_s.size, 2)
    assert result.cutting_force_n.shape == (result.time_s.size, 2)
    assert result.sensor_signal.shape == (result.time_s.size, 2)
    assert np.isfinite(result.sensor_signal).all()


def test_unstable_scenario_has_higher_risk_than_stable():
    stable = _scenario_risk("stable")
    unstable = _scenario_risk("unstable")
    assert unstable.risk_chatter_now > stable.risk_chatter_now
    assert unstable.margin_physics < stable.margin_physics
    assert unstable.label in {"slight", "severe"}


def _scenario_risk(name):
    modal, tool, cut, config = make_scenario(name)
    result = simulate_milling(modal, tool, cut, config)
    features = extract_signal_features(
        result.sensor_signal,
        config.sample_rate_hz,
        cut.spindle_rpm,
        tool.flute_count,
        modal.natural_frequency_hz,
    )
    margin = signed_stability_margin(modal, tool, cut)
    return estimate_chatter_risk(features, margin)
