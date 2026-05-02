import numpy as np

from chatter_twin.features import extract_signal_features
from chatter_twin.risk import estimate_chatter_risk


def test_feature_extraction_finds_dominant_tooth_frequency():
    sample_rate = 5000.0
    spindle_rpm = 6000.0
    flute_count = 4
    tooth_frequency = spindle_rpm * flute_count / 60.0
    t = np.arange(0, 1.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * tooth_frequency * t)
    features = extract_signal_features(signal, sample_rate, spindle_rpm, flute_count)
    assert abs(features.dominant_frequency_hz - tooth_frequency) < 8.0
    assert features.tooth_band_energy > features.chatter_band_energy


def test_risk_estimator_marks_unhealthy_sensor_unknown():
    sample_rate = 2000.0
    t = np.arange(0, 0.5, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 200.0 * t)
    features = extract_signal_features(signal, sample_rate, 6000.0, 2)
    risk = estimate_chatter_risk(features, margin_physics=0.4, sensor_healthy=False)
    assert risk.label == "unknown"
    assert risk.uncertainty > 0.9
    assert "sensor_unhealthy" in risk.reason_codes


def test_negative_margin_increases_risk_label():
    sample_rate = 5000.0
    t = np.arange(0, 0.5, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 900.0 * t)
    features = extract_signal_features(signal, sample_rate, 6000.0, 4, modal_frequency_hz=900.0)
    risk = estimate_chatter_risk(features, margin_physics=-0.8)
    assert risk.risk_chatter_now > 0.7
    assert risk.label in {"slight", "severe"}
    assert "negative_physics_margin" in risk.reason_codes
