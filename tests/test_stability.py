from dataclasses import replace

import numpy as np

from chatter_twin.scenarios import make_scenario
from chatter_twin.stability import (
    critical_axial_depth_m,
    estimate_stability,
    modal_receptance,
    regenerative_phase_factor,
    signed_stability_margin,
)


def test_stability_estimate_has_positive_physical_terms():
    modal, tool, cut, _ = make_scenario("near_boundary")
    estimate = estimate_stability(modal, tool, cut)
    assert estimate.critical_axial_depth_m > 0
    assert estimate.tooth_frequency_hz > 0
    assert estimate.limiting_frequency_hz > 0
    assert estimate.dynamic_compliance_m_n > 0
    assert estimate.directional_factor > 0
    assert np.isfinite(estimate.signed_margin)
    assert critical_axial_depth_m(modal, tool, cut) == estimate.critical_axial_depth_m
    assert signed_stability_margin(modal, tool, cut) == estimate.signed_margin


def test_stability_margin_decreases_with_axial_depth():
    modal, tool, cut, _ = make_scenario("stable")
    deeper_cut = replace(cut, axial_depth_m=cut.axial_depth_m * 3.0)
    assert signed_stability_margin(modal, tool, deeper_cut) < signed_stability_margin(modal, tool, cut)


def test_regenerative_phase_factor_varies_with_spindle_ratio():
    factors = regenerative_phase_factor(np.array([600.0, 750.0, 900.0]), tooth_frequency_hz=600.0)
    assert factors[0] < factors[1]
    assert factors[2] > factors[0]


def test_modal_receptance_peaks_near_natural_frequency():
    modal, _, _, _ = make_scenario("stable")
    near = np.mean([modal.natural_frequency_hz])
    low = near * 0.5
    high = near * 1.5
    near_mag = np.max(np.abs(modal_receptance(modal, near)))
    low_mag = np.max(np.abs(modal_receptance(modal, low)))
    high_mag = np.max(np.abs(modal_receptance(modal, high)))
    assert near_mag > low_mag
    assert near_mag > high_mag


def test_spindle_speed_changes_lobe_limit():
    modal, tool, cut, _ = make_scenario("near_boundary")
    slower = estimate_stability(modal, tool, replace(cut, spindle_rpm=cut.spindle_rpm * 0.92))
    faster = estimate_stability(modal, tool, replace(cut, spindle_rpm=cut.spindle_rpm * 1.08))
    assert abs(slower.critical_axial_depth_m - faster.critical_axial_depth_m) > 1.0e-6
