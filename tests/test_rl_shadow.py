from pathlib import Path
import json

import pytest

from chatter_twin.cli import main
from chatter_twin.replay import DomainRandomizationConfig
from chatter_twin.rl import Sb3TrainingConfig
from chatter_twin.rl_shadow import (
    RLShadowGateConfig,
    RLShadowReplayConfig,
    gate_rl_shadow_replay,
    recommendations_from_action_trace,
    summarize_rl_shadow_recommendations,
)


def test_shadow_recommendations_are_advisory_only():
    recommendations = recommendations_from_action_trace(
        _action_trace_rows(),
        source_model_path=Path("runs/td3_seed_616/model.zip"),
        profile_label="fixed_mrr2_safety_first",
    )

    assert len(recommendations) == 2
    assert all(row["shadow_only"] for row in recommendations)
    assert all(row["cnc_write_enabled"] is False for row in recommendations)
    assert recommendations[0]["action_active"] is False
    assert recommendations[0]["shadow_warning"] is False
    assert recommendations[1]["action_active"] is True
    assert recommendations[1]["shadow_warning"] is True
    assert recommendations[1]["relative_mrr_proxy"] == pytest.approx(0.99)


def test_shadow_summary_reports_boundary_and_window_metrics():
    recommendations = recommendations_from_action_trace(
        _action_trace_rows(),
        source_model_path=Path("runs/td3_seed_616/model.zip"),
        profile_label="fixed_mrr2_safety_first",
    )

    payload = summarize_rl_shadow_recommendations(
        recommendations,
        selection={
            "selected": {
                "seed": 616,
                "algorithm": "td3",
                "source_model_path": "runs/td3_seed_616/model.zip",
            }
        },
        replay_config=RLShadowReplayConfig(scenarios=("stable", "unstable"), episodes=1, steps=2),
        sb3_config=Sb3TrainingConfig(algorithm="td3", scenarios=("stable", "unstable"), steps=2),
        randomization=DomainRandomizationConfig(enabled=True),
        selection_path=Path("selected_policy.json"),
    )

    assert payload["deployment_boundary"]["mode"] == "shadow_only"
    assert payload["deployment_boundary"]["cnc_writes_enabled"] is False
    assert payload["window_metrics"]["recommendation_windows"] == 2
    assert payload["window_metrics"]["action_windows"] == 1
    assert payload["window_metrics"]["action_fraction"] == 0.5
    assert payload["window_metrics"]["guard_fallbacks"] == 1
    assert payload["window_metrics"]["shield_rejections"] == 1
    assert payload["scenario_metrics"][0]["scenario"] == "stable"
    assert payload["scenario_metrics"][1]["scenario"] == "unstable"


def test_gate_rl_shadow_replay_passes_shadow_review_candidate(tmp_path):
    shadow_dir = tmp_path / "shadow"
    _write_shadow_metrics(shadow_dir, mean_risk=0.50)

    payload = gate_rl_shadow_replay(
        shadow_dir=shadow_dir,
        out_dir=tmp_path / "gate",
        config=RLShadowGateConfig.from_profile("shadow_review"),
    )

    assert payload["decision"]["status"] == "pass"
    assert payload["decision"]["promotion_level"] == "shadow_review_candidate"
    assert payload["decision"]["hardware_actuation_allowed"] is False
    assert payload["decision"]["failed_checks"] == []
    assert (tmp_path / "gate" / "gate_checks.csv").exists()
    assert "Passing this gate does not authorize CNC writes" in (tmp_path / "gate" / "gate_report.md").read_text(
        encoding="utf-8"
    )


def test_gate_rl_shadow_replay_blocks_failed_risk(tmp_path):
    shadow_dir = tmp_path / "shadow"
    _write_shadow_metrics(shadow_dir, mean_risk=0.62)

    payload = gate_rl_shadow_replay(
        shadow_dir=shadow_dir,
        out_dir=tmp_path / "gate",
        config=RLShadowGateConfig(max_mean_risk=0.55),
    )

    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["promotion_level"] == "do_not_promote"
    assert payload["decision"]["failed_checks"] == ["mean_risk"]


def test_live_shadow_profile_requires_real_machine_evidence(tmp_path):
    shadow_dir = tmp_path / "shadow"
    _write_shadow_metrics(
        shadow_dir,
        mean_risk=0.32,
        max_risk=0.42,
        unstable_risk=0.34,
        action_fraction=0.25,
        guard_fallback_fraction=0.0,
        recommendation_windows=400,
    )

    payload = gate_rl_shadow_replay(
        shadow_dir=shadow_dir,
        out_dir=tmp_path / "gate",
        config=RLShadowGateConfig.from_profile("live_shadow"),
    )

    assert payload["decision"]["status"] == "blocked"
    assert payload["decision"]["failed_checks"] == ["real_machine_data", "operator_approval_evidence"]


def test_hardware_profile_is_stricter_than_shadow_review():
    config = RLShadowGateConfig.from_profile("hardware_actuation")

    assert config.profile == "hardware_actuation"
    assert config.min_recommendation_windows == 1000
    assert config.require_real_machine_data is True
    assert config.require_operator_approval_evidence is True
    assert config.require_hardware_interlock_evidence is True


def test_cli_gate_rl_shadow_returns_zero_when_passed(tmp_path):
    shadow_dir = tmp_path / "shadow"
    out_dir = tmp_path / "gate"
    _write_shadow_metrics(shadow_dir, mean_risk=0.50)

    status = main(["gate-rl-shadow", "--shadow-dir", str(shadow_dir), "--out", str(out_dir), "--profile", "shadow_review"])

    assert status == 0
    assert json.loads((out_dir / "gate_metrics.json").read_text(encoding="utf-8"))["decision"]["status"] == "pass"


def test_cli_gate_rl_shadow_returns_two_when_profile_blocks(tmp_path):
    shadow_dir = tmp_path / "shadow"
    out_dir = tmp_path / "gate"
    _write_shadow_metrics(shadow_dir, mean_risk=0.50)

    status = main(["gate-rl-shadow", "--shadow-dir", str(shadow_dir), "--out", str(out_dir), "--profile", "live_shadow"])

    assert status == 2
    assert json.loads((out_dir / "gate_metrics.json").read_text(encoding="utf-8"))["decision"]["status"] == "blocked"


def _action_trace_rows():
    return [
        {
            "scenario": "stable",
            "episode": 0,
            "step": 0,
            "controller": "td3",
            "risk_now": 0.22,
            "risk_horizon": 0.25,
            "risk_label": "stable",
            "risk_uncertainty": 0.12,
            "current_margin": 0.44,
            "current_uncertainty": 0.12,
            "previous_feed_override": 1.0,
            "previous_spindle_override": 1.0,
            "raw_feed_override": 1.02,
            "raw_spindle_override": 1.01,
            "guard_feed_override": 1.0,
            "guard_spindle_override": 1.0,
            "shield_feed_override": 1.0,
            "shield_spindle_override": 1.0,
            "candidate_guard_enabled": 1,
            "guard_fallback": 0,
            "guard_reasons": "",
            "shield_rejected": 0,
            "shield_reasons": "",
        },
        {
            "scenario": "unstable",
            "episode": 0,
            "step": 1,
            "controller": "td3",
            "risk_now": 0.68,
            "risk_horizon": 0.72,
            "risk_label": "severe",
            "risk_uncertainty": 0.28,
            "current_margin": -0.17,
            "current_uncertainty": 0.28,
            "previous_feed_override": 1.0,
            "previous_spindle_override": 1.0,
            "raw_feed_override": 0.9,
            "raw_spindle_override": 1.1,
            "guard_feed_override": 0.9,
            "guard_spindle_override": 1.1,
            "shield_feed_override": 0.9,
            "shield_spindle_override": 1.1,
            "candidate_guard_enabled": 1,
            "guard_fallback": 1,
            "guard_reasons": "high_uncertainty",
            "shield_rejected": 1,
            "shield_reasons": "high_uncertainty",
        },
    ]


def _write_shadow_metrics(
    shadow_dir: Path,
    *,
    mean_risk: float,
    max_risk: float = 0.74,
    unstable_risk: float = 0.57,
    action_fraction: float = 0.46,
    guard_fallback_fraction: float = 0.10,
    recommendation_windows: int = 160,
    real_machine_data: bool = False,
    operator_approval_evidence: bool = False,
    hardware_interlock_evidence: bool = False,
) -> None:
    shadow_dir.mkdir(parents=True)
    payload = {
        "deployment_boundary": {
            "mode": "shadow_only",
            "cnc_writes_enabled": False,
            "requires_safety_shield": True,
            "requires_human_review": True,
        },
        "validation_context": {
            "data_source": "real_machine" if real_machine_data else "simulated_twin",
            "real_machine_data": real_machine_data,
            "operator_approval_evidence": operator_approval_evidence,
            "hardware_interlock_evidence": hardware_interlock_evidence,
        },
        "window_metrics": {
            "recommendation_windows": recommendation_windows,
            "action_fraction": action_fraction,
            "guard_fallback_fraction": guard_fallback_fraction,
            "guard_fallbacks": 16,
            "shield_rejections": 0,
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "relative_mrr_proxy": 1.13,
        },
        "scenario_metrics": [
            {"scenario": "stable", "mean_risk": 0.49, "max_risk": 0.72, "relative_mrr_proxy": 1.13},
            {"scenario": "unstable", "mean_risk": unstable_risk, "max_risk": 0.71, "relative_mrr_proxy": 1.14},
        ],
    }
    (shadow_dir / "shadow_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
