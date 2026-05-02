import csv
import json
from pathlib import Path

from chatter_twin.cli import main
from chatter_twin.demo import InternalDemoConfig, write_internal_demo_report


def test_write_internal_demo_report_builds_readiness_summary(tmp_path):
    config = _write_demo_fixture(tmp_path)

    payload = write_internal_demo_report(
        out_path=tmp_path / "INTERNAL_DEMO_REPORT.md",
        summary_path=tmp_path / "INTERNAL_DEMO_SUMMARY.json",
        config=config,
    )

    report = (tmp_path / "INTERNAL_DEMO_REPORT.md").read_text(encoding="utf-8")
    summary = json.loads((tmp_path / "INTERNAL_DEMO_SUMMARY.json").read_text(encoding="utf-8"))
    assert payload["readiness"]["status"] == "good_internal_demo_complete"
    assert summary["readiness"]["status"] == "good_internal_demo_complete"
    assert "Software demo complete" in report
    assert "real CNC validation" in report
    assert "`shadow_review` | `pass`" in report
    assert "`hardware_actuation` | `blocked`" in report


def test_cli_internal_demo_report_runs(tmp_path):
    config = _write_demo_fixture(tmp_path)
    out_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.json"

    status = main(
        [
            "internal-demo-report",
            "--out",
            str(out_path),
            "--summary-out",
            str(summary_path),
            "--calibration-dir",
            str(config.calibration_dir),
            "--risk-model-dir",
            str(config.risk_model_dir),
            "--closed-loop-dir",
            str(config.closed_loop_dir),
            "--rl-comparison-dir",
            str(config.rl_comparison_dir),
            "--champion-dir",
            str(config.champion_dir),
            "--rl-shadow-replay-dir",
            str(config.rl_shadow_replay_dir),
            "--shadow-review-gate-dir",
            str(config.shadow_review_gate_dir),
            "--live-shadow-gate-dir",
            str(config.live_shadow_gate_dir),
            "--hardware-gate-dir",
            str(config.hardware_gate_dir),
            "--shadow-eval-dir",
            str(config.shadow_eval_dir),
            "--counterfactual-dir",
            str(config.counterfactual_dir),
            "--test-status",
            "fixture tests passed",
        ]
    )

    assert status == 0
    assert out_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["test_status"] == "fixture tests passed"
    assert payload["promotion_gates"][0]["status"] == "pass"


def _write_demo_fixture(root: Path) -> InternalDemoConfig:
    calibration_dir = root / "calibration"
    risk_model_dir = root / "risk"
    closed_loop_dir = root / "closed_loop"
    rl_comparison_dir = root / "rl_comparison"
    champion_dir = root / "champion"
    replay_dir = root / "replay"
    shadow_review_gate_dir = root / "gate_shadow"
    live_shadow_gate_dir = root / "gate_live"
    hardware_gate_dir = root / "gate_hardware"
    shadow_eval_dir = root / "shadow_eval"
    counterfactual_dir = root / "counterfactual"

    _write_json(
        calibration_dir / "calibration.json",
        {
            "metrics": {
                "sample_count": 100,
                "train": {"roc_auc": 0.82},
                "holdout": {
                    "roc_auc": 0.76,
                    "accuracy_at_0_5": 0.74,
                    "mean_absolute_error_to_time_domain_risk": 0.15,
                },
            }
        },
    )
    _write_json(
        risk_model_dir / "metrics.json",
        {
            "test": {
                "accuracy": 0.84,
                "binary_chatter_f1": 0.93,
                "binary_chatter_precision": 0.98,
                "binary_chatter_recall": 0.88,
            },
            "event_warning": {
                "threshold_selection": {
                    "test_at_selected_threshold": {
                        "f1": 0.73,
                        "precision": 1.0,
                        "recall": 0.57,
                        "mean_detected_lead_time_s": 0.19,
                    }
                }
            },
        },
    )
    _write_csv(
        closed_loop_dir / "summary.csv",
        [
            {"controller": "fixed", "scenario": "stable", "mean_risk": 0.40, "relative_mrr_proxy": 1.0, "shield_rejections": 0},
            {"controller": "fixed", "scenario": "unstable", "mean_risk": 0.55, "relative_mrr_proxy": 1.0, "shield_rejections": 0},
            {"controller": "sld", "scenario": "stable", "mean_risk": 0.34, "relative_mrr_proxy": 0.91, "shield_rejections": 0},
            {"controller": "sld", "scenario": "unstable", "mean_risk": 0.50, "relative_mrr_proxy": 0.96, "shield_rejections": 0},
        ],
    )
    _write_csv(
        rl_comparison_dir / "candidate_summary.csv",
        [
            {
                "profile": "td3_fixed",
                "mean_risk_avg": 0.47,
                "mean_risk_worst": 0.55,
                "unstable_risk": 0.55,
                "relative_mrr_avg": 1.08,
                "guard_fallbacks_sum": 52,
                "is_pareto_frontier": 1,
            },
            {
                "profile": "td3_headroom",
                "mean_risk_avg": 0.48,
                "mean_risk_worst": 0.56,
                "unstable_risk": 0.56,
                "relative_mrr_avg": 1.13,
                "guard_fallbacks_sum": 46,
                "is_pareto_frontier": 1,
            },
        ],
    )
    _write_json(
        champion_dir / "selected_policy.json",
        {
            "selected": {
                "seed": 616,
                "selection_score": 0.99,
                "mean_risk_avg": 0.48,
                "relative_mrr_avg": 1.13,
                "unstable_risk": 0.52,
            }
        },
    )
    _write_json(
        replay_dir / "shadow_metrics.json",
        {
            "deployment_boundary": {"mode": "shadow_only", "cnc_writes_enabled": False},
            "window_metrics": {
                "recommendation_windows": 160,
                "action_fraction": 0.46,
                "mean_risk": 0.51,
                "max_risk": 0.75,
                "relative_mrr_proxy": 1.13,
                "shield_rejections": 0,
            },
        },
    )
    _write_gate(shadow_review_gate_dir, "shadow_review", "pass", "shadow_review_candidate", [])
    _write_gate(live_shadow_gate_dir, "live_shadow", "blocked", "do_not_promote", ["real_machine_data"])
    _write_gate(hardware_gate_dir, "hardware_actuation", "blocked", "do_not_promote", ["hardware_interlock_evidence"])
    _write_json(
        shadow_eval_dir / "shadow_metrics.json",
        {
            "window_metrics": {"warning_fraction": 0.4, "relative_mrr_proxy": 0.98},
            "event_metrics": {"f1": 0.73, "precision": 1.0, "recall": 0.57},
        },
    )
    _write_json(
        counterfactual_dir / "counterfactual_metrics.json",
        {
            "windows": {"mean_risk_reduction": 0.007, "mean_relative_mrr_proxy": 1.0},
            "episodes": {"mitigated_event_episodes": 0, "worsened_event_episodes": 0},
        },
    )
    return InternalDemoConfig(
        calibration_dir=calibration_dir,
        risk_model_dir=risk_model_dir,
        closed_loop_dir=closed_loop_dir,
        rl_comparison_dir=rl_comparison_dir,
        champion_dir=champion_dir,
        rl_shadow_replay_dir=replay_dir,
        shadow_review_gate_dir=shadow_review_gate_dir,
        live_shadow_gate_dir=live_shadow_gate_dir,
        hardware_gate_dir=hardware_gate_dir,
        shadow_eval_dir=shadow_eval_dir,
        counterfactual_dir=counterfactual_dir,
        test_status="fixture tests passed",
    )


def _write_gate(path: Path, profile: str, status: str, promotion_level: str, failed_checks: list[str]) -> None:
    _write_json(
        path / "gate_metrics.json",
        {
            "gate_config": {"profile": profile},
            "decision": {
                "status": status,
                "promotion_level": promotion_level,
                "hardware_actuation_allowed": False,
                "failed_checks": failed_checks,
            },
        },
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
