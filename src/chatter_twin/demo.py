from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InternalDemoConfig:
    calibration_dir: Path = Path("results/margin_calibration_context_family_holdout_demo")
    risk_model_dir: Path = Path("results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo")
    closed_loop_dir: Path = Path("results/closed_loop_benchmark_context_calibrated_randomized_stress_demo")
    rl_comparison_dir: Path = Path("results/rl_td3_harsh_stress_comparison")
    champion_dir: Path = Path("results/rl_td3_fixed_mrr2_shadow_champion")
    rl_shadow_replay_dir: Path = Path("results/rl_td3_fixed_mrr2_shadow_replay_demo")
    shadow_review_gate_dir: Path = Path("results/rl_td3_fixed_mrr2_shadow_gate_demo")
    live_shadow_gate_dir: Path = Path("results/rl_td3_fixed_mrr2_live_shadow_gate_demo")
    hardware_gate_dir: Path = Path("results/rl_td3_fixed_mrr2_hardware_actuation_gate_demo")
    shadow_eval_dir: Path = Path("results/shadow_policy_onset_events_axial_depth_holdout_validated_demo")
    counterfactual_dir: Path = Path("results/shadow_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo")
    test_status: str = ""


def write_internal_demo_report(
    *,
    out_path: Path,
    summary_path: Path | None = None,
    config: InternalDemoConfig = InternalDemoConfig(),
) -> dict[str, Any]:
    payload = build_internal_demo_payload(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_internal_demo_report(payload), encoding="utf-8")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_internal_demo_payload(config: InternalDemoConfig = InternalDemoConfig()) -> dict[str, Any]:
    calibration = _calibration_summary(config.calibration_dir)
    risk_model = _risk_model_summary(config.risk_model_dir)
    baselines = _baseline_summary(config.closed_loop_dir)
    rl = _rl_summary(config.rl_comparison_dir, config.champion_dir)
    shadow_replay = _shadow_replay_summary(config.rl_shadow_replay_dir)
    gates = _gate_summaries(
        {
            "shadow_review": config.shadow_review_gate_dir,
            "live_shadow": config.live_shadow_gate_dir,
            "hardware_actuation": config.hardware_gate_dir,
        }
    )
    shadow_eval = _shadow_eval_summary(config.shadow_eval_dir)
    counterfactual = _counterfactual_summary(config.counterfactual_dir)
    readiness = _readiness_summary(gates)
    payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config": _stringify_paths(asdict(config)),
        "conclusion": {
            "headline": "Software demo complete; real CNC validation is still the decisive missing result.",
            "current_stage": "offline_shadow_review",
            "rl_champion": "td3 seed 616",
            "hardware_ready": False,
        },
        "validation_boundary": {
            "validated_now": [
                "Simulator, risk estimator, controller comparisons, shielded RL replay, and promotion gates run end to end.",
                "The selected TD3 policy passes the offline shadow-review gate.",
                "The same policy is blocked from live-shadow and hardware-actuation profiles.",
            ],
            "not_validated_yet": [
                "No MTConnect or controller API connection is implemented.",
                "No synchronized high-rate accelerometer/audio/current data has been ingested.",
                "No real FRF/cutting-coefficient calibration or real CNC chatter cut has been run through this stack.",
                "No CNC write path exists or is approved.",
            ],
        },
        "calibration": calibration,
        "risk_model": risk_model,
        "controller_baselines": baselines,
        "shadow_risk_policy": shadow_eval,
        "shadow_counterfactual": counterfactual,
        "rl": rl,
        "rl_shadow_replay": shadow_replay,
        "promotion_gates": gates,
        "readiness": readiness,
        "artifacts": _artifact_summary(config),
        "test_status": config.test_status,
    }
    return payload


def render_internal_demo_report(payload: dict[str, Any]) -> str:
    conclusion = payload["conclusion"]
    calibration = payload["calibration"]
    risk = payload["risk_model"]
    baselines = payload["controller_baselines"]
    shadow_eval = payload["shadow_risk_policy"]
    counterfactual = payload["shadow_counterfactual"]
    rl = payload["rl"]
    replay = payload["rl_shadow_replay"]
    gates = payload["promotion_gates"]
    readiness = payload["readiness"]
    lines = [
        "# Chatter Twin Internal Demo Report",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Executive Conclusion",
        "",
        conclusion["headline"],
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Current stage | `{conclusion['current_stage']}` |",
        f"| RL champion | `{conclusion['rl_champion']}` |",
        f"| Hardware ready | `{str(conclusion['hardware_ready']).lower()}` |",
        f"| Test status | `{payload['test_status'] or 'not recorded in report command'}` |",
        "",
        "## What This Demo Proves",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["validation_boundary"]["validated_now"])
    lines.extend(
        [
            "",
            "## What It Does Not Prove Yet",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["validation_boundary"]["not_validated_yet"])
    lines.extend(
        [
            "",
            "## Pipeline Status",
            "",
            "| Layer | Evidence | Status |",
            "|---|---|---|",
            f"| Calibration twin | {calibration['one_line']} | `{calibration['status']}` |",
            f"| Risk estimator | {risk['one_line']} | `{risk['status']}` |",
            f"| Classical/controller baselines | {baselines['one_line']} | `{baselines['status']}` |",
            f"| Offline shadow policy | {shadow_eval['one_line']} | `{shadow_eval['status']}` |",
            f"| Shadow counterfactual | {counterfactual['one_line']} | `{counterfactual['status']}` |",
            f"| RL controller candidate | {rl['one_line']} | `{rl['status']}` |",
            f"| RL shadow replay | {replay['one_line']} | `{replay['status']}` |",
            f"| Promotion gates | {readiness['one_line']} | `{readiness['status']}` |",
            "",
            "## Calibration",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Samples | {_fmt(calibration['sample_count'])} |",
            f"| Train ROC AUC | {_fmt(calibration['train_roc_auc'])} |",
            f"| Holdout ROC AUC | {_fmt(calibration['holdout_roc_auc'])} |",
            f"| Holdout accuracy @ 0.5 | {_fmt(calibration['holdout_accuracy'])} |",
            f"| Holdout MAE to time-domain risk | {_fmt(calibration['holdout_mae'])} |",
            "",
            "## Risk Estimator",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Test accuracy | {_fmt(risk['test_accuracy'])} |",
            f"| Binary chatter F1 | {_fmt(risk['binary_chatter_f1'])} |",
            f"| Binary chatter precision | {_fmt(risk['binary_chatter_precision'])} |",
            f"| Binary chatter recall | {_fmt(risk['binary_chatter_recall'])} |",
            f"| Event warning F1 | {_fmt(risk['event_f1'])} |",
            f"| Event precision | {_fmt(risk['event_precision'])} |",
            f"| Event recall | {_fmt(risk['event_recall'])} |",
            f"| Mean detected lead time, s | {_fmt(risk['event_mean_detected_lead_time_s'])} |",
            "",
            "## Controller Baselines",
            "",
            "| Controller | Avg risk | Worst risk | Avg relative MRR | Shield rejections |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in baselines["controllers"]:
        lines.append(
            f"| `{row['controller']}` | {_fmt(row['mean_risk_avg'])} | {_fmt(row['mean_risk_worst'])} | "
            f"{_fmt(row['relative_mrr_avg'])} | {_fmt(row['shield_rejections_sum'])} |"
        )
    lines.extend(
        [
            "",
            f"Best baseline by average risk: `{baselines['best_by_risk']['controller']}` "
            f"({_fmt(baselines['best_by_risk']['mean_risk_avg'])}).",
            "",
            "## RL Candidate And Shadow Replay",
            "",
            "| Item | Value |",
            "|---|---:|",
            f"| Selected seed | {rl['selected_seed']} |",
            f"| Selected policy score | {_fmt(rl['selection_score'])} |",
            f"| Champion average risk | {_fmt(rl['selected_mean_risk'])} |",
            f"| Champion average relative MRR | {_fmt(rl['selected_relative_mrr'])} |",
            f"| Shadow replay windows | {_fmt(replay['recommendation_windows'])} |",
            f"| Shadow action fraction | {_fmt(replay['action_fraction'])} |",
            f"| Shadow mean risk | {_fmt(replay['mean_risk'])} |",
            f"| Shadow max risk | {_fmt(replay['max_risk'])} |",
            f"| Shadow relative MRR proxy | {_fmt(replay['relative_mrr_proxy'])} |",
            f"| Shadow shield rejections | {_fmt(replay['shield_rejections'])} |",
            "",
            "RL stress candidates:",
            "",
            "| Profile | Avg risk | Worst risk | Unstable risk | Avg relative MRR | Guard fallbacks | Pareto |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rl["stress_candidates"]:
        lines.append(
            f"| `{row['profile']}` | {_fmt(row['mean_risk_avg'])} | {_fmt(row['mean_risk_worst'])} | "
            f"{_fmt(row['unstable_risk'])} | {_fmt(row['relative_mrr_avg'])} | "
            f"{_fmt(row['guard_fallbacks_sum'])} | {row['is_pareto_frontier']} |"
        )
    lines.extend(
        [
            "",
            "## Promotion Gates",
            "",
            "| Profile | Status | Promotion level | Failed checks |",
            "|---|---|---|---|",
        ]
    )
    for row in gates:
        failed = ", ".join(row["failed_checks"]) if row["failed_checks"] else "none"
        lines.append(f"| `{row['profile']}` | `{row['status']}` | `{row['promotion_level']}` | {failed} |")
    lines.extend(
        [
            "",
            "## Demo Runbook",
            "",
            "1. Re-run tests: `rtk uv run pytest -q`.",
            "2. Regenerate the RL shadow replay: `rtk uv run --extra rl chatter-twin shadow-rl-policy ...` using the command in `docs/RL_CONTROLLERS.md`.",
            "3. Re-run gates: `rtk uv run chatter-twin gate-rl-shadow --profile shadow_review ...`, then repeat for `live_shadow` and `hardware_actuation`.",
            "4. Regenerate this report: "
            f"`rtk uv run chatter-twin internal-demo-report --out docs/INTERNAL_DEMO_REPORT.md --summary-out docs/INTERNAL_DEMO_SUMMARY.json --test-status \"{payload['test_status'] or 'latest test result'}\"`.",
            "",
            "## Artifact Map",
            "",
            "| Artifact | Path |",
            "|---|---|",
        ]
    )
    for name, path in payload["artifacts"].items():
        lines.append(f"| {name} | `{path}` |")
    lines.extend(
        [
            "",
            "## Next Result Barrier",
            "",
            "The good internal demo is complete when judged as an offline software demo. "
            "The next result barrier is real-machine validation: synchronized CNC context plus high-rate sensor data, FRF/cutting calibration, and replay through the same estimator, controllers, and gate profiles.",
            "",
        ]
    )
    return "\n".join(lines)


def _calibration_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "calibration.json")
    metrics = payload.get("metrics", {})
    train = metrics.get("train", {})
    holdout = metrics.get("holdout", {})
    sample_count = int(metrics.get("sample_count", 0))
    holdout_auc = _float_or_none(holdout.get("roc_auc"))
    status = "validated_synthetic_holdout" if sample_count and holdout_auc is not None else "missing"
    return {
        "path": str(path),
        "status": status,
        "one_line": f"{sample_count} synthetic calibration samples, holdout ROC AUC {_fmt(holdout_auc)}",
        "sample_count": sample_count,
        "train_roc_auc": _float_or_none(train.get("roc_auc")),
        "holdout_roc_auc": holdout_auc,
        "holdout_accuracy": _float_or_none(holdout.get("accuracy_at_0_5")),
        "holdout_mae": _float_or_none(holdout.get("mean_absolute_error_to_time_domain_risk")),
    }


def _risk_model_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "metrics.json")
    test = payload.get("test", {})
    event = payload.get("event_warning", {}).get("threshold_selection", {}).get("test_at_selected_threshold", {})
    status = "validated_synthetic_holdout" if test else "missing"
    event_f1 = _float_or_none(event.get("f1"))
    return {
        "path": str(path),
        "status": status,
        "one_line": f"test chatter F1 {_fmt(test.get('binary_chatter_f1'))}, event F1 {_fmt(event_f1)}",
        "test_accuracy": _float_or_none(test.get("accuracy")),
        "binary_chatter_f1": _float_or_none(test.get("binary_chatter_f1")),
        "binary_chatter_precision": _float_or_none(test.get("binary_chatter_precision")),
        "binary_chatter_recall": _float_or_none(test.get("binary_chatter_recall")),
        "event_f1": event_f1,
        "event_precision": _float_or_none(event.get("precision")),
        "event_recall": _float_or_none(event.get("recall")),
        "event_mean_detected_lead_time_s": _float_or_none(event.get("mean_detected_lead_time_s")),
    }


def _baseline_summary(path: Path) -> dict[str, Any]:
    rows = _read_csv(path / "summary.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["controller"], []).append(row)
    controllers = []
    for controller, group in sorted(grouped.items()):
        controllers.append(
            {
                "controller": controller,
                "scenario_count": len(group),
                "mean_risk_avg": _mean(_float(row.get("mean_risk")) for row in group),
                "mean_risk_worst": max(_float(row.get("mean_risk")) for row in group),
                "relative_mrr_avg": _mean(_float(row.get("relative_mrr_proxy")) for row in group),
                "shield_rejections_sum": sum(_float(row.get("shield_rejections")) for row in group),
            }
        )
    best_by_risk = min(controllers, key=lambda row: row["mean_risk_avg"]) if controllers else {}
    status = "validated_synthetic_benchmark" if controllers else "missing"
    return {
        "path": str(path),
        "status": status,
        "one_line": f"{len(controllers)} controllers compared; best risk `{best_by_risk.get('controller', 'missing')}`",
        "controllers": controllers,
        "best_by_risk": best_by_risk,
    }


def _rl_summary(comparison_dir: Path, champion_dir: Path) -> dict[str, Any]:
    candidates = _read_csv(comparison_dir / "candidate_summary.csv")
    stress_candidates = [
        {
            "profile": row["profile"],
            "mean_risk_avg": _float(row.get("mean_risk_avg")),
            "mean_risk_worst": _float(row.get("mean_risk_worst")),
            "unstable_risk": _float(row.get("unstable_risk")),
            "relative_mrr_avg": _float(row.get("relative_mrr_avg")),
            "guard_fallbacks_sum": _float(row.get("guard_fallbacks_sum")),
            "is_pareto_frontier": "yes" if _float(row.get("is_pareto_frontier")) else "no",
        }
        for row in candidates
    ]
    champion = _read_json(champion_dir / "selected_policy.json").get("selected", {})
    selected_seed = champion.get("seed", "")
    status = "shadow_champion_selected" if champion else "missing"
    return {
        "comparison_path": str(comparison_dir),
        "champion_path": str(champion_dir),
        "status": status,
        "one_line": f"TD3 seed {selected_seed} selected; avg MRR {_fmt(champion.get('relative_mrr_avg'))}",
        "selected_seed": selected_seed,
        "selection_score": _float_or_none(champion.get("selection_score")),
        "selected_mean_risk": _float_or_none(champion.get("mean_risk_avg")),
        "selected_relative_mrr": _float_or_none(champion.get("relative_mrr_avg")),
        "selected_unstable_risk": _float_or_none(champion.get("unstable_risk")),
        "stress_candidates": stress_candidates,
    }


def _shadow_replay_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "shadow_metrics.json")
    window = payload.get("window_metrics", {})
    boundary = payload.get("deployment_boundary", {})
    status = "shadow_only_replay" if boundary.get("mode") == "shadow_only" else "missing"
    return {
        "path": str(path),
        "status": status,
        "one_line": f"{window.get('recommendation_windows', 0)} windows, MRR {_fmt(window.get('relative_mrr_proxy'))}, shield rejects {_fmt(window.get('shield_rejections'))}",
        "recommendation_windows": _float_or_none(window.get("recommendation_windows")),
        "action_fraction": _float_or_none(window.get("action_fraction")),
        "mean_risk": _float_or_none(window.get("mean_risk")),
        "max_risk": _float_or_none(window.get("max_risk")),
        "relative_mrr_proxy": _float_or_none(window.get("relative_mrr_proxy")),
        "shield_rejections": _float_or_none(window.get("shield_rejections")),
    }


def _gate_summaries(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for profile, path in paths.items():
        payload = _read_json(path / "gate_metrics.json")
        decision = payload.get("decision", {})
        rows.append(
            {
                "profile": profile,
                "path": str(path),
                "status": decision.get("status", "missing"),
                "promotion_level": decision.get("promotion_level", "missing"),
                "hardware_actuation_allowed": bool(decision.get("hardware_actuation_allowed", False)),
                "failed_checks": list(decision.get("failed_checks", [])),
            }
        )
    return rows


def _shadow_eval_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "shadow_metrics.json")
    windows = payload.get("window_metrics", {})
    event = payload.get("event_metrics", {})
    status = "shadow_policy_evaluated" if windows else "missing"
    return {
        "path": str(path),
        "status": status,
        "one_line": f"event F1 {_fmt(event.get('f1'))}, warning fraction {_fmt(windows.get('warning_fraction'))}",
        "event_f1": _float_or_none(event.get("f1")),
        "event_precision": _float_or_none(event.get("precision")),
        "event_recall": _float_or_none(event.get("recall")),
        "warning_fraction": _float_or_none(windows.get("warning_fraction")),
        "relative_mrr_proxy": _float_or_none(windows.get("relative_mrr_proxy")),
    }


def _counterfactual_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path / "counterfactual_metrics.json")
    windows = payload.get("windows", {})
    episodes = payload.get("episodes", {})
    status = "counterfactual_replayed" if windows else "missing"
    return {
        "path": str(path),
        "status": status,
        "one_line": f"mean risk reduction {_fmt(windows.get('mean_risk_reduction'))}, mitigated events {_fmt(episodes.get('mitigated_event_episodes'))}",
        "mean_risk_reduction": _float_or_none(windows.get("mean_risk_reduction")),
        "mean_relative_mrr_proxy": _float_or_none(windows.get("mean_relative_mrr_proxy")),
        "mitigated_event_episodes": _float_or_none(episodes.get("mitigated_event_episodes")),
        "worsened_event_episodes": _float_or_none(episodes.get("worsened_event_episodes")),
    }


def _readiness_summary(gates: list[dict[str, Any]]) -> dict[str, Any]:
    status_by_profile = {row["profile"]: row["status"] for row in gates}
    shadow_passed = status_by_profile.get("shadow_review") == "pass"
    live_blocked = status_by_profile.get("live_shadow") == "blocked"
    hardware_blocked = status_by_profile.get("hardware_actuation") == "blocked"
    status = "good_internal_demo_complete" if shadow_passed and live_blocked and hardware_blocked else "needs_attention"
    return {
        "status": status,
        "one_line": "offline shadow review passes; live and hardware gates block as intended"
        if status == "good_internal_demo_complete"
        else "gate profile outcomes need review",
    }


def _artifact_summary(config: InternalDemoConfig) -> dict[str, str]:
    return {
        "Calibration": str(config.calibration_dir),
        "Risk model": str(config.risk_model_dir),
        "Closed-loop benchmark": str(config.closed_loop_dir),
        "Risk-model shadow policy": str(config.shadow_eval_dir),
        "Shadow counterfactual": str(config.counterfactual_dir),
        "RL stress comparison": str(config.rl_comparison_dir),
        "RL champion": str(config.champion_dir),
        "RL shadow replay": str(config.rl_shadow_replay_dir),
        "Shadow-review gate": str(config.shadow_review_gate_dir),
        "Live-shadow gate": str(config.live_shadow_gate_dir),
        "Hardware-actuation gate": str(config.hardware_gate_dir),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    if isinstance(value, str):
        return value
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _stringify_paths(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
