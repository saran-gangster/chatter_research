from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chatter_twin.calibration import MarginCalibration
from chatter_twin.models import RewardConfig, ShieldConfig
from chatter_twin.replay import DomainRandomizationConfig
from chatter_twin.rl import Sb3TrainingConfig, trace_sb3_policy_actions


@dataclass(frozen=True)
class RLShadowReplayConfig:
    scenarios: tuple[str, ...] = ("stable", "near_boundary", "onset", "unstable")
    episodes: int = 4
    steps: int = 10
    decision_interval_s: float = 0.08
    seed: int = 3_000_000

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be positive")


@dataclass(frozen=True)
class RLShadowGateConfig:
    profile: str = "shadow_review"
    min_recommendation_windows: int = 1
    max_mean_risk: float = 0.55
    max_max_risk: float = 0.80
    max_unstable_mean_risk: float = 0.60
    min_relative_mrr: float = 1.0
    max_guard_fallback_fraction: float = 0.20
    max_action_fraction: float = 0.80
    max_shield_rejections: int = 0
    require_shadow_only: bool = True
    require_no_cnc_writes: bool = True
    require_real_machine_data: bool = False
    require_operator_approval_evidence: bool = False
    require_hardware_interlock_evidence: bool = False

    def __post_init__(self) -> None:
        if self.profile not in GATE_PROFILE_CHOICES:
            raise ValueError(f"profile must be one of {', '.join(GATE_PROFILE_CHOICES)}")
        if self.min_recommendation_windows < 1:
            raise ValueError("min_recommendation_windows must be at least 1")
        for name in (
            "max_mean_risk",
            "max_max_risk",
            "max_unstable_mean_risk",
            "min_relative_mrr",
            "max_guard_fallback_fraction",
            "max_action_fraction",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_shield_rejections < 0:
            raise ValueError("max_shield_rejections cannot be negative")

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "RLShadowGateConfig":
        if profile not in GATE_PROFILE_DEFAULTS:
            raise ValueError(f"profile must be one of {', '.join(GATE_PROFILE_CHOICES)}")
        values = dict(GATE_PROFILE_DEFAULTS[profile])
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


GATE_PROFILE_CHOICES = ("shadow_review", "live_shadow", "hardware_actuation")
GATE_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "shadow_review": {
        "profile": "shadow_review",
        "min_recommendation_windows": 1,
        "max_mean_risk": 0.55,
        "max_max_risk": 0.80,
        "max_unstable_mean_risk": 0.60,
        "min_relative_mrr": 1.0,
        "max_guard_fallback_fraction": 0.20,
        "max_action_fraction": 0.80,
        "max_shield_rejections": 0,
        "require_shadow_only": True,
        "require_no_cnc_writes": True,
        "require_real_machine_data": False,
        "require_operator_approval_evidence": False,
        "require_hardware_interlock_evidence": False,
    },
    "live_shadow": {
        "profile": "live_shadow",
        "min_recommendation_windows": 300,
        "max_mean_risk": 0.45,
        "max_max_risk": 0.70,
        "max_unstable_mean_risk": 0.50,
        "min_relative_mrr": 1.0,
        "max_guard_fallback_fraction": 0.10,
        "max_action_fraction": 0.65,
        "max_shield_rejections": 0,
        "require_shadow_only": True,
        "require_no_cnc_writes": True,
        "require_real_machine_data": True,
        "require_operator_approval_evidence": True,
        "require_hardware_interlock_evidence": False,
    },
    "hardware_actuation": {
        "profile": "hardware_actuation",
        "min_recommendation_windows": 1_000,
        "max_mean_risk": 0.30,
        "max_max_risk": 0.50,
        "max_unstable_mean_risk": 0.35,
        "min_relative_mrr": 1.0,
        "max_guard_fallback_fraction": 0.02,
        "max_action_fraction": 0.50,
        "max_shield_rejections": 0,
        "require_shadow_only": True,
        "require_no_cnc_writes": True,
        "require_real_machine_data": True,
        "require_operator_approval_evidence": True,
        "require_hardware_interlock_evidence": True,
    },
}
PROMOTION_LEVEL_BY_PROFILE = {
    "shadow_review": "shadow_review_candidate",
    "live_shadow": "live_shadow_candidate",
    "hardware_actuation": "hardware_actuation_review_candidate",
}


def run_selected_rl_shadow_policy(
    *,
    selection_path: Path,
    out_dir: Path,
    config: RLShadowReplayConfig = RLShadowReplayConfig(),
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    selection = _load_selection(selection_path)
    selected = selection["selected"]
    model_path = Path(str(selected["source_model_path"]))
    if not model_path.exists():
        raise ValueError(f"Selected model does not exist: {model_path}")

    source_run = Path(str(selected["source_run"]))
    source_metrics = _read_json(source_run / "metrics.json")
    algorithm = str(selected.get("algorithm", source_metrics.get("algorithm", "")))
    model = _load_sb3_model(algorithm, model_path)
    replay_config = _sb3_replay_config_from_metrics(
        source_metrics,
        algorithm=algorithm,
        replay=config,
        seed=config.seed + int(selected["seed"]),
    )

    action_rows = trace_sb3_policy_actions(
        model=model,
        controller_name=algorithm,
        scenarios=config.scenarios,
        episodes=config.episodes,
        steps=config.steps,
        decision_interval_s=config.decision_interval_s,
        seed=replay_config.seed + 1_000_000,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=replay_config.reward_config,
        shield_config=replay_config.shield_config,
        candidate_guard=replay_config.candidate_guard,
        action_mode=replay_config.action_mode,
        delta_action_scale=replay_config.delta_action_scale,
        delta_feed_scale=replay_config.delta_feed_scale,
        delta_spindle_scale=replay_config.delta_spindle_scale,
        delta_mapping=replay_config.delta_mapping,
    )
    recommendations = recommendations_from_action_trace(
        action_rows,
        source_model_path=model_path,
        profile_label=str(selection.get("profile_label", "")),
    )
    payload = summarize_rl_shadow_recommendations(
        recommendations,
        selection=selection,
        replay_config=config,
        sb3_config=replay_config,
        randomization=randomization,
        selection_path=selection_path,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "recommendations.csv", recommendations)
    _write_csv(out_dir / "action_trace.csv", action_rows)
    _write_json(out_dir / "shadow_metrics.json", payload)
    _write_report(out_dir / "report.md", payload)
    return payload


def gate_rl_shadow_replay(
    *,
    shadow_dir: Path,
    out_dir: Path,
    config: RLShadowGateConfig = RLShadowGateConfig(),
) -> dict[str, Any]:
    metrics_path = shadow_dir / "shadow_metrics.json" if shadow_dir.is_dir() else shadow_dir
    metrics = _read_json(metrics_path)
    checks = rl_shadow_gate_checks(metrics, config)
    failed_checks = [row for row in checks if not bool(row["passed"])]
    passed = not failed_checks
    payload = {
        "shadow_metrics_path": str(metrics_path),
        "gate_config": asdict(config),
        "decision": {
            "passed": passed,
            "status": "pass" if passed else "blocked",
            "promotion_level": PROMOTION_LEVEL_BY_PROFILE[config.profile] if passed else "do_not_promote",
            "hardware_actuation_allowed": False,
            "failed_checks": [row["check"] for row in failed_checks],
            "recommendation": _gate_recommendation(config.profile, passed),
        },
        "checks": checks,
        "source_shadow_metrics": metrics,
        "artifacts": ["gate_checks.csv", "gate_metrics.json", "gate_report.md"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "gate_checks.csv", checks)
    _write_json(out_dir / "gate_metrics.json", payload)
    _write_gate_report(out_dir / "gate_report.md", payload)
    return payload


def rl_shadow_gate_checks(metrics: dict[str, Any], config: RLShadowGateConfig) -> list[dict[str, Any]]:
    window_metrics = metrics.get("window_metrics", {})
    boundary = metrics.get("deployment_boundary", {})
    validation_context = metrics.get("validation_context", {})
    unstable_metrics = _scenario_row(metrics, "unstable")
    checks = [
        _gte_check(
            "minimum_recommendation_windows",
            "recommendation_windows",
            _metric(window_metrics, "recommendation_windows"),
            float(config.min_recommendation_windows),
        ),
        _lte_check("mean_risk", "mean_risk", _metric(window_metrics, "mean_risk"), config.max_mean_risk),
        _lte_check("max_risk", "max_risk", _metric(window_metrics, "max_risk"), config.max_max_risk),
        _lte_check(
            "unstable_mean_risk",
            "unstable.mean_risk",
            _metric(unstable_metrics, "mean_risk"),
            config.max_unstable_mean_risk,
        ),
        _gte_check(
            "relative_mrr_proxy",
            "relative_mrr_proxy",
            _metric(window_metrics, "relative_mrr_proxy"),
            config.min_relative_mrr,
        ),
        _lte_check(
            "guard_fallback_fraction",
            "guard_fallback_fraction",
            _metric(window_metrics, "guard_fallback_fraction"),
            config.max_guard_fallback_fraction,
        ),
        _lte_check(
            "action_fraction",
            "action_fraction",
            _metric(window_metrics, "action_fraction"),
            config.max_action_fraction,
        ),
        _lte_check(
            "shield_rejections",
            "shield_rejections",
            _metric(window_metrics, "shield_rejections"),
            float(config.max_shield_rejections),
        ),
    ]
    if config.require_shadow_only:
        checks.append(
            _eq_check(
                "deployment_mode_shadow_only",
                "deployment_boundary.mode",
                str(boundary.get("mode", "")),
                "shadow_only",
            )
        )
    if config.require_no_cnc_writes:
        checks.append(
            _eq_check(
                "cnc_writes_disabled",
                "deployment_boundary.cnc_writes_enabled",
                bool(boundary.get("cnc_writes_enabled", True)),
                False,
            )
        )
    if config.require_real_machine_data:
        checks.append(
            _eq_check(
                "real_machine_data",
                "validation_context.real_machine_data",
                bool(validation_context.get("real_machine_data", False)),
                True,
            )
        )
    if config.require_operator_approval_evidence:
        checks.append(
            _eq_check(
                "operator_approval_evidence",
                "validation_context.operator_approval_evidence",
                bool(validation_context.get("operator_approval_evidence", False)),
                True,
            )
        )
    if config.require_hardware_interlock_evidence:
        checks.append(
            _eq_check(
                "hardware_interlock_evidence",
                "validation_context.hardware_interlock_evidence",
                bool(validation_context.get("hardware_interlock_evidence", False)),
                True,
            )
        )
    return checks


def recommendations_from_action_trace(
    rows: list[dict[str, Any]],
    *,
    source_model_path: Path,
    profile_label: str,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for row in rows:
        feed = float(row["shield_feed_override"])
        spindle = float(row["shield_spindle_override"])
        previous_feed = float(row["previous_feed_override"])
        previous_spindle = float(row["previous_spindle_override"])
        action_active = abs(feed - previous_feed) > 1.0e-9 or abs(spindle - previous_spindle) > 1.0e-9
        recommendations.append(
            {
                "profile_label": profile_label,
                "source_model_path": str(source_model_path),
                "scenario": row["scenario"],
                "episode": row["episode"],
                "step": row["step"],
                "controller": row["controller"],
                "risk_chatter_now": row["risk_now"],
                "risk_chatter_horizon": row["risk_horizon"],
                "risk_label": row["risk_label"],
                "risk_uncertainty": row["risk_uncertainty"],
                "current_margin": row["current_margin"],
                "current_uncertainty": row["current_uncertainty"],
                "previous_feed_override": previous_feed,
                "previous_spindle_override": previous_spindle,
                "raw_feed_override": row["raw_feed_override"],
                "raw_spindle_override": row["raw_spindle_override"],
                "guard_feed_override": row["guard_feed_override"],
                "guard_spindle_override": row["guard_spindle_override"],
                "feed_override": feed,
                "spindle_override": spindle,
                "relative_mrr_proxy": feed * spindle,
                "action_active": action_active,
                "shadow_warning": action_active,
                "candidate_guard_enabled": row["candidate_guard_enabled"],
                "guard_fallback": row["guard_fallback"],
                "guard_reasons": row["guard_reasons"],
                "shield_rejected": row["shield_rejected"],
                "shield_reasons": row["shield_reasons"],
                "cnc_write_enabled": False,
                "shadow_only": True,
            }
        )
    return recommendations


def summarize_rl_shadow_recommendations(
    recommendations: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    replay_config: RLShadowReplayConfig,
    sb3_config: Sb3TrainingConfig,
    randomization: DomainRandomizationConfig | None,
    selection_path: Path,
) -> dict[str, Any]:
    total = len(recommendations)
    action_windows = sum(_bool(row["action_active"]) for row in recommendations)
    guard_fallbacks = sum(int(row["guard_fallback"]) for row in recommendations)
    shield_rejections = sum(int(row["shield_rejected"]) for row in recommendations)
    payload = {
        "selection_path": str(selection_path),
        "selected_policy": selection["selected"],
        "deployment_boundary": {
            "mode": "shadow_only",
            "cnc_writes_enabled": False,
            "requires_safety_shield": True,
            "requires_human_review": True,
        },
        "validation_context": {
            "data_source": "simulated_twin",
            "real_machine_data": False,
            "operator_approval_evidence": False,
            "hardware_interlock_evidence": False,
        },
        "replay_config": asdict(replay_config),
        "policy_config": {
            "algorithm": sb3_config.algorithm,
            "action_mode": sb3_config.action_mode,
            "delta_action_scale": sb3_config.delta_action_scale,
            "delta_feed_scale": sb3_config.delta_feed_scale,
            "delta_spindle_scale": sb3_config.delta_spindle_scale,
            "delta_mapping": sb3_config.delta_mapping,
            "candidate_guard": sb3_config.candidate_guard,
            "reward_config": asdict(sb3_config.reward_config),
            "shield_config": asdict(sb3_config.shield_config),
        },
        "randomization": asdict(randomization or DomainRandomizationConfig()),
        "window_metrics": {
            "recommendation_windows": total,
            "episodes": len({(row["scenario"], row["episode"]) for row in recommendations}),
            "action_windows": action_windows,
            "action_fraction": action_windows / max(total, 1),
            "guard_fallbacks": guard_fallbacks,
            "guard_fallback_fraction": guard_fallbacks / max(total, 1),
            "shield_rejections": shield_rejections,
            "mean_risk": _mean(row["risk_chatter_now"] for row in recommendations),
            "mean_horizon_risk": _mean(row["risk_chatter_horizon"] for row in recommendations),
            "max_risk": max((float(row["risk_chatter_now"]) for row in recommendations), default=0.0),
            "mean_feed_override": _mean(row["feed_override"] for row in recommendations),
            "mean_spindle_override": _mean(row["spindle_override"] for row in recommendations),
            "relative_mrr_proxy": _mean(row["relative_mrr_proxy"] for row in recommendations),
        },
        "scenario_metrics": _scenario_metrics(recommendations),
        "artifacts": ["recommendations.csv", "action_trace.csv", "shadow_metrics.json", "report.md"],
    }
    return payload


def _scenario_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scenario"]), []).append(row)
    metrics: list[dict[str, Any]] = []
    for scenario, group in sorted(groups.items()):
        metrics.append(
            {
                "scenario": scenario,
                "windows": len(group),
                "action_fraction": sum(_bool(row["action_active"]) for row in group) / max(len(group), 1),
                "guard_fallbacks": sum(int(row["guard_fallback"]) for row in group),
                "shield_rejections": sum(int(row["shield_rejected"]) for row in group),
                "mean_risk": _mean(row["risk_chatter_now"] for row in group),
                "max_risk": max(float(row["risk_chatter_now"]) for row in group),
                "mean_feed_override": _mean(row["feed_override"] for row in group),
                "mean_spindle_override": _mean(row["spindle_override"] for row in group),
                "relative_mrr_proxy": _mean(row["relative_mrr_proxy"] for row in group),
            }
        )
    return metrics


def _load_selection(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "selected_policy.json"
    payload = _read_json(path)
    if "selected" not in payload:
        raise ValueError(f"{path} does not contain a selected policy payload")
    return payload


def _load_sb3_model(algorithm: str, model_path: Path):
    try:
        from stable_baselines3 import SAC, TD3
    except ImportError as exc:  # pragma: no cover - optional dependency guard.
        raise RuntimeError("RL shadow replay requires optional RL dependencies. Run with `rtk uv run --extra rl ...`.") from exc
    if algorithm == "sac":
        return SAC.load(str(model_path), device="cpu")
    if algorithm == "td3":
        return TD3.load(str(model_path), device="cpu")
    raise ValueError(f"Unsupported SB3 algorithm {algorithm!r}")


def _sb3_replay_config_from_metrics(
    payload: dict[str, Any],
    *,
    algorithm: str,
    replay: RLShadowReplayConfig,
    seed: int,
) -> Sb3TrainingConfig:
    config = payload.get("config", {})
    return Sb3TrainingConfig(
        algorithm=algorithm,
        scenarios=replay.scenarios,
        total_timesteps=max(1, int(config.get("total_timesteps", 1))),
        eval_episodes=replay.episodes,
        steps=replay.steps,
        decision_interval_s=replay.decision_interval_s,
        seed=seed,
        learning_rate=float(config.get("learning_rate", 3.0e-4)),
        buffer_size=max(1, int(config.get("buffer_size", 1))),
        learning_starts=max(0, int(config.get("learning_starts", 0))),
        batch_size=max(1, int(config.get("batch_size", 1))),
        gamma=float(config.get("gamma", 0.92)),
        train_freq=max(1, int(config.get("train_freq", 1))),
        gradient_steps=max(0, int(config.get("gradient_steps", 0))),
        baseline_controllers=tuple(config.get("baseline_controllers", ("sld", "mpc"))),
        candidate_guard=bool(config.get("candidate_guard", True)),
        reward_config=RewardConfig(**config.get("reward_config", {})),
        shield_config=ShieldConfig(**config.get("shield_config", {})),
        action_mode=str(config.get("action_mode", "absolute")),
        delta_action_scale=float(config.get("delta_action_scale", 1.0)),
        delta_feed_scale=_optional_float(config.get("delta_feed_scale")),
        delta_spindle_scale=_optional_float(config.get("delta_spindle_scale")),
        delta_mapping=str(config.get("delta_mapping", "fixed")),
    )


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_policy"]
    window_metrics = payload["window_metrics"]
    lines = [
        "# RL Shadow Recommendation Report",
        "",
        f"Selected seed: `{selected['seed']}`",
        f"Source model: `{selected['source_model_path']}`",
        "",
        "## Deployment Boundary",
        "",
        "| Boundary | Value |",
        "|---|---|",
        "| Mode | `shadow_only` |",
        "| CNC writes enabled | `false` |",
        "| Safety shield required | `true` |",
        "| Human review required | `true` |",
        "",
        "## Window Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Recommendation windows | {window_metrics['recommendation_windows']} |",
        f"| Action fraction | {window_metrics['action_fraction']:.3f} |",
        f"| Guard fallback fraction | {window_metrics['guard_fallback_fraction']:.3f} |",
        f"| Shield rejections | {window_metrics['shield_rejections']} |",
        f"| Mean risk | {window_metrics['mean_risk']:.3f} |",
        f"| Max risk | {window_metrics['max_risk']:.3f} |",
        f"| Mean feed override | {window_metrics['mean_feed_override']:.3f} |",
        f"| Mean spindle override | {window_metrics['mean_spindle_override']:.3f} |",
        f"| Relative MRR proxy | {window_metrics['relative_mrr_proxy']:.3f} |",
        "",
        "## Scenario Metrics",
        "",
        "| Scenario | Windows | Mean risk | Max risk | Relative MRR | Guard fallbacks | Shield rejections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["scenario_metrics"]:
        lines.append(
            "| "
            f"{row['scenario']} | {row['windows']} | {row['mean_risk']:.3f} | {row['max_risk']:.3f} | "
            f"{row['relative_mrr_proxy']:.3f} | {row['guard_fallbacks']} | {row['shield_rejections']} |"
        )
    lines.extend(
        [
            "",
            "This is a shadow-only recommendation report. No CNC command is issued.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_gate_report(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    gate_config = payload["gate_config"]
    lines = [
        "# RL Shadow Gate Report",
        "",
        f"Profile: `{gate_config['profile']}`",
        f"Status: `{decision['status']}`",
        f"Promotion level: `{decision['promotion_level']}`",
        f"Hardware actuation allowed: `{str(decision['hardware_actuation_allowed']).lower()}`",
        "",
        decision["recommendation"],
        "",
        "## Checks",
        "",
        "| Check | Metric | Comparator | Actual | Threshold | Result |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in payload["checks"]:
        actual = _format_gate_value(row["actual"])
        threshold = _format_gate_value(row["threshold"])
        result = "pass" if row["passed"] else "fail"
        lines.append(
            "| "
            f"{row['check']} | {row['metric']} | {row['comparator']} | {actual} | {threshold} | {result} |"
        )
    lines.extend(["", "Passing this gate does not authorize CNC writes.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def _gate_recommendation(profile: str, passed: bool) -> str:
    if not passed:
        return "Do not promote this policy; inspect failed gate checks before further replay."
    if profile == "shadow_review":
        return "Eligible for human-reviewed shadow monitoring only; CNC writes remain disabled."
    if profile == "live_shadow":
        return "Eligible for live shadow monitoring only; CNC writes remain disabled."
    return "Eligible for hardware-actuation review only; independent safety approval is still required."


def _scenario_row(metrics: dict[str, Any], scenario: str) -> dict[str, Any]:
    for row in metrics.get("scenario_metrics", []):
        if str(row.get("scenario", "")) == scenario:
            return dict(row)
    return {}


def _metric(container: dict[str, Any], key: str) -> float | None:
    value = container.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def _lte_check(check: str, metric: str, actual: float | None, threshold: float) -> dict[str, Any]:
    return _gate_row(check, metric, actual, "<=", threshold, actual is not None and actual <= threshold)


def _gte_check(check: str, metric: str, actual: float | None, threshold: float) -> dict[str, Any]:
    return _gate_row(check, metric, actual, ">=", threshold, actual is not None and actual >= threshold)


def _eq_check(check: str, metric: str, actual: Any, threshold: Any) -> dict[str, Any]:
    return _gate_row(check, metric, actual, "==", threshold, actual == threshold)


def _gate_row(check: str, metric: str, actual: Any, comparator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "check": check,
        "metric": metric,
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _format_gate_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
