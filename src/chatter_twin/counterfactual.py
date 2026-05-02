from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.models import CutConfig, ModalParams, SimulationConfig, ToolConfig
from chatter_twin.replay import CHATTER_POSITIVE_LABELS, WindowSpec, slice_result_windows
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.scenarios import make_scenario
from chatter_twin.stability import estimate_stability


@dataclass(frozen=True)
class CounterfactualConfig:
    sample_rate_hz: float | None = None
    sensor_noise_std: float | None = None
    seed: int = 707
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.sensor_noise_std is not None and self.sensor_noise_std < 0:
            raise ValueError("sensor_noise_std cannot be negative")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be at least one")


@dataclass(frozen=True)
class StabilityPolicyConfig:
    min_spindle_override: float = 0.90
    max_spindle_override: float = 1.10
    candidate_count: int = 21
    feed_override: float = 1.0
    min_margin_improvement: float = 0.05
    margin_weight: float = 1.0
    productivity_weight: float = 0.03
    move_penalty: float = 0.08

    def __post_init__(self) -> None:
        if self.min_spindle_override <= 0 or self.max_spindle_override <= 0:
            raise ValueError("spindle override bounds must be positive")
        if self.min_spindle_override > self.max_spindle_override:
            raise ValueError("min_spindle_override cannot exceed max_spindle_override")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least two")
        if self.feed_override <= 0:
            raise ValueError("feed_override must be positive")
        if self.min_margin_improvement < 0:
            raise ValueError("min_margin_improvement cannot be negative")


@dataclass(frozen=True)
class CounterfactualPolicyConfig:
    feed_values: tuple[float, ...] = (1.0,)
    spindle_values: tuple[float, ...] = (0.96, 1.0, 1.04, 1.08)
    min_risk_reduction: float = 0.005
    risk_weight: float = 1.0
    productivity_weight: float = 0.02
    move_penalty: float = 0.04

    def __post_init__(self) -> None:
        if not self.feed_values:
            raise ValueError("feed_values must not be empty")
        if not self.spindle_values:
            raise ValueError("spindle_values must not be empty")
        if any(value <= 0 for value in (*self.feed_values, *self.spindle_values)):
            raise ValueError("feed_values and spindle_values must be positive")
        if self.min_risk_reduction < 0:
            raise ValueError("min_risk_reduction cannot be negative")


def run_stability_margin_shadow_policy(
    *,
    shadow_dir: Path,
    out_dir: Path,
    dataset_dir: Path | None = None,
    config: StabilityPolicyConfig | None = None,
    max_windows: int | None = None,
) -> dict[str, Any]:
    config = config or StabilityPolicyConfig()
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be at least one")
    dataset_dir = dataset_dir or infer_dataset_dir_from_shadow(shadow_dir)
    joined_rows = _load_joined_recommendations(shadow_dir, dataset_dir, max_windows)
    recommendations = [_stability_policy_row(row, config) for row in joined_rows]
    payload = summarize_stability_policy(
        recommendations,
        source_shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        config=config,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "recommendations.csv", recommendations)
    _write_json(out_dir / "shadow_metrics.json", payload)
    _write_stability_policy_report(out_dir / "report.md", payload)
    return payload


def run_counterfactual_risk_shadow_policy(
    *,
    shadow_dir: Path,
    out_dir: Path,
    dataset_dir: Path | None = None,
    counterfactual_config: CounterfactualConfig | None = None,
    policy_config: CounterfactualPolicyConfig | None = None,
) -> dict[str, Any]:
    counterfactual_config = counterfactual_config or CounterfactualConfig()
    policy_config = policy_config or CounterfactualPolicyConfig()
    dataset_dir = dataset_dir or infer_dataset_dir_from_shadow(shadow_dir)
    joined_rows = _load_joined_recommendations(shadow_dir, dataset_dir, counterfactual_config.max_windows)
    recommendations = [
        _counterfactual_policy_row(row, idx, counterfactual_config, policy_config)
        for idx, row in enumerate(joined_rows)
    ]
    payload = summarize_counterfactual_policy(
        recommendations,
        source_shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        counterfactual_config=counterfactual_config,
        policy_config=policy_config,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "recommendations.csv", recommendations)
    _write_json(out_dir / "shadow_metrics.json", payload)
    _write_counterfactual_policy_report(out_dir / "report.md", payload)
    return payload


def run_shadow_counterfactual(
    *,
    shadow_dir: Path,
    out_dir: Path,
    dataset_dir: Path | None = None,
    config: CounterfactualConfig | None = None,
) -> dict[str, Any]:
    config = config or CounterfactualConfig()
    dataset_dir = dataset_dir or infer_dataset_dir_from_shadow(shadow_dir)
    joined_rows = _load_joined_recommendations(shadow_dir, dataset_dir, config.max_windows)

    counterfactual_rows = [_simulate_counterfactual_row(row, idx, config) for idx, row in enumerate(joined_rows)]
    payload = summarize_counterfactual(counterfactual_rows, shadow_dir=shadow_dir, dataset_dir=dataset_dir, config=config)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "counterfactual_windows.csv", counterfactual_rows)
    _write_json(out_dir / "counterfactual_metrics.json", payload)
    _write_report(out_dir / "report.md", payload)
    return payload


def run_shadow_episode_counterfactual(
    *,
    shadow_dir: Path,
    out_dir: Path,
    dataset_dir: Path | None = None,
    config: CounterfactualConfig | None = None,
) -> dict[str, Any]:
    config = config or CounterfactualConfig()
    dataset_dir = dataset_dir or infer_dataset_dir_from_shadow(shadow_dir)
    joined_rows = _load_joined_recommendations(shadow_dir, dataset_dir, config.max_windows)
    window_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for episode_idx, rows in enumerate(_group_episode_rows(joined_rows).values()):
        episode_window_rows, episode_summary = _simulate_episode_counterfactual(
            rows,
            episode_idx=episode_idx,
            config=config,
        )
        window_rows.extend(episode_window_rows)
        episode_rows.append(episode_summary)

    payload = summarize_episode_counterfactual(
        window_rows,
        episode_rows,
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        config=config,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "episode_windows.csv", window_rows)
    _write_csv(out_dir / "episode_summary.csv", episode_rows)
    _write_json(out_dir / "episode_metrics.json", payload)
    _write_episode_counterfactual_report(out_dir / "report.md", payload)
    return payload


def run_shadow_action_sweep(
    *,
    shadow_dir: Path,
    out_dir: Path,
    dataset_dir: Path | None = None,
    config: CounterfactualConfig | None = None,
    feed_values: tuple[float, ...] = (0.88, 0.92, 0.96, 1.0),
    spindle_values: tuple[float, ...] = (0.96, 1.0, 1.04, 1.08),
) -> dict[str, Any]:
    config = config or CounterfactualConfig()
    if not feed_values or not spindle_values:
        raise ValueError("feed_values and spindle_values must not be empty")
    if any(value <= 0 for value in (*feed_values, *spindle_values)):
        raise ValueError("feed_values and spindle_values must be positive")

    dataset_dir = dataset_dir or infer_dataset_dir_from_shadow(shadow_dir)
    joined_rows = _load_joined_recommendations(shadow_dir, dataset_dir, config.max_windows)
    baseline_cache = [_simulate_baseline(row, idx, config) for idx, row in enumerate(joined_rows)]

    sweep_rows: list[dict[str, object]] = []
    best_payload: dict[str, Any] | None = None
    best_counterfactual_rows: list[dict[str, object]] = []
    for feed in feed_values:
        for spindle in spindle_values:
            policy_rows = [_with_action_pair(row, feed, spindle) for row in joined_rows]
            counterfactual_rows = [
                _simulate_counterfactual_row(row, idx, config, baseline=baseline)
                for idx, (row, baseline) in enumerate(zip(policy_rows, baseline_cache, strict=True))
            ]
            payload = summarize_counterfactual(
                counterfactual_rows,
                shadow_dir=shadow_dir,
                dataset_dir=dataset_dir,
                config=config,
            )
            summary = {
                "feed_override": feed,
                "spindle_override": spindle,
                "windows": payload["windows"]["count"],
                "action_fraction": payload["windows"]["action_fraction"],
                "relative_mrr_proxy": payload["windows"]["mean_relative_mrr_proxy"],
                "mean_baseline_risk": payload["windows"]["mean_baseline_risk"],
                "mean_shadow_risk": payload["windows"]["mean_shadow_risk"],
                "mean_risk_reduction": payload["windows"]["mean_risk_reduction"],
                "positive_risk_reduction_fraction": payload["windows"]["positive_risk_reduction_fraction"],
                "baseline_chatter_fraction": payload["windows"]["baseline_chatter_fraction"],
                "shadow_chatter_fraction": payload["windows"]["shadow_chatter_fraction"],
                "mitigated_event_episodes": payload["episodes"]["mitigated_event_episodes"],
                "worsened_event_episodes": payload["episodes"]["worsened_event_episodes"],
            }
            sweep_rows.append(summary)
            if best_payload is None or _sweep_score(summary) > _sweep_score(best_payload["best_policy"]):
                best_payload = {**payload, "best_policy": summary}
                best_counterfactual_rows = counterfactual_rows

    assert best_payload is not None
    result = {
        "shadow_dir": str(shadow_dir),
        "dataset_dir": str(dataset_dir),
        "feed_values": list(feed_values),
        "spindle_values": list(spindle_values),
        "candidate_count": len(sweep_rows),
        "best_policy": best_payload["best_policy"],
        "best_windows": best_payload["windows"],
        "best_episodes": best_payload["episodes"],
        "artifacts": ["sweep.csv", "sweep_metrics.json", "best_counterfactual_windows.csv", "report.md"],
        "caveat": best_payload["caveat"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "sweep.csv", sweep_rows)
    _write_csv(out_dir / "best_counterfactual_windows.csv", best_counterfactual_rows)
    _write_json(out_dir / "sweep_metrics.json", result)
    _write_sweep_report(out_dir / "report.md", result)
    return result


def infer_dataset_dir_from_shadow(shadow_dir: Path) -> Path:
    shadow_metrics = _read_json(shadow_dir / "shadow_metrics.json")
    model_dir = Path(str(shadow_metrics["model_dir"]))
    model_metrics = _read_json(model_dir / "metrics.json")
    dataset_path = model_metrics.get("dataset", {}).get("path")
    if not dataset_path:
        raise ValueError("Could not infer dataset path; pass --dataset-dir explicitly")
    return Path(str(dataset_path))


def summarize_counterfactual(
    rows: list[dict[str, object]],
    *,
    shadow_dir: Path,
    dataset_dir: Path,
    config: CounterfactualConfig,
) -> dict[str, Any]:
    total = len(rows)
    baseline_risks = [float(row["baseline_risk_chatter_now"]) for row in rows]
    shadow_risks = [float(row["shadow_risk_chatter_now"]) for row in rows]
    risk_reductions = [float(row["risk_reduction"]) for row in rows]
    baseline_chatter = [_bool(row["baseline_chatter_positive"]) for row in rows]
    shadow_chatter = [_bool(row["shadow_chatter_positive"]) for row in rows]
    actions = [_bool(row["action_active"]) for row in rows]
    mrr = [float(row["relative_mrr_proxy"]) for row in rows]
    event = _episode_counterfactual_metrics(rows)
    return {
        "shadow_dir": str(shadow_dir),
        "dataset_dir": str(dataset_dir),
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "sensor_noise_std": config.sensor_noise_std,
            "seed": config.seed,
            "max_windows": config.max_windows,
        },
        "windows": {
            "count": total,
            "action_windows": sum(actions),
            "action_fraction": sum(actions) / max(total, 1),
            "mean_relative_mrr_proxy": _mean(mrr),
            "mean_baseline_risk": _mean(baseline_risks),
            "mean_shadow_risk": _mean(shadow_risks),
            "mean_risk_reduction": _mean(risk_reductions),
            "positive_risk_reduction_fraction": sum(value > 0.0 for value in risk_reductions) / max(total, 1),
            "baseline_chatter_fraction": sum(baseline_chatter) / max(total, 1),
            "shadow_chatter_fraction": sum(shadow_chatter) / max(total, 1),
        },
        "episodes": event,
        "artifacts": ["counterfactual_windows.csv", "counterfactual_metrics.json", "report.md"],
        "caveat": (
            "Local per-window counterfactual. It applies shadow feed/spindle overrides "
            "to the replay-window process context, but it does not yet propagate vibration "
            "state continuously across an entire cut."
        ),
    }


def summarize_episode_counterfactual(
    window_rows: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
    *,
    shadow_dir: Path,
    dataset_dir: Path,
    config: CounterfactualConfig,
) -> dict[str, Any]:
    total = len(window_rows)
    baseline_risks = [float(row["baseline_risk_chatter_now"]) for row in window_rows]
    shadow_risks = [float(row["shadow_risk_chatter_now"]) for row in window_rows]
    risk_reductions = [float(row["risk_reduction"]) for row in window_rows]
    actions = [_bool(row["action_active"]) for row in window_rows]
    mrr = [float(row["relative_mrr_proxy"]) for row in window_rows]
    baseline_chatter = [_bool(row["baseline_chatter_positive"]) for row in window_rows]
    shadow_chatter = [_bool(row["shadow_chatter_positive"]) for row in window_rows]
    return {
        "shadow_dir": str(shadow_dir),
        "dataset_dir": str(dataset_dir),
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "sensor_noise_std": config.sensor_noise_std,
            "seed": config.seed,
            "max_windows": config.max_windows,
        },
        "windows": {
            "count": total,
            "action_windows": sum(actions),
            "action_fraction": sum(actions) / max(total, 1),
            "mean_relative_mrr_proxy": _mean(mrr),
            "mean_baseline_risk": _mean(baseline_risks),
            "mean_shadow_risk": _mean(shadow_risks),
            "mean_risk_reduction": _mean(risk_reductions),
            "positive_risk_reduction_fraction": sum(value > 0.0 for value in risk_reductions) / max(total, 1),
            "baseline_chatter_fraction": sum(baseline_chatter) / max(total, 1),
            "shadow_chatter_fraction": sum(shadow_chatter) / max(total, 1),
        },
        "episodes": {
            "episode_count": len(episode_rows),
            "baseline_event_episodes": sum(_bool(row["baseline_event"]) for row in episode_rows),
            "shadow_event_episodes": sum(_bool(row["shadow_event"]) for row in episode_rows),
            "mitigated_event_episodes": sum(_bool(row["mitigated_event"]) for row in episode_rows),
            "worsened_event_episodes": sum(_bool(row["worsened_event"]) for row in episode_rows),
            "mean_episode_risk_reduction": _mean([float(row["mean_risk_reduction"]) for row in episode_rows]),
        },
        "artifacts": ["episode_windows.csv", "episode_summary.csv", "episode_metrics.json", "report.md"],
        "caveat": (
            "Full-episode counterfactual with persistent feed/spindle profiles. "
            "The dynamics are still an MVP low-order model and do not yet include controller servo dynamics."
        ),
    }


def summarize_stability_policy(
    rows: list[dict[str, object]],
    *,
    source_shadow_dir: Path,
    dataset_dir: Path,
    config: StabilityPolicyConfig,
) -> dict[str, Any]:
    total = len(rows)
    warnings = [_bool(row["shadow_warning"]) for row in rows]
    actions = [_bool(row["action_active"]) for row in rows]
    relative_mrr = [float(row["relative_mrr_proxy"]) for row in rows]
    improvements = [float(row["selected_margin_improvement"]) for row in rows if _bool(row["shadow_warning"])]
    source_metrics = _read_json(source_shadow_dir / "shadow_metrics.json")
    return {
        "model_dir": source_metrics["model_dir"],
        "source_shadow_dir": str(source_shadow_dir),
        "dataset_dir": str(dataset_dir),
        "policy": {
            "type": "stability_margin",
            "min_spindle_override": config.min_spindle_override,
            "max_spindle_override": config.max_spindle_override,
            "candidate_count": config.candidate_count,
            "feed_override": config.feed_override,
            "min_margin_improvement": config.min_margin_improvement,
            "margin_weight": config.margin_weight,
            "productivity_weight": config.productivity_weight,
            "move_penalty": config.move_penalty,
        },
        "window_metrics": {
            "windows": total,
            "warning_windows": sum(warnings),
            "warning_fraction": sum(warnings) / max(total, 1),
            "action_windows": sum(actions),
            "action_fraction": sum(actions) / max(total, 1),
            "mean_relative_mrr_proxy": _mean(relative_mrr),
            "mean_selected_margin_improvement": _mean(improvements),
        },
        "event_metrics": _event_warning_metrics(rows),
        "artifacts": ["recommendations.csv", "shadow_metrics.json", "report.md"],
    }


def summarize_counterfactual_policy(
    rows: list[dict[str, object]],
    *,
    source_shadow_dir: Path,
    dataset_dir: Path,
    counterfactual_config: CounterfactualConfig,
    policy_config: CounterfactualPolicyConfig,
) -> dict[str, Any]:
    total = len(rows)
    warnings = [_bool(row["shadow_warning"]) for row in rows]
    actions = [_bool(row["action_active"]) for row in rows]
    relative_mrr = [float(row["relative_mrr_proxy"]) for row in rows]
    reductions = [float(row["selected_risk_reduction"]) for row in rows if _bool(row["action_active"])]
    source_metrics = _read_json(source_shadow_dir / "shadow_metrics.json")
    return {
        "model_dir": source_metrics["model_dir"],
        "source_shadow_dir": str(source_shadow_dir),
        "dataset_dir": str(dataset_dir),
        "policy": {
            "type": "counterfactual_risk",
            "feed_values": list(policy_config.feed_values),
            "spindle_values": list(policy_config.spindle_values),
            "min_risk_reduction": policy_config.min_risk_reduction,
            "risk_weight": policy_config.risk_weight,
            "productivity_weight": policy_config.productivity_weight,
            "move_penalty": policy_config.move_penalty,
        },
        "counterfactual_config": {
            "sample_rate_hz": counterfactual_config.sample_rate_hz,
            "sensor_noise_std": counterfactual_config.sensor_noise_std,
            "seed": counterfactual_config.seed,
            "max_windows": counterfactual_config.max_windows,
        },
        "window_metrics": {
            "windows": total,
            "warning_windows": sum(warnings),
            "warning_fraction": sum(warnings) / max(total, 1),
            "action_windows": sum(actions),
            "action_fraction": sum(actions) / max(total, 1),
            "mean_relative_mrr_proxy": _mean(relative_mrr),
            "mean_selected_risk_reduction": _mean(reductions),
        },
        "event_metrics": _event_warning_metrics(rows),
        "artifacts": ["recommendations.csv", "shadow_metrics.json", "report.md"],
    }


def _load_joined_recommendations(shadow_dir: Path, dataset_dir: Path, max_windows: int | None) -> list[dict[str, str]]:
    recommendation_rows = _read_csv(shadow_dir / "recommendations.csv")
    dataset_rows = _read_csv(dataset_dir / "windows.csv")
    if not recommendation_rows:
        raise ValueError("recommendations.csv contains no rows")
    if not dataset_rows:
        raise ValueError("windows.csv contains no rows")

    dataset_by_window_id = {str(row["window_id"]): row for row in dataset_rows}
    joined_rows: list[dict[str, str]] = []
    for row in recommendation_rows:
        window_id = str(row["window_id"])
        if window_id not in dataset_by_window_id:
            raise ValueError(f"Window {window_id} from recommendations.csv is missing from dataset windows.csv")
        joined_rows.append({**dataset_by_window_id[window_id], **row})
        if max_windows is not None and len(joined_rows) >= max_windows:
            break
    return joined_rows


def _group_episode_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(f"{row['scenario']}::episode={row['episode']}", []).append(row)
    for key, values in groups.items():
        groups[key] = sorted(values, key=lambda row: _float(row, "start_time_s", default=_float(row, "window_id")))
    return groups


def _simulate_episode_counterfactual(
    rows: list[dict[str, str]],
    *,
    episode_idx: int,
    config: CounterfactualConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not rows:
        raise ValueError("episode rows must not be empty")
    modal, tool, cut, sim_config = _episode_configs_from_rows(rows)
    sample_rate_hz = config.sample_rate_hz or sim_config.sample_rate_hz
    duration_s = _episode_duration_s(rows, sample_rate_hz)
    sim_config = replace(
        sim_config,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        sensor_noise_std=sim_config.sensor_noise_std if config.sensor_noise_std is None else config.sensor_noise_std,
    )
    n_steps = max(2, int(round(duration_s * sample_rate_hz)))
    baseline_feed = np.ones(n_steps, dtype=float)
    baseline_spindle = np.ones(n_steps, dtype=float)
    shadow_feed, shadow_spindle = _control_profiles_from_rows(rows, n_steps, sample_rate_hz)

    baseline_result = simulate_milling(
        modal,
        tool,
        cut,
        replace(sim_config, random_seed=config.seed + episode_idx * 10),
        feed_override_profile=baseline_feed,
        spindle_override_profile=baseline_spindle,
    )
    shadow_result = simulate_milling(
        modal,
        tool,
        cut,
        replace(sim_config, random_seed=config.seed + episode_idx * 10 + 1),
        feed_override_profile=shadow_feed,
        spindle_override_profile=shadow_spindle,
    )
    window_spec = _window_spec_from_rows(rows, sample_rate_hz)
    _, baseline_records = slice_result_windows(
        result=baseline_result,
        scenario=rows[0]["scenario"],
        episode=int(float(rows[0]["episode"])),
        window_spec=window_spec,
    )
    _, shadow_records = slice_result_windows(
        result=shadow_result,
        scenario=rows[0]["scenario"],
        episode=int(float(rows[0]["episode"])),
        window_spec=window_spec,
    )
    count = min(len(rows), len(baseline_records), len(shadow_records))
    window_rows: list[dict[str, object]] = []
    for source, baseline, shadow in zip(rows[:count], baseline_records[:count], shadow_records[:count], strict=False):
        feed_override = _float(source, "feed_override", default=1.0)
        spindle_override = _float(source, "spindle_override", default=1.0)
        window_rows.append(
            {
                "window_id": source["window_id"],
                "scenario": source["scenario"],
                "episode": source["episode"],
                "start_time_s": source.get("start_time_s", ""),
                "shadow_warning": source.get("shadow_warning", False),
                "action_active": source.get("action_active", False),
                "feed_override": feed_override,
                "spindle_override": spindle_override,
                "relative_mrr_proxy": feed_override * spindle_override,
                "baseline_risk_chatter_now": baseline.risk_chatter_now,
                "shadow_risk_chatter_now": shadow.risk_chatter_now,
                "risk_reduction": baseline.risk_chatter_now - shadow.risk_chatter_now,
                "baseline_label": baseline.label,
                "shadow_label": shadow.label,
                "baseline_chatter_positive": baseline.label in CHATTER_POSITIVE_LABELS,
                "shadow_chatter_positive": shadow.label in CHATTER_POSITIVE_LABELS,
                "baseline_margin_physics": baseline.margin_physics,
                "shadow_margin_physics": shadow.margin_physics,
            }
        )

    baseline_event = any(_bool(row["baseline_chatter_positive"]) for row in window_rows)
    shadow_event = any(_bool(row["shadow_chatter_positive"]) for row in window_rows)
    episode_summary = {
        "scenario": rows[0]["scenario"],
        "episode": rows[0]["episode"],
        "windows": len(window_rows),
        "action_windows": sum(_bool(row["action_active"]) for row in window_rows),
        "mean_relative_mrr_proxy": _mean([float(row["relative_mrr_proxy"]) for row in window_rows]),
        "mean_baseline_risk": _mean([float(row["baseline_risk_chatter_now"]) for row in window_rows]),
        "mean_shadow_risk": _mean([float(row["shadow_risk_chatter_now"]) for row in window_rows]),
        "mean_risk_reduction": _mean([float(row["risk_reduction"]) for row in window_rows]),
        "baseline_event": baseline_event,
        "shadow_event": shadow_event,
        "mitigated_event": baseline_event and not shadow_event,
        "worsened_event": shadow_event and not baseline_event,
    }
    return window_rows, episode_summary


def _episode_configs_from_rows(rows: list[dict[str, str]]) -> tuple[ModalParams, ToolConfig, CutConfig, SimulationConfig]:
    first = rows[0]
    modal, tool, _, sim_config = make_scenario(first["scenario"])
    modal = replace(
        modal,
        stiffness_x_n_m=modal.stiffness_x_n_m * _float(first, "stiffness_scale", default=1.0),
        stiffness_y_n_m=modal.stiffness_y_n_m * _float(first, "stiffness_scale", default=1.0),
        damping_x_n_s_m=modal.damping_x_n_s_m * _float(first, "damping_scale", default=1.0),
        damping_y_n_s_m=modal.damping_y_n_s_m * _float(first, "damping_scale", default=1.0),
    )
    axial_profile_scale = _float(first, "axial_depth_profile_scale", default=1.0)
    coeff_profile_scale = _float(first, "cutting_coeff_profile_scale", default=1.0)
    cut = CutConfig(
        spindle_rpm=_float(first, "spindle_rpm"),
        feed_per_tooth_m=_float(first, "feed_per_tooth_m"),
        axial_depth_m=_float(first, "axial_depth_m") / max(axial_profile_scale, 1.0e-18),
        radial_depth_m=_float(first, "radial_depth_m"),
        cutting_coeff_t_n_m2=_float(first, "cutting_coeff_t_n_m2") / max(coeff_profile_scale, 1.0e-18),
        cutting_coeff_r_n_m2=_float(first, "cutting_coeff_r_n_m2") / max(coeff_profile_scale, 1.0e-18),
    )
    sim_config = replace(
        sim_config,
        sensor_noise_std=sim_config.sensor_noise_std * _float(first, "noise_scale", default=1.0),
    )
    return modal, tool, cut, sim_config


def _control_profiles_from_rows(rows: list[dict[str, str]], n_steps: int, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    feed = np.ones(n_steps, dtype=float)
    spindle = np.ones(n_steps, dtype=float)
    starts = [min(max(0, int(round(_float(row, "start_time_s", default=0.0) * sample_rate_hz))), n_steps - 1) for row in rows]
    for idx, row in enumerate(rows):
        start = starts[idx]
        stop = starts[idx + 1] if idx + 1 < len(starts) else n_steps
        feed[start:stop] = _float(row, "feed_override", default=1.0)
        spindle[start:stop] = _float(row, "spindle_override", default=1.0)
    return feed, spindle


def _episode_duration_s(rows: list[dict[str, str]], sample_rate_hz: float) -> float:
    dt = 1.0 / sample_rate_hz
    end = max(_float(row, "end_time_s", default=_float(row, "start_time_s", default=0.0) + 0.1) for row in rows)
    return max(end + dt, dt)


def _window_spec_from_rows(rows: list[dict[str, str]], sample_rate_hz: float) -> WindowSpec:
    dt = 1.0 / sample_rate_hz
    window_lengths = [
        _float(row, "end_time_s", default=_float(row, "start_time_s", default=0.0) + 0.1)
        - _float(row, "start_time_s", default=0.0)
        + dt
        for row in rows
    ]
    starts = [_float(row, "start_time_s", default=0.0) for row in rows]
    strides = [b - a for a, b in zip(starts, starts[1:], strict=False) if b > a]
    window_s = float(np.median(window_lengths)) if window_lengths else 0.1
    stride_s = float(np.median(strides)) if strides else window_s
    return WindowSpec(window_s=window_s, stride_s=stride_s)


def _counterfactual_policy_row(
    row: dict[str, str],
    row_idx: int,
    counterfactual_config: CounterfactualConfig,
    policy_config: CounterfactualPolicyConfig,
) -> dict[str, object]:
    warning = _bool(row.get("shadow_warning", False))
    baseline = _simulate_baseline(row, row_idx, counterfactual_config)
    baseline_risk = float(baseline["risk_chatter_now"])
    best_feed = 1.0
    best_spindle = 1.0
    best_reduction = 0.0
    best_score = _counterfactual_action_score(
        risk_reduction=0.0,
        feed_override=1.0,
        spindle_override=1.0,
        policy_config=policy_config,
    )
    best_risk = baseline_risk
    best_label = str(baseline["label"])
    best_margin = float(baseline["margin_physics"])

    if warning:
        for feed in _candidate_values_with_identity(policy_config.feed_values):
            for spindle in _candidate_values_with_identity(policy_config.spindle_values):
                feed_override = float(feed)
                spindle_override = float(spindle)
                candidate_row = _with_action_pair_forced(row, feed_override, spindle_override)
                result = _simulate_counterfactual_row(
                    candidate_row,
                    row_idx,
                    counterfactual_config,
                    baseline=baseline,
                )
                risk_reduction = float(result["risk_reduction"])
                score = _counterfactual_action_score(
                    risk_reduction=risk_reduction,
                    feed_override=feed_override,
                    spindle_override=spindle_override,
                    policy_config=policy_config,
                )
                if score > best_score:
                    best_feed = feed_override
                    best_spindle = spindle_override
                    best_reduction = risk_reduction
                    best_score = score
                    best_risk = float(result["shadow_risk_chatter_now"])
                    best_label = str(result["shadow_label"])
                    best_margin = float(result["shadow_margin_physics"])

    action_active = (
        warning
        and best_reduction >= policy_config.min_risk_reduction
        and (abs(best_feed - 1.0) > 1.0e-12 or abs(best_spindle - 1.0) > 1.0e-12)
    )
    if not action_active:
        best_feed = 1.0
        best_spindle = 1.0
        best_reduction = 0.0
        best_risk = baseline_risk
        best_label = str(baseline["label"])
        best_margin = float(baseline["margin_physics"])

    return {
        **row,
        "shadow_warning": warning,
        "action_active": action_active,
        "feed_override": best_feed,
        "spindle_override": best_spindle,
        "relative_mrr_proxy": best_feed * best_spindle,
        "baseline_counterfactual_risk": baseline_risk,
        "selected_counterfactual_risk": best_risk,
        "selected_risk_reduction": best_reduction,
        "selected_counterfactual_label": best_label,
        "selected_counterfactual_margin": best_margin,
        "selected_action_score": best_score,
    }


def _counterfactual_action_score(
    *,
    risk_reduction: float,
    feed_override: float,
    spindle_override: float,
    policy_config: CounterfactualPolicyConfig,
) -> float:
    relative_mrr = feed_override * spindle_override
    move = abs(feed_override - 1.0) + abs(spindle_override - 1.0)
    return (
        policy_config.risk_weight * risk_reduction
        + policy_config.productivity_weight * relative_mrr
        - policy_config.move_penalty * move
    )


def _candidate_values_with_identity(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted({1.0, *values}))


def _with_action_pair_forced(row: dict[str, str], feed_override: float, spindle_override: float) -> dict[str, str]:
    return {
        **row,
        "feed_override": str(feed_override),
        "spindle_override": str(spindle_override),
        "action_active": str(abs(feed_override - 1.0) > 1.0e-12 or abs(spindle_override - 1.0) > 1.0e-12),
        "relative_mrr_proxy": str(feed_override * spindle_override),
    }


def _stability_policy_row(row: dict[str, str], config: StabilityPolicyConfig) -> dict[str, object]:
    warning = _bool(row.get("shadow_warning", False))
    baseline_margin = _stability_margin_for_row(row, feed_override=1.0, spindle_override=1.0)
    selected_spindle = 1.0
    selected_margin = baseline_margin
    selected_score = _stability_action_score(
        margin=baseline_margin,
        margin_baseline=baseline_margin,
        spindle_override=1.0,
        config=config,
    )
    if warning:
        for spindle in np.linspace(config.min_spindle_override, config.max_spindle_override, config.candidate_count):
            spindle_override = float(spindle)
            margin = _stability_margin_for_row(row, feed_override=config.feed_override, spindle_override=spindle_override)
            score = _stability_action_score(
                margin=margin,
                margin_baseline=baseline_margin,
                spindle_override=spindle_override,
                config=config,
            )
            if score > selected_score:
                selected_spindle = spindle_override
                selected_margin = margin
                selected_score = score

    improvement = selected_margin - baseline_margin
    action_active = warning and improvement >= config.min_margin_improvement and abs(selected_spindle - 1.0) > 1.0e-12
    feed_override = config.feed_override if action_active else 1.0
    spindle_override = selected_spindle if action_active else 1.0
    if not action_active:
        selected_spindle = 1.0
        selected_margin = baseline_margin
        improvement = 0.0
    return {
        **row,
        "shadow_warning": warning,
        "action_active": action_active,
        "feed_override": feed_override,
        "spindle_override": spindle_override,
        "relative_mrr_proxy": feed_override * spindle_override,
        "baseline_margin_physics": baseline_margin,
        "selected_margin_physics": selected_margin,
        "selected_margin_improvement": improvement,
        "selected_action_score": selected_score,
    }


def _stability_margin_for_row(row: dict[str, str], *, feed_override: float, spindle_override: float) -> float:
    modal, tool, cut, _ = _configs_from_row(row)
    cut = replace(
        cut,
        feed_per_tooth_m=cut.feed_per_tooth_m * feed_override,
        spindle_rpm=cut.spindle_rpm * spindle_override,
    )
    return estimate_stability(modal, tool, cut).signed_margin


def _stability_action_score(
    *,
    margin: float,
    margin_baseline: float,
    spindle_override: float,
    config: StabilityPolicyConfig,
) -> float:
    improvement = margin - margin_baseline
    productivity = config.feed_override * spindle_override
    move = abs(spindle_override - 1.0)
    return config.margin_weight * improvement + config.productivity_weight * productivity - config.move_penalty * move


def _simulate_counterfactual_row(
    row: dict[str, str],
    row_idx: int,
    config: CounterfactualConfig,
    *,
    baseline: dict[str, float | str] | None = None,
) -> dict[str, object]:
    baseline = baseline or _simulate_baseline(row, row_idx, config)
    modal, tool, cut, sim_config = _configs_from_row(row)
    duration_s = _duration_from_row(row)
    sample_rate_hz = config.sample_rate_hz or sim_config.sample_rate_hz
    sensor_noise_std = sim_config.sensor_noise_std if config.sensor_noise_std is None else config.sensor_noise_std
    shadow_config = replace(
        sim_config,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        sensor_noise_std=sensor_noise_std,
        random_seed=config.seed + row_idx * 2 + 1,
    )
    feed_override = _float(row, "feed_override", default=1.0)
    spindle_override = _float(row, "spindle_override", default=1.0)
    shadow_cut = replace(
        cut,
        feed_per_tooth_m=cut.feed_per_tooth_m * feed_override,
        spindle_rpm=cut.spindle_rpm * spindle_override,
    )

    shadow = _simulate_window(modal, tool, shadow_cut, shadow_config)
    return {
        "window_id": row["window_id"],
        "scenario": row["scenario"],
        "episode": row["episode"],
        "start_time_s": row.get("start_time_s", ""),
        "label": row.get("label", ""),
        "predicted_chatter_score": row.get("predicted_chatter_score", ""),
        "shadow_warning": row.get("shadow_warning", False),
        "action_active": row.get("action_active", False),
        "feed_override": feed_override,
        "spindle_override": spindle_override,
        "relative_mrr_proxy": feed_override * spindle_override,
        "baseline_risk_chatter_now": baseline["risk_chatter_now"],
        "shadow_risk_chatter_now": shadow["risk_chatter_now"],
        "risk_reduction": baseline["risk_chatter_now"] - shadow["risk_chatter_now"],
        "baseline_label": baseline["label"],
        "shadow_label": shadow["label"],
        "baseline_chatter_positive": baseline["label"] in CHATTER_POSITIVE_LABELS,
        "shadow_chatter_positive": shadow["label"] in CHATTER_POSITIVE_LABELS,
        "baseline_margin_physics": baseline["margin_physics"],
        "shadow_margin_physics": shadow["margin_physics"],
        "baseline_rms": baseline["rms"],
        "shadow_rms": shadow["rms"],
        "baseline_chatter_band_energy": baseline["chatter_band_energy"],
        "shadow_chatter_band_energy": shadow["chatter_band_energy"],
    }


def _simulate_baseline(row: dict[str, str], row_idx: int, config: CounterfactualConfig) -> dict[str, float | str]:
    modal, tool, cut, sim_config = _configs_from_row(row)
    baseline_config = replace(
        sim_config,
        duration_s=_duration_from_row(row),
        sample_rate_hz=config.sample_rate_hz or sim_config.sample_rate_hz,
        sensor_noise_std=sim_config.sensor_noise_std if config.sensor_noise_std is None else config.sensor_noise_std,
        random_seed=config.seed + row_idx * 2,
    )
    return _simulate_window(modal, tool, cut, baseline_config)


def _with_action_pair(row: dict[str, str], feed_override: float, spindle_override: float) -> dict[str, str]:
    active = _bool(row.get("shadow_warning", row.get("action_active", False)))
    applied_feed = feed_override if active else 1.0
    applied_spindle = spindle_override if active else 1.0
    return {
        **row,
        "feed_override": str(applied_feed),
        "spindle_override": str(applied_spindle),
        "action_active": str(active and (abs(applied_feed - 1.0) > 1.0e-12 or abs(applied_spindle - 1.0) > 1.0e-12)),
        "relative_mrr_proxy": str(applied_feed * applied_spindle),
    }


def _configs_from_row(row: dict[str, str]) -> tuple[ModalParams, ToolConfig, CutConfig, SimulationConfig]:
    modal, tool, _, sim_config = make_scenario(row["scenario"])
    modal = replace(
        modal,
        stiffness_x_n_m=modal.stiffness_x_n_m * _float(row, "stiffness_scale", default=1.0),
        stiffness_y_n_m=modal.stiffness_y_n_m * _float(row, "stiffness_scale", default=1.0),
        damping_x_n_s_m=modal.damping_x_n_s_m * _float(row, "damping_scale", default=1.0),
        damping_y_n_s_m=modal.damping_y_n_s_m * _float(row, "damping_scale", default=1.0),
    )
    cut = CutConfig(
        spindle_rpm=_float(row, "spindle_rpm"),
        feed_per_tooth_m=_float(row, "feed_per_tooth_m"),
        axial_depth_m=_float(row, "axial_depth_m"),
        radial_depth_m=_float(row, "radial_depth_m"),
        cutting_coeff_t_n_m2=_float(row, "cutting_coeff_t_n_m2"),
        cutting_coeff_r_n_m2=_float(row, "cutting_coeff_r_n_m2"),
    )
    sim_config = replace(
        sim_config,
        axial_depth_start_scale=1.0,
        axial_depth_ramp_per_s=0.0,
        cutting_coeff_start_scale=1.0,
        drift_per_s=0.0,
        sensor_noise_std=sim_config.sensor_noise_std * _float(row, "noise_scale", default=1.0),
    )
    return modal, tool, cut, sim_config


def _simulate_window(
    modal: ModalParams,
    tool: ToolConfig,
    cut: CutConfig,
    sim_config: SimulationConfig,
) -> dict[str, float | str]:
    result = simulate_milling(modal, tool, cut, sim_config)
    features = extract_signal_features(
        result.sensor_signal,
        sim_config.sample_rate_hz,
        cut.spindle_rpm,
        tool.flute_count,
        modal.natural_frequency_hz,
    )
    stability = estimate_stability(modal, tool, cut)
    risk = estimate_chatter_risk(features, stability.signed_margin)
    return {
        "risk_chatter_now": risk.risk_chatter_now,
        "label": risk.label,
        "margin_physics": risk.margin_physics,
        "rms": features.rms,
        "chatter_band_energy": features.chatter_band_energy,
    }


def _episode_counterfactual_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(f"{row['scenario']}::episode={row['episode']}", []).append(row)

    baseline_event_episodes = 0
    shadow_event_episodes = 0
    mitigated_event_episodes = 0
    worsened_event_episodes = 0
    mean_risk_reductions: list[float] = []
    for episode_rows in groups.values():
        baseline_event = any(_bool(row["baseline_chatter_positive"]) for row in episode_rows)
        shadow_event = any(_bool(row["shadow_chatter_positive"]) for row in episode_rows)
        if baseline_event:
            baseline_event_episodes += 1
        if shadow_event:
            shadow_event_episodes += 1
        if baseline_event and not shadow_event:
            mitigated_event_episodes += 1
        if shadow_event and not baseline_event:
            worsened_event_episodes += 1
        mean_risk_reductions.append(_mean([float(row["risk_reduction"]) for row in episode_rows]))

    return {
        "episode_count": len(groups),
        "baseline_event_episodes": baseline_event_episodes,
        "shadow_event_episodes": shadow_event_episodes,
        "mitigated_event_episodes": mitigated_event_episodes,
        "worsened_event_episodes": worsened_event_episodes,
        "mean_episode_risk_reduction": _mean(mean_risk_reductions),
    }


def _event_warning_metrics(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(f"{row['scenario']}::episode={row['episode']}", []).append(row)
    event_episodes = 0
    quiet_episodes = 0
    detected = 0
    missed = 0
    false_warnings = 0
    lead_times: list[float] = []

    for episode_rows in groups.values():
        ordered = sorted(episode_rows, key=lambda row: _float_object(row, "start_time_s", default=_float_object(row, "window_id")))
        chatter_rows = [row for row in ordered if _label_id(row) >= 2]
        first_chatter = chatter_rows[0] if chatter_rows else None
        warning_rows = [row for row in ordered if _bool(row["shadow_warning"])]
        if first_chatter is None:
            quiet_episodes += 1
            if warning_rows:
                false_warnings += 1
            continue
        event_episodes += 1
        first_chatter_time = _float_object(first_chatter, "start_time_s", default=_float_object(first_chatter, "window_id"))
        early_warnings = [
            row
            for row in warning_rows
            if _label_id(row) < 2
            and _float_object(row, "start_time_s", default=_float_object(row, "window_id")) < first_chatter_time
        ]
        if early_warnings:
            detected += 1
            first_warning_time = _float_object(
                early_warnings[0],
                "start_time_s",
                default=_float_object(early_warnings[0], "window_id"),
            )
            lead_times.append(max(0.0, first_chatter_time - first_warning_time))
        else:
            missed += 1

    precision = detected / max(detected + false_warnings, 1)
    recall = detected / max(event_episodes, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "episode_count": len(groups),
        "event_episodes": event_episodes,
        "quiet_episodes": quiet_episodes,
        "detected_event_episodes": detected,
        "missed_event_episodes": missed,
        "false_warning_episodes": false_warnings,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_detected_lead_time_s": _mean(lead_times) if lead_times else None,
    }


def _duration_from_row(row: dict[str, str]) -> float:
    start = _float(row, "start_time_s", default=0.0)
    end = _float(row, "end_time_s", default=start + 0.10)
    return max(end - start, 1.0e-3)


def _float(row: dict[str, str], column: str, default: float | None = None) -> float:
    value = row.get(column)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"Missing required column {column!r}")
        return default
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _label_id(row: dict[str, object]) -> int:
    value = row.get("label_id")
    if value not in (None, ""):
        return int(value)
    label = str(row.get("label", "unknown"))
    return {"stable": 0, "transition": 1, "slight": 2, "severe": 3}.get(label, 4)


def _float_object(row: dict[str, object], column: str, default: float | None = None) -> float:
    value = row.get(column)
    if value in (None, ""):
        if default is None:
            raise ValueError(f"Missing required column {column!r}")
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _sweep_score(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(row["mean_risk_reduction"]),
        float(row["relative_mrr_proxy"]),
        -float(row["worsened_event_episodes"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    windows = payload["windows"]
    episodes = payload["episodes"]
    lines = [
        "# Shadow Counterfactual Report",
        "",
        f"Shadow directory: `{payload['shadow_dir']}`",
        f"Dataset directory: `{payload['dataset_dir']}`",
        "",
        "## Window Outcome",
        "",
        "| Windows | Action fraction | Relative MRR proxy | Baseline risk | Shadow risk | Risk reduction | Baseline chatter | Shadow chatter |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {windows['count']} | {windows['action_fraction']:.3f} | "
        f"{windows['mean_relative_mrr_proxy']:.3f} | {windows['mean_baseline_risk']:.3f} | "
        f"{windows['mean_shadow_risk']:.3f} | {windows['mean_risk_reduction']:.3f} | "
        f"{windows['baseline_chatter_fraction']:.3f} | {windows['shadow_chatter_fraction']:.3f} |",
        "",
        "## Episode Outcome",
        "",
        "| Episodes | Baseline events | Shadow events | Mitigated events | Worsened events | Mean episode risk reduction |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {episodes['episode_count']} | {episodes['baseline_event_episodes']} | "
        f"{episodes['shadow_event_episodes']} | {episodes['mitigated_event_episodes']} | "
        f"{episodes['worsened_event_episodes']} | {episodes['mean_episode_risk_reduction']:.3f} |",
        "",
        payload["caveat"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_episode_counterfactual_report(path: Path, payload: dict[str, Any]) -> None:
    windows = payload["windows"]
    episodes = payload["episodes"]
    lines = [
        "# Shadow Episode Counterfactual Report",
        "",
        f"Shadow directory: `{payload['shadow_dir']}`",
        f"Dataset directory: `{payload['dataset_dir']}`",
        "",
        "## Window Outcome",
        "",
        "| Windows | Action fraction | Relative MRR proxy | Baseline risk | Shadow risk | Risk reduction | Baseline chatter | Shadow chatter |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {windows['count']} | {windows['action_fraction']:.3f} | "
        f"{windows['mean_relative_mrr_proxy']:.3f} | {windows['mean_baseline_risk']:.3f} | "
        f"{windows['mean_shadow_risk']:.3f} | {windows['mean_risk_reduction']:.3f} | "
        f"{windows['baseline_chatter_fraction']:.3f} | {windows['shadow_chatter_fraction']:.3f} |",
        "",
        "## Episode Outcome",
        "",
        "| Episodes | Baseline events | Shadow events | Mitigated events | Worsened events | Mean episode risk reduction |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {episodes['episode_count']} | {episodes['baseline_event_episodes']} | "
        f"{episodes['shadow_event_episodes']} | {episodes['mitigated_event_episodes']} | "
        f"{episodes['worsened_event_episodes']} | {episodes['mean_episode_risk_reduction']:.3f} |",
        "",
        payload["caveat"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_counterfactual_policy_report(path: Path, payload: dict[str, Any]) -> None:
    window = payload["window_metrics"]
    event = payload["event_metrics"]
    policy = payload["policy"]
    lines = [
        "# Counterfactual-Risk Shadow Policy Report",
        "",
        f"Source shadow directory: `{payload['source_shadow_dir']}`",
        f"Dataset directory: `{payload['dataset_dir']}`",
        f"Feed candidates: `{policy['feed_values']}`",
        f"Spindle candidates: `{policy['spindle_values']}`",
        "",
        "## Window Burden",
        "",
        "| Windows | Warning fraction | Action fraction | Relative MRR proxy | Mean selected risk reduction |",
        "|---:|---:|---:|---:|---:|",
        f"| {window['windows']} | {window['warning_fraction']:.3f} | {window['action_fraction']:.3f} | "
        f"{window['mean_relative_mrr_proxy']:.3f} | {window['mean_selected_risk_reduction']:.3f} |",
        "",
        "## Event Outcome",
        "",
        "| Episodes | Event episodes | Detected | Missed | False warnings | Precision | Recall | F1 | Mean lead time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {event['episode_count']} | {event['event_episodes']} | {event['detected_event_episodes']} | "
        f"{event['missed_event_episodes']} | {event['false_warning_episodes']} | "
        f"{event['precision']:.3f} | {event['recall']:.3f} | {event['f1']:.3f} | "
        f"{_format_seconds(event['mean_detected_lead_time_s'])} |",
        "",
        "This policy changes only the recommendation action; it reuses the source warning timing.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_stability_policy_report(path: Path, payload: dict[str, Any]) -> None:
    window = payload["window_metrics"]
    event = payload["event_metrics"]
    lines = [
        "# Stability-Margin Shadow Policy Report",
        "",
        f"Source shadow directory: `{payload['source_shadow_dir']}`",
        f"Dataset directory: `{payload['dataset_dir']}`",
        "",
        "## Window Burden",
        "",
        "| Windows | Warning fraction | Action fraction | Relative MRR proxy | Mean selected margin improvement |",
        "|---:|---:|---:|---:|---:|",
        f"| {window['windows']} | {window['warning_fraction']:.3f} | {window['action_fraction']:.3f} | "
        f"{window['mean_relative_mrr_proxy']:.3f} | {window['mean_selected_margin_improvement']:.3f} |",
        "",
        "## Event Outcome",
        "",
        "| Episodes | Event episodes | Detected | Missed | False warnings | Precision | Recall | F1 | Mean lead time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {event['episode_count']} | {event['event_episodes']} | {event['detected_event_episodes']} | "
        f"{event['missed_event_episodes']} | {event['false_warning_episodes']} | "
        f"{event['precision']:.3f} | {event['recall']:.3f} | {event['f1']:.3f} | "
        f"{_format_seconds(event['mean_detected_lead_time_s'])} |",
        "",
        "This policy changes only the recommendation action; it reuses the source warning timing.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sweep_report(path: Path, payload: dict[str, Any]) -> None:
    best = payload["best_policy"]
    windows = payload["best_windows"]
    episodes = payload["best_episodes"]
    lines = [
        "# Shadow Action Sweep Report",
        "",
        f"Shadow directory: `{payload['shadow_dir']}`",
        f"Dataset directory: `{payload['dataset_dir']}`",
        f"Candidates: `{payload['candidate_count']}`",
        "",
        "## Best Local Policy",
        "",
        "| Feed | Spindle | Relative MRR | Baseline risk | Shadow risk | Risk reduction | Action fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {best['feed_override']:.3f} | {best['spindle_override']:.3f} | "
        f"{best['relative_mrr_proxy']:.3f} | {best['mean_baseline_risk']:.3f} | "
        f"{best['mean_shadow_risk']:.3f} | {best['mean_risk_reduction']:.3f} | "
        f"{best['action_fraction']:.3f} |",
        "",
        "## Best Episode Outcome",
        "",
        "| Episodes | Baseline events | Shadow events | Mitigated events | Worsened events |",
        "|---:|---:|---:|---:|---:|",
        f"| {episodes['episode_count']} | {episodes['baseline_event_episodes']} | "
        f"{episodes['shadow_event_episodes']} | {episodes['mitigated_event_episodes']} | "
        f"{episodes['worsened_event_episodes']} |",
        "",
        "## Best Window Outcome",
        "",
        "| Windows | Positive risk-reduction fraction | Baseline chatter | Shadow chatter |",
        "|---:|---:|---:|---:|",
        f"| {windows['count']} | {windows['positive_risk_reduction_fraction']:.3f} | "
        f"{windows['baseline_chatter_fraction']:.3f} | {windows['shadow_chatter_fraction']:.3f} |",
        "",
        payload["caveat"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_seconds(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"
