from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chatter_twin.replay import LABEL_TO_ID


@dataclass(frozen=True)
class ShadowPolicyConfig:
    threshold_source: str = "event"
    warning_threshold: float | None = None
    clear_threshold: float | None = None
    warning_feed_override: float = 0.92
    warning_spindle_override: float = 1.04
    min_feed_override: float = 0.80
    max_feed_override: float = 1.05
    min_spindle_override: float = 0.90
    max_spindle_override: float = 1.10
    max_feed_delta: float = 0.05
    max_spindle_delta: float = 0.05

    def __post_init__(self) -> None:
        if self.threshold_source not in {"event", "lead", "default", "manual"}:
            raise ValueError("threshold_source must be one of event, lead, default, manual")
        if self.threshold_source == "manual" and self.warning_threshold is None:
            raise ValueError("manual threshold_source requires warning_threshold")
        for name, value in (
            ("warning_feed_override", self.warning_feed_override),
            ("warning_spindle_override", self.warning_spindle_override),
            ("min_feed_override", self.min_feed_override),
            ("max_feed_override", self.max_feed_override),
            ("min_spindle_override", self.min_spindle_override),
            ("max_spindle_override", self.max_spindle_override),
            ("max_feed_delta", self.max_feed_delta),
            ("max_spindle_delta", self.max_spindle_delta),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_feed_override > self.max_feed_override:
            raise ValueError("min_feed_override cannot exceed max_feed_override")
        if self.min_spindle_override > self.max_spindle_override:
            raise ValueError("min_spindle_override cannot exceed max_spindle_override")
        if self.warning_threshold is not None and not 0.0 <= self.warning_threshold <= 1.0:
            raise ValueError("warning_threshold must be in [0, 1]")
        if self.clear_threshold is not None and not 0.0 <= self.clear_threshold <= 1.0:
            raise ValueError("clear_threshold must be in [0, 1]")


def run_shadow_evaluation(
    *,
    model_dir: Path,
    out_dir: Path,
    config: ShadowPolicyConfig | None = None,
) -> dict[str, Any]:
    config = config or ShadowPolicyConfig()
    metrics = _read_json(model_dir / "metrics.json")
    rows = _read_csv(model_dir / "predictions.csv")
    if not rows:
        raise ValueError("predictions.csv contains no rows")

    threshold = resolve_warning_threshold(metrics, config)
    clear_threshold = config.clear_threshold if config.clear_threshold is not None else max(0.0, threshold * 0.75)
    recommendations = recommend_shadow_actions(rows, config, threshold=threshold, clear_threshold=clear_threshold)
    payload = summarize_shadow_recommendations(
        recommendations,
        config=config,
        threshold=threshold,
        clear_threshold=clear_threshold,
        model_metrics=metrics,
        model_dir=model_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "recommendations.csv", recommendations)
    _write_json(out_dir / "shadow_metrics.json", payload)
    _write_report(out_dir / "report.md", payload)
    return payload


def resolve_warning_threshold(metrics: dict[str, Any], config: ShadowPolicyConfig) -> float:
    if config.threshold_source == "manual":
        assert config.warning_threshold is not None
        return float(config.warning_threshold)
    if config.warning_threshold is not None:
        return float(config.warning_threshold)
    if config.threshold_source == "event":
        return float(metrics["event_warning"]["threshold_selection"]["selected_threshold"])
    if config.threshold_source == "lead":
        return float(metrics["lead_time"]["threshold_selection"]["selected_threshold"])
    return float(metrics.get("event_warning", {}).get("threshold", 0.5))


def recommend_shadow_actions(
    rows: list[dict[str, str]],
    config: ShadowPolicyConfig,
    *,
    threshold: float,
    clear_threshold: float,
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    for _, episode_rows in _group_rows(rows).items():
        active = False
        feed = 1.0
        spindle = 1.0
        for row in episode_rows:
            score = _float(row, "predicted_chatter_score")
            if score >= threshold:
                active = True
            elif score <= clear_threshold:
                active = False

            target_feed = config.warning_feed_override if active else 1.0
            target_spindle = config.warning_spindle_override if active else 1.0
            feed = _rate_limit(
                _clip(target_feed, config.min_feed_override, config.max_feed_override),
                feed,
                config.max_feed_delta,
            )
            spindle = _rate_limit(
                _clip(target_spindle, config.min_spindle_override, config.max_spindle_override),
                spindle,
                config.max_spindle_delta,
            )
            action_active = abs(feed - 1.0) > 1.0e-9 or abs(spindle - 1.0) > 1.0e-9
            recommendations.append(
                {
                    **row,
                    "shadow_warning": active,
                    "feed_override": feed,
                    "spindle_override": spindle,
                    "action_active": action_active,
                    "relative_mrr_proxy": feed * spindle,
                }
            )
    return recommendations


def summarize_shadow_recommendations(
    recommendations: list[dict[str, object]],
    *,
    config: ShadowPolicyConfig,
    threshold: float,
    clear_threshold: float,
    model_metrics: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    event_metrics = shadow_event_metrics(recommendations)
    total = len(recommendations)
    warning_windows = sum(_bool(row["shadow_warning"]) for row in recommendations)
    action_windows = sum(_bool(row["action_active"]) for row in recommendations)
    relative_mrr = _mean([float(row["relative_mrr_proxy"]) for row in recommendations])
    mean_feed = _mean([float(row["feed_override"]) for row in recommendations])
    mean_spindle = _mean([float(row["spindle_override"]) for row in recommendations])
    payload = {
        "model_dir": str(model_dir),
        "policy": {
            **asdict(config),
            "resolved_warning_threshold": threshold,
            "resolved_clear_threshold": clear_threshold,
        },
        "dataset": {
            "prediction_windows": total,
            "episodes": len(_group_rows(recommendations)),
        },
        "window_metrics": {
            "warning_windows": warning_windows,
            "warning_fraction": warning_windows / max(total, 1),
            "action_windows": action_windows,
            "action_fraction": action_windows / max(total, 1),
            "mean_feed_override": mean_feed,
            "mean_spindle_override": mean_spindle,
            "relative_mrr_proxy": relative_mrr,
            "relative_mrr_loss_proxy": max(0.0, 1.0 - relative_mrr),
        },
        "event_metrics": event_metrics,
        "source_model": {
            "test_accuracy": model_metrics["test"]["accuracy"],
            "test_chatter_f1": model_metrics["test"]["binary_chatter_f1"],
            "test_event_warning_f1": model_metrics["event_warning"]["test"]["f1"],
            "test_event_warning_recall": model_metrics["event_warning"]["test"]["recall"],
        },
        "artifacts": ["recommendations.csv", "shadow_metrics.json", "report.md"],
    }
    return payload


def shadow_event_metrics(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    groups = _group_rows(rows)
    event_episodes = 0
    quiet_episodes = 0
    detected_events = 0
    missed_events = 0
    false_warning_episodes = 0
    true_quiet_episodes = 0
    lead_times: list[float] = []

    for episode_rows in groups.values():
        chatter_rows = [row for row in episode_rows if _label_id(row) >= LABEL_TO_ID["slight"]]
        first_chatter = chatter_rows[0] if chatter_rows else None
        warnings = [row for row in episode_rows if _bool(row["shadow_warning"])]
        if first_chatter is None:
            quiet_episodes += 1
            if warnings:
                false_warning_episodes += 1
            else:
                true_quiet_episodes += 1
            continue

        event_episodes += 1
        first_chatter_order = _row_order(first_chatter)
        early_warnings = [
            row
            for row in warnings
            if _label_id(row) < LABEL_TO_ID["slight"] and _row_order(row) < first_chatter_order
        ]
        if early_warnings:
            detected_events += 1
            lead_times.append(max(0.0, _time_or_order(first_chatter) - _time_or_order(early_warnings[0])))
        else:
            missed_events += 1

    precision = detected_events / max(detected_events + false_warning_episodes, 1)
    recall = detected_events / max(event_episodes, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "episode_count": len(groups),
        "event_episodes": event_episodes,
        "quiet_episodes": quiet_episodes,
        "detected_event_episodes": detected_events,
        "missed_event_episodes": missed_events,
        "false_warning_episodes": false_warning_episodes,
        "true_quiet_episodes": true_quiet_episodes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_detected_lead_time_s": _mean_or_none(lead_times),
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row.get('scenario', '')}::episode={row.get('episode', '')}", []).append(row)
    for key, values in groups.items():
        groups[key] = sorted(values, key=_row_order)
    return groups


def _row_order(row: dict[str, Any]) -> float:
    if row.get("start_time_s") not in (None, ""):
        return _float(row, "start_time_s")
    return _float(row, "window_id")


def _time_or_order(row: dict[str, Any]) -> float:
    return _row_order(row)


def _label_id(row: dict[str, Any]) -> int:
    if row.get("label_id") not in (None, ""):
        return int(row["label_id"])
    return LABEL_TO_ID.get(str(row.get("label", "unknown")), LABEL_TO_ID["unknown"])


def _float(row: dict[str, Any], column: str) -> float:
    return float(row[column])


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    valid = [value for value in values if value >= 0.0]
    return _mean(valid) if valid else None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    return _clip(value, previous - max_delta, previous + max_delta)


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
    window = payload["window_metrics"]
    event = payload["event_metrics"]
    lines = [
        "# Shadow Recommendation Report",
        "",
        f"Model directory: `{payload['model_dir']}`",
        f"Windows: `{payload['dataset']['prediction_windows']}`",
        f"Episodes: `{payload['dataset']['episodes']}`",
        f"Warning threshold: `{payload['policy']['resolved_warning_threshold']:.3f}`",
        f"Clear threshold: `{payload['policy']['resolved_clear_threshold']:.3f}`",
        "",
        "## Window Burden",
        "",
        "| Warning fraction | Action fraction | Mean feed | Mean spindle | Relative MRR proxy | MRR loss proxy |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {window['warning_fraction']:.3f} | {window['action_fraction']:.3f} | "
        f"{window['mean_feed_override']:.3f} | {window['mean_spindle_override']:.3f} | "
        f"{window['relative_mrr_proxy']:.3f} | {window['relative_mrr_loss_proxy']:.3f} |",
        "",
        "## Event Outcome",
        "",
        "| Episodes | Event episodes | Detected | Missed | False warnings | Precision | Recall | F1 | Mean lead time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {event['episode_count']} | {event['event_episodes']} | {event['detected_event_episodes']} | "
        f"{event['missed_event_episodes']} | {event['false_warning_episodes']} | "
        f"{event['precision']:.3f} | {event['recall']:.3f} | {event['f1']:.3f} | "
        f"{_format_optional_seconds(event['mean_detected_lead_time_s'])} |",
        "",
        "This is an offline shadow-mode recommendation report. No CNC command is issued.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional_seconds(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"
