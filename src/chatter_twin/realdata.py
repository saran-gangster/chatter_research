from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CHATTER_LABELS = frozenset({"slight", "severe"})
LABEL_ORDER = ("stable", "transition", "slight", "severe", "unknown")


@dataclass(frozen=True)
class RealDataRunSpec:
    dataset: str
    modality: str
    result_dir: Path
    claim_allowed: str
    note: str = ""


DEFAULT_REAL_DATA_RUNS: tuple[RealDataRunSpec, ...] = (
    RealDataRunSpec(
        dataset="i-CNC Zenodo",
        modality="10 kHz vibration X/Y packages",
        result_dir=Path("results/icnc_risk_temporal_pkg_baseline_1000pkg"),
        claim_allowed="current-window signal sanity only; package labels and no calibrated process context",
    ),
    RealDataRunSpec(
        dataset="KIT force+accel",
        modality="10 kHz synchronized acceleration + force",
        result_dir=Path("results/kit_mat_force_accel_pseudo_onset_horizon_baseline"),
        claim_allowed="pipeline sanity only; row split with exploratory pseudo-onset labels",
    ),
    RealDataRunSpec(
        dataset="KIT force+accel",
        modality="10 kHz synchronized acceleration + force",
        result_dir=Path("results/kit_mat_force_accel_pseudo_onset_horizon_time_block"),
        claim_allowed="honest time-forward stress test; currently no early-warning claim",
    ),
    RealDataRunSpec(
        dataset="Purdue MT cutting",
        modality="48 kHz sound sensor 0",
        result_dir=Path("results/mt_cutting_sensor0_current_scenario_baseline_12k"),
        claim_allowed="best current-window Purdue unseen-experiment audio baseline",
    ),
    RealDataRunSpec(
        dataset="Purdue MT cutting",
        modality="48 kHz sound sensor 1",
        result_dir=Path("results/mt_cutting_sensor1_current_episode_baseline_12k"),
        claim_allowed="current-window audio chatter validation across held-out cutting paths",
    ),
    RealDataRunSpec(
        dataset="Purdue MT cutting",
        modality="48 kHz sound sensor 1",
        result_dir=Path("results/mt_cutting_sensor1_current_scenario_baseline_12k"),
        claim_allowed="stricter current-window audio validation across unseen experiment folders",
    ),
    RealDataRunSpec(
        dataset="Purdue MT cutting",
        modality="48 kHz sound sensor 2",
        result_dir=Path("results/mt_cutting_sensor2_current_scenario_baseline_12k"),
        claim_allowed="single-sensor diagnostic; not current champion on experiment holdout",
    ),
    RealDataRunSpec(
        dataset="Purdue MT cutting",
        modality="48 kHz sound sensors 0,1,2 norm",
        result_dir=Path("results/mt_cutting_3sensor_current_scenario_baseline_12k"),
        claim_allowed="sensor-fusion diagnostic; not current champion on experiment holdout",
    ),
)


def parse_real_data_run_spec(text: str) -> RealDataRunSpec:
    """Parse a CLI run spec: dataset=...,modality=...,path=...,claim=...,note=..."""

    fields: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Run spec part {part!r} must be KEY=VALUE")
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    missing = {"dataset", "modality", "path", "claim"} - set(fields)
    if missing:
        raise ValueError(f"Run spec is missing fields: {', '.join(sorted(missing))}")
    return RealDataRunSpec(
        dataset=fields["dataset"],
        modality=fields["modality"],
        result_dir=Path(fields["path"]),
        claim_allowed=fields["claim"],
        note=fields.get("note", ""),
    )


def write_real_data_benchmark(
    *,
    out_dir: Path,
    runs: tuple[RealDataRunSpec, ...] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = runs or DEFAULT_REAL_DATA_RUNS
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for spec in specs:
        metrics_path = spec.result_dir / "metrics.json"
        if not metrics_path.exists():
            skipped.append(
                {
                    "dataset": spec.dataset,
                    "modality": spec.modality,
                    "path": str(spec.result_dir),
                    "reason": "missing metrics.json",
                }
            )
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_benchmark_row(spec, metrics))

    payload = {
        "rows": rows,
        "skipped": skipped,
        "artifacts": ["real_data_benchmark.csv", "real_data_benchmark.json", "report.md"],
    }
    _write_dict_csv(out_dir / "real_data_benchmark.csv", rows)
    (out_dir / "real_data_benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_benchmark_report(out_dir / "report.md", payload)
    return payload


def write_risk_error_analysis(
    *,
    model_dir: Path,
    out_dir: Path,
    group_column: str = "scenario",
) -> dict[str, Any]:
    predictions_path = model_dir / "predictions.csv"
    if not predictions_path.exists():
        raise ValueError(f"{predictions_path} does not exist")
    rows = _read_csv(predictions_path)
    if not rows:
        raise ValueError(f"{predictions_path} contains no rows")
    if group_column not in rows[0]:
        raise ValueError(f"predictions.csv is missing group column {group_column!r}")

    label_metrics = _label_metrics(rows)
    confusion_rows = _confusion_rows(rows)
    pair_rows = _confusion_pair_rows(rows)
    group_rows = _group_metrics(rows, group_column)
    failure_rows = [
        row
        for row in sorted(group_rows, key=lambda item: (_none_last(item["chatter_f1"]), -int(item["chatter_support"])))
        if int(row["chatter_support"]) > 0
    ]
    payload = {
        "model_dir": str(model_dir),
        "group_column": group_column,
        "windows": len(rows),
        "label_metrics": label_metrics,
        "group_metrics": group_rows,
        "worst_groups": failure_rows[:10],
        "artifacts": [
            "label_metrics.csv",
            "confusion_matrix.csv",
            "confusion_pairs.csv",
            "group_metrics.csv",
            "error_analysis.json",
            "report.md",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_dict_csv(out_dir / "label_metrics.csv", label_metrics)
    _write_dict_csv(out_dir / "confusion_matrix.csv", confusion_rows)
    _write_dict_csv(out_dir / "confusion_pairs.csv", pair_rows)
    _write_dict_csv(out_dir / "group_metrics.csv", group_rows)
    (out_dir / "error_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_error_report(out_dir / "report.md", payload)
    return payload


def _benchmark_row(spec: RealDataRunSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test") or {}
    lead_time = metrics.get("lead_time") or {}
    lead_test = lead_time.get("test") or {}
    selected = ((lead_time.get("threshold_selection") or {}).get("test_at_selected_threshold") or {})
    split = metrics.get("split") or {}
    target = metrics.get("target") or {}
    return {
        "dataset": spec.dataset,
        "sensor_modality": spec.modality,
        "result_dir": str(spec.result_dir),
        "split_type": split.get("mode", ""),
        "target": target.get("mode", ""),
        "test_accuracy": _round_or_blank(test.get("accuracy")),
        "chatter_f1": _round_or_blank(test.get("binary_chatter_f1")),
        "intervention_f1": _round_or_blank(test.get("intervention_f1")),
        "lead_time_f1": _round_or_blank(lead_test.get("f1")),
        "selected_lead_time_f1": _round_or_blank(selected.get("f1")),
        "event_warning_f1": _round_or_blank(((metrics.get("event_warning") or {}).get("test") or {}).get("f1")),
        "claim_allowed": spec.claim_allowed,
        "note": spec.note,
    }


def _label_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = [label for label in LABEL_ORDER if any(_truth(row) == label or _pred(row) == label for row in rows)]
    out: list[dict[str, Any]] = []
    for label in labels:
        tp = sum(_truth(row) == label and _pred(row) == label for row in rows)
        fp = sum(_truth(row) != label and _pred(row) == label for row in rows)
        fn = sum(_truth(row) == label and _pred(row) != label for row in rows)
        support = sum(_truth(row) == label for row in rows)
        out.append(
            {
                "label": label,
                "support": support,
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, tp + fn),
                "f1": _f1(tp, fp, fn),
            }
        )
    tp, fp, fn = _binary_counts(rows)
    out.append(
        {
            "label": "chatter_binary",
            "support": sum(_is_chatter(_truth(row)) for row in rows),
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "f1": _f1(tp, fp, fn),
        }
    )
    return out


def _confusion_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = [label for label in LABEL_ORDER if any(_truth(row) == label or _pred(row) == label for row in rows)]
    matrix = Counter((_truth(row), _pred(row)) for row in rows)
    return [{"truth": truth, **{pred: matrix[(truth, pred)] for pred in labels}} for truth in labels]


def _confusion_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    pairs = Counter((_truth(row), _pred(row)) for row in rows if _truth(row) != _pred(row))
    return [
        {"truth": truth, "predicted": pred, "count": count}
        for (truth, pred), count in sorted(pairs.items(), key=lambda item: item[1], reverse=True)
    ]


def _group_metrics(rows: list[dict[str, str]], group_column: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(group_column, "")].append(row)
    out: list[dict[str, Any]] = []
    for group, group_rows in sorted(groups.items()):
        correct = sum(_truth(row) == _pred(row) for row in group_rows)
        tp, fp, fn = _binary_counts(group_rows)
        truth_counts = Counter(_truth(row) for row in group_rows)
        pred_counts = Counter(_pred(row) for row in group_rows)
        out.append(
            {
                group_column: group,
                "windows": len(group_rows),
                "accuracy": _safe_div(correct, len(group_rows)),
                "chatter_support": sum(_is_chatter(_truth(row)) for row in group_rows),
                "chatter_precision": _safe_div(tp, tp + fp),
                "chatter_recall": _safe_div(tp, tp + fn),
                "chatter_f1": _f1(tp, fp, fn),
                "stable_support": truth_counts.get("stable", 0),
                "slight_support": truth_counts.get("slight", 0),
                "severe_support": truth_counts.get("severe", 0),
                "predicted_stable": pred_counts.get("stable", 0),
                "predicted_slight": pred_counts.get("slight", 0),
                "predicted_severe": pred_counts.get("severe", 0),
                "false_positive_windows": fp,
                "false_negative_windows": fn,
            }
        )
    return out


def _binary_counts(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for row in rows:
        truth_chatter = _is_chatter(_truth(row))
        pred_chatter = _is_chatter(_pred(row))
        if truth_chatter and pred_chatter:
            tp += 1
        elif not truth_chatter and pred_chatter:
            fp += 1
        elif truth_chatter and not pred_chatter:
            fn += 1
    return tp, fp, fn


def _truth(row: dict[str, str]) -> str:
    return row.get("target_label") or row.get("label") or "unknown"


def _pred(row: dict[str, str]) -> str:
    return row.get("predicted_label", "unknown")


def _is_chatter(label: str) -> bool:
    return label in CHATTER_LABELS


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def _round_or_blank(value: object) -> float | str:
    if value is None:
        return ""
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return ""


def _none_last(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 2.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Real-Data Benchmark",
        "",
        "| Dataset | Sensor Modality | Split | Target | Accuracy | Chatter F1 | Lead-Time F1 | Claim Allowed |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['sensor_modality']} | `{row['split_type']}` | `{row['target']}` | "
            f"{_fmt(row['test_accuracy'])} | {_fmt(row['chatter_f1'])} | {_fmt(row['lead_time_f1'])} | "
            f"{row['claim_allowed']} |"
        )
    if payload["skipped"]:
        lines.extend(["", "## Skipped", "", "| Dataset | Path | Reason |", "|---|---|---|"])
        for item in payload["skipped"]:
            lines.append(f"| {item['dataset']} | `{item['path']}` | {item['reason']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error_report(path: Path, payload: dict[str, Any]) -> None:
    group_column = payload["group_column"]
    lines = [
        "# Risk Error Analysis",
        "",
        f"Model: `{payload['model_dir']}`",
        f"Windows: `{payload['windows']}`",
        f"Group column: `{group_column}`",
        "",
        "## Label Metrics",
        "",
        "| Label | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["label_metrics"]:
        lines.append(
            f"| {row['label']} | {row['support']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Worst Chatter Groups",
            "",
            f"| {group_column} | Windows | Chatter Support | Chatter F1 | False Negatives | False Positives |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["worst_groups"]:
        lines.append(
            f"| `{row[group_column]}` | {row['windows']} | {row['chatter_support']} | {_fmt(row['chatter_f1'])} | "
            f"{row['false_negative_windows']} | {row['false_positive_windows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    if value == "":
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
