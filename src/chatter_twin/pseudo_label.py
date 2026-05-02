from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chatter_twin.replay import CHATTER_POSITIVE_LABELS, LABEL_SEVERITY, LABEL_TO_ID

DEFAULT_PSEUDO_SCORE_COLUMNS = (
    "rms",
    "chatter_band_energy",
    "non_tooth_harmonic_ratio",
    "chatter_band_energy_ewma",
    "non_tooth_harmonic_ratio_ewma",
    "chatter_band_energy_growth_rate",
)
LOG_SCORE_COLUMNS = frozenset(
    {
        "rms",
        "chatter_band_energy",
        "tooth_band_energy",
        "non_tooth_harmonic_ratio",
        "rms_ewma",
        "chatter_band_energy_ewma",
        "non_tooth_harmonic_ratio_ewma",
    }
)
POSITIVE_MODES = frozenset({"scenario", "episode", "label"})


@dataclass(frozen=True)
class PseudoLabelConfig:
    score_columns: tuple[str, ...] = DEFAULT_PSEUDO_SCORE_COLUMNS
    positive_scenarios: tuple[str, ...] = ()
    positive_mode: str = "scenario"
    horizon_s: float | None = None
    transition_quantile: float = 0.95
    slight_quantile: float = 0.99
    severe_quantile: float = 0.997
    transition_floor: float = 1.0
    slight_floor: float = 2.0
    severe_floor: float = 3.5

    def __post_init__(self) -> None:
        if not self.score_columns:
            raise ValueError("score_columns cannot be empty")
        if self.positive_mode not in POSITIVE_MODES:
            raise ValueError(f"positive_mode must be one of: {', '.join(sorted(POSITIVE_MODES))}")
        if self.horizon_s is not None and self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        for name, value in (
            ("transition_quantile", self.transition_quantile),
            ("slight_quantile", self.slight_quantile),
            ("severe_quantile", self.severe_quantile),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.transition_quantile <= self.slight_quantile <= self.severe_quantile:
            raise ValueError("pseudo-label quantiles must be ordered")


def pseudo_label_replay_dataset(
    *,
    dataset_dir: Path,
    out_dir: Path,
    config: PseudoLabelConfig | None = None,
) -> dict[str, object]:
    config = config or PseudoLabelConfig()
    rows = _read_windows(dataset_dir / "windows.csv")
    if not rows:
        raise ValueError(f"No replay windows found in {dataset_dir / 'windows.csv'}")
    _validate_score_columns(rows, config.score_columns)

    stable_rows = [row for row in rows if row.get("label") == "stable"]
    if not stable_rows:
        raise ValueError("Need at least one stable row to build pseudo-label baseline")
    positive_scenarios = set(config.positive_scenarios) or {
        row["scenario"] for row in rows if row.get("label") in CHATTER_POSITIVE_LABELS
    }
    if not positive_scenarios:
        raise ValueError("No positive scenarios found; pass positive_scenarios explicitly")
    positive_episodes = {
        (row["scenario"], row["episode"])
        for row in rows
        if row.get("label") in CHATTER_POSITIVE_LABELS and row["scenario"] in positive_scenarios
    }

    baseline = _fit_baseline(stable_rows, config.score_columns)
    stable_scores = _score_rows(stable_rows, baseline, config.score_columns)
    thresholds = _thresholds(stable_scores, config)

    updated_rows: list[dict[str, str]] = []
    changed = 0
    candidate_windows = 0
    label_counts_before = Counter(row["label"] for row in rows)
    score_by_window_id: dict[str, float] = {}
    for row in rows:
        score = _score_row(row, baseline, config.score_columns)
        score_by_window_id[row["window_id"]] = score
        updated = dict(row)
        candidate = _is_pseudo_label_candidate(row, config, positive_scenarios, positive_episodes)
        updated["source_label"] = row.get("source_label") or row.get("label", "unknown")
        updated["source_label_id"] = row.get("source_label_id") or row.get("label_id", "")
        updated["pseudo_label_candidate"] = str(candidate)
        updated["pseudo_label_score"] = f"{score:.12g}"
        if candidate:
            candidate_windows += 1
            new_label = _score_to_label(score, thresholds)
            if new_label != row["label"]:
                changed += 1
            _set_current_label(updated, new_label)
        updated_rows.append(updated)

    horizon_s = float(config.horizon_s or _first_float(rows, "horizon_s", default=0.25))
    _attach_horizon_labels(updated_rows, horizon_s=horizon_s)

    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_npz_with_labels(dataset_dir / "dataset.npz", out_dir / "dataset.npz", updated_rows)
    _write_windows(out_dir / "windows.csv", updated_rows)
    manifest = _write_manifest(dataset_dir / "manifest.json", out_dir / "manifest.json", updated_rows, config, thresholds)
    _write_readme(out_dir / "README.md", updated_rows, config, thresholds, dataset_dir, changed, score_by_window_id)

    return {
        "out_dir": str(out_dir),
        "source_dataset": str(dataset_dir),
        "total_windows": len(updated_rows),
        "changed_windows": changed,
        "candidate_windows": candidate_windows,
        "label_counts_before": dict(sorted(label_counts_before.items())),
        "label_counts_after": dict(sorted(Counter(row["label"] for row in updated_rows).items())),
        "positive_scenarios": sorted(positive_scenarios),
        "positive_mode": config.positive_mode,
        "thresholds": thresholds,
        "manifest": manifest,
        "artifacts": ["dataset.npz", "windows.csv", "manifest.json", "README.md"],
    }


def _read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_windows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_score_columns(rows: list[dict[str, str]], score_columns: tuple[str, ...]) -> None:
    missing = [column for column in score_columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing pseudo-label score columns: {', '.join(missing)}")


def _fit_baseline(rows: list[dict[str, str]], score_columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    baseline: dict[str, dict[str, float]] = {}
    for column in score_columns:
        values = np.array([_score_value(row, column) for row in rows], dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        robust_scale = max(1.4826 * mad, float((q75 - q25) / 1.349), 1.0)
        baseline[column] = {"median": median, "scale": robust_scale}
    return baseline


def _score_rows(
    rows: list[dict[str, str]],
    baseline: dict[str, dict[str, float]],
    score_columns: tuple[str, ...],
) -> NDArray[np.float64]:
    return np.array([_score_row(row, baseline, score_columns) for row in rows], dtype=float)


def _score_row(row: dict[str, str], baseline: dict[str, dict[str, float]], score_columns: tuple[str, ...]) -> float:
    parts: list[float] = []
    for column in score_columns:
        value = _score_value(row, column)
        center = baseline[column]["median"]
        scale = baseline[column]["scale"]
        parts.append(max(0.0, (value - center) / scale))
    return float(np.mean(parts))


def _score_value(row: dict[str, str], column: str) -> float:
    value = _float(row.get(column), default=0.0)
    if column in LOG_SCORE_COLUMNS:
        return float(np.log10(max(value, 1.0e-18)))
    return value


def _thresholds(scores: NDArray[np.float64], config: PseudoLabelConfig) -> dict[str, float]:
    transition = max(float(np.quantile(scores, config.transition_quantile)), config.transition_floor)
    slight = max(float(np.quantile(scores, config.slight_quantile)), config.slight_floor, transition + 1.0e-6)
    severe = max(float(np.quantile(scores, config.severe_quantile)), config.severe_floor, slight + 1.0e-6)
    return {"transition": transition, "slight": slight, "severe": severe}


def _score_to_label(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["severe"]:
        return "severe"
    if score >= thresholds["slight"]:
        return "slight"
    if score >= thresholds["transition"]:
        return "transition"
    return "stable"


def _is_pseudo_label_candidate(
    row: dict[str, str],
    config: PseudoLabelConfig,
    positive_scenarios: set[str],
    positive_episodes: set[tuple[str, str]],
) -> bool:
    if row["scenario"] not in positive_scenarios:
        return False
    match config.positive_mode:
        case "scenario":
            return True
        case "episode":
            return (row["scenario"], row["episode"]) in positive_episodes
        case "label":
            return row.get("label") in CHATTER_POSITIVE_LABELS
    raise ValueError(f"Unsupported positive_mode: {config.positive_mode}")


def _set_current_label(row: dict[str, str], label: str) -> None:
    row["label"] = label
    row["label_id"] = str(LABEL_TO_ID[label])
    row["chatter_within_horizon"] = str(label in CHATTER_POSITIVE_LABELS)
    row["time_to_chatter_s"] = "0.0" if label in CHATTER_POSITIVE_LABELS else "-1.0"


def _attach_horizon_labels(rows: list[dict[str, str]], *, horizon_s: float) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["episode"]), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: _float(row["start_time_s"]))
        starts = np.array([_float(row["start_time_s"]) for row in group_rows], dtype=float)
        for idx, row in enumerate(group_rows):
            stop_time = starts[idx] + horizon_s + 1.0e-12
            stop_idx = int(np.searchsorted(starts, stop_time, side="right"))
            horizon_rows = group_rows[idx:stop_idx]
            future_rows = group_rows[idx + 1 : stop_idx]
            horizon_label = max((candidate["label"] for candidate in horizon_rows), key=lambda label: LABEL_SEVERITY[label])
            first_chatter = next((candidate for candidate in horizon_rows if candidate["label"] in CHATTER_POSITIVE_LABELS), None)
            future_chatter = next((candidate for candidate in future_rows if candidate["label"] in CHATTER_POSITIVE_LABELS), None)
            row["horizon_label"] = horizon_label
            row["horizon_label_id"] = str(LABEL_TO_ID[horizon_label])
            row["horizon_s"] = str(horizon_s)
            row["chatter_within_horizon"] = str(first_chatter is not None)
            row["future_chatter_within_horizon"] = str(future_chatter is not None)
            row["time_to_chatter_s"] = (
                str(max(0.0, _float(first_chatter["start_time_s"]) - starts[idx])) if first_chatter is not None else "-1.0"
            )


def _copy_npz_with_labels(source: Path, target: Path, rows: list[dict[str, str]]) -> None:
    if not source.exists():
        raise ValueError(f"Source dataset.npz not found: {source}")
    source_data = np.load(source, allow_pickle=True)
    arrays = {key: source_data[key] for key in source_data.files}
    arrays["labels"] = np.array([row["label"] for row in rows])
    arrays["label_ids"] = np.array([int(row["label_id"]) for row in rows], dtype=np.int64)
    arrays["horizon_labels"] = np.array([row["horizon_label"] for row in rows])
    arrays["horizon_label_ids"] = np.array([int(row["horizon_label_id"]) for row in rows], dtype=np.int64)
    arrays["chatter_within_horizon"] = np.array([_bool(row["chatter_within_horizon"]) for row in rows], dtype=np.bool_)
    arrays["future_chatter_within_horizon"] = np.array(
        [_bool(row["future_chatter_within_horizon"]) for row in rows],
        dtype=np.bool_,
    )
    arrays["time_to_chatter_s"] = np.array([_float(row["time_to_chatter_s"]) for row in rows], dtype=np.float32)
    np.savez_compressed(target, **arrays)


def _write_manifest(
    source: Path,
    target: Path,
    rows: list[dict[str, str]],
    config: PseudoLabelConfig,
    thresholds: dict[str, float],
) -> dict[str, object]:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["label_counts"] = dict(sorted(Counter(row["label"] for row in rows).items()))
    manifest["total_windows"] = len(rows)
    manifest["horizon"] = {"horizon_s": _first_float(rows, "horizon_s", default=0.25)}
    strategy = dict(manifest.get("sampling_strategy", {}))
    strategy["pseudo_labeling"] = {
        "method": "stable_robust_score_quantile",
        "config": asdict(config),
        "thresholds": thresholds,
    }
    manifest["sampling_strategy"] = strategy
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_readme(
    path: Path,
    rows: list[dict[str, str]],
    config: PseudoLabelConfig,
    thresholds: dict[str, float],
    source_dataset: Path,
    changed: int,
    score_by_window_id: dict[str, float],
) -> None:
    label_counts = Counter(row["label"] for row in rows)
    candidate_count = sum(_bool(row.get("pseudo_label_candidate")) for row in rows)
    positive_rows = [row for row in rows if row["label"] in CHATTER_POSITIVE_LABELS]
    lead_rows = [
        row
        for row in rows
        if row["label"] not in CHATTER_POSITIVE_LABELS and _bool(row["future_chatter_within_horizon"])
    ]
    score_values = np.array(list(score_by_window_id.values()), dtype=float)
    lines = [
        "# Pseudo-Labeled Replay Dataset",
        "",
        f"Source dataset: `{source_dataset}`",
        "",
        f"Total windows: `{len(rows)}`",
        f"Pseudo-label mode: `{config.positive_mode}`",
        f"Pseudo-label candidate windows: `{candidate_count}`",
        f"Changed current labels: `{changed}`",
        f"Lead-time candidate windows: `{len(lead_rows)}`",
        f"Current chatter-positive windows: `{len(positive_rows)}`",
        "",
        "## Thresholds",
        "",
        "| Label threshold | Score |",
        "|---|---:|",
        f"| transition | {thresholds['transition']:.6g} |",
        f"| slight | {thresholds['slight']:.6g} |",
        f"| severe | {thresholds['severe']:.6g} |",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Score Columns",
            "",
            ", ".join(f"`{column}`" for column in config.score_columns),
            "",
            "## Score Summary",
            "",
            f"- min: `{float(np.min(score_values)):.6g}`",
            f"- median: `{float(np.median(score_values)):.6g}`",
            f"- p95: `{float(np.quantile(score_values, 0.95)):.6g}`",
            f"- max: `{float(np.max(score_values)):.6g}`",
            "",
            "## Caveats",
            "",
            "- These are pseudo-labels, not human or surface-metrology labels.",
            "- Use them to prototype early-warning logic and select segments for manual review.",
            "- Do not report pseudo-label metrics as final real-machine validation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _first_float(rows: list[dict[str, str]], column: str, *, default: float) -> float:
    for row in rows:
        if column in row and row[column] not in {"", None}:
            return _float(row[column], default=default)
    return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def copy_dataset_artifacts(source_dir: Path, out_dir: Path) -> None:
    for name in ("README.md", "manifest.json", "windows.csv", "dataset.npz"):
        source = source_dir / name
        if source.exists() and not (out_dir / name).exists():
            shutil.copy2(source, out_dir / name)
