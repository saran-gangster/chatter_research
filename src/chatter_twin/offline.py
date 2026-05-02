from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chatter_twin.replay import ID_TO_LABEL, LABEL_TO_ID

MODEL_TYPES = ("softmax", "hist_gb")
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
SPLIT_MODES = ("row", "episode", "parameter_family", "time_block")
HOLDOUT_TAILS = ("high", "low")
FEATURE_SETS = ("base", "temporal", "profile", "profile_temporal", "interaction", "interaction_temporal")
TARGETS = ("current", "horizon")
LEAD_TIME_THRESHOLDS = tuple(float(round(value, 2)) for value in np.linspace(0.05, 0.95, 19))
BASE_FEATURE_COLUMNS = (
    "margin_physics",
    "margin_signal",
    "uncertainty",
    "spindle_rpm",
    "feed_per_tooth_m",
    "axial_depth_m",
    "radial_depth_m",
    "tooth_frequency_hz",
    "limiting_frequency_hz",
    "dominant_frequency_hz",
    "rms",
    "chatter_band_energy",
    "tooth_band_energy",
    "non_tooth_harmonic_ratio",
)
PROFILE_ONLY_FEATURE_COLUMNS = (
    "axial_depth_profile_scale",
    "cutting_coeff_profile_scale",
    "cutting_coeff_t_n_m2",
    "cutting_coeff_r_n_m2",
)
INTERACTION_ONLY_FEATURE_COLUMNS = (
    "depth_to_critical_ratio",
    "critical_depth_proxy_m",
    "distance_to_boundary_m",
    "process_force_proxy_n",
    "process_severity_proxy",
    "profile_progress",
    "margin_chatter_energy_interaction",
    "margin_rms_interaction",
    "chatter_growth_pressure",
    "rms_growth_pressure",
)
TEMPORAL_ONLY_FEATURE_COLUMNS = (
    "episode_progress",
    "dominant_frequency_delta_hz",
    "rms_delta",
    "rms_growth_rate",
    "chatter_band_energy_delta",
    "chatter_band_energy_growth_rate",
    "tooth_band_energy_delta",
    "tooth_band_energy_growth_rate",
    "non_tooth_harmonic_ratio_delta",
    "non_tooth_harmonic_ratio_growth_rate",
    "rms_ewma",
    "chatter_band_energy_ewma",
    "non_tooth_harmonic_ratio_ewma",
)
TEMPORAL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TEMPORAL_ONLY_FEATURE_COLUMNS
PROFILE_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PROFILE_ONLY_FEATURE_COLUMNS
PROFILE_TEMPORAL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PROFILE_ONLY_FEATURE_COLUMNS + TEMPORAL_ONLY_FEATURE_COLUMNS
INTERACTION_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + PROFILE_ONLY_FEATURE_COLUMNS + INTERACTION_ONLY_FEATURE_COLUMNS
INTERACTION_TEMPORAL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS + PROFILE_ONLY_FEATURE_COLUMNS + TEMPORAL_ONLY_FEATURE_COLUMNS + INTERACTION_ONLY_FEATURE_COLUMNS
)
DEFAULT_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
FEATURE_COLUMNS_BY_SET = {
    "base": BASE_FEATURE_COLUMNS,
    "temporal": TEMPORAL_FEATURE_COLUMNS,
    "profile": PROFILE_FEATURE_COLUMNS,
    "profile_temporal": PROFILE_TEMPORAL_FEATURE_COLUMNS,
    "interaction": INTERACTION_FEATURE_COLUMNS,
    "interaction_temporal": INTERACTION_TEMPORAL_FEATURE_COLUMNS,
}
LOG_FEATURE_COLUMNS = frozenset(
    {
        "rms",
        "chatter_band_energy",
        "tooth_band_energy",
        "non_tooth_harmonic_ratio",
        "rms_ewma",
        "chatter_band_energy_ewma",
        "non_tooth_harmonic_ratio_ewma",
        "cutting_coeff_t_n_m2",
        "cutting_coeff_r_n_m2",
    }
)
DERIVED_FEATURE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "depth_to_critical_ratio": ("margin_physics",),
    "critical_depth_proxy_m": ("margin_physics", "axial_depth_m"),
    "distance_to_boundary_m": ("margin_physics", "axial_depth_m"),
    "process_force_proxy_n": ("axial_depth_m", "feed_per_tooth_m", "cutting_coeff_t_n_m2", "cutting_coeff_r_n_m2"),
    "process_severity_proxy": ("axial_depth_profile_scale", "cutting_coeff_profile_scale"),
    "profile_progress": ("episode_progress", "axial_depth_profile_scale"),
    "margin_chatter_energy_interaction": ("margin_physics", "chatter_band_energy"),
    "margin_rms_interaction": ("margin_physics", "rms"),
    "chatter_growth_pressure": ("margin_physics", "chatter_band_energy_growth_rate"),
    "rms_growth_pressure": ("margin_physics", "rms_growth_rate"),
}


@dataclass(frozen=True)
class RiskTrainingConfig:
    model_type: str = "softmax"
    calibration: str = "none"
    feature_set: str = "base"
    target: str = "current"
    epochs: int = 800
    learning_rate: float = 0.08
    l2: float = 1.0e-3
    test_fraction: float = 0.30
    validation_fraction: float = 0.0
    seed: int = 303
    split_mode: str = "row"
    holdout_column: str = "axial_depth_scale"
    holdout_tail: str = "high"

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}")
        if self.calibration not in CALIBRATION_METHODS:
            raise ValueError(f"calibration must be one of {CALIBRATION_METHODS}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2 < 0:
            raise ValueError("l2 cannot be negative")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}")
        if self.holdout_tail not in HOLDOUT_TAILS:
            raise ValueError(f"holdout_tail must be one of {HOLDOUT_TAILS}")


@dataclass(frozen=True)
class RiskModel:
    feature_columns: tuple[str, ...]
    label_names: tuple[str, ...]
    mean: list[float]
    scale: list[float]
    weights: list[list[float]]
    bias: list[float]


@dataclass(frozen=True)
class SplitResult:
    train_idx: NDArray[np.int64]
    test_idx: NDArray[np.int64]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ValidationSplitResult:
    fit_idx: NDArray[np.int64]
    validation_idx: NDArray[np.int64]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ModelTrainingResult:
    model_summary: dict[str, object]
    train_probs: NDArray[np.float64]
    test_probs: NDArray[np.float64]
    train_predictions: NDArray[np.int64]
    test_predictions: NDArray[np.int64]
    loss: dict[str, float | None]
    artifacts: tuple[str, ...]


def train_risk_model(
    *,
    dataset_dir: Path,
    out_dir: Path,
    config: RiskTrainingConfig | None = None,
    feature_columns: tuple[str, ...] | None = None,
) -> dict:
    config = config or RiskTrainingConfig()
    feature_columns = feature_columns or feature_columns_for_set(config.feature_set)
    target_id_column = target_id_column_for_mode(config.target)
    records = load_window_records(dataset_dir / "windows.csv")
    if len(records) < 2:
        raise ValueError("Need at least two windows to train/evaluate a risk model")

    x_raw, y = make_feature_matrix(records, feature_columns, target_column=target_id_column)
    split = make_train_test_split(records, y, config)
    train_idx, test_idx = split.train_idx, split.test_idx
    validation_split = make_validation_split(records, train_idx, y, config)
    fit_idx, validation_idx = validation_split.fit_idx, validation_split.validation_idx
    eval_idx = np.concatenate([validation_idx, test_idx]) if validation_idx.size else test_idx
    x_fit_raw = x_raw[fit_idx]
    x_eval_raw = x_raw[eval_idx]
    y_fit = y[fit_idx]
    y_validation = y[validation_idx] if validation_idx.size else np.array([], dtype=np.int64)
    y_test = y[test_idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    if config.model_type == "softmax":
        fitted = train_softmax_model(x_fit_raw, x_eval_raw, y_fit, feature_columns, config)
    else:
        fitted = train_hist_gradient_boosting_model(x_fit_raw, x_eval_raw, y_fit, feature_columns, config, out_dir)

    validation_count = int(validation_idx.size)
    validation_probs = fitted.test_probs[:validation_count] if validation_count else np.zeros((0, len(ID_TO_LABEL)), dtype=float)
    validation_predictions = fitted.test_predictions[:validation_count] if validation_count else np.array([], dtype=np.int64)
    test_probs = fitted.test_probs[validation_count:]
    test_predictions = fitted.test_predictions[validation_count:]

    train_metrics = classification_metrics(y_fit, fitted.train_predictions, fitted.train_probs)
    validation_metrics = classification_metrics(y_validation, validation_predictions, validation_probs) if validation_count else None
    test_metrics = classification_metrics(y_test, test_predictions, test_probs)
    train_lead_time_metrics = lead_time_metrics(records, fit_idx, fitted.train_probs)
    validation_lead_time_metrics = lead_time_metrics(records, validation_idx, validation_probs) if validation_count else None
    test_lead_time_metrics = lead_time_metrics(records, test_idx, test_probs)
    train_lead_time_sweep = lead_time_threshold_sweep(records, fit_idx, fitted.train_probs)
    validation_lead_time_sweep = (
        lead_time_threshold_sweep(records, validation_idx, validation_probs) if validation_count else None
    )
    test_lead_time_sweep = lead_time_threshold_sweep(records, test_idx, test_probs)
    threshold_source = _threshold_selection_source(validation_lead_time_sweep)
    selection_sweep = validation_lead_time_sweep if threshold_source == "validation" else train_lead_time_sweep
    selected_lead_time_threshold = float(selection_sweep["best_f1"]["threshold"])
    test_lead_time_at_selected_threshold = lead_time_metrics(
        records,
        test_idx,
        test_probs,
        threshold=selected_lead_time_threshold,
    )
    train_event_sweep = event_warning_threshold_sweep(records, fit_idx, fitted.train_probs)
    validation_event_sweep = event_warning_threshold_sweep(records, validation_idx, validation_probs) if validation_count else None
    test_event_sweep = event_warning_threshold_sweep(records, test_idx, test_probs)
    event_threshold_source = _threshold_selection_source(validation_event_sweep)
    event_selection_sweep = validation_event_sweep if event_threshold_source == "validation" else train_event_sweep
    selected_event_threshold = float(event_selection_sweep["best_f1"]["threshold"])
    payload = {
        "config": asdict(config),
        "model": {
            "type": config.model_type,
            "calibration": fitted.model_summary.get("calibration"),
        },
        "target": {
            "mode": config.target,
            "id_column": target_id_column,
            "label_column": target_label_column_for_mode(config.target),
        },
        "feature_columns": list(feature_columns),
        "split": split.metadata,
        "validation_split": validation_split.metadata,
        "dataset": {
            "path": str(dataset_dir),
            "windows": len(records),
            "train_windows": int(fit_idx.size),
            "validation_windows": int(validation_idx.size),
            "test_windows": int(test_idx.size),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "lead_time": {
            "score": "predicted_chatter_score",
            "threshold": 0.5,
            "definition": "current stable/transition rows with future slight/severe chatter inside the horizon",
            "train": train_lead_time_metrics,
            "validation": validation_lead_time_metrics,
            "test": test_lead_time_metrics,
            "threshold_selection": {
                "method": f"maximize_{threshold_source}_f1",
                "source": threshold_source,
                "selected_threshold": selected_lead_time_threshold,
                "train_best_f1": train_lead_time_sweep["best_f1"],
                "validation_best_f1": validation_lead_time_sweep["best_f1"] if validation_lead_time_sweep else None,
                "test_at_selected_threshold": test_lead_time_at_selected_threshold,
                "test_oracle_best_f1": test_lead_time_sweep["best_f1"],
            },
            "sweeps": {
                "thresholds": list(LEAD_TIME_THRESHOLDS),
                "train": train_lead_time_sweep["rows"],
                "validation": validation_lead_time_sweep["rows"] if validation_lead_time_sweep else [],
                "test": test_lead_time_sweep["rows"],
            },
        },
        "event_warning": {
            "score": "predicted_chatter_score",
            "threshold": 0.5,
            "definition": "episode-level warning before first current slight/severe chatter window",
            "train": event_warning_metrics(records, fit_idx, fitted.train_probs),
            "validation": event_warning_metrics(records, validation_idx, validation_probs) if validation_count else None,
            "test": event_warning_metrics(records, test_idx, test_probs),
            "threshold_selection": {
                "method": f"maximize_{event_threshold_source}_event_f1",
                "source": event_threshold_source,
                "selected_threshold": selected_event_threshold,
                "train_best_f1": train_event_sweep["best_f1"],
                "validation_best_f1": validation_event_sweep["best_f1"] if validation_event_sweep else None,
                "test_at_selected_threshold": event_warning_metrics(
                    records,
                    test_idx,
                    test_probs,
                    threshold=selected_event_threshold,
                ),
                "test_oracle_best_f1": test_event_sweep["best_f1"],
            },
            "sweeps": {
                "thresholds": list(LEAD_TIME_THRESHOLDS),
                "train": train_event_sweep["rows"],
                "validation": validation_event_sweep["rows"] if validation_event_sweep else [],
                "test": test_event_sweep["rows"],
            },
        },
        "loss": fitted.loss,
        "artifacts": list(fitted.artifacts),
    }

    (out_dir / "model.json").write_text(json.dumps(fitted.model_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_predictions_csv(
        out_dir / "predictions.csv",
        records,
        test_idx,
        test_probs,
        test_predictions,
        target_label_column=target_label_column_for_mode(config.target),
    )
    write_confusion_matrix_csv(out_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"])
    write_report(out_dir / "report.md", payload)
    return payload


def load_window_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_feature_matrix(
    records: list[dict[str, str]],
    feature_columns: tuple[str, ...],
    target_column: str = "label_id",
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if records:
        missing = sorted(
            {
                required
                for column in (*feature_columns, target_column)
                for required in _required_columns_for_feature(column)
                if required not in records[0]
            }
        )
        if missing:
            raise ValueError(
                "windows.csv is missing required columns "
                f"{missing}; regenerate the dataset or choose a compatible --feature-set"
            )
    x = np.zeros((len(records), len(feature_columns)), dtype=float)
    y = np.zeros(len(records), dtype=np.int64)
    for row_idx, record in enumerate(records):
        for col_idx, column in enumerate(feature_columns):
            value = feature_value(record, column)
            if column in LOG_FEATURE_COLUMNS:
                value = np.log10(max(value, 1.0e-18))
            x[row_idx, col_idx] = value
        y[row_idx] = int(record[target_column])
    return x, y


def _required_columns_for_feature(column: str) -> tuple[str, ...]:
    return DERIVED_FEATURE_REQUIREMENTS.get(column, (column,))


def feature_value(record: dict[str, str], column: str) -> float:
    if column == "depth_to_critical_ratio":
        return depth_to_critical_ratio(record)
    if column == "critical_depth_proxy_m":
        ratio = max(depth_to_critical_ratio(record), 1.0e-6)
        return _float(record, "axial_depth_m") / ratio
    if column == "distance_to_boundary_m":
        ratio = max(depth_to_critical_ratio(record), 1.0e-6)
        critical_depth_m = _float(record, "axial_depth_m") / ratio
        return critical_depth_m - _float(record, "axial_depth_m")
    if column == "process_force_proxy_n":
        cutting_coeff = _float(record, "cutting_coeff_t_n_m2") + _float(record, "cutting_coeff_r_n_m2")
        return cutting_coeff * _float(record, "axial_depth_m") * _float(record, "feed_per_tooth_m")
    if column == "process_severity_proxy":
        return _float(record, "axial_depth_profile_scale") * _float(record, "cutting_coeff_profile_scale")
    if column == "profile_progress":
        return _float(record, "episode_progress") * _float(record, "axial_depth_profile_scale")
    if column == "margin_chatter_energy_interaction":
        return depth_to_critical_ratio(record) * _float(record, "chatter_band_energy")
    if column == "margin_rms_interaction":
        return depth_to_critical_ratio(record) * _float(record, "rms")
    if column == "chatter_growth_pressure":
        return depth_to_critical_ratio(record) * _float(record, "chatter_band_energy_growth_rate")
    if column == "rms_growth_pressure":
        return depth_to_critical_ratio(record) * _float(record, "rms_growth_rate")
    return _float(record, column)


def depth_to_critical_ratio(record: dict[str, str]) -> float:
    return 1.0 - _float(record, "margin_physics")


def _float(record: dict[str, str], column: str) -> float:
    return float(record[column])


def feature_columns_for_set(feature_set: str) -> tuple[str, ...]:
    try:
        return FEATURE_COLUMNS_BY_SET[feature_set]
    except KeyError as exc:
        raise ValueError(f"Unknown feature_set {feature_set!r}") from exc


def target_id_column_for_mode(target: str) -> str:
    if target == "current":
        return "label_id"
    if target == "horizon":
        return "horizon_label_id"
    raise ValueError(f"Unknown target {target!r}")


def target_label_column_for_mode(target: str) -> str:
    if target == "current":
        return "label"
    if target == "horizon":
        return "horizon_label"
    raise ValueError(f"Unknown target {target!r}")


def train_test_indices(y: NDArray[np.int64], test_fraction: float, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        if indices.size == 1 or test_fraction == 0:
            train.extend(indices.tolist())
            continue
        n_test = max(1, int(round(indices.size * test_fraction)))
        n_test = min(n_test, indices.size - 1)
        test.extend(indices[:n_test].tolist())
        train.extend(indices[n_test:].tolist())
    if not test:
        test = train.copy()
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(test), dtype=np.int64)


def make_train_test_split(records: list[dict[str, str]], y: NDArray[np.int64], config: RiskTrainingConfig) -> SplitResult:
    if config.split_mode == "row":
        train_idx, test_idx = train_test_indices(y, config.test_fraction, config.seed)
        return SplitResult(
            train_idx=train_idx,
            test_idx=test_idx,
            metadata={
                "mode": "row",
                "test_fraction": config.test_fraction,
                "seed": config.seed,
                "train_groups": None,
                "test_groups": None,
            },
        )

    groups = episode_group_keys(records)
    if config.split_mode == "episode":
        strata = np.array([record.get("scenario", "") for record in records])
        return grouped_train_test_split(
            groups=groups,
            test_fraction=config.test_fraction,
            seed=config.seed,
            mode="episode",
            strata=strata,
        )
    if config.split_mode == "time_block":
        return time_block_split(records=records, groups=groups, test_fraction=config.test_fraction, mode="time_block")

    return parameter_family_split(
        records=records,
        groups=groups,
        test_fraction=config.test_fraction,
        column=config.holdout_column,
        tail=config.holdout_tail,
    )


def make_validation_split(
    records: list[dict[str, str]],
    train_idx: NDArray[np.int64],
    y: NDArray[np.int64],
    config: RiskTrainingConfig,
) -> ValidationSplitResult:
    disabled = ValidationSplitResult(
        fit_idx=train_idx,
        validation_idx=np.array([], dtype=np.int64),
        metadata={
            "enabled": False,
            "validation_fraction": config.validation_fraction,
            "reason": "validation_fraction_zero" if config.validation_fraction == 0 else "not_enough_data",
        },
    )
    if config.validation_fraction <= 0 or train_idx.size < 3:
        return disabled

    local_y = y[train_idx]
    if config.split_mode == "row":
        local_fit, local_validation = train_test_indices(local_y, config.validation_fraction, config.seed + 911)
        metadata: dict[str, object] = {
            "enabled": True,
            "mode": "row",
            "validation_fraction": config.validation_fraction,
            "seed": config.seed + 911,
        }
    elif config.split_mode == "time_block":
        train_records = [records[int(index)] for index in train_idx]
        local_groups = episode_group_keys(train_records)
        local_split = time_block_split(
            records=train_records,
            groups=local_groups,
            test_fraction=config.validation_fraction,
            mode="validation_time_block",
        )
        local_fit, local_validation = local_split.train_idx, local_split.test_idx
        metadata = {
            "enabled": True,
            "mode": "time_block",
            "validation_fraction": config.validation_fraction,
            **local_split.metadata,
        }
    else:
        train_records = [records[int(index)] for index in train_idx]
        local_groups = episode_group_keys(train_records)
        local_strata = np.array([record.get("scenario", "") for record in train_records])
        local_split = grouped_train_test_split(
            groups=local_groups,
            test_fraction=config.validation_fraction,
            seed=config.seed + 911,
            mode="validation",
            strata=local_strata,
        )
        local_fit, local_validation = local_split.train_idx, local_split.test_idx
        metadata = {
            "enabled": True,
            "mode": "episode",
            "validation_fraction": config.validation_fraction,
            "seed": config.seed + 911,
            **local_split.metadata,
        }

    if (
        local_validation.size == 0
        or local_fit.size == 0
        or np.intersect1d(local_fit, local_validation).size
        or np.unique(local_y[local_fit]).size < 2
    ):
        return ValidationSplitResult(
            fit_idx=train_idx,
            validation_idx=np.array([], dtype=np.int64),
            metadata={
                "enabled": False,
                "validation_fraction": config.validation_fraction,
                "reason": "could_not_create_disjoint_fit_validation_split",
            },
        )

    fit_idx = train_idx[local_fit]
    validation_idx = train_idx[local_validation]
    metadata.update(
        {
            "fit_windows": int(fit_idx.size),
            "validation_windows": int(validation_idx.size),
        }
    )
    return ValidationSplitResult(fit_idx=fit_idx, validation_idx=validation_idx, metadata=metadata)


def episode_group_keys(records: list[dict[str, str]]) -> NDArray[np.str_]:
    return np.array([f"{record.get('scenario', '')}::episode={record.get('episode', '')}" for record in records])


def grouped_train_test_split(
    *,
    groups: NDArray[np.str_],
    test_fraction: float,
    seed: int,
    mode: str,
    strata: NDArray[np.str_] | None = None,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    selected_test_groups: list[str] = []
    strata = strata if strata is not None else np.full(groups.shape, "all")

    for stratum in sorted(set(strata.tolist())):
        stratum_groups = np.array(sorted(set(groups[strata == stratum].tolist())))
        if stratum_groups.size <= 1 or test_fraction == 0:
            continue
        rng.shuffle(stratum_groups)
        n_test = max(1, int(round(stratum_groups.size * test_fraction)))
        n_test = min(n_test, stratum_groups.size - 1)
        selected_test_groups.extend(stratum_groups[:n_test].tolist())

    if not selected_test_groups:
        row_count = groups.size
        train_idx = np.arange(row_count, dtype=np.int64)
        test_idx = train_idx.copy()
    else:
        test_mask = np.isin(groups, np.array(selected_test_groups))
        train_idx = np.flatnonzero(~test_mask).astype(np.int64)
        test_idx = np.flatnonzero(test_mask).astype(np.int64)

    train_groups = sorted(set(groups[train_idx].tolist()))
    test_groups = sorted(set(groups[test_idx].tolist()))
    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        metadata={
            "mode": mode,
            "test_fraction": test_fraction,
            "seed": seed,
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "test_groups": test_groups,
        },
    )


def time_block_split(
    *,
    records: list[dict[str, str]],
    groups: NDArray[np.str_],
    test_fraction: float,
    mode: str,
) -> SplitResult:
    train: list[int] = []
    test: list[int] = []
    group_ranges: list[dict[str, object]] = []

    for group in sorted(set(groups.tolist())):
        group_idx = np.flatnonzero(groups == group).astype(np.int64)
        ordered = sorted(group_idx.tolist(), key=lambda idx: (_record_start_time(records[idx]), _record_window_index(records[idx])))
        if len(ordered) <= 1 or test_fraction == 0:
            train.extend(ordered)
            split_at = len(ordered)
        else:
            n_test = max(1, int(round(len(ordered) * test_fraction)))
            n_test = min(n_test, len(ordered) - 1)
            split_at = len(ordered) - n_test
            train.extend(ordered[:split_at])
            test.extend(ordered[split_at:])

        group_ranges.append(
            {
                "group": group,
                "windows": len(ordered),
                "train_windows": split_at,
                "test_windows": len(ordered) - split_at,
                "train_start_s": _index_time_or_none(records, ordered[0]) if split_at else None,
                "train_end_s": _index_time_or_none(records, ordered[split_at - 1]) if split_at else None,
                "test_start_s": _index_time_or_none(records, ordered[split_at]) if split_at < len(ordered) else None,
                "test_end_s": _index_time_or_none(records, ordered[-1]) if split_at < len(ordered) else None,
            }
        )

    if not test:
        test = train.copy()

    train_idx = np.array(sorted(train), dtype=np.int64)
    test_idx = np.array(sorted(test), dtype=np.int64)
    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        metadata={
            "mode": mode,
            "test_fraction": test_fraction,
            "train_group_count": len(set(groups[train_idx].tolist())) if train_idx.size else 0,
            "test_group_count": len(set(groups[test_idx].tolist())) if test_idx.size else 0,
            "test_groups": sorted(set(groups[test_idx].tolist())) if test_idx.size else [],
            "block_policy": "train_early_test_late_within_each_episode",
            "group_time_ranges": group_ranges,
        },
    )


def _record_start_time(record: dict[str, str]) -> float:
    try:
        return float(record.get("start_time_s", "0"))
    except ValueError:
        return 0.0


def _record_window_index(record: dict[str, str]) -> int:
    try:
        return int(float(record.get("window_index_in_episode", "0")))
    except ValueError:
        return 0


def _index_time_or_none(records: list[dict[str, str]], index: int) -> float | None:
    if not records:
        return None
    return _record_start_time(records[index])


def parameter_family_split(
    *,
    records: list[dict[str, str]],
    groups: NDArray[np.str_],
    test_fraction: float,
    column: str,
    tail: str,
) -> SplitResult:
    if not records or column not in records[0]:
        raise ValueError(f"holdout_column {column!r} is not present in windows.csv")

    values = np.array([float(record[column]) for record in records], dtype=float)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) <= 1 or test_fraction == 0:
        train_idx = np.arange(groups.size, dtype=np.int64)
        test_idx = train_idx.copy()
        return SplitResult(
            train_idx=train_idx,
            test_idx=test_idx,
            metadata={
                "mode": "parameter_family",
                "test_fraction": test_fraction,
                "holdout_column": column,
                "holdout_tail": tail,
                "train_group_count": len(unique_groups),
                "test_group_count": len(unique_groups),
                "test_groups": unique_groups,
            },
        )

    group_values = {group: float(np.mean(values[groups == group])) for group in unique_groups}
    ordered_groups = sorted(unique_groups, key=lambda group: group_values[group], reverse=tail == "high")
    target_windows = max(1, int(round(groups.size * test_fraction)))
    selected_test_groups: list[str] = []
    selected_windows = 0
    for group in ordered_groups:
        if len(selected_test_groups) >= len(unique_groups) - 1:
            break
        selected_test_groups.append(group)
        selected_windows += int(np.sum(groups == group))
        if selected_windows >= target_windows:
            break

    test_mask = np.isin(groups, np.array(selected_test_groups))
    train_idx = np.flatnonzero(~test_mask).astype(np.int64)
    test_idx = np.flatnonzero(test_mask).astype(np.int64)
    train_values = values[train_idx]
    test_values = values[test_idx]
    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        metadata={
            "mode": "parameter_family",
            "test_fraction": test_fraction,
            "holdout_column": column,
            "holdout_tail": tail,
            "train_group_count": len(set(groups[train_idx].tolist())),
            "test_group_count": len(set(groups[test_idx].tolist())),
            "train_value_min": float(np.min(train_values)),
            "train_value_max": float(np.max(train_values)),
            "test_value_min": float(np.min(test_values)),
            "test_value_max": float(np.max(test_values)),
            "test_groups": sorted(selected_test_groups),
        },
    )


def train_softmax_model(
    x_train_raw: NDArray[np.float64],
    x_test_raw: NDArray[np.float64],
    y_train: NDArray[np.int64],
    feature_columns: tuple[str, ...],
    config: RiskTrainingConfig,
) -> ModelTrainingResult:
    mean = x_train_raw.mean(axis=0)
    scale = x_train_raw.std(axis=0)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    x_train = (x_train_raw - mean) / scale
    x_test = (x_test_raw - mean) / scale

    weights, bias, losses = fit_softmax(x_train, y_train, config)
    train_probs = predict_proba_array(x_train, weights, bias)
    test_probs = predict_proba_array(x_test, weights, bias)
    model = RiskModel(
        feature_columns=tuple(feature_columns),
        label_names=tuple(ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))),
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=weights.tolist(),
        bias=bias.tolist(),
    )
    model_summary = {
        "model_type": "softmax",
        "target": config.target,
        "feature_columns": list(feature_columns),
        "label_names": [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))],
        "preprocessing": {
            "standardization": "zscore_from_training_split",
            "log10_columns": sorted(LOG_FEATURE_COLUMNS),
        },
        **asdict(model),
    }
    return ModelTrainingResult(
        model_summary=model_summary,
        train_probs=train_probs,
        test_probs=test_probs,
        train_predictions=np.argmax(train_probs, axis=1).astype(np.int64),
        test_predictions=np.argmax(test_probs, axis=1).astype(np.int64),
        loss={"initial": losses[0], "final": losses[-1]},
        artifacts=("model.json", "metrics.json", "predictions.csv", "confusion_matrix.csv", "report.md"),
    )


def train_hist_gradient_boosting_model(
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_train: NDArray[np.int64],
    feature_columns: tuple[str, ...],
    config: RiskTrainingConfig,
    out_dir: Path,
) -> ModelTrainingResult:
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:  # pragma: no cover - exercised only when optional dependency is missing
        raise RuntimeError("The hist_gb model requires scikit-learn. Install project dependencies with uv.") from exc

    observed_classes = np.unique(y_train)
    if observed_classes.size < 2:
        raise ValueError("hist_gb requires at least two labels in the training split")

    base_estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=config.l2,
        early_stopping=False,
        random_state=config.seed,
    )
    sample_weights = _balanced_sample_weights(y_train)
    estimator: Any = base_estimator
    calibration = calibration_plan(y_train, config.calibration)
    if calibration["status"] == "enabled":
        estimator = CalibratedClassifierCV(base_estimator, method=str(calibration["method"]), cv=int(calibration["cv"]))

    _fit_estimator(estimator, x_train, y_train, sample_weights)
    train_probs = align_class_probabilities(estimator.predict_proba(x_train), estimator.classes_)
    test_probs = align_class_probabilities(estimator.predict_proba(x_test), estimator.classes_)
    train_predictions = np.argmax(train_probs, axis=1).astype(np.int64)
    test_predictions = np.argmax(test_probs, axis=1).astype(np.int64)

    bundle = {
        "estimator": estimator,
        "feature_columns": tuple(feature_columns),
        "label_names": tuple(ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))),
        "log10_columns": tuple(sorted(LOG_FEATURE_COLUMNS)),
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    model_summary = {
        "model_type": "hist_gb",
        "target": config.target,
        "feature_columns": list(feature_columns),
        "label_names": [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))],
        "classes_seen": [ID_TO_LABEL[int(label_id)] for label_id in estimator.classes_],
        "serialized_artifact": "model.joblib",
        "preprocessing": {
            "standardization": "none",
            "log10_columns": sorted(LOG_FEATURE_COLUMNS),
        },
        "calibration": calibration,
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 5,
            "l2_regularization": config.l2,
            "early_stopping": False,
        },
    }
    return ModelTrainingResult(
        model_summary=model_summary,
        train_probs=train_probs,
        test_probs=test_probs,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        loss={"initial": None, "final": None},
        artifacts=("model.json", "model.joblib", "metrics.json", "predictions.csv", "confusion_matrix.csv", "report.md"),
    )


def calibration_plan(y_train: NDArray[np.int64], requested: str) -> dict[str, object]:
    if requested == "none":
        return {"requested": requested, "status": "disabled", "method": None, "cv": None, "reason": "not_requested"}
    _, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts)) if counts.size else 0
    if min_count < 2:
        return {
            "requested": requested,
            "status": "skipped",
            "method": requested,
            "cv": None,
            "reason": "fewer_than_two_samples_in_at_least_one_training_class",
        }
    cv = min(3, min_count)
    return {"requested": requested, "status": "enabled", "method": requested, "cv": cv, "reason": None}


def _fit_estimator(estimator: Any, x: NDArray[np.float64], y: NDArray[np.int64], sample_weights: NDArray[np.float64]) -> None:
    try:
        estimator.fit(x, y, sample_weight=sample_weights)
    except TypeError:
        estimator.fit(x, y)


def align_class_probabilities(probs: NDArray[np.float64], classes: NDArray[np.int64]) -> NDArray[np.float64]:
    aligned = np.zeros((probs.shape[0], len(ID_TO_LABEL)), dtype=float)
    for source_col, label_id in enumerate(classes):
        aligned[:, int(label_id)] = probs[:, source_col]
    return aligned


def fit_softmax(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    config: RiskTrainingConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    rng = np.random.default_rng(config.seed)
    n_samples, n_features = x.shape
    n_classes = len(ID_TO_LABEL)
    weights = rng.normal(0.0, 0.01, size=(n_features, n_classes))
    bias = np.zeros(n_classes, dtype=float)
    one_hot = np.eye(n_classes)[y]
    sample_weights = _balanced_sample_weights(y)
    losses: list[float] = []

    for _ in range(config.epochs):
        logits = x @ weights + bias
        probs = _softmax(logits)
        weighted_error = (probs - one_hot) * sample_weights[:, np.newaxis]
        grad_w = (x.T @ weighted_error) / n_samples + config.l2 * weights
        grad_b = weighted_error.mean(axis=0)
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b
        losses.append(_cross_entropy(probs, y, sample_weights) + 0.5 * config.l2 * float(np.sum(weights**2)))

    return weights, bias, losses


def predict_proba(model: RiskModel, rows: list[dict[str, str]]) -> NDArray[np.float64]:
    x_raw, _ = make_feature_matrix(rows, model.feature_columns)
    mean = np.array(model.mean, dtype=float)
    scale = np.array(model.scale, dtype=float)
    x = (x_raw - mean) / scale
    return predict_proba_array(x, np.array(model.weights, dtype=float), np.array(model.bias, dtype=float))


def predict_proba_array(x: NDArray[np.float64], weights: NDArray[np.float64], bias: NDArray[np.float64]) -> NDArray[np.float64]:
    return _softmax(x @ weights + bias)


def classification_metrics(y_true: NDArray[np.int64], y_pred: NDArray[np.int64], probs: NDArray[np.float64]) -> dict:
    confusion = np.zeros((len(ID_TO_LABEL), len(ID_TO_LABEL)), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=False):
        confusion[int(truth), int(pred)] += 1

    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    f1_scores: list[float] = []
    for label_id in range(len(ID_TO_LABEL)):
        support = int(np.sum(y_true == label_id))
        if support == 0:
            continue
        tp = confusion[label_id, label_id]
        fp = int(np.sum(confusion[:, label_id]) - tp)
        fn = int(np.sum(confusion[label_id, :]) - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_scores.append(2.0 * precision * recall / max(precision + recall, 1.0e-12))

    binary_truth = y_true >= LABEL_TO_ID["slight"]
    binary_score = probs[:, LABEL_TO_ID["slight"]] + probs[:, LABEL_TO_ID["severe"]]
    binary_pred = binary_score >= 0.5
    binary_tp = int(np.sum(binary_truth & binary_pred))
    binary_fp = int(np.sum(~binary_truth & binary_pred))
    binary_fn = int(np.sum(binary_truth & ~binary_pred))
    binary_precision = binary_tp / max(binary_tp + binary_fp, 1)
    binary_recall = binary_tp / max(binary_tp + binary_fn, 1)
    binary_f1 = 2.0 * binary_precision * binary_recall / max(binary_precision + binary_recall, 1.0e-12)
    brier = float(np.mean((binary_score - binary_truth.astype(float)) ** 2)) if y_true.size else 0.0

    intervention_truth = (y_true >= LABEL_TO_ID["transition"]) & (y_true <= LABEL_TO_ID["severe"])
    intervention_score = probs[:, LABEL_TO_ID["transition"]] + probs[:, LABEL_TO_ID["slight"]] + probs[:, LABEL_TO_ID["severe"]]
    intervention_pred = intervention_score >= 0.5
    intervention_tp = int(np.sum(intervention_truth & intervention_pred))
    intervention_fp = int(np.sum(~intervention_truth & intervention_pred))
    intervention_fn = int(np.sum(intervention_truth & ~intervention_pred))
    intervention_precision = intervention_tp / max(intervention_tp + intervention_fp, 1)
    intervention_recall = intervention_tp / max(intervention_tp + intervention_fn, 1)
    intervention_f1 = 2.0 * intervention_precision * intervention_recall / max(intervention_precision + intervention_recall, 1.0e-12)
    intervention_brier = (
        float(np.mean((intervention_score - intervention_truth.astype(float)) ** 2)) if y_true.size else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "binary_chatter_precision": binary_precision,
        "binary_chatter_recall": binary_recall,
        "binary_chatter_f1": binary_f1,
        "binary_chatter_brier": brier,
        "intervention_precision": intervention_precision,
        "intervention_recall": intervention_recall,
        "intervention_f1": intervention_f1,
        "intervention_brier": intervention_brier,
        "confusion_matrix": confusion.tolist(),
    }


def lead_time_metrics(
    records: list[dict[str, str]],
    indices: NDArray[np.int64],
    probs: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    if indices.size == 0:
        return _empty_lead_time_metrics()

    selected_records = [records[int(index)] for index in indices]
    current_label_ids = np.array([int(record.get("label_id", LABEL_TO_ID["unknown"])) for record in selected_records], dtype=int)
    candidate_mask = current_label_ids < LABEL_TO_ID["slight"]
    truth = np.array([_csv_bool(record.get("future_chatter_within_horizon", "False")) for record in selected_records], dtype=bool)
    truth &= candidate_mask
    score = probs[:, LABEL_TO_ID["slight"]] + probs[:, LABEL_TO_ID["severe"]]
    pred = (score >= threshold) & candidate_mask

    candidate_count = int(np.sum(candidate_mask))
    positive_count = int(np.sum(truth))
    if candidate_count == 0:
        return _empty_lead_time_metrics()

    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred & candidate_mask))
    fn = int(np.sum(truth & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    brier = float(np.mean((score[candidate_mask] - truth[candidate_mask].astype(float)) ** 2))
    lead_times = np.array(
        [
            float(record.get("time_to_chatter_s", "-1.0"))
            for record, is_positive in zip(selected_records, truth, strict=False)
            if is_positive
        ],
        dtype=float,
    )
    detected_lead_times = np.array(
        [
            float(record.get("time_to_chatter_s", "-1.0"))
            for record, is_tp in zip(selected_records, truth & pred, strict=False)
            if is_tp
        ],
        dtype=float,
    )

    return {
        "candidate_windows": candidate_count,
        "positive_windows": positive_count,
        "warning_windows": int(np.sum(pred)),
        "true_positive_windows": tp,
        "false_positive_windows": fp,
        "false_negative_windows": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "mean_time_to_chatter_s": _mean_or_none(lead_times),
        "mean_detected_time_to_chatter_s": _mean_or_none(detected_lead_times),
    }


def lead_time_threshold_sweep(
    records: list[dict[str, str]],
    indices: NDArray[np.int64],
    probs: NDArray[np.float64],
    thresholds: tuple[float, ...] = LEAD_TIME_THRESHOLDS,
) -> dict[str, object]:
    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        metrics = lead_time_metrics(records, indices, probs, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "candidate_windows": int(metrics["candidate_windows"] or 0),
                "positive_windows": int(metrics["positive_windows"] or 0),
                "warning_windows": int(metrics["warning_windows"] or 0),
                "precision": float(metrics["precision"] or 0.0),
                "recall": float(metrics["recall"] or 0.0),
                "f1": float(metrics["f1"] or 0.0),
                "mean_detected_time_to_chatter_s": metrics["mean_detected_time_to_chatter_s"],
            }
        )

    if not rows:
        best = {
            "threshold": 0.5,
            "candidate_windows": 0,
            "positive_windows": 0,
            "warning_windows": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_detected_time_to_chatter_s": None,
        }
    else:
        best = max(
            rows,
            key=lambda row: (
                float(row["f1"] or 0.0),
                float(row["recall"] or 0.0),
                float(row["precision"] or 0.0),
                -float(row["threshold"] or 0.0),
            ),
        )
    return {"rows": rows, "best_f1": best}


def event_warning_metrics(
    records: list[dict[str, str]],
    indices: NDArray[np.int64],
    probs: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    if indices.size == 0:
        return _empty_event_warning_metrics()

    groups: dict[str, list[tuple[dict[str, str], float]]] = {}
    scores = probs[:, LABEL_TO_ID["slight"]] + probs[:, LABEL_TO_ID["severe"]]
    for record_idx, score in zip(indices, scores, strict=False):
        record = records[int(record_idx)]
        groups.setdefault(_event_group_key(record), []).append((record, float(score)))

    event_count = 0
    quiet_count = 0
    detected_events = 0
    missed_events = 0
    false_warning_events = 0
    true_quiet_events = 0
    warning_event_count = 0
    lead_times: list[float] = []

    for rows in groups.values():
        rows = sorted(rows, key=lambda item: _float(item[0], "start_time_s"))
        chatter_rows = [item for item in rows if _label_id(item[0]) >= LABEL_TO_ID["slight"]]
        first_chatter_time = _float(chatter_rows[0][0], "start_time_s") if chatter_rows else None
        if first_chatter_time is None:
            quiet_count += 1
            warning_rows = [
                item
                for item in rows
                if _label_id(item[0]) < LABEL_TO_ID["slight"] and item[1] >= threshold
            ]
            if warning_rows:
                false_warning_events += 1
                warning_event_count += 1
            else:
                true_quiet_events += 1
            continue

        event_count += 1
        warning_rows = [
            item
            for item in rows
            if _label_id(item[0]) < LABEL_TO_ID["slight"]
            and _float(item[0], "start_time_s") < first_chatter_time
            and item[1] >= threshold
        ]
        if warning_rows:
            warning_event_count += 1
            detected_events += 1
            first_warning_time = _float(warning_rows[0][0], "start_time_s")
            lead_times.append(max(0.0, first_chatter_time - first_warning_time))
        else:
            missed_events += 1

    precision = detected_events / max(detected_events + false_warning_events, 1)
    recall = detected_events / max(event_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "episode_count": len(groups),
        "event_episodes": event_count,
        "quiet_episodes": quiet_count,
        "warning_episodes": warning_event_count,
        "detected_event_episodes": detected_events,
        "missed_event_episodes": missed_events,
        "false_warning_episodes": false_warning_events,
        "true_quiet_episodes": true_quiet_events,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_detected_lead_time_s": _mean_or_none(np.array(lead_times, dtype=float)),
    }


def event_warning_threshold_sweep(
    records: list[dict[str, str]],
    indices: NDArray[np.int64],
    probs: NDArray[np.float64],
    thresholds: tuple[float, ...] = LEAD_TIME_THRESHOLDS,
) -> dict[str, object]:
    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        metrics = event_warning_metrics(records, indices, probs, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "episode_count": int(metrics["episode_count"] or 0),
                "event_episodes": int(metrics["event_episodes"] or 0),
                "warning_episodes": int(metrics["warning_episodes"] or 0),
                "precision": float(metrics["precision"] or 0.0),
                "recall": float(metrics["recall"] or 0.0),
                "f1": float(metrics["f1"] or 0.0),
                "mean_detected_lead_time_s": metrics["mean_detected_lead_time_s"],
            }
        )

    if not rows:
        best = {
            "threshold": 0.5,
            "episode_count": 0,
            "event_episodes": 0,
            "warning_episodes": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_detected_lead_time_s": None,
        }
    else:
        best = max(
            rows,
            key=lambda row: (
                float(row["f1"] or 0.0),
                float(row["recall"] or 0.0),
                float(row["precision"] or 0.0),
                -float(row["threshold"] or 0.0),
            ),
        )
    return {"rows": rows, "best_f1": best}


def _empty_event_warning_metrics() -> dict[str, float | int | None]:
    return {
        "episode_count": 0,
        "event_episodes": 0,
        "quiet_episodes": 0,
        "warning_episodes": 0,
        "detected_event_episodes": 0,
        "missed_event_episodes": 0,
        "false_warning_episodes": 0,
        "true_quiet_episodes": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_detected_lead_time_s": None,
    }


def _event_group_key(record: dict[str, str]) -> str:
    return f"{record.get('scenario', '')}::episode={record.get('episode', '')}"


def _label_id(record: dict[str, str]) -> int:
    return int(record.get("label_id", LABEL_TO_ID["unknown"]))


def _threshold_selection_source(validation_sweep: dict[str, object] | None) -> str:
    if validation_sweep is None:
        return "train"
    best = validation_sweep["best_f1"]
    if isinstance(best, dict) and (
        int(best.get("positive_windows") or 0) > 0 or int(best.get("event_episodes") or 0) > 0
    ):
        return "validation"
    return "train"


def _empty_lead_time_metrics() -> dict[str, float | int | None]:
    return {
        "candidate_windows": 0,
        "positive_windows": 0,
        "warning_windows": 0,
        "true_positive_windows": 0,
        "false_positive_windows": 0,
        "false_negative_windows": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "brier": 0.0,
        "mean_time_to_chatter_s": None,
        "mean_detected_time_to_chatter_s": None,
    }


def _csv_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _mean_or_none(values: NDArray[np.float64]) -> float | None:
    valid = values[values >= 0.0]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def write_predictions_csv(
    path: Path,
    records: list[dict[str, str]],
    indices: NDArray[np.int64],
    probs: NDArray[np.float64],
    predictions: NDArray[np.int64],
    target_label_column: str = "label",
) -> None:
    fieldnames = [
        "window_id",
        "scenario",
        "episode",
        "start_time_s",
        "label_id",
        "label",
        "horizon_label",
        "target_label",
        "predicted_label",
        "predicted_chatter_score",
        "risk_chatter_now",
        "future_chatter_within_horizon",
        "time_to_chatter_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, pred, prob in zip(indices, predictions, probs, strict=False):
            record = records[int(row_idx)]
            writer.writerow(
                {
                    "window_id": record["window_id"],
                    "scenario": record["scenario"],
                    "episode": record["episode"],
                    "start_time_s": record.get("start_time_s", ""),
                    "label_id": record.get("label_id", ""),
                    "label": record["label"],
                    "horizon_label": record.get("horizon_label", record["label"]),
                    "target_label": record.get(target_label_column, record["label"]),
                    "predicted_label": ID_TO_LABEL[int(pred)],
                    "predicted_chatter_score": float(prob[LABEL_TO_ID["slight"]] + prob[LABEL_TO_ID["severe"]]),
                    "risk_chatter_now": record["risk_chatter_now"],
                    "future_chatter_within_horizon": record.get("future_chatter_within_horizon", "False"),
                    "time_to_chatter_s": record.get("time_to_chatter_s", "-1.0"),
                }
            )


def write_confusion_matrix_csv(path: Path, matrix: list[list[int]]) -> None:
    labels = [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth/pred", *labels])
        for label, row in zip(labels, matrix, strict=False):
            writer.writerow([label, *row])


def write_report(path: Path, payload: dict) -> None:
    split_meta = payload["split"]
    model_meta = payload["model"]
    lines = [
        "# Offline Risk Model Report",
        "",
        f"Dataset windows: `{payload['dataset']['windows']}`",
        f"Train/validation/test: `{payload['dataset']['train_windows']}` / "
        f"`{payload['dataset']['validation_windows']}` / `{payload['dataset']['test_windows']}`",
        f"Features: `{len(payload['feature_columns'])}`",
        f"Model: `{model_meta['type']}`",
        f"Target: `{payload['target']['mode']}`",
        f"Split: `{split_meta['mode']}`",
        "",
        "| Split | Accuracy | Macro F1 | Chatter F1 | Intervention F1 | Chatter Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    scored_splits = ("train", "validation", "test") if payload.get("validation") else ("train", "test")
    for split_name in scored_splits:
        metrics = payload[split_name]
        lines.append(
            f"| {split_name} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | "
            f"{metrics['binary_chatter_f1']:.3f} | {metrics['intervention_f1']:.3f} | "
            f"{metrics['binary_chatter_brier']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lead-Time Early Warning",
            "",
            "Only current `stable`/`transition` windows are scored here. A positive means a future window "
            "within the replay horizon reaches `slight` or `severe`.",
            "",
            "| Split | Candidates | Positives | Warnings | Precision | Recall | F1 | Mean Detected Lead Time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in scored_splits:
        metrics = payload["lead_time"][split_name]
        detected_lead = metrics["mean_detected_time_to_chatter_s"]
        detected_text = "n/a" if detected_lead is None else f"{detected_lead:.3f}s"
        lines.append(
            f"| {split_name} | {metrics['candidate_windows']} | {metrics['positive_windows']} | "
            f"{metrics['warning_windows']} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {detected_text} |"
        )
    selected = payload["lead_time"]["threshold_selection"]
    tuned_metrics = selected["test_at_selected_threshold"]
    oracle_metrics = selected["test_oracle_best_f1"]
    tuned_lead = tuned_metrics["mean_detected_time_to_chatter_s"]
    oracle_lead = oracle_metrics["mean_detected_time_to_chatter_s"]
    tuned_lead_text = "n/a" if tuned_lead is None else f"{tuned_lead:.3f}s"
    oracle_lead_text = "n/a" if oracle_lead is None else f"{oracle_lead:.3f}s"
    lines.extend(
        [
            "",
            "| Threshold Mode | Threshold | Precision | Recall | F1 | Mean Detected Lead Time |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {selected['source']}-selected | {selected['selected_threshold']:.2f} | "
            f"{tuned_metrics['precision']:.3f} | {tuned_metrics['recall']:.3f} | "
            f"{tuned_metrics['f1']:.3f} | {tuned_lead_text} |",
            f"| test oracle diagnostic | {oracle_metrics['threshold']:.2f} | "
            f"{oracle_metrics['precision']:.3f} | {oracle_metrics['recall']:.3f} | "
            f"{oracle_metrics['f1']:.3f} | {oracle_lead_text} |",
        ]
    )
    event_selected = payload["event_warning"]["threshold_selection"]
    event_tuned = event_selected["test_at_selected_threshold"]
    event_oracle = event_selected["test_oracle_best_f1"]
    lines.extend(
        [
            "",
            "## Event-Level Warning",
            "",
            "Each scenario/episode is scored once. An event is detected only if a warning appears before the "
            "first current-window `slight` or `severe` label in that episode.",
            "",
            "| Split | Episodes | Event Episodes | Warning Episodes | Precision | Recall | F1 | Mean Lead Time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in scored_splits:
        metrics = payload["event_warning"][split_name]
        lead = metrics["mean_detected_lead_time_s"]
        lead_text = "n/a" if lead is None else f"{lead:.3f}s"
        lines.append(
            f"| {split_name} | {metrics['episode_count']} | {metrics['event_episodes']} | "
            f"{metrics['warning_episodes']} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {lead_text} |"
        )
    event_tuned_lead = event_tuned["mean_detected_lead_time_s"]
    event_oracle_lead = event_oracle["mean_detected_lead_time_s"]
    event_tuned_lead_text = "n/a" if event_tuned_lead is None else f"{event_tuned_lead:.3f}s"
    event_oracle_lead_text = "n/a" if event_oracle_lead is None else f"{event_oracle_lead:.3f}s"
    lines.extend(
        [
            "",
            "| Event Threshold Mode | Threshold | Precision | Recall | F1 | Mean Lead Time |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {event_selected['source']}-selected | {event_selected['selected_threshold']:.2f} | "
            f"{event_tuned['precision']:.3f} | {event_tuned['recall']:.3f} | "
            f"{event_tuned['f1']:.3f} | {event_tuned_lead_text} |",
            f"| test oracle diagnostic | {event_oracle['threshold']:.2f} | "
            f"{event_oracle['precision']:.3f} | {event_oracle['recall']:.3f} | "
            f"{event_oracle['f1']:.3f} | {event_oracle_lead_text} |",
        ]
    )
    lines.append("")
    if payload["loss"]["initial"] is None:
        lines.append("Initial/final loss: `not reported for this model type`")
    else:
        lines.append(f"Initial/final loss: `{payload['loss']['initial']:.4f}` / `{payload['loss']['final']:.4f}`")
    if model_meta.get("calibration"):
        calibration = model_meta["calibration"]
        lines.append(f"Calibration: `{calibration['status']}`")
    lines.extend(
        [
            "",
            "The model is trained on replay-window metadata/features, not raw waveform CNN/GRU inputs.",
            "Use episode, time-block, or parameter-family splits when reporting generalization.",
            "",
        ]
    )
    if split_meta["mode"] == "parameter_family":
        lines.insert(
            8,
            f"Holdout: `{split_meta['holdout_tail']}` `{split_meta['holdout_column']}` "
            f"(test range `{split_meta['test_value_min']:.4g}`-`{split_meta['test_value_max']:.4g}`)",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _balanced_sample_weights(y: NDArray[np.int64]) -> NDArray[np.float64]:
    weights = np.ones(y.size, dtype=float)
    for label in np.unique(y):
        count = max(int(np.sum(y == label)), 1)
        weights[y == label] = y.size / (len(np.unique(y)) * count)
    return weights


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy(probs: NDArray[np.float64], y: NDArray[np.int64], sample_weights: NDArray[np.float64]) -> float:
    selected = probs[np.arange(y.size), y]
    return float(np.mean(-np.log(selected + 1.0e-12) * sample_weights))
