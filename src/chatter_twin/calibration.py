from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.models import SignalFeatures
from chatter_twin.scenarios import make_scenario
from chatter_twin.stability import signed_stability_margin


CALIBRATION_MODELS = ("raw", "context")
CONTEXT_FEATURE_NAMES = (
    "raw_margin",
    "log_spindle_rpm",
    "log_tooth_frequency_hz",
    "log_axial_depth_m",
    "log_radial_depth_m",
    "log_feed_per_tooth_m",
    "log_cutting_coeff_t_n_m2",
    "log_cutting_coeff_r_n_m2",
    "log_modal_frequency_hz",
    "modal_damping_ratio",
    "stiffness_ratio_xy",
    "radial_immersion_ratio",
    "runout_to_feed_ratio",
)
DEFAULT_AXIAL_DEPTH_SCALES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
DEFAULT_SPINDLE_SCALES = (0.86, 0.90, 0.94, 0.98, 1.02, 1.06, 1.10, 1.14)
DEFAULT_STIFFNESS_SCALE = (0.75, 1.25)
DEFAULT_DAMPING_SCALE = (0.70, 1.30)
DEFAULT_CUTTING_COEFF_SCALE = (0.75, 1.35)
DEFAULT_RADIAL_DEPTH_SCALE = (0.70, 1.25)
DEFAULT_FEED_SCALE = (0.80, 1.20)
DEFAULT_RUNOUT_SCALE = (0.50, 2.00)


@dataclass(frozen=True)
class CalibrationFamily:
    family: int
    stiffness_x_scale: float = 1.0
    stiffness_y_scale: float = 1.0
    damping_x_scale: float = 1.0
    damping_y_scale: float = 1.0
    cutting_coeff_t_scale: float = 1.0
    cutting_coeff_r_scale: float = 1.0
    radial_depth_scale: float = 1.0
    feed_scale: float = 1.0
    runout_scale: float = 1.0


@dataclass(frozen=True)
class MarginCalibration:
    """Map raw physics margin to simulator-calibrated chatter probability."""

    intercept: float
    slope: float
    margin_uncertainty: float = 0.20
    target_threshold: float = 0.50
    sample_count: int = 0
    positive_count: int = 0
    source: str = "time_domain_signal"
    model_type: str = "raw"
    feature_names: tuple[str, ...] = ("raw_margin",)
    coefficients: tuple[float, ...] = ()
    feature_means: tuple[float, ...] = ()
    feature_stds: tuple[float, ...] = ()
    train_brier_score: float | None = None
    holdout_brier_score: float | None = None
    uncertainty_floor: float = 0.05

    def physics_risk(self, raw_margin: float, modal: Any = None, tool: Any = None, cut: Any = None, row: dict[str, Any] | None = None) -> float:
        return _sigmoid(self.logit(raw_margin, modal=modal, tool=tool, cut=cut, row=row))

    def calibrated_margin(self, raw_margin: float, modal: Any = None, tool: Any = None, cut: Any = None, row: dict[str, Any] | None = None) -> float:
        probability = min(max(self.physics_risk(raw_margin, modal=modal, tool=tool, cut=cut, row=row), 1.0e-6), 1.0 - 1.0e-6)
        logit = math.log(probability / (1.0 - probability))
        return float(-self.margin_uncertainty * logit)

    def uncertainty(self, raw_margin: float, modal: Any = None, tool: Any = None, cut: Any = None, row: dict[str, Any] | None = None) -> float:
        probability = self.physics_risk(raw_margin, modal=modal, tool=tool, cut=cut, row=row)
        boundary = 4.0 * probability * (1.0 - probability)
        brier = self.holdout_brier_score if self.holdout_brier_score is not None else self.train_brier_score
        calibration_penalty = 0.0 if brier is None else min(0.35, brier)
        return float(min(0.95, max(self.uncertainty_floor, 0.08 + 0.35 * boundary + calibration_penalty)))

    def logit(self, raw_margin: float, modal: Any = None, tool: Any = None, cut: Any = None, row: dict[str, Any] | None = None) -> float:
        if self.model_type == "context" and self.coefficients:
            means = self.feature_means or tuple(0.0 for _ in self.feature_names)
            stds = self.feature_stds or tuple(1.0 for _ in self.feature_names)
            if row is None and modal is None and tool is None and cut is None:
                values = tuple(raw_margin if name == "raw_margin" else means[idx] for idx, name in enumerate(self.feature_names))
            else:
                values = context_feature_values(raw_margin, modal=modal, tool=tool, cut=cut, row=row, feature_names=self.feature_names)
            normalized = [(value - mean) / max(std, 1.0e-12) for value, mean, std in zip(values, means, stds, strict=True)]
            return float(self.intercept + sum(coef * value for coef, value in zip(self.coefficients, normalized, strict=True)))
        return float(self.intercept + self.slope * raw_margin)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarginCalibration:
        model_type = str(payload.get("model_type", "raw"))
        feature_names = tuple(str(value) for value in payload.get("feature_names", ("raw_margin",)))
        coefficients = tuple(float(value) for value in payload.get("coefficients", ()))
        feature_means = tuple(float(value) for value in payload.get("feature_means", ()))
        feature_stds = tuple(float(value) for value in payload.get("feature_stds", ()))
        return cls(
            intercept=float(payload["intercept"]),
            slope=float(payload["slope"]),
            margin_uncertainty=float(payload.get("margin_uncertainty", 0.20)),
            target_threshold=float(payload.get("target_threshold", 0.50)),
            sample_count=int(payload.get("sample_count", 0)),
            positive_count=int(payload.get("positive_count", 0)),
            source=str(payload.get("source", "time_domain_signal")),
            model_type=model_type,
            feature_names=feature_names,
            coefficients=coefficients,
            feature_means=feature_means,
            feature_stds=feature_stds,
            train_brier_score=_optional_float(payload.get("train_brier_score")),
            holdout_brier_score=_optional_float(payload.get("holdout_brier_score")),
            uncertainty_floor=float(payload.get("uncertainty_floor", 0.05)),
        )


def apply_margin_calibration(raw_margin: float, calibration: MarginCalibration | None, modal: Any = None, tool: Any = None, cut: Any = None) -> float:
    if calibration is None:
        return raw_margin
    return calibration.calibrated_margin(raw_margin, modal=modal, tool=tool, cut=cut)


def calibrated_margin_uncertainty(raw_margin: float, calibration: MarginCalibration | None, modal: Any = None, tool: Any = None, cut: Any = None) -> float | None:
    if calibration is None:
        return None
    return calibration.uncertainty(raw_margin, modal=modal, tool=tool, cut=cut)


def load_margin_calibration(path: Path) -> MarginCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("calibration", payload)
    return MarginCalibration.from_dict(calibration)


def calibrate_margin_surrogate(
    *,
    scenarios: list[str],
    out_dir: Path,
    axial_depth_scales: list[float] | tuple[float, ...] = DEFAULT_AXIAL_DEPTH_SCALES,
    spindle_scales: list[float] | tuple[float, ...] = DEFAULT_SPINDLE_SCALES,
    duration_s: float = 0.25,
    sample_rate_hz: float | None = None,
    sensor_noise_std: float = 0.0,
    target_threshold: float = 0.50,
    calibration_model: str = "raw",
    seed: int = 909,
    family_count: int = 1,
    holdout_family: int | None = None,
    stiffness_scale: tuple[float, float] = DEFAULT_STIFFNESS_SCALE,
    damping_scale: tuple[float, float] = DEFAULT_DAMPING_SCALE,
    cutting_coeff_scale: tuple[float, float] = DEFAULT_CUTTING_COEFF_SCALE,
    radial_depth_scale: tuple[float, float] = DEFAULT_RADIAL_DEPTH_SCALE,
    feed_scale: tuple[float, float] = DEFAULT_FEED_SCALE,
    runout_scale: tuple[float, float] = DEFAULT_RUNOUT_SCALE,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("scenarios must contain at least one scenario")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not axial_depth_scales or not spindle_scales:
        raise ValueError("axial_depth_scales and spindle_scales must be non-empty")
    if family_count < 1:
        raise ValueError("family_count must be at least 1")
    if holdout_family is not None and not 0 <= holdout_family < family_count:
        raise ValueError("holdout_family must be between 0 and family_count - 1")
    if calibration_model not in CALIBRATION_MODELS:
        raise ValueError(f"calibration_model must be one of {CALIBRATION_MODELS}")

    out_dir.mkdir(parents=True, exist_ok=True)
    families = _make_calibration_families(
        family_count=family_count,
        seed=seed,
        stiffness_scale=stiffness_scale,
        damping_scale=damping_scale,
        cutting_coeff_scale=cutting_coeff_scale,
        radial_depth_scale=radial_depth_scale,
        feed_scale=feed_scale,
        runout_scale=runout_scale,
    )
    rows = _sample_margin_grid(
        scenarios=scenarios,
        axial_depth_scales=tuple(float(value) for value in axial_depth_scales),
        spindle_scales=tuple(float(value) for value in spindle_scales),
        families=families,
        holdout_family=holdout_family,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
        target_threshold=target_threshold,
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    calibration, metrics = _fit_margin_calibration(
        train_rows,
        target_threshold=target_threshold,
        calibration_model=calibration_model,
        all_rows=rows,
        holdout_rows=holdout_rows,
    )
    payload = {
        "calibration": calibration.to_dict(),
        "metrics": metrics,
        "families": [asdict(family) for family in families],
        "config": {
            "scenarios": scenarios,
            "axial_depth_scales": list(axial_depth_scales),
            "spindle_scales": list(spindle_scales),
            "family_count": family_count,
            "holdout_family": holdout_family,
            "stiffness_scale": list(stiffness_scale),
            "damping_scale": list(damping_scale),
            "cutting_coeff_scale": list(cutting_coeff_scale),
            "radial_depth_scale": list(radial_depth_scale),
            "feed_scale": list(feed_scale),
            "runout_scale": list(runout_scale),
            "duration_s": duration_s,
            "sample_rate_hz": sample_rate_hz,
            "sensor_noise_std": sensor_noise_std,
            "target_threshold": target_threshold,
            "calibration_model": calibration_model,
            "seed": seed,
        },
        "artifacts": ["samples.csv", "calibration.json", "report.md", "margin_calibration.svg"],
    }
    _write_csv(out_dir / "samples.csv", rows)
    _write_json(out_dir / "calibration.json", payload)
    _write_report(out_dir / "report.md", payload)
    _write_svg(out_dir / "margin_calibration.svg", rows, calibration)
    return payload


def time_domain_chatter_risk(features: SignalFeatures, sensor_signal: np.ndarray) -> float:
    """Signal-only target used to calibrate physics margin without circularity."""

    chatter_ratio = features.chatter_band_energy / max(features.tooth_band_energy, 1.0e-18)
    ratio_score = _sigmoid((math.log10(chatter_ratio + 1.0e-12) + 0.4) * 2.0)
    crest_score = _sigmoid((features.crest_factor - 4.0) / 1.5)
    entropy_score = _sigmoid((features.spectral_entropy - 0.55) * 6.0)
    signal_score = min(1.0, 0.55 * ratio_score + 0.25 * crest_score + 0.20 * entropy_score)

    magnitude = np.linalg.norm(sensor_signal, axis=1)
    third = max(2, magnitude.size // 3)
    head_rms = float(np.sqrt(np.mean(magnitude[:third] ** 2)))
    tail_rms = float(np.sqrt(np.mean(magnitude[-third:] ** 2)))
    growth_score = _sigmoid((math.log10((tail_rms + 1.0e-12) / (head_rms + 1.0e-12)) + 0.05) * 3.0)
    return float(min(1.0, max(0.0, 0.65 * signal_score + 0.35 * growth_score)))


def context_feature_values(
    raw_margin: float,
    *,
    modal: Any = None,
    tool: Any = None,
    cut: Any = None,
    row: dict[str, Any] | None = None,
    feature_names: tuple[str, ...] = CONTEXT_FEATURE_NAMES,
) -> tuple[float, ...]:
    source = _context_source(raw_margin, modal=modal, tool=tool, cut=cut, row=row)
    return tuple(float(source.get(name, 0.0)) for name in feature_names)


def _context_source(raw_margin: float, *, modal: Any = None, tool: Any = None, cut: Any = None, row: dict[str, Any] | None = None) -> dict[str, float]:
    if row is not None:
        spindle_rpm = _row_float(row, "spindle_rpm")
        flute_count = _row_float(row, "flute_count", 4.0)
        tool_diameter_m = _row_float(row, "tool_diameter_m", 0.010)
        feed_per_tooth_m = _row_float(row, "feed_per_tooth_m", 45.0e-6)
        radial_depth_m = _row_float(row, "radial_depth_m")
        runout_m = _row_float(row, "runout_m", 1.0e-6)
        stiffness_x = _row_float(row, "stiffness_x_n_m", 1.55e7)
        stiffness_y = _row_float(row, "stiffness_y_n_m", 1.25e7)
        values = {
            "raw_margin": float(raw_margin),
            "spindle_rpm": spindle_rpm,
            "tooth_frequency_hz": _row_float(row, "tooth_frequency_hz", spindle_rpm * flute_count / 60.0),
            "axial_depth_m": _row_float(row, "axial_depth_m"),
            "radial_depth_m": radial_depth_m,
            "feed_per_tooth_m": feed_per_tooth_m,
            "cutting_coeff_t_n_m2": _row_float(row, "cutting_coeff_t_n_m2", 7.0e8),
            "cutting_coeff_r_n_m2": _row_float(row, "cutting_coeff_r_n_m2", 2.1e8),
            "modal_frequency_hz": _row_float(row, "modal_frequency_hz", 700.0),
            "modal_damping_ratio": _row_float(row, "modal_damping_ratio", 0.03),
            "stiffness_ratio_xy": _row_float(row, "stiffness_ratio_xy", stiffness_x / max(stiffness_y, 1.0e-12)),
            "radial_immersion_ratio": _row_float(row, "radial_immersion_ratio", radial_depth_m / max(tool_diameter_m, 1.0e-12)),
            "runout_to_feed_ratio": _row_float(row, "runout_to_feed_ratio", runout_m / max(feed_per_tooth_m, 1.0e-12)),
        }
    elif modal is not None and tool is not None and cut is not None:
        values = {
            "raw_margin": float(raw_margin),
            "spindle_rpm": float(cut.spindle_rpm),
            "tooth_frequency_hz": float(cut.spindle_rpm * tool.flute_count / 60.0),
            "axial_depth_m": float(cut.axial_depth_m),
            "radial_depth_m": float(cut.radial_depth_m),
            "feed_per_tooth_m": float(cut.feed_per_tooth_m),
            "cutting_coeff_t_n_m2": float(cut.cutting_coeff_t_n_m2),
            "cutting_coeff_r_n_m2": float(cut.cutting_coeff_r_n_m2),
            "modal_frequency_hz": float(modal.natural_frequency_hz),
            "modal_damping_ratio": float(modal.damping_ratio),
            "stiffness_ratio_xy": float(modal.stiffness_x_n_m / max(modal.stiffness_y_n_m, 1.0e-12)),
            "radial_immersion_ratio": float(cut.radial_depth_m / max(tool.diameter_m, 1.0e-12)),
            "runout_to_feed_ratio": float(tool.runout_m / max(cut.feed_per_tooth_m, 1.0e-12)),
        }
    else:
        values = {"raw_margin": float(raw_margin)}

    values.update(
        {
            "log_spindle_rpm": _safe_log(values.get("spindle_rpm", 1.0)),
            "log_tooth_frequency_hz": _safe_log(values.get("tooth_frequency_hz", 1.0)),
            "log_axial_depth_m": _safe_log(values.get("axial_depth_m", 1.0e-6)),
            "log_radial_depth_m": _safe_log(values.get("radial_depth_m", 1.0e-6)),
            "log_feed_per_tooth_m": _safe_log(values.get("feed_per_tooth_m", 1.0e-6)),
            "log_cutting_coeff_t_n_m2": _safe_log(values.get("cutting_coeff_t_n_m2", 1.0)),
            "log_cutting_coeff_r_n_m2": _safe_log(values.get("cutting_coeff_r_n_m2", 1.0)),
            "log_modal_frequency_hz": _safe_log(values.get("modal_frequency_hz", 1.0)),
        }
    )
    return values


def _make_calibration_families(
    *,
    family_count: int,
    seed: int,
    stiffness_scale: tuple[float, float],
    damping_scale: tuple[float, float],
    cutting_coeff_scale: tuple[float, float],
    radial_depth_scale: tuple[float, float],
    feed_scale: tuple[float, float],
    runout_scale: tuple[float, float],
) -> list[CalibrationFamily]:
    _validate_range("stiffness_scale", stiffness_scale)
    _validate_range("damping_scale", damping_scale)
    _validate_range("cutting_coeff_scale", cutting_coeff_scale)
    _validate_range("radial_depth_scale", radial_depth_scale)
    _validate_range("feed_scale", feed_scale)
    _validate_range("runout_scale", runout_scale)

    if family_count == 1:
        return [CalibrationFamily(family=0)]

    rng = np.random.default_rng(seed + 31_337)
    return [
        CalibrationFamily(
            family=family,
            stiffness_x_scale=_sample_range(rng, stiffness_scale),
            stiffness_y_scale=_sample_range(rng, stiffness_scale),
            damping_x_scale=_sample_range(rng, damping_scale),
            damping_y_scale=_sample_range(rng, damping_scale),
            cutting_coeff_t_scale=_sample_range(rng, cutting_coeff_scale),
            cutting_coeff_r_scale=_sample_range(rng, cutting_coeff_scale),
            radial_depth_scale=_sample_range(rng, radial_depth_scale),
            feed_scale=_sample_range(rng, feed_scale),
            runout_scale=_sample_range(rng, runout_scale),
        )
        for family in range(family_count)
    ]


def _apply_family(base_modal, base_tool, base_cut, family: CalibrationFamily):
    modal = replace(
        base_modal,
        stiffness_x_n_m=base_modal.stiffness_x_n_m * family.stiffness_x_scale,
        stiffness_y_n_m=base_modal.stiffness_y_n_m * family.stiffness_y_scale,
        damping_x_n_s_m=base_modal.damping_x_n_s_m * family.damping_x_scale,
        damping_y_n_s_m=base_modal.damping_y_n_s_m * family.damping_y_scale,
    )
    tool = replace(base_tool, runout_m=base_tool.runout_m * family.runout_scale)
    cut = replace(
        base_cut,
        feed_per_tooth_m=base_cut.feed_per_tooth_m * family.feed_scale,
        radial_depth_m=base_cut.radial_depth_m * family.radial_depth_scale,
        cutting_coeff_t_n_m2=base_cut.cutting_coeff_t_n_m2 * family.cutting_coeff_t_scale,
        cutting_coeff_r_n_m2=base_cut.cutting_coeff_r_n_m2 * family.cutting_coeff_r_scale,
    )
    return modal, tool, cut


def _sample_margin_grid(
    *,
    scenarios: list[str],
    axial_depth_scales: tuple[float, ...],
    spindle_scales: tuple[float, ...],
    families: list[CalibrationFamily],
    holdout_family: int | None,
    duration_s: float,
    sample_rate_hz: float | None,
    sensor_noise_std: float,
    seed: int,
    target_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_idx = 0
    for family in families:
        split = "holdout" if holdout_family is not None and family.family == holdout_family else "train"
        for scenario_idx, scenario in enumerate(scenarios):
            base_modal, base_tool, base_cut, base_config = make_scenario(scenario)
            modal, tool, family_cut = _apply_family(base_modal, base_tool, base_cut, family)
            for axial_depth_scale in axial_depth_scales:
                for spindle_scale in spindle_scales:
                    cut = replace(
                        family_cut,
                        axial_depth_m=family_cut.axial_depth_m * axial_depth_scale,
                        spindle_rpm=family_cut.spindle_rpm * spindle_scale,
                    )
                    config_updates: dict[str, Any] = {
                        "duration_s": duration_s,
                        "sensor_noise_std": sensor_noise_std,
                        "random_seed": seed + family.family * 100_000 + scenario_idx * 10_000 + sample_idx,
                    }
                    if sample_rate_hz is not None:
                        config_updates["sample_rate_hz"] = sample_rate_hz
                    config = replace(base_config, **config_updates)
                    result = simulate_milling(modal, tool, cut, config)
                    features = extract_signal_features(
                        result.sensor_signal,
                        config.sample_rate_hz,
                        cut.spindle_rpm,
                        tool.flute_count,
                        modal.natural_frequency_hz,
                    )
                    raw_margin = signed_stability_margin(modal, tool, cut)
                    time_risk = time_domain_chatter_risk(features, result.sensor_signal)
                    context = _context_source(raw_margin, modal=modal, tool=tool, cut=cut)
                    rows.append(
                        {
                            "sample": sample_idx,
                            "split": split,
                            "family": family.family,
                            "scenario": scenario,
                            "axial_depth_scale": axial_depth_scale,
                            "spindle_scale": spindle_scale,
                            "stiffness_x_scale": family.stiffness_x_scale,
                            "stiffness_y_scale": family.stiffness_y_scale,
                            "damping_x_scale": family.damping_x_scale,
                            "damping_y_scale": family.damping_y_scale,
                            "cutting_coeff_t_scale": family.cutting_coeff_t_scale,
                            "cutting_coeff_r_scale": family.cutting_coeff_r_scale,
                            "radial_depth_scale": family.radial_depth_scale,
                            "feed_scale": family.feed_scale,
                            "runout_scale": family.runout_scale,
                            "axial_depth_m": cut.axial_depth_m,
                            "radial_depth_m": cut.radial_depth_m,
                            "feed_per_tooth_m": cut.feed_per_tooth_m,
                            "spindle_rpm": cut.spindle_rpm,
                            "tooth_frequency_hz": cut.spindle_rpm * tool.flute_count / 60.0,
                            "tool_diameter_m": tool.diameter_m,
                            "flute_count": tool.flute_count,
                            "runout_m": tool.runout_m,
                            "stiffness_x_n_m": modal.stiffness_x_n_m,
                            "stiffness_y_n_m": modal.stiffness_y_n_m,
                            "cutting_coeff_t_n_m2": cut.cutting_coeff_t_n_m2,
                            "cutting_coeff_r_n_m2": cut.cutting_coeff_r_n_m2,
                            "modal_frequency_hz": modal.natural_frequency_hz,
                            "modal_damping_ratio": modal.damping_ratio,
                            "stiffness_ratio_xy": context["stiffness_ratio_xy"],
                            "radial_immersion_ratio": context["radial_immersion_ratio"],
                            "runout_to_feed_ratio": context["runout_to_feed_ratio"],
                            "raw_margin": raw_margin,
                            "time_domain_risk": time_risk,
                            "target_positive": time_risk >= target_threshold,
                            "rms": features.rms,
                            "chatter_band_energy": features.chatter_band_energy,
                            "tooth_band_energy": features.tooth_band_energy,
                            "non_tooth_harmonic_ratio": features.non_tooth_harmonic_ratio,
                            "spectral_entropy": features.spectral_entropy,
                        }
                    )
                    sample_idx += 1
    return rows


def _fit_margin_calibration(
    rows: list[dict[str, Any]],
    *,
    target_threshold: float,
    calibration_model: str,
    all_rows: list[dict[str, Any]] | None = None,
    holdout_rows: list[dict[str, Any]] | None = None,
) -> tuple[MarginCalibration, dict[str, Any]]:
    y = np.array([1 if row["target_positive"] else 0 for row in rows], dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("calibration grid produced only one target class; widen the depth/spindle sweep or lower target threshold")

    feature_names = ("raw_margin",) if calibration_model == "raw" else CONTEXT_FEATURE_NAMES
    x_raw = _feature_matrix(rows, feature_names)
    feature_means = np.mean(x_raw, axis=0)
    feature_stds = np.std(x_raw, axis=0)
    feature_stds = np.where(feature_stds <= 1.0e-12, 1.0, feature_stds)
    x = (x_raw - feature_means) / feature_stds
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(class_weight="balanced", solver="lbfgs")
    model.fit(x, y)
    intercept = float(model.intercept_[0])
    coefficients = tuple(float(value) for value in model.coef_[0])
    if calibration_model == "raw":
        slope = float(coefficients[0] / feature_stds[0])
        intercept = float(intercept - coefficients[0] * feature_means[0] / feature_stds[0])
        feature_means_tuple: tuple[float, ...] = ()
        feature_stds_tuple: tuple[float, ...] = ()
        coefficients_tuple: tuple[float, ...] = ()
    else:
        slope = 0.0
        feature_means_tuple = tuple(float(value) for value in feature_means)
        feature_stds_tuple = tuple(float(value) for value in feature_stds)
        coefficients_tuple = coefficients
    calibration = MarginCalibration(
        intercept=intercept,
        slope=slope,
        target_threshold=target_threshold,
        sample_count=len(rows),
        positive_count=int(np.sum(y)),
        model_type=calibration_model,
        feature_names=feature_names,
        coefficients=coefficients_tuple,
        feature_means=feature_means_tuple,
        feature_stds=feature_stds_tuple,
    )
    all_rows = all_rows or rows
    holdout_rows = holdout_rows or []
    metrics = _evaluate_calibration(all_rows, calibration)
    metrics["train"] = _evaluate_calibration(rows, calibration)
    metrics["holdout"] = _evaluate_calibration(holdout_rows, calibration) if holdout_rows else None
    calibration = replace(
        calibration,
        train_brier_score=metrics["train"]["brier_score"],
        holdout_brier_score=metrics["holdout"]["brier_score"] if metrics["holdout"] else None,
    )
    return calibration, metrics


def _evaluate_calibration(rows: list[dict[str, Any]], calibration: MarginCalibration) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_fraction": 0.0,
            "accuracy_at_0_5": None,
            "roc_auc": None,
            "brier_score": None,
            "mean_absolute_error_to_time_domain_risk": None,
            "raw_margin_time_risk_correlation": None,
            "boundary_raw_margin_at_p_0_5": None,
            "risk_at_raw_margin_zero": calibration.physics_risk(0.0),
            "calibrated_margin_at_raw_margin_zero": calibration.calibrated_margin(0.0),
        }

    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    y = np.array([1 if row["target_positive"] else 0 for row in rows], dtype=int)
    raw_margins = np.array([float(row["raw_margin"]) for row in rows], dtype=float)
    probabilities = np.array([calibration.physics_risk(float(row["raw_margin"]), row=row) for row in rows], dtype=float)
    time_risks = np.array([float(row["time_domain_risk"]) for row in rows], dtype=float)
    boundary_margin = _boundary_raw_margin_at_mean_context(calibration)
    metrics = {
        "sample_count": len(rows),
        "positive_count": int(np.sum(y)),
        "negative_count": int(len(rows) - np.sum(y)),
        "positive_fraction": float(np.mean(y)),
        "accuracy_at_0_5": float(accuracy_score(y, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "mean_absolute_error_to_time_domain_risk": float(np.mean(np.abs(probabilities - time_risks))),
        "raw_margin_time_risk_correlation": _safe_corr(raw_margins, time_risks),
        "boundary_raw_margin_at_p_0_5": boundary_margin,
        "risk_at_raw_margin_zero": calibration.physics_risk(0.0),
        "calibrated_margin_at_raw_margin_zero": calibration.calibrated_margin(0.0),
    }
    return metrics


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    calibration = payload["calibration"]
    metrics = payload["metrics"]
    lines = [
        "# Margin Calibration Report",
        "",
        "This calibrates the raw FRF/regenerative stability margin against a signal-only time-domain simulator target.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| samples | {metrics['sample_count']} |",
        f"| positives | {metrics['positive_count']} |",
        f"| positive fraction | {metrics['positive_fraction']:.3f} |",
        f"| ROC AUC | {_fmt(metrics['roc_auc'])} |",
        f"| Brier score | {_fmt(metrics['brier_score'])} |",
        f"| accuracy at 0.5 | {_fmt(metrics['accuracy_at_0_5'])} |",
        f"| raw-margin/time-risk correlation | {_fmt(metrics['raw_margin_time_risk_correlation'])} |",
        f"| raw margin at p=0.5 | {_fmt(metrics['boundary_raw_margin_at_p_0_5'])} |",
        f"| risk at raw margin 0 | {metrics['risk_at_raw_margin_zero']:.3f} |",
        "",
        "| Split | Samples | Positives | Positive Fraction | ROC AUC | Brier | Accuracy | MAE to Time Risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _split_metric_row("train", metrics["train"]),
        _split_metric_row("holdout", metrics["holdout"]),
        "",
        "Calibration:",
        "",
        f"* model: `{calibration['model_type']}`",
        f"* intercept: `{calibration['intercept']:.6g}`",
        f"* slope: `{calibration['slope']:.6g}`",
        f"* target threshold: `{calibration['target_threshold']:.3f}`",
        f"* features: `{', '.join(calibration['feature_names'])}`",
        "",
        "Use `calibration.json` with benchmark commands that accept `--margin-calibration`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg(path: Path, rows: list[dict[str, Any]], calibration: MarginCalibration) -> None:
    width = 860
    height = 460
    left = 76
    right = 32
    top = 44
    bottom = 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    margins = np.array([float(row["raw_margin"]) for row in rows], dtype=float)
    risks = np.array([float(row["time_domain_risk"]) for row in rows], dtype=float)
    x_min, x_max = float(np.min(margins)), float(np.max(margins))
    y_min, y_max = 0.0, max(1.0, float(np.max(risks)) + 0.05)

    def x_pos(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1.0e-9) * plot_w

    def y_pos(value: float) -> float:
        return top + (1.0 - (value - y_min) / max(y_max - y_min, 1.0e-9)) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="600">Margin calibration: raw margin vs time-domain risk</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w - 120}" y="{height - 18}" font-family="sans-serif" font-size="12">raw margin</text>',
        f'<text x="18" y="{top + 14}" font-family="sans-serif" font-size="12">risk</text>',
    ]
    color_by_scenario = {
        "stable": "#2b7a78",
        "near_boundary": "#3f7fbf",
        "onset": "#d18b2f",
        "unstable": "#c94f4f",
    }
    for row in rows:
        color = color_by_scenario.get(str(row["scenario"]), "#666")
        svg.append(
            f'<circle cx="{x_pos(float(row["raw_margin"])):.1f}" cy="{y_pos(float(row["time_domain_risk"])):.1f}" '
            f'r="4" fill="{color}" opacity="0.62"/>'
        )
    curve_x = np.linspace(x_min, x_max, 160)
    points = " ".join(f'{x_pos(float(x)):.1f},{y_pos(calibration.physics_risk(float(x))):.1f}' for x in curve_x)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#111" stroke-width="2"/>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [context_feature_values(float(row["raw_margin"]), row=row, feature_names=feature_names) for row in rows],
        dtype=float,
    )


def _boundary_raw_margin_at_mean_context(calibration: MarginCalibration) -> float | None:
    if calibration.model_type == "context" and calibration.coefficients and "raw_margin" in calibration.feature_names:
        idx = calibration.feature_names.index("raw_margin")
        coef = calibration.coefficients[idx]
        if abs(coef) <= 1.0e-12:
            return None
        mean = calibration.feature_means[idx]
        std = max(calibration.feature_stds[idx], 1.0e-12)
        return float(mean - calibration.intercept * std / coef)
    if abs(calibration.slope) <= 1.0e-12:
        return None
    return float(-calibration.intercept / calibration.slope)


def _split_metric_row(name: str, metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return f"| {name} | 0 | 0 | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {name} | {metrics['sample_count']} | {metrics['positive_count']} | "
        f"{_fmt(metrics['positive_fraction'])} | {_fmt(metrics['roc_auc'])} | "
        f"{_fmt(metrics['brier_score'])} | {_fmt(metrics['accuracy_at_0_5'])} | "
        f"{_fmt(metrics['mean_absolute_error_to_time_domain_risk'])} |"
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_log(value: float) -> float:
    return math.log(max(float(value), 1.0e-18))


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if abs(high - low) <= 1.0e-12:
        return float(low)
    return float(rng.uniform(low, high))


def _validate_range(name: str, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if low <= 0 or high <= 0:
        raise ValueError(f"{name} values must be positive")
    if high < low:
        raise ValueError(f"{name} upper bound must be greater than or equal to lower bound")
