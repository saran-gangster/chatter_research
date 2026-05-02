import csv
from pathlib import Path

from chatter_twin.calibration import calibrate_margin_surrogate, load_margin_calibration
from chatter_twin.cli import main


def test_calibrate_margin_surrogate_writes_artifacts(tmp_path: Path):
    payload = calibrate_margin_surrogate(
        scenarios=["stable", "unstable"],
        axial_depth_scales=(0.5, 1.5, 3.0),
        spindle_scales=(0.90, 1.10),
        duration_s=0.18,
        sensor_noise_std=0.0,
        target_threshold=0.45,
        out_dir=tmp_path,
        seed=123,
    )

    assert payload["metrics"]["sample_count"] == 12
    assert payload["metrics"]["positive_count"] > 0
    for name in ["samples.csv", "calibration.json", "report.md", "margin_calibration.svg"]:
        assert (tmp_path / name).exists()

    calibration = load_margin_calibration(tmp_path / "calibration.json")
    assert 0.0 <= calibration.physics_risk(0.0) <= 1.0
    assert calibration.calibrated_margin(0.0) == payload["metrics"]["calibrated_margin_at_raw_margin_zero"]


def test_cli_calibrate_margin_runs(tmp_path: Path):
    out_dir = tmp_path / "calibration"
    status = main(
        [
            "calibrate-margin",
            "--scenarios",
            "stable,unstable",
            "--axial-depth-scales",
            "0.5,1.5,3.0",
            "--spindle-scales",
            "0.90,1.10",
            "--duration",
            "0.14",
            "--sensor-noise",
            "0.0",
            "--target-threshold",
            "0.35",
            "--family-count",
            "3",
            "--holdout-family",
            "2",
            "--out",
            str(out_dir),
        ]
    )

    assert status == 0
    assert (out_dir / "calibration.json").exists()
    assert load_margin_calibration(out_dir / "calibration.json").sample_count == 24

    bench_dir = tmp_path / "bench"
    status = main(
        [
            "closed-loop-benchmark",
            "--controllers",
            "fixed,hybrid",
            "--scenarios",
            "stable",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--decision-interval",
            "0.05",
            "--margin-calibration",
            str(out_dir / "calibration.json"),
            "--out",
            str(bench_dir),
        ]
    )
    assert status == 0
    assert (bench_dir / "summary.csv").exists()


def test_randomized_family_holdout_reports_split_metrics(tmp_path: Path):
    payload = calibrate_margin_surrogate(
        scenarios=["stable", "unstable"],
        axial_depth_scales=(0.5, 1.5, 3.0),
        spindle_scales=(0.90, 1.10),
        duration_s=0.14,
        sensor_noise_std=0.0,
        target_threshold=0.35,
        family_count=3,
        holdout_family=2,
        out_dir=tmp_path,
        seed=222,
    )

    assert payload["metrics"]["sample_count"] == 36
    assert payload["metrics"]["train"]["sample_count"] == 24
    assert payload["metrics"]["holdout"]["sample_count"] == 12
    assert payload["metrics"]["holdout"]["roc_auc"] is not None
    assert len(payload["families"]) == 3

    with (tmp_path / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "holdout"}
    assert {row["family"] for row in rows if row["split"] == "holdout"} == {"2"}


def test_context_calibration_loads_and_scores_with_context(tmp_path: Path):
    payload = calibrate_margin_surrogate(
        scenarios=["stable", "unstable"],
        axial_depth_scales=(0.5, 1.5, 3.0),
        spindle_scales=(0.90, 1.10),
        duration_s=0.14,
        sensor_noise_std=0.0,
        target_threshold=0.35,
        calibration_model="context",
        family_count=3,
        holdout_family=2,
        out_dir=tmp_path,
        seed=222,
    )

    calibration = load_margin_calibration(tmp_path / "calibration.json")
    assert calibration.model_type == "context"
    assert "log_axial_depth_m" in calibration.feature_names
    assert len(calibration.coefficients) == len(calibration.feature_names)
    assert payload["metrics"]["holdout"]["sample_count"] == 12
    assert 0.0 <= calibration.uncertainty(0.0) <= 0.95
