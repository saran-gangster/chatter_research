import csv
import json
from pathlib import Path

from chatter_twin.cli import main
from chatter_twin.realdata import RealDataRunSpec, write_real_data_benchmark, write_risk_error_analysis


def test_write_real_data_benchmark_summarizes_metrics(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_metrics(model_dir / "metrics.json")

    payload = write_real_data_benchmark(
        out_dir=tmp_path / "benchmark",
        runs=(
            RealDataRunSpec(
                dataset="Fixture",
                modality="sensor",
                result_dir=model_dir,
                claim_allowed="fixture claim",
            ),
        ),
    )

    assert payload["rows"][0]["dataset"] == "Fixture"
    assert payload["rows"][0]["split_type"] == "scenario"
    assert payload["rows"][0]["chatter_f1"] == 0.75
    report = (tmp_path / "benchmark" / "report.md").read_text(encoding="utf-8")
    assert "Real-Data Benchmark" in report
    assert "fixture claim" in report


def test_cli_real_data_benchmark_accepts_run_specs(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_metrics(model_dir / "metrics.json")
    out_dir = tmp_path / "benchmark"

    status = main(
        [
            "real-data-benchmark",
            "--out",
            str(out_dir),
            "--run",
            f"dataset=Fixture,modality=sensor,path={model_dir},claim=fixture claim",
        ]
    )

    assert status == 0
    rows = list(csv.DictReader((out_dir / "real_data_benchmark.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["dataset"] == "Fixture"
    assert rows[0]["lead_time_f1"] == "0.5"


def test_write_risk_error_analysis_reports_group_failures(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_predictions(model_dir / "predictions.csv")

    payload = write_risk_error_analysis(model_dir=model_dir, out_dir=tmp_path / "errors")

    assert payload["windows"] == 6
    binary = next(row for row in payload["label_metrics"] if row["label"] == "chatter_binary")
    assert binary["support"] == 4
    worst = payload["worst_groups"][0]
    assert worst["scenario"] == "ExpB"
    assert worst["false_negative_windows"] == 2
    report = (tmp_path / "errors" / "report.md").read_text(encoding="utf-8")
    assert "Worst Chatter Groups" in report
    assert "ExpB" in report


def test_cli_risk_error_analysis_runs(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_predictions(model_dir / "predictions.csv")
    out_dir = tmp_path / "errors"

    status = main(["risk-error-analysis", "--model-dir", str(model_dir), "--out", str(out_dir)])

    assert status == 0
    payload = json.loads((out_dir / "error_analysis.json").read_text(encoding="utf-8"))
    assert payload["group_column"] == "scenario"
    assert (out_dir / "group_metrics.csv").exists()


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": {"mode": "scenario"},
        "target": {"mode": "current"},
        "test": {
            "accuracy": 0.7,
            "binary_chatter_f1": 0.75,
            "intervention_f1": 0.75,
        },
        "lead_time": {
            "test": {"f1": 0.5},
            "threshold_selection": {"test_at_selected_threshold": {"f1": 0.6}},
        },
        "event_warning": {"test": {"f1": 0.2}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"scenario": "ExpA", "target_label": "stable", "label": "stable", "predicted_label": "stable"},
        {"scenario": "ExpA", "target_label": "slight", "label": "slight", "predicted_label": "slight"},
        {"scenario": "ExpA", "target_label": "severe", "label": "severe", "predicted_label": "stable"},
        {"scenario": "ExpB", "target_label": "slight", "label": "slight", "predicted_label": "stable"},
        {"scenario": "ExpB", "target_label": "severe", "label": "severe", "predicted_label": "stable"},
        {"scenario": "ExpB", "target_label": "stable", "label": "stable", "predicted_label": "slight"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
