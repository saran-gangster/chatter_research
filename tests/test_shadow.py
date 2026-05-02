import csv
import json
from pathlib import Path

from chatter_twin.cli import main
from chatter_twin.shadow import ShadowPolicyConfig, run_shadow_evaluation


def _write_shadow_fixture(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    metrics = {
        "test": {"accuracy": 0.9, "binary_chatter_f1": 0.8},
        "event_warning": {
            "threshold": 0.5,
            "test": {"f1": 0.7, "recall": 0.6},
            "threshold_selection": {"selected_threshold": 0.5},
        },
        "lead_time": {
            "threshold_selection": {"selected_threshold": 0.4},
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = [
        {
            "window_id": 0,
            "scenario": "onset",
            "episode": 0,
            "start_time_s": 0.0,
            "label_id": 1,
            "label": "transition",
            "horizon_label": "slight",
            "target_label": "slight",
            "predicted_label": "transition",
            "predicted_chatter_score": 0.75,
            "risk_chatter_now": 0.4,
            "future_chatter_within_horizon": "True",
            "time_to_chatter_s": 0.1,
        },
        {
            "window_id": 1,
            "scenario": "onset",
            "episode": 0,
            "start_time_s": 0.1,
            "label_id": 2,
            "label": "slight",
            "horizon_label": "slight",
            "target_label": "slight",
            "predicted_label": "slight",
            "predicted_chatter_score": 0.9,
            "risk_chatter_now": 0.7,
            "future_chatter_within_horizon": "False",
            "time_to_chatter_s": 0.0,
        },
        {
            "window_id": 2,
            "scenario": "stable",
            "episode": 0,
            "start_time_s": 0.0,
            "label_id": 0,
            "label": "stable",
            "horizon_label": "stable",
            "target_label": "stable",
            "predicted_label": "stable",
            "predicted_chatter_score": 0.2,
            "risk_chatter_now": 0.1,
            "future_chatter_within_horizon": "False",
            "time_to_chatter_s": -1.0,
        },
    ]
    with (model_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_shadow_evaluation_detects_event_before_chatter(tmp_path: Path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "shadow"
    _write_shadow_fixture(model_dir)

    payload = run_shadow_evaluation(model_dir=model_dir, out_dir=out_dir, config=ShadowPolicyConfig())

    assert payload["event_metrics"]["event_episodes"] == 1
    assert payload["event_metrics"]["detected_event_episodes"] == 1
    assert payload["event_metrics"]["false_warning_episodes"] == 0
    assert payload["event_metrics"]["f1"] == 1.0
    assert payload["window_metrics"]["relative_mrr_proxy"] < 1.0
    for name in ["recommendations.csv", "shadow_metrics.json", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_eval_runs(tmp_path: Path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "shadow"
    _write_shadow_fixture(model_dir)

    status = main(["shadow-eval", "--model-dir", str(model_dir), "--out", str(out_dir)])

    assert status == 0
    metrics = json.loads((out_dir / "shadow_metrics.json").read_text(encoding="utf-8"))
    assert metrics["event_metrics"]["detected_event_episodes"] == 1
