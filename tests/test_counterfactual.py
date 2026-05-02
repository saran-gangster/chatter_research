import csv
import json
from pathlib import Path

from chatter_twin.cli import main
from chatter_twin.counterfactual import (
    CounterfactualConfig,
    CounterfactualPolicyConfig,
    StabilityPolicyConfig,
    run_counterfactual_risk_shadow_policy,
    run_shadow_action_sweep,
    run_shadow_counterfactual,
    run_shadow_episode_counterfactual,
    run_stability_margin_shadow_policy,
)


def _write_counterfactual_fixture(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "dataset"
    shadow_dir = root / "shadow"
    model_dir = root / "model"
    dataset_dir.mkdir(parents=True)
    shadow_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    (model_dir / "metrics.json").write_text(
        json.dumps({"dataset": {"path": str(dataset_dir)}}),
        encoding="utf-8",
    )
    (shadow_dir / "shadow_metrics.json").write_text(
        json.dumps({"model_dir": str(model_dir)}),
        encoding="utf-8",
    )

    dataset_rows = [
        {
            "window_id": 0,
            "scenario": "unstable",
            "episode": 0,
            "start_time_s": 0.0,
            "end_time_s": 0.08,
            "label": "transition",
            "stiffness_scale": 1.0,
            "damping_scale": 1.0,
            "noise_scale": 0.0,
            "spindle_rpm": 8800.0,
            "feed_per_tooth_m": 45.0e-6,
            "axial_depth_m": 0.0018,
            "radial_depth_m": 0.004,
            "cutting_coeff_t_n_m2": 7.0e8,
            "cutting_coeff_r_n_m2": 2.1e8,
        },
        {
            "window_id": 1,
            "scenario": "stable",
            "episode": 0,
            "start_time_s": 0.0,
            "end_time_s": 0.08,
            "label": "stable",
            "stiffness_scale": 1.0,
            "damping_scale": 1.0,
            "noise_scale": 0.0,
            "spindle_rpm": 9200.0,
            "feed_per_tooth_m": 45.0e-6,
            "axial_depth_m": 0.00025,
            "radial_depth_m": 0.004,
            "cutting_coeff_t_n_m2": 7.0e8,
            "cutting_coeff_r_n_m2": 2.1e8,
        },
    ]
    _write_csv(dataset_dir / "windows.csv", dataset_rows)

    recommendation_rows = [
        {
            "window_id": 0,
            "scenario": "unstable",
            "episode": 0,
            "start_time_s": 0.0,
            "label": "transition",
            "predicted_chatter_score": 0.9,
            "shadow_warning": True,
            "action_active": True,
            "feed_override": 0.92,
            "spindle_override": 1.04,
            "relative_mrr_proxy": 0.9568,
        },
        {
            "window_id": 1,
            "scenario": "stable",
            "episode": 0,
            "start_time_s": 0.0,
            "label": "stable",
            "predicted_chatter_score": 0.1,
            "shadow_warning": False,
            "action_active": False,
            "feed_override": 1.0,
            "spindle_override": 1.0,
            "relative_mrr_proxy": 1.0,
        },
    ]
    _write_csv(shadow_dir / "recommendations.csv", recommendation_rows)
    return shadow_dir, dataset_dir


def test_shadow_counterfactual_writes_artifacts(tmp_path: Path):
    shadow_dir, dataset_dir = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "counterfactual"

    payload = run_shadow_counterfactual(
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        config=CounterfactualConfig(sensor_noise_std=0.0),
    )

    assert payload["windows"]["count"] == 2
    assert payload["windows"]["action_windows"] == 1
    assert "mean_risk_reduction" in payload["windows"]
    for name in ["counterfactual_windows.csv", "counterfactual_metrics.json", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_counterfactual_infers_dataset_dir(tmp_path: Path):
    shadow_dir, _ = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "counterfactual"

    status = main(
        [
            "shadow-counterfactual",
            "--shadow-dir",
            str(shadow_dir),
            "--out",
            str(out_dir),
            "--sensor-noise",
            "0.0",
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "counterfactual_metrics.json").read_text(encoding="utf-8"))
    assert metrics["windows"]["count"] == 2


def test_shadow_episode_counterfactual_writes_artifacts(tmp_path: Path):
    shadow_dir, dataset_dir = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "episode_counterfactual"

    payload = run_shadow_episode_counterfactual(
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        config=CounterfactualConfig(sensor_noise_std=0.0),
    )

    assert payload["windows"]["count"] == 2
    assert payload["episodes"]["episode_count"] == 2
    for name in ["episode_windows.csv", "episode_summary.csv", "episode_metrics.json", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_episode_counterfactual_runs(tmp_path: Path):
    shadow_dir, _ = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "episode_counterfactual"

    status = main(
        [
            "shadow-episode-counterfactual",
            "--shadow-dir",
            str(shadow_dir),
            "--out",
            str(out_dir),
            "--sensor-noise",
            "0.0",
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "episode_metrics.json").read_text(encoding="utf-8"))
    assert metrics["episodes"]["episode_count"] == 2


def test_shadow_action_sweep_writes_candidates(tmp_path: Path):
    shadow_dir, dataset_dir = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "sweep"

    payload = run_shadow_action_sweep(
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        feed_values=(0.92, 1.0),
        spindle_values=(1.0, 1.04),
        config=CounterfactualConfig(sensor_noise_std=0.0),
    )

    assert payload["candidate_count"] == 4
    assert "best_policy" in payload
    for name in ["sweep.csv", "sweep_metrics.json", "best_counterfactual_windows.csv", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_action_sweep_runs(tmp_path: Path):
    shadow_dir, _ = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "sweep"

    status = main(
        [
            "shadow-action-sweep",
            "--shadow-dir",
            str(shadow_dir),
            "--out",
            str(out_dir),
            "--feed-values",
            "0.92,1.0",
            "--spindle-values",
            "1.0,1.04",
            "--sensor-noise",
            "0.0",
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "sweep_metrics.json").read_text(encoding="utf-8"))
    assert metrics["candidate_count"] == 4


def test_stability_margin_shadow_policy_writes_recommendations(tmp_path: Path):
    shadow_dir, dataset_dir = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "stability_policy"

    payload = run_stability_margin_shadow_policy(
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        config=StabilityPolicyConfig(candidate_count=5, min_margin_improvement=0.0),
    )

    assert payload["window_metrics"]["windows"] == 2
    assert "mean_selected_margin_improvement" in payload["window_metrics"]
    for name in ["recommendations.csv", "shadow_metrics.json", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_stability_policy_runs(tmp_path: Path):
    shadow_dir, _ = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "stability_policy"

    status = main(
        [
            "shadow-stability-policy",
            "--shadow-dir",
            str(shadow_dir),
            "--out",
            str(out_dir),
            "--candidates",
            "5",
            "--min-margin-improvement",
            "0.0",
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "shadow_metrics.json").read_text(encoding="utf-8"))
    assert metrics["policy"]["type"] == "stability_margin"


def test_counterfactual_risk_shadow_policy_writes_recommendations(tmp_path: Path):
    shadow_dir, dataset_dir = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "counterfactual_policy"

    payload = run_counterfactual_risk_shadow_policy(
        shadow_dir=shadow_dir,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        counterfactual_config=CounterfactualConfig(sensor_noise_std=0.0),
        policy_config=CounterfactualPolicyConfig(
            feed_values=(1.0,),
            spindle_values=(1.0, 1.04),
            min_risk_reduction=0.0,
        ),
    )

    assert payload["window_metrics"]["windows"] == 2
    assert "mean_selected_risk_reduction" in payload["window_metrics"]
    for name in ["recommendations.csv", "shadow_metrics.json", "report.md"]:
        assert (out_dir / name).exists()


def test_cli_shadow_counterfactual_policy_runs(tmp_path: Path):
    shadow_dir, _ = _write_counterfactual_fixture(tmp_path)
    out_dir = tmp_path / "counterfactual_policy"

    status = main(
        [
            "shadow-counterfactual-policy",
            "--shadow-dir",
            str(shadow_dir),
            "--out",
            str(out_dir),
            "--feed-values",
            "1.0",
            "--spindle-values",
            "1.0,1.04",
            "--min-risk-reduction",
            "0.0",
            "--sensor-noise",
            "0.0",
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "shadow_metrics.json").read_text(encoding="utf-8"))
    assert metrics["policy"]["type"] == "counterfactual_risk"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
