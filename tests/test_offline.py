import csv
import json
from pathlib import Path

import numpy as np

from chatter_twin.cli import main
from chatter_twin.offline import (
    RiskTrainingConfig,
    episode_group_keys,
    event_warning_metrics,
    event_warning_threshold_sweep,
    feature_columns_for_set,
    feature_value,
    lead_time_metrics,
    lead_time_threshold_sweep,
    load_window_records,
    make_feature_matrix,
    make_train_test_split,
    make_validation_split,
    scenario_group_keys,
    train_risk_model,
    train_test_indices,
)
from chatter_twin.replay import DomainRandomizationConfig, WindowSpec, export_synthetic_dataset


def _make_dataset(path: Path) -> Path:
    export_synthetic_dataset(
        scenarios=["stable", "near_boundary", "unstable"],
        episodes=2,
        duration_s=0.35,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        out_dir=path,
    )
    return path


def _make_randomized_dataset(path: Path) -> Path:
    export_synthetic_dataset(
        scenarios=["stable", "near_boundary", "unstable"],
        episodes=4,
        duration_s=0.35,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        out_dir=path,
        randomization=DomainRandomizationConfig(enabled=True),
    )
    return path


def _make_onset_dataset(path: Path) -> Path:
    export_synthetic_dataset(
        scenarios=["onset"],
        episodes=2,
        duration_s=0.9,
        window_spec=WindowSpec(window_s=0.1, stride_s=0.05),
        out_dir=path,
    )
    return path


def test_make_feature_matrix_loads_numeric_columns(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    x, y = make_feature_matrix(records, ("margin_physics", "rms", "chatter_band_energy"))
    assert x.shape == (len(records), 3)
    assert y.shape == (len(records),)
    assert np.isfinite(x).all()


def test_make_feature_matrix_loads_temporal_feature_set(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    x, y = make_feature_matrix(records, feature_columns_for_set("temporal"))
    assert x.shape[0] == len(records)
    assert x.shape[1] > len(feature_columns_for_set("base"))
    assert y.shape == (len(records),)
    assert np.isfinite(x).all()


def test_make_feature_matrix_loads_profile_temporal_feature_set(tmp_path: Path):
    dataset = _make_onset_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    x, y = make_feature_matrix(records, feature_columns_for_set("profile_temporal"))
    assert x.shape[0] == len(records)
    assert x.shape[1] > len(feature_columns_for_set("temporal"))
    assert y.shape == (len(records),)
    assert np.isfinite(x).all()


def test_make_feature_matrix_loads_interaction_temporal_feature_set(tmp_path: Path):
    dataset = _make_onset_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    x, y = make_feature_matrix(records, feature_columns_for_set("interaction_temporal"))
    assert x.shape[0] == len(records)
    assert x.shape[1] > len(feature_columns_for_set("profile_temporal"))
    assert y.shape == (len(records),)
    assert np.isfinite(x).all()
    assert feature_value(records[0], "depth_to_critical_ratio") == 1.0 - float(records[0]["margin_physics"])


def test_make_feature_matrix_loads_horizon_target(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    x, y = make_feature_matrix(records, feature_columns_for_set("base"), target_column="horizon_label_id")
    assert x.shape[0] == len(records)
    assert y.shape == (len(records),)
    assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})


def test_lead_time_metrics_score_only_pre_chatter_windows(tmp_path: Path):
    dataset = _make_onset_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    indices = np.arange(len(records), dtype=np.int64)
    probs = np.zeros((len(records), 5), dtype=float)
    for row_idx, record in enumerate(records):
        if record["future_chatter_within_horizon"] == "True":
            probs[row_idx, 2] = 0.8
            probs[row_idx, 0] = 0.2
        else:
            probs[row_idx, 0] = 0.8
            probs[row_idx, 2] = 0.2

    metrics = lead_time_metrics(records, indices, probs)

    assert metrics["candidate_windows"] > 0
    assert metrics["positive_windows"] > 0
    assert metrics["recall"] == 1.0
    assert metrics["mean_detected_time_to_chatter_s"] is not None


def test_lead_time_threshold_sweep_selects_best_threshold(tmp_path: Path):
    dataset = _make_onset_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    indices = np.arange(len(records), dtype=np.int64)
    probs = np.zeros((len(records), 5), dtype=float)
    for row_idx, record in enumerate(records):
        if record["future_chatter_within_horizon"] == "True":
            probs[row_idx, 2] = 0.7
            probs[row_idx, 0] = 0.3
        else:
            probs[row_idx, 0] = 0.7
            probs[row_idx, 2] = 0.3

    sweep = lead_time_threshold_sweep(records, indices, probs, thresholds=(0.25, 0.5, 0.75))

    assert len(sweep["rows"]) == 3
    assert sweep["best_f1"]["threshold"] == 0.5
    assert sweep["best_f1"]["f1"] > 0.0


def test_event_warning_metrics_score_episode_before_first_chatter(tmp_path: Path):
    dataset = _make_onset_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    indices = np.arange(len(records), dtype=np.int64)
    probs = np.zeros((len(records), 5), dtype=float)
    for row_idx, record in enumerate(records):
        if record["future_chatter_within_horizon"] == "True" and record["label"] not in {"slight", "severe"}:
            probs[row_idx, 2] = 0.8
            probs[row_idx, 0] = 0.2
        else:
            probs[row_idx, 0] = 0.8
            probs[row_idx, 2] = 0.2

    metrics = event_warning_metrics(records, indices, probs)
    sweep = event_warning_threshold_sweep(records, indices, probs, thresholds=(0.25, 0.5, 0.75))

    assert metrics["episode_count"] > 0
    assert metrics["event_episodes"] > 0
    assert metrics["detected_event_episodes"] > 0
    assert metrics["mean_detected_lead_time_s"] is not None
    assert sweep["best_f1"]["f1"] > 0.0


def test_train_test_indices_keep_training_examples():
    y = np.array([0, 0, 0, 2, 2, 2])
    train_idx, test_idx = train_test_indices(y, test_fraction=0.5, seed=1)
    assert train_idx.size >= 2
    assert test_idx.size >= 2
    assert set(y[train_idx]) == {0, 2}


def test_episode_split_keeps_groups_disjoint(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(records, y, RiskTrainingConfig(split_mode="episode", test_fraction=0.5, seed=9))
    groups = episode_group_keys(records)
    train_groups = set(groups[split.train_idx].tolist())
    test_groups = set(groups[split.test_idx].tolist())

    assert split.metadata["mode"] == "episode"
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)


def test_scenario_split_keeps_scenarios_disjoint(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(records, y, RiskTrainingConfig(split_mode="scenario", test_fraction=0.34, seed=9))
    groups = scenario_group_keys(records)
    train_groups = set(groups[split.train_idx].tolist())
    test_groups = set(groups[split.test_idx].tolist())

    assert split.metadata["mode"] == "scenario"
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)


def test_time_block_split_holds_out_late_windows_per_episode(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(records, y, RiskTrainingConfig(split_mode="time_block", test_fraction=0.4))
    groups = episode_group_keys(records)

    assert split.metadata["mode"] == "time_block"
    assert split.metadata["block_policy"] == "train_early_test_late_within_each_episode"
    for group in sorted(set(groups.tolist())):
        train_times = [float(records[index]["start_time_s"]) for index in split.train_idx if groups[index] == group]
        test_times = [float(records[index]["start_time_s"]) for index in split.test_idx if groups[index] == group]
        assert train_times
        assert test_times
        assert max(train_times) < min(test_times)


def test_parameter_family_split_holds_out_extreme_scale(tmp_path: Path):
    dataset = _make_randomized_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(
        records,
        y,
        RiskTrainingConfig(split_mode="parameter_family", holdout_column="axial_depth_scale", test_fraction=0.25),
    )
    values = np.array([float(record["axial_depth_scale"]) for record in records])

    assert split.metadata["mode"] == "parameter_family"
    assert split.metadata["holdout_column"] == "axial_depth_scale"
    assert np.min(values[split.test_idx]) >= np.max(values[split.train_idx])


def test_validation_split_is_disjoint_from_fit_subset(tmp_path: Path):
    dataset = _make_randomized_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(records, y, RiskTrainingConfig(split_mode="episode", test_fraction=0.25, seed=3))
    validation = make_validation_split(
        records,
        split.train_idx,
        y,
        RiskTrainingConfig(split_mode="episode", test_fraction=0.25, validation_fraction=0.25, seed=3),
    )

    assert validation.metadata["enabled"] is True
    assert validation.fit_idx.size > 0
    assert validation.validation_idx.size > 0
    assert set(validation.fit_idx.tolist()).isdisjoint(set(validation.validation_idx.tolist()))


def test_time_block_validation_split_is_disjoint_from_fit_subset(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    records = load_window_records(dataset / "windows.csv")
    _, y = make_feature_matrix(records, ("margin_physics",))
    split = make_train_test_split(records, y, RiskTrainingConfig(split_mode="time_block", test_fraction=0.25))
    validation = make_validation_split(
        records,
        split.train_idx,
        y,
        RiskTrainingConfig(split_mode="time_block", test_fraction=0.25, validation_fraction=0.25),
    )

    assert validation.metadata["enabled"] is True
    assert validation.fit_idx.size > 0
    assert validation.validation_idx.size > 0
    assert set(validation.fit_idx.tolist()).isdisjoint(set(validation.validation_idx.tolist()))


def test_train_risk_model_writes_artifacts(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    payload = train_risk_model(
        dataset_dir=dataset,
        out_dir=out_dir,
        config=RiskTrainingConfig(epochs=120, learning_rate=0.08, validation_fraction=0.2, seed=4),
    )
    assert 0.0 <= payload["test"]["accuracy"] <= 1.0
    assert 0.0 <= payload["test"]["intervention_f1"] <= 1.0
    assert "lead_time" in payload
    assert "event_warning" in payload
    assert 0.0 <= payload["lead_time"]["test"]["f1"] <= 1.0
    assert 0.0 <= payload["event_warning"]["test"]["f1"] <= 1.0
    assert payload["loss"]["final"] <= payload["loss"]["initial"]
    assert payload["split"]["mode"] == "row"
    assert payload["dataset"]["validation_windows"] > 0
    assert payload["validation"] is not None
    assert payload["lead_time"]["threshold_selection"]["source"] in {"train", "validation"}
    for name in ["model.json", "metrics.json", "predictions.csv", "confusion_matrix.csv", "report.md"]:
        assert (out_dir / name).exists()

    model = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
    assert "weights" in model
    assert len(model["feature_columns"]) > 0
    with (out_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "Lead-Time Early Warning" in (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Offline Risk Model Report" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_train_risk_model_supports_parameter_family_split(tmp_path: Path):
    dataset = _make_randomized_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    payload = train_risk_model(
        dataset_dir=dataset,
        out_dir=out_dir,
        config=RiskTrainingConfig(
            epochs=80,
            learning_rate=0.08,
            seed=5,
            split_mode="parameter_family",
            holdout_column="axial_depth_scale",
            test_fraction=0.25,
        ),
    )
    assert payload["split"]["mode"] == "parameter_family"
    assert payload["split"]["test_group_count"] > 0
    assert "Holdout:" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_train_risk_model_supports_hist_gradient_boosting(tmp_path: Path):
    dataset = _make_randomized_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    payload = train_risk_model(
        dataset_dir=dataset,
        out_dir=out_dir,
        config=RiskTrainingConfig(
            model_type="hist_gb",
            calibration="sigmoid",
            feature_set="profile_temporal",
            target="horizon",
            split_mode="episode",
            test_fraction=0.25,
            seed=7,
        ),
    )
    assert payload["model"]["type"] == "hist_gb"
    assert payload["config"]["feature_set"] == "profile_temporal"
    assert payload["target"]["mode"] == "horizon"
    assert payload["model"]["calibration"]["requested"] == "sigmoid"
    assert "threshold_selection" in payload["lead_time"]
    assert "threshold_selection" in payload["event_warning"]
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "model.json").exists()
    model = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
    assert model["model_type"] == "hist_gb"
    assert model["serialized_artifact"] == "model.joblib"
    assert "Model: `hist_gb`" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_cli_train_risk_runs(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    status = main(
        [
            "train-risk",
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--epochs",
            "80",
            "--split-mode",
            "episode",
        ]
    )
    assert status == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"]["mode"] == "episode"


def test_cli_train_risk_runs_time_block_split(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    status = main(
        [
            "train-risk",
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--epochs",
            "80",
            "--split-mode",
            "time_block",
        ]
    )
    assert status == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"]["mode"] == "time_block"
    assert metrics["split"]["block_policy"] == "train_early_test_late_within_each_episode"


def test_cli_train_risk_runs_scenario_split(tmp_path: Path):
    dataset = _make_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    status = main(
        [
            "train-risk",
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--epochs",
            "80",
            "--split-mode",
            "scenario",
        ]
    )
    assert status == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"]["mode"] == "scenario"


def test_cli_train_risk_runs_hist_gradient_boosting(tmp_path: Path):
    dataset = _make_randomized_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "model"
    status = main(
        [
            "train-risk",
            "--dataset",
            str(dataset),
            "--out",
            str(out_dir),
            "--model",
            "hist_gb",
            "--calibration",
            "sigmoid",
            "--feature-set",
            "temporal",
            "--target",
            "horizon",
            "--split-mode",
            "episode",
        ]
    )
    assert status == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model"]["type"] == "hist_gb"
    assert metrics["target"]["mode"] == "horizon"
    assert "lead_time" in metrics
    assert "event_warning" in metrics
    assert (out_dir / "model.joblib").exists()
