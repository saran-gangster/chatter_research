import csv
import json
from pathlib import Path

import numpy as np

from chatter_twin.cli import main
from chatter_twin.dynamics import simulate_milling
from chatter_twin.replay import (
    DomainRandomizationConfig,
    HorizonConfig,
    LABEL_TO_ID,
    TransitionFocusConfig,
    WindowSpec,
    export_synthetic_dataset,
    slice_result_windows,
)
from chatter_twin.scenarios import make_scenario


def test_slice_result_windows_returns_aligned_records():
    modal, tool, cut, config = make_scenario("stable")
    result = simulate_milling(modal, tool, cut, config)
    windows, records = slice_result_windows(
        result=result,
        scenario="stable",
        episode=0,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        horizon=HorizonConfig(horizon_s=0.10),
    )
    assert windows
    assert len(windows) == len(records)
    assert windows[0].ndim == 2
    assert records[0].window_id == 0
    assert records[0].label_id == LABEL_TO_ID[records[0].label]
    assert records[0].scenario == "stable"
    assert records[0].window_index_in_episode == 0
    assert records[0].randomized is False
    assert records[0].spindle_scale == 1.0
    assert records[0].episode_progress == 0.0
    assert records[0].rms_growth_rate == 0.0
    assert records[0].horizon_s == 0.10
    assert records[0].horizon_label_id == LABEL_TO_ID[records[0].horizon_label]
    assert records[-1].window_index_in_episode == len(records) - 1


def test_export_synthetic_dataset_writes_schema_artifacts(tmp_path: Path):
    manifest = export_synthetic_dataset(
        scenarios=["stable", "unstable"],
        episodes=1,
        duration_s=0.25,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        out_dir=tmp_path,
    )
    assert manifest.total_windows > 0
    assert manifest.schema_version == "chatter-window-v5"
    assert manifest.domain_randomization["enabled"] is False
    assert manifest.sampling_strategy["transition_focus"]["enabled"] is False
    assert manifest.horizon["horizon_s"] == 0.20
    for name in ["dataset.npz", "windows.csv", "manifest.json", "README.md"]:
        assert (tmp_path / name).exists()

    data = np.load(tmp_path / "dataset.npz")
    assert data["sensor_windows"].ndim == 3
    assert data["sensor_windows"].shape[0] == manifest.total_windows
    assert data["labels"].shape[0] == manifest.total_windows
    assert data["horizon_labels"].shape[0] == manifest.total_windows
    assert data["horizon_label_ids"].shape[0] == manifest.total_windows
    assert data["future_chatter_within_horizon"].shape[0] == manifest.total_windows
    assert data["time_to_chatter_s"].shape[0] == manifest.total_windows
    assert set(data["channel_names"].tolist()) == {"accel_x", "accel_y"}
    assert data["randomized"].shape[0] == manifest.total_windows
    assert not data["randomized"].any()
    assert np.allclose(data["spindle_scale"], 1.0)
    assert data["episode_progress"].shape[0] == manifest.total_windows
    assert data["axial_depth_profile_scale"].shape[0] == manifest.total_windows
    assert data["cutting_coeff_profile_scale"].shape[0] == manifest.total_windows
    assert np.allclose(data["axial_depth_profile_scale"], 1.0)
    assert data["rms_growth_rate"].shape[0] == manifest.total_windows
    assert np.isfinite(data["chatter_band_energy_ewma"]).all()

    with (tmp_path / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest.total_windows
    assert rows[0]["randomized"] == "False"
    assert float(rows[0]["axial_depth_scale"]) == 1.0
    assert "rms_growth_rate" in rows[0]
    assert "chatter_band_energy_ewma" in rows[0]
    assert "axial_depth_profile_scale" in rows[0]
    assert "cutting_coeff_profile_scale" in rows[0]
    assert "horizon_label" in rows[0]
    assert "future_chatter_within_horizon" in rows[0]
    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["total_windows"] == manifest.total_windows
    assert saved_manifest["domain_randomization"]["enabled"] is False
    assert saved_manifest["sampling_strategy"]["transition_focus"]["enabled"] is False
    assert saved_manifest["horizon"]["horizon_s"] == 0.2


def test_onset_scenario_produces_lead_time_horizon_labels(tmp_path: Path):
    manifest = export_synthetic_dataset(
        scenarios=["onset"],
        episodes=1,
        duration_s=0.90,
        window_spec=WindowSpec(window_s=0.10, stride_s=0.05),
        out_dir=tmp_path,
        horizon=HorizonConfig(horizon_s=0.25),
    )
    assert manifest.total_windows > 0

    with (tmp_path / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lead_rows = [
        row
        for row in rows
        if row["future_chatter_within_horizon"] == "True" and row["label"] not in {"slight", "severe"}
    ]
    assert lead_rows
    assert any(float(row["time_to_chatter_s"]) > 0 for row in lead_rows)
    assert any(float(row["axial_depth_profile_scale"]) != 1.0 for row in rows)


def test_export_synthetic_dataset_with_randomization_records_scales(tmp_path: Path):
    manifest = export_synthetic_dataset(
        scenarios=["stable", "near_boundary", "unstable"],
        episodes=2,
        duration_s=0.25,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        out_dir=tmp_path,
        randomization=DomainRandomizationConfig(enabled=True),
    )
    assert manifest.total_windows > 0
    assert manifest.domain_randomization["enabled"] is True

    data = np.load(tmp_path / "dataset.npz")
    assert data["randomized"].all()
    assert data["axial_depth_scale"].shape[0] == manifest.total_windows
    assert np.any(np.abs(data["axial_depth_scale"] - 1.0) > 1.0e-6)
    assert np.any(np.abs(data["stiffness_scale"] - 1.0) > 1.0e-6)

    with (tmp_path / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["randomized"] == "True"
    assert any(abs(float(row["noise_scale"]) - 1.0) > 1.0e-6 for row in rows)


def test_export_synthetic_dataset_with_transition_focus_records_strategy(tmp_path: Path):
    manifest = export_synthetic_dataset(
        scenarios=["near_boundary"],
        episodes=2,
        duration_s=0.30,
        window_spec=WindowSpec(window_s=0.05, stride_s=0.05),
        out_dir=tmp_path,
        randomization=DomainRandomizationConfig(enabled=True),
        transition_focus=TransitionFocusConfig(enabled=True, candidates_per_episode=3, min_transition_windows=1),
    )
    assert manifest.total_windows > 0
    assert manifest.sampling_strategy["transition_focus"]["enabled"] is True
    assert manifest.sampling_strategy["transition_focus"]["candidates_per_episode"] == 3

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["sampling_strategy"]["transition_focus"]["enabled"] is True


def test_cli_export_synthetic_runs(tmp_path: Path):
    out_dir = tmp_path / "synthetic"
    status = main(
        [
            "export-synthetic",
            "--scenarios",
            "stable,unstable",
            "--episodes",
            "1",
            "--duration",
            "0.25",
            "--window",
            "0.05",
            "--stride",
            "0.05",
            "--out",
            str(out_dir),
            "--randomize",
            "--focus-transitions",
            "--transition-candidates",
            "2",
        ]
    )
    assert status == 0
    assert (out_dir / "dataset.npz").exists()
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["domain_randomization"]["enabled"] is True
    assert manifest["sampling_strategy"]["transition_focus"]["enabled"] is True
