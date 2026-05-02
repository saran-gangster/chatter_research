import csv
import json
from pathlib import Path

import numpy as np

from chatter_twin.cli import main
from chatter_twin.pseudo_label import PseudoLabelConfig, pseudo_label_replay_dataset


def test_pseudo_label_replay_dataset_creates_onset_labels(tmp_path: Path):
    source = tmp_path / "source"
    _write_pseudo_source(source)

    payload = pseudo_label_replay_dataset(
        dataset_dir=source,
        out_dir=tmp_path / "pseudo",
        config=PseudoLabelConfig(
            score_columns=("rms", "chatter_band_energy", "non_tooth_harmonic_ratio"),
            positive_scenarios=("chatter_trial",),
            horizon_s=0.2,
        ),
    )

    assert payload["changed_windows"] == 3
    assert payload["label_counts_after"] == {"severe": 1, "slight": 1, "stable": 4, "transition": 1}
    rows = list(csv.DictReader((tmp_path / "pseudo" / "windows.csv").open(newline="", encoding="utf-8")))
    chatter_rows = [row for row in rows if row["scenario"] == "chatter_trial"]
    assert [row["label"] for row in chatter_rows] == ["stable", "transition", "slight", "severe"]
    assert chatter_rows[0]["future_chatter_within_horizon"] == "True"
    assert float(chatter_rows[0]["time_to_chatter_s"]) == 0.2

    data = np.load(tmp_path / "pseudo" / "dataset.npz")
    assert data["labels"].tolist().count("slight") == 1
    assert data["horizon_labels"].tolist()[3] == "slight"


def test_cli_pseudo_label_replay(tmp_path: Path):
    source = tmp_path / "source"
    _write_pseudo_source(source)
    out = tmp_path / "pseudo_cli"

    status = main(
        [
            "pseudo-label-replay",
            "--dataset",
            str(source),
            "--out",
            str(out),
            "--score-columns",
            "rms,chatter_band_energy,non_tooth_harmonic_ratio",
            "--positive-scenarios",
            "chatter_trial",
            "--horizon",
            "0.2",
        ]
    )

    assert status == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_counts"]["transition"] == 1
    assert manifest["sampling_strategy"]["pseudo_labeling"]["method"] == "stable_robust_score_quantile"


def test_pseudo_label_label_mode_preserves_stable_paths_in_mixed_scenario(tmp_path: Path):
    source = tmp_path / "source"
    _write_pseudo_source_from_specs(
        source,
        [
            ("stable_trial", "stable", "0", 0.0, 1.0),
            ("stable_trial", "stable", "0", 0.1, 1.0),
            ("mixed_exp", "stable", "1", 0.0, 100000.0),
            ("mixed_exp", "slight", "2", 0.0, 1.0),
            ("mixed_exp", "slight", "2", 0.1, 1000000.0),
            ("mixed_exp", "slight", "2", 0.2, 100000000.0),
        ],
    )

    payload = pseudo_label_replay_dataset(
        dataset_dir=source,
        out_dir=tmp_path / "pseudo",
        config=PseudoLabelConfig(
            score_columns=("rms", "chatter_band_energy", "non_tooth_harmonic_ratio"),
            positive_mode="label",
            horizon_s=0.2,
        ),
    )

    assert payload["positive_mode"] == "label"
    assert payload["candidate_windows"] == 3
    rows = list(csv.DictReader((tmp_path / "pseudo" / "windows.csv").open(newline="", encoding="utf-8")))
    high_stable = next(row for row in rows if row["scenario"] == "mixed_exp" and row["episode"] == "1")
    assert high_stable["label"] == "stable"
    assert high_stable["source_label"] == "stable"
    assert high_stable["pseudo_label_candidate"] == "False"
    positive_rows = [row for row in rows if row["scenario"] == "mixed_exp" and row["episode"] == "2"]
    assert [row["label"] for row in positive_rows] == ["stable", "slight", "severe"]
    assert all(row["source_label"] == "slight" for row in positive_rows)


def _write_pseudo_source(path: Path) -> None:
    specs = [
        ("stable_trial", "stable", "0", 0.0, 1.0),
        ("stable_trial", "stable", "0", 0.1, 1.0),
        ("stable_trial", "stable", "0", 0.2, 1.0),
        ("chatter_trial", "slight", "1", 0.0, 1.0),
        ("chatter_trial", "slight", "1", 0.1, 31.6227766017),
        ("chatter_trial", "slight", "1", 0.2, 1000.0),
        ("chatter_trial", "slight", "1", 0.3, 100000.0),
    ]
    _write_pseudo_source_from_specs(path, specs)


def _write_pseudo_source_from_specs(path: Path, specs: list[tuple[str, str, str, float, float]]) -> None:
    path.mkdir(parents=True)
    rows = []
    for idx, (scenario, label, episode, start, value) in enumerate(specs):
        rows.append(
            {
                "window_id": str(idx),
                "scenario": scenario,
                "episode": episode,
                "start_time_s": str(start),
                "end_time_s": str(start + 0.09),
                "label": label,
                "label_id": "0" if label == "stable" else "2",
                "horizon_label": label,
                "horizon_label_id": "0" if label == "stable" else "2",
                "horizon_s": "0.2",
                "chatter_within_horizon": str(label == "slight"),
                "future_chatter_within_horizon": "False",
                "time_to_chatter_s": "0.0" if label == "slight" else "-1.0",
                "rms": str(value),
                "chatter_band_energy": str(value),
                "non_tooth_harmonic_ratio": str(value),
            }
        )
    with (path / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        path / "dataset.npz",
        sensor_windows=np.zeros((len(rows), 4, 2), dtype=np.float32),
        labels=np.array([row["label"] for row in rows]),
        label_ids=np.array([int(row["label_id"]) for row in rows], dtype=np.int64),
        horizon_labels=np.array([row["horizon_label"] for row in rows]),
        horizon_label_ids=np.array([int(row["horizon_label_id"]) for row in rows], dtype=np.int64),
        chatter_within_horizon=np.array([row["chatter_within_horizon"] == "True" for row in rows], dtype=np.bool_),
        future_chatter_within_horizon=np.zeros(len(rows), dtype=np.bool_),
        time_to_chatter_s=np.array([float(row["time_to_chatter_s"]) for row in rows], dtype=np.float32),
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "chatter-window-v5",
                "sample_rate_hz": 10.0,
                "window_s": 0.1,
                "stride_s": 0.1,
                "channel_names": ["x", "y"],
                "scenarios": ["stable_trial", "chatter_trial"],
                "episodes_per_scenario": 1,
                "total_windows": len(rows),
                "label_counts": {"stable": 3, "slight": 4},
                "domain_randomization": {"enabled": False},
                "sampling_strategy": {},
                "horizon": {"horizon_s": 0.2},
                "artifacts": ["dataset.npz", "windows.csv", "manifest.json"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
