from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from chatter_twin.features import extract_signal_features
from chatter_twin.replay import (
    CHATTER_POSITIVE_LABELS,
    CHANNEL_NAMES,
    LABEL_TO_ID,
    DatasetManifest,
    HorizonConfig,
    WindowRecord,
    WindowSpec,
    attach_horizon_targets,
    result_sample_rate,
)
from chatter_twin.risk import estimate_chatter_risk

ICNC_RECORD_URL = "https://zenodo.org/records/15308467"
ICNC_API_URL = "https://zenodo.org/api/records/15308467"
ICNC_DOWNLOAD_URL = "https://zenodo.org/records/15308467/files/i-CNC%20Dataset.zip?download=1"
ICNC_FILENAME = "i-CNC Dataset.zip"
ICNC_SIZE_BYTES = 2_989_101_603
ICNC_MD5 = "5edb62f27b89159f804f423134bb6ac3"

KIT_INDUSTRIAL_RECORD_URL = "https://radar.kit.edu/radar/en/dataset/hvvwn1kfwf7qt48z"
KIT_INDUSTRIAL_METADATA_URL = "https://radar.kit.edu/radar/en/export/hvvwn1kfwf7qt48z/exportdatacite"
KIT_INDUSTRIAL_DOWNLOAD_URL = "https://radar.kit.edu/radar-backend/archives/hvvwn1kfwf7qt48z/versions/1/content"
KIT_INDUSTRIAL_FILENAME = "10.35097-hvvwn1kfwf7qt48z.tar"
KIT_INDUSTRIAL_SIZE_BYTES = 44_584_092_672
KIT_INDUSTRIAL_DOI = "10.35097/hvvwn1kfwf7qt48z"

MT_CUTTING_GITHUB_URL = "https://github.com/purduelamm/mt_cutting_dataset"
MT_CUTTING_DATASET_NAME = "CNC Machine Tool Cutting Sound Dataset"


@dataclass(frozen=True)
class ICNCIngestConfig:
    window_s: float = 0.10
    stride_s: float = 0.05
    horizon_s: float = 0.25
    flute_count: int = 4
    modal_frequency_hz: float | None = None
    default_sample_rate_hz: float = 10_000.0
    default_spindle_rpm: float = 9_000.0
    default_feed_per_tooth_m: float = 45.0e-6
    include_unknown: bool = False
    max_packages_per_file: int | None = None
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be positive")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if self.flute_count < 1:
            raise ValueError("flute_count must be at least 1")
        if self.default_sample_rate_hz <= 0:
            raise ValueError("default_sample_rate_hz must be positive")
        if self.default_spindle_rpm <= 0:
            raise ValueError("default_spindle_rpm must be positive")
        if self.default_feed_per_tooth_m <= 0:
            raise ValueError("default_feed_per_tooth_m must be positive")
        if self.max_packages_per_file is not None and self.max_packages_per_file < 1:
            raise ValueError("max_packages_per_file must be at least 1")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be at least 1")


@dataclass(frozen=True)
class KITIndustrialIngestConfig:
    window_s: float = 1.0
    stride_s: float = 0.5
    horizon_s: float = 2.0
    sample_rate_hz: float = 500.0
    signal_columns: tuple[str, str] = ("LOAD|6", "CURRENT|6")
    include_other_anomalies: bool = False
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be positive")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if len(self.signal_columns) != 2:
            raise ValueError("signal_columns must contain exactly two columns")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be at least 1")


@dataclass(frozen=True)
class KITMatIngestConfig:
    window_s: float = 0.10
    stride_s: float = 0.05
    horizon_s: float = 0.25
    signal_names: tuple[str, ...] = ("xAcceleration", "yAcceleration")
    standardize_signals: bool = False
    include_other_anomalies: bool = False
    max_windows: int | None = None
    max_samples_per_trial: int | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be positive")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if not self.signal_names:
            raise ValueError("signal_names cannot be empty")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be at least 1")
        if self.max_samples_per_trial is not None and self.max_samples_per_trial < 4:
            raise ValueError("max_samples_per_trial must be at least 4")


@dataclass(frozen=True)
class MTCuttingIngestConfig:
    window_s: float = 0.10
    stride_s: float = 0.05
    horizon_s: float = 0.25
    sensors: tuple[int, ...] = (0, 1, 2)
    include_unknown: bool = False
    max_experiments: int | None = None
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be positive")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if not self.sensors:
            raise ValueError("sensors cannot be empty")
        if self.max_experiments is not None and self.max_experiments < 1:
            raise ValueError("max_experiments must be at least 1")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be at least 1")


def icnc_source_manifest() -> dict[str, object]:
    return {
        "dataset": "i-CNC Use Case: Vibration data and associated Chatter indication during milling process with CNC machines",
        "record_url": ICNC_RECORD_URL,
        "api_url": ICNC_API_URL,
        "download_url": ICNC_DOWNLOAD_URL,
        "filename": ICNC_FILENAME,
        "size_bytes": ICNC_SIZE_BYTES,
        "md5": ICNC_MD5,
        "license": "CC-BY-4.0",
        "published": "2025-04-30",
        "files_inside_zip": ("test1_with_status.csv", "test2_with_status.csv"),
        "columns": {
            "timestamp": "measurement timestamp",
            "fs": "DAQ sampling frequency in Hz",
            "numscans": "samples generated in the measurement package",
            "spindlespeed": "spindle speed in RPM",
            "x_channel": "X-axis acceleration samples at 10 kHz, m/s^2",
            "y_channel": "Y-axis acceleration samples at 10 kHz, m/s^2",
            "status": "AI-generated chatter occurrence indicator",
        },
    }


def write_icnc_source_manifest(path: Path) -> dict[str, object]:
    manifest = icnc_source_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def kit_industrial_source_manifest() -> dict[str, object]:
    return {
        "dataset": "A Multimodal Dataset for Process Monitoring and Anomaly Detection in Industrial CNC Milling",
        "record_url": KIT_INDUSTRIAL_RECORD_URL,
        "metadata_url": KIT_INDUSTRIAL_METADATA_URL,
        "download_url": KIT_INDUSTRIAL_DOWNLOAD_URL,
        "filename": KIT_INDUSTRIAL_FILENAME,
        "size_bytes": KIT_INDUSTRIAL_SIZE_BYTES,
        "doi": KIT_INDUSTRIAL_DOI,
        "license": "CC-BY-4.0",
        "published": "2025-07-01",
        "machine": "DMC 60 H with Siemens SINUMERIK 840D",
        "experiments": 33,
        "duration": "about 6 hours",
        "modalities": {
            "controller": "SINUMERIK Edge JSON/CSV at 500 Hz",
            "external_sensors": "force and acceleration MATLAB files at 10 kHz",
            "synchronized": "combined MATLAB timetable per experiment",
            "context": "NC programs, CAD/STP models, design-of-experiment documentation",
        },
        "bridge_value": (
            "continuous multimodal process-context data with explicit anomaly types, "
            "including chatter; suitable for signal/context validation before real closed-loop CNC data"
        ),
        "download_note": "The public content URL is a 44.6 GB tar; use a resumable tool such as curl -L -C -.",
    }


def write_kit_industrial_source_manifest(path: Path) -> dict[str, object]:
    manifest = kit_industrial_source_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def mt_cutting_source_manifest() -> dict[str, object]:
    return {
        "dataset": MT_CUTTING_DATASET_NAME,
        "repository_url": MT_CUTTING_GITHUB_URL,
        "license": "unspecified in importer; verify before publication",
        "modalities": {
            "IMI": "laboratory CNC milling sound, three synchronized 48 kHz sensors, cutting intervals, operator chatter labels",
            "KRPM": "industry CNC machining sound and cutting intervals, no detailed chatter labels in the IMI workbook",
        },
        "bridge_value": (
            "many independent labeled sound cuts for real-signal current-window chatter validation; "
            "not closed-loop intervention data and not a calibrated force/FRF process twin"
        ),
    }


def write_mt_cutting_source_manifest(path: Path) -> dict[str, object]:
    manifest = mt_cutting_source_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def download_icnc_dataset(
    *,
    out_path: Path,
    manifest_path: Path | None = None,
    skip_existing: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    if out_path.is_dir():
        out_path = out_path / ICNC_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        write_icnc_source_manifest(manifest_path)

    if skip_existing and out_path.exists() and out_path.stat().st_size == ICNC_SIZE_BYTES:
        md5 = _md5_file(out_path)
        return _download_payload(out_path, md5, downloaded=False)

    digest = hashlib.md5()
    downloaded_bytes = 0
    with urllib.request.urlopen(ICNC_DOWNLOAD_URL) as response, out_path.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            downloaded_bytes += len(chunk)

    md5 = digest.hexdigest()
    if downloaded_bytes != ICNC_SIZE_BYTES:
        raise ValueError(f"Downloaded {downloaded_bytes} bytes, expected {ICNC_SIZE_BYTES}")
    if md5 != ICNC_MD5:
        raise ValueError(f"Downloaded file md5 {md5} does not match expected {ICNC_MD5}")
    return _download_payload(out_path, md5, downloaded=True)


def ingest_icnc_dataset(
    *,
    source: Path,
    out_dir: Path,
    config: ICNCIngestConfig | None = None,
) -> dict[str, object]:
    config = config or ICNCIngestConfig()
    csv_sources = discover_icnc_csv_sources(source)
    if not csv_sources:
        raise ValueError(f"No i-CNC CSV files found under {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_windows: list[NDArray[np.float64]] = []
    all_records: list[WindowRecord] = []
    source_summaries: list[dict[str, object]] = []
    next_episode_id = 0

    for csv_source in csv_sources:
        if config.max_windows is not None and len(all_records) >= config.max_windows:
            break
        remaining_windows = None if config.max_windows is None else config.max_windows - len(all_records)
        windows, records, summary = _ingest_icnc_csv(
            csv_source,
            episode_start=next_episode_id,
            starting_window_id=len(all_records),
            config=config,
            max_windows=remaining_windows,
        )
        all_windows.extend(windows)
        all_records.extend(records)
        source_summaries.append(summary)
        next_episode_id += int(summary["packages_kept"])

    if not all_records:
        raise ValueError("No replay windows produced; reduce the window size or raise package/window limits")

    sensor_windows = np.stack(all_windows).astype(np.float32)
    manifest = _icnc_manifest(all_records, csv_sources, source_summaries, config, sensor_windows)
    _write_real_replay_dataset(out_dir, all_records, sensor_windows, manifest)
    _write_icnc_readme(out_dir / "README.md", manifest, source_summaries)
    return {
        "manifest": asdict(manifest),
        "sources": source_summaries,
        "out_dir": str(out_dir),
        "artifacts": list(manifest.artifacts),
    }


def inspect_kit_industrial_dataset(source: Path) -> dict[str, object]:
    """Inspect the KIT/RADAR payload without extracting the full dataset."""

    with _open_kit_source(source) as kit:
        members = kit.members()
        trials = _kit_doe_trials(kit)
    trial_count = len(trials)
    chatter_trials = [trial for trial in trials if _kit_comment_to_label(trial.get("comment", "")) in CHATTER_POSITIVE_LABELS]
    component_counts = Counter(str(trial["component"]) for trial in trials)
    return {
        "source": str(source),
        "members": len(members),
        "processed_hfdata_files": sum(1 for member in members if member.endswith("_hfdata.csv")),
        "synchronized_mat_files": sum(1 for member in members if member.endswith("_synchronized.mat")),
        "trials": trial_count,
        "component_counts": dict(sorted(component_counts.items())),
        "chatter_trials": chatter_trials,
        "trial_comments": [
            {
                "trial": trial["trial"],
                "component": trial["component"],
                "comment": trial.get("comment", ""),
            }
            for trial in trials
            if trial.get("comment")
        ],
    }


def inspect_kit_synchronized_mat(
    *,
    source: Path,
    trial: str,
    max_datasets: int = 200,
) -> dict[str, object]:
    """Inspect one KIT synchronized MATLAB v7.3 file and list HDF5 datasets."""

    if max_datasets < 1:
        raise ValueError("max_datasets must be at least 1")
    with _open_kit_source(source) as kit:
        trial_record = _kit_trial_by_id(kit, trial)
        member = _kit_synchronized_mat_member(trial_record)
        with _open_kit_mat_h5(kit, member) as handle:
            datasets = _h5_dataset_summary(handle, max_datasets=max_datasets)
            root_keys = sorted(str(key) for key in handle.keys())
    return {
        "source": str(source),
        "trial": trial,
        "member": member,
        "root_keys": root_keys,
        "datasets_listed": len(datasets),
        "datasets_truncated": len(datasets) >= max_datasets,
        "datasets": datasets,
        "candidate_numeric_time_series": [
            item
            for item in datasets
            if item["kind"] == "numeric"
            and len(item["shape"]) >= 1
            and max(item["shape"] or [0]) >= 100
        ],
    }


def ingest_kit_industrial_dataset(
    *,
    source: Path,
    out_dir: Path,
    trials: list[str] | None = None,
    config: KITIndustrialIngestConfig | None = None,
) -> dict[str, object]:
    """Import KIT industrial controller replay windows from processed hfdata CSV files."""

    config = config or KITIndustrialIngestConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {trial.strip() for trial in trials or [] if trial.strip()}
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    summaries: list[dict[str, object]] = []

    with _open_kit_source(source) as kit:
        doe_trials = _kit_doe_trials(kit)
        trial_by_id = {str(trial["trial"]): trial for trial in doe_trials}
        if selected:
            missing = sorted(selected - set(trial_by_id))
            if missing:
                raise ValueError(f"KIT trials not found in DoE: {', '.join(missing)}")
            candidates = [trial_by_id[trial_id] for trial_id in sorted(selected)]
        else:
            candidates = [
                trial
                for trial in doe_trials
                if _kit_comment_to_label(trial.get("comment", "")) != "unknown"
                or bool(config.include_other_anomalies)
            ]

        for episode_id, trial in enumerate(candidates):
            if config.max_windows is not None and len(records) >= config.max_windows:
                break
            label = _kit_comment_to_label(trial.get("comment", ""))
            if label == "unknown" and not config.include_other_anomalies:
                continue
            remaining = None if config.max_windows is None else config.max_windows - len(records)
            trial_windows, trial_records, summary = _ingest_kit_trial(
                kit,
                trial=trial,
                label=label,
                episode=episode_id,
                starting_window_id=len(records),
                config=config,
                max_windows=remaining,
            )
            windows.extend(trial_windows)
            records.extend(trial_records)
            summaries.append(summary)

    if not records:
        raise ValueError("No KIT replay windows produced; select at least one stable or chatter trial")

    sensor_windows = np.stack(windows).astype(np.float32)
    manifest = _kit_manifest(records, summaries, config, sensor_windows)
    _write_real_replay_dataset(out_dir, records, sensor_windows, manifest)
    _write_kit_readme(out_dir / "README.md", manifest, summaries, config)
    return {
        "manifest": asdict(manifest),
        "sources": summaries,
        "out_dir": str(out_dir),
        "artifacts": list(manifest.artifacts),
    }


def ingest_kit_mat_dataset(
    *,
    source: Path,
    out_dir: Path,
    trials: list[str] | None = None,
    config: KITMatIngestConfig | None = None,
) -> dict[str, object]:
    """Import KIT synchronized MATLAB acceleration/force replay windows."""

    config = config or KITMatIngestConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {trial.strip() for trial in trials or [] if trial.strip()}
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    summaries: list[dict[str, object]] = []

    with _open_kit_source(source) as kit:
        doe_trials = _kit_doe_trials(kit)
        trial_by_id = {str(trial["trial"]): trial for trial in doe_trials}
        if selected:
            missing = sorted(selected - set(trial_by_id))
            if missing:
                raise ValueError(f"KIT trials not found in DoE: {', '.join(missing)}")
            candidates = [trial_by_id[trial_id] for trial_id in sorted(selected)]
        else:
            candidates = [
                trial
                for trial in doe_trials
                if _kit_comment_to_label(trial.get("comment", "")) != "unknown"
                or bool(config.include_other_anomalies)
            ]

        for episode_id, trial in enumerate(candidates):
            if config.max_windows is not None and len(records) >= config.max_windows:
                break
            label = _kit_comment_to_label(trial.get("comment", ""))
            if label == "unknown" and not config.include_other_anomalies:
                continue
            remaining = None if config.max_windows is None else config.max_windows - len(records)
            trial_windows, trial_records, summary = _ingest_kit_mat_trial(
                kit,
                trial=trial,
                label=label,
                episode=episode_id,
                starting_window_id=len(records),
                config=config,
                max_windows=remaining,
            )
            windows.extend(trial_windows)
            records.extend(trial_records)
            summaries.append(summary)

    if not records:
        raise ValueError("No KIT MAT replay windows produced; select at least one stable or chatter trial")

    sensor_windows = np.stack(windows).astype(np.float32)
    manifest = _kit_mat_manifest(records, summaries, config, sensor_windows)
    _write_real_replay_dataset(out_dir, records, sensor_windows, manifest)
    _write_kit_mat_readme(out_dir / "README.md", manifest, summaries, config)
    return {
        "manifest": asdict(manifest),
        "sources": summaries,
        "out_dir": str(out_dir),
        "artifacts": list(manifest.artifacts),
    }


def inspect_mt_cutting_dataset(source: Path) -> dict[str, object]:
    """Inspect the Purdue CNC cutting sound dataset layout and IMI labels."""

    labels = _read_mt_label_workbook(source / "IMI" / "labeling_all_details.xlsx")
    experiment_dirs = _mt_experiment_dirs(source)
    experiments: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        cutting_intervals = _read_cutting_intervals(exp_dir / "cutting.csv")
        exp_labels = labels.get(exp_name, [])
        chatter_values = [_mt_row_chatter_value(row) for row in exp_labels]
        for value in chatter_values:
            label_counts[_mt_chatter_value_to_label(value)] += 1
        wavs = sorted(path.name for path in exp_dir.glob("*_sensor*.wav"))
        experiments.append(
            {
                "experiment": exp_name,
                "cutting_intervals": len(cutting_intervals),
                "label_rows": len(exp_labels),
                "wav_files": wavs,
                "label_counts": dict(sorted(Counter(_mt_chatter_value_to_label(value) for value in chatter_values).items())),
            }
        )
    return {
        "source": str(source),
        "repository_url": MT_CUTTING_GITHUB_URL,
        "experiments": len(experiment_dirs),
        "labeled_experiments": len(labels),
        "label_counts": dict(sorted(label_counts.items())),
        "experiment_summaries": experiments,
    }


def ingest_mt_cutting_dataset(
    *,
    source: Path,
    out_dir: Path,
    experiments: list[str] | None = None,
    config: MTCuttingIngestConfig | None = None,
) -> dict[str, object]:
    """Import Purdue IMI cutting-sound windows into the replay schema."""

    config = config or MTCuttingIngestConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {experiment.strip() for experiment in experiments or [] if experiment.strip()}
    labels_by_exp = _read_mt_label_workbook(source / "IMI" / "labeling_all_details.xlsx")
    candidates = _mt_experiment_dirs(source)
    if selected:
        available = {path.name for path in candidates}
        missing = sorted(selected - available)
        if missing:
            raise ValueError(f"MT cutting experiments not found: {', '.join(missing)}")
        candidates = [path for path in candidates if path.name in selected]
    if config.max_experiments is not None:
        candidates = candidates[: config.max_experiments]

    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    summaries: list[dict[str, object]] = []
    episode_id = 0

    for exp_dir in candidates:
        if config.max_windows is not None and len(records) >= config.max_windows:
            break
        exp_name = exp_dir.name
        label_rows = labels_by_exp.get(exp_name)
        if not label_rows:
            continue
        intervals = _read_cutting_intervals(exp_dir / "cutting.csv")
        if not intervals:
            continue
        sample_rate_hz, signal = _read_mt_audio_matrix(exp_dir, config.sensors)
        path_summaries: list[dict[str, object]] = []
        for path_idx, interval in enumerate(intervals):
            if config.max_windows is not None and len(records) >= config.max_windows:
                break
            label_row = label_rows[path_idx] if path_idx < len(label_rows) else {}
            chatter_value = _mt_row_chatter_value(label_row)
            label = _mt_chatter_value_to_label(chatter_value)
            if label == "unknown" and not config.include_unknown:
                continue
            start_sample = max(0, int(round(interval["start_s"] * sample_rate_hz)))
            stop_sample = min(signal.shape[0], int(round(interval["end_s"] * sample_rate_hz)))
            if stop_sample <= start_sample:
                continue
            interval_signal = signal[start_sample:stop_sample]
            remaining = None if config.max_windows is None else config.max_windows - len(records)
            path_windows, path_records, _ = _slice_icnc_package(
                sensor_signal=interval_signal,
                scenario=exp_name,
                episode=episode_id,
                package_index=path_idx,
                starting_window_id=len(records),
                starting_window_index=0,
                sample_rate_hz=sample_rate_hz,
                spindle_rpm=max(_mt_float(label_row, "Spindle speed [RPM]", 1.0), 1.0),
                label=label,
                previous=_TemporalState(),
                config=_mt_slice_config(label_row, sample_rate_hz, config),
                max_windows=remaining,
                feed_per_tooth_m=_mt_feed_per_tooth_m(label_row),
                axial_depth_m=_mt_depth_m(label_row, ("Depth of cut [inch]", "Depth of Cut [inch]", "Depth [inch]")),
                radial_depth_m=_mt_depth_m(label_row, ("Width of cut [inch]", "Radial depth of cut [inch]", "Width [inch]")),
            )
            path_records = attach_horizon_targets(path_records, HorizonConfig(horizon_s=config.horizon_s))
            windows.extend(path_windows)
            records.extend(path_records)
            path_summaries.append(
                {
                    "path_index": path_idx + 1,
                    "start_s": interval["start_s"],
                    "end_s": interval["end_s"],
                    "label": label,
                    "chatter_value": chatter_value,
                    "windows": len(path_records),
                }
            )
            episode_id += 1
        summaries.append(
            {
                "experiment": exp_name,
                "sample_rate_hz": sample_rate_hz,
                "sensors": list(config.sensors),
                "cutting_intervals": len(intervals),
                "label_rows": len(label_rows),
                "paths_imported": len(path_summaries),
                "windows": sum(int(item["windows"]) for item in path_summaries),
                "label_counts": dict(sorted(Counter(item["label"] for item in path_summaries).items())),
                "paths": path_summaries,
            }
        )

    if not records:
        raise ValueError("No MT cutting replay windows produced; check experiment selection and window size")

    sensor_windows = np.stack(windows).astype(np.float32)
    manifest = _mt_cutting_manifest(records, summaries, config, sensor_windows)
    _write_real_replay_dataset(out_dir, records, sensor_windows, manifest)
    _write_mt_cutting_readme(out_dir / "README.md", manifest, summaries, config)
    return {
        "manifest": asdict(manifest),
        "sources": summaries,
        "out_dir": str(out_dir),
        "artifacts": list(manifest.artifacts),
    }


def _mt_experiment_dirs(source: Path) -> list[Path]:
    imi_dir = source / "IMI"
    if not imi_dir.exists():
        raise ValueError(f"{source} does not contain an IMI directory")
    return sorted(path for path in imi_dir.iterdir() if path.is_dir() and (path / "cutting.csv").exists())


def _read_mt_label_workbook(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        raise ValueError(f"MT cutting label workbook not found: {path}")
    sheets = _read_simple_xlsx(path)
    parsed: dict[str, list[dict[str, str]]] = {}
    for sheet_name, rows in sheets.items():
        if sheet_name == "Summary" or not rows:
            continue
        headers = [str(value).strip() for value in rows[0]]
        sheet_rows: list[dict[str, str]] = []
        for row in rows[1:]:
            if not any(str(value).strip() for value in row):
                continue
            record = {header: str(row[idx]).strip() if idx < len(row) else "" for idx, header in enumerate(headers) if header}
            if record.get("No."):
                sheet_rows.append(record)
        parsed[sheet_name] = sheet_rows
    return parsed


def _read_simple_xlsx(path: Path) -> dict[str, list[list[str]]]:
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns = {"a": spreadsheet_ns}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared = _xlsx_shared_strings(archive, ns)
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall("a:sheets/a:sheet", ns):
            name = str(sheet.attrib["name"])
            rel_id = sheet.attrib[f"{{{rel_ns}}}id"]
            target = relmap[rel_id]
            sheet_path = "xl/" + target if not target.startswith("/") else target[1:]
            root = ET.fromstring(archive.read(sheet_path))
            rows: list[list[str]] = []
            for xml_row in root.findall("a:sheetData/a:row", ns):
                values: dict[int, str] = {}
                for cell in xml_row.findall("a:c", ns):
                    ref = cell.attrib.get("r", "A0")
                    match = re.match(r"([A-Z]+)([0-9]+)", ref)
                    if not match:
                        continue
                    col_idx = _excel_col_to_index(match.group(1))
                    values[col_idx] = _xlsx_cell_value(cell, shared, ns)
                if values:
                    rows.append([values.get(idx, "") for idx in range(max(values) + 1)])
            sheets[name] = rows
    return sheets


def _xlsx_shared_strings(archive: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared: list[str] = []
    text_tag = f"{{{ns['a']}}}t"
    for item in root.findall("a:si", ns):
        shared.append("".join(text.text or "" for text in item.iter(text_tag)))
    return shared


def _xlsx_cell_value(cell: ET.Element, shared: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text = cell.find("a:is/a:t", ns)
        return "" if text is None else text.text or ""
    value = cell.find("a:v", ns)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def _excel_col_to_index(column: str) -> int:
    out = 0
    for char in column:
        out = out * 26 + ord(char) - ord("A") + 1
    return out - 1


def _read_cutting_intervals(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        intervals: list[dict[str, float]] = []
        for row in reader:
            start = _first_float(row, ("start", "Start", "START"))
            end = _first_float(row, ("end", "End", "END"))
            if start is None or end is None or end <= start:
                continue
            intervals.append({"start_s": start, "end_s": end})
    return intervals


def _read_mt_audio_matrix(exp_dir: Path, sensors: tuple[int, ...]) -> tuple[float, NDArray[np.float64]]:
    channels: list[NDArray[np.float64]] = []
    sample_rates: list[int] = []
    for sensor in sensors:
        matches = sorted(exp_dir.glob(f"*_sensor{sensor}.wav"))
        if not matches:
            raise ValueError(f"{exp_dir} is missing sensor {sensor} WAV")
        sample_rate, data = wavfile.read(matches[0])
        sample_rates.append(int(sample_rate))
        channels.append(_wav_to_float_mono(data))
    if len(set(sample_rates)) != 1:
        raise ValueError(f"{exp_dir} has mismatched sensor sample rates: {sample_rates}")
    sample_count = min(channel.size for channel in channels)
    if sample_count < 4:
        raise ValueError(f"{exp_dir} audio is too short")
    signal = np.column_stack([channel[:sample_count] for channel in channels])
    return float(sample_rates[0]), signal


def _wav_to_float_mono(data: NDArray[np.generic]) -> NDArray[np.float64]:
    array = np.asarray(data)
    if array.ndim == 2:
        array = np.mean(array.astype(np.float64), axis=1)
    else:
        array = array.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(max(abs(np.iinfo(data.dtype).min), abs(np.iinfo(data.dtype).max)))
        array = array / max_abs
    return array - float(np.mean(array))


def _mt_row_chatter_value(row: dict[str, str]) -> float:
    op1 = _mt_float(row, "Chatter (operator 1)", 0.0)
    op2 = _mt_float(row, "Chatter (operator 2)", 0.0)
    return max(op1, op2)


def _mt_chatter_value_to_label(value: float) -> str:
    if value <= 0:
        return "stable"
    if value < 2:
        return "slight"
    return "severe"


def _mt_slice_config(row: dict[str, str], sample_rate_hz: float, config: MTCuttingIngestConfig) -> ICNCIngestConfig:
    spindle_rpm = max(_mt_float(row, "Spindle speed [RPM]", 1.0), 1.0)
    return ICNCIngestConfig(
        window_s=config.window_s,
        stride_s=config.stride_s,
        horizon_s=config.horizon_s,
        flute_count=_mt_flute_count(row.get("Tool", "")),
        modal_frequency_hz=None,
        default_sample_rate_hz=sample_rate_hz,
        default_spindle_rpm=spindle_rpm,
        default_feed_per_tooth_m=_mt_feed_per_tooth_m(row),
    )


def _mt_feed_per_tooth_m(row: dict[str, str]) -> float:
    rpm = max(_mt_float(row, "Spindle speed [RPM]", 1.0), 1.0)
    feed_ipm = _mt_float(row, "Feedrate [IPM]", 0.0)
    flutes = _mt_flute_count(row.get("Tool", ""))
    feed_m_per_min = feed_ipm * 0.0254
    if feed_m_per_min <= 0:
        return 1.0e-9
    return max(feed_m_per_min / (rpm * flutes), 1.0e-9)


def _mt_depth_m(row: dict[str, str], columns: tuple[str, ...]) -> float:
    value = _first_float(row, columns)
    return 0.0 if value is None else max(value * 0.0254, 0.0)


def _mt_flute_count(tool: str) -> int:
    match = re.search(r"(\d+)\s*[- ]?\s*flute", tool, flags=re.IGNORECASE)
    if not match:
        return 4
    return max(int(match.group(1)), 1)


def _first_float(row: dict[str, str], columns: tuple[str, ...]) -> float | None:
    for column in columns:
        if column in row:
            try:
                return float(str(row[column]).strip())
            except ValueError:
                continue
    return None


def _mt_float(row: dict[str, str], column: str, default: float) -> float:
    value = _first_float(row, (column,))
    return default if value is None else value


class _KITSource:
    def __init__(self, source: Path):
        self.source = source
        self._archive: zipfile.ZipFile | None = None

    def __enter__(self) -> "_KITSource":
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._archive = zipfile.ZipFile(self.source)
        elif not self.source.exists():
            raise ValueError(f"KIT source does not exist: {self.source}")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._archive is not None:
            self._archive.close()

    def members(self) -> list[str]:
        if self._archive is not None:
            return sorted(self._archive.namelist())
        root = _kit_data_root(self.source)
        return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file())

    def read_bytes(self, member: str) -> bytes:
        if self._archive is not None:
            return self._archive.read(member)
        root = _kit_data_root(self.source)
        return (root / member).read_bytes()

    def filesystem_member_path(self, member: str) -> Path | None:
        if self._archive is not None:
            return None
        root = _kit_data_root(self.source)
        candidate = root / member
        if candidate.exists():
            return candidate
        if member.startswith("Data/"):
            candidate = root / member.removeprefix("Data/")
            if candidate.exists():
                return candidate
        return None

    @contextmanager
    def open_text(self, member: str) -> Iterator[object]:
        if self._archive is not None:
            with self._archive.open(member, "r") as raw:
                wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                try:
                    yield wrapper
                finally:
                    wrapper.detach()
            return
        root = _kit_data_root(self.source)
        with (root / member).open(newline="", encoding="utf-8-sig") as handle:
            yield handle


def _open_kit_source(source: Path) -> _KITSource:
    return _KITSource(source)


def _kit_data_root(source: Path) -> Path:
    if source.name == "Data" and source.is_dir():
        return source
    if (source / "Data").is_dir():
        return source / "Data"
    return source


def _kit_doe_trials(kit: _KITSource) -> list[dict[str, object]]:
    rows = _read_xlsx_rows(kit.read_bytes("Data/Descriptive/DoE/DoE.xlsx" if kit._archive is not None else "Descriptive/DoE/DoE.xlsx"))
    trials: list[dict[str, object]] = []
    component: str | None = None
    headers: list[str] | None = None
    for row in rows:
        first = row[0].strip() if row else ""
        if first in {"Thermoforming mold", "Injection mold", "Impeller"}:
            component = first
            headers = None
            continue
        if first == "Trial":
            headers = row
            continue
        if component is None or headers is None or not first:
            continue
        values = {headers[idx]: row[idx] for idx in range(min(len(headers), len(row))) if headers[idx]}
        trial = {
            "trial": first,
            "component": component,
            "comment": values.get("Comment", ""),
            "nc_code": values.get("NC-Code", ""),
            "spindle_rpm": _positive_float(values.get("Spindle Speed [U/min]"), 0.0),
            "feedrate_mm_min": _positive_float(values.get("Feedrate [mm/min]"), 0.0),
            "feed_per_tooth_mm": _positive_float(values.get("fz [mm]"), 0.0),
            "flutes": int(round(_positive_float(values.get("Number of teeth"), 1.0))),
            "axial_depth_mm": _positive_float(
                values.get("Cutting depth [mm]") or values.get("Optimal Cutting depth [mm]"),
                0.0,
            ),
            "radial_depth_mm": _kit_overlap_midpoint(
                values.get("Overlap [mm]") or values.get("Optimal Overlap [mm]"),
            ),
            "material": values.get("Material", ""),
            "tool": values.get("Tools", ""),
        }
        trials.append(trial)
    return trials


def _kit_trial_by_id(kit: _KITSource, trial_id: str) -> dict[str, object]:
    trials = _kit_doe_trials(kit)
    for trial in trials:
        if str(trial["trial"]) == trial_id:
            return trial
    raise ValueError(f"KIT trial not found in DoE: {trial_id}")


def _read_xlsx_rows(payload: bytes) -> list[list[str]]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(io.BytesIO(payload)) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))
        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        rows: list[list[str]] = []
        for sheet in book.findall("a:sheets/a:sheet", ns):
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            root = ET.fromstring(workbook.read("xl/" + relmap[rel_id].lstrip("/")))
            for row in root.findall("a:sheetData/a:row", ns):
                values: list[str] = []
                for cell in row.findall("a:c", ns):
                    idx = _xlsx_column_index(cell.attrib.get("r", ""))
                    while len(values) <= idx:
                        values.append("")
                    value = cell.find("a:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                    values[idx] = text
                rows.append(values)
        return rows


def _xlsx_column_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - 64
    return index - 1


def _kit_overlap_midpoint(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not numbers:
        return 0.0
    return float(sum(numbers) / len(numbers))


def _kit_comment_to_label(comment: object) -> str:
    text = str(comment or "").strip().lower()
    if not text:
        return "stable"
    if "chatter" in text:
        return "slight"
    if text in {"roughing", "finishing"}:
        return "stable"
    return "unknown"


def _kit_hfdata_member(trial: dict[str, object]) -> str:
    component = str(trial["component"])
    trial_id = str(trial["trial"])
    return f"Data/Dataset/{component}/{trial_id}/processed_data/{trial_id}_hfdata.csv"


def _kit_synchronized_mat_member(trial: dict[str, object]) -> str:
    component = str(trial["component"])
    trial_id = str(trial["trial"])
    return f"Data/Dataset/{component}/{trial_id}/processed_data/{trial_id}_synchronized.mat"


@contextmanager
def _open_kit_mat_h5(kit: _KITSource, member: str) -> Iterator[object]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install the `mat` extra to inspect KIT MATLAB v7.3 files: uv run --extra mat ...") from exc

    filesystem_path = kit.filesystem_member_path(member)
    if filesystem_path is not None:
        with h5py.File(filesystem_path, "r") as handle:
            yield handle
        return

    payload = kit.read_bytes(member)
    with h5py.File(io.BytesIO(payload), "r") as handle:
        yield handle


def _h5_dataset_summary(handle: object, *, max_datasets: int) -> list[dict[str, object]]:
    import h5py

    datasets: list[dict[str, object]] = []

    def visitor(path: str, obj: object) -> None:
        if len(datasets) >= max_datasets:
            return
        if not isinstance(obj, h5py.Dataset):
            return
        dtype = str(obj.dtype)
        shape = tuple(int(dim) for dim in obj.shape)
        attrs = {str(key): _jsonable_h5_attr(value) for key, value in obj.attrs.items()}
        datasets.append(
            {
                "path": path,
                "shape": list(shape),
                "dtype": dtype,
                "kind": _h5_kind(dtype),
                "attrs": attrs,
                "sample": _h5_sample(obj),
            }
        )

    handle.visititems(visitor)
    return datasets


def _h5_kind(dtype: str) -> str:
    if dtype.startswith("float") or dtype.startswith("int") or dtype.startswith("uint"):
        return "numeric"
    if "ref" in dtype.lower() or "object" in dtype.lower():
        return "reference"
    return "other"


def _h5_sample(dataset: object) -> list[object]:
    try:
        array = np.asarray(dataset)
        if array.size == 0:
            return []
        flat = array.reshape(-1)[:5]
        return [_jsonable_h5_attr(value) for value in flat]
    except Exception:
        return []


def _jsonable_h5_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 12:
            value = value.reshape(-1)[:12]
        return [_jsonable_h5_attr(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable_h5_attr(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ingest_kit_trial(
    kit: _KITSource,
    *,
    trial: dict[str, object],
    label: str,
    episode: int,
    starting_window_id: int,
    config: KITIndustrialIngestConfig,
    max_windows: int | None,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], dict[str, object]]:
    member = _kit_hfdata_member(trial)
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    rows_read = 0
    samples: list[tuple[float, float]] = []
    with kit.open_text(member if kit._archive is not None else member.removeprefix("Data/")) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{member} has no CSV header")
        missing = [column for column in config.signal_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{member} is missing signal columns: {', '.join(missing)}")
        for row in reader:
            rows_read += 1
            try:
                samples.append(tuple(float(row[column]) for column in config.signal_columns))
            except ValueError:
                continue
    signal = np.asarray(samples, dtype=float)
    if signal.shape[0] < 4:
        return [], [], _kit_trial_summary(trial, member, rows_read, 0, label, config)
    slice_config = ICNCIngestConfig(
        window_s=config.window_s,
        stride_s=config.stride_s,
        horizon_s=config.horizon_s,
        flute_count=max(1, int(trial.get("flutes") or 1)),
        modal_frequency_hz=None,
        default_sample_rate_hz=config.sample_rate_hz,
        default_spindle_rpm=float(trial.get("spindle_rpm") or 1.0),
        default_feed_per_tooth_m=max(float(trial.get("feed_per_tooth_mm") or 0.0) / 1_000.0, 1.0e-9),
    )
    windows, records, _ = _slice_icnc_package(
        sensor_signal=signal,
        scenario=str(trial["trial"]),
        episode=episode,
        package_index=0,
        starting_window_id=starting_window_id,
        starting_window_index=0,
        sample_rate_hz=config.sample_rate_hz,
        spindle_rpm=max(float(trial.get("spindle_rpm") or 1.0), 1.0),
        label=label,
        previous=_TemporalState(),
        config=slice_config,
        max_windows=max_windows,
        feed_per_tooth_m=max(float(trial.get("feed_per_tooth_mm") or 0.0) / 1_000.0, 1.0e-9),
        axial_depth_m=float(trial.get("axial_depth_mm") or 0.0) / 1_000.0,
        radial_depth_m=float(trial.get("radial_depth_mm") or 0.0) / 1_000.0,
    )
    records = attach_horizon_targets(records, HorizonConfig(horizon_s=config.horizon_s))
    return windows, records, _kit_trial_summary(trial, member, rows_read, len(records), label, config)


def _ingest_kit_mat_trial(
    kit: _KITSource,
    *,
    trial: dict[str, object],
    label: str,
    episode: int,
    starting_window_id: int,
    config: KITMatIngestConfig,
    max_windows: int | None,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], dict[str, object]]:
    member = _kit_synchronized_mat_member(trial)
    with _open_kit_mat_h5(kit, member) as handle:
        table = _read_kit_mat_timetable(handle)
        missing = [name for name in config.signal_names if name not in table["signals"]]
        if missing:
            available = ", ".join(sorted(table["signals"]))
            raise ValueError(f"{member} is missing MAT signals {', '.join(missing)}; available: {available}")
        channels = [np.asarray(table["signals"][name], dtype=float).reshape(-1) for name in config.signal_names]
        sample_count = min(channel.size for channel in channels)
        if config.max_samples_per_trial is not None:
            sample_count = min(sample_count, config.max_samples_per_trial)
        signal = np.column_stack([channel[:sample_count] for channel in channels])
        if config.standardize_signals:
            signal = _standardize_signal_columns(signal)
        sample_rate_hz = float(table["sample_rate_hz"])
        available_signals = sorted(table["signals"])
        units = {name: table["units"].get(name, "") for name in config.signal_names}

    if signal.shape[0] < 4:
        return [], [], _kit_mat_trial_summary(trial, member, 0, 0, label, sample_rate_hz, units, available_signals, config)

    slice_config = ICNCIngestConfig(
        window_s=config.window_s,
        stride_s=config.stride_s,
        horizon_s=config.horizon_s,
        flute_count=max(1, int(trial.get("flutes") or 1)),
        modal_frequency_hz=None,
        default_sample_rate_hz=sample_rate_hz,
        default_spindle_rpm=float(trial.get("spindle_rpm") or 1.0),
        default_feed_per_tooth_m=max(float(trial.get("feed_per_tooth_mm") or 0.0) / 1_000.0, 1.0e-9),
    )
    windows, records, _ = _slice_icnc_package(
        sensor_signal=signal,
        scenario=str(trial["trial"]),
        episode=episode,
        package_index=0,
        starting_window_id=starting_window_id,
        starting_window_index=0,
        sample_rate_hz=sample_rate_hz,
        spindle_rpm=max(float(trial.get("spindle_rpm") or 1.0), 1.0),
        label=label,
        previous=_TemporalState(),
        config=slice_config,
        max_windows=max_windows,
        feed_per_tooth_m=max(float(trial.get("feed_per_tooth_mm") or 0.0) / 1_000.0, 1.0e-9),
        axial_depth_m=float(trial.get("axial_depth_mm") or 0.0) / 1_000.0,
        radial_depth_m=float(trial.get("radial_depth_mm") or 0.0) / 1_000.0,
    )
    records = attach_horizon_targets(records, HorizonConfig(horizon_s=config.horizon_s))
    return windows, records, _kit_mat_trial_summary(
        trial,
        member,
        signal.shape[0],
        len(records),
        label,
        sample_rate_hz,
        units,
        available_signals,
        config,
    )


def _read_kit_mat_timetable(handle: object) -> dict[str, object]:
    import h5py

    table = handle["#refs#/j"]
    names = [_h5_deref_string(handle, ref) for ref in np.asarray(table["varNames"]).reshape(-1)]
    units = [_h5_deref_string(handle, ref) for ref in np.asarray(table["varUnits"]).reshape(-1)]
    descriptions = [_h5_deref_string(handle, ref) for ref in np.asarray(table["varDescriptions"]).reshape(-1)]
    refs = np.asarray(table["data"]).reshape(-1)
    signals: dict[str, NDArray[np.float64]] = {}
    signal_units: dict[str, str] = {}
    signal_descriptions: dict[str, str] = {}
    for name, unit, description, ref in zip(names, units, descriptions, refs, strict=False):
        obj = handle[ref]
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in {"f", "i", "u"}:
            signals[name] = np.asarray(obj, dtype=float).reshape(-1)
            signal_units[name] = unit
            signal_descriptions[name] = description
    sample_rate = float(np.asarray(table["rowTimes"]["sampleRate"]).reshape(-1)[0])
    return {
        "signals": signals,
        "units": signal_units,
        "descriptions": signal_descriptions,
        "sample_rate_hz": sample_rate,
    }


def _standardize_signal_columns(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    center = np.median(signal, axis=0)
    mad = np.median(np.abs(signal - center), axis=0)
    q25, q75 = np.quantile(signal, [0.25, 0.75], axis=0)
    scale = np.maximum.reduce([1.4826 * mad, (q75 - q25) / 1.349, np.full(signal.shape[1], 1.0e-9)])
    return (signal - center) / scale


def _h5_deref_string(handle: object, ref: object) -> str:
    array = np.asarray(handle[ref])
    if array.dtype.kind not in {"u", "i"}:
        return ""
    return "".join(chr(int(value)) for value in array.reshape(-1) if int(value) != 0)


def _kit_mat_trial_summary(
    trial: dict[str, object],
    member: str,
    samples_read: int,
    windows: int,
    label: str,
    sample_rate_hz: float,
    units: dict[str, str],
    available_signals: list[str],
    config: KITMatIngestConfig,
) -> dict[str, object]:
    return {
        "trial": trial["trial"],
        "component": trial["component"],
        "comment": trial.get("comment", ""),
        "label": label,
        "member": member,
        "samples_read": samples_read,
        "windows": windows,
        "signal_names": list(config.signal_names),
        "standardized": config.standardize_signals,
        "signal_units": units,
        "sample_rate_hz": sample_rate_hz,
        "available_signal_count": len(available_signals),
        "available_signals_preview": available_signals[:20],
    }


def _kit_trial_summary(
    trial: dict[str, object],
    member: str,
    rows_read: int,
    windows: int,
    label: str,
    config: KITIndustrialIngestConfig,
) -> dict[str, object]:
    return {
        "trial": trial["trial"],
        "component": trial["component"],
        "comment": trial.get("comment", ""),
        "label": label,
        "member": member,
        "rows_read": rows_read,
        "windows": windows,
        "signal_columns": list(config.signal_columns),
        "sample_rate_hz": config.sample_rate_hz,
    }


@dataclass(frozen=True)
class _CsvSource:
    path: str
    archive_path: Path | None
    filesystem_path: Path | None


def discover_icnc_csv_sources(source: Path) -> list[_CsvSource]:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        return [_CsvSource(path=name, archive_path=source, filesystem_path=None) for name in names]
    if source.is_file() and source.suffix.lower() == ".csv":
        return [_CsvSource(path=source.name, archive_path=None, filesystem_path=source)]
    if source.is_dir():
        return [
            _CsvSource(path=str(path.relative_to(source)), archive_path=None, filesystem_path=path)
            for path in sorted(source.rglob("*.csv"))
        ]
    raise ValueError(f"Unsupported i-CNC source path: {source}")


@contextmanager
def _open_csv_source(source: _CsvSource) -> Iterator[object]:
    if source.archive_path is not None:
        with zipfile.ZipFile(source.archive_path) as archive:
            with archive.open(source.path, "r") as raw:
                import io

                wrapper = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                try:
                    yield wrapper
                finally:
                    wrapper.detach()
        return
    if source.filesystem_path is None:
        raise ValueError(f"CSV source has no readable path: {source.path}")
    with source.filesystem_path.open(newline="", encoding="utf-8-sig") as handle:
        yield handle


def _ingest_icnc_csv(
    csv_source: _CsvSource,
    *,
    episode_start: int,
    starting_window_id: int,
    config: ICNCIngestConfig,
    max_windows: int | None,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], dict[str, object]]:
    _raise_csv_field_limit()
    package_count = 0
    kept_packages = 0
    skipped_packages = 0
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    with _open_csv_source(csv_source) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_source.path} has no CSV header")
        channel_spec = _infer_channel_spec(reader.fieldnames)
        for row in reader:
            if config.max_packages_per_file is not None and package_count >= config.max_packages_per_file:
                break
            if max_windows is not None and len(records) >= max_windows:
                break
            package_count += 1
            try:
                x_channel, y_channel = _extract_channels(row, channel_spec)
            except ValueError:
                skipped_packages += 1
                continue
            if x_channel.size < 4 or y_channel.size < 4:
                skipped_packages += 1
                continue
            sample_count = min(x_channel.size, y_channel.size)
            x_channel = x_channel[:sample_count]
            y_channel = y_channel[:sample_count]
            sample_rate_hz = _positive_float(row.get("fs"), config.default_sample_rate_hz)
            spindle_rpm = _positive_float(row.get("spindlespeed"), config.default_spindle_rpm)
            label = _status_to_label(row.get("status", "unknown"))
            if label == "unknown" and not config.include_unknown:
                skipped_packages += 1
                continue
            episode_id = episode_start + kept_packages
            package_windows, package_records, previous = _slice_icnc_package(
                sensor_signal=np.column_stack([x_channel, y_channel]),
                scenario=Path(csv_source.path).stem,
                episode=episode_id,
                package_index=package_count - 1,
                starting_window_id=starting_window_id + len(records),
                starting_window_index=0,
                sample_rate_hz=sample_rate_hz,
                spindle_rpm=spindle_rpm,
                label=label,
                previous=_TemporalState(),
                config=config,
                max_windows=None if max_windows is None else max_windows - len(records),
            )
            package_records = attach_horizon_targets(package_records, HorizonConfig(horizon_s=config.horizon_s))
            windows.extend(package_windows)
            records.extend(package_records)
            kept_packages += 1

    summary = {
        "path": csv_source.path,
        "packages_read": package_count,
        "packages_kept": kept_packages,
        "packages_skipped": skipped_packages,
        "windows": len(records),
        "label_counts": dict(sorted(Counter(record.label for record in records).items())),
    }
    return windows, records, summary


@dataclass(frozen=True)
class _TemporalState:
    previous_start_time_s: float | None = None
    previous_dominant_frequency_hz: float | None = None
    previous_rms: float | None = None
    previous_chatter_band_energy: float | None = None
    previous_tooth_band_energy: float | None = None
    previous_non_tooth_harmonic_ratio: float | None = None
    rms_ewma: float | None = None
    chatter_band_energy_ewma: float | None = None
    non_tooth_harmonic_ratio_ewma: float | None = None


def _slice_icnc_package(
    *,
    sensor_signal: NDArray[np.float64],
    scenario: str,
    episode: int,
    package_index: int,
    starting_window_id: int,
    starting_window_index: int,
    sample_rate_hz: float,
    spindle_rpm: float,
    label: str,
    previous: _TemporalState,
    config: ICNCIngestConfig,
    max_windows: int | None,
    feed_per_tooth_m: float | None = None,
    axial_depth_m: float = 0.0,
    radial_depth_m: float = 0.0,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], _TemporalState]:
    window_samples = int(round(config.window_s * sample_rate_hz))
    stride_samples = int(round(config.stride_s * sample_rate_hz))
    if window_samples < 4:
        raise ValueError("Window must contain at least four samples")
    if stride_samples < 1:
        raise ValueError("Stride must contain at least one sample")
    if sensor_signal.shape[0] < window_samples:
        return [], [], previous

    episode_duration_s = max(sensor_signal.shape[0] / sample_rate_hz, config.window_s)
    ewma_alpha = 0.35
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    state = previous

    for start in range(0, sensor_signal.shape[0] - window_samples + 1, stride_samples):
        if max_windows is not None and len(records) >= max_windows:
            break
        stop = start + window_samples
        window = sensor_signal[start:stop]
        start_time_s = start / sample_rate_hz
        end_time_s = (stop - 1) / sample_rate_hz
        features = extract_signal_features(
            window,
            sample_rate_hz,
            spindle_rpm,
            config.flute_count,
            config.modal_frequency_hz,
        )
        risk = estimate_chatter_risk(features, margin_physics=0.0, margin_uncertainty=1.0)
        dt = start_time_s - state.previous_start_time_s if state.previous_start_time_s is not None else 0.0
        dt = max(dt, 1.0e-12)
        rms_ewma = _ewma(features.rms, state.rms_ewma, ewma_alpha)
        chatter_ewma = _ewma(features.chatter_band_energy, state.chatter_band_energy_ewma, ewma_alpha)
        ratio_ewma = _ewma(features.non_tooth_harmonic_ratio, state.non_tooth_harmonic_ratio_ewma, ewma_alpha)
        dominant_delta = _delta(features.dominant_frequency_hz, state.previous_dominant_frequency_hz)
        rms_delta = _delta(features.rms, state.previous_rms)
        chatter_delta = _delta(features.chatter_band_energy, state.previous_chatter_band_energy)
        tooth_delta = _delta(features.tooth_band_energy, state.previous_tooth_band_energy)
        ratio_delta = _delta(features.non_tooth_harmonic_ratio, state.previous_non_tooth_harmonic_ratio)
        record = WindowRecord(
            window_id=starting_window_id + len(records),
            scenario=scenario,
            episode=episode,
            window_index_in_episode=starting_window_index + len(records),
            randomized=False,
            spindle_scale=1.0,
            feed_scale=1.0,
            axial_depth_scale=1.0,
            radial_depth_scale=1.0,
            stiffness_scale=1.0,
            damping_scale=1.0,
            cutting_coeff_scale=1.0,
            noise_scale=1.0,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            episode_progress=float(np.clip(start_time_s / episode_duration_s, 0.0, 1.0)),
            label=label,
            label_id=LABEL_TO_ID[label],
            horizon_label=label,
            horizon_label_id=LABEL_TO_ID[label],
            horizon_s=config.horizon_s,
            chatter_within_horizon=label in CHATTER_POSITIVE_LABELS,
            future_chatter_within_horizon=False,
            time_to_chatter_s=0.0 if label in CHATTER_POSITIVE_LABELS else -1.0,
            risk_chatter_now=risk.risk_chatter_now,
            risk_chatter_horizon=risk.risk_chatter_horizon,
            margin_physics=0.0,
            margin_signal=risk.margin_signal,
            uncertainty=max(risk.uncertainty, 0.80),
            spindle_rpm=spindle_rpm,
            feed_per_tooth_m=feed_per_tooth_m or config.default_feed_per_tooth_m,
            axial_depth_m=axial_depth_m,
            radial_depth_m=radial_depth_m,
            axial_depth_profile_scale=1.0,
            cutting_coeff_profile_scale=1.0,
            cutting_coeff_t_n_m2=1.0,
            cutting_coeff_r_n_m2=1.0,
            tooth_frequency_hz=features.tooth_frequency_hz,
            limiting_frequency_hz=config.modal_frequency_hz or 0.0,
            dominant_frequency_hz=features.dominant_frequency_hz,
            rms=features.rms,
            chatter_band_energy=features.chatter_band_energy,
            tooth_band_energy=features.tooth_band_energy,
            non_tooth_harmonic_ratio=features.non_tooth_harmonic_ratio,
            dominant_frequency_delta_hz=dominant_delta,
            rms_delta=rms_delta,
            rms_growth_rate=rms_delta / dt if state.previous_start_time_s is not None else 0.0,
            chatter_band_energy_delta=chatter_delta,
            chatter_band_energy_growth_rate=chatter_delta / dt if state.previous_start_time_s is not None else 0.0,
            tooth_band_energy_delta=tooth_delta,
            tooth_band_energy_growth_rate=tooth_delta / dt if state.previous_start_time_s is not None else 0.0,
            non_tooth_harmonic_ratio_delta=ratio_delta,
            non_tooth_harmonic_ratio_growth_rate=ratio_delta / dt if state.previous_start_time_s is not None else 0.0,
            rms_ewma=rms_ewma,
            chatter_band_energy_ewma=chatter_ewma,
            non_tooth_harmonic_ratio_ewma=ratio_ewma,
        )
        records.append(record)
        windows.append(window)
        state = _TemporalState(
            previous_start_time_s=start_time_s,
            previous_dominant_frequency_hz=features.dominant_frequency_hz,
            previous_rms=features.rms,
            previous_chatter_band_energy=features.chatter_band_energy,
            previous_tooth_band_energy=features.tooth_band_energy,
            previous_non_tooth_harmonic_ratio=features.non_tooth_harmonic_ratio,
            rms_ewma=rms_ewma,
            chatter_band_energy_ewma=chatter_ewma,
            non_tooth_harmonic_ratio_ewma=ratio_ewma,
        )
    return windows, records, state


@dataclass(frozen=True)
class _ChannelSpec:
    x_vector_column: str | None
    y_vector_column: str | None
    x_sample_columns: tuple[str, ...]
    y_sample_columns: tuple[str, ...]


def _infer_channel_spec(fieldnames: list[str]) -> _ChannelSpec:
    by_normalised = {_normalise_column_name(name): name for name in fieldnames}
    x_vector = _first_existing(by_normalised, ("x_channel", "xchannel", "x", "accel_x", "accelx"))
    y_vector = _first_existing(by_normalised, ("y_channel", "ychannel", "y", "accel_y", "accely"))
    x_samples = _sample_columns(fieldnames, "x")
    y_samples = _sample_columns(fieldnames, "y")
    if x_vector is None and not x_samples:
        raise ValueError("Could not find an X vibration channel column")
    if y_vector is None and not y_samples:
        raise ValueError("Could not find a Y vibration channel column")
    return _ChannelSpec(
        x_vector_column=x_vector,
        y_vector_column=y_vector,
        x_sample_columns=tuple(x_samples),
        y_sample_columns=tuple(y_samples),
    )


def _extract_channels(row: dict[str, str], spec: _ChannelSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = _extract_one_channel(row, spec.x_vector_column, spec.x_sample_columns)
    y = _extract_one_channel(row, spec.y_vector_column, spec.y_sample_columns)
    if x.size != y.size:
        sample_count = min(x.size, y.size)
        if sample_count < 4:
            raise ValueError("X/Y channels do not share enough samples")
        x = x[:sample_count]
        y = y[:sample_count]
    return x, y


def _extract_one_channel(row: dict[str, str], vector_column: str | None, sample_columns: tuple[str, ...]) -> NDArray[np.float64]:
    if sample_columns:
        return np.array([float(row[column]) for column in sample_columns], dtype=float)
    if vector_column is None:
        raise ValueError("No channel column available")
    return _parse_vector_cell(row[vector_column])


def _parse_vector_cell(value: str) -> NDArray[np.float64]:
    text = value.strip()
    if not text:
        raise ValueError("Empty channel vector")
    text = text.strip("[]()")
    parsed = np.fromstring(text.replace(";", " ").replace(",", " "), sep=" ", dtype=float)
    if parsed.size == 0:
        raise ValueError("Could not parse channel vector")
    return parsed


def _sample_columns(fieldnames: list[str], axis: str) -> list[str]:
    candidates: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{axis}(?:_)?channel(?:[_\s\[(]?)(\d+)", re.IGNORECASE)
    fallback = re.compile(rf"^(?:accel[_\s]?)?{axis}(?:[_\s\[(]?)(\d+)$", re.IGNORECASE)
    for name in fieldnames:
        normalised = name.strip()
        match = pattern.match(normalised) or fallback.match(normalised)
        if match:
            candidates.append((int(match.group(1)), name))
    return [name for _, name in sorted(candidates)]


def _status_to_label(value: str | None) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"", "unknown", "nan", "none", "null", "-1", "no machining", "nomachining", "sensorerror", "sensor error"}:
        return "unknown"
    if text in {"0", "false", "no", "normal", "stable", "non-chatter", "non_chatter", "nochatter"}:
        return "stable"
    if text in {"transition", "incipient", "warning"}:
        return "transition"
    if text in {"severe", "heavy", "strong"}:
        return "severe"
    if text in {"1", "true", "yes", "chatter", "slight", "mild", "unstable"}:
        return "slight"
    try:
        return "slight" if float(text) > 0.5 else "stable"
    except ValueError:
        return "unknown"


def _icnc_manifest(
    records: list[WindowRecord],
    csv_sources: list[_CsvSource],
    source_summaries: list[dict[str, object]],
    config: ICNCIngestConfig,
    sensor_windows: NDArray[np.float32],
) -> DatasetManifest:
    counts = Counter(record.label for record in records)
    return DatasetManifest(
        schema_version="chatter-window-v5",
        sample_rate_hz=result_sample_rate(records, sensor_windows),
        window_s=config.window_s,
        stride_s=config.stride_s,
        channel_names=CHANNEL_NAMES,
        scenarios=tuple(Path(source.path).stem for source in csv_sources),
        episodes_per_scenario=1,
        total_windows=len(records),
        label_counts=dict(sorted(counts.items())),
        domain_randomization={
            "enabled": False,
            "source": "real_i_cnc_zenodo",
        },
        sampling_strategy={
            "dataset": "i_cnc_zenodo_15308467",
            "max_packages_per_file": config.max_packages_per_file,
            "max_windows": config.max_windows,
            "source_files": source_summaries,
        },
        horizon={"horizon_s": config.horizon_s},
        artifacts=("dataset.npz", "windows.csv", "manifest.json", "README.md"),
    )


def _kit_manifest(
    records: list[WindowRecord],
    source_summaries: list[dict[str, object]],
    config: KITIndustrialIngestConfig,
    sensor_windows: NDArray[np.float32],
) -> DatasetManifest:
    counts = Counter(record.label for record in records)
    return DatasetManifest(
        schema_version="chatter-window-v5",
        sample_rate_hz=result_sample_rate(records, sensor_windows),
        window_s=config.window_s,
        stride_s=config.stride_s,
        channel_names=tuple(config.signal_columns),
        scenarios=tuple(str(summary["trial"]) for summary in source_summaries),
        episodes_per_scenario=1,
        total_windows=len(records),
        label_counts=dict(sorted(counts.items())),
        domain_randomization={
            "enabled": False,
            "source": "real_kit_industrial_radar",
        },
        sampling_strategy={
            "dataset": "kit_industrial_radar_hvvwn1kfwf7qt48z",
            "max_windows": config.max_windows,
            "source_files": source_summaries,
            "modality": "processed controller hfdata CSV",
            "note": "The synchronized MATLAB acceleration/force files are not imported by this controller-only adapter.",
        },
        horizon={"horizon_s": config.horizon_s},
        artifacts=("dataset.npz", "windows.csv", "manifest.json", "README.md"),
    )


def _kit_mat_manifest(
    records: list[WindowRecord],
    source_summaries: list[dict[str, object]],
    config: KITMatIngestConfig,
    sensor_windows: NDArray[np.float32],
) -> DatasetManifest:
    counts = Counter(record.label for record in records)
    return DatasetManifest(
        schema_version="chatter-window-v5",
        sample_rate_hz=result_sample_rate(records, sensor_windows),
        window_s=config.window_s,
        stride_s=config.stride_s,
        channel_names=tuple(config.signal_names),
        scenarios=tuple(str(summary["trial"]) for summary in source_summaries),
        episodes_per_scenario=1,
        total_windows=len(records),
        label_counts=dict(sorted(counts.items())),
        domain_randomization={
            "enabled": False,
            "source": "real_kit_industrial_radar_mat",
        },
        sampling_strategy={
            "dataset": "kit_industrial_radar_hvvwn1kfwf7qt48z",
            "max_windows": config.max_windows,
            "max_samples_per_trial": config.max_samples_per_trial,
            "standardize_signals": config.standardize_signals,
            "source_files": source_summaries,
            "modality": "synchronized MATLAB timetable",
            "note": "DoE comments provide coarse trial-level labels; no time-local onset labels are available yet.",
        },
        horizon={"horizon_s": config.horizon_s},
        artifacts=("dataset.npz", "windows.csv", "manifest.json", "README.md"),
    )


def _mt_cutting_manifest(
    records: list[WindowRecord],
    source_summaries: list[dict[str, object]],
    config: MTCuttingIngestConfig,
    sensor_windows: NDArray[np.float32],
) -> DatasetManifest:
    counts = Counter(record.label for record in records)
    return DatasetManifest(
        schema_version="chatter-window-v5",
        sample_rate_hz=result_sample_rate(records, sensor_windows),
        window_s=config.window_s,
        stride_s=config.stride_s,
        channel_names=tuple(f"sensor{sensor}" for sensor in config.sensors),
        scenarios=tuple(str(summary["experiment"]) for summary in source_summaries),
        episodes_per_scenario=0,
        total_windows=len(records),
        label_counts=dict(sorted(counts.items())),
        domain_randomization={
            "enabled": False,
            "source": "real_purdue_mt_cutting_sound",
        },
        sampling_strategy={
            "dataset": "purdue_lamm_mt_cutting_dataset",
            "repository_url": MT_CUTTING_GITHUB_URL,
            "max_experiments": config.max_experiments,
            "max_windows": config.max_windows,
            "source_files": source_summaries,
            "modality": "48 kHz sound windows from IMI cutting intervals",
            "label_source": "labeling_all_details.xlsx operator chatter labels; max(operator1, operator2)",
            "note": "Labels are per cutting path, not time-local onset labels.",
        },
        horizon={"horizon_s": config.horizon_s},
        artifacts=("dataset.npz", "windows.csv", "manifest.json", "README.md"),
    )


def _write_real_replay_dataset(
    out_dir: Path,
    records: list[WindowRecord],
    sensor_windows: NDArray[np.float32],
    manifest: DatasetManifest,
) -> None:
    np.savez_compressed(
        out_dir / "dataset.npz",
        sensor_windows=sensor_windows,
        labels=np.array([record.label for record in records]),
        label_ids=np.array([record.label_id for record in records], dtype=np.int64),
        horizon_labels=np.array([record.horizon_label for record in records]),
        horizon_label_ids=np.array([record.horizon_label_id for record in records], dtype=np.int64),
        horizon_s=np.array([manifest.horizon["horizon_s"]], dtype=np.float32),
        chatter_within_horizon=np.array([record.chatter_within_horizon for record in records], dtype=np.bool_),
        future_chatter_within_horizon=np.array([record.future_chatter_within_horizon for record in records], dtype=np.bool_),
        time_to_chatter_s=np.array([record.time_to_chatter_s for record in records], dtype=np.float32),
        risk_chatter_now=np.array([record.risk_chatter_now for record in records], dtype=np.float32),
        risk_chatter_horizon=np.array([record.risk_chatter_horizon for record in records], dtype=np.float32),
        margin_physics=np.array([record.margin_physics for record in records], dtype=np.float32),
        uncertainty=np.array([record.uncertainty for record in records], dtype=np.float32),
        scenarios=np.array([record.scenario for record in records]),
        episodes=np.array([record.episode for record in records], dtype=np.int64),
        window_index_in_episode=np.array([record.window_index_in_episode for record in records], dtype=np.int64),
        randomized=np.array([record.randomized for record in records], dtype=np.bool_),
        spindle_scale=np.array([record.spindle_scale for record in records], dtype=np.float32),
        feed_scale=np.array([record.feed_scale for record in records], dtype=np.float32),
        axial_depth_scale=np.array([record.axial_depth_scale for record in records], dtype=np.float32),
        radial_depth_scale=np.array([record.radial_depth_scale for record in records], dtype=np.float32),
        stiffness_scale=np.array([record.stiffness_scale for record in records], dtype=np.float32),
        damping_scale=np.array([record.damping_scale for record in records], dtype=np.float32),
        cutting_coeff_scale=np.array([record.cutting_coeff_scale for record in records], dtype=np.float32),
        noise_scale=np.array([record.noise_scale for record in records], dtype=np.float32),
        start_time_s=np.array([record.start_time_s for record in records], dtype=np.float64),
        episode_progress=np.array([record.episode_progress for record in records], dtype=np.float32),
        axial_depth_profile_scale=np.array([record.axial_depth_profile_scale for record in records], dtype=np.float32),
        cutting_coeff_profile_scale=np.array([record.cutting_coeff_profile_scale for record in records], dtype=np.float32),
        cutting_coeff_t_n_m2=np.array([record.cutting_coeff_t_n_m2 for record in records], dtype=np.float32),
        cutting_coeff_r_n_m2=np.array([record.cutting_coeff_r_n_m2 for record in records], dtype=np.float32),
        dominant_frequency_delta_hz=np.array([record.dominant_frequency_delta_hz for record in records], dtype=np.float32),
        rms_delta=np.array([record.rms_delta for record in records], dtype=np.float32),
        rms_growth_rate=np.array([record.rms_growth_rate for record in records], dtype=np.float32),
        chatter_band_energy_delta=np.array([record.chatter_band_energy_delta for record in records], dtype=np.float32),
        chatter_band_energy_growth_rate=np.array([record.chatter_band_energy_growth_rate for record in records], dtype=np.float32),
        tooth_band_energy_delta=np.array([record.tooth_band_energy_delta for record in records], dtype=np.float32),
        tooth_band_energy_growth_rate=np.array([record.tooth_band_energy_growth_rate for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_delta=np.array([record.non_tooth_harmonic_ratio_delta for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_growth_rate=np.array(
            [record.non_tooth_harmonic_ratio_growth_rate for record in records],
            dtype=np.float32,
        ),
        rms_ewma=np.array([record.rms_ewma for record in records], dtype=np.float32),
        chatter_band_energy_ewma=np.array([record.chatter_band_energy_ewma for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_ewma=np.array([record.non_tooth_harmonic_ratio_ewma for record in records], dtype=np.float32),
        sample_rate_hz=np.array([manifest.sample_rate_hz]),
        channel_names=np.array(manifest.channel_names),
    )
    _write_records_csv(out_dir / "windows.csv", records)
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_csv(path: Path, records: list[WindowRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_icnc_readme(path: Path, manifest: DatasetManifest, source_summaries: list[dict[str, object]]) -> None:
    lines = [
        "# i-CNC Replay Dataset",
        "",
        "Imported from Zenodo record `15308467` into the local chatter replay schema.",
        "",
        f"Total windows: `{manifest.total_windows}`",
        f"Sample rate: `{manifest.sample_rate_hz:.1f} Hz`",
        f"Window/stride: `{manifest.window_s:.3f}s / {manifest.stride_s:.3f}s`",
        f"Horizon target: `{manifest.horizon['horizon_s']:.3f}s`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in manifest.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Sources", "", "| Source | Packages | Kept | Skipped | Windows |", "|---|---:|---:|---:|---:|"])
    for source in source_summaries:
        lines.append(
            f"| `{source['path']}` | {source['packages_read']} | {source['packages_kept']} | "
            f"{source['packages_skipped']} | {source['windows']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `status` is treated as the label supplied by the dataset authors' AI chatter indicator.",
            "- The dataset does not include axial/radial depth, feed, tool, FRF, or cutting coefficients, so physics-margin fields are placeholders.",
            "- Use this for signal-estimator validation and domain-shift checks, not for claiming a calibrated process-dynamics twin.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_kit_mat_readme(
    path: Path,
    manifest: DatasetManifest,
    source_summaries: list[dict[str, object]],
    config: KITMatIngestConfig,
) -> None:
    lines = [
        "# KIT Industrial Synchronized MAT Replay Dataset",
        "",
        "Imported from RADAR/KIT synchronized MATLAB timetables into the local chatter replay schema.",
        "",
        f"Total windows: `{manifest.total_windows}`",
        f"Sample rate: `{manifest.sample_rate_hz:.1f} Hz`",
        f"Window/stride: `{manifest.window_s:.3f}s / {manifest.stride_s:.3f}s`",
        f"Horizon target: `{manifest.horizon['horizon_s']:.3f}s`",
        "Signals: " + ", ".join(f"`{name}`" for name in config.signal_names),
        f"Robust standardized signals: `{config.standardize_signals}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in manifest.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Trials", "", "| Trial | Component | Label | Comment | Samples | Windows |", "|---|---|---|---|---:|---:|"])
    for source in source_summaries:
        lines.append(
            f"| `{source['trial']}` | {source['component']} | {source['label']} | "
            f"{source['comment']} | {source['samples_read']} | {source['windows']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Labels are still coarse DoE trial labels, not time-local onset annotations.",
            "- Use this for high-rate signal validation and pseudo-labeling experiments before claiming early warning.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_mt_cutting_readme(
    path: Path,
    manifest: DatasetManifest,
    source_summaries: list[dict[str, object]],
    config: MTCuttingIngestConfig,
) -> None:
    lines = [
        "# Purdue MT Cutting Sound Replay Dataset",
        "",
        f"Imported from `{MT_CUTTING_GITHUB_URL}` into the local chatter replay schema.",
        "",
        f"Total windows: `{manifest.total_windows}`",
        f"Sample rate: `{manifest.sample_rate_hz:.1f} Hz`",
        f"Window/stride: `{manifest.window_s:.3f}s / {manifest.stride_s:.3f}s`",
        f"Horizon target: `{manifest.horizon['horizon_s']:.3f}s`",
        f"Sensors: `{','.join(str(sensor) for sensor in config.sensors)}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in manifest.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Source Experiments", "", "| Experiment | Paths imported | Windows | Label counts |", "|---|---:|---:|---|"])
    for source in source_summaries:
        lines.append(
            f"| `{source['experiment']}` | {source['paths_imported']} | {source['windows']} | "
            f"`{json.dumps(source['label_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Labels are assigned per cutting path from the operator workbook, not per time-local onset.",
            "- This dataset is useful for real audio/current-window chatter validation, not closed-loop control validation.",
            "- Cutting context is imported when available, but no FRF, force, surface-finish, or controller-intervention trace is included.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_kit_readme(
    path: Path,
    manifest: DatasetManifest,
    source_summaries: list[dict[str, object]],
    config: KITIndustrialIngestConfig,
) -> None:
    lines = [
        "# KIT Industrial Controller Replay Dataset",
        "",
        "Imported from RADAR/KIT record `hvvwn1kfwf7qt48z` into the local chatter replay schema.",
        "",
        f"Total windows: `{manifest.total_windows}`",
        f"Sample rate: `{manifest.sample_rate_hz:.1f} Hz`",
        f"Window/stride: `{manifest.window_s:.3f}s / {manifest.stride_s:.3f}s`",
        f"Horizon target: `{manifest.horizon['horizon_s']:.3f}s`",
        f"Signal columns: `{config.signal_columns[0]}`, `{config.signal_columns[1]}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in manifest.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Trials", "", "| Trial | Component | Label | Comment | Rows | Windows |", "|---|---|---|---|---:|---:|"])
    for source in source_summaries:
        lines.append(
            f"| `{source['trial']}` | {source['component']} | {source['label']} | "
            f"{source['comment']} | {source['rows_read']} | {source['windows']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This first KIT adapter uses processed SINUMERIK controller `hfdata.csv` signals only.",
            "- DoE comments provide coarse trial-level labels; they are not onset timestamps.",
            "- Use this for industrial controller-signal domain checks, then upgrade to synchronized MATLAB acceleration/force ingestion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _download_payload(out_path: Path, md5: str, *, downloaded: bool) -> dict[str, object]:
    return {
        "path": str(out_path),
        "downloaded": downloaded,
        "size_bytes": out_path.stat().st_size,
        "md5": md5,
        "expected_size_bytes": ICNC_SIZE_BYTES,
        "expected_md5": ICNC_MD5,
        "ok": out_path.stat().st_size == ICNC_SIZE_BYTES and md5 == ICNC_MD5,
    }


def _md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _first_existing(columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    normalised_candidates = {_normalise_column_name(candidate) for candidate in candidates}
    for candidate in normalised_candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _positive_float(value: str | None, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _delta(current: float, previous: float | None) -> float:
    if previous is None:
        return 0.0
    return float(current - previous)


def _ewma(current: float, previous: float | None, alpha: float) -> float:
    if previous is None:
        return float(current)
    return float(alpha * current + (1.0 - alpha) * previous)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
