from __future__ import annotations

import csv
import hashlib
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

import numpy as np
from numpy.typing import NDArray

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

    for episode, csv_source in enumerate(csv_sources):
        if config.max_windows is not None and len(all_records) >= config.max_windows:
            break
        remaining_windows = None if config.max_windows is None else config.max_windows - len(all_records)
        windows, records, summary = _ingest_icnc_csv(
            csv_source,
            episode=episode,
            starting_window_id=len(all_records),
            config=config,
            max_windows=remaining_windows,
        )
        all_windows.extend(windows)
        all_records.extend(records)
        source_summaries.append(summary)

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
    episode: int,
    starting_window_id: int,
    config: ICNCIngestConfig,
    max_windows: int | None,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], dict[str, object]]:
    _raise_csv_field_limit()
    package_count = 0
    skipped_packages = 0
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    previous = _TemporalState()
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
            package_windows, package_records, previous = _slice_icnc_package(
                sensor_signal=np.column_stack([x_channel, y_channel]),
                scenario=Path(csv_source.path).stem,
                episode=episode,
                package_index=package_count - 1,
                starting_window_id=starting_window_id + len(records),
                starting_window_index=len(records),
                sample_rate_hz=sample_rate_hz,
                spindle_rpm=spindle_rpm,
                label=label,
                previous=previous,
                config=config,
                max_windows=None if max_windows is None else max_windows - len(records),
            )
            windows.extend(package_windows)
            records.extend(package_records)

    records = attach_horizon_targets(records, HorizonConfig(horizon_s=config.horizon_s))
    summary = {
        "path": csv_source.path,
        "packages_read": package_count,
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
) -> tuple[list[NDArray[np.float64]], list[WindowRecord], _TemporalState]:
    window_samples = int(round(config.window_s * sample_rate_hz))
    stride_samples = int(round(config.stride_s * sample_rate_hz))
    if window_samples < 4:
        raise ValueError("Window must contain at least four samples")
    if stride_samples < 1:
        raise ValueError("Stride must contain at least one sample")
    if sensor_signal.shape[0] < window_samples:
        return [], [], previous

    package_duration_s = sensor_signal.shape[0] / sample_rate_hz
    package_offset_s = package_index * package_duration_s
    episode_duration_s = max((package_index + 1) * package_duration_s, config.window_s)
    ewma_alpha = 0.35
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    state = previous

    for start in range(0, sensor_signal.shape[0] - window_samples + 1, stride_samples):
        if max_windows is not None and len(records) >= max_windows:
            break
        stop = start + window_samples
        window = sensor_signal[start:stop]
        start_time_s = package_offset_s + start / sample_rate_hz
        end_time_s = package_offset_s + (stop - 1) / sample_rate_hz
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
            feed_per_tooth_m=config.default_feed_per_tooth_m,
            axial_depth_m=0.0,
            radial_depth_m=0.0,
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
    lines.extend(["", "## Sources", "", "| Source | Packages | Skipped | Windows |", "|---|---:|---:|---:|"])
    for source in source_summaries:
        lines.append(
            f"| `{source['path']}` | {source['packages_read']} | {source['packages_skipped']} | {source['windows']} |"
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
