import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from chatter_twin.cli import main
from chatter_twin.datasets import (
    ICNCIngestConfig,
    discover_icnc_csv_sources,
    ingest_icnc_dataset,
    write_icnc_source_manifest,
)


def test_write_icnc_source_manifest(tmp_path: Path):
    out = tmp_path / "source_manifest.json"
    manifest = write_icnc_source_manifest(out)
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["record_url"] == manifest["record_url"]
    assert saved["filename"] == "i-CNC Dataset.zip"
    assert saved["md5"]


def test_ingest_icnc_dataset_from_vector_csv(tmp_path: Path):
    source = tmp_path / "test1_with_status.csv"
    _write_vector_icnc_csv(source)

    payload = ingest_icnc_dataset(
        source=source,
        out_dir=tmp_path / "out",
        config=ICNCIngestConfig(
            window_s=0.008,
            stride_s=0.004,
            horizon_s=0.02,
            default_sample_rate_hz=1_000.0,
            max_packages_per_file=2,
        ),
    )

    manifest = payload["manifest"]
    assert manifest["total_windows"] == 6
    assert manifest["label_counts"] == {"slight": 3, "stable": 3}
    assert (tmp_path / "out" / "dataset.npz").exists()
    assert (tmp_path / "out" / "windows.csv").exists()
    data = np.load(tmp_path / "out" / "dataset.npz")
    assert data["sensor_windows"].shape == (6, 8, 2)
    assert set(data["channel_names"].tolist()) == {"accel_x", "accel_y"}
    assert data["sample_rate_hz"][0] == 1_000.0

    with (tmp_path / "out" / "windows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scenario"] == "test1_with_status"
    assert rows[0]["label"] == "stable"
    assert rows[-1]["label"] == "slight"
    assert "rms_growth_rate" in rows[0]
    assert "future_chatter_within_horizon" in rows[0]


def test_ingest_icnc_dataset_from_zip_with_expanded_columns(tmp_path: Path):
    csv_path = tmp_path / "test2_with_status.csv"
    _write_expanded_icnc_csv(csv_path)
    zip_path = tmp_path / "icnc.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="nested/test2_with_status.csv")

    sources = discover_icnc_csv_sources(zip_path)
    assert [source.path for source in sources] == ["nested/test2_with_status.csv"]

    payload = ingest_icnc_dataset(
        source=zip_path,
        out_dir=tmp_path / "out_zip",
        config=ICNCIngestConfig(window_s=0.004, stride_s=0.004, default_sample_rate_hz=1_000.0),
    )
    assert payload["manifest"]["total_windows"] == 1
    data = np.load(tmp_path / "out_zip" / "dataset.npz")
    assert data["sensor_windows"].shape == (1, 4, 2)
    assert data["labels"].tolist() == ["slight"]


def test_cli_icnc_manifest_and_ingest(tmp_path: Path):
    source = tmp_path / "test1_with_status.csv"
    _write_vector_icnc_csv(source)
    manifest_out = tmp_path / "manifest.json"
    ingest_out = tmp_path / "ingested"

    assert main(["icnc-manifest", "--out", str(manifest_out)]) == 0
    assert manifest_out.exists()
    status = main(
        [
            "ingest-icnc",
            "--source",
            str(source),
            "--out",
            str(ingest_out),
            "--window",
            "0.008",
            "--stride",
            "0.004",
            "--default-sample-rate",
            "1000",
            "--max-packages-per-file",
            "1",
        ]
    )
    assert status == 0
    saved_manifest = json.loads((ingest_out / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["total_windows"] == 3


def _write_vector_icnc_csv(path: Path) -> None:
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "fs": "1000",
            "numscans": "16",
            "spindlespeed": "6000",
            "x_channel": " ".join(str(value) for value in range(16)),
            "y_channel": " ".join(str(value * 0.5) for value in range(16)),
            "status": "0",
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "fs": "1000",
            "numscans": "16",
            "spindlespeed": "6200",
            "x_channel": " ".join(str(16 - value) for value in range(16)),
            "y_channel": " ".join(str(value % 3) for value in range(16)),
            "status": "1",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_expanded_icnc_csv(path: Path) -> None:
    fieldnames = ["timestamp", "fs", "numscans", "spindlespeed", "status"]
    fieldnames.extend(f"x_channel_{idx}" for idx in range(4))
    fieldnames.extend(f"y_channel_{idx}" for idx in range(4))
    row = {
        "timestamp": "2026-01-01T00:00:00",
        "fs": "1000",
        "numscans": "4",
        "spindlespeed": "7000",
        "status": "chatter",
    }
    row.update({f"x_channel_{idx}": str(idx) for idx in range(4)})
    row.update({f"y_channel_{idx}": str(idx + 0.25) for idx in range(4)})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
