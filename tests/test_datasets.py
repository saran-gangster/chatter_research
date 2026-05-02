import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from chatter_twin.cli import main
from chatter_twin.datasets import (
    ICNCIngestConfig,
    KITIndustrialIngestConfig,
    KITMatIngestConfig,
    discover_icnc_csv_sources,
    ingest_kit_industrial_dataset,
    ingest_kit_mat_dataset,
    ingest_icnc_dataset,
    inspect_kit_industrial_dataset,
    inspect_kit_synchronized_mat,
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


def test_ingest_icnc_skips_unknown_rows_by_default(tmp_path: Path):
    source = tmp_path / "test_unknown_status.csv"
    _write_unknown_icnc_csv(source)

    cut_only = ingest_icnc_dataset(
        source=source,
        out_dir=tmp_path / "cut_only",
        config=ICNCIngestConfig(window_s=0.008, stride_s=0.004, default_sample_rate_hz=1_000.0),
    )
    assert cut_only["manifest"]["label_counts"] == {"stable": 3}
    assert cut_only["sources"][0]["packages_read"] == 2
    assert cut_only["sources"][0]["packages_skipped"] == 1

    with_unknown = ingest_icnc_dataset(
        source=source,
        out_dir=tmp_path / "with_unknown",
        config=ICNCIngestConfig(
            window_s=0.008,
            stride_s=0.004,
            default_sample_rate_hz=1_000.0,
            include_unknown=True,
        ),
    )
    assert with_unknown["manifest"]["label_counts"] == {"stable": 3, "unknown": 3}


def test_inspect_and_ingest_kit_industrial_directory(tmp_path: Path):
    source = tmp_path / "Data"
    _write_minimal_kit_source(source)

    inspection = inspect_kit_industrial_dataset(source)
    assert inspection["trials"] == 2
    assert inspection["processed_hfdata_files"] == 2
    assert inspection["chatter_trials"][0]["trial"] == "IM-02F-A01"

    payload = ingest_kit_industrial_dataset(
        source=source,
        out_dir=tmp_path / "kit_out",
        trials=["IM-01F", "IM-02F-A01"],
        config=KITIndustrialIngestConfig(window_s=0.4, stride_s=0.4, horizon_s=0.8, sample_rate_hz=10.0),
    )
    assert payload["manifest"]["total_windows"] == 4
    assert payload["manifest"]["label_counts"] == {"slight": 2, "stable": 2}
    data = np.load(tmp_path / "kit_out" / "dataset.npz")
    assert data["sensor_windows"].shape == (4, 4, 2)
    assert data["channel_names"].tolist() == ["LOAD|6", "CURRENT|6"]


def test_cli_inspect_and_ingest_kit_industrial(tmp_path: Path):
    source = tmp_path / "Data"
    _write_minimal_kit_source(source)
    inspection_out = tmp_path / "kit_inspection.json"
    ingest_out = tmp_path / "kit_cli_out"

    assert main(["inspect-kit-industrial", "--source", str(source), "--out", str(inspection_out)]) == 0
    assert json.loads(inspection_out.read_text(encoding="utf-8"))["chatter_trials"][0]["trial"] == "IM-02F-A01"

    status = main(
        [
            "ingest-kit-industrial",
            "--source",
            str(source),
            "--out",
            str(ingest_out),
            "--window",
            "0.4",
            "--stride",
            "0.4",
            "--horizon",
            "0.8",
            "--sample-rate",
            "10",
        ]
    )
    assert status == 0
    saved_manifest = json.loads((ingest_out / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["label_counts"] == {"slight": 2, "stable": 2}


def test_inspect_kit_synchronized_mat_directory(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "Data"
    _write_minimal_kit_source(source)
    mat_path = source / "Dataset" / "Injection mold" / "IM-02F-A01" / "processed_data" / "IM-02F-A01_synchronized.mat"
    _write_minimal_kit_mat(h5py, mat_path, ["xAcceleration", "yAcceleration"])

    payload = inspect_kit_synchronized_mat(source=source, trial="IM-02F-A01", max_datasets=10)

    paths = {item["path"] for item in payload["datasets"]}
    assert "#refs#/j/data" in paths
    assert "#refs#/j/varNames" in paths


def test_ingest_kit_mat_directory(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "Data"
    _write_minimal_kit_source(source)
    for trial in ("IM-01F", "IM-02F-A01"):
        mat_path = source / "Dataset" / "Injection mold" / trial / "processed_data" / f"{trial}_synchronized.mat"
        _write_minimal_kit_mat(h5py, mat_path, ["xAcceleration", "yAcceleration"])

    payload = ingest_kit_mat_dataset(
        source=source,
        out_dir=tmp_path / "kit_mat_out",
        trials=["IM-01F", "IM-02F-A01"],
        config=KITMatIngestConfig(window_s=0.4, stride_s=0.4, horizon_s=0.8),
    )

    assert payload["manifest"]["total_windows"] == 4
    assert payload["manifest"]["sample_rate_hz"] == 10.0
    assert payload["manifest"]["label_counts"] == {"slight": 2, "stable": 2}
    data = np.load(tmp_path / "kit_mat_out" / "dataset.npz")
    assert data["sensor_windows"].shape == (4, 4, 2)
    assert data["channel_names"].tolist() == ["xAcceleration", "yAcceleration"]


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


def _write_minimal_kit_source(root: Path) -> None:
    doe = root / "Descriptive" / "DoE" / "DoE.xlsx"
    doe.parent.mkdir(parents=True, exist_ok=True)
    _write_minimal_xlsx(
        doe,
        [
            ["Injection mold"],
            [
                "Trial",
                "Optimal Overlap [mm]",
                "Optimal Cutting depth [mm]",
                "Cooling lubricant",
                "Tools",
                "Number of teeth",
                "fz [mm]",
                "Dc [mm]",
                "Material",
                "Vc [mm/min]",
                "Calculated spindle speed [1/min]",
                "Factor",
                "Exact spindle speed [U/min]",
                "Spindle Speed [U/min]",
                "Feedrate [mm/min]",
                "NC-Code",
                "Comment",
            ],
            ["IM-01F", "0.5", "0.5", "no", "VHM", "4", "0.02246", "10", "S235JR", "140", "4456", "100", "4456", "4450", "399", "IM-01F", "Finishing"],
            [
                "IM-02F-A01",
                "0.5",
                "0.5",
                "no",
                "VHM",
                "4",
                "0.04492",
                "10",
                "S235JR",
                "140",
                "4456",
                "100",
                "4456",
                "4450",
                "799",
                "IM-02F-A01",
                "Finishing with Chatter",
            ],
        ],
    )
    for trial in ("IM-01F", "IM-02F-A01"):
        trial_dir = root / "Dataset" / "Injection mold" / trial / "processed_data"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with (trial_dir / f"{trial}_hfdata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["CYCLE", "LOAD|6", "CURRENT|6"])
            writer.writeheader()
            for idx in range(8):
                scale = 2.0 if trial.endswith("A01") else 1.0
                writer.writerow({"CYCLE": idx, "LOAD|6": scale * (idx + 1), "CURRENT|6": scale * (idx % 3 + 1)})


def _write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    strings: list[str] = []
    index: dict[str, int] = {}
    for row in rows:
        for value in row:
            if value not in index:
                index[value] = len(strings)
                strings.append(value)

    def cell_ref(row_idx: int, col_idx: int) -> str:
        letters = ""
        col = col_idx
        while col:
            col, rem = divmod(col - 1, 26)
            letters = chr(65 + rem) + letters
        return f"{letters}{row_idx}"

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cells.append(f'<c r="{cell_ref(row_idx, col_idx)}" t="s"><v>{index[value]}</v></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="DoE" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + "".join(f"<si><t>{value}</t></si>" for value in strings)
            + "</sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(sheet_rows)
            + "</sheetData></worksheet>",
        )


def _write_minimal_kit_mat(h5py: object, path: Path, signal_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("messdaten_sync", data=np.array([[0, 2, 1, 1, 1, 3]], dtype=np.uint32))
        refs = handle.create_group("#refs#")
        table = refs.create_group("j")
        data_refs = np.empty((len(signal_names), 1), dtype=ref_dtype)
        name_refs = np.empty((len(signal_names), 1), dtype=ref_dtype)
        unit_refs = np.empty((len(signal_names), 1), dtype=ref_dtype)
        desc_refs = np.empty((len(signal_names), 1), dtype=ref_dtype)
        for idx, name in enumerate(signal_names):
            values = np.arange(8, dtype=np.float64).reshape(1, -1) + idx
            data_refs[idx, 0] = refs.create_dataset(f"data_{idx}", data=values).ref
            name_refs[idx, 0] = _h5_char_dataset(refs, f"name_{idx}", name).ref
            unit_refs[idx, 0] = _h5_char_dataset(refs, f"unit_{idx}", "g").ref
            desc_refs[idx, 0] = _h5_char_dataset(refs, f"desc_{idx}", name).ref
        table.create_dataset("data", data=data_refs)
        table.create_dataset("varNames", data=name_refs)
        table.create_dataset("varUnits", data=unit_refs)
        table.create_dataset("varDescriptions", data=desc_refs)
        row_times = table.create_group("rowTimes")
        row_times.create_dataset("sampleRate", data=np.array([[10.0]], dtype=np.float64))


def _h5_char_dataset(group: object, name: str, value: str) -> object:
    data = np.array([ord(char) for char in value], dtype=np.uint16).reshape(-1, 1)
    return group.create_dataset(name, data=data)


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


def _write_unknown_icnc_csv(path: Path) -> None:
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "fs": "1000",
            "numscans": "16",
            "spindlespeed": "6000",
            "x_channel": " ".join(str(value) for value in range(16)),
            "y_channel": " ".join(str(value * 0.5) for value in range(16)),
            "status": "No Machining",
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "fs": "1000",
            "numscans": "16",
            "spindlespeed": "6000",
            "x_channel": " ".join(str(value) for value in range(16)),
            "y_channel": " ".join(str(value * 0.5) for value in range(16)),
            "status": "Stable",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
