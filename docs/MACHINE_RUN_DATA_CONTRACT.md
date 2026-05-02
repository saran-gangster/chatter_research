# Real CNC Machine Run Data Contract

This is the first hardware-facing contract for the chatter twin. It does not
write to a CNC controller. It only validates and imports synchronized machine
context plus high-rate sensor logs into the existing replay schema.

## Folder Shape

Each captured run should live in one folder:

```text
data/raw/machine_run_001/
  run_metadata.json
  sensors.csv
  cnc_context.csv
  labels.csv
```

Create a starter folder:

```bash
rtk uv run chatter-twin machine-run-template --out data/raw/machine_run_001
```

## Required Files

`run_metadata.json` records the setup:

| Section | Required Purpose |
|---|---|
| `run_id` | Stable run name used as the replay scenario |
| `machine` | Machine/controller identifiers and MTConnect/API notes |
| `tool` | diameter, flute count, overhang, tool id |
| `process` | material, operation, axial/radial depth, cutting coefficients |
| `modal` | low-order x/y modal mass, stiffness, and damping |
| `sensors` | sample rate, channel names, placement |
| `time_sync` | how DAQ and controller clocks were aligned |

`sensors.csv` is the high-rate DAQ stream:

```csv
time_s,accel_x,accel_y,accel_z
0.0000,0.000,0.000,0.000
0.0001,0.012,-0.004,0.002
```

`cnc_context.csv` is the lower-rate controller stream:

```csv
time_s,spindle_rpm,feed_per_tooth_m,feed_override,spindle_override,program_line
0.0,9000,0.000045,1.0,1.0,N100
0.1,9000,0.000045,1.0,1.0,N100
```

`feed_mm_min` can replace `feed_per_tooth_m`; the importer converts it using
spindle speed and flute count.

`labels.csv` is optional but strongly recommended:

```csv
start_time_s,end_time_s,label,source
0.0,1.5,stable,operator/surface/spectrum
1.5,1.9,transition,spectrum
1.9,2.4,slight,surface+audio
```

Allowed labels are `stable`, `transition`, `slight`, `severe`, and `unknown`.

## Validate

```bash
rtk uv run chatter-twin validate-machine-run \
  --source data/raw/machine_run_001 \
  --out data/raw/machine_run_001/validation.json
```

Validation checks that:

- required files and columns exist;
- sensor time is strictly increasing;
- CNC context time is strictly increasing;
- labels have valid intervals;
- high-rate sensor jitter is reported before ingestion.

## Ingest

```bash
rtk uv run chatter-twin ingest-machine-run \
  --source data/raw/machine_run_001 \
  --out results/machine_run_001_replay \
  --window 0.1 \
  --stride 0.05 \
  --horizon 0.25 \
  --default-label unknown
```

The output is compatible with `train-risk`, `shadow-eval`, and the rest of the
offline replay tooling:

```text
results/machine_run_001_replay/
  dataset.npz
  windows.csv
  manifest.json
  README.md
```

## Claim Boundary

This importer enables real-machine shadow validation. It does not by itself
prove chatter suppression. Strong claims still require calibrated FRF/cutting
coefficients, trustworthy labels, surface/tool validation, and eventually a
separate safety-approved CNC write path.
