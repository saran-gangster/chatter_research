# Real Dataset On-Ramp

## Dataset Ladder

| Dataset | Use Now | Gap It Covers | Main Limitation |
|---|---|---|---|
| i-CNC Zenodo | current-window chatter signal validation | real 10 kHz vibration plus chatter status | package-level labels, no process depth/feed/FRF/context |
| KIT industrial CNC milling | process-context/anomaly validation | continuous controller data, force, acceleration, NC/CAD, anomaly documentation | 44.6 GB tar, still not closed-loop intervention data |
| Purdue MT cutting sound | external audio chatter validation | many labeled cutting paths with 48 kHz sound and operator chatter labels | path-level labels, no force/FRF/surface validation or interventions |

## i-CNC Zenodo

The first external validation target is the i-CNC Zenodo dataset:

- Record: <https://zenodo.org/records/15308467>
- DOI: `10.5281/zenodo.15308467`
- License: `CC-BY-4.0`
- Download: `i-CNC Dataset.zip`
- Size: about `3.0 GB`
- Files inside the zip: `test1_with_status.csv`, `test2_with_status.csv`
- Signals: X/Y vibration packages sampled at `10 kHz`, spindle speed, and a
  `status` indicator for chatter occurrence.

This dataset is useful for signal-estimator/domain-shift validation. It is not
enough to validate the full process-dynamics twin because it does not include
tool geometry, feed/depth, material coefficients, FRFs, surface finish, or
controller interventions.

## CPU SSH Workflow

Clone the repo on the data machine:

```bash
git clone https://github.com/saran-gangster/chatter_research.git
cd chatter_research
```

Write the source manifest without downloading the large zip:

```bash
uv run chatter-twin icnc-manifest --out data/raw/icnc/source_manifest.json
```

Download and verify the Zenodo zip:

```bash
uv run chatter-twin download-icnc \
  --out "data/raw/icnc/i-CNC Dataset.zip" \
  --manifest-out data/raw/icnc/source_manifest.json
```

Import a bounded first pass into the replay schema:

```bash
uv run chatter-twin ingest-icnc \
  --source "data/raw/icnc/i-CNC Dataset.zip" \
  --out results/icnc_replay_subset \
  --window 0.1 \
  --stride 0.05 \
  --horizon 0.25 \
  --max-packages-per-file 1000
```

By default the importer skips `No Machining`, `SensorError`, and other unknown
status rows so the first replay dataset is cutting-only. Add
`--include-unknown` only when you intentionally want a raw operational-state
dataset with `unknown` labels preserved. Each retained measurement package is
stored as one replay episode so episode splits hold out whole packages.

Then train a first real-data signal baseline:

```bash
uv run chatter-twin train-risk \
  --dataset results/icnc_replay_subset \
  --out results/icnc_risk_signal_baseline \
  --model hist_gb \
  --feature-set temporal \
  --target current \
  --split-mode episode \
  --validation-fraction 0.25
```

## Importer Caveats

The importer treats the dataset `status` column as the label. The physics fields
in `windows.csv` are intentionally conservative placeholders because the public
dataset does not expose a calibrated machining context. Use `temporal` or
`base` feature sets first; `profile` and `interaction` feature sets are more
appropriate for synthetic/calibrated replay datasets with real feed, depth, and
cutting-coefficient metadata.

The first bounded run on the Lightning CPU box used 1000 raw packages from each
CSV, kept 578 cutting packages, and produced 4046 replay windows. A package-held
out histogram gradient boosting baseline reached `0.544` chatter F1 and `0.648`
chatter recall on current-window labels. Event-warning and lead-time metrics
were not meaningful on that subset because the dataset labels whole packages,
not stable-to-chatter onset trajectories.

## KIT Industrial CNC Milling

The next bridge dataset is KIT/RADAR record `hvvwn1kfwf7qt48z`:

- Record: <https://radar.kit.edu/radar/en/dataset/hvvwn1kfwf7qt48z>
- DOI: `10.35097/hvvwn1kfwf7qt48z`
- License: `CC-BY-4.0`
- Download format: `application/x-tar`
- Size: `44,584,092,672` bytes, about `44.6 GB`
- Machine: DMC 60 H with Siemens SINUMERIK 840D
- Coverage: `33` experiments and about `6` hours of milling data
- Signals: controller data at `500 Hz`, force and acceleration at `10 kHz`,
  plus synchronized MATLAB files, NC programs, CAD/STP models, and
  design-of-experiment documentation.

Write the source manifest:

```bash
uv run chatter-twin kit-industrial-manifest \
  --out data/raw/kit_industrial/source_manifest.json
```

Download the tar with resume support:

```bash
mkdir -p data/raw/kit_industrial
curl -L -C - --fail \
  -o data/raw/kit_industrial/10.35097-hvvwn1kfwf7qt48z.tar \
  "https://radar.kit.edu/radar-backend/archives/hvvwn1kfwf7qt48z/versions/1/content"
```

Verify the downloaded size before extracting:

```bash
python - <<'PY'
from pathlib import Path
path = Path("data/raw/kit_industrial/10.35097-hvvwn1kfwf7qt48z.tar")
print(path.stat().st_size)
assert path.stat().st_size == 44_584_092_672
PY
```

Do not extract the whole archive blindly. First inspect the members:

```bash
tar -tf data/raw/kit_industrial/10.35097-hvvwn1kfwf7qt48z.tar \
  > data/raw/kit_industrial/tar_members.txt
```

Then extract only documentation and one experiment folder for schema discovery.
Inspect the payload and DoE labels:

```bash
uv run chatter-twin inspect-kit-industrial \
  --source data/raw/kit_industrial/extracted/10.35097-hvvwn1kfwf7qt48z/data/dataset/Data.zip \
  --out data/raw/kit_industrial/kit_inspection.json
```

The first adapter imports processed controller `hfdata.csv` signals into the
replay schema. It uses DoE comments for coarse trial-level labels; the explicit
chatter trial is `IM-02F-A01`.

```bash
uv run chatter-twin ingest-kit-industrial \
  --source data/raw/kit_industrial/extracted/10.35097-hvvwn1kfwf7qt48z/data/dataset/Data.zip \
  --out results/kit_controller_replay_stable_set_vs_chatter \
  --trials TF-01,TF-02,IM-01F,IM-01R,IMP-BASE,IMP-01,IM-02F-A01 \
  --window 1.0 \
  --stride 0.5 \
  --horizon 2.0 \
  --sample-rate 500
```

Current caveat: this is a controller-only bridge using `LOAD|6` and
`CURRENT|6`. It is useful for industrial-domain smoke tests, but the serious
next adapter should target the synchronized MATLAB acceleration/force files and
derive time-local chatter/onset labels inside the chatter trial.

The synchronized MATLAB v7.3 path now has an optional HDF5 reader:

```bash
uv run --extra mat chatter-twin inspect-kit-mat \
  --source data/raw/kit_industrial/extracted/10.35097-hvvwn1kfwf7qt48z/data/dataset/Data.zip \
  --trial IM-02F-A01 \
  --out data/raw/kit_industrial/kit_mat_inspection_IM-02F-A01.json
```

The decoded timetable exposes `97` variables at `10 kHz`, including
`xForce`, `yForce`, `zForce`, `xAcceleration`, `yAcceleration`, and
`zAcceleration`.

Import a bounded acceleration replay:

```bash
uv run --extra mat chatter-twin ingest-kit-mat \
  --source data/raw/kit_industrial/extracted/10.35097-hvvwn1kfwf7qt48z/data/dataset/Data.zip \
  --out results/kit_mat_replay_im01f_vs_chatter_200k \
  --trials IM-01F,IM-02F-A01 \
  --window 0.1 \
  --stride 0.05 \
  --horizon 0.25 \
  --signal-names xAcceleration,yAcceleration \
  --max-samples-per-trial 200000
```

This produced `798` high-rate acceleration windows on the first bounded run,
balanced between stable `IM-01F` and chatter-labeled `IM-02F-A01`. Treat that
as a high-rate ingestion validation result only; the labels are still trial
level. The next useful layer is pseudo-onset labeling inside `IM-02F-A01`.

## KIT Synchronized Force+Acceleration Replay

The current strongest public-data bridge uses the synchronized MATLAB files
directly from `Data.zip` and imports all six calibrated high-rate channels:

```bash
uv run --extra mat chatter-twin ingest-kit-mat \
  --source data/raw/kit_industrial/extracted/10.35097-hvvwn1kfwf7qt48z/data/dataset/Data.zip \
  --out results/kit_mat_replay_force_accel_standardized_full \
  --trials IM-01F,IM-02F-A01 \
  --window 0.1 \
  --stride 0.05 \
  --horizon 0.5 \
  --signal-names xAcceleration,yAcceleration,zAcceleration,xForce,yForce,zForce \
  --standardize-signals
```

This produced `18,203` windows:

| Trial | Label | Samples read | Windows | Signals |
|---|---|---:|---:|---|
| `IM-01F` | stable | 6,021,595 | 12,042 | 3-axis acceleration + 3-axis force |
| `IM-02F-A01` | slight | 3,081,459 | 6,161 | 3-axis acceleration + 3-axis force |

The channels are robust-standardized per trial and signal before feature
extraction. This prevents force units from numerically swamping acceleration
features in the pseudo-label score.

Then generate exploratory time-local pseudo labels inside the chatter-labeled
trial:

```bash
uv run chatter-twin pseudo-label-replay \
  --dataset results/kit_mat_replay_force_accel_standardized_full \
  --out results/kit_mat_pseudo_onset_force_accel_standardized_full_exploratory \
  --positive-scenarios IM-02F-A01 \
  --horizon 0.5 \
  --transition-floor 0.35 \
  --slight-floor 0.6 \
  --severe-floor 1.0
```

Pseudo-label output:

| Item | Value |
|---|---:|
| Total windows | 18,203 |
| Changed current labels | 6,148 |
| Lead-time candidate windows | 102 |
| Current chatter-positive windows | 22 |
| Stable windows after relabeling | 17,925 |
| Transition windows | 256 |
| Slight windows | 13 |
| Severe windows | 9 |

The resulting horizon-risk sanity baseline is:

```bash
uv run chatter-twin train-risk \
  --dataset results/kit_mat_pseudo_onset_force_accel_standardized_full_exploratory \
  --out results/kit_mat_force_accel_pseudo_onset_horizon_baseline \
  --model hist_gb \
  --feature-set temporal \
  --target horizon \
  --split-mode row \
  --test-fraction 0.3 \
  --seed 509
```

| Metric | Value |
|---|---:|
| Test accuracy | 0.986 |
| Test chatter F1 | 0.959 |
| Test intervention F1 | 0.887 |
| Test lead-time F1 | 0.952 |
| Test lead-time recall | 0.938 |
| Test lead-time mean detected lead | 0.217 s |

And a conservative manual-threshold shadow replay:

```bash
uv run chatter-twin shadow-eval \
  --model-dir results/kit_mat_force_accel_pseudo_onset_horizon_baseline \
  --out results/kit_mat_force_accel_pseudo_onset_shadow_eval_manual_050 \
  --threshold-source manual \
  --warning-threshold 0.5 \
  --warning-feed 0.92 \
  --warning-spindle 1.04
```

| Metric | Value |
|---|---:|
| Warning fraction | 0.0066 |
| Action fraction | 0.0075 |
| Event F1 | 1.000 |
| Event recall | 1.000 |
| False warning episodes | 0 |
| Relative MRR proxy | 0.9997 |

Caveat: this is still an exploratory pseudo-label experiment, not final
machine validation. Window-level lead-time metrics are the most useful sanity
check here. Event-level lead time is not a publishable result because the row
split scatters one physical chatter episode across train and test windows.

A stricter temporal holdout now exists:

```bash
uv run chatter-twin train-risk \
  --dataset results/kit_mat_pseudo_onset_force_accel_standardized_full_exploratory \
  --out results/kit_mat_force_accel_pseudo_onset_horizon_time_block \
  --model hist_gb \
  --feature-set temporal \
  --target horizon \
  --split-mode time_block \
  --test-fraction 0.3 \
  --seed 510
```

This trains on early windows and tests on later windows inside both `IM-01F`
and `IM-02F-A01`. It is a more honest check for the single long KIT chatter
trial:

| Metric | Row split | Time-block split |
|---|---:|---:|
| Test accuracy | 0.986 | 0.933 |
| Test chatter F1 | 0.959 | 0.043 |
| Test intervention F1 | 0.887 | 0.444 |
| Default lead-time F1 | 0.952 | 0.026 |
| Selected-threshold lead-time F1 | 0.938 | 0.301 |
| Event warning F1 | 1.000 | 0.000 |

Interpretation: the row-split result is useful as a pipeline sanity check, but
the time-block result is the current honest benchmark. The model reacts to some
late chatter windows but does not warn before the first later chatter event.
Increasing the pseudo-label horizon to `1.0 s` did not fix this; time-block
lead-time and event-warning F1 both fell to `0.000`.

## Purdue MT Cutting Sound Dataset

The second public bridge is Purdue LAMM's CNC machine tool cutting sound
dataset:

- Repository: <https://github.com/purduelamm/mt_cutting_dataset>
- IMI lab subset: 18 experiment folders, three synchronized 48 kHz sound
  sensors, one `cutting.csv` per experiment, and `labeling_all_details.xlsx`
  with operator chatter labels.
- KRPM industry subset: one longer industry recording with cutting intervals,
  useful later for domain shift but without the same detailed IMI chatter
  workbook.

Write the source manifest:

```bash
uv run chatter-twin mt-cutting-manifest \
  --out data/raw/mt_cutting_dataset/source_manifest.json
```

Clone the dataset on the data machine:

```bash
git clone --depth 1 https://github.com/purduelamm/mt_cutting_dataset.git \
  data/raw/mt_cutting_dataset/repo
```

Inspect labels:

```bash
uv run chatter-twin inspect-mt-cutting \
  --source data/raw/mt_cutting_dataset/repo \
  --out data/raw/mt_cutting_dataset/inspection.json
```

The current importer reads the IMI workbook with the standard-library XLSX
parser, maps `max(Chatter operator 1, Chatter operator 2)` as `0 -> stable`,
`1 -> slight`, and `2 -> severe`, and slices WAV intervals from `cutting.csv`.

First bounded replay from the near-spindle/internal sound sensor:

```bash
uv run chatter-twin ingest-mt-cutting \
  --source data/raw/mt_cutting_dataset/repo \
  --out results/mt_cutting_sensor1_replay_12k \
  --sensors 1 \
  --window 0.1 \
  --stride 0.05 \
  --horizon 0.25 \
  --max-windows 12000
```

Replay label counts:

| Label | Windows |
|---|---:|
| stable | 5,913 |
| slight | 2,389 |
| severe | 3,698 |

Episode-held-out current-window baseline:

```bash
uv run chatter-twin train-risk \
  --dataset results/mt_cutting_sensor1_replay_12k \
  --out results/mt_cutting_sensor1_current_episode_baseline_12k \
  --model hist_gb \
  --feature-set temporal \
  --target current \
  --split-mode episode \
  --test-fraction 0.3 \
  --validation-fraction 0.2 \
  --seed 512
```

| Metric | Value |
|---|---:|
| Test accuracy | 0.692 |
| Test chatter F1 | 0.777 |
| Test intervention F1 | 0.777 |

Experiment-folder holdout is stricter:

```bash
uv run chatter-twin train-risk \
  --dataset results/mt_cutting_sensor1_replay_12k \
  --out results/mt_cutting_sensor1_current_scenario_baseline_12k \
  --model hist_gb \
  --feature-set temporal \
  --target current \
  --split-mode scenario \
  --test-fraction 0.3 \
  --validation-fraction 0.2 \
  --seed 513
```

| Metric | Value |
|---|---:|
| Held-out experiments | `Exp0-2`, `Exp1-1-2`, `Exp1-3`, `Exp1-8`, `Exp2-1` |
| Test accuracy | 0.566 |
| Test chatter F1 | 0.637 |
| Test intervention F1 | 0.637 |

An all-three-sensor import was also tested with the same 12k cap and scenario
holdout:

| Input | Test accuracy | Test chatter F1 |
|---|---:|---:|
| sensor 1 only | 0.566 | 0.637 |
| sensors 0,1,2 norm | 0.680 | 0.550 |

Interpretation: Purdue gives a much better multi-cut public chatter benchmark
than KIT for current-window signal validation. It still does not solve
early-warning validation because labels are per cutting path, not onset
timestamps.
