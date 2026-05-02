# Real Dataset On-Ramp

## Dataset Ladder

| Dataset | Use Now | Gap It Covers | Main Limitation |
|---|---|---|---|
| i-CNC Zenodo | current-window chatter signal validation | real 10 kHz vibration plus chatter status | package-level labels, no process depth/feed/FRF/context |
| KIT industrial CNC milling | process-context/anomaly validation | continuous controller data, force, acceleration, NC/CAD, anomaly documentation | 44.6 GB tar, still not closed-loop intervention data |

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
The first useful adapter should target the synchronized MATLAB files and/or
preprocessed controller CSV files, then map acceleration/force/controller
windows into the existing replay schema.
