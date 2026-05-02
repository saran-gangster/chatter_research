# Real Dataset On-Ramp

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
