# Offline Risk Training

`chatter-twin train-risk` trains CPU-only risk estimators from a replay dataset
produced by `chatter-twin export-synthetic`.

The default model is intentionally small and inspectable: a softmax classifier
over per-window metadata/features from `windows.csv`. A stronger histogram
gradient-boosted tree baseline is also available with optional probability
calibration. Both stay laptop/CPU friendly and create baselines before heavier
sequence models are introduced.

## Command

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_replay_demo \
  --out results/risk_model_demo \
  --epochs 800
```

Train the stronger tree baseline with sigmoid calibration:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_randomized_demo \
  --out results/risk_model_hist_gb_episode_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode episode \
  --validation-fraction 0.25
```

For a less optimistic evaluation, split by complete synthetic episodes:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_randomized_demo \
  --out results/risk_model_episode_holdout_demo \
  --epochs 800 \
  --split-mode episode
```

To test generalization on a held-out parameter family, hold out the high or low
tail of a randomization/context column:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_randomized_demo \
  --out results/risk_model_axial_depth_holdout_demo \
  --epochs 800 \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

To choose the lead-time warning threshold without peeking at the test split,
reserve part of the training split for validation:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_interaction_validated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high \
  --validation-fraction 0.25
```

Available split modes:

| Mode | Meaning |
|---|---|
| `row` | Stratified random row split. Fast smoke test, but optimistic for overlapping replay windows. |
| `episode` | Holds out complete scenario/episode cuts so adjacent windows do not leak across train/test. |
| `scenario` | Holds out complete scenario or experiment names. Useful for checking whether a signal model generalizes to unseen experiments. |
| `time_block` | Trains on early windows and tests on later windows inside each episode. Useful for single long real-machine trials where complete episode holdout would leave too few chatter examples. |
| `parameter_family` | Holds out complete episodes from the high or low tail of a selected context/randomization column. |

Available model types:

| Model | Meaning |
|---|---|
| `softmax` | Transparent NumPy multiclass baseline with z-score scaling. |
| `hist_gb` | Scikit-learn histogram gradient-boosted tree baseline. Supports `--calibration sigmoid` or `--calibration isotonic` when the training split has enough examples per class. |

Available feature sets:

| Feature set | Meaning |
|---|---|
| `base` | Static physics/context/spectral features. Compatible with older replay exports. |
| `temporal` | Adds within-episode progress, per-window deltas/growth rates, and EWMA features to help identify transition/onset behavior. Requires replay schema `chatter-window-v3` or newer. |
| `profile` | Adds per-window process-profile features such as effective axial-depth scale and cutting coefficients. Requires replay schema `chatter-window-v5` or newer. |
| `profile_temporal` | Combines profile and temporal features. This is a stable CPU baseline for onset/lead-time experiments. |
| `interaction` | Adds physics-shaped derived features such as depth-to-critical ratio, critical-depth proxy, distance-to-boundary, process-force proxy, and profile severity. Requires replay schema `chatter-window-v5` or newer. |
| `interaction_temporal` | Combines temporal, profile, and derived interaction features. This is the strongest current CPU diagnostic baseline for hard onset holdouts. |

Available targets:

| Target | Meaning |
|---|---|
| `current` | Train against the current-window label from `label_id`. |
| `horizon` | Train against `horizon_label_id`, the worst current-or-upcoming label within the export horizon. Requires replay schema `chatter-window-v4` or newer. |

## Outputs

| File | Meaning |
|---|---|
| `model.json` | Model summary, feature columns, preprocessing, class labels, and either softmax parameters or tree metadata. |
| `model.joblib` | Serialized scikit-learn estimator when `--model hist_gb` is used. |
| `metrics.json` | Train/test accuracy, macro F1, binary chatter F1, intervention F1, Brier scores, and confusion matrix. |
| `predictions.csv` | Test-window predictions, chatter scores, row timing, current labels, and horizon/lead-time fields. |
| `confusion_matrix.csv` | Confusion matrix using the replay labels. |
| `report.md` | Human-readable summary. |

The binary chatter metric treats `slight` and `severe` as chatter-positive for
the selected target. With `--target horizon`, this measures early-warning
classification for chatter that is already present or imminent within the
configured replay horizon.

The intervention metric treats `transition`, `slight`, and `severe` as
control-positive. This is the more useful score for a safety shield or
supervisory controller that should react before severe chatter.

For true lead-time experiments, prefer a schema `chatter-window-v5` replay
export that includes the `onset` scenario. That scenario ramps process severity
inside each episode, so some current `stable`/`transition` windows have
`future_chatter_within_horizon=True` before the current label reaches `slight`
or `severe`.

The lead-time section in `metrics.json` and `report.md` scores only current
`stable`/`transition` rows. Its positive class is:

```text
future_chatter_within_horizon=True
```

This separates true early warning from the easier task of recognizing chatter
after the current window is already labeled `slight` or `severe`.

The same section also stores a threshold sweep over the predicted chatter score.
When `--validation-fraction` is nonzero, `threshold_selection.selected_threshold`
is chosen on the validation split by maximum lead-time F1, then evaluated on
the test split. Without a validation split, the code falls back to training
split selection for backward-compatible smoke tests. `test_oracle_best_f1` is
diagnostic only; do not report it as deployable performance.

`event_warning` in `metrics.json` and `report.md` is the controller-facing
version of the same idea. It groups rows by `scenario::episode`, finds the
first current-window `slight`/`severe` label, and counts the episode as detected
only if a warning appeared before that first chatter window. Quiet episodes with
a warning are counted as false-warning episodes. Use this metric when reporting
shadow-mode readiness; window-level lead-time F1 can be high while episode-level
false warnings are still unacceptable.

After training, use `chatter-twin shadow-eval` to convert `predictions.csv` into
offline recommendation traces and MRR/event-level policy metrics. That step is
documented in `docs/SHADOW_RECOMMENDATIONS.md`.
