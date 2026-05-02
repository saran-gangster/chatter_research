# Chatter Twin Experiment Log

This log captures reproducible local artifacts and the main lesson from each
offline experiment. All commands are CPU-only.

## 2026-05-01 - Randomized Replay Baseline

Dataset:

```bash
rtk uv run chatter-twin export-synthetic \
  --scenarios stable,near_boundary,unstable \
  --episodes 4 \
  --duration 0.6 \
  --window 0.1 \
  --stride 0.05 \
  --randomize \
  --out results/synthetic_randomized_demo
```

Label counts:

| Label | Windows |
|---|---:|
| stable | 24 |
| transition | 31 |
| slight | 63 |
| severe | 14 |

Best CPU model tested on high axial-depth holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_randomized_demo \
  --out results/risk_model_hist_gb_temporal_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

Result:

| Split | Accuracy | Macro F1 | Chatter F1 | Lesson |
|---|---:|---:|---:|---|
| high axial-depth holdout | 0.545 | 0.369 | 0.800 | Transition windows were still confused with stable/slight. |

## 2026-05-01 - Transition-Focused Replay

Dataset:

```bash
rtk uv run chatter-twin export-synthetic \
  --scenarios stable,near_boundary,unstable \
  --episodes 4 \
  --duration 0.6 \
  --window 0.1 \
  --stride 0.05 \
  --randomize \
  --focus-transitions \
  --transition-candidates 12 \
  --min-transition-windows 3 \
  --out results/synthetic_transition_focus_demo
```

Label counts:

| Label | Windows |
|---|---:|
| stable | 20 |
| transition | 68 |
| slight | 44 |

Episode holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_transition_focus_demo \
  --out results/risk_model_hist_gb_transition_focus_episode_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --split-mode episode
```

| Split | Accuracy | Macro F1 | Chatter F1 |
|---|---:|---:|---:|
| episode holdout | 0.818 | 0.771 | 1.000 |

High axial-depth holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_transition_focus_demo \
  --out results/risk_model_hist_gb_transition_focus_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

| Split | Accuracy | Macro F1 | Chatter F1 | Lesson |
|---|---:|---:|---:|---|
| high axial-depth holdout | 0.864 | 0.864 | 1.000 | Targeted boundary sampling fixed much of the transition collapse. |

Remaining caveat: this is still synthetic and label-derived from the current
hybrid risk estimator. The next step is to add horizon/onset labels and compare
whether the model predicts upcoming chatter before the current-window label
becomes slight/severe.

## 2026-05-01 - Horizon Target

Dataset regenerated with `chatter-window-v4` horizon labels:

```bash
rtk uv run chatter-twin export-synthetic \
  --scenarios stable,near_boundary,unstable \
  --episodes 4 \
  --duration 0.6 \
  --window 0.1 \
  --stride 0.05 \
  --randomize \
  --focus-transitions \
  --transition-candidates 12 \
  --min-transition-windows 3 \
  --horizon 0.2 \
  --out results/synthetic_transition_focus_demo
```

Current-label counts:

| Label | Windows |
|---|---:|
| stable | 20 |
| transition | 68 |
| slight | 44 |

Horizon-label counts:

| Label | Windows |
|---|---:|
| stable | 2 |
| transition | 86 |
| slight | 44 |

Episode holdout with horizon target:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_transition_focus_demo \
  --out results/risk_model_hist_gb_horizon_episode_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --target horizon \
  --split-mode episode
```

| Split | Accuracy | Macro F1 | Chatter F1 | Intervention F1 |
|---|---:|---:|---:|---:|
| episode holdout | 0.970 | 0.659 | 1.000 | 0.985 |

High axial-depth holdout with horizon target:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_transition_focus_demo \
  --out results/risk_model_hist_gb_horizon_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

| Split | Accuracy | Macro F1 | Chatter F1 | Intervention F1 |
|---|---:|---:|---:|---:|
| high axial-depth holdout | 1.000 | 1.000 | 1.000 | 1.000 |

Important caveat: this horizon target mostly converts stable windows into
transition windows. In this dataset, there are no current stable/transition
windows that become `slight` or `severe` within 0.2 seconds; `future_chatter`
positives are already current `slight` windows. The next dataset improvement is
to generate progressive onset episodes where stable/transition windows precede
slight/severe chatter by a measurable lead time.

## 2026-05-01 - Progressive Onset Replay

Dataset regenerated with schema `chatter-window-v5`, including the new `onset`
scenario with within-episode axial-depth/cutting-coefficient ramping:

```bash
rtk uv run chatter-twin export-synthetic \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 4 \
  --duration 0.9 \
  --window 0.1 \
  --stride 0.05 \
  --randomize \
  --focus-transitions \
  --transition-candidates 12 \
  --min-transition-windows 3 \
  --horizon 0.25 \
  --out results/synthetic_onset_demo
```

Current-label counts:

| Label | Windows |
|---|---:|
| stable | 38 |
| transition | 122 |
| slight | 100 |
| severe | 12 |

Horizon-label counts:

| Label | Windows |
|---|---:|
| stable | 11 |
| transition | 120 |
| slight | 107 |
| severe | 34 |

Lead-time check:

| Measure | Value |
|---|---:|
| `future_chatter_within_horizon=True` windows | 132 |
| current stable/transition windows that become slight/severe within horizon | 29 |
| lead-time values observed | 0.05s, 0.10s, 0.15s, 0.20s, 0.25s |

Episode holdout with horizon target:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_onset_horizon_episode_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --target horizon \
  --split-mode episode
```

| Split | Accuracy | Macro F1 | Chatter F1 | Intervention F1 | Lead-Time F1 | Lead-Time Recall |
|---|---:|---:|---:|---:|---:|---:|
| episode holdout | 0.897 | 0.694 | 0.861 | 0.954 | 0.476 | 1.000 |

Lead-time detail: on the test split, 5 of 39 current stable/transition windows
became slight/severe within the 0.25s horizon. The model caught all 5, but
raised 11 extra early warnings, so precision was 0.312 and mean detected lead
time was 0.15s.

High axial-depth-scale holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_onset_horizon_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

| Split | Accuracy | Macro F1 | Chatter F1 | Intervention F1 | Lead-Time F1 | Lead-Time Recall |
|---|---:|---:|---:|---:|---:|---:|
| high axial-depth-scale holdout | 0.671 | 0.423 | 0.686 | 0.975 | 0.000 | 0.000 |

Lesson: the replay generator now creates genuine lead-time positives, so the
horizon target is meaningful. The episode split catches incipient chatter but
with many extra warnings. The high axial-depth holdout misses all lead-time
positives at the default 0.5 chatter threshold, so the next modeling target is
profile-aware features, threshold calibration, and/or a small sequence model
after preserving this harder split.

## 2026-05-01 - Profile-Aware Lead-Time Baseline

This run adds schema-v5 process-profile features to the CPU tree baseline:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_profile_onset_horizon_episode_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set profile_temporal \
  --target horizon \
  --split-mode episode
```

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_profile_onset_horizon_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set profile_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

Comparison against the previous temporal-only model:

| Split | Feature Set | Accuracy | Chatter F1 | Intervention F1 | Lead-Time F1 | Lead-Time Recall | Warnings / Positives |
|---|---|---:|---:|---:|---:|---:|---:|
| episode holdout | temporal | 0.897 | 0.861 | 0.954 | 0.476 | 1.000 | 16 / 5 |
| episode holdout | profile_temporal | 0.735 | 1.000 | 0.954 | 1.000 | 1.000 | 5 / 5 |
| high axial-depth holdout | temporal | 0.671 | 0.686 | 0.975 | 0.000 | 0.000 | 0 / 9 |
| high axial-depth holdout | profile_temporal | 0.612 | 0.694 | 0.982 | 0.364 | 0.444 | 13 / 9 |

Threshold-sweep diagnostic:

| Split | Feature Set | Train-Selected Threshold | Test F1 at Train Threshold | Test Oracle F1 |
|---|---|---:|---:|---:|
| episode holdout | temporal | 0.40 | 0.256 | 1.000 |
| episode holdout | profile_temporal | 0.50 | 1.000 | 1.000 |
| high axial-depth holdout | temporal | 0.40 | 0.000 | 0.234 |
| high axial-depth holdout | profile_temporal | 0.50 | 0.364 | 0.500 |

Lesson: profile-aware features reduce false warnings on episode holdout and
recover some high axial-depth lead-time positives, but the hard holdout is still
not solved. The oracle sweep shows that threshold tuning alone cannot make the
temporal model useful on the hard split; profile information is carrying real
signal. Next best modeling step is either interaction features around
margin-depth growth or a small CPU sequence model before moving to GPU RL.

## 2026-05-01 - Physics-Interaction Lead-Time Baseline

This run adds derived physics-shaped features on top of the profile/temporal
feature set:

* depth-to-critical ratio from the signed physics margin,
* critical-depth and distance-to-boundary proxies,
* cutting-force/process-severity proxies,
* interactions between margin pressure and RMS/chatter-energy growth.

Calibrated interaction baseline:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_interaction_onset_horizon_axial_depth_holdout_demo \
  --model hist_gb \
  --calibration sigmoid \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

Uncalibrated interaction baseline:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_interaction_onset_horizon_axial_depth_holdout_uncalibrated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high
```

High axial-depth holdout comparison:

| Feature Set / Calibration | Accuracy | Chatter F1 | Intervention F1 | Lead-Time F1 | Lead-Time Recall | Warnings / Positives |
|---|---:|---:|---:|---:|---:|---:|
| profile_temporal / sigmoid | 0.612 | 0.694 | 0.982 | 0.364 | 0.444 | 13 / 9 |
| interaction_temporal / sigmoid | 0.624 | 0.720 | 0.987 | 0.364 | 0.444 | 13 / 9 |
| interaction_temporal / none | 0.706 | 0.807 | 0.981 | 0.621 | 1.000 | 20 / 9 |

Threshold diagnostic on high axial-depth holdout:

| Feature Set / Calibration | Train-Selected Threshold | Test F1 at Train Threshold | Test Oracle F1 |
|---|---:|---:|---:|
| profile_temporal / sigmoid | 0.50 | 0.364 | 0.500 |
| interaction_temporal / sigmoid | 0.40 | 0.412 | 0.432 |
| interaction_temporal / none | 0.05 | 0.391 | 0.621 |

Lesson: the derived interaction features help overall hard-holdout
classification, and the uncalibrated tree recovers every lead-time positive at
the default 0.5 warning threshold. Calibration is not automatically helpful in
this tiny synthetic regime; sigmoid calibration smoothed scores enough to hurt
hard-holdout early warning. The train-selected threshold remains unreliable
because the training split is too small and too optimistic, so the next serious
step is a dedicated validation split or event-level sequence model before any
RL policy uses these scores.

## 2026-05-01 - Validation-Selected Warning Threshold

This run uses the same uncalibrated `interaction_temporal` CPU baseline but
reserves 25% of the training episodes for threshold selection:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_interaction_onset_horizon_axial_depth_holdout_validated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high \
  --validation-fraction 0.25
```

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_demo \
  --out results/risk_model_hist_gb_interaction_onset_horizon_episode_validated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode episode \
  --validation-fraction 0.25
```

Comparison:

| Split | Threshold Source | Fit / Validation / Test Windows | Accuracy | Chatter F1 | Lead-Time F1 | Lead-Time Recall | Warnings / Positives |
|---|---|---:|---:|---:|---:|---:|---:|
| high axial-depth holdout | train | 187 / 0 / 85 | 0.706 | 0.807 | 0.621 | 1.000 | 20 / 9 |
| high axial-depth holdout | validation | 136 / 51 / 85 | 0.694 | 0.821 | 0.615 | 0.444 | 4 / 9 |
| episode holdout | train | 204 / 0 / 68 | 0.809 | 0.944 | 0.714 | 1.000 | 9 / 5 |
| episode holdout | validation | 136 / 68 / 68 | 0.471 | 0.805 | 0.400 | 1.000 | 20 / 5 |

Lesson: validation-selected thresholding gives a cleaner protocol but exposes
how data-starved the current synthetic split is. Holding out validation episodes
reduces fit data enough to hurt episode-holdout accuracy, and on the hard
high-depth split the validation-selected model is more conservative: fewer
warnings, higher precision, but lower recall. Before hardware/RL, the next best
step is event-level sequence evaluation or more onset episodes; threshold
tuning alone is not a stable answer yet.

## 2026-05-01 - Event-Level Warning Evaluation

This update adds `event_warning` metrics to the same offline reports. Instead of
scoring every replay window independently, each scenario/episode is scored once:

* event episode: contains at least one current-window `slight` or `severe`
  chatter label;
* detected event: the model warned on a stable/transition window before that
  first chatter label;
* false-warning episode: a quiet episode that still received a pre-chatter
  warning.

Hard high axial-depth holdout:

| Model / Protocol | Episodes | Event Episodes | Warning Episodes | Detected | False Warnings | Event Precision | Event Recall | Event F1 | Mean Event Lead Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interaction_temporal / no validation / default 0.5 threshold | 5 | 2 | 2 | 2 | 0 | 1.000 | 1.000 | 1.000 | 0.500s |
| interaction_temporal / validation threshold | 5 | 2 | 1 | 1 | 0 | 1.000 | 0.500 | 0.667 | 0.200s |

Episode holdout with validation threshold:

| Episodes | Event Episodes | Warning Episodes | Detected | False Warnings | Event Precision | Event Recall | Event F1 | Mean Event Lead Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 3 | 1 | 2 | 0.333 | 0.500 | 0.400 | 0.250s |

Lesson: event-level metrics are the right reporting unit for shadow-mode
control. The no-validation hard-holdout result looks strong, but its threshold
procedure is less defensible. The validation-threshold result is more honest:
it catches one of two hard-holdout chatter events with no quiet-episode false
warnings. The immediate next step is to generate more onset episodes so
validation threshold selection is not starved, then rerun event-level metrics.

## 2026-05-01 - Larger Onset Event Dataset

The previous event-level validation was starved: only 4 episodes per scenario.
This dataset increases to 12 episodes per scenario while keeping the same onset,
randomization, transition-focus, and horizon settings.

```bash
rtk uv run chatter-twin export-synthetic \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 12 \
  --duration 0.9 \
  --window 0.1 \
  --stride 0.05 \
  --randomize \
  --focus-transitions \
  --transition-candidates 12 \
  --min-transition-windows 3 \
  --horizon 0.25 \
  --out results/synthetic_onset_events_demo
```

Dataset counts:

| Measure | Value |
|---|---:|
| replay windows | 816 |
| episodes | 48 |
| current stable windows | 88 |
| current transition windows | 378 |
| current slight windows | 328 |
| current severe windows | 22 |
| event episodes with current slight/severe chatter | 27 |
| episodes with pre-chatter lead windows | 18 |
| current stable/transition windows that become slight/severe within horizon | 87 |

Validated hard high axial-depth holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_events_demo \
  --out results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode parameter_family \
  --holdout-column axial_depth_scale \
  --holdout-tail high \
  --validation-fraction 0.25
```

Validated episode holdout:

```bash
rtk uv run chatter-twin train-risk \
  --dataset results/synthetic_onset_events_demo \
  --out results/risk_model_hist_gb_interaction_onset_events_episode_validated_demo \
  --model hist_gb \
  --calibration none \
  --feature-set interaction_temporal \
  --target horizon \
  --split-mode episode \
  --validation-fraction 0.25
```

Comparison to the smaller onset dataset:

| Dataset / Split | Fit / Val / Test Windows | Test Event Episodes | Accuracy | Chatter F1 | Window Lead-Time F1 | Event F1 | Event Recall | False-Warning Episodes | Mean Event Lead Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small / high axial-depth holdout | 136 / 51 / 85 | 2 | 0.694 | 0.821 | 0.615 | 0.667 | 0.500 | 0 | 0.200s |
| large / high axial-depth holdout | 425 / 136 / 255 | 7 | 0.839 | 0.927 | 0.821 | 0.727 | 0.571 | 0 | 0.200s |
| large / episode holdout | 408 / 136 / 272 | 9 | 0.882 | 0.964 | 0.800 | 0.800 | 0.667 | 0 | 0.300s |

Validation-selected event threshold results:

| Split | Selected Event Threshold | Detected / Event Episodes | Warning Episodes | Event F1 | Mean Event Lead Time |
|---|---:|---:|---:|---:|---:|
| large high axial-depth holdout | 0.95 | 4 / 7 | 4 | 0.727 | 0.188s |
| large episode holdout | 0.05 | 6 / 9 | 7 | 0.750 | 0.308s |

Lesson: more onset episodes substantially stabilize the event-level story. The
validated hard-holdout model now sees 7 chatter-event episodes instead of 2,
improves window lead-time F1 from 0.615 to 0.821, and improves event F1 from
0.667 to 0.727 with no false-warning episodes at the default threshold. This is
now a credible CPU-only early-warning baseline. The next step is to add a
shadow-mode policy/recommendation evaluator that turns these warnings into
bounded feed/spindle override recommendations and scores event detection,
warning burden, and MRR impact together.

## 2026-05-01 - Shadow-Mode Recommendation Evaluation

This update adds `chatter-twin shadow-eval`, an offline recommendation layer
that turns test-set risk predictions into bounded feed/spindle override traces.
It applies event-threshold hysteresis, clips/rate-limits actions, and reports
event detection together with warning burden and MRR proxy. No CNC command is
issued.

Event-threshold policy:

```bash
rtk uv run chatter-twin shadow-eval \
  --model-dir results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --threshold-source event \
  --warning-feed 0.92 \
  --warning-spindle 1.04
```

Default-threshold diagnostic:

```bash
rtk uv run chatter-twin shadow-eval \
  --model-dir results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_policy_onset_events_axial_depth_holdout_default_threshold_demo \
  --threshold-source default \
  --warning-feed 0.92 \
  --warning-spindle 1.04
```

Results on the hard high axial-depth holdout:

| Policy | Threshold | Windows | Episodes | Warning Fraction | Relative MRR Proxy | Detected / Event Episodes | False Warnings | Event Precision | Event Recall | Event F1 | Mean Event Lead Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| event-selected | 0.95 | 255 | 15 | 0.400 | 0.984 | 4 / 7 | 0 | 1.000 | 0.571 | 0.727 | 0.188s |
| default | 0.50 | 255 | 15 | 0.404 | 0.983 | 4 / 7 | 0 | 1.000 | 0.571 | 0.727 | 0.200s |

Artifacts:

| Policy | Output |
|---|---|
| event-selected | `results/shadow_policy_onset_events_axial_depth_holdout_validated_demo` |
| default | `results/shadow_policy_onset_events_axial_depth_holdout_default_threshold_demo` |

Lesson: the CPU-only estimator is now connected to a deployable-style shadow
recommendation trace. On this hard holdout it catches 4 of 7 chatter-event
episodes before current chatter labels, produces no false-warning quiet
episodes, and keeps the simple MRR proxy around 0.984. The next best step is to
add a rule/MPC counterfactual simulator that applies these shadow actions to the
2-DOF process model, so recommendations can be scored by estimated chatter
suppression rather than only by pre-event warning quality.

## 2026-05-02 - Shadow Counterfactual and Action Sweep

This update adds two CPU-only checks after `shadow-eval`:

* `shadow-counterfactual`: joins `recommendations.csv` with the source replay
  `windows.csv`, rebuilds each window's process context, and simulates baseline
  vs recommended feed/spindle overrides.
* `shadow-action-sweep`: keeps the same warning timing but tests a small grid
  of feed/spindle action pairs.

Counterfactual replay of the event-selected policy:

```bash
rtk uv run chatter-twin shadow-counterfactual \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

Result:

| Windows | Action Fraction | Relative MRR Proxy | Baseline Risk | Shadow Risk | Risk Reduction | Baseline Chatter Fraction | Shadow Chatter Fraction | Mitigated Events | Worsened Events |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 255 | 0.400 | 0.984 | 0.400 | 0.406 | -0.005 | 0.294 | 0.310 | 0 | 0 |

Action sweep:

```bash
rtk uv run chatter-twin shadow-action-sweep \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_action_sweep_onset_events_axial_depth_holdout_validated_demo \
  --feed-values 0.88,0.92,1.0 \
  --spindle-values 0.96,1.0,1.04 \
  --sensor-noise 0.0
```

Best sweep result:

| Feed | Spindle | Relative MRR Proxy | Baseline Risk | Shadow Risk | Risk Reduction | Mitigated Events | Worsened Events |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 1.000 | 0.400 | 0.400 | 0.000 | 0 | 0 |

Selected sweep rows:

| Feed | Spindle | Relative MRR Proxy | Shadow Risk | Risk Reduction | Shadow Chatter Fraction | Worsened Events |
|---:|---:|---:|---:|---:|---:|---:|
| 0.92 | 1.04 | 0.983 | 0.406 | -0.005 | 0.310 | 0 |
| 0.92 | 1.00 | 0.968 | 0.401 | -0.000 | 0.294 | 0 |
| 1.00 | 0.96 | 0.984 | 0.425 | -0.025 | 0.357 | 1 |

Artifacts:

| Check | Output |
|---|---|
| counterfactual replay | `results/shadow_counterfactual_onset_events_axial_depth_holdout_validated_demo` |
| action sweep | `results/shadow_action_sweep_onset_events_axial_depth_holdout_validated_demo` |

Lesson: the early-warning model is more mature than the action policy. The
fixed recommendation `feed=0.92, spindle=1.04` catches events in shadow mode,
but the local process counterfactual does not show chatter-risk reduction.
Within the small tested grid, the best local action is no action. The next best
technical step is to move from static override pairs to a controller that
chooses spindle moves using the stability-margin model, then evaluate it with
the same counterfactual harness before any RL policy is trained.

## 2026-05-02 - Stability-Margin Shadow Policy

This update adds `shadow-stability-policy`. It keeps the event-warning timing
from `shadow-eval`, reconstructs each replay window's modal/cut context from
`windows.csv`, and searches bounded spindle overrides using the current
stability-margin estimate. Feed is held at 1.0 by default.

Policy generation:

```bash
rtk uv run chatter-twin shadow-stability-policy \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_stability_policy_onset_events_axial_depth_holdout_validated_demo \
  --min-spindle 0.90 \
  --max-spindle 1.10 \
  --candidates 21 \
  --feed 1.0 \
  --min-margin-improvement 0.05
```

Counterfactual replay:

```bash
rtk uv run chatter-twin shadow-counterfactual \
  --shadow-dir results/shadow_stability_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_stability_policy_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

Policy result:

| Windows | Warning Fraction | Action Fraction | Relative MRR Proxy | Mean Selected Margin Improvement | Event F1 | False Warnings |
|---:|---:|---:|---:|---:|---:|---:|
| 255 | 0.400 | 0.337 | 1.012 | 0.540 | 0.727 | 0 |

Counterfactual result:

| Baseline Risk | Shadow Risk | Risk Reduction | Baseline Chatter Fraction | Shadow Chatter Fraction | Mitigated Events | Worsened Events |
|---:|---:|---:|---:|---:|---:|---:|
| 0.400 | 0.407 | -0.006 | 0.294 | 0.278 | 0 | 0 |

Selected spindle moves were mostly `1.03`, `1.10`, `1.04`, `0.99`, and `1.01`.
The policy improved the explicit stability-margin score on warning windows, but
the local time-domain counterfactual still did not reduce mean chatter risk.
It did slightly reduce the chatter-positive window fraction without mitigating
whole episodes. This exposes the next model gap: the stability-margin surrogate
and the simulator/risk-estimator objective are not yet aligned. Before RL, the
next controller baseline should optimize the same counterfactual objective used
for evaluation, or the stability estimate should be recalibrated against the
time-domain simulator.

## 2026-05-02 - Counterfactual-Risk Shadow Policy

This update adds `shadow-counterfactual-policy`. It keeps the same event-warning
timing, but instead of using a fixed override or a stability-margin surrogate,
it simulates candidate actions on each warning window and selects only actions
that reduce the local hybrid chatter risk by at least a small threshold.

Policy generation:

```bash
rtk uv run chatter-twin shadow-counterfactual-policy \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --feed-values 1.0 \
  --spindle-values 0.96,1.0,1.04,1.08 \
  --min-risk-reduction 0.005 \
  --sensor-noise 0.0
```

Counterfactual replay:

```bash
rtk uv run chatter-twin shadow-counterfactual \
  --shadow-dir results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

Policy result:

| Windows | Warning Fraction | Action Fraction | Relative MRR Proxy | Mean Selected Risk Reduction | Event F1 | False Warnings |
|---:|---:|---:|---:|---:|---:|---:|
| 255 | 0.400 | 0.110 | 1.005 | 0.066 | 0.727 | 0 |

Counterfactual result:

| Baseline Risk | Shadow Risk | Risk Reduction | Baseline Chatter Fraction | Shadow Chatter Fraction | Mitigated Events | Worsened Events |
|---:|---:|---:|---:|---:|---:|---:|
| 0.400 | 0.393 | 0.007 | 0.294 | 0.282 | 0 | 0 |

Selected actions:

| Measure | Value |
|---|---:|
| action windows | 28 / 255 |
| warning windows | 102 / 255 |
| selected spindle `1.04` | 26 windows |
| selected spindle `1.08` | 2 windows |
| action-window risk reduction range | 0.006 to 0.199 |

Lesson: matching the action objective to the counterfactual evaluation finally
produces a positive local control result. The policy is selective, preserves
feed, slightly increases the MRR proxy, and reduces mean local risk without
worsening any event episode. It still does not mitigate whole episodes, because
the replay is local per window and does not propagate machine vibration state.
The next best step is to turn this into a sequential receding-horizon baseline
inside the Gym environment or a full-episode counterfactual simulator, so action
effects persist across the cut before RL is introduced.

## 2026-05-02 - Full-Episode Counterfactual Replay

This update extends the simulator to accept time-varying feed/spindle override
profiles and adds `shadow-episode-counterfactual`. Instead of resetting each
window independently, the command reconstructs each held-out replay episode,
applies the shadow policy as persistent controls, simulates the full episode,
and then slices the resulting signal back into the same window grid.

Counterfactual-risk policy replay:

```bash
rtk uv run chatter-twin shadow-episode-counterfactual \
  --shadow-dir results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_episode_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

Comparison across policies:

| Policy | Action Fraction | Relative MRR Proxy | Baseline Risk | Shadow Risk | Risk Reduction | Baseline Chatter Fraction | Shadow Chatter Fraction | Mitigated Events | Worsened Events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed `feed=0.92`, `spindle=1.04` | 0.400 | 0.984 | 0.484 | 0.485 | -0.001 | 0.396 | 0.388 | 0 | 0 |
| stability-margin selector | 0.337 | 1.012 | 0.484 | 0.488 | -0.004 | 0.396 | 0.369 | 0 | 0 |
| counterfactual-risk selector | 0.110 | 1.005 | 0.484 | 0.476 | 0.009 | 0.396 | 0.365 | 0 | 0 |

Artifacts:

| Policy | Output |
|---|---|
| fixed | `results/shadow_episode_counterfactual_fixed_policy_onset_events_axial_depth_holdout_validated_demo` |
| stability-margin | `results/shadow_episode_counterfactual_stability_policy_onset_events_axial_depth_holdout_validated_demo` |
| counterfactual-risk | `results/shadow_episode_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo` |

Lesson: the counterfactual-risk policy remains the only action policy that
improves mean risk when controls persist through full synthetic episodes. The
effect is still modest and does not eliminate whole chatter-event episodes, but
this is now a sequential, CPU-only baseline with aligned warning, action,
productivity, and risk metrics. The next step is to move this selector into a
receding-horizon controller interface and compare it against SAC/TD3 later,
only after the full-episode baseline is stable.

## 2026-05-02 - Counterfactual-Risk Controller Baseline

This update adds `cf` as a controller option in the existing rollout,
benchmark, and Gym-style evaluation path. Unlike the offline
`shadow-counterfactual-policy`, this controller proposes actions online from
the current risk estimate and machine state. It simulates candidate feed/spindle
windows with planning sensor noise disabled, compares each candidate against the
current-action baseline, and only moves when predicted local risk reduction
clears a threshold.

Smoke rollout:

```bash
rtk uv run chatter-twin rollout \
  --controller cf \
  --scenario unstable \
  --steps 10
```

Benchmark:

```bash
rtk uv run chatter-twin benchmark \
  --controllers fixed,rule,sld,mpc,cf \
  --scenarios near_boundary,unstable \
  --episodes 3 \
  --steps 12 \
  --out results/benchmark_counterfactual_controller_demo
```

Result:

| Scenario | Controller | Mean Risk | Final Risk | Relative MRR Proxy | Mean Reward |
|---|---|---:|---:|---:|---:|
| near_boundary | cf | 0.484 | 0.464 | 1.066 | -14.951 |
| near_boundary | fixed | 0.714 | 0.723 | 1.000 | -28.435 |
| near_boundary | rule | 0.486 | 0.520 | 1.052 | -15.142 |
| near_boundary | sld | 0.367 | 0.346 | 1.096 | -7.957 |
| near_boundary | mpc | 0.365 | 0.344 | 1.146 | -7.349 |
| unstable | cf | 0.701 | 0.701 | 0.960 | -28.281 |
| unstable | fixed | 0.722 | 0.709 | 1.000 | -29.227 |
| unstable | rule | 0.740 | 0.735 | 0.926 | -31.980 |
| unstable | sld | 0.731 | 0.728 | 1.096 | -29.666 |
| unstable | mpc | 0.731 | 0.728 | 1.096 | -29.666 |

Artifacts:

| Check | Output |
|---|---|
| controller benchmark | `results/benchmark_counterfactual_controller_demo` |

Lesson: `cf` is now a real controller candidate, not only an offline shadow
artifact. It improves over fixed/rule on both tested scenarios and is strongest
on the unstable cut in this small benchmark. SLD/MPC still outperform it on the
near-boundary case, so the next controller step should be a hybrid score that
combines counterfactual simulated risk with the stability-margin schedule rather
than treating them as separate baselines.

## 2026-05-02 - Hybrid Controller and Closed-Loop Benchmark

This update completes three controller-track tasks in code:

* `hybrid` combines counterfactual short-horizon risk scoring with
  stability-margin improvement.
* `closed-loop-benchmark` carries simulated displacement, velocity, and accepted
  controller actions between decision steps.
* `benchmark` and `closed-loop-benchmark` now emit Pareto artifacts for
  risk-vs-productivity comparison.

Benchmark:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 2 \
  --steps 8 \
  --decision-interval 0.10 \
  --out results/closed_loop_benchmark_hybrid_demo
```

Selected results:

| Scenario | Controller | Mean Risk | Final Risk | Severe Fraction | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| near_boundary | fixed | 0.743 | 0.764 | 0.000 | 1.000 | -19.859 | no |
| near_boundary | cf | 0.533 | 0.497 | 0.000 | 1.065 | -11.509 | no |
| near_boundary | hybrid | 0.533 | 0.497 | 0.000 | 1.065 | -11.509 | no |
| near_boundary | sld | 0.481 | 0.418 | 0.000 | 1.081 | -9.623 | yes |
| near_boundary | mpc | 0.487 | 0.419 | 0.000 | 1.129 | -9.415 | yes |
| onset | fixed | 0.736 | 0.729 | 0.000 | 1.000 | -19.635 | no |
| onset | cf | 0.536 | 0.498 | 0.000 | 1.057 | -11.873 | no |
| onset | hybrid | 0.497 | 0.431 | 0.000 | 1.065 | -10.251 | no |
| onset | sld | 0.418 | 0.348 | 0.000 | 1.081 | -7.352 | no |
| onset | mpc | 0.417 | 0.343 | 0.000 | 1.122 | -6.966 | yes |
| unstable | fixed | 0.705 | 0.718 | 0.000 | 1.000 | -18.947 | yes |
| unstable | hybrid | 0.798 | 0.826 | 0.562 | 1.065 | -30.852 | no |
| unstable | sld | 0.784 | 0.792 | 0.375 | 1.039 | -27.873 | yes |
| unstable | mpc | 0.788 | 0.800 | 0.500 | 1.081 | -29.546 | yes |

Artifacts:

| Check | Output |
|---|---|
| closed-loop benchmark | `results/closed_loop_benchmark_hybrid_demo` |
| episode rows | `results/closed_loop_benchmark_hybrid_demo/episodes.csv` |
| summary table | `results/closed_loop_benchmark_hybrid_demo/summary.csv` |
| Pareto table | `results/closed_loop_benchmark_hybrid_demo/pareto.csv` |
| plots | `results/closed_loop_benchmark_hybrid_demo/risk_vs_mrr.svg`, `results/closed_loop_benchmark_hybrid_demo/pareto.svg` |

Lesson: the next three implementation tasks are now done: hybrid controller,
persistent closed-loop benchmark, and Pareto-style benchmark reporting. The
results are useful but not flattering enough to hide. Hybrid improves over
fixed/rule on onset and matches `cf` on near-boundary, but it is dominated by
SLD/MPC on the tested Pareto front. On the fully unstable case, aggressive
spindle moves can make persistent dynamics worse. Before adding SAC/TD3, the
next best technical task is to calibrate the stability-margin surrogate against
the time-domain simulated risk and use that calibration to tune the controller
shield/objective.

## 2026-05-02 - Time-Domain Margin Calibration

This update adds `chatter-twin calibrate-margin`, which sweeps axial-depth and
spindle-speed scale factors, simulates each cut, extracts signal features, and
fits a logistic mapping from raw physics margin to a signal-only time-domain
chatter target. The resulting `calibration.json` can be passed to
`--margin-calibration` for rollout and benchmark commands. The calibrated margin
is used both in risk estimation and controller scoring.

Severe-target calibration:

```bash
rtk uv run chatter-twin calibrate-margin \
  --scenarios stable,near_boundary,onset,unstable \
  --duration 0.25 \
  --sensor-noise 0.0 \
  --target-threshold 0.50 \
  --out results/margin_calibration_time_domain_demo
```

Incipient-target calibration:

```bash
rtk uv run chatter-twin calibrate-margin \
  --scenarios stable,near_boundary,onset,unstable \
  --duration 0.25 \
  --sensor-noise 0.0 \
  --target-threshold 0.35 \
  --out results/margin_calibration_time_domain_incipient_demo
```

Calibration results:

| Target | Samples | Positive Fraction | ROC AUC | Brier | Raw Margin at p=0.5 | Risk at Raw Margin 0 |
|---|---:|---:|---:|---:|---:|---:|
| severe-like `0.50` | 320 | 0.078 | 0.969 | 0.068 | -4.088 | 0.016 |
| incipient `0.35` | 320 | 0.181 | 0.888 | 0.119 | -2.593 | 0.144 |

Closed-loop benchmark with the incipient calibration:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 2 \
  --steps 8 \
  --decision-interval 0.10 \
  --margin-calibration results/margin_calibration_time_domain_incipient_demo/calibration.json \
  --out results/closed_loop_benchmark_incipient_calibrated_margin_guarded_demo
```

Selected calibrated benchmark results:

| Scenario | Controller | Mean Risk | Final Risk | Severe Fraction | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| near_boundary | fixed | 0.331 | 0.352 | 0.000 | 1.000 | -3.915 | no |
| near_boundary | sld | 0.329 | 0.334 | 0.000 | 1.081 | -3.878 | yes |
| near_boundary | mpc | 0.334 | 0.335 | 0.000 | 1.129 | -3.698 | yes |
| onset | fixed | 0.325 | 0.318 | 0.000 | 1.000 | -3.702 | no |
| onset | sld | 0.265 | 0.263 | 0.000 | 1.081 | -1.602 | no |
| onset | mpc | 0.263 | 0.259 | 0.000 | 1.129 | -1.157 | yes |
| unstable | fixed | 0.445 | 0.458 | 0.000 | 1.000 | -8.472 | no |
| unstable | hybrid | 0.451 | 0.477 | 0.000 | 1.049 | -8.415 | no |
| unstable | sld | 0.411 | 0.396 | 0.000 | 1.081 | -6.871 | yes |
| unstable | mpc | 0.411 | 0.396 | 0.000 | 1.129 | -6.520 | yes |

Artifacts:

| Check | Output |
|---|---|
| severe calibration | `results/margin_calibration_time_domain_demo` |
| incipient calibration | `results/margin_calibration_time_domain_incipient_demo` |
| calibrated benchmark | `results/closed_loop_benchmark_incipient_calibrated_margin_guarded_demo` |

Lesson: the previous raw margin scale was much too pessimistic for the
time-domain simulator. A raw margin of zero is not close to a 50% simulated
signal-risk boundary; for the incipient target, that boundary is around raw
margin `-2.59`. After calibration, the controllers stop producing severe
labels from physics-margin pessimism alone. SLD/MPC still form the strongest
Pareto front, while hybrid remains useful mainly as a guarded experimental
controller. The next step is not GPU RL; it is a richer calibration set with
domain randomization and held-out tool/material/modal families, then retuning
MPC/hybrid against those calibrated margins.

## 2026-05-02 - Random-Family Margin Holdout

This update extends `calibrate-margin` with synthetic process families. Each
family perturbs modal stiffness/damping, cutting coefficients, radial depth,
feed, and runout. The calibration can now fit on train families and report a
held-out family, which is closer to the real question: does the stability-margin
surrogate generalize when the tool/setup/material changes?

Calibration:

```bash
rtk uv run chatter-twin calibrate-margin \
  --scenarios stable,near_boundary,onset,unstable \
  --axial-depth-scales 0.35,0.60,0.85,1.10,1.35,1.60,2.00,2.50,3.00 \
  --spindle-scales 0.88,0.94,1.00,1.06,1.12 \
  --duration 0.20 \
  --sensor-noise 0.0 \
  --target-threshold 0.35 \
  --family-count 5 \
  --holdout-family 4 \
  --out results/margin_calibration_random_family_holdout_demo
```

Result:

| Split | Samples | Positives | Positive Fraction | ROC AUC | Brier | Accuracy | MAE to Time Risk |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 900 | 181 | 0.201 | 0.743 | 0.187 | 0.764 | 0.155 |
| train | 720 | 148 | 0.206 | 0.770 | 0.181 | 0.779 | 0.151 |
| holdout | 180 | 33 | 0.183 | 0.623 | 0.211 | 0.706 | 0.170 |

The fitted boundary is similar to the single-family incipient calibration:
raw margin at `p=0.5` is `-2.598`, and risk at raw margin zero is `0.293`.

Closed-loop benchmark with the random-family calibration:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 2 \
  --steps 8 \
  --decision-interval 0.10 \
  --margin-calibration results/margin_calibration_random_family_holdout_demo/calibration.json \
  --out results/closed_loop_benchmark_random_family_calibrated_demo
```

Selected benchmark results:

| Scenario | Controller | Mean Risk | Final Risk | Severe Fraction | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| onset | fixed | 0.400 | 0.392 | 0.000 | 1.000 | -6.384 | no |
| onset | sld | 0.347 | 0.347 | 0.000 | 1.081 | -4.553 | no |
| onset | mpc | 0.345 | 0.343 | 0.000 | 1.129 | -4.107 | yes |
| unstable | fixed | 0.437 | 0.450 | 0.000 | 1.000 | -8.145 | yes |
| unstable | hybrid | 0.480 | 0.496 | 0.000 | 1.065 | -9.372 | no |
| unstable | sld | 0.460 | 0.458 | 0.000 | 1.081 | -8.674 | yes |
| unstable | mpc | 0.461 | 0.458 | 0.000 | 1.129 | -8.321 | yes |

Artifacts:

| Check | Output |
|---|---|
| randomized family calibration | `results/margin_calibration_random_family_holdout_demo` |
| calibrated benchmark | `results/closed_loop_benchmark_random_family_calibrated_demo` |

Lesson: this is the first calibration result that looks like a real research
warning. The train-family fit is acceptable for a one-feature surrogate, but
holdout-family AUC drops to `0.623`, so raw margin alone is not robust enough
under setup/material drift. The next engineering step is to make calibration
multi-feature and uncertainty-aware: raw margin plus modal/cutting/process
context should predict the time-domain target, and high holdout uncertainty
should make the safety shield more conservative.

## 2026-05-02 - Context-Aware Margin Calibration

This update adds `--calibration-model context`. The model still uses the raw
physics margin, but it also includes process context available at runtime:
spindle speed, tooth frequency, axial/radial depth, feed per tooth, cutting
coefficients, modal natural frequency, modal damping ratio, stiffness ratio,
radial immersion, and runout-to-feed ratio. The saved calibration artifact now
also carries holdout Brier score into a calibration uncertainty estimate. That
uncertainty is merged into `RiskEstimate.uncertainty`, so the existing safety
shield can reject controller actions when the calibration model is near its
boundary or has weak holdout reliability.

Calibration:

```bash
rtk uv run chatter-twin calibrate-margin \
  --calibration-model context \
  --scenarios stable,near_boundary,onset,unstable \
  --axial-depth-scales 0.35,0.60,0.85,1.10,1.35,1.60,2.00,2.50,3.00 \
  --spindle-scales 0.88,0.94,1.00,1.06,1.12 \
  --duration 0.20 \
  --sensor-noise 0.0 \
  --target-threshold 0.35 \
  --family-count 5 \
  --holdout-family 4 \
  --out results/margin_calibration_context_family_holdout_demo
```

Raw-margin vs context calibration:

| Model | Train ROC AUC | Holdout ROC AUC | Train Brier | Holdout Brier | Risk at Raw Margin 0 |
|---|---:|---:|---:|---:|---:|
| raw margin | 0.770 | 0.623 | 0.181 | 0.211 | 0.293 |
| context | 0.829 | 0.764 | 0.166 | 0.177 | 0.240 |

Context-calibrated closed-loop benchmark:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 2 \
  --steps 8 \
  --decision-interval 0.10 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/closed_loop_benchmark_context_calibrated_demo
```

Selected benchmark results:

| Scenario | Controller | Mean Risk | Final Risk | Shield Rejects | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| onset | fixed | 0.408 | 0.401 | 0 | 1.000 | -6.702 | yes |
| onset | sld | 0.375 | 0.370 | 0 | 0.919 | -5.533 | yes |
| onset | mpc | 0.410 | 0.402 | 0 | 1.044 | -6.391 | yes |
| stable | fixed | 0.380 | 0.378 | 0 | 1.000 | -5.681 | yes |
| stable | sld | 0.348 | 0.356 | 0 | 0.919 | -4.582 | yes |
| stable | mpc | 0.352 | 0.359 | 0 | 0.959 | -4.334 | yes |
| unstable | fixed | 0.451 | 0.463 | 0 | 1.000 | -8.685 | yes |
| unstable | sld | 0.513 | 0.520 | 0 | 1.081 | -10.597 | yes |
| unstable | mpc | 0.517 | 0.524 | 12 | 1.129 | -14.869 | yes |
| unstable | hybrid | 0.526 | 0.549 | 12 | 1.065 | -15.532 | no |

Artifacts:

| Check | Output |
|---|---|
| context calibration | `results/margin_calibration_context_family_holdout_demo` |
| context-calibrated benchmark | `results/closed_loop_benchmark_context_calibrated_demo` |

Lesson: adding context materially improves held-out calibration quality:
holdout AUC rises from `0.623` to `0.764`, and holdout Brier improves from
`0.211` to `0.177`. That is the first credible uncertainty-aware margin twin
step. The controller result is also a warning: once uncertainty is tied to the
shield, aggressive controllers are rejected or penalized in unstable cuts. That
is acceptable behavior before hardware, but the controller objective now needs
to optimize under calibration uncertainty instead of treating shield rejection
as an afterthought.

## 2026-05-02 - Uncertainty-Aware Controller Scoring

This update retunes the planning controllers to reason about calibration
uncertainty before proposing an action. `mpc` now scores candidate
feed/spindle pairs with both calibration uncertainty and boundary proximity
uncertainty, and it avoids non-current candidates above the shield's uncertainty
limit. `cf` and `hybrid` now predict full `RiskEstimate` objects for candidate
actions, merge calibration uncertainty into those estimates, and skip
high-uncertainty moves before the shield has to reject them.

Benchmark:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 2 \
  --steps 8 \
  --decision-interval 0.10 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/closed_loop_benchmark_context_calibrated_uncertainty_aware_demo
```

Selected benchmark results:

| Scenario | Controller | Mean Risk | Final Risk | Shield Rejects | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| onset | fixed | 0.408 | 0.401 | 0 | 1.000 | -6.702 | yes |
| onset | sld | 0.375 | 0.370 | 0 | 0.919 | -5.533 | yes |
| onset | mpc | 0.410 | 0.402 | 0 | 1.044 | -6.391 | yes |
| stable | fixed | 0.380 | 0.378 | 0 | 1.000 | -5.681 | yes |
| stable | sld | 0.348 | 0.356 | 0 | 0.919 | -4.582 | yes |
| stable | mpc | 0.352 | 0.359 | 0 | 0.959 | -4.334 | yes |
| unstable | fixed | 0.451 | 0.463 | 0 | 1.000 | -8.685 | yes |
| unstable | sld | 0.513 | 0.520 | 0 | 1.081 | -10.597 | yes |
| unstable | mpc | 0.514 | 0.522 | 0 | 1.108 | -10.440 | yes |
| unstable | cf | 0.517 | 0.529 | 0 | 1.035 | -10.682 | no |
| unstable | hybrid | 0.517 | 0.529 | 0 | 1.035 | -10.682 | no |
| unstable | rule | 0.515 | 0.525 | 10 | 1.045 | -14.385 | no |

Artifacts:

| Check | Output |
|---|---|
| uncertainty-aware benchmark | `results/closed_loop_benchmark_context_calibrated_uncertainty_aware_demo` |
| summary table | `results/closed_loop_benchmark_context_calibrated_uncertainty_aware_demo/summary.csv` |
| Pareto table | `results/closed_loop_benchmark_context_calibrated_uncertainty_aware_demo/pareto.csv` |

Lesson: the controller-side uncertainty fix removes the shield-rejection failure
for `mpc`, `cf`, and `hybrid` in the context-calibrated unstable cut. Before the
retune, `mpc` and `hybrid` each produced 12 rejected actions on the same small
demo; after the retune they produce zero. The simple `rule` baseline still
produces 10 rejected actions, which is useful as a sanity check that the shield
is still active. The remaining research question is controller quality, not
safety plumbing: `fixed` has the lowest unstable-cut mean risk in this tiny
context-calibrated demo, while `sld`/`mpc` preserve more MRR. The next best
step is to train/evaluate SAC only after this uncertainty-aware shielded
benchmark remains stable on a larger randomized episode set.

## 2026-05-02 - Randomized Closed-Loop Stress Test

This update adds domain randomization directly to `benchmark` and
`closed-loop-benchmark`. Episode rows now record the sampled spindle, feed,
depth, modal, cutting-coefficient, and noise scales, so controller performance
can be audited against the synthetic setup instead of only the scenario name.
The `sld` controller was also made rate-limit-aware: it now scores only
one-step-reachable spindle candidates, so it does not ask for a far stability
lobe target that the shield will partially rate-limit into a high-uncertainty
intermediate state.

Stress benchmark:

```bash
rtk uv run chatter-twin closed-loop-benchmark \
  --controllers fixed,rule,sld,mpc,cf,hybrid \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 8 \
  --steps 12 \
  --decision-interval 0.10 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/closed_loop_benchmark_context_calibrated_randomized_stress_demo
```

Selected randomized stress results:

| Scenario | Controller | Mean Risk | Final Risk | Shield Rejects | Relative MRR Proxy | Mean Reward | Pareto Efficient |
|---|---|---:|---:|---:|---:|---:|---|
| stable | fixed | 0.393 | 0.402 | 0 | 1.000 | -9.248 | yes |
| stable | sld | 0.342 | 0.338 | 0 | 0.913 | -6.529 | yes |
| stable | mpc | 0.352 | 0.352 | 0 | 0.971 | -6.464 | yes |
| near_boundary | fixed | 0.440 | 0.444 | 0 | 1.000 | -11.976 | yes |
| near_boundary | sld | 0.394 | 0.400 | 0 | 0.930 | -9.594 | yes |
| near_boundary | hybrid | 0.429 | 0.430 | 0 | 0.991 | -11.447 | yes |
| onset | fixed | 0.446 | 0.446 | 0 | 1.000 | -12.544 | no |
| onset | sld | 0.414 | 0.409 | 0 | 0.948 | -10.760 | yes |
| onset | mpc | 0.423 | 0.423 | 0 | 1.039 | -10.661 | yes |
| unstable | fixed | 0.543 | 0.554 | 0 | 1.000 | -18.030 | no |
| unstable | sld | 0.510 | 0.514 | 0 | 0.987 | -16.221 | yes |
| unstable | hybrid | 0.519 | 0.522 | 0 | 1.013 | -16.747 | yes |
| unstable | mpc | 0.532 | 0.538 | 0 | 1.099 | -16.637 | yes |
| unstable | rule | 0.542 | 0.552 | 29 | 1.027 | -20.977 | no |

Artifacts:

| Check | Output |
|---|---|
| randomized stress benchmark | `results/closed_loop_benchmark_context_calibrated_randomized_stress_demo` |
| episode rows with scale metadata | `results/closed_loop_benchmark_context_calibrated_randomized_stress_demo/episodes.csv` |
| summary table | `results/closed_loop_benchmark_context_calibrated_randomized_stress_demo/summary.csv` |
| Pareto table | `results/closed_loop_benchmark_context_calibrated_randomized_stress_demo/pareto.csv` |

Lesson: under 192 randomized closed-loop episodes, `sld`, `mpc`, `cf`, and
`hybrid` all finish with zero shield rejections. The only remaining rejected
actions come from the intentionally naive `rule` baseline: 8 in `stable` and
29 in `unstable`. In this stress test, `sld` is the most attractive low-risk
non-RL baseline, while `mpc` gives the strongest MRR proxy in onset/unstable
cases. This is the right point to start SAC/TD3 as controller candidates, but
they should be compared against `sld` and `mpc` under this exact randomized,
calibrated, shielded benchmark.

## 2026-05-02 - CPU SAC/TD3 Controller Smoke

This update adds `chatter-twin train-rl` with three algorithms:

* `q_learning`: a tiny tabular CPU baseline for validating the artifact path.
* `sac`: Stable-Baselines3 SAC on CPU.
* `td3`: Stable-Baselines3 TD3 on CPU.

SAC/TD3 are optional via `--extra rl`. `torch` is pinned to the PyTorch CPU
wheel index in `pyproject.toml`; the lockfile contains no CUDA/NVIDIA packages.
Learned actions pass through a candidate guard before execution: the twin
evaluates the proposed feed/spindle pair and falls back to the current override
if the candidate would move into high calibration uncertainty.

SAC smoke:

```bash
rtk uv run --extra rl chatter-twin train-rl \
  --algorithm sac \
  --scenarios stable,unstable \
  --total-timesteps 64 \
  --eval-episodes 1 \
  --steps 4 \
  --decision-interval 0.05 \
  --learning-starts 8 \
  --batch-size 8 \
  --buffer-size 512 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_sac_cpu_smoke_demo
```

TD3 smoke:

```bash
rtk uv run --extra rl chatter-twin train-rl \
  --algorithm td3 \
  --scenarios stable,unstable \
  --total-timesteps 64 \
  --eval-episodes 1 \
  --steps 4 \
  --decision-interval 0.05 \
  --learning-starts 8 \
  --batch-size 8 \
  --buffer-size 512 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_cpu_smoke_demo
```

Selected smoke results:

| Controller | Scenario | Mean Risk | Final Risk | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---:|---:|---:|---:|---:|
| sac | stable | 0.338 | 0.358 | 0 | 0.999 | -1.994 |
| sac | unstable | 0.410 | 0.419 | 0 | 0.939 | -4.145 |
| td3 | stable | 0.330 | 0.360 | 0 | 1.078 | -1.936 |
| td3 | unstable | 0.528 | 0.542 | 0 | 1.137 | -6.102 |
| mpc | unstable | 0.484 | 0.471 | 0 | 1.018 | -4.693 |
| sld | unstable | 0.521 | 0.553 | 0 | 0.966 | -5.529 |

Artifacts:

| Check | Output |
|---|---|
| SAC CPU smoke | `results/rl_sac_cpu_smoke_demo` |
| TD3 CPU smoke | `results/rl_td3_cpu_smoke_demo` |
| SAC/TD3 guide | `docs/RL_CONTROLLERS.md` |

Lesson: SAC and TD3 are now real shielded controller candidates in the repo:
they train, checkpoint, evaluate, and emit matched `sld`/`mpc` comparisons on
CPU. These smoke runs are intentionally tiny, so they should not be interpreted
as final policy quality. SAC looks promising in the short unstable smoke, TD3
is more aggressive on MRR and worse on unstable risk, and `mpc` remains the
baseline to beat. The next non-GPU step is longer CPU sweeps and multi-seed
reports before moving to GPU-scale training.

## 2026-05-02 - CPU SAC/TD3 Randomized Sweeps

This follow-up runs modest 1,000-timestep CPU sweeps for SAC and TD3 over all
four scenarios with domain randomization and context-calibrated margins. These
are still small runs, but they exercise the complete learned-controller loop:
training, candidate guard, checkpointing, evaluation, and matched `sld`/`mpc`
baseline comparison.

SAC:

```bash
rtk uv run --extra rl chatter-twin train-rl \
  --algorithm sac \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_sac_cpu_randomized_demo
```

TD3:

```bash
rtk uv run --extra rl chatter-twin train-rl \
  --algorithm td3 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_cpu_randomized_demo
```

Selected randomized CPU sweep results:

| Controller | Scenario | Mean Risk | Final Risk | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---:|---:|---:|---:|---:|
| sac | stable | 0.316 | 0.323 | 0 | 0.917 | -4.073 |
| sac | near_boundary | 0.383 | 0.384 | 0 | 0.934 | -7.166 |
| sac | onset | 0.369 | 0.372 | 0 | 0.915 | -6.243 |
| sac | unstable | 0.390 | 0.398 | 0 | 0.932 | -7.630 |
| td3 | stable | 0.288 | 0.281 | 0 | 0.952 | -2.033 |
| td3 | near_boundary | 0.382 | 0.374 | 0 | 0.952 | -6.023 |
| td3 | onset | 0.370 | 0.367 | 0 | 0.952 | -5.253 |
| td3 | unstable | 0.442 | 0.441 | 0 | 0.952 | -8.412 |
| mpc | unstable | 0.464 | 0.493 | 0 | 1.041 | -8.686 |
| sld | unstable | 0.460 | 0.483 | 0 | 0.998 | -8.916 |

Artifacts:

| Check | Output |
|---|---|
| SAC randomized CPU sweep | `results/rl_sac_cpu_randomized_demo` |
| TD3 randomized CPU sweep | `results/rl_td3_cpu_randomized_demo` |
| RL controller guide | `docs/RL_CONTROLLERS.md` |

Lesson: this is the first useful learned-controller result. Both SAC and TD3
stay inside the candidate guard with zero shield rejections. SAC is conservative
on feed and gives low unstable risk at the cost of MRR. TD3 converges to a
high-feed/low-spindle posture with better reward in stable/near-boundary/onset,
but higher unstable risk than SAC. The next non-GPU step would be multi-seed
CPU sweeps and reward/guard ablations; GPU only becomes necessary for larger
timesteps or repeated sweeps.

## 2026-05-02 - Multi-Seed RL Harness Smoke

This update adds `chatter-twin train-rl-multiseed`, a first-class runner for
repeating shielded RL controller training over several seeds and aggregating
the learned policies against matched `sld`/`mpc` baselines. It writes each
per-seed run under `runs/<algorithm>_seed_<seed>/` plus top-level
`run_summary.csv`, `comparison_summary.csv`, `aggregate_summary.csv`,
`metrics.json`, and `report.md`.

Smoke command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,unstable \
  --total-timesteps 64 \
  --eval-episodes 1 \
  --steps 4 \
  --decision-interval 0.05 \
  --learning-starts 8 \
  --batch-size 8 \
  --buffer-size 512 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_smoke_demo
```

Selected aggregate results:

| Training algo | Controller | Scenario | Seeds | Mean Risk | Risk Std | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sac | sac | stable | 2 | 0.291 | 0.066 | 0 | 0.965 | -1.426 |
| sac | sac | unstable | 2 | 0.502 | 0.130 | 0 | 1.028 | -5.822 |
| td3 | td3 | stable | 2 | 0.256 | 0.105 | 0 | 0.958 | -0.876 |
| td3 | td3 | unstable | 2 | 0.546 | 0.026 | 0 | 1.048 | -6.443 |
| sac | mpc | unstable | 2 | 0.539 | 0.077 | 0 | 1.033 | -5.797 |
| sac | sld | unstable | 2 | 0.557 | 0.052 | 0 | 0.985 | -6.292 |

Artifacts:

| Check | Output |
|---|---|
| Multi-seed smoke | `results/rl_multiseed_cpu_smoke_demo` |
| Aggregate table | `results/rl_multiseed_cpu_smoke_demo/aggregate_summary.csv` |
| Multi-seed report | `results/rl_multiseed_cpu_smoke_demo/report.md` |

Lesson: SAC/TD3 comparison is no longer limited to one manual seed. The smoke
is deliberately small, but the repo now has the harness needed for larger CPU
or GPU-scale multi-seed studies with the same shielded evaluation contract.

## 2026-05-02 - Short Randomized Multi-Seed RL Run

I attempted a larger three-seed, 1,000-timestep, all-scenario SAC/TD3 run on
CPU. It completed the SAC seeds but was stopped before TD3 because the
simulator-backed training loop was too slow for an interactive turn. The partial
folder is marked with
`results/rl_multiseed_cpu_randomized_demo/PARTIAL_ABORTED_README.md`.

Balanced CPU command completed:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_short_randomized_demo
```

Selected aggregate results:

| Training algo | Controller | Scenario | Seeds | Mean Risk | Risk Std | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sac | sac | stable | 2 | 0.309 | 0.041 | 0 | 0.937 | -2.838 |
| sac | sac | near_boundary | 2 | 0.357 | 0.034 | 0 | 0.959 | -4.360 |
| sac | sac | onset | 2 | 0.422 | 0.058 | 0 | 0.951 | -6.025 |
| sac | sac | unstable | 2 | 0.400 | 0.158 | 0 | 0.897 | -5.988 |
| td3 | td3 | stable | 2 | 0.279 | 0.029 | 0 | 0.864 | -1.939 |
| td3 | td3 | near_boundary | 2 | 0.358 | 0.069 | 0 | 0.868 | -4.275 |
| td3 | td3 | onset | 2 | 0.385 | 0.014 | 0 | 0.876 | -5.125 |
| td3 | td3 | unstable | 2 | 0.452 | 0.149 | 0 | 0.869 | -7.173 |
| sac | mpc | unstable | 2 | 0.479 | 0.130 | 0 | 1.081 | -6.994 |
| sac | sld | unstable | 2 | 0.477 | 0.135 | 0 | 1.039 | -7.197 |

Artifacts:

| Check | Output |
|---|---|
| Short randomized multi-seed run | `results/rl_multiseed_cpu_short_randomized_demo` |
| Aggregate table | `results/rl_multiseed_cpu_short_randomized_demo/aggregate_summary.csv` |
| Report | `results/rl_multiseed_cpu_short_randomized_demo/report.md` |

Lesson: SAC currently produces the lowest unstable-risk learned behavior in
this short run, but it sacrifices productivity and regresses on onset. TD3 has
better stable/onset reward but also lowers MRR and does not beat SAC on
unstable risk. The next best non-GPU step is reward/constraint ablation:
separate the risk penalty, MRR penalty, action smoothness, and candidate guard
so we can tell whether the learned policies are genuinely controlling chatter
or simply choosing conservative override postures.

## 2026-05-02 - Reward Config and MRR Ablations

This update centralizes the control reward into a shared `RewardConfig` used by
the Gym environment, RL training/evaluation, and matched closed-loop baselines.
The RL commands now expose:

* `--productivity-mode feed|mrr`
* `--productivity-weight`
* `--risk-now-weight`
* `--risk-horizon-weight`
* `--severe-penalty`
* `--smoothness-weight`
* `--rejection-penalty`

The default reward remains behavior-compatible with the previous formula:

```text
feed_productivity
- 3.0 * current_risk
- 1.5 * horizon_risk
- severe_penalty
- 0.5 * override_motion
- shield_rejection_penalty
```

MRR proxy ablation:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --productivity-mode mrr \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_mrr_reward_ablation_demo
```

Stronger MRR-weight ablation:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_mrr2_reward_ablation_demo
```

Selected learned-policy comparison:

| Profile | Controller | Scenario | Mean Risk | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---|---:|---:|---:|---:|
| feed default | sac | stable | 0.309 | 0 | 0.937 | -2.838 |
| feed default | sac | unstable | 0.400 | 0 | 0.897 | -5.988 |
| feed default | td3 | stable | 0.279 | 0 | 0.864 | -1.939 |
| feed default | td3 | unstable | 0.452 | 0 | 0.869 | -7.173 |
| mrr weight 1 | sac | stable | 0.302 | 0 | 0.908 | -2.805 |
| mrr weight 1 | sac | unstable | 0.420 | 0 | 0.918 | -6.453 |
| mrr weight 1 | td3 | stable | 0.279 | 0 | 0.865 | -2.451 |
| mrr weight 1 | td3 | unstable | 0.458 | 0 | 0.864 | -7.827 |
| mrr weight 2 | sac | stable | 0.300 | 0 | 0.905 | 2.671 |
| mrr weight 2 | sac | unstable | 0.419 | 0 | 0.845 | -1.833 |
| mrr weight 2 | td3 | stable | 0.284 | 0 | 0.954 | 3.688 |
| mrr weight 2 | td3 | unstable | 0.465 | 0 | 0.954 | -1.774 |

Artifacts:

| Check | Output |
|---|---|
| MRR reward ablation | `results/rl_multiseed_cpu_mrr_reward_ablation_demo` |
| MRR x2 reward ablation | `results/rl_multiseed_cpu_mrr2_reward_ablation_demo` |
| Reward config docs | `docs/RL_CONTROLLERS.md` |

Lesson: changing productivity from feed to MRR is not enough. A stronger MRR
weight recovers TD3 productivity, but unstable risk worsens relative to SAC and
still does not beat the safety/productivity tradeoff of the baselines. This
suggests the next ablation should target horizon-risk and smoothness terms,
especially because onset behavior remains weak.

## 2026-05-02 - Horizon-Risk and Smoothness Ablations

These ablations use the stronger MRR reward profile from
`results/rl_multiseed_cpu_mrr2_reward_ablation_demo` as the base:

* `results/rl_multiseed_cpu_mrr2_horizon3_ablation_demo`: increase
  `--risk-horizon-weight` from `1.5` to `3.0`.
* `results/rl_multiseed_cpu_mrr2_smooth01_ablation_demo`: reduce
  `--smoothness-weight` from `0.5` to `0.1`.

Horizon-risk command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --risk-horizon-weight 3.0 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_mrr2_horizon3_ablation_demo
```

Smoothness command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_multiseed_cpu_mrr2_smooth01_ablation_demo
```

Selected learned-policy results:

| Profile | Controller | Scenario | Mean Risk | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---|---:|---:|---:|---:|
| mrr weight 2 | sac | onset | 0.406 | 0 | 0.895 | -0.633 |
| mrr weight 2 | sac | unstable | 0.419 | 0 | 0.845 | -1.833 |
| mrr weight 2 | td3 | onset | 0.404 | 0 | 0.954 | 0.182 |
| mrr weight 2 | td3 | unstable | 0.465 | 0 | 0.954 | -1.774 |
| horizon weight 3 | sac | onset | 0.423 | 0 | 0.915 | -4.868 |
| horizon weight 3 | sac | unstable | 0.421 | 0 | 0.866 | -5.963 |
| horizon weight 3 | td3 | onset | 0.447 | 0 | 1.032 | -4.655 |
| horizon weight 3 | td3 | unstable | 0.451 | 0 | 0.984 | -5.775 |
| smoothness 0.1 | sac | onset | 0.416 | 0 | 0.932 | -0.353 |
| smoothness 0.1 | sac | unstable | 0.412 | 0 | 0.865 | -1.310 |
| smoothness 0.1 | td3 | onset | 0.403 | 0 | 0.954 | 0.244 |
| smoothness 0.1 | td3 | unstable | 0.465 | 0 | 0.954 | -1.714 |

One important caveat: the smoothness run produced SAC shield rejections on
near-boundary cuts, even though the selected onset/unstable rows above show no
rejections. That suggests lower motion penalty can make exploration/action
selection less shield-compatible.

Lesson: scalar reward changes alone are not solving the policy quality issue.
Higher horizon penalty does not improve onset risk, and lower smoothness is not
the main productivity bottleneck. The next best non-GPU step is action/guard
diagnostics: record raw proposed actions, guard fallbacks, and shield reasons
so we can see whether the learned policies are being constrained by action
space, candidate guard, or true simulator risk.

## 2026-05-02 - RL Action and Guard Diagnostics

This update adds action-level diagnostics to SAC/TD3 training artifacts.

New per-run artifacts:

* `evaluation_actions.csv`: one row per learned-policy evaluation step,
  including raw action, current risk/uncertainty, candidate-guard action,
  candidate risk/uncertainty, final shielded action, shield reasons, and reward.
* `action_diagnostics_summary.csv`: scenario-level summary for each run.

New top-level multi-seed artifacts:

* `action_diagnostics_summary.csv`: all per-seed diagnostic summaries.
* `action_diagnostics_aggregate.csv`: aggregate diagnostics over seeds.

Diagnostic command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_action_diagnostics_mrr2_smooth01_demo
```

Selected aggregate diagnostics:

| Controller | Scenario | Steps | Guard Fallbacks | Shield Rejects | Current Unc | Candidate Unc | Raw Feed | Shield Feed | Raw Spindle | Shield Spindle | Guard Reasons | Shield Reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| sac | near_boundary | 24 | 2 | 5 | 0.585 | 0.583 | 0.915 | 0.930 | 1.002 | 1.002 | high_uncertainty:2 | feed_rate_limited:4;high_uncertainty:5 |
| sac | onset | 24 | 6 | 0 | 0.549 | 0.555 | 0.900 | 0.938 | 0.992 | 0.994 | high_uncertainty:6 | feed_rate_limited:4 |
| sac | unstable | 24 | 0 | 0 | 0.537 | 0.537 | 0.865 | 0.889 | 0.970 | 0.972 | spindle_clipped:6 | feed_rate_limited:8;spindle_clipped:6;spindle_rate_limited:1 |
| td3 | near_boundary | 24 | 0 | 0 | 0.574 | 0.571 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:18 | spindle_clipped:18;spindle_rate_limited:4 |
| td3 | onset | 24 | 0 | 0 | 0.517 | 0.505 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:12 | spindle_clipped:12;spindle_rate_limited:4 |
| td3 | unstable | 24 | 0 | 0 | 0.543 | 0.544 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:24 | spindle_clipped:24;spindle_rate_limited:4 |

Artifacts:

| Check | Output |
|---|---|
| Action diagnostics run | `results/rl_action_diagnostics_mrr2_smooth01_demo` |
| Aggregate diagnostics | `results/rl_action_diagnostics_mrr2_smooth01_demo/action_diagnostics_aggregate.csv` |
| Per-step traces | `results/rl_action_diagnostics_mrr2_smooth01_demo/runs/*/evaluation_actions.csv` |

Lesson: SAC's near-boundary shield rejections are not primarily caused by raw
actions outside the shield bounds. The candidate guard falls back only twice,
while the final shield rejects five steps due to high uncertainty around the
current/candidate operating point. TD3 avoids guard fallbacks and shield
rejections, but it does so by saturating to max feed and min spindle; the
shield then repeatedly clips/rate-limits spindle. The next best non-GPU design
change is to move from absolute override actions to delta/rate-aware actions,
or to make the candidate guard explicitly optimize over shield-reachable
actions before the policy is evaluated.

## 2026-05-02 - Delta Action Mode For SAC/TD3

This update adds `--action-mode delta` for Stable-Baselines3 policies. In
default `absolute` mode, the policy emits feed/spindle override targets. In
`delta` mode, the two action components are interpreted as normalized changes
from the current override and scaled by the shield's feed/spindle rate limits.
The resulting absolute action still passes through the candidate guard and
final safety shield.

Delta-action command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac,td3 \
  --seeds 616,717 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 256 \
  --eval-episodes 2 \
  --steps 6 \
  --decision-interval 0.08 \
  --learning-starts 32 \
  --batch-size 16 \
  --buffer-size 4096 \
  --randomize \
  --action-mode delta \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_delta_action_mrr2_smooth01_demo
```

Selected comparison with the absolute-action diagnostic profile:

| Action Mode | Controller | Scenario | Mean Risk | Shield Rejects | Guard Fallbacks | Relative MRR Proxy | Mean Reward |
|---|---|---|---:|---:|---:|---:|---:|
| absolute | sac | near_boundary | 0.350 | 5 | 2 | 0.931 | 0.424 |
| delta | sac | near_boundary | 0.369 | 0 | 0 | 0.990 | 1.522 |
| absolute | sac | onset | 0.416 | 0 | 6 | 0.932 | -0.353 |
| delta | sac | onset | 0.424 | 0 | 6 | 1.001 | 0.254 |
| absolute | sac | unstable | 0.412 | 0 | 0 | 0.865 | -1.310 |
| delta | sac | unstable | 0.400 | 0 | 0 | 1.017 | 0.846 |
| absolute | td3 | near_boundary | 0.364 | 0 | 0 | 0.954 | 1.391 |
| delta | td3 | near_boundary | 0.476 | 0 | 0 | 1.093 | -0.291 |
| absolute | td3 | unstable | 0.465 | 0 | 0 | 0.954 | -1.714 |
| delta | td3 | unstable | 0.431 | 0 | 6 | 1.080 | 0.603 |

Artifacts:

| Check | Output |
|---|---|
| Delta-action run | `results/rl_delta_action_mrr2_smooth01_demo` |
| Aggregate results | `results/rl_delta_action_mrr2_smooth01_demo/aggregate_summary.csv` |
| Action diagnostics | `results/rl_delta_action_mrr2_smooth01_demo/action_diagnostics_aggregate.csv` |

Lesson: delta action mode is a meaningful improvement for SAC in this CPU
budget. It removes near-boundary shield rejections, recovers MRR around 1.0,
and preserves the lower unstable-risk behavior. TD3 still saturates, but the
saturation changes direction: it drives high feed and high spindle, producing
high MRR with higher risk. The next best non-GPU run should focus on SAC delta
only with longer training and more seeds; GPU is still optional until we scale
that sweep substantially.

## 2026-05-02 - Longer SAC Delta CPU Run

This run scales the most promising learned-controller profile so far: SAC with
delta/rate-aware actions, MRR productivity, low smoothness penalty, three
seeds, and 1,000 training timesteps per seed.

Command:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_sac_delta_cpu_long_randomized_demo
```

Selected aggregate results:

| Controller | Scenario | Seeds | Mean Risk | Risk Std | Shield Rejects | Relative MRR Proxy | Mean Reward |
|---|---|---:|---:|---:|---:|---:|---:|
| sac | stable | 3 | 0.326 | 0.004 | 0 | 1.003 | 4.200 |
| sac | near_boundary | 3 | 0.369 | 0.044 | 8 | 1.017 | 2.030 |
| sac | onset | 3 | 0.400 | 0.016 | 0 | 0.995 | 1.269 |
| sac | unstable | 3 | 0.477 | 0.012 | 0 | 1.028 | -1.710 |
| mpc | unstable | 3 | 0.499 | 0.033 | 0 | 1.074 | -1.150 |
| sld | unstable | 3 | 0.481 | 0.019 | 0 | 1.021 | -1.534 |

Selected action diagnostics:

| Scenario | Steps | Guard Fallbacks | Shield Rejects | Current Unc | Candidate Unc | Raw Feed | Shield Feed | Raw Spindle | Shield Spindle | Shield Reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable | 96 | 7 | 0 | 0.542 | 0.545 | 0.981 | 0.981 | 1.023 | 1.023 | spindle_clipped:1 |
| near_boundary | 96 | 8 | 8 | 0.564 | 0.564 | 0.988 | 0.986 | 1.035 | 1.031 | high_uncertainty:8;spindle_clipped:17 |
| onset | 96 | 3 | 0 | 0.539 | 0.538 | 0.980 | 0.980 | 1.017 | 1.017 | spindle_clipped:1 |
| unstable | 96 | 0 | 0 | 0.518 | 0.518 | 0.999 | 0.991 | 1.050 | 1.039 | spindle_clipped:33 |

Artifacts:

| Check | Output |
|---|---|
| Longer SAC delta run | `results/rl_sac_delta_cpu_long_randomized_demo` |
| Aggregate results | `results/rl_sac_delta_cpu_long_randomized_demo/aggregate_summary.csv` |
| Action diagnostics | `results/rl_sac_delta_cpu_long_randomized_demo/action_diagnostics_aggregate.csv` |

Lesson: SAC delta remains the best learned-controller candidate so far, but
the hard uncertainty shield is now the visible bottleneck. Near-boundary
rejections occur for SAC, SLD, and MPC in matched evaluation, so the next
non-GPU work should refine uncertainty handling rather than just train longer:
either add an advisory/strict shield distinction or make high uncertainty
produce conservative rate-limited fallback instead of a full rejection.

## 2026-05-02 - Uncertainty Hold Shield Mode

This update adds a configurable uncertainty behavior to the safety shield.
The default remains `--uncertainty-mode reject`, preserving the earlier hard
reject behavior. The new `--uncertainty-mode hold` mode keeps hard failures
strict, but treats high calibration uncertainty alone as a conservative hold:
the shield returns the previous feed/spindle override, marks the action
accepted, and records `high_uncertainty;uncertainty_hold` in the reasons.

Code changes:

* `ShieldConfig.uncertainty_mode` supports `reject`, `hold`, and `advisory`.
* `train-rl`, `train-rl-multiseed`, `benchmark`, and `closed-loop-benchmark`
  expose `--uncertainty-mode`.
* RL training, learned-policy evaluation, action diagnostics, and matched
  SLD/MPC baselines all receive the same shield config.
* Tests cover the strict default, hold-mode behavior, and hard-failure
  rejection under hold mode.

Validation:

```bash
rtk uv run --extra dev pytest tests/test_shield_env_cli.py -q
rtk uv run --extra dev pytest -q
```

Results:

* `tests/test_shield_env_cli.py`: 18 passed.
* Full suite: 79 passed.

Hold-mode CPU run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms sac \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_sac_delta_uncertainty_hold_cpu_long_demo
```

Selected aggregate results:

| Controller | Scenario | Seeds | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |
|---|---|---:|---:|---:|---:|---:|---:|
| sac | stable | 3 | 0.331 | 0.014 | 0.988 | 3.838 | 0 |
| sac | near_boundary | 3 | 0.363 | 0.036 | 0.990 | 2.309 | 0 |
| sac | onset | 3 | 0.390 | 0.017 | 0.998 | 1.680 | 0 |
| sac | unstable | 3 | 0.485 | 0.056 | 1.047 | -1.777 | 0 |
| mpc | near_boundary | 3 | 0.446 | 0.031 | 1.045 | 0.623 | 0 |
| sld | near_boundary | 3 | 0.431 | 0.040 | 0.967 | -0.258 | 0 |

Selected SAC action diagnostics:

| Scenario | Steps | Guard fallbacks | Shield rejects | Current unc | Candidate unc | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable | 96 | 8 | 0 | 0.533 | 0.535 | 0.986 | 0.986 | 1.001 | 1.002 | |
| near_boundary | 96 | 10 | 0 | 0.568 | 0.569 | 0.977 | 0.977 | 1.016 | 1.014 | high_uncertainty:8;spindle_clipped:6;uncertainty_hold:8 |
| onset | 96 | 12 | 0 | 0.541 | 0.543 | 0.991 | 0.992 | 1.006 | 1.006 | |
| unstable | 96 | 3 | 0 | 0.508 | 0.506 | 1.000 | 0.992 | 1.064 | 1.055 | feed_rate_limited:3;spindle_clipped:32 |

Artifacts:

| Check | Output |
|---|---|
| Hold-mode SAC delta run | `results/rl_sac_delta_uncertainty_hold_cpu_long_demo` |
| Aggregate results | `results/rl_sac_delta_uncertainty_hold_cpu_long_demo/aggregate_summary.csv` |
| Action diagnostics | `results/rl_sac_delta_uncertainty_hold_cpu_long_demo/action_diagnostics_aggregate.csv` |

Lesson: the uncertainty bottleneck is now handled in a more deployable way.
Hold mode removes near-boundary shield-rejection accounting failures across
SAC, SLD, and MPC while still leaving high uncertainty visible in the reason
codes. Compared with the strict longer SAC delta run, near-boundary SAC moves
from 8 shield rejections to 0, mean risk improves from `0.369` to `0.363`,
MRR drops from `1.017` to `0.990`, and reward improves from `2.030` to
`2.309`. The next useful non-GPU step is to add a dedicated strict-vs-hold
comparison report or run the same mode on TD3 to confirm whether TD3's
saturation remains the limiting behavior.

## 2026-05-02 - TD3 Hold Mode And RL Run Comparison

This step runs the same long delta/action hold-mode profile for TD3 and adds a
reusable `compare-rl-runs` command. The comparison command reads
`aggregate_summary.csv`, `action_diagnostics_aggregate.csv`, and `metrics.json`
from any number of RL result directories and writes:

* `profiles.csv`
* `combined_summary.csv`
* `learned_policy_summary.csv`
* `combined_action_diagnostics.csv`
* `delta_summary.csv`
* `metrics.json`
* `report.md`

Commands:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta_uncertainty_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run sac_strict=results/rl_sac_delta_cpu_long_randomized_demo \
  --run sac_hold=results/rl_sac_delta_uncertainty_hold_cpu_long_demo \
  --run td3_hold=results/rl_td3_delta_uncertainty_hold_cpu_long_demo \
  --baseline-label sac_strict \
  --out results/rl_strict_hold_delta_comparison_demo
```

Selected learned-policy comparison:

| Profile | Algorithm | Scenario | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |
|---|---|---|---:|---:|---:|---:|---:|
| sac_strict | sac | near_boundary | 0.369 | 0.044 | 1.017 | 2.030 | 8 |
| sac_hold | sac | near_boundary | 0.363 | 0.036 | 0.990 | 2.309 | 0 |
| td3_hold | td3 | near_boundary | 0.372 | 0.032 | 1.022 | 2.442 | 0 |
| sac_strict | sac | unstable | 0.477 | 0.012 | 1.028 | -1.710 | 0 |
| sac_hold | sac | unstable | 0.485 | 0.056 | 1.047 | -1.777 | 0 |
| td3_hold | td3 | unstable | 0.463 | 0.018 | 1.051 | -0.917 | 0 |

Selected TD3 action diagnostics:

| Scenario | Steps | Guard fallbacks | Shield rejects | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| near_boundary | 96 | 8 | 0 | 1.090 | 1.046 | 0.972 | 0.977 | high_uncertainty:8;spindle_clipped:72;spindle_rate_limited:7;uncertainty_hold:8 |
| onset | 96 | 0 | 0 | 1.094 | 1.050 | 0.957 | 0.969 | spindle_clipped:75;spindle_rate_limited:4 |
| stable | 96 | 0 | 0 | 1.094 | 1.050 | 0.960 | 0.969 | spindle_clipped:77;spindle_rate_limited:6 |
| unstable | 96 | 7 | 0 | 1.094 | 1.050 | 1.003 | 1.001 | spindle_clipped:62;spindle_rate_limited:9 |

Artifacts:

| Check | Output |
|---|---|
| TD3 hold-mode run | `results/rl_td3_delta_uncertainty_hold_cpu_long_demo` |
| Strict/hold comparison | `results/rl_strict_hold_delta_comparison_demo` |
| Comparison report | `results/rl_strict_hold_delta_comparison_demo/report.md` |

Lesson: TD3 hold mode removes uncertainty-related shield rejections and looks
stronger than SAC on unstable reward in this CPU budget, but it is still a
blunt high-feed policy. Raw TD3 feed averages about `1.09` and is clipped to
the `1.05` limit in every scenario. The useful next implementation step is to
make action bounds configurable per run or add a saturation penalty, then
rerun TD3 to see whether the lower unstable risk survives without leaning on
constant clipping.

## 2026-05-02 - TD3 Saturation Penalty Ablation

This update adds reward knobs for shield/candidate-guard adjustments:

* `RewardConfig.clip_penalty`
* `RewardConfig.rate_limit_penalty`
* CLI flags `--clip-penalty` and `--rate-limit-penalty`

The default for both is `0.0`, so all previous artifacts remain comparable.
For SAC/TD3, candidate-guard clipping is also penalized during training and
evaluation because the guard clips raw actions before the final environment
shield sees them.

Validation before the run:

```bash
rtk uv run --extra dev pytest tests/test_shield_env_cli.py tests/test_rl.py -q
```

Result: 25 passed.

TD3 penalty run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --clip-penalty 0.25 \
  --rate-limit-penalty 0.05 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta_hold_clip025_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_hold=results/rl_td3_delta_uncertainty_hold_cpu_long_demo \
  --run td3_clip025=results/rl_td3_delta_hold_clip025_cpu_long_demo \
  --baseline-label td3_hold \
  --out results/rl_td3_saturation_penalty_comparison_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Mean reward | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| td3_hold | near_boundary | 0.372 | 1.022 | 2.442 | 1.090 | 1.046 | 0.972 | 0.977 |
| td3_clip025 | near_boundary | 0.406 | 0.960 | -3.505 | 0.874 | 0.885 | 1.124 | 1.086 |
| td3_hold | unstable | 0.463 | 1.051 | -0.917 | 1.094 | 1.050 | 1.003 | 1.001 |
| td3_clip025 | unstable | 0.500 | 0.978 | -7.482 | 0.892 | 0.898 | 1.128 | 1.090 |

Artifacts:

| Check | Output |
|---|---|
| TD3 clipping-penalty run | `results/rl_td3_delta_hold_clip025_cpu_long_demo` |
| Penalty comparison | `results/rl_td3_saturation_penalty_comparison_demo` |
| Comparison report | `results/rl_td3_saturation_penalty_comparison_demo/report.md` |

Lesson: this blunt penalty is not the fix. It reduces raw feed saturation, but
TD3 compensates by dropping feed too far and pushing spindle high. Risk gets
worse in every scenario, MRR falls, and reward collapses. Keep the penalty
knobs for ablations, but the next better non-GPU direction is configurable
action bounds or a much smaller/asymmetric penalty profile.

## 2026-05-02 - TD3 Narrow Delta Action Scale

This step adds `--delta-action-scale` for SAC/TD3 delta mode. The default is
`1.0`, matching previous behavior. With `--delta-action-scale 0.5`, the
normalized policy action range is halved, so the policy can still choose
directions freely but can request smaller per-decision override moves.

Implementation notes:

* `Sb3TrainingConfig.delta_action_scale`
* `MultiSeedTrainingConfig.delta_action_scale`
* `ScenarioSamplingEnv` delta action space scaled by that value
* CLI flags for `train-rl` and `train-rl-multiseed`
* Comparison reports now include delta scale in `profiles.csv` and `report.md`

Validation before the run:

```bash
rtk uv run --extra dev pytest tests/test_rl.py -q
```

Result: 6 passed.

Run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_hold=results/rl_td3_delta_uncertainty_hold_cpu_long_demo \
  --run td3_delta05=results/rl_td3_delta05_hold_cpu_long_demo \
  --baseline-label td3_hold \
  --out results/rl_td3_delta_scale_comparison_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Mean reward | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| td3_hold | near_boundary | 0.372 | 1.022 | 2.442 | 1.090 | 1.046 | 0.972 | 0.977 |
| td3_delta05 | near_boundary | 0.373 | 1.022 | 2.451 | 1.062 | 1.043 | 0.974 | 0.980 |
| td3_hold | unstable | 0.463 | 1.051 | -0.917 | 1.094 | 1.050 | 1.003 | 1.001 |
| td3_delta05 | unstable | 0.449 | 1.019 | -0.908 | 1.066 | 1.047 | 0.969 | 0.973 |

Artifacts:

| Check | Output |
|---|---|
| TD3 narrow-delta run | `results/rl_td3_delta05_hold_cpu_long_demo` |
| Delta-scale comparison | `results/rl_td3_delta_scale_comparison_demo` |
| Comparison report | `results/rl_td3_delta_scale_comparison_demo/report.md` |

Lesson: narrowing delta actions works better than a blunt clipping penalty.
It reduces raw feed saturation, reduces spindle clipping counts, preserves
near-boundary performance, and slightly improves unstable risk. It still clips
feed often, so the next non-GPU ablation should try `--delta-action-scale 0.35`
or split the scale into separate feed and spindle factors.

## 2026-05-02 - TD3 Split Delta Scale Sweep

This step adds feed- and spindle-specific delta scales:

* `Sb3TrainingConfig.delta_feed_scale`
* `Sb3TrainingConfig.delta_spindle_scale`
* `MultiSeedTrainingConfig.delta_feed_scale`
* `MultiSeedTrainingConfig.delta_spindle_scale`
* CLI flags `--delta-feed-scale` and `--delta-spindle-scale`

Unset per-axis scales fall back to `--delta-action-scale`, preserving previous
behavior. The sweep compares global delta `0.5` against feed/spindle-specific
profiles:

* `results/rl_td3_delta_feed035_spindle05_hold_cpu_long_demo`
* `results/rl_td3_delta_feed025_spindle05_hold_cpu_long_demo`
* `results/rl_td3_delta_feed025_spindle035_hold_cpu_long_demo`
* `results/rl_td3_split_delta_scale_sweep_demo`

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Mean reward | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| td3_delta05 | unstable | 0.449 | 1.019 | -0.908 | 1.066 | 1.047 | 0.969 | 0.973 |
| td3_feed035_spindle05 | unstable | 0.452 | 0.999 | -1.310 | 1.055 | 1.044 | 0.950 | 0.957 |
| td3_feed025_spindle05 | unstable | 0.470 | 0.956 | -2.621 | 1.047 | 1.041 | 0.906 | 0.919 |
| td3_feed025_spindle035 | unstable | 0.475 | 1.057 | -1.148 | 1.047 | 1.041 | 1.017 | 1.016 |
| td3_delta05 | near_boundary | 0.373 | 1.022 | 2.451 | 1.062 | 1.043 | 0.974 | 0.980 |
| td3_feed025_spindle05 | near_boundary | 0.349 | 0.960 | 2.381 | 1.044 | 1.037 | 0.912 | 0.926 |

Lesson: global delta `0.5` remains the best TD3 tradeoff. Feed-specific
narrowing can reduce raw feed overshoot and improve stable/near-boundary risk,
but it gives up productivity and worsens unstable behavior. Spindle-specific
narrowing reduces spindle clipping counts, but it does not improve risk. The
remaining feed clipping is caused by repeated positive deltas walking into the
ceiling, so the next structural fix should make delta mapping headroom-aware
instead of only shrinking the action range.

## 2026-05-02 - TD3 Headroom Delta Mapping

Implemented `--delta-mapping headroom` for SAC/TD3 delta-action policies. The
default remains `fixed`, preserving previous behavior. Headroom mapping keeps
the same normalized action range but maps positive/negative deltas to the
remaining safe override headroom before candidate guarding, reducing requests
that immediately collide with the shield boundary.

Code and interface changes:

* `Sb3TrainingConfig.delta_mapping`
* `MultiSeedTrainingConfig.delta_mapping`
* `ScenarioSamplingEnv.delta_mapping`
* CLI flags `--delta-mapping fixed|headroom`
* `compare-rl-runs` profile output now records `delta_mapping`
* regression coverage for headroom mapping at the feed/spindle override limits

Run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --delta-mapping headroom \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_headroom_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_delta05=results/rl_td3_delta05_hold_cpu_long_demo \
  --run td3_headroom=results/rl_td3_delta05_headroom_hold_cpu_long_demo \
  --baseline-label td3_delta05 \
  --out results/rl_td3_headroom_mapping_comparison_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Mean reward | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| td3_delta05 | near_boundary | 0.373 | 1.022 | 2.451 | 1.062 | 1.043 | 0.974 | 0.980 |
| td3_headroom | near_boundary | 0.354 | 0.991 | 2.699 | 1.045 | 1.043 | 0.949 | 0.951 |
| td3_delta05 | unstable | 0.449 | 1.019 | -0.908 | 1.066 | 1.047 | 0.969 | 0.973 |
| td3_headroom | unstable | 0.462 | 0.975 | -2.044 | 1.047 | 1.047 | 0.931 | 0.931 |

Artifacts:

| Check | Output |
|---|---|
| TD3 headroom run | `results/rl_td3_delta05_headroom_hold_cpu_long_demo` |
| Headroom comparison | `results/rl_td3_headroom_mapping_comparison_demo` |
| Comparison report | `results/rl_td3_headroom_mapping_comparison_demo/report.md` |

Lesson: headroom mapping does what it is supposed to do mechanically: raw feed
requests stop overshooting the feed ceiling, and spindle clipping counts drop.
It improves near-boundary risk and reward, but the unstable scenario loses too
much MRR and reward. The current best overall TD3 CPU profile is still fixed
delta mode with `--delta-action-scale 0.5`; headroom mapping is a safer
candidate for a follow-up reward/action prior tune.

## 2026-05-02 - TD3 Headroom MRR-Weight Ablation

Ran a follow-up ablation that keeps `--delta-mapping headroom` but increases
`--productivity-weight` from `2.0` to `3.0`. The comparison report was also
updated so profile tables include productivity mode/weight and warn when reward
settings differ, because raw reward deltas are not normalized across reward
configurations.

Run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --delta-mapping headroom \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 3.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_headroom_mrr3_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_delta05=results/rl_td3_delta05_hold_cpu_long_demo \
  --run td3_headroom_mrr2=results/rl_td3_delta05_headroom_hold_cpu_long_demo \
  --run td3_headroom_mrr3=results/rl_td3_delta05_headroom_mrr3_hold_cpu_long_demo \
  --baseline-label td3_delta05 \
  --out results/rl_td3_headroom_reward_sweep_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|
| td3_delta05 | unstable | 0.449 | 1.019 | 1.066 | 1.047 | 0.969 | 0.973 |
| td3_headroom_mrr2 | unstable | 0.462 | 0.975 | 1.047 | 1.047 | 0.931 | 0.931 |
| td3_headroom_mrr3 | unstable | 0.453 | 1.036 | 1.017 | 1.015 | 1.024 | 1.022 |
| td3_delta05 | near_boundary | 0.373 | 1.022 | 1.062 | 1.043 | 0.974 | 0.980 |
| td3_headroom_mrr3 | near_boundary | 0.388 | 1.037 | 1.006 | 1.005 | 1.032 | 1.034 |

Artifacts:

| Check | Output |
|---|---|
| Headroom MRR 3 run | `results/rl_td3_delta05_headroom_mrr3_hold_cpu_long_demo` |
| Headroom reward sweep | `results/rl_td3_headroom_reward_sweep_demo` |
| Comparison report | `results/rl_td3_headroom_reward_sweep_demo/report.md` |

Lesson: the stronger productivity reward recovers MRR under headroom mapping
and avoids the low-spindle behavior from the MRR 2 headroom profile. It also
raises risk in stable/near-boundary/onset scenarios and still does not beat the
fixed delta `0.5` profile on unstable risk. Current best overall CPU TD3
candidate remains `td3_delta05`; `td3_headroom_mrr3` is a useful productivity
oriented comparator.

## 2026-05-02 - TD3 MRR 3 Fixed-vs-Headroom Mapping

Ran the fair mapping comparison at equal `--productivity-weight 3.0`. This
separates the effect of headroom mapping from the effect of changing reward
scale.

Run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 1000 \
  --eval-episodes 4 \
  --steps 8 \
  --decision-interval 0.08 \
  --learning-starts 100 \
  --batch-size 32 \
  --buffer-size 10000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 3.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_fixed_mrr3_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_fixed_mrr3=results/rl_td3_delta05_fixed_mrr3_hold_cpu_long_demo \
  --run td3_headroom_mrr3=results/rl_td3_delta05_headroom_mrr3_hold_cpu_long_demo \
  --baseline-label td3_fixed_mrr3 \
  --out results/rl_td3_mrr3_mapping_comparison_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr3 | unstable | 0.475 | 0.980 | 1.015 | 1.002 | 0.974 | 0.976 |
| td3_headroom_mrr3 | unstable | 0.453 | 1.036 | 1.017 | 1.015 | 1.024 | 1.022 |
| td3_fixed_mrr3 | near_boundary | 0.364 | 0.973 | 1.003 | 0.990 | 0.974 | 0.980 |
| td3_headroom_mrr3 | near_boundary | 0.388 | 1.037 | 1.006 | 1.005 | 1.032 | 1.034 |

Artifacts:

| Check | Output |
|---|---|
| Fixed MRR 3 run | `results/rl_td3_delta05_fixed_mrr3_hold_cpu_long_demo` |
| MRR 3 mapping comparison | `results/rl_td3_mrr3_mapping_comparison_demo` |
| Comparison report | `results/rl_td3_mrr3_mapping_comparison_demo/report.md` |

Lesson: at equal MRR weight `3.0`, headroom mapping gives better unstable
MRR and lower unstable risk, but higher risk in near-boundary/onset/stable
scenarios. Fixed MRR 2 remains the balanced CPU baseline; headroom MRR 3 is now
the most interesting productivity-oriented CPU candidate.

## 2026-05-02 - RL Candidate Summary Reports

Extended `chatter-twin compare-rl-runs` with a whole-profile
`candidate_summary.csv` and a matching report section. This avoids judging
policies one scenario row at a time when the real decision is whether a profile
is balanced enough to promote to longer training.

Candidate summary columns include:

* average and worst learned-policy risk;
* unstable and near-boundary risk;
* average, minimum, and unstable relative MRR;
* guard fallback count;
* mean raw/shielded feed and spindle overrides;
* a Pareto-frontier flag over average risk, average MRR, and shield rejects.

The latest mapping comparison was regenerated at:

| Check | Output |
|---|---|
| Candidate summary CSV | `results/rl_td3_mrr3_mapping_comparison_demo/candidate_summary.csv` |
| Regenerated report | `results/rl_td3_mrr3_mapping_comparison_demo/report.md` |

Current whole-profile readout:

| Profile | Avg risk | Worst risk | Unstable risk | Avg MRR | Min MRR | Unstable MRR | Guard fallbacks | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr3 | 0.388 | 0.475 | 0.475 | 0.991 | 0.973 | 0.980 | 18 | 1 |
| td3_headroom_mrr3 | 0.398 | 0.453 | 0.453 | 1.040 | 1.036 | 1.036 | 25 | 1 |

Lesson: both are non-dominated. `td3_fixed_mrr3` is safer on average, while
`td3_headroom_mrr3` is better for productivity and the unstable case. This
reinforces keeping two candidates for the next longer-training pass rather than
collapsing to a single winner too early.

## 2026-05-02 - TD3 Medium Validation

Promoted the two surviving CPU profiles into a medium validation run:

* `td3_fixed_mrr2`: fixed delta mapping, `--delta-action-scale 0.5`,
  `--productivity-weight 2.0`;
* `td3_headroom_mrr3`: headroom delta mapping, `--delta-action-scale 0.5`,
  `--productivity-weight 3.0`.

Both used four seeds (`616,717,818,919`), `2000` TD3 timesteps per seed,
`6` evaluation episodes per scenario, `10` control steps per episode, random
domain sampling, hold-mode uncertainty handling, and the calibrated
margin model.

Run:

```bash
rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818,919 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 2000 \
  --eval-episodes 6 \
  --steps 10 \
  --decision-interval 0.08 \
  --learning-starts 150 \
  --batch-size 32 \
  --buffer-size 20000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation

rtk uv run --extra rl chatter-twin train-rl-multiseed \
  --algorithms td3 \
  --seeds 616,717,818,919 \
  --scenarios stable,near_boundary,onset,unstable \
  --total-timesteps 2000 \
  --eval-episodes 6 \
  --steps 10 \
  --decision-interval 0.08 \
  --learning-starts 150 \
  --batch-size 32 \
  --buffer-size 20000 \
  --randomize \
  --action-mode delta \
  --delta-action-scale 0.5 \
  --delta-mapping headroom \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 3.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta05_headroom_mrr3_cpu_medium_validation

rtk uv run chatter-twin compare-rl-runs \
  --run td3_fixed_mrr2=results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation \
  --run td3_headroom_mrr3=results/rl_td3_delta05_headroom_mrr3_cpu_medium_validation \
  --baseline-label td3_fixed_mrr2 \
  --out results/rl_td3_medium_validation_comparison
```

Whole-profile readout:

| Profile | Avg risk | Worst risk | Unstable risk | Avg MRR | Min MRR | Unstable MRR | Guard fallbacks | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr2 | 0.410 | 0.490 | 0.490 | 1.070 | 1.054 | 1.054 | 31 | 1 |
| td3_headroom_mrr3 | 0.428 | 0.494 | 0.494 | 1.130 | 1.125 | 1.132 | 46 | 1 |

Scenario-level readout:

| Profile | Scenario | Mean risk | Relative MRR | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr2 | near_boundary | 0.387 | 1.071 | 1.044 | 1.028 | 1.050 | 1.042 |
| td3_headroom_mrr3 | near_boundary | 0.403 | 1.125 | 1.046 | 1.044 | 1.080 | 1.078 |
| td3_fixed_mrr2 | unstable | 0.490 | 1.054 | 1.030 | 1.014 | 1.048 | 1.040 |
| td3_headroom_mrr3 | unstable | 0.494 | 1.132 | 1.047 | 1.047 | 1.083 | 1.082 |

Artifacts:

| Check | Output |
|---|---|
| Fixed MRR 2 medium run | `results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation` |
| Headroom MRR 3 medium run | `results/rl_td3_delta05_headroom_mrr3_cpu_medium_validation` |
| Medium validation comparison | `results/rl_td3_medium_validation_comparison` |
| Candidate summary | `results/rl_td3_medium_validation_comparison/candidate_summary.csv` |
| Comparison report | `results/rl_td3_medium_validation_comparison/report.md` |

Lesson: the medium validation confirms the tradeoff rather than crowning a
winner. Fixed MRR 2 is safer on average and has fewer guard fallbacks.
Headroom MRR 3 delivers substantially higher MRR, especially unstable MRR, but
it is not lower-risk. The next best non-GPU step is a stress test under harsher
domain randomization and drift rather than another same-distribution training
run.

## 2026-05-02 - Saved-Policy Harsh Stress Evaluation

Added `chatter-twin eval-rl-run`, a saved-policy evaluator that loads SAC/TD3
`model.zip` files from a single run or multi-seed run and evaluates those
already-trained policies under a fresh scenario/randomization profile. This is
the right stress-test shape because it tests generalization instead of
retraining the policy on the stress distribution.

Smoke check:

```bash
rtk uv run --extra rl chatter-twin eval-rl-run \
  --run-dir results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation \
  --scenarios stable \
  --eval-episodes 1 \
  --steps 2 \
  --decision-interval 0.05 \
  --randomize \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_saved_eval_smoke_demo
```

Harsh stress profile:

| Parameter | Range |
|---|---|
| Spindle scale | `0.75, 1.25` |
| Feed scale | `0.55, 1.45` |
| Axial-depth scale | `0.75, 2.30` |
| Radial-depth scale | `0.55, 1.55` |
| Stiffness scale | `0.60, 1.40` |
| Damping scale | `0.45, 1.65` |
| Cutting-coefficient scale | `0.60, 1.80` |
| Noise scale | `1.00, 3.50` |

Run:

```bash
rtk uv run --extra rl chatter-twin eval-rl-run \
  --run-dir results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation \
  --scenarios stable,near_boundary,onset,unstable \
  --eval-episodes 6 \
  --steps 10 \
  --decision-interval 0.08 \
  --randomize \
  --spindle-scale 0.75 1.25 \
  --feed-scale 0.55 1.45 \
  --axial-depth-scale 0.75 2.30 \
  --radial-depth-scale 0.55 1.55 \
  --stiffness-scale 0.60 1.40 \
  --damping-scale 0.45 1.65 \
  --cutting-coeff-scale 0.60 1.80 \
  --noise-scale 1.00 3.50 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_fixed_mrr2_harsh_stress_eval

rtk uv run --extra rl chatter-twin eval-rl-run \
  --run-dir results/rl_td3_delta05_headroom_mrr3_cpu_medium_validation \
  --scenarios stable,near_boundary,onset,unstable \
  --eval-episodes 6 \
  --steps 10 \
  --decision-interval 0.08 \
  --randomize \
  --spindle-scale 0.75 1.25 \
  --feed-scale 0.55 1.45 \
  --axial-depth-scale 0.75 2.30 \
  --radial-depth-scale 0.55 1.55 \
  --stiffness-scale 0.60 1.40 \
  --damping-scale 0.45 1.65 \
  --cutting-coeff-scale 0.60 1.80 \
  --noise-scale 1.00 3.50 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_headroom_mrr3_harsh_stress_eval

rtk uv run chatter-twin compare-rl-runs \
  --run td3_fixed_mrr2_stress=results/rl_td3_fixed_mrr2_harsh_stress_eval \
  --run td3_headroom_mrr3_stress=results/rl_td3_headroom_mrr3_harsh_stress_eval \
  --baseline-label td3_fixed_mrr2_stress \
  --out results/rl_td3_harsh_stress_comparison
```

Whole-profile readout:

| Profile | Avg risk | Worst risk | Unstable risk | Avg MRR | Min MRR | Unstable MRR | Guard fallbacks | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr2_stress | 0.471 | 0.554 | 0.554 | 1.083 | 1.070 | 1.079 | 52 | 1 |
| td3_headroom_mrr3_stress | 0.483 | 0.563 | 0.563 | 1.129 | 1.116 | 1.131 | 46 | 1 |

Scenario-level readout:

| Profile | Scenario | Mean risk | Relative MRR | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|
| td3_fixed_mrr2_stress | near_boundary | 0.446 | 1.070 | 1.052 | 1.034 | 1.043 | 1.035 |
| td3_headroom_mrr3_stress | near_boundary | 0.452 | 1.116 | 1.046 | 1.042 | 1.076 | 1.071 |
| td3_fixed_mrr2_stress | unstable | 0.554 | 1.079 | 1.054 | 1.036 | 1.050 | 1.042 |
| td3_headroom_mrr3_stress | unstable | 0.563 | 1.131 | 1.047 | 1.046 | 1.082 | 1.081 |

Artifacts:

| Check | Output |
|---|---|
| Saved eval smoke | `results/rl_saved_eval_smoke_demo` |
| Fixed-policy harsh eval | `results/rl_td3_fixed_mrr2_harsh_stress_eval` |
| Headroom-policy harsh eval | `results/rl_td3_headroom_mrr3_harsh_stress_eval` |
| Harsh stress comparison | `results/rl_td3_harsh_stress_comparison` |
| Candidate summary | `results/rl_td3_harsh_stress_comparison/candidate_summary.csv` |
| Comparison report | `results/rl_td3_harsh_stress_comparison/report.md` |

Lesson: both policies degrade under harsh randomization, as expected. Fixed MRR
2 remains the safety-first candidate with lower average/worst/unstable risk.
Headroom MRR 3 remains the productivity candidate, preserving higher MRR with
slightly fewer guard fallbacks, but it is not the lower-risk option. For a
hardware-facing shadow-mode policy, promote fixed MRR 2 first and keep headroom
MRR 3 as the high-productivity comparator.

## 2026-05-02 - Shadow Champion Policy Selection

Added `chatter-twin select-rl-policy`, which ranks seed checkpoints from a
saved-policy evaluation directory and writes a shadow-only policy card. The
selection objective uses:

* average risk;
* worst scenario risk;
* unstable risk;
* average relative MRR reward;
* minimum-MRR shortfall penalty;
* guard fallback fraction;
* shield rejection penalty.

The minimum-MRR shortfall penalty prevents choosing a policy that looks safe by
slowing the process below the economic target.

Run:

```bash
rtk uv run chatter-twin select-rl-policy \
  --eval-dir results/rl_td3_fixed_mrr2_harsh_stress_eval \
  --out results/rl_td3_fixed_mrr2_shadow_champion \
  --profile-label fixed_mrr2_safety_first \
  --min-relative-mrr 1.0 \
  --mrr-shortfall-weight 3.0
```

Selected policy:

| Field | Value |
|---|---|
| Selected seed | `616` |
| Source model | `results/rl_td3_delta05_fixed_mrr2_cpu_medium_validation/runs/td3_seed_616/model.zip` |
| Selection score | `0.993` |
| Average risk | `0.476` |
| Worst scenario risk | `0.589` |
| Max window risk | `0.729` |
| Unstable risk | `0.524` |
| Near-boundary risk | `0.392` |
| Average relative MRR | `1.128` |
| Minimum relative MRR | `1.101` |
| Unstable relative MRR | `1.137` |
| Guard fallbacks | `26` |
| Shield rejections | `0` |

Seed ranking:

| Seed | Score | Avg risk | Worst risk | Unstable risk | Avg MRR | Min MRR | Guard fallbacks | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 616 | 0.993 | 0.476 | 0.589 | 0.524 | 1.128 | 1.101 | 26 | yes |
| 919 | 1.015 | 0.491 | 0.572 | 0.572 | 1.110 | 1.082 | 12 | no |
| 717 | 1.030 | 0.495 | 0.583 | 0.583 | 1.133 | 1.130 | 14 | no |
| 818 | 1.034 | 0.421 | 0.536 | 0.536 | 0.962 | 0.958 | 0 | no |

Artifacts:

| Check | Output |
|---|---|
| Champion directory | `results/rl_td3_fixed_mrr2_shadow_champion` |
| Candidate ranking | `results/rl_td3_fixed_mrr2_shadow_champion/candidate_ranking.csv` |
| Selected policy JSON | `results/rl_td3_fixed_mrr2_shadow_champion/selected_policy.json` |
| Policy card | `results/rl_td3_fixed_mrr2_shadow_champion/policy_card.md` |

Lesson: seed `818` had the best average risk and zero guard fallbacks, but it
dropped below the MRR target, so it was rejected as a potential reward-hacking
candidate. Seed `616` is the current fixed-MRR2 shadow champion because it
maintains MRR above target and has the best selection score among viable
policies. The generated policy card explicitly marks the boundary as
`shadow_only`, with CNC writes disabled and human review required.

## 2026-05-02 - RL Shadow Recommendation Replay

Added `chatter-twin shadow-rl-policy`, which loads a selected SAC/TD3 policy
card and exports the learned policy's proposed feed/spindle overrides as an
offline recommendation trace. This is the first hardware-facing shape for the
selected RL policy, but it remains shadow-only: CNC writes are disabled in the
artifact boundary, the safety shield is required, and human review is required.

Run:

```bash
rtk uv run --extra rl chatter-twin shadow-rl-policy \
  --selection results/rl_td3_fixed_mrr2_shadow_champion \
  --scenarios stable,near_boundary,onset,unstable \
  --episodes 4 \
  --steps 10 \
  --decision-interval 0.08 \
  --randomize \
  --spindle-scale 0.75 1.25 \
  --feed-scale 0.55 1.45 \
  --axial-depth-scale 0.75 2.30 \
  --radial-depth-scale 0.55 1.55 \
  --stiffness-scale 0.60 1.40 \
  --damping-scale 0.45 1.65 \
  --cutting-coeff-scale 0.60 1.80 \
  --noise-scale 1.00 3.50 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_fixed_mrr2_shadow_replay_demo
```

Window metrics:

| Metric | Value |
|---|---:|
| Recommendation windows | 160 |
| Action windows | 74 |
| Action fraction | 0.463 |
| Guard fallbacks | 16 |
| Guard fallback fraction | 0.100 |
| Shield rejections | 0 |
| Mean risk | 0.507 |
| Max risk | 0.746 |
| Mean feed override | 1.046 |
| Mean spindle override | 1.080 |
| Relative MRR proxy | 1.130 |

Scenario metrics:

| Scenario | Windows | Mean risk | Max risk | Relative MRR | Guard fallbacks | Shield rejections |
|---|---:|---:|---:|---:|---:|---:|
| near_boundary | 40 | 0.463 | 0.746 | 1.115 | 9 | 0 |
| onset | 40 | 0.496 | 0.727 | 1.137 | 0 | 0 |
| stable | 40 | 0.498 | 0.720 | 1.132 | 7 | 0 |
| unstable | 40 | 0.573 | 0.714 | 1.137 | 0 | 0 |

Artifacts:

| Check | Output |
|---|---|
| Shadow replay directory | `results/rl_td3_fixed_mrr2_shadow_replay_demo` |
| Recommendations | `results/rl_td3_fixed_mrr2_shadow_replay_demo/recommendations.csv` |
| Action trace | `results/rl_td3_fixed_mrr2_shadow_replay_demo/action_trace.csv` |
| Metrics | `results/rl_td3_fixed_mrr2_shadow_replay_demo/shadow_metrics.json` |
| Report | `results/rl_td3_fixed_mrr2_shadow_replay_demo/report.md` |

Lesson: the selected seed-616 TD3 policy now produces a concrete
recommendation artifact with no shield rejections and MRR above target under
the harsh randomization profile. The risk values are still too high to claim
hardware readiness, so this is a shadow-mode review artifact and the next
non-GPU step is to add replay gates/acceptance checks before any live-machine
integration work.

## 2026-05-02 - RL Shadow Replay Acceptance Gate

Added `chatter-twin gate-rl-shadow`, which reads a shadow replay directory and
writes explicit pass/fail checks for staged promotion. The gate checks risk,
MRR, guard fallback burden, action burden, shield rejections, the deployment
boundary, and profile-specific validation evidence. Passing any profile still
records `hardware_actuation_allowed=false`; this is not a CNC write approval.

Run:

```bash
rtk uv run chatter-twin gate-rl-shadow \
  --profile shadow_review \
  --shadow-dir results/rl_td3_fixed_mrr2_shadow_replay_demo \
  --out results/rl_td3_fixed_mrr2_shadow_gate_demo
```

Profile results:

| Profile | Status | Promotion level | Failed checks |
|---|---|---|---|
| `shadow_review` | pass | `shadow_review_candidate` | none |
| `live_shadow` | blocked | `do_not_promote` | windows, mean risk, max risk, unstable risk, real-machine data, operator approval |
| `hardware_actuation` | blocked | `do_not_promote` | windows, mean risk, max risk, unstable risk, guard fallback fraction, real-machine data, operator approval, hardware interlock |

Shadow-review result:

| Field | Value |
|---|---|
| Status | `pass` |
| Promotion level | `shadow_review_candidate` |
| Hardware actuation allowed | `false` |
| Failed checks | none |

Checks:

| Check | Actual | Threshold | Result |
|---|---:|---:|---|
| Recommendation windows | 160 | 1 | pass |
| Mean risk | 0.507 | 0.550 | pass |
| Max risk | 0.746 | 0.800 | pass |
| Unstable mean risk | 0.573 | 0.600 | pass |
| Relative MRR proxy | 1.130 | 1.000 | pass |
| Guard fallback fraction | 0.100 | 0.200 | pass |
| Action fraction | 0.463 | 0.800 | pass |
| Shield rejections | 0 | 0 | pass |
| Deployment mode | `shadow_only` | `shadow_only` | pass |
| CNC writes enabled | `false` | `false` | pass |

Artifacts:

| Check | Output |
|---|---|
| Shadow-review gate | `results/rl_td3_fixed_mrr2_shadow_gate_demo` |
| Live-shadow gate | `results/rl_td3_fixed_mrr2_live_shadow_gate_demo` |
| Hardware-actuation gate | `results/rl_td3_fixed_mrr2_hardware_actuation_gate_demo` |

Lesson: the selected TD3 policy now has a reproducible promotion ladder. It is
reviewable offline, but correctly blocked from live-shadow and hardware
promotion because it is synthetic, below the stricter sample-count bar, and too
risky for those profiles.

## 2026-05-02 - Good Internal Demo Report

Added `chatter-twin internal-demo-report`, which collects the current ignored
`results/` artifacts into a tracked internal demo report and JSON summary.
This gives the project a single review artifact instead of requiring someone
to open calibration reports, risk-model metrics, RL comparisons, policy cards,
shadow replay outputs, and gate reports one by one.

Run:

```bash
rtk uv run chatter-twin internal-demo-report \
  --out docs/INTERNAL_DEMO_REPORT.md \
  --summary-out docs/INTERNAL_DEMO_SUMMARY.json \
  --test-status "95 passed in 15.58s"
```

Report outputs:

| Artifact | Output |
|---|---|
| Internal demo report | `docs/INTERNAL_DEMO_REPORT.md` |
| Machine-readable summary | `docs/INTERNAL_DEMO_SUMMARY.json` |

Demo conclusion:

| Field | Value |
|---|---|
| Current stage | `offline_shadow_review` |
| Readiness | `good_internal_demo_complete` |
| RL champion | `td3 seed 616` |
| Shadow-review gate | `pass` |
| Live-shadow gate | `blocked` |
| Hardware-actuation gate | `blocked` |
| Hardware ready | `false` |

Lesson: the good internal demo is now complete as a software artifact. It shows
the full offline decision stack working end to end and makes the boundary clear:
real CNC validation is still the next result barrier.
