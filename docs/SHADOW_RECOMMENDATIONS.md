# Shadow-Mode Recommendations

`chatter-twin shadow-eval` is the bridge between early-warning risk estimation
and controller/RL work. It does not command a CNC controller. It replays a
risk model's test predictions, emits bounded feed/spindle override
recommendations, and scores the warning burden, productivity proxy, and
event-level lead-time behavior.

## Command

```bash
rtk uv run chatter-twin shadow-eval \
  --model-dir results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --threshold-source event \
  --warning-feed 0.92 \
  --warning-spindle 1.04
```

Useful threshold modes:

| Mode | Meaning |
|---|---|
| `event` | Use `metrics.json -> event_warning.threshold_selection.selected_threshold`. This is the default and best current deployable proxy. |
| `lead` | Use the window-level lead-time selected threshold. Useful for diagnostics. |
| `default` | Use the model/report default threshold, currently 0.5 when no event threshold is present. |
| `manual` | Require `--warning-threshold` and use that exact value. |

## Policy

The current shadow policy is deliberately simple:

| Piece | Behavior |
|---|---|
| Warning score | `predicted_chatter_score` from the trained risk model. |
| Activation | Warning becomes active when score is at or above the resolved warning threshold. |
| Clearing | Warning clears when score falls below `clear_threshold`, defaulting to 75% of the warning threshold. |
| Action | While warning is active, target feed override is `--warning-feed` and target spindle override is `--warning-spindle`. |
| Bounds | Feed and spindle overrides are clipped by min/max policy limits. |
| Rate limit | Per-window changes are limited by `--max-feed-delta` and `--max-spindle-delta`. |

This is a shadow-mode supervisory policy, not the final controller. It is meant
to answer whether the estimator is good enough to drive bounded recommendations
before building a learned controller.

## Outputs

| File | Meaning |
|---|---|
| `recommendations.csv` | One row per prediction window with `shadow_warning`, `feed_override`, `spindle_override`, `action_active`, and `relative_mrr_proxy`. |
| `shadow_metrics.json` | Machine-readable policy, window burden, source-model metrics, and event metrics. |
| `report.md` | Compact human-readable summary. |

## Metrics

Window metrics summarize how expensive the recommendation policy would be:

| Metric | Meaning |
|---|---|
| `warning_fraction` | Fraction of windows where the warning state is active. |
| `action_fraction` | Fraction of windows where feed or spindle override differs from 1.0. |
| `relative_mrr_proxy` | Mean `feed_override * spindle_override`. This is only a proxy, but it catches policies that suppress chatter by starving the cut. |
| `relative_mrr_loss_proxy` | `max(0, 1 - relative_mrr_proxy)`. |

Event metrics summarize whether the policy warned before current-window chatter:

| Metric | Meaning |
|---|---|
| `event_episodes` | Episodes containing at least one current `slight` or `severe` chatter window. |
| `detected_event_episodes` | Event episodes with a warning on a prior stable/transition window. |
| `false_warning_episodes` | Quiet episodes that still received a warning. |
| `mean_detected_lead_time_s` | Time between first early warning and first current chatter window, for detected events. |

For real hardware, this remains only a recommendation layer until MTConnect/DAQ
integration, sensor-health gates, controller-state checks, and operator
approval are added.

## Counterfactual Replay

Use `shadow-counterfactual` after `shadow-eval` to apply the recommended
feed/spindle overrides back through the local 2-DOF process model:

```bash
rtk uv run chatter-twin shadow-counterfactual \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

This writes:

| File | Meaning |
|---|---|
| `counterfactual_windows.csv` | Baseline-vs-shadow local simulation result per replay window. |
| `counterfactual_metrics.json` | Mean risk, chatter fractions, MRR proxy, and episode-level mitigation counts. |
| `report.md` | Compact summary. |

The counterfactual is local per window. It joins `recommendations.csv` against
the original replay `windows.csv`, rebuilds the process context, simulates the
baseline and shadow action, and compares the resulting hybrid risk estimates.
It does not yet propagate vibration state continuously across a whole cut.

## Stability-Margin Policy

Use `shadow-stability-policy` to keep the same warning timing but replace the
fixed action with a physics-margin spindle selector:

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

For warning windows, the policy reconstructs the modal and cut context from
`windows.csv`, evaluates candidate spindle overrides against the
stability-margin estimate, and accepts a move only when the selected margin
improves over the baseline by at least `--min-margin-improvement`. Feed remains
at 1.0 by default to avoid productivity cheating.

The output directory is compatible with `shadow-counterfactual` because it
writes a fresh `recommendations.csv` plus `shadow_metrics.json`.

## Counterfactual-Risk Policy

Use `shadow-counterfactual-policy` when you want the action selector to optimize
the same local risk objective used by `shadow-counterfactual`:

```bash
rtk uv run chatter-twin shadow-counterfactual-policy \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --feed-values 1.0 \
  --spindle-values 0.96,1.0,1.04,1.08 \
  --min-risk-reduction 0.005 \
  --sensor-noise 0.0
```

For warning windows, it simulates each candidate feed/spindle pair, scores
predicted local risk reduction with productivity and movement penalties, and
only accepts an action when the selected candidate clears
`--min-risk-reduction`. This is still a local per-window selector, but it is the
first policy path whose objective matches the counterfactual evaluation metric.

## Episode Counterfactual

Use `shadow-episode-counterfactual` when you want actions to persist across the
whole synthetic cut instead of resetting each replay window independently:

```bash
rtk uv run chatter-twin shadow-episode-counterfactual \
  --shadow-dir results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_episode_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo \
  --sensor-noise 0.0
```

This command rebuilds each held-out replay episode, converts the policy's
`recommendations.csv` into time-varying feed/spindle profiles, simulates the
full episode once with baseline controls and once with shadow controls, then
slices both signals back into comparable windows.

It writes:

| File | Meaning |
|---|---|
| `episode_windows.csv` | Baseline-vs-shadow risk/label comparison for each replay window after full-episode simulation. |
| `episode_summary.csv` | One row per episode with mean risk, event flags, and MRR proxy. |
| `episode_metrics.json` | Aggregate window and episode metrics. |
| `report.md` | Compact summary. |

## Action Sweep

Use `shadow-action-sweep` to keep the same warning timing but try several
override pairs:

```bash
rtk uv run chatter-twin shadow-action-sweep \
  --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo \
  --out results/shadow_action_sweep_onset_events_axial_depth_holdout_validated_demo \
  --feed-values 0.88,0.92,1.0 \
  --spindle-values 0.96,1.0,1.04 \
  --sensor-noise 0.0
```

This writes `sweep.csv`, `sweep_metrics.json`, `best_counterfactual_windows.csv`,
and `report.md`. The best policy is chosen by mean local risk reduction, with
relative MRR as a tie-breaker. If the best policy is `feed=1.0` and
`spindle=1.0`, the current action set is not justified by the local twin even
if the warning detector is working.
