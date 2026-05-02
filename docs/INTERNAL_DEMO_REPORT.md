# Chatter Twin Internal Demo Report

Generated: `2026-05-02T17:15:36+00:00`

## Executive Conclusion

Software demo complete; real CNC validation is still the decisive missing result.

| Field | Value |
|---|---|
| Current stage | `offline_shadow_review` |
| RL champion | `td3 seed 616` |
| Hardware ready | `false` |
| Test status | `106 passed in 15.69s` |

## What This Demo Proves

- Simulator, risk estimator, controller comparisons, shielded RL replay, and promotion gates run end to end.
- The selected TD3 policy passes the offline shadow-review gate.
- The same policy is blocked from live-shadow and hardware-actuation profiles.
- Public synchronized high-rate KIT force/acceleration data has been ingested for offline replay sanity checks.

## What It Does Not Prove Yet

- No MTConnect or controller API connection is implemented.
- No user-owned synchronized high-rate accelerometer/audio/current stream has been ingested.
- No real FRF/cutting-coefficient calibration or real CNC chatter cut has been run through this stack.
- No CNC write path exists or is approved.

## Pipeline Status

| Layer | Evidence | Status |
|---|---|---|
| Calibration twin | 900 synthetic calibration samples, holdout ROC AUC 0.764 | `validated_synthetic_holdout` |
| Risk estimator | test chatter F1 0.927, event F1 0.727 | `validated_synthetic_holdout` |
| Classical/controller baselines | 6 controllers compared; best risk `sld` | `validated_synthetic_benchmark` |
| Offline shadow policy | event F1 0.727, warning fraction 0.400 | `shadow_policy_evaluated` |
| Shadow counterfactual | mean risk reduction 0.007, mitigated events 0 | `counterfactual_replayed` |
| RL controller candidate | TD3 seed 616 selected; avg MRR 1.128 | `shadow_champion_selected` |
| RL shadow replay | 160 windows, MRR 1.130, shield rejects 0 | `shadow_only_replay` |
| Promotion gates | offline shadow review passes; live and hardware gates block as intended | `good_internal_demo_complete` |

## Calibration

| Metric | Value |
|---|---:|
| Samples | 900 |
| Train ROC AUC | 0.829 |
| Holdout ROC AUC | 0.764 |
| Holdout accuracy @ 0.5 | 0.739 |
| Holdout MAE to time-domain risk | 0.147 |

## Risk Estimator

| Metric | Value |
|---|---:|
| Test accuracy | 0.839 |
| Binary chatter F1 | 0.927 |
| Binary chatter precision | 0.981 |
| Binary chatter recall | 0.878 |
| Event warning F1 | 0.727 |
| Event precision | 1 |
| Event recall | 0.571 |
| Mean detected lead time, s | 0.188 |

## Controller Baselines

| Controller | Avg risk | Worst risk | Avg relative MRR | Shield rejections |
|---|---:|---:|---:|---:|
| `cf` | 0.448 | 0.532 | 0.996 | 0 |
| `fixed` | 0.456 | 0.543 | 1 | 0 |
| `hybrid` | 0.440 | 0.519 | 0.996 | 0 |
| `mpc` | 0.439 | 0.532 | 1.043 | 0 |
| `rule` | 0.466 | 0.542 | 1.024 | 37 |
| `sld` | 0.415 | 0.510 | 0.944 | 0 |

Best baseline by average risk: `sld` (0.415).

## RL Candidate And Shadow Replay

| Item | Value |
|---|---:|
| Selected seed | 616 |
| Selected policy score | 0.993 |
| Champion average risk | 0.476 |
| Champion average relative MRR | 1.128 |
| Shadow replay windows | 160 |
| Shadow action fraction | 0.463 |
| Shadow mean risk | 0.507 |
| Shadow max risk | 0.746 |
| Shadow relative MRR proxy | 1.130 |
| Shadow shield rejections | 0 |

RL stress candidates:

| Profile | Avg risk | Worst risk | Unstable risk | Avg relative MRR | Guard fallbacks | Pareto |
|---|---:|---:|---:|---:|---:|---|
| `td3_fixed_mrr2_stress` | 0.471 | 0.554 | 0.554 | 1.083 | 52 | yes |
| `td3_headroom_mrr3_stress` | 0.483 | 0.563 | 0.563 | 1.129 | 46 | yes |

## Promotion Gates

| Profile | Status | Promotion level | Failed checks |
|---|---|---|---|
| `shadow_review` | `pass` | `shadow_review_candidate` | none |
| `live_shadow` | `blocked` | `do_not_promote` | minimum_recommendation_windows, mean_risk, max_risk, unstable_mean_risk, real_machine_data, operator_approval_evidence |
| `hardware_actuation` | `blocked` | `do_not_promote` | minimum_recommendation_windows, mean_risk, max_risk, unstable_mean_risk, guard_fallback_fraction, real_machine_data, operator_approval_evidence, hardware_interlock_evidence |

## Demo Runbook

1. Re-run tests: `rtk uv run pytest -q`.
2. Regenerate the RL shadow replay: `rtk uv run --extra rl chatter-twin shadow-rl-policy ...` using the command in `docs/RL_CONTROLLERS.md`.
3. Re-run gates: `rtk uv run chatter-twin gate-rl-shadow --profile shadow_review ...`, then repeat for `live_shadow` and `hardware_actuation`.
4. Regenerate this report: `rtk uv run chatter-twin internal-demo-report --out docs/INTERNAL_DEMO_REPORT.md --summary-out docs/INTERNAL_DEMO_SUMMARY.json --test-status "106 passed in 15.69s"`.

## Artifact Map

| Artifact | Path |
|---|---|
| Calibration | `results/margin_calibration_context_family_holdout_demo` |
| Risk model | `results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo` |
| Closed-loop benchmark | `results/closed_loop_benchmark_context_calibrated_randomized_stress_demo` |
| Risk-model shadow policy | `results/shadow_policy_onset_events_axial_depth_holdout_validated_demo` |
| Shadow counterfactual | `results/shadow_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo` |
| RL stress comparison | `results/rl_td3_harsh_stress_comparison` |
| RL champion | `results/rl_td3_fixed_mrr2_shadow_champion` |
| RL shadow replay | `results/rl_td3_fixed_mrr2_shadow_replay_demo` |
| Shadow-review gate | `results/rl_td3_fixed_mrr2_shadow_gate_demo` |
| Live-shadow gate | `results/rl_td3_fixed_mrr2_live_shadow_gate_demo` |
| Hardware-actuation gate | `results/rl_td3_fixed_mrr2_hardware_actuation_gate_demo` |

## Next Result Barrier

The good internal demo is complete when judged as an offline software demo. The next result barrier is real-machine validation: synchronized CNC context plus high-rate sensor data, FRF/cutting calibration, and replay through the same estimator, controllers, and gate profiles.
