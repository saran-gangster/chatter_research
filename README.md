# Chatter Twin

Software-first MVP for a CNC milling chatter-margin digital twin.

This package implements the first safe slice of the project described in
`CNC_CHATTER_DIGITAL_TWIN_RL_SYSTEM_DESIGN.md`: a simulation/research-code
scaffold with no CNC hardware writes and no live-machine dependency.

## What Is Included

- 2-DOF regenerative milling simulator with tooth engagement, delay, runout,
  sensor noise, drift knobs, and progressive onset profiles.
- FRF/regenerative-phase stability-margin model that can be replaced by a full
  Altintas/Budak implementation later.
- Signal feature extraction and hybrid chatter-risk estimation.
- Feed/spindle override safety shield.
- Gymnasium environment for controller/RL experiments.
- Fixed, rule-based, SLD-aware, counterfactual-risk, hybrid risk/margin, and
  lookahead MPC-style controller baselines.
- Synthetic replay-window export with optional domain randomization.
- i-CNC Zenodo dataset manifest/download/import helpers for a first real-data
  signal-validation pass.
- Offline shadow-mode recommendation, counterfactual replay, and action-sweep
  evaluation from risk-model predictions.
- Reproducible internal demo report that summarizes calibration, risk,
  controller, RL, shadow replay, and gate-readiness artifacts.
- CLI smoke demos.

## Quick Start

```bash
rtk uv run --extra dev pytest -q
rtk uv run chatter-twin env-smoke
rtk uv run chatter-twin simulate --scenario unstable --duration 0.5 --out /tmp/chatter_unstable.npz
rtk uv run chatter-twin rollout --controller rule --steps 20
rtk uv run chatter-twin rollout --controller cf --scenario unstable --steps 10
rtk uv run chatter-twin rollout --controller hybrid --scenario onset --steps 10
rtk uv run chatter-twin benchmark --controllers fixed,rule,sld,mpc,cf --scenarios stable,near_boundary,unstable --episodes 5 --steps 20 --out results/benchmark_demo
rtk uv run chatter-twin closed-loop-benchmark --controllers fixed,rule,sld,mpc,cf,hybrid --scenarios stable,near_boundary,onset,unstable --episodes 2 --steps 8 --decision-interval 0.10 --out results/closed_loop_benchmark_hybrid_demo
rtk uv run chatter-twin calibrate-margin --scenarios stable,near_boundary,onset,unstable --duration 0.25 --sensor-noise 0.0 --target-threshold 0.35 --out results/margin_calibration_time_domain_incipient_demo
rtk uv run chatter-twin calibrate-margin --scenarios stable,near_boundary,onset,unstable --axial-depth-scales 0.35,0.60,0.85,1.10,1.35,1.60,2.00,2.50,3.00 --spindle-scales 0.88,0.94,1.00,1.06,1.12 --duration 0.20 --sensor-noise 0.0 --target-threshold 0.35 --family-count 5 --holdout-family 4 --out results/margin_calibration_random_family_holdout_demo
rtk uv run chatter-twin calibrate-margin --calibration-model context --scenarios stable,near_boundary,onset,unstable --axial-depth-scales 0.35,0.60,0.85,1.10,1.35,1.60,2.00,2.50,3.00 --spindle-scales 0.88,0.94,1.00,1.06,1.12 --duration 0.20 --sensor-noise 0.0 --target-threshold 0.35 --family-count 5 --holdout-family 4 --out results/margin_calibration_context_family_holdout_demo
rtk uv run chatter-twin closed-loop-benchmark --controllers fixed,rule,sld,mpc,cf,hybrid --scenarios stable,near_boundary,onset,unstable --episodes 2 --steps 8 --decision-interval 0.10 --margin-calibration results/margin_calibration_time_domain_incipient_demo/calibration.json --out results/closed_loop_benchmark_incipient_calibrated_margin_guarded_demo
rtk uv run chatter-twin closed-loop-benchmark --controllers fixed,rule,sld,mpc,cf,hybrid --scenarios stable,near_boundary,onset,unstable --episodes 8 --steps 12 --decision-interval 0.10 --randomize --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/closed_loop_benchmark_context_calibrated_randomized_stress_demo
rtk uv run chatter-twin export-synthetic --episodes 3 --duration 1.0 --window 0.1 --stride 0.05 --out results/synthetic_replay_demo
rtk uv run chatter-twin export-synthetic --episodes 4 --duration 0.6 --window 0.1 --stride 0.05 --randomize --out results/synthetic_randomized_demo
rtk uv run chatter-twin export-synthetic --scenarios stable,near_boundary,unstable --episodes 4 --duration 0.6 --window 0.1 --stride 0.05 --randomize --focus-transitions --transition-candidates 12 --min-transition-windows 3 --horizon 0.2 --out results/synthetic_transition_focus_demo
rtk uv run chatter-twin export-synthetic --scenarios stable,near_boundary,onset,unstable --episodes 4 --duration 0.9 --window 0.1 --stride 0.05 --randomize --focus-transitions --transition-candidates 12 --min-transition-windows 3 --horizon 0.25 --out results/synthetic_onset_demo
rtk uv run chatter-twin icnc-manifest --out data/raw/icnc/source_manifest.json
rtk uv run chatter-twin download-icnc --out "data/raw/icnc/i-CNC Dataset.zip" --manifest-out data/raw/icnc/source_manifest.json
rtk uv run chatter-twin ingest-icnc --source "data/raw/icnc/i-CNC Dataset.zip" --out results/icnc_replay_subset --window 0.1 --stride 0.05 --horizon 0.25 --max-packages-per-file 1000
rtk uv run chatter-twin train-risk --dataset results/synthetic_replay_demo --out results/risk_model_demo --epochs 800
rtk uv run chatter-twin train-risk --dataset results/synthetic_randomized_demo --out results/risk_model_axial_depth_holdout_demo --epochs 800 --split-mode parameter_family --holdout-column axial_depth_scale --holdout-tail high
rtk uv run chatter-twin train-risk --dataset results/synthetic_onset_demo --out results/risk_model_hist_gb_interaction_onset_horizon_episode_validated_demo --model hist_gb --calibration none --feature-set interaction_temporal --target horizon --split-mode episode --validation-fraction 0.25
rtk uv run chatter-twin shadow-eval --model-dir results/risk_model_hist_gb_interaction_onset_events_axial_depth_holdout_validated_demo --out results/shadow_policy_onset_events_axial_depth_holdout_validated_demo --threshold-source event --warning-feed 0.92 --warning-spindle 1.04
rtk uv run chatter-twin shadow-stability-policy --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo --out results/shadow_stability_policy_onset_events_axial_depth_holdout_validated_demo --min-spindle 0.90 --max-spindle 1.10 --candidates 21 --feed 1.0 --min-margin-improvement 0.05
rtk uv run chatter-twin shadow-counterfactual-policy --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo --out results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo --feed-values 1.0 --spindle-values 0.96,1.0,1.04,1.08 --min-risk-reduction 0.005 --sensor-noise 0.0
rtk uv run chatter-twin shadow-counterfactual --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo --out results/shadow_counterfactual_onset_events_axial_depth_holdout_validated_demo --sensor-noise 0.0
rtk uv run chatter-twin shadow-episode-counterfactual --shadow-dir results/shadow_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo --out results/shadow_episode_counterfactual_counterfactual_policy_onset_events_axial_depth_holdout_validated_demo --sensor-noise 0.0
rtk uv run chatter-twin shadow-action-sweep --shadow-dir results/shadow_policy_onset_events_axial_depth_holdout_validated_demo --out results/shadow_action_sweep_onset_events_axial_depth_holdout_validated_demo --feed-values 0.88,0.92,1.0 --spindle-values 0.96,1.0,1.04 --sensor-noise 0.0
rtk uv run --extra rl chatter-twin train-rl --algorithm sac --scenarios stable,unstable --total-timesteps 64 --eval-episodes 1 --steps 4 --decision-interval 0.05 --learning-starts 8 --batch-size 8 --buffer-size 512 --randomize --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_sac_cpu_smoke_demo
rtk uv run --extra rl chatter-twin train-rl --algorithm td3 --scenarios stable,unstable --total-timesteps 64 --eval-episodes 1 --steps 4 --decision-interval 0.05 --learning-starts 8 --batch-size 8 --buffer-size 512 --randomize --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_td3_cpu_smoke_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac,td3 --seeds 616,717 --scenarios stable,unstable --total-timesteps 64 --eval-episodes 1 --steps 4 --decision-interval 0.05 --learning-starts 8 --batch-size 8 --buffer-size 512 --randomize --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_multiseed_cpu_smoke_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac,td3 --seeds 616,717 --scenarios stable,near_boundary,onset,unstable --total-timesteps 256 --eval-episodes 2 --steps 6 --decision-interval 0.08 --learning-starts 32 --batch-size 16 --buffer-size 4096 --randomize --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_multiseed_cpu_short_randomized_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac,td3 --seeds 616,717 --scenarios stable,near_boundary,onset,unstable --total-timesteps 256 --eval-episodes 2 --steps 6 --decision-interval 0.08 --learning-starts 32 --batch-size 16 --buffer-size 4096 --randomize --productivity-mode mrr --productivity-weight 2.0 --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_multiseed_cpu_mrr2_reward_ablation_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac,td3 --seeds 616,717 --scenarios stable,near_boundary,onset,unstable --total-timesteps 256 --eval-episodes 2 --steps 6 --decision-interval 0.08 --learning-starts 32 --batch-size 16 --buffer-size 4096 --randomize --action-mode delta --productivity-mode mrr --productivity-weight 2.0 --smoothness-weight 0.1 --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_delta_action_mrr2_smooth01_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac --seeds 616,717,818 --scenarios stable,near_boundary,onset,unstable --total-timesteps 1000 --eval-episodes 4 --steps 8 --decision-interval 0.08 --learning-starts 100 --batch-size 32 --buffer-size 10000 --randomize --action-mode delta --productivity-mode mrr --productivity-weight 2.0 --smoothness-weight 0.1 --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_sac_delta_cpu_long_randomized_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms sac --seeds 616,717,818 --scenarios stable,near_boundary,onset,unstable --total-timesteps 1000 --eval-episodes 4 --steps 8 --decision-interval 0.08 --learning-starts 100 --batch-size 32 --buffer-size 10000 --randomize --action-mode delta --uncertainty-mode hold --productivity-mode mrr --productivity-weight 2.0 --smoothness-weight 0.1 --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_sac_delta_uncertainty_hold_cpu_long_demo
rtk uv run --extra rl chatter-twin train-rl-multiseed --algorithms td3 --seeds 616,717,818 --scenarios stable,near_boundary,onset,unstable --total-timesteps 1000 --eval-episodes 4 --steps 8 --decision-interval 0.08 --learning-starts 100 --batch-size 32 --buffer-size 10000 --randomize --action-mode delta --uncertainty-mode hold --productivity-mode mrr --productivity-weight 2.0 --smoothness-weight 0.1 --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json --out results/rl_td3_delta_uncertainty_hold_cpu_long_demo
rtk uv run chatter-twin compare-rl-runs --run sac_strict=results/rl_sac_delta_cpu_long_randomized_demo --run sac_hold=results/rl_sac_delta_uncertainty_hold_cpu_long_demo --run td3_hold=results/rl_td3_delta_uncertainty_hold_cpu_long_demo --baseline-label sac_strict --out results/rl_strict_hold_delta_comparison_demo
rtk uv run chatter-twin internal-demo-report --out docs/INTERNAL_DEMO_REPORT.md --summary-out docs/INTERNAL_DEMO_SUMMARY.json
```

## Benchmark Outputs

`chatter-twin benchmark` writes reset-style controller comparison artifacts.
`chatter-twin closed-loop-benchmark` writes the same report shape, but carries
the simulated displacement/velocity state and accepted controller actions
between decision steps. Add `--randomize` to either benchmark mode to perturb
spindle, feed, depth, modal, cutting-coefficient, and noise scales per episode.

- `episodes.csv`: one row per controller/scenario/episode.
- `summary.csv`: aggregated controller/scenario metrics.
- `summary.json`: machine-readable episode and summary payload.
- `summary.md`: compact human-readable report.
- `risk_vs_mrr.svg`: dependency-free visual comparison of mean risk and relative MRR proxy.
- `pareto.csv`: scenario-level risk/productivity Pareto flags.
- `pareto.svg`: dependency-free Pareto scatter plot.

`chatter-twin calibrate-margin` sweeps spindle/depth conditions, compares the
raw physics margin with a signal-only time-domain simulator target, and writes
`calibration.json`. Add `--family-count` plus `--holdout-family` to randomize
modal, cutting, feed, radial-depth, and runout families and report train vs
held-out calibration quality. Use `--calibration-model context` to include
modal, cutting, feed, depth, spindle, and runout context instead of fitting only
raw margin. Pass `calibration.json` to `--margin-calibration` on rollout or
benchmark commands to evaluate controllers with the calibrated margin scale and
calibration-driven uncertainty. Planning controllers (`mpc`, `cf`, and
`hybrid`) also score candidate actions against that uncertainty so they can
avoid moves that the safety shield would reject.

## Replay Dataset Outputs

`chatter-twin export-synthetic` writes a replay-window dataset compatible with
the schema in `docs/REPLAY_SCHEMA.md`:

- `dataset.npz`: windowed sensor arrays plus aligned labels/targets.
- `windows.csv`: per-window machining context, features, and risk summaries.
- `manifest.json`: dataset-level schema metadata.
- `README.md`: compact dataset summary.

Add `--randomize` to perturb spindle speed, feed, axial/radial depth, modal
stiffness/damping, cutting coefficients, and sensor noise per synthetic
episode. The applied scale factors are saved in both `dataset.npz` and
`windows.csv`, with the active ranges recorded in `manifest.json`.
Add `--focus-transitions` to mine several candidate perturbations per requested
episode and keep cuts that better populate the stable/transition/slight
boundary. Exports also include horizon/onset targets; set `--horizon` to control
the early-warning window. Include the `onset` scenario when you need progressive
stable-to-chatter episodes with measurable lead time before the current-window
label becomes `slight` or `severe`.

`chatter-twin icnc-manifest`, `download-icnc`, and `ingest-icnc` support the
first public real-data validation pass against the i-CNC Zenodo dataset. The
raw download is about 3 GB and is kept under ignored `data/`; imported replay
subsets are written under ignored `results/`. `ingest-icnc` skips
`No Machining`/unknown packages by default; pass `--include-unknown` only when
you want an operational-state dataset instead of cutting-only validation data.
See `docs/REAL_DATASETS.md`.

## Offline Risk Training

`chatter-twin train-risk` trains CPU-only risk estimators over replay-window
metadata/features and writes `model.json`, `metrics.json`, `predictions.csv`,
`confusion_matrix.csv`, and `report.md`. The default model is an inspectable
softmax baseline; `--model hist_gb` adds a stronger histogram gradient-boosted
tree baseline and writes `model.joblib`. Use `--feature-set temporal` with
schema v3+ replay exports to include within-cut growth/EMA features, and
`--feature-set profile_temporal` with schema v5 onset exports to also include
within-episode process profile scales. Use `--feature-set interaction_temporal`
to add physics-shaped derived features such as depth-to-critical ratio and
process force proxy. Use `--target horizon` with schema v4+ exports for
early-warning labels. See
`docs/OFFLINE_RISK_TRAINING.md`.
Use `--split-mode episode` or `--split-mode parameter_family` for more honest
generalization checks on synthetic replay data. Reports also include lead-time
metrics that score only current stable/transition windows with future chatter
inside the configured horizon. Add `--validation-fraction` to select the
lead-time warning threshold on a held-out validation subset instead of the fit
or test split. Reports also include event-level warning metrics that score each
cut episode once: did the model warn before the first current chatter window?

## Shadow-Mode Recommendations

`chatter-twin shadow-eval` converts a trained risk model's `predictions.csv`
into offline feed/spindle override recommendations. It uses the selected
event-warning threshold by default, applies hysteresis and rate limits, and
writes `recommendations.csv`, `shadow_metrics.json`, and `report.md`.
`shadow-counterfactual` replays those recommendations through the local
process model, while `shadow-action-sweep` tests simple override pairs on the
same warning timing.
`shadow-stability-policy` keeps the warning timing but chooses spindle overrides
from the local stability-margin estimate, preserving feed by default.
`shadow-counterfactual-policy` chooses actions by directly minimizing local
counterfactual risk on the same process twin used for evaluation.
`shadow-episode-counterfactual` applies a policy as persistent feed/spindle
profiles across each full replay episode.
See `docs/SHADOW_RECOMMENDATIONS.md`.

## RL Controllers

`chatter-twin train-rl` trains shielded RL controller candidates. `q_learning`
is a tiny CPU-only artifact-pipeline baseline; `sac` and `td3` use
Stable-Baselines3 through the optional `rl` extra. The optional PyTorch
dependency is pinned to the CPU wheel index, so these commands do not require a
GPU. SAC/TD3 runs write a learned policy checkpoint plus matched `sld`/`mpc`
baseline comparisons. `chatter-twin train-rl-multiseed` repeats those runs over
a seed list and writes aggregate mean/std summaries for learned policies and
matched baselines. Reward ablations are exposed through `--productivity-mode`,
`--productivity-weight`, `--risk-now-weight`, `--risk-horizon-weight`,
`--severe-penalty`, `--smoothness-weight`, `--rejection-penalty`,
`--clip-penalty`, and `--rate-limit-penalty`. SAC/TD3
runs also write raw-action, candidate-guard, and shield diagnostics so policy
behavior can be debugged without guessing from aggregate reward alone. See
`docs/RL_CONTROLLERS.md`.
Use `--action-mode delta` for SAC/TD3 policies that propose bounded override
changes from the current machine state instead of absolute override targets.
Use `--delta-action-scale` with delta mode to narrow the normalized SAC/TD3
action range without changing the machine shield limits.
Use `--delta-feed-scale` and `--delta-spindle-scale` when feed and spindle
need different delta ranges.
Use `--delta-mapping headroom` when delta actions should consume only the
remaining safe override headroom instead of asking beyond the shield boundary
and relying on candidate clipping.
Use `--uncertainty-mode hold` to keep the strict hard-failure shield while
treating high calibration uncertainty as a conservative hold instead of a full
shield rejection.
Use `chatter-twin compare-rl-runs` to combine several RL result directories
into one strict/hold and algorithm comparison report, including a
whole-profile candidate summary for risk/MRR tradeoffs.
Use `chatter-twin eval-rl-run` to load saved SAC/TD3 `model.zip` artifacts and
evaluate the already-trained policies under a fresh stress randomization
profile without retraining.
Use `chatter-twin select-rl-policy` on a saved-policy evaluation directory to
rank seed checkpoints and write a shadow-only policy card for the selected
candidate.
Use `chatter-twin shadow-rl-policy` to load that selected policy and export a
shadow-only recommendations trace. It writes `recommendations.csv`,
`action_trace.csv`, `shadow_metrics.json`, and `report.md`; CNC writes remain
disabled in the artifact boundary.
Use `chatter-twin gate-rl-shadow` to check a shadow replay against explicit
risk, productivity, fallback, action-burden, shield-rejection, and deployment
boundary gates before promoting it. Gate profiles are staged as
`shadow_review`, `live_shadow`, and `hardware_actuation`; passing any gate still
does not enable CNC writes.

## Internal Demo

`chatter-twin internal-demo-report` collects the current ignored `results/`
artifacts into a tracked demo narrative at `docs/INTERNAL_DEMO_REPORT.md` and a
machine-readable summary at `docs/INTERNAL_DEMO_SUMMARY.json`. The current
report conclusion is: the good internal offline demo is complete, the selected
TD3 policy is a shadow-review candidate, and real CNC validation remains the
next result barrier.

## Safety Boundary

This repository currently does not connect to MTConnect, DAQ hardware, or any
CNC controller. Hardware-facing work should remain behind explicit interfaces
and shadow-mode testing until the safety shield and validation plan are mature.
