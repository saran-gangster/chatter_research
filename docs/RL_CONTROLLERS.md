# Shielded RL Controllers

This repo supports a CPU-first RL path before any GPU training:

1. `q_learning`: tiny tabular baseline for validating the RL artifact pipeline.
2. `sac`: Stable-Baselines3 SAC running on CPU.
3. `td3`: Stable-Baselines3 TD3 running on CPU.

SAC and TD3 are optional because they require PyTorch. The project pins `torch`
to the PyTorch CPU wheel index in `pyproject.toml`, so `--extra rl` does not
pull the CUDA/NVIDIA wheel stack.

## CPU Smoke Commands

SAC:

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

TD3:

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

## Outputs

Each SAC/TD3 run writes:

| Artifact | Purpose |
|---|---|
| `model.zip` | Stable-Baselines3 policy checkpoint. |
| `evaluation_actions.csv` | Per-step raw action, candidate guard, shield, risk, and reward trace. |
| `evaluation_episodes.csv` | Greedy learned-policy evaluation episodes. |
| `evaluation_summary.csv` | Learned-policy summary by scenario. |
| `action_diagnostics_summary.csv` | Scenario-level action/guard/shield diagnostics. |
| `baseline_episodes.csv` | Matched `sld`/`mpc` closed-loop episodes. |
| `baseline_summary.csv` | Matched baseline summary. |
| `comparison_summary.csv` | Learned controller plus baselines in one table. |
| `metrics.json` | Machine-readable config and metrics. |
| `report.md` | Human-readable comparison report. |

`train-rl-multiseed` repeats the same training/evaluation loop over several
seeds and writes:

| Artifact | Purpose |
|---|---|
| `runs/<algorithm>_seed_<seed>/` | Full per-seed `train-rl` artifacts. |
| `run_summary.csv` | One row per algorithm/seed run. |
| `comparison_summary.csv` | Per-seed learned-controller and baseline summaries. |
| `aggregate_summary.csv` | Mean/std aggregate by training algorithm, controller, and scenario. |
| `action_diagnostics_summary.csv` | Per-seed action-diagnostic summaries collected at the top level. |
| `action_diagnostics_aggregate.csv` | Mean/sum action diagnostics aggregated over seeds. |
| `metrics.json` | Machine-readable aggregate payload. |
| `report.md` | Human-readable multi-seed report. |

`compare-rl-runs` combines multiple result directories and writes:

| Artifact | Purpose |
|---|---|
| `profiles.csv` | One row per input run with action, shield, and reward settings. |
| `combined_summary.csv` | Concatenated aggregate summaries from all input runs. |
| `learned_policy_summary.csv` | Learned policy rows only, excluding matched SLD/MPC baselines. |
| `candidate_summary.csv` | Whole-profile learned-policy risk/MRR summary, including worst/unstable risk, min/unstable MRR, guard fallbacks, and Pareto flag. |
| `combined_action_diagnostics.csv` | Concatenated action diagnostics from all input runs. |
| `delta_summary.csv` | Scenario-level deltas against the baseline profile. |
| `metrics.json` | Machine-readable comparison payload. |
| `report.md` | Human-readable comparison report. |

`eval-rl-run` loads saved SAC/TD3 `model.zip` artifacts from either a single
`train-rl` directory or a multi-seed `train-rl-multiseed` directory, then
evaluates those already-trained policies under a fresh scenario/randomization
profile. It writes the same aggregate shape as `train-rl-multiseed`, so its
outputs can be fed directly into `compare-rl-runs`.

`select-rl-policy` ranks seed checkpoints from a saved-policy evaluation
directory and writes a shadow-only deployment card:

| Artifact | Purpose |
|---|---|
| `candidate_ranking.csv` | Ranked seed checkpoints with risk, MRR, fallback, and shield metrics. |
| `selected_policy.json` | Machine-readable selected checkpoint plus deployment boundary. |
| `policy_card.md` | Human-readable shadow-mode policy card for review. |

Reward shaping is explicit and reproducible:

| Option | Default | Meaning |
|---|---:|---|
| `--action-mode` | `absolute` | SAC/TD3 action semantics: absolute override targets or `delta` changes from the current override. |
| `--delta-action-scale` | `1.0` | Scale for the normalized delta action range; `0.5` halves the largest requested per-decision override change. |
| `--delta-feed-scale` | unset | Optional feed-specific delta scale. Overrides `--delta-action-scale` for feed only. |
| `--delta-spindle-scale` | unset | Optional spindle-specific delta scale. Overrides `--delta-action-scale` for spindle only. |
| `--delta-mapping` | `fixed` | `fixed` maps raw deltas to fixed per-step override requests; `headroom` maps them to the remaining safe override headroom before candidate guarding. |
| `--uncertainty-mode` | `reject` | Shield behavior for high calibration uncertainty: strict `reject`, conservative `hold`, or non-rejecting `advisory`. |
| `--productivity-mode` | `feed` | Reward productivity as feed override, or `mrr` for feed times spindle. |
| `--productivity-weight` | `1.0` | Positive productivity weight. |
| `--risk-now-weight` | `3.0` | Current chatter-risk penalty weight. |
| `--risk-horizon-weight` | `1.5` | Horizon chatter-risk penalty weight. |
| `--severe-penalty` | `2.0` | Extra penalty for severe chatter labels. |
| `--smoothness-weight` | `0.5` | Override movement penalty. |
| `--rejection-penalty` | `0.75` | Penalty when the shield rejects an action. |
| `--clip-penalty` | `0.0` | Per-component penalty when the candidate guard or shield clips feed/spindle. |
| `--rate-limit-penalty` | `0.0` | Per-component penalty when the shield rate-limits feed/spindle. |

The learned action is still supervised by a candidate guard before it reaches
the environment shield. The guard evaluates the proposed feed/spindle pair in
the twin and falls back to the current override when the candidate would move
into high calibration uncertainty.

## Current CPU Smoke Results

The smoke runs are intentionally tiny. They prove that SAC/TD3 training,
checkpointing, evaluation, and baseline comparison work on CPU; they are not
evidence that either policy is tuned.

| Run | Output | Shield Rejects |
|---|---|---:|
| SAC CPU smoke | `results/rl_sac_cpu_smoke_demo` | 0 on `stable`, 0 on `unstable` |
| TD3 CPU smoke | `results/rl_td3_cpu_smoke_demo` | 0 on `stable`, 0 on `unstable` |

## Current CPU Randomized Sweeps

These runs use all four scenarios, randomized process parameters, the
context-calibrated margin artifact, 1,000 training timesteps, and matched
`sld`/`mpc` baseline evaluation.

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

Use the same command with `--algorithm td3` and
`--out results/rl_td3_cpu_randomized_demo` for TD3.

Selected results:

| Controller | Scenario | Mean risk | Relative MRR | Mean reward | Shield rejects |
|---|---|---:|---:|---:|---:|
| sac | stable | 0.316 | 0.917 | -4.073 | 0 |
| sac | near_boundary | 0.383 | 0.934 | -7.166 | 0 |
| sac | onset | 0.369 | 0.915 | -6.243 | 0 |
| sac | unstable | 0.390 | 0.932 | -7.630 | 0 |
| td3 | stable | 0.288 | 0.952 | -2.033 | 0 |
| td3 | near_boundary | 0.382 | 0.952 | -6.023 | 0 |
| td3 | onset | 0.370 | 0.952 | -5.253 | 0 |
| td3 | unstable | 0.442 | 0.952 | -8.412 | 0 |

The 1,000-step CPU runs are still small, but they are now real learned-policy
comparisons rather than command smokes. GPU only becomes useful when increasing
timesteps substantially or running multi-seed sweeps.

## Current Multi-Seed CPU Smoke

This command exercises the multi-seed runner over SAC and TD3 with two seeds.
It is intentionally tiny, but it verifies the repeatable artifact layout and
aggregate mean/std reporting.

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

| Training algo | Controller | Scenario | Seeds | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sac | sac | stable | 2 | 0.291 | 0.066 | 0.965 | -1.426 | 0 |
| sac | sac | unstable | 2 | 0.502 | 0.130 | 1.028 | -5.822 | 0 |
| td3 | td3 | stable | 2 | 0.256 | 0.105 | 0.958 | -0.876 | 0 |
| td3 | td3 | unstable | 2 | 0.546 | 0.026 | 1.048 | -6.443 | 0 |
| sac | mpc | unstable | 2 | 0.539 | 0.077 | 1.033 | -5.797 | 0 |
| sac | sld | unstable | 2 | 0.557 | 0.052 | 0.985 | -6.292 | 0 |

This does not replace the longer 1,000-step runs; it gives us the repeatable
multi-seed harness needed to run those longer comparisons without manual glue.

## Current Short Randomized Multi-Seed CPU Run

The full 1,000-step, three-seed all-scenario run was too slow for an
interactive CPU turn; it completed the SAC seeds but was stopped before TD3.
The balanced all-scenario result below uses two seeds and 256 timesteps per
learned policy.

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

| Training algo | Controller | Scenario | Seeds | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sac | sac | stable | 2 | 0.309 | 0.041 | 0.937 | -2.838 | 0 |
| sac | sac | near_boundary | 2 | 0.357 | 0.034 | 0.959 | -4.360 | 0 |
| sac | sac | onset | 2 | 0.422 | 0.058 | 0.951 | -6.025 | 0 |
| sac | sac | unstable | 2 | 0.400 | 0.158 | 0.897 | -5.988 | 0 |
| td3 | td3 | stable | 2 | 0.279 | 0.029 | 0.864 | -1.939 | 0 |
| td3 | td3 | near_boundary | 2 | 0.358 | 0.069 | 0.868 | -4.275 | 0 |
| td3 | td3 | onset | 2 | 0.385 | 0.014 | 0.876 | -5.125 | 0 |
| td3 | td3 | unstable | 2 | 0.452 | 0.149 | 0.869 | -7.173 | 0 |
| sac | mpc | unstable | 2 | 0.479 | 0.130 | 1.081 | -6.994 | 0 |
| sac | sld | unstable | 2 | 0.477 | 0.135 | 1.039 | -7.197 | 0 |

Interpretation: SAC gives the lowest unstable risk in this short run but gives
up MRR and performs poorly on onset. TD3 is stronger on stable/onset reward but
also sacrifices MRR. The baselines preserve more MRR, especially MPC, so longer
training or reward shaping must prove that the learned controllers can recover
productivity while keeping the lower-risk behavior.

## Reward Ablation Snapshot

Two short CPU ablations used the same setup as
`results/rl_multiseed_cpu_short_randomized_demo`:

1. `results/rl_multiseed_cpu_mrr_reward_ablation_demo`: switch productivity
   from feed override to MRR proxy.
2. `results/rl_multiseed_cpu_mrr2_reward_ablation_demo`: use MRR proxy with
   `--productivity-weight 2.0`.

Selected learned-policy results:

| Profile | Controller | Scenario | Mean risk | Relative MRR | Mean reward | Shield rejects |
|---|---|---|---:|---:|---:|---:|
| feed default | sac | stable | 0.309 | 0.937 | -2.838 | 0 |
| feed default | sac | unstable | 0.400 | 0.897 | -5.988 | 0 |
| feed default | td3 | stable | 0.279 | 0.864 | -1.939 | 0 |
| feed default | td3 | unstable | 0.452 | 0.869 | -7.173 | 0 |
| mrr weight 1 | sac | stable | 0.302 | 0.908 | -2.805 | 0 |
| mrr weight 1 | sac | unstable | 0.420 | 0.918 | -6.453 | 0 |
| mrr weight 1 | td3 | stable | 0.279 | 0.865 | -2.451 | 0 |
| mrr weight 1 | td3 | unstable | 0.458 | 0.864 | -7.827 | 0 |
| mrr weight 2 | sac | stable | 0.300 | 0.905 | 2.671 | 0 |
| mrr weight 2 | sac | unstable | 0.419 | 0.845 | -1.833 | 0 |
| mrr weight 2 | td3 | stable | 0.284 | 0.954 | 3.688 | 0 |
| mrr weight 2 | td3 | unstable | 0.465 | 0.954 | -1.774 | 0 |

Interpretation: changing the productivity proxy alone does not recover MRR.
Increasing MRR weight helps TD3 recover productivity, but it does not improve
unstable-risk behavior. SAC remains the lower-risk learned policy on unstable
cuts, while MPC/SLD still preserve more MRR. The next useful ablation is to
separate horizon-risk and smoothness penalties to see why onset handling is
weak.

## Horizon And Smoothness Ablations

Two follow-up ablations used the stronger MRR reward profile
(`--productivity-mode mrr --productivity-weight 2.0`) as the base:

1. `results/rl_multiseed_cpu_mrr2_horizon3_ablation_demo`: increase
   `--risk-horizon-weight` from `1.5` to `3.0`.
2. `results/rl_multiseed_cpu_mrr2_smooth01_ablation_demo`: reduce
   `--smoothness-weight` from `0.5` to `0.1`.

Selected learned-policy results:

| Profile | Controller | Scenario | Mean risk | Relative MRR | Mean reward | Shield rejects |
|---|---|---|---:|---:|---:|---:|
| mrr weight 2 | sac | onset | 0.406 | 0.895 | -0.633 | 0 |
| mrr weight 2 | sac | unstable | 0.419 | 0.845 | -1.833 | 0 |
| mrr weight 2 | td3 | onset | 0.404 | 0.954 | 0.182 | 0 |
| mrr weight 2 | td3 | unstable | 0.465 | 0.954 | -1.774 | 0 |
| horizon weight 3 | sac | onset | 0.423 | 0.915 | -4.868 | 0 |
| horizon weight 3 | sac | unstable | 0.421 | 0.866 | -5.963 | 0 |
| horizon weight 3 | td3 | onset | 0.447 | 1.032 | -4.655 | 0 |
| horizon weight 3 | td3 | unstable | 0.451 | 0.984 | -5.775 | 0 |
| smoothness 0.1 | sac | onset | 0.416 | 0.932 | -0.353 | 0 |
| smoothness 0.1 | sac | unstable | 0.412 | 0.865 | -1.310 | 0 |
| smoothness 0.1 | td3 | onset | 0.403 | 0.954 | 0.244 | 0 |
| smoothness 0.1 | td3 | unstable | 0.465 | 0.954 | -1.714 | 0 |

Interpretation: higher horizon-risk weight does not improve onset behavior in
these short runs; TD3 recovers more MRR but with worse onset risk. Lower
smoothness penalty is also not the main unlock, and SAC produced shield
rejections on near-boundary cuts under that profile. The remaining likely
bottlenecks are action-space/guard behavior and insufficient training horizon,
not just scalar reward weights.

## Action Diagnostics

The diagnostic run below repeats the low-smoothness MRR profile and writes
per-step action traces plus top-level aggregate diagnostics:

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

Selected action diagnostics:

| Controller | Scenario | Steps | Guard fallbacks | Shield rejects | Current unc | Candidate unc | Raw feed | Shield feed | Raw spindle | Shield spindle | Guard reasons | Shield reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| sac | near_boundary | 24 | 2 | 5 | 0.585 | 0.583 | 0.915 | 0.930 | 1.002 | 1.002 | high_uncertainty:2 | feed_rate_limited:4;high_uncertainty:5 |
| sac | onset | 24 | 6 | 0 | 0.549 | 0.555 | 0.900 | 0.938 | 0.992 | 0.994 | high_uncertainty:6 | feed_rate_limited:4 |
| sac | unstable | 24 | 0 | 0 | 0.537 | 0.537 | 0.865 | 0.889 | 0.970 | 0.972 | spindle_clipped:6 | feed_rate_limited:8;spindle_clipped:6;spindle_rate_limited:1 |
| td3 | near_boundary | 24 | 0 | 0 | 0.574 | 0.571 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:18 | spindle_clipped:18;spindle_rate_limited:4 |
| td3 | onset | 24 | 0 | 0 | 0.517 | 0.505 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:12 | spindle_clipped:12;spindle_rate_limited:4 |
| td3 | unstable | 24 | 0 | 0 | 0.543 | 0.544 | 1.050 | 1.050 | 0.900 | 0.908 | spindle_clipped:24 | spindle_clipped:24;spindle_rate_limited:4 |

Interpretation: SAC's near-boundary shield rejections are not mainly raw-action
out-of-range errors. The candidate guard falls back only twice, while the final
shield rejects five steps because the current/candidate uncertainty is close to
the configured limit. TD3 is easier for the shield to accept, but it has learned
a blunt saturated posture: max feed and min spindle, with repeated spindle
clipping/rate limiting. The next design change should focus on action-space and
guard semantics, not more scalar reward tuning.

## Delta Action Mode

`--action-mode delta` changes SAC/TD3 from proposing absolute feed/spindle
override targets to proposing bounded changes from the current override. The
delta action components are scaled by the shield's configured feed/spindle rate
limits before candidate-guard and shield evaluation.

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

Selected comparison against the absolute-action diagnostic profile:

| Action mode | Controller | Scenario | Mean risk | Relative MRR | Shield rejects | Guard fallbacks |
|---|---|---|---:|---:|---:|---:|
| absolute | sac | near_boundary | 0.350 | 0.931 | 5 | 2 |
| delta | sac | near_boundary | 0.369 | 0.990 | 0 | 0 |
| absolute | sac | unstable | 0.412 | 0.865 | 0 | 0 |
| delta | sac | unstable | 0.400 | 1.017 | 0 | 0 |
| absolute | td3 | near_boundary | 0.364 | 0.954 | 0 | 0 |
| delta | td3 | near_boundary | 0.476 | 1.093 | 0 | 0 |
| absolute | td3 | unstable | 0.465 | 0.954 | 0 | 0 |
| delta | td3 | unstable | 0.431 | 1.080 | 0 | 6 |

Interpretation: delta actions are a strong improvement for SAC in this short
run: they remove near-boundary shield rejections and recover MRR without losing
the lower unstable-risk behavior. TD3 still saturates, but now it pushes toward
high feed and high spindle instead of high feed and low spindle, so it trades
more productivity for more risk. The next serious CPU run should use SAC with
delta actions, more timesteps, and multiple seeds before spending GPU time.

## Longer SAC Delta CPU Run

The first longer CPU run focuses on SAC only, because the shorter delta-action
comparison showed SAC was the learned controller worth scaling before GPU work.

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

| Controller | Scenario | Seeds | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |
|---|---|---:|---:|---:|---:|---:|---:|
| sac | stable | 3 | 0.326 | 0.004 | 1.003 | 4.200 | 0 |
| sac | near_boundary | 3 | 0.369 | 0.044 | 1.017 | 2.030 | 8 |
| sac | onset | 3 | 0.400 | 0.016 | 0.995 | 1.269 | 0 |
| sac | unstable | 3 | 0.477 | 0.012 | 1.028 | -1.710 | 0 |
| mpc | unstable | 3 | 0.499 | 0.033 | 1.074 | -1.150 | 0 |
| sld | unstable | 3 | 0.481 | 0.019 | 1.021 | -1.534 | 0 |

Selected action diagnostics:

| Scenario | Steps | Guard fallbacks | Shield rejects | Current unc | Candidate unc | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable | 96 | 7 | 0 | 0.542 | 0.545 | 0.981 | 0.981 | 1.023 | 1.023 | spindle_clipped:1 |
| near_boundary | 96 | 8 | 8 | 0.564 | 0.564 | 0.988 | 0.986 | 1.035 | 1.031 | high_uncertainty:8;spindle_clipped:17 |
| onset | 96 | 3 | 0 | 0.539 | 0.538 | 0.980 | 0.980 | 1.017 | 1.017 | spindle_clipped:1 |
| unstable | 96 | 0 | 0 | 0.518 | 0.518 | 0.999 | 0.991 | 1.050 | 1.039 | spindle_clipped:33 |

Interpretation: longer SAC-delta training preserves the useful pattern: MRR is
around or above 1.0, stable risk is controlled, and unstable risk is slightly
better than MPC and comparable to SLD. The near-boundary shield rejects are not
unique to SAC in this run; matched SLD/MPC also hit eight near-boundary rejects,
so this looks like a randomized uncertainty edge near the calibrated margin
limit. The next CPU step is to make uncertainty handling more graded than a
hard reject, or to split near-boundary evaluation into strict and advisory
shield modes.

## Uncertainty Hold Shield Mode

The shield now supports `--uncertainty-mode hold`. The default remains strict:
`reject` still treats high calibration uncertainty as a rejected action. Hold
mode keeps hard failures strict, including sensor/controller/not-in-cut/unknown
state failures, but high uncertainty alone returns the previous feed/spindle
override as an accepted conservative hold. The result still records
`high_uncertainty` and `uncertainty_hold`, so the event is visible in action
diagnostics without being counted as a shield violation.

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

Interpretation: hold mode fixes the previous accounting bottleneck without
silencing uncertainty. In the strict longer SAC run, near-boundary SAC had 8
shield rejections, mean risk `0.369`, MRR `1.017`, and reward `2.030`. With
hold mode, near-boundary SAC has 0 shield rejections, lower mean risk `0.363`,
MRR `0.990`, and reward `2.309`. Matched SLD/MPC near-boundary rejects also
drop to zero, confirming that the earlier failures were a calibrated-margin
uncertainty edge rather than a SAC-only action pathology.

## TD3 Hold-Mode And Comparison Report

The same hold-mode setup was run for TD3, then combined with the SAC strict and
SAC hold runs using the reusable comparison command:

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

| Scenario | Guard fallbacks | Shield rejects | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |
|---|---:|---:|---:|---:|---:|---:|---|
| near_boundary | 8 | 0 | 1.090 | 1.046 | 0.972 | 0.977 | high_uncertainty:8;spindle_clipped:72;spindle_rate_limited:7;uncertainty_hold:8 |
| onset | 0 | 0 | 1.094 | 1.050 | 0.957 | 0.969 | spindle_clipped:75;spindle_rate_limited:4 |
| stable | 0 | 0 | 1.094 | 1.050 | 0.960 | 0.969 | spindle_clipped:77;spindle_rate_limited:6 |
| unstable | 7 | 0 | 1.094 | 1.050 | 1.003 | 1.001 | spindle_clipped:62;spindle_rate_limited:9 |

Artifacts:

| Check | Output |
|---|---|
| TD3 hold-mode run | `results/rl_td3_delta_uncertainty_hold_cpu_long_demo` |
| Strict/hold comparison | `results/rl_strict_hold_delta_comparison_demo` |
| Comparison report | `results/rl_strict_hold_delta_comparison_demo/report.md` |

Interpretation: TD3 hold mode removes shield rejections and gives the best
unstable reward in this CPU comparison, but it is still visibly saturating the
feed action. The shield clips TD3 raw feed from about `1.09` to the `1.05`
limit across all scenarios, with heavy spindle clipping too. SAC hold remains
the cleaner policy posture near the boundary, while TD3 is a useful high-MRR
candidate that needs either stronger action regularization or narrower action
bounds before it should be trusted.

## TD3 Saturation Penalty Ablation

To test whether TD3's clipped posture can be discouraged through reward
shaping, this run adds per-component penalties for candidate-guard/shield
clipping and shield rate limiting.

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

Interpretation: the blunt clipping penalty is too strong by itself. It reduces
raw feed saturation, but TD3 shifts into low-feed/high-spindle behavior,
increases risk, loses MRR, and collapses reward. Keep the knobs because they
are useful for controlled ablations, but the next better TD3 fix is likely
configurable/narrower action bounds or a smaller asymmetric penalty, not the
`0.25` clip penalty profile.

## TD3 Narrow Delta Action Scale

The cleaner TD3 saturation fix is to narrow the policy's normalized delta
action range instead of punishing clipping after the fact. This keeps the same
machine shield limits, but with `--delta-action-scale 0.5`, each policy
decision can request only half of the normal per-step feed/spindle override
change.

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

Interpretation: this is better than the blunt clipping penalty. Narrow delta
actions reduce raw feed saturation from about `1.09` to `1.06`, cut spindle
clipping from roughly `62-77` events to `44-48`, preserve near-boundary reward,
and improve unstable risk from `0.463` to `0.449`. MRR drops on unstable from
`1.051` to `1.019`, but that is a reasonable tradeoff for a less violent
policy. Feed clipping still remains, so the next CPU ablation should test
`--delta-action-scale 0.35` or separate feed/spindle delta scales.

## TD3 Split Delta Scale Sweep

This sweep tests whether feed and spindle should have separate delta ranges.
The profiles compare global `0.5` against feed-specific narrowing while keeping
spindle at `0.5`, then narrowing both feed and spindle.

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
  --delta-feed-scale 0.25 \
  --delta-spindle-scale 0.35 \
  --uncertainty-mode hold \
  --productivity-mode mrr \
  --productivity-weight 2.0 \
  --smoothness-weight 0.1 \
  --margin-calibration results/margin_calibration_context_family_holdout_demo/calibration.json \
  --out results/rl_td3_delta_feed025_spindle035_hold_cpu_long_demo

rtk uv run chatter-twin compare-rl-runs \
  --run td3_delta05=results/rl_td3_delta05_hold_cpu_long_demo \
  --run td3_feed035_spindle05=results/rl_td3_delta_feed035_spindle05_hold_cpu_long_demo \
  --run td3_feed025_spindle05=results/rl_td3_delta_feed025_spindle05_hold_cpu_long_demo \
  --run td3_feed025_spindle035=results/rl_td3_delta_feed025_spindle035_hold_cpu_long_demo \
  --baseline-label td3_delta05 \
  --out results/rl_td3_split_delta_scale_sweep_demo
```

Selected comparison:

| Profile | Scenario | Mean risk | Relative MRR | Mean reward | Raw feed | Shield feed | Raw spindle | Shield spindle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| td3_delta05 | unstable | 0.449 | 1.019 | -0.908 | 1.066 | 1.047 | 0.969 | 0.973 |
| td3_feed035_spindle05 | unstable | 0.452 | 0.999 | -1.310 | 1.055 | 1.044 | 0.950 | 0.957 |
| td3_feed025_spindle05 | unstable | 0.470 | 0.956 | -2.621 | 1.047 | 1.041 | 0.906 | 0.919 |
| td3_feed025_spindle035 | unstable | 0.475 | 1.057 | -1.148 | 1.047 | 1.041 | 1.017 | 1.016 |
| td3_delta05 | near_boundary | 0.373 | 1.022 | 2.451 | 1.062 | 1.043 | 0.974 | 0.980 |
| td3_feed025_spindle05 | near_boundary | 0.349 | 0.960 | 2.381 | 1.044 | 1.037 | 0.912 | 0.926 |

Artifacts:

| Check | Output |
|---|---|
| Feed 0.35 / spindle 0.5 run | `results/rl_td3_delta_feed035_spindle05_hold_cpu_long_demo` |
| Feed 0.25 / spindle 0.5 run | `results/rl_td3_delta_feed025_spindle05_hold_cpu_long_demo` |
| Feed 0.25 / spindle 0.35 run | `results/rl_td3_delta_feed025_spindle035_hold_cpu_long_demo` |
| Split-scale sweep comparison | `results/rl_td3_split_delta_scale_sweep_demo` |
| Comparison report | `results/rl_td3_split_delta_scale_sweep_demo/report.md` |

Interpretation: global delta `0.5` remains the best TD3 tradeoff. Feed
`0.25` reduces raw feed overshoot and improves stable/near-boundary risk, but
it gives up MRR and worsens unstable behavior. Spindle `0.35` reduces spindle
clipping counts, but it does not improve risk. The remaining feed clipping is
less a per-step scale issue and more a headroom issue: repeated positive
deltas walk the override into the feed ceiling.

## TD3 Headroom Delta Mapping

This run tests a structural alternative to shrinking the action range:
`--delta-mapping headroom`. In fixed mapping, a positive delta near the override
ceiling can still ask for an unreachable value and depend on candidate/shield
clipping. In headroom mapping, the same normalized action consumes only the
remaining safe feed/spindle override headroom.

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

Interpretation: headroom mapping successfully removes most of the feed
overshoot and reduces spindle clipping, and it improves near-boundary risk and
reward. It is not the best overall TD3 profile yet because the unstable case
loses MRR and reward. Keep it as a safer candidate to tune, but the current
best CPU TD3 tradeoff remains fixed delta mode with `--delta-action-scale 0.5`.

## TD3 Headroom MRR-Weight Ablation

Because the first headroom profile was mechanically safer but too conservative
on unstable cuts, this ablation keeps `--delta-mapping headroom` and raises
`--productivity-weight` from `2.0` to `3.0`.

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

Interpretation: MRR weight `3.0` recovers productivity under headroom mapping
and avoids the very low spindle behavior from the MRR 2 headroom run. It pays
for that with higher risk in stable/near-boundary/onset cases and still does
not beat fixed delta `0.5` on unstable risk. Since reward weights differ, use
risk/MRR/actions for this comparison; raw reward is not normalized across these
profiles.

## TD3 MRR 3 Fixed-vs-Headroom Mapping

This isolates mapping choice at the same productivity reward weight:

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

Interpretation: at equal MRR weight `3.0`, headroom mapping improves unstable
MRR and unstable risk, but it worsens stable/near-boundary/onset risk. Fixed
mapping is more conservative on average at this reward weight; headroom is the
more productivity-oriented mapping. This gives two clear CPU candidates for
later longer training: fixed delta `0.5` with MRR 2 as the balanced baseline,
and headroom delta `0.5` with MRR 3 as the productivity-oriented candidate.

## TD3 Medium Validation

This validation promotes the two short-run survivors to a larger CPU run: four
seeds, `2000` TD3 timesteps per seed, `6` evaluation episodes per scenario, and
`10` control steps per episode. It is still CPU-sized, but it is more useful
than the earlier 1000-step ablations for deciding which profiles deserve a
larger stress test.

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

Interpretation: both candidates remain non-dominated. The fixed MRR 2 profile
has lower average risk and fewer guard fallbacks. The headroom MRR 3 profile
buys about `+0.06` average MRR and a much higher unstable MRR, but it does not
lower risk in this medium run. The next useful CPU step is therefore not more
basic training; it is a stress test over harsher randomization and drift to see
which candidate fails more gracefully.

## Saved-Policy Harsh Stress Evaluation

This step adds `chatter-twin eval-rl-run`, which loads saved SAC/TD3
`model.zip` artifacts and evaluates the already-trained policies under a new
randomization profile. This is stricter than retraining under the stress
distribution because it tests whether the promoted medium-run policies
generalize.

The harsh stress profile widens process and sensing uncertainty:

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
| Fixed-policy harsh eval | `results/rl_td3_fixed_mrr2_harsh_stress_eval` |
| Headroom-policy harsh eval | `results/rl_td3_headroom_mrr3_harsh_stress_eval` |
| Harsh stress comparison | `results/rl_td3_harsh_stress_comparison` |
| Candidate summary | `results/rl_td3_harsh_stress_comparison/candidate_summary.csv` |
| Comparison report | `results/rl_td3_harsh_stress_comparison/report.md` |

Interpretation: harsh stress keeps both candidates non-dominated. Fixed MRR 2
is the safer stress candidate: lower average risk, lower worst risk, and lower
unstable risk. Headroom MRR 3 remains the productivity candidate: roughly
`+0.046` average MRR and `+0.052` unstable MRR, with slightly fewer guard
fallbacks, but at higher risk. This means the current champion for safety-first
closed-loop experiments should be fixed MRR 2; headroom MRR 3 is useful as a
high-productivity comparator.

## Shadow Champion Selection

This step turns the fixed-MRR2 harsh stress evaluation into a shadow-only
policy selection artifact. The selector ranks per-seed checkpoints using a
safety-first score that includes average risk, worst-scenario risk, unstable
risk, guard fallback fraction, shield rejections, and a minimum-MRR shortfall
penalty. The MRR shortfall term is important: without it, a policy can look
safer by falling below the minimum economic MRR target.

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

Interpretation: seed `818` had the lowest average risk and zero guard
fallbacks, but it fell below the minimum MRR target, so it is not the
safety-first champion. Seed `616` is selected because it keeps MRR above target
while retaining the best selection score among the non-starving candidates.
The policy card is explicitly shadow-only; it is not hardware actuation
approval.

## Shadow Champion Replay

This step loads the selected seed-616 TD3 checkpoint and converts its greedy
policy actions into a reviewable shadow recommendation trace. The output keeps
the same deployment boundary as the policy card: shadow-only, CNC writes
disabled, safety shield required, and human review required.

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

Shadow replay metrics:

| Metric | Value |
|---|---:|
| Recommendation windows | 160 |
| Action fraction | 0.463 |
| Guard fallback fraction | 0.100 |
| Shield rejections | 0 |
| Mean risk | 0.507 |
| Max risk | 0.746 |
| Mean feed override | 1.046 |
| Mean spindle override | 1.080 |
| Relative MRR proxy | 1.130 |

Scenario-level replay:

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

Interpretation: this is the first concrete learned-policy recommendation
artifact from the selected RL champion. It preserves MRR above the target and
does not trigger shield rejections in this replay, but the mean/max risk are
still stress-test values. Treat it as a review artifact for shadow mode, not as
permission to write CNC overrides.

## Shadow Replay Gate

This step adds explicit acceptance gates for a replay artifact. The gate is not
a hardware approval: every profile still records
`hardware_actuation_allowed=false`. Profiles form a promotion ladder:
`shadow_review` for offline review, `live_shadow` for live recommendation-only
monitoring, and `hardware_actuation` as a deliberately strict review gate before
any CNC-write discussion.

```bash
rtk uv run chatter-twin gate-rl-shadow \
  --profile shadow_review \
  --shadow-dir results/rl_td3_fixed_mrr2_shadow_replay_demo \
  --out results/rl_td3_fixed_mrr2_shadow_gate_demo
```

Profile thresholds:

| Gate | Shadow review | Live shadow | Hardware actuation |
|---|---:|---:|---:|
| Minimum recommendation windows | 1 | 300 | 1000 |
| Maximum mean risk | 0.550 | 0.450 | 0.300 |
| Maximum max risk | 0.800 | 0.700 | 0.500 |
| Maximum unstable mean risk | 0.600 | 0.500 | 0.350 |
| Minimum relative MRR proxy | 1.000 | 1.000 | 1.000 |
| Maximum guard fallback fraction | 0.200 | 0.100 | 0.020 |
| Maximum action fraction | 0.800 | 0.650 | 0.500 |
| Maximum shield rejections | 0 | 0 | 0 |
| Real-machine data required | no | yes | yes |
| Operator approval evidence required | no | yes | yes |
| Hardware interlock evidence required | no | no | yes |

Profile results on the selected TD3 replay:

| Profile | Status | Promotion level | Failed checks |
|---|---|---|---|
| `shadow_review` | pass | `shadow_review_candidate` | none |
| `live_shadow` | blocked | `do_not_promote` | windows, mean risk, max risk, unstable risk, real-machine data, operator approval |
| `hardware_actuation` | blocked | `do_not_promote` | windows, mean risk, max risk, unstable risk, guard fallback fraction, real-machine data, operator approval, hardware interlock |

Shadow-review check details:

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

Artifacts:

| Check | Output |
|---|---|
| Shadow-review gate | `results/rl_td3_fixed_mrr2_shadow_gate_demo` |
| Live-shadow gate | `results/rl_td3_fixed_mrr2_live_shadow_gate_demo` |
| Hardware-actuation gate | `results/rl_td3_fixed_mrr2_hardware_actuation_gate_demo` |

Interpretation: the selected TD3 replay now passes the current shadow-review
gate with no failed checks, while the stricter profiles block it. That is the
right current behavior: the artifact is reviewable, but it is synthetic and
not eligible for live-machine or hardware-actuation promotion.
