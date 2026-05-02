# Replay Window Schema

This schema is the handoff point between synthetic simulation data now and real
DAQ/MTConnect data later.

## `dataset.npz`

Required arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `sensor_windows` | `(n_windows, window_samples, n_channels)` | Windowed high-rate sensor samples. Current synthetic channels are `accel_x`, `accel_y`. |
| `labels` | `(n_windows,)` | String label: `stable`, `transition`, `slight`, `severe`, or `unknown`. |
| `label_ids` | `(n_windows,)` | Integer label ids: stable `0`, transition `1`, slight `2`, severe `3`, unknown `4`. |
| `horizon_labels` | `(n_windows,)` | Worst current-or-future label within the configured horizon. |
| `horizon_label_ids` | `(n_windows,)` | Integer ids for `horizon_labels`. |
| `horizon_s` | `(1,)` | Horizon duration used to compute horizon/onset labels. |
| `chatter_within_horizon` | `(n_windows,)` | True when current or future windows within the horizon include `slight` or `severe`. |
| `future_chatter_within_horizon` | `(n_windows,)` | True when a future window, excluding the current one, reaches `slight` or `severe` within the horizon. |
| `time_to_chatter_s` | `(n_windows,)` | Time until first current/future `slight` or `severe` label within the horizon, or `-1` if none. |
| `risk_chatter_now` | `(n_windows,)` | Hybrid risk estimate for the current window. |
| `risk_chatter_horizon` | `(n_windows,)` | Short-horizon risk estimate. |
| `margin_physics` | `(n_windows,)` | Signed stability margin from the physics model. |
| `uncertainty` | `(n_windows,)` | Risk estimate uncertainty. |
| `scenarios` | `(n_windows,)` | Source scenario name. |
| `episodes` | `(n_windows,)` | Source episode index. |
| `window_index_in_episode` | `(n_windows,)` | Zero-based replay-window index inside each scenario/episode cut. |
| `randomized` | `(n_windows,)` | Boolean marker for domain-randomized synthetic episodes. Real imported data can set this to false. |
| `spindle_scale` | `(n_windows,)` | Synthetic randomization multiplier for spindle speed. Defaults to `1.0` when disabled. |
| `feed_scale` | `(n_windows,)` | Synthetic randomization multiplier for feed per tooth. Defaults to `1.0` when disabled. |
| `axial_depth_scale` | `(n_windows,)` | Synthetic randomization multiplier for axial depth. Defaults to `1.0` when disabled. |
| `radial_depth_scale` | `(n_windows,)` | Synthetic randomization multiplier for radial depth. Defaults to `1.0` when disabled. |
| `stiffness_scale` | `(n_windows,)` | Synthetic randomization multiplier for modal stiffness. Defaults to `1.0` when disabled. |
| `damping_scale` | `(n_windows,)` | Synthetic randomization multiplier for modal damping. Defaults to `1.0` when disabled. |
| `cutting_coeff_scale` | `(n_windows,)` | Synthetic randomization multiplier for cutting coefficients. Defaults to `1.0` when disabled. |
| `noise_scale` | `(n_windows,)` | Synthetic randomization multiplier for sensor noise. Defaults to `1.0` when disabled. |
| `start_time_s` | `(n_windows,)` | Window start time in seconds. |
| `episode_progress` | `(n_windows,)` | Window start time normalized by simulated cut duration. |
| `axial_depth_profile_scale` | `(n_windows,)` | Effective axial-depth scale for the current window relative to the scenario/randomized base cut. |
| `cutting_coeff_profile_scale` | `(n_windows,)` | Effective cutting-coefficient scale for the current window relative to the scenario/randomized base cut. |
| `cutting_coeff_t_n_m2` | `(n_windows,)` | Effective tangential cutting coefficient for the current window. |
| `cutting_coeff_r_n_m2` | `(n_windows,)` | Effective radial cutting coefficient for the current window. |
| `dominant_frequency_delta_hz` | `(n_windows,)` | Change in dominant frequency from the previous window in the same episode. |
| `rms_delta` | `(n_windows,)` | Change in RMS from the previous window in the same episode. |
| `rms_growth_rate` | `(n_windows,)` | `rms_delta` divided by elapsed time between window starts. |
| `chatter_band_energy_delta` | `(n_windows,)` | Change in chatter-band energy from the previous window. |
| `chatter_band_energy_growth_rate` | `(n_windows,)` | Chatter-band energy growth per second between window starts. |
| `tooth_band_energy_delta` | `(n_windows,)` | Change in tooth-band energy from the previous window. |
| `tooth_band_energy_growth_rate` | `(n_windows,)` | Tooth-band energy growth per second between window starts. |
| `non_tooth_harmonic_ratio_delta` | `(n_windows,)` | Change in non-tooth-harmonic ratio from the previous window. |
| `non_tooth_harmonic_ratio_growth_rate` | `(n_windows,)` | Non-tooth-harmonic ratio growth per second between window starts. |
| `rms_ewma` | `(n_windows,)` | Exponentially weighted moving average of RMS within the episode. |
| `chatter_band_energy_ewma` | `(n_windows,)` | Exponentially weighted moving average of chatter-band energy. |
| `non_tooth_harmonic_ratio_ewma` | `(n_windows,)` | Exponentially weighted moving average of non-tooth-harmonic ratio. |
| `sample_rate_hz` | `(1,)` | Sensor window sample rate. |
| `channel_names` | `(n_channels,)` | Channel name strings. |

## `windows.csv`

One row per window. This file repeats the label/risk targets and adds machining
context such as spindle speed, feed per tooth, axial/radial depth, tooth
frequency, limiting chatter frequency, RMS, band-energy features, temporal
growth/EMA features, process-profile scales, and synthetic
domain-randomization scales.

## Horizon Labels

Schema `chatter-window-v4` adds horizon/onset labels for early-warning training.
The current label still describes the present window. The horizon label describes
the worst state from the current window through the configured future horizon.
For example:

```bash
rtk uv run chatter-twin export-synthetic \
  --randomize \
  --focus-transitions \
  --horizon 0.2 \
  --out results/synthetic_transition_focus_demo
```

Use `train-risk --target horizon` to train against this current-or-upcoming
target. `future_chatter_within_horizon` is useful for post-hoc lead-time
analysis because it excludes chatter already present in the current window.

## Progressive Onset Profiles

Schema `chatter-window-v5` adds per-window process-profile columns. The
simulator can now ramp axial depth and cutting coefficients within an episode,
and replay slicing computes each window's physics margin from that effective
cut state instead of using one constant episode-level depth. This is mainly for
early-warning experiments where stable/transition windows should precede
`slight` or `severe` chatter by a measurable lead time.

Use the `onset` scenario to generate these progressive cuts:

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

`future_chatter_within_horizon` should now contain rows where the current label
is still `stable` or `transition`, which are the replay windows needed for true
incipient-chatter training.

## Domain Randomization

Synthetic export can perturb spindle speed, feed, axial/radial depth, modal
stiffness, modal damping, cutting coefficients, and sensor noise per episode:

```bash
rtk uv run chatter-twin export-synthetic --randomize --out results/synthetic_randomized_demo
```

The applied scales are stored per window so training and validation reports can
audit whether a result came from nominal synthetic data or randomized synthetic
data. The manifest stores the active randomization ranges under
`domain_randomization`.

## Transition-Focused Export

Synthetic export can also search several randomized candidates for each
requested episode and keep the candidate with the most transition windows:

```bash
rtk uv run chatter-twin export-synthetic \
  --randomize \
  --focus-transitions \
  --transition-candidates 12 \
  --min-transition-windows 3 \
  --out results/synthetic_transition_focus_demo
```

This is useful for building boundary/onset datasets instead of waiting for
broad domain randomization to accidentally sample the chatter boundary. The
manifest records this under `sampling_strategy.transition_focus`.

## Real Data Contract

Future hardware log importers should write the same `dataset.npz` and
`windows.csv` shapes. Real datasets can add extra channels, but must preserve
the required arrays above so the risk estimator, benchmark tools, and training
code can consume synthetic and real replay windows interchangeably.
