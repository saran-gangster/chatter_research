from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chatter_twin.dynamics import simulate_milling
from chatter_twin.features import extract_signal_features
from chatter_twin.models import ChatterLabel, CutConfig, ModalParams, SimulationConfig, SimulationResult
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.scenarios import make_scenario
from chatter_twin.stability import estimate_stability

LABEL_TO_ID: dict[str, int] = {
    "stable": 0,
    "transition": 1,
    "slight": 2,
    "severe": 3,
    "unknown": 4,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CHANNEL_NAMES = ("accel_x", "accel_y")
CHATTER_POSITIVE_LABELS = frozenset({"slight", "severe"})
LABEL_SEVERITY = {
    "unknown": -1,
    "stable": 0,
    "transition": 1,
    "slight": 2,
    "severe": 3,
}


@dataclass(frozen=True)
class WindowSpec:
    window_s: float = 0.10
    stride_s: float = 0.05

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be positive")


@dataclass(frozen=True)
class DomainRandomizationConfig:
    enabled: bool = False
    spindle_scale: tuple[float, float] = (0.85, 1.15)
    feed_scale: tuple[float, float] = (0.70, 1.30)
    axial_depth_scale: tuple[float, float] = (0.60, 1.80)
    radial_depth_scale: tuple[float, float] = (0.70, 1.30)
    stiffness_scale: tuple[float, float] = (0.80, 1.20)
    damping_scale: tuple[float, float] = (0.70, 1.35)
    cutting_coeff_scale: tuple[float, float] = (0.75, 1.35)
    noise_scale: tuple[float, float] = (0.50, 2.00)

    def __post_init__(self) -> None:
        for name, bounds in (
            ("spindle_scale", self.spindle_scale),
            ("feed_scale", self.feed_scale),
            ("axial_depth_scale", self.axial_depth_scale),
            ("radial_depth_scale", self.radial_depth_scale),
            ("stiffness_scale", self.stiffness_scale),
            ("damping_scale", self.damping_scale),
            ("cutting_coeff_scale", self.cutting_coeff_scale),
            ("noise_scale", self.noise_scale),
        ):
            _validate_scale_range(name, bounds)


@dataclass(frozen=True)
class TransitionFocusConfig:
    enabled: bool = False
    candidates_per_episode: int = 8
    min_transition_windows: int = 1

    def __post_init__(self) -> None:
        if self.candidates_per_episode < 1:
            raise ValueError("candidates_per_episode must be at least 1")
        if self.min_transition_windows < 0:
            raise ValueError("min_transition_windows cannot be negative")


@dataclass(frozen=True)
class HorizonConfig:
    horizon_s: float = 0.20

    def __post_init__(self) -> None:
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")


@dataclass(frozen=True)
class WindowRecord:
    window_id: int
    scenario: str
    episode: int
    window_index_in_episode: int
    randomized: bool
    spindle_scale: float
    feed_scale: float
    axial_depth_scale: float
    radial_depth_scale: float
    stiffness_scale: float
    damping_scale: float
    cutting_coeff_scale: float
    noise_scale: float
    start_time_s: float
    end_time_s: float
    episode_progress: float
    label: ChatterLabel
    label_id: int
    horizon_label: ChatterLabel
    horizon_label_id: int
    horizon_s: float
    chatter_within_horizon: bool
    future_chatter_within_horizon: bool
    time_to_chatter_s: float
    risk_chatter_now: float
    risk_chatter_horizon: float
    margin_physics: float
    margin_signal: float
    uncertainty: float
    spindle_rpm: float
    feed_per_tooth_m: float
    axial_depth_m: float
    radial_depth_m: float
    axial_depth_profile_scale: float
    cutting_coeff_profile_scale: float
    cutting_coeff_t_n_m2: float
    cutting_coeff_r_n_m2: float
    tooth_frequency_hz: float
    limiting_frequency_hz: float
    dominant_frequency_hz: float
    rms: float
    chatter_band_energy: float
    tooth_band_energy: float
    non_tooth_harmonic_ratio: float
    dominant_frequency_delta_hz: float
    rms_delta: float
    rms_growth_rate: float
    chatter_band_energy_delta: float
    chatter_band_energy_growth_rate: float
    tooth_band_energy_delta: float
    tooth_band_energy_growth_rate: float
    non_tooth_harmonic_ratio_delta: float
    non_tooth_harmonic_ratio_growth_rate: float
    rms_ewma: float
    chatter_band_energy_ewma: float
    non_tooth_harmonic_ratio_ewma: float


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    sample_rate_hz: float
    window_s: float
    stride_s: float
    channel_names: tuple[str, ...]
    scenarios: tuple[str, ...]
    episodes_per_scenario: int
    total_windows: int
    label_counts: dict[str, int]
    domain_randomization: dict[str, object]
    sampling_strategy: dict[str, object]
    horizon: dict[str, object]
    artifacts: tuple[str, ...]


def export_synthetic_dataset(
    *,
    scenarios: list[str],
    episodes: int,
    duration_s: float,
    window_spec: WindowSpec,
    out_dir: Path,
    seed: int = 202,
    randomization: DomainRandomizationConfig | None = None,
    transition_focus: TransitionFocusConfig | None = None,
    horizon: HorizonConfig | None = None,
) -> DatasetManifest:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    randomization = randomization or DomainRandomizationConfig()
    transition_focus = transition_focus or TransitionFocusConfig()
    horizon = horizon or HorizonConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []

    for scenario_idx, scenario in enumerate(scenarios):
        for episode in range(episodes):
            episode_seed = seed + scenario_idx * 10_000 + episode
            scenario_windows, scenario_records = select_episode_windows(
                scenario=scenario,
                episode=episode,
                episode_seed=episode_seed,
                duration_s=duration_s,
                window_spec=window_spec,
                starting_window_id=len(records),
                randomization=randomization,
                transition_focus=transition_focus,
                horizon=horizon,
            )
            windows.extend(scenario_windows)
            records.extend(scenario_records)

    if not records:
        raise ValueError("No windows produced; increase duration or reduce window size")

    sensor_windows = np.stack(windows).astype(np.float32)
    labels = np.array([record.label for record in records])
    label_ids = np.array([record.label_id for record in records], dtype=np.int64)
    horizon_labels = np.array([record.horizon_label for record in records])
    horizon_label_ids = np.array([record.horizon_label_id for record in records], dtype=np.int64)
    risks = np.array([record.risk_chatter_now for record in records], dtype=np.float32)
    horizons = np.array([record.risk_chatter_horizon for record in records], dtype=np.float32)
    margins = np.array([record.margin_physics for record in records], dtype=np.float32)
    uncertainties = np.array([record.uncertainty for record in records], dtype=np.float32)
    scenarios_array = np.array([record.scenario for record in records])
    episodes_array = np.array([record.episode for record in records], dtype=np.int64)
    window_indices = np.array([record.window_index_in_episode for record in records], dtype=np.int64)
    start_times = np.array([record.start_time_s for record in records], dtype=np.float64)
    episode_progress = np.array([record.episode_progress for record in records], dtype=np.float32)
    sample_rate_hz = result_sample_rate(records, sensor_windows)

    np.savez_compressed(
        out_dir / "dataset.npz",
        sensor_windows=sensor_windows,
        labels=labels,
        label_ids=label_ids,
        horizon_labels=horizon_labels,
        horizon_label_ids=horizon_label_ids,
        horizon_s=np.array([horizon.horizon_s], dtype=np.float32),
        chatter_within_horizon=np.array([record.chatter_within_horizon for record in records], dtype=np.bool_),
        future_chatter_within_horizon=np.array([record.future_chatter_within_horizon for record in records], dtype=np.bool_),
        time_to_chatter_s=np.array([record.time_to_chatter_s for record in records], dtype=np.float32),
        risk_chatter_now=risks,
        risk_chatter_horizon=horizons,
        margin_physics=margins,
        uncertainty=uncertainties,
        scenarios=scenarios_array,
        episodes=episodes_array,
        window_index_in_episode=window_indices,
        randomized=np.array([record.randomized for record in records], dtype=np.bool_),
        spindle_scale=np.array([record.spindle_scale for record in records], dtype=np.float32),
        feed_scale=np.array([record.feed_scale for record in records], dtype=np.float32),
        axial_depth_scale=np.array([record.axial_depth_scale for record in records], dtype=np.float32),
        radial_depth_scale=np.array([record.radial_depth_scale for record in records], dtype=np.float32),
        stiffness_scale=np.array([record.stiffness_scale for record in records], dtype=np.float32),
        damping_scale=np.array([record.damping_scale for record in records], dtype=np.float32),
        cutting_coeff_scale=np.array([record.cutting_coeff_scale for record in records], dtype=np.float32),
        noise_scale=np.array([record.noise_scale for record in records], dtype=np.float32),
        start_time_s=start_times,
        episode_progress=episode_progress,
        axial_depth_profile_scale=np.array([record.axial_depth_profile_scale for record in records], dtype=np.float32),
        cutting_coeff_profile_scale=np.array([record.cutting_coeff_profile_scale for record in records], dtype=np.float32),
        cutting_coeff_t_n_m2=np.array([record.cutting_coeff_t_n_m2 for record in records], dtype=np.float32),
        cutting_coeff_r_n_m2=np.array([record.cutting_coeff_r_n_m2 for record in records], dtype=np.float32),
        dominant_frequency_delta_hz=np.array([record.dominant_frequency_delta_hz for record in records], dtype=np.float32),
        rms_delta=np.array([record.rms_delta for record in records], dtype=np.float32),
        rms_growth_rate=np.array([record.rms_growth_rate for record in records], dtype=np.float32),
        chatter_band_energy_delta=np.array([record.chatter_band_energy_delta for record in records], dtype=np.float32),
        chatter_band_energy_growth_rate=np.array([record.chatter_band_energy_growth_rate for record in records], dtype=np.float32),
        tooth_band_energy_delta=np.array([record.tooth_band_energy_delta for record in records], dtype=np.float32),
        tooth_band_energy_growth_rate=np.array([record.tooth_band_energy_growth_rate for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_delta=np.array([record.non_tooth_harmonic_ratio_delta for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_growth_rate=np.array([record.non_tooth_harmonic_ratio_growth_rate for record in records], dtype=np.float32),
        rms_ewma=np.array([record.rms_ewma for record in records], dtype=np.float32),
        chatter_band_energy_ewma=np.array([record.chatter_band_energy_ewma for record in records], dtype=np.float32),
        non_tooth_harmonic_ratio_ewma=np.array([record.non_tooth_harmonic_ratio_ewma for record in records], dtype=np.float32),
        sample_rate_hz=np.array([sample_rate_hz]),
        channel_names=np.array(CHANNEL_NAMES),
    )
    _write_records_csv(out_dir / "windows.csv", records)
    manifest = _manifest(records, scenarios, episodes, window_spec, sensor_windows, randomization, transition_focus, horizon)
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(out_dir / "README.md", manifest)
    return manifest


def select_episode_windows(
    *,
    scenario: str,
    episode: int,
    episode_seed: int,
    duration_s: float,
    window_spec: WindowSpec,
    starting_window_id: int,
    randomization: DomainRandomizationConfig,
    transition_focus: TransitionFocusConfig,
    horizon: HorizonConfig,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord]]:
    candidates = transition_focus.candidates_per_episode if transition_focus.enabled else 1
    best_windows: list[NDArray[np.float64]] = []
    best_records: list[WindowRecord] = []
    best_transition_count = -1
    best_score = -1.0

    for candidate_id in range(candidates):
        candidate_seed = episode_seed + candidate_id * 1_000_003
        candidate_windows, candidate_records = simulate_episode_windows(
            scenario=scenario,
            episode=episode,
            seed=candidate_seed,
            duration_s=duration_s,
            window_spec=window_spec,
            starting_window_id=starting_window_id,
            randomization=randomization,
            horizon=horizon,
        )
        transition_count = sum(record.label == "transition" for record in candidate_records)
        score = transition_candidate_score(candidate_records)
        if transition_count > best_transition_count or (transition_count == best_transition_count and score > best_score):
            best_transition_count = transition_count
            best_score = score
            best_windows = candidate_windows
            best_records = candidate_records
        if transition_focus.enabled and transition_count >= transition_focus.min_transition_windows:
            break

    return best_windows, best_records


def simulate_episode_windows(
    *,
    scenario: str,
    episode: int,
    seed: int,
    duration_s: float,
    window_spec: WindowSpec,
    starting_window_id: int,
    randomization: DomainRandomizationConfig,
    horizon: HorizonConfig,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord]]:
    modal, tool, cut, sim_config = make_scenario(scenario)
    modal, cut, sim_config, randomization_metadata = apply_domain_randomization(
        modal=modal,
        cut=cut,
        sim_config=sim_config,
        rng=np.random.default_rng(seed),
        config=randomization,
    )
    sim_config = replace(
        sim_config,
        duration_s=duration_s,
        random_seed=seed,
    )
    result = simulate_milling(modal, tool, cut, sim_config)
    return slice_result_windows(
        result=result,
        scenario=scenario,
        episode=episode,
        window_spec=window_spec,
        starting_window_id=starting_window_id,
        randomization_metadata=randomization_metadata,
        horizon=horizon,
    )


def transition_candidate_score(records: list[WindowRecord]) -> float:
    if not records:
        return -1.0
    transition_bonus = 5.0 * sum(record.label == "transition" for record in records)
    lead_time_bonus = 10.0 * sum(
        record.future_chatter_within_horizon and record.label not in CHATTER_POSITIVE_LABELS for record in records
    )
    near_transition = sum(1.0 - min(abs(record.risk_chatter_now - 0.425) / 0.425, 1.0) for record in records)
    near_boundary = sum(max(0.0, 1.0 - abs(record.margin_physics) / 1.0) for record in records)
    return lead_time_bonus + transition_bonus + near_transition + 0.25 * near_boundary


def slice_result_windows(
    *,
    result: SimulationResult,
    scenario: str,
    episode: int,
    window_spec: WindowSpec,
    starting_window_id: int = 0,
    randomization_metadata: dict[str, float | bool] | None = None,
    horizon: HorizonConfig | None = None,
) -> tuple[list[NDArray[np.float64]], list[WindowRecord]]:
    horizon = horizon or HorizonConfig()
    sample_rate = _sample_rate(result)
    window_samples = int(round(window_spec.window_s * sample_rate))
    stride_samples = int(round(window_spec.stride_s * sample_rate))
    if window_samples < 4:
        raise ValueError("Window must contain at least four samples")
    if stride_samples < 1:
        raise ValueError("Stride must contain at least one sample")
    if window_samples > result.sensor_signal.shape[0]:
        return [], []

    randomization_metadata = _normalise_randomization_metadata(randomization_metadata)
    episode_duration_s = max(float(result.time_s[-1] - result.time_s[0]), 1.0e-12)
    previous_start_time: float | None = None
    previous_dominant_frequency_hz: float | None = None
    previous_rms: float | None = None
    previous_chatter_band_energy: float | None = None
    previous_tooth_band_energy: float | None = None
    previous_non_tooth_harmonic_ratio: float | None = None
    rms_ewma: float | None = None
    chatter_band_energy_ewma: float | None = None
    non_tooth_harmonic_ratio_ewma: float | None = None
    ewma_alpha = 0.35
    windows: list[NDArray[np.float64]] = []
    records: list[WindowRecord] = []
    for start in range(0, result.sensor_signal.shape[0] - window_samples + 1, stride_samples):
        stop = start + window_samples
        window = result.sensor_signal[start:stop]
        window_cut, axial_depth_profile_scale, cutting_coeff_profile_scale = _effective_window_cut(result, start, stop)
        stability = estimate_stability(result.modal, result.tool, window_cut)
        features = extract_signal_features(
            window,
            sample_rate,
            window_cut.spindle_rpm,
            result.tool.flute_count,
            result.modal.natural_frequency_hz,
        )
        risk = estimate_chatter_risk(features, stability.signed_margin)
        window_id = starting_window_id + len(records)
        window_index = len(records)
        start_time_s = float(result.time_s[start])
        end_time_s = float(result.time_s[stop - 1])
        dt = start_time_s - previous_start_time if previous_start_time is not None else 0.0
        dt = max(dt, 1.0e-12)
        dominant_frequency_delta_hz = _delta(features.dominant_frequency_hz, previous_dominant_frequency_hz)
        rms_delta = _delta(features.rms, previous_rms)
        chatter_band_energy_delta = _delta(features.chatter_band_energy, previous_chatter_band_energy)
        tooth_band_energy_delta = _delta(features.tooth_band_energy, previous_tooth_band_energy)
        non_tooth_harmonic_ratio_delta = _delta(features.non_tooth_harmonic_ratio, previous_non_tooth_harmonic_ratio)
        rms_ewma = _ewma(features.rms, rms_ewma, ewma_alpha)
        chatter_band_energy_ewma = _ewma(features.chatter_band_energy, chatter_band_energy_ewma, ewma_alpha)
        non_tooth_harmonic_ratio_ewma = _ewma(features.non_tooth_harmonic_ratio, non_tooth_harmonic_ratio_ewma, ewma_alpha)
        records.append(
            WindowRecord(
                window_id=window_id,
                scenario=scenario,
                episode=episode,
                window_index_in_episode=window_index,
                randomized=bool(randomization_metadata["randomized"]),
                spindle_scale=float(randomization_metadata["spindle_scale"]),
                feed_scale=float(randomization_metadata["feed_scale"]),
                axial_depth_scale=float(randomization_metadata["axial_depth_scale"]),
                radial_depth_scale=float(randomization_metadata["radial_depth_scale"]),
                stiffness_scale=float(randomization_metadata["stiffness_scale"]),
                damping_scale=float(randomization_metadata["damping_scale"]),
                cutting_coeff_scale=float(randomization_metadata["cutting_coeff_scale"]),
                noise_scale=float(randomization_metadata["noise_scale"]),
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                episode_progress=float(np.clip(start_time_s / episode_duration_s, 0.0, 1.0)),
                label=risk.label,
                label_id=LABEL_TO_ID[risk.label],
                horizon_label=risk.label,
                horizon_label_id=LABEL_TO_ID[risk.label],
                horizon_s=horizon.horizon_s,
                chatter_within_horizon=risk.label in CHATTER_POSITIVE_LABELS,
                future_chatter_within_horizon=False,
                time_to_chatter_s=0.0 if risk.label in CHATTER_POSITIVE_LABELS else -1.0,
                risk_chatter_now=risk.risk_chatter_now,
                risk_chatter_horizon=risk.risk_chatter_horizon,
                margin_physics=risk.margin_physics,
                margin_signal=risk.margin_signal,
                uncertainty=risk.uncertainty,
                spindle_rpm=window_cut.spindle_rpm,
                feed_per_tooth_m=window_cut.feed_per_tooth_m,
                axial_depth_m=window_cut.axial_depth_m,
                radial_depth_m=window_cut.radial_depth_m,
                axial_depth_profile_scale=axial_depth_profile_scale,
                cutting_coeff_profile_scale=cutting_coeff_profile_scale,
                cutting_coeff_t_n_m2=window_cut.cutting_coeff_t_n_m2,
                cutting_coeff_r_n_m2=window_cut.cutting_coeff_r_n_m2,
                tooth_frequency_hz=features.tooth_frequency_hz,
                limiting_frequency_hz=stability.limiting_frequency_hz,
                dominant_frequency_hz=features.dominant_frequency_hz,
                rms=features.rms,
                chatter_band_energy=features.chatter_band_energy,
                tooth_band_energy=features.tooth_band_energy,
                non_tooth_harmonic_ratio=features.non_tooth_harmonic_ratio,
                dominant_frequency_delta_hz=dominant_frequency_delta_hz,
                rms_delta=rms_delta,
                rms_growth_rate=rms_delta / dt if previous_start_time is not None else 0.0,
                chatter_band_energy_delta=chatter_band_energy_delta,
                chatter_band_energy_growth_rate=chatter_band_energy_delta / dt if previous_start_time is not None else 0.0,
                tooth_band_energy_delta=tooth_band_energy_delta,
                tooth_band_energy_growth_rate=tooth_band_energy_delta / dt if previous_start_time is not None else 0.0,
                non_tooth_harmonic_ratio_delta=non_tooth_harmonic_ratio_delta,
                non_tooth_harmonic_ratio_growth_rate=non_tooth_harmonic_ratio_delta / dt if previous_start_time is not None else 0.0,
                rms_ewma=rms_ewma,
                chatter_band_energy_ewma=chatter_band_energy_ewma,
                non_tooth_harmonic_ratio_ewma=non_tooth_harmonic_ratio_ewma,
            )
        )
        windows.append(window)
        previous_start_time = start_time_s
        previous_dominant_frequency_hz = features.dominant_frequency_hz
        previous_rms = features.rms
        previous_chatter_band_energy = features.chatter_band_energy
        previous_tooth_band_energy = features.tooth_band_energy
        previous_non_tooth_harmonic_ratio = features.non_tooth_harmonic_ratio
    return windows, attach_horizon_targets(records, horizon)


def _effective_window_cut(result: SimulationResult, start: int, stop: int) -> tuple[CutConfig, float, float]:
    if result.axial_depth_m is None or result.cutting_coeff_t_n_m2 is None or result.cutting_coeff_r_n_m2 is None:
        base_cut = result.cut
        axial_depth_profile_scale = 1.0
        cutting_coeff_profile_scale = 1.0
    else:
        axial_depth_m = float(np.mean(result.axial_depth_m[start:stop]))
        cutting_coeff_t_n_m2 = float(np.mean(result.cutting_coeff_t_n_m2[start:stop]))
        cutting_coeff_r_n_m2 = float(np.mean(result.cutting_coeff_r_n_m2[start:stop]))
        axial_depth_profile_scale = axial_depth_m / max(result.cut.axial_depth_m, 1.0e-18)
        cutting_coeff_profile_scale = cutting_coeff_t_n_m2 / max(result.cut.cutting_coeff_t_n_m2, 1.0e-18)
        base_cut = replace(
            result.cut,
            axial_depth_m=axial_depth_m,
            cutting_coeff_t_n_m2=cutting_coeff_t_n_m2,
            cutting_coeff_r_n_m2=cutting_coeff_r_n_m2,
        )

    spindle_rpm = (
        float(np.mean(result.spindle_rpm[start:stop]))
        if result.spindle_rpm is not None
        else result.cut.spindle_rpm
    )
    feed_per_tooth_m = (
        float(np.mean(result.feed_per_tooth_m[start:stop]))
        if result.feed_per_tooth_m is not None
        else result.cut.feed_per_tooth_m
    )
    return (
        replace(
            base_cut,
            spindle_rpm=spindle_rpm,
            feed_per_tooth_m=feed_per_tooth_m,
        ),
        float(axial_depth_profile_scale),
        float(cutting_coeff_profile_scale),
    )


def attach_horizon_targets(records: list[WindowRecord], horizon: HorizonConfig) -> list[WindowRecord]:
    updated: list[WindowRecord] = []
    for idx, record in enumerate(records):
        horizon_records = [
            candidate
            for candidate in records[idx:]
            if candidate.start_time_s - record.start_time_s <= horizon.horizon_s + 1.0e-12
        ]
        future_records = horizon_records[1:]
        horizon_label = max((candidate.label for candidate in horizon_records), key=lambda label: LABEL_SEVERITY[label])
        first_chatter = next((candidate for candidate in horizon_records if candidate.label in CHATTER_POSITIVE_LABELS), None)
        future_chatter = next((candidate for candidate in future_records if candidate.label in CHATTER_POSITIVE_LABELS), None)
        updated.append(
            replace(
                record,
                horizon_label=horizon_label,
                horizon_label_id=LABEL_TO_ID[horizon_label],
                horizon_s=horizon.horizon_s,
                chatter_within_horizon=first_chatter is not None,
                future_chatter_within_horizon=future_chatter is not None,
                time_to_chatter_s=float(first_chatter.start_time_s - record.start_time_s) if first_chatter is not None else -1.0,
            )
        )
    return updated


def apply_domain_randomization(
    *,
    modal: ModalParams,
    cut: CutConfig,
    sim_config: SimulationConfig,
    rng: np.random.Generator,
    config: DomainRandomizationConfig,
) -> tuple[ModalParams, CutConfig, SimulationConfig, dict[str, float | bool]]:
    if not config.enabled:
        return modal, cut, sim_config, _normalise_randomization_metadata(None)

    spindle_scale = _uniform_scale(rng, config.spindle_scale)
    feed_scale = _uniform_scale(rng, config.feed_scale)
    axial_depth_scale = _uniform_scale(rng, config.axial_depth_scale)
    radial_depth_scale = _uniform_scale(rng, config.radial_depth_scale)
    stiffness_scale = _uniform_scale(rng, config.stiffness_scale)
    damping_scale = _uniform_scale(rng, config.damping_scale)
    cutting_coeff_scale = _uniform_scale(rng, config.cutting_coeff_scale)
    noise_scale = _uniform_scale(rng, config.noise_scale)

    randomized_modal = replace(
        modal,
        stiffness_x_n_m=modal.stiffness_x_n_m * stiffness_scale,
        stiffness_y_n_m=modal.stiffness_y_n_m * stiffness_scale,
        damping_x_n_s_m=modal.damping_x_n_s_m * damping_scale,
        damping_y_n_s_m=modal.damping_y_n_s_m * damping_scale,
    )
    randomized_cut = replace(
        cut,
        spindle_rpm=cut.spindle_rpm * spindle_scale,
        feed_per_tooth_m=cut.feed_per_tooth_m * feed_scale,
        axial_depth_m=cut.axial_depth_m * axial_depth_scale,
        radial_depth_m=cut.radial_depth_m * radial_depth_scale,
        cutting_coeff_t_n_m2=cut.cutting_coeff_t_n_m2 * cutting_coeff_scale,
        cutting_coeff_r_n_m2=cut.cutting_coeff_r_n_m2 * cutting_coeff_scale,
    )
    randomized_sim_config = replace(
        sim_config,
        sensor_noise_std=sim_config.sensor_noise_std * noise_scale,
    )
    return (
        randomized_modal,
        randomized_cut,
        randomized_sim_config,
        {
            "randomized": True,
            "spindle_scale": spindle_scale,
            "feed_scale": feed_scale,
            "axial_depth_scale": axial_depth_scale,
            "radial_depth_scale": radial_depth_scale,
            "stiffness_scale": stiffness_scale,
            "damping_scale": damping_scale,
            "cutting_coeff_scale": cutting_coeff_scale,
            "noise_scale": noise_scale,
        },
    )


def _manifest(
    records: list[WindowRecord],
    scenarios: list[str],
    episodes: int,
    window_spec: WindowSpec,
    sensor_windows: NDArray[np.float32],
    randomization: DomainRandomizationConfig,
    transition_focus: TransitionFocusConfig,
    horizon: HorizonConfig,
) -> DatasetManifest:
    counts = Counter(record.label for record in records)
    return DatasetManifest(
        schema_version="chatter-window-v5",
        sample_rate_hz=result_sample_rate(records, sensor_windows),
        window_s=window_spec.window_s,
        stride_s=window_spec.stride_s,
        channel_names=CHANNEL_NAMES,
        scenarios=tuple(scenarios),
        episodes_per_scenario=episodes,
        total_windows=len(records),
        label_counts=dict(sorted(counts.items())),
        domain_randomization=asdict(randomization),
        sampling_strategy={
            "transition_focus": asdict(transition_focus),
        },
        horizon=asdict(horizon),
        artifacts=("dataset.npz", "windows.csv", "manifest.json", "README.md"),
    )


def result_sample_rate(records: list[WindowRecord], sensor_windows: NDArray[np.float32]) -> float:
    if not records:
        return 0.0
    duration = max(records[0].end_time_s - records[0].start_time_s, 1.0e-12)
    return float((sensor_windows.shape[1] - 1) / duration)


def _sample_rate(result: SimulationResult) -> float:
    if result.time_s.size < 2:
        raise ValueError("Simulation result needs at least two timestamps")
    return float(1.0 / np.mean(np.diff(result.time_s)))


def _delta(current: float, previous: float | None) -> float:
    if previous is None:
        return 0.0
    return float(current - previous)


def _ewma(current: float, previous_ewma: float | None, alpha: float) -> float:
    if previous_ewma is None:
        return float(current)
    return float(alpha * current + (1.0 - alpha) * previous_ewma)


def _validate_scale_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = bounds
    if lower <= 0 or upper <= 0:
        raise ValueError(f"{name} bounds must be positive")
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound")


def _uniform_scale(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(rng.uniform(lower, upper))


def _normalise_randomization_metadata(metadata: dict[str, float | bool] | None) -> dict[str, float | bool]:
    normalised: dict[str, float | bool] = {
        "randomized": False,
        "spindle_scale": 1.0,
        "feed_scale": 1.0,
        "axial_depth_scale": 1.0,
        "radial_depth_scale": 1.0,
        "stiffness_scale": 1.0,
        "damping_scale": 1.0,
        "cutting_coeff_scale": 1.0,
        "noise_scale": 1.0,
    }
    if metadata:
        normalised.update(metadata)
    return normalised


def _write_records_csv(path: Path, records: list[WindowRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_readme(path: Path, manifest: DatasetManifest) -> None:
    lines = [
        "# Chatter Twin Replay Dataset",
        "",
        f"Schema version: `{manifest.schema_version}`",
        f"Total windows: `{manifest.total_windows}`",
        f"Sample rate: `{manifest.sample_rate_hz:.1f} Hz`",
        f"Window/stride: `{manifest.window_s:.3f}s / {manifest.stride_s:.3f}s`",
        f"Horizon target: `{manifest.horizon['horizon_s']:.3f}s`",
        f"Domain randomization: `{'enabled' if manifest.domain_randomization['enabled'] else 'disabled'}`",
        f"Transition focus: `{'enabled' if manifest.sampling_strategy['transition_focus']['enabled'] else 'disabled'}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in manifest.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `dataset.npz`: sensor windows and aligned labels/targets.",
            "- `windows.csv`: per-window metadata and feature/risk summaries.",
            "- `manifest.json`: schema and dataset-level metadata.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
