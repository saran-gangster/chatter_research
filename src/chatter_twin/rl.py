from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

from chatter_twin.calibration import MarginCalibration
from chatter_twin.env import ChatterSuppressEnv
from chatter_twin.models import RewardConfig, ShieldConfig, SimulationConfig
from chatter_twin.replay import DomainRandomizationConfig, apply_domain_randomization
from chatter_twin.scenarios import make_scenario


DEFAULT_ACTION_DELTAS: tuple[tuple[float, float], ...] = (
    (-0.05, -0.05),
    (-0.05, 0.00),
    (-0.05, 0.05),
    (0.00, -0.05),
    (0.00, 0.00),
    (0.00, 0.05),
    (0.05, -0.05),
    (0.05, 0.00),
    (0.05, 0.05),
)
STATE_BINS: dict[str, tuple[float, ...]] = {
    "feed_override": (0.85, 0.95, 1.00, 1.05),
    "spindle_override": (0.92, 0.98, 1.02, 1.08),
    "risk_now": (0.35, 0.45, 0.60, 0.75),
    "risk_horizon": (0.35, 0.45, 0.60, 0.75),
    "margin_physics": (-0.30, -0.05, 0.05, 0.30),
    "uncertainty": (0.35, 0.55, 0.65),
}


@dataclass(frozen=True)
class QLearningConfig:
    scenarios: tuple[str, ...] = ("stable", "near_boundary", "onset", "unstable")
    episodes: int = 200
    eval_episodes: int = 8
    steps: int = 20
    decision_interval_s: float = 0.10
    seed: int = 515
    learning_rate: float = 0.20
    discount: float = 0.92
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    reward_config: RewardConfig = RewardConfig()
    shield_config: ShieldConfig = ShieldConfig()
    action_deltas: tuple[tuple[float, float], ...] = DEFAULT_ACTION_DELTAS

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not self.action_deltas:
            raise ValueError("action_deltas must be non-empty")


@dataclass(frozen=True)
class Sb3TrainingConfig:
    algorithm: str = "sac"
    scenarios: tuple[str, ...] = ("stable", "near_boundary", "onset", "unstable")
    total_timesteps: int = 5_000
    eval_episodes: int = 8
    steps: int = 20
    decision_interval_s: float = 0.10
    seed: int = 616
    learning_rate: float = 3.0e-4
    buffer_size: int = 50_000
    learning_starts: int = 100
    batch_size: int = 64
    gamma: float = 0.92
    train_freq: int = 1
    gradient_steps: int = 1
    baseline_controllers: tuple[str, ...] = ("sld", "mpc")
    candidate_guard: bool = True
    reward_config: RewardConfig = RewardConfig()
    shield_config: ShieldConfig = ShieldConfig()
    action_mode: str = "absolute"
    delta_action_scale: float = 1.0
    delta_feed_scale: float | None = None
    delta_spindle_scale: float | None = None
    delta_mapping: str = "fixed"

    def __post_init__(self) -> None:
        if self.algorithm not in {"sac", "td3"}:
            raise ValueError("algorithm must be 'sac' or 'td3'")
        if self.action_mode not in {"absolute", "delta"}:
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        if not 0 < self.delta_action_scale <= 1:
            raise ValueError("delta_action_scale must be in (0, 1]")
        _validate_optional_delta_scale(self.delta_feed_scale, "delta_feed_scale")
        _validate_optional_delta_scale(self.delta_spindle_scale, "delta_spindle_scale")
        if self.delta_mapping not in {"fixed", "headroom"}:
            raise ValueError("delta_mapping must be 'fixed' or 'headroom'")
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be at least 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.train_freq < 1:
            raise ValueError("train_freq must be at least 1")
        if self.gradient_steps < 0:
            raise ValueError("gradient_steps cannot be negative")


@dataclass(frozen=True)
class MultiSeedTrainingConfig:
    algorithms: tuple[str, ...] = ("sac", "td3")
    seeds: tuple[int, ...] = (616, 717, 818)
    scenarios: tuple[str, ...] = ("stable", "near_boundary", "onset", "unstable")
    episodes: int = 200
    total_timesteps: int = 5_000
    eval_episodes: int = 8
    steps: int = 20
    decision_interval_s: float = 0.10
    learning_rate: float = 3.0e-4
    buffer_size: int = 50_000
    learning_starts: int = 100
    batch_size: int = 64
    gamma: float = 0.92
    train_freq: int = 1
    gradient_steps: int = 1
    baseline_controllers: tuple[str, ...] = ("sld", "mpc")
    candidate_guard: bool = True
    reward_config: RewardConfig = RewardConfig()
    shield_config: ShieldConfig = ShieldConfig()
    action_mode: str = "absolute"
    delta_action_scale: float = 1.0
    delta_feed_scale: float | None = None
    delta_spindle_scale: float | None = None
    delta_mapping: str = "fixed"

    def __post_init__(self) -> None:
        if not self.algorithms:
            raise ValueError("algorithms must be non-empty")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        unsupported = [algorithm for algorithm in self.algorithms if algorithm not in {"q_learning", "sac", "td3"}]
        if unsupported:
            raise ValueError(f"unsupported algorithms: {', '.join(unsupported)}")
        if self.action_mode not in {"absolute", "delta"}:
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        if not 0 < self.delta_action_scale <= 1:
            raise ValueError("delta_action_scale must be in (0, 1]")
        _validate_optional_delta_scale(self.delta_feed_scale, "delta_feed_scale")
        _validate_optional_delta_scale(self.delta_spindle_scale, "delta_spindle_scale")
        if self.delta_mapping not in {"fixed", "headroom"}:
            raise ValueError("delta_mapping must be 'fixed' or 'headroom'")
        if not self.scenarios:
            raise ValueError("scenarios must be non-empty")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be at least 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.train_freq < 1:
            raise ValueError("train_freq must be at least 1")
        if self.gradient_steps < 0:
            raise ValueError("gradient_steps cannot be negative")


@dataclass(frozen=True)
class TabularQPolicy:
    q_table: dict[str, tuple[float, ...]]
    action_deltas: tuple[tuple[float, float], ...] = DEFAULT_ACTION_DELTAS
    state_bins: dict[str, tuple[float, ...]] = field(default_factory=lambda: STATE_BINS)
    shield_config: ShieldConfig = ShieldConfig()

    def state_key(self, observation: np.ndarray) -> str:
        bins = (
            _bin(float(observation[0]), self.state_bins["feed_override"]),
            _bin(float(observation[1]), self.state_bins["spindle_override"]),
            _bin(float(observation[4]), self.state_bins["risk_now"]),
            _bin(float(observation[5]), self.state_bins["risk_horizon"]),
            _bin(float(observation[6]), self.state_bins["margin_physics"]),
            _bin(float(observation[8]), self.state_bins["uncertainty"]),
        )
        return ":".join(str(value) for value in bins)

    def q_values(self, state_key: str) -> np.ndarray:
        values = self.q_table.get(state_key)
        if values is None:
            return np.zeros(len(self.action_deltas), dtype=float)
        return np.array(values, dtype=float)

    def greedy_action(self, observation: np.ndarray) -> int:
        values = self.q_values(self.state_key(observation))
        best_value = float(np.max(values))
        candidates = np.flatnonzero(np.isclose(values, best_value))
        return int(candidates[0])

    def action_to_overrides(self, observation: np.ndarray, action_index: int) -> tuple[float, float]:
        feed_delta, spindle_delta = self.action_deltas[action_index]
        feed = _clip(float(observation[0]) + feed_delta, self.shield_config.min_feed_override, self.shield_config.max_feed_override)
        spindle = _clip(float(observation[1]) + spindle_delta, self.shield_config.min_spindle_override, self.shield_config.max_spindle_override)
        return feed, spindle

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "tabular_q_learning",
            "action_deltas": [list(value) for value in self.action_deltas],
            "state_bins": {key: list(value) for key, value in self.state_bins.items()},
            "shield_config": asdict(self.shield_config),
            "q_table": {key: list(value) for key, value in sorted(self.q_table.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TabularQPolicy:
        return cls(
            q_table={str(key): tuple(float(item) for item in value) for key, value in payload["q_table"].items()},
            action_deltas=tuple(tuple(float(item) for item in value) for value in payload.get("action_deltas", DEFAULT_ACTION_DELTAS)),
            state_bins={str(key): tuple(float(item) for item in value) for key, value in payload.get("state_bins", STATE_BINS).items()},
            shield_config=ShieldConfig(**payload.get("shield_config", {})),
        )


class ScenarioSamplingEnv(gym.Env):
    """Sample one calibrated/randomized chatter environment per training episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenarios: tuple[str, ...],
        steps: int,
        decision_interval_s: float,
        seed: int,
        margin_calibration: MarginCalibration | None = None,
        randomization: DomainRandomizationConfig | None = None,
        reward_config: RewardConfig | None = None,
        shield_config: ShieldConfig | None = None,
        action_mode: str = "absolute",
        delta_action_scale: float = 1.0,
        delta_feed_scale: float | None = None,
        delta_spindle_scale: float | None = None,
        delta_mapping: str = "fixed",
        candidate_guard: bool = True,
    ) -> None:
        super().__init__()
        if not scenarios:
            raise ValueError("scenarios must be non-empty")
        self.scenarios = scenarios
        self.steps = steps
        self.decision_interval_s = decision_interval_s
        self.base_seed = seed
        self.margin_calibration = margin_calibration
        self.randomization = randomization or DomainRandomizationConfig()
        self.reward_config = reward_config or RewardConfig()
        self.shield_config = shield_config or ShieldConfig()
        self.action_mode = action_mode
        self.delta_action_scale = delta_action_scale
        self.delta_feed_scale = delta_feed_scale
        self.delta_spindle_scale = delta_spindle_scale
        self.delta_mapping = delta_mapping
        self.candidate_guard = candidate_guard
        self._rng = np.random.default_rng(seed)
        self._episode_idx = 0
        self._env = _make_env(
            scenario=scenarios[0],
            seed=seed,
            steps=steps,
            decision_interval_s=decision_interval_s,
            margin_calibration=margin_calibration,
            randomization=self.randomization,
            reward_config=self.reward_config,
            shield_config=self.shield_config,
        )
        self.action_space = _policy_action_space(
            self._env,
            action_mode,
            delta_action_scale,
            delta_feed_scale,
            delta_spindle_scale,
        )
        self.observation_space = self._env.observation_space
        self.current_scenario = scenarios[0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.base_seed = seed
            self._episode_idx = 0
        scenario = str(self._rng.choice(self.scenarios))
        episode_seed = int(self.base_seed + self._episode_idx * 10_000 + self._rng.integers(0, 9_999))
        self._episode_idx += 1
        self.current_scenario = scenario
        self._env = _make_env(
            scenario=scenario,
            seed=episode_seed,
            steps=self.steps,
            decision_interval_s=self.decision_interval_s,
            margin_calibration=self.margin_calibration,
            randomization=self.randomization,
            reward_config=self.reward_config,
            shield_config=self.shield_config,
        )
        obs, info = self._env.reset(seed=episode_seed)
        info["scenario"] = scenario
        return obs, info

    def step(self, action):
        action = _policy_action_to_absolute(
            self._env,
            action,
            self.action_mode,
            self.delta_action_scale,
            self.delta_feed_scale,
            self.delta_spindle_scale,
            self.delta_mapping,
        )
        guard_info = None
        if self.candidate_guard:
            action, guard_info = _guard_continuous_action_with_info(self._env, action, enabled=True)
        obs, reward, terminated, truncated, info = self._env.step(action)
        reward = _apply_guard_reward_adjustment(reward, info, guard_info, self.reward_config)
        info["scenario"] = self.current_scenario
        return obs, reward, terminated, truncated, info


def train_q_learning(
    *,
    config: QLearningConfig,
    out_dir: Path,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    q_table: dict[str, list[float]] = {}
    train_rows: list[dict[str, Any]] = []
    epsilon = config.epsilon_start

    for episode in range(config.episodes):
        scenario = config.scenarios[episode % len(config.scenarios)]
        episode_seed = config.seed + episode * 10_000
        env = _make_env(
            scenario=scenario,
            seed=episode_seed,
            steps=config.steps,
            decision_interval_s=config.decision_interval_s,
            margin_calibration=margin_calibration,
            randomization=randomization,
            reward_config=config.reward_config,
            shield_config=config.shield_config,
        )
        policy = TabularQPolicy(
            q_table={key: tuple(value) for key, value in q_table.items()},
            action_deltas=config.action_deltas,
            shield_config=config.shield_config,
        )
        obs, _ = env.reset(seed=episode_seed)
        row = _new_episode_row("q_learning", scenario, episode, config.steps)

        for _ in range(config.steps):
            state_key = policy.state_key(obs)
            _ensure_state(q_table, state_key, len(config.action_deltas))
            if rng.random() < epsilon:
                action_index = int(rng.integers(0, len(config.action_deltas)))
            else:
                action_index = _argmax_stable(q_table[state_key])

            feed, spindle = policy.action_to_overrides(obs, action_index)
            next_obs, reward, _, truncated, info = env.step(np.array([feed, spindle], dtype=np.float32))
            next_key = policy.state_key(next_obs)
            _ensure_state(q_table, next_key, len(config.action_deltas))
            target = float(reward) + config.discount * max(q_table[next_key])
            current = q_table[state_key][action_index]
            q_table[state_key][action_index] = current + config.learning_rate * (target - current)
            _update_episode_row(row, reward, info)
            obs = next_obs
            if truncated:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row["epsilon"] = epsilon
        train_rows.append(_finalize_episode_row(row))

    policy = TabularQPolicy(
        q_table={key: tuple(value) for key, value in q_table.items()},
        action_deltas=config.action_deltas,
        shield_config=config.shield_config,
    )
    eval_rows = evaluate_q_policy(
        policy=policy,
        scenarios=config.scenarios,
        episodes=config.eval_episodes,
        steps=config.steps,
        decision_interval_s=config.decision_interval_s,
        seed=config.seed + 1_000_000,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=config.reward_config,
        shield_config=config.shield_config,
    )
    summary_rows = _summarize(eval_rows)
    payload = {
        "algorithm": "tabular_q_learning",
        "config": asdict(config),
        "state_count": len(q_table),
        "training": {
            "episodes": len(train_rows),
            "mean_reward_last_10": _mean([row["total_reward"] for row in train_rows[-10:]]),
            "mean_shield_rejections_last_10": _mean([row["shield_rejections"] for row in train_rows[-10:]]),
        },
        "evaluation": {
            "episodes": len(eval_rows),
            "summary": summary_rows,
        },
        "policy": policy.to_dict(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "training_episodes.csv", train_rows)
    _write_csv(out_dir / "evaluation_episodes.csv", eval_rows)
    _write_csv(out_dir / "evaluation_summary.csv", summary_rows)
    _write_json(out_dir / "policy.json", policy.to_dict())
    _write_json(out_dir / "metrics.json", payload)
    _write_report(out_dir / "report.md", payload)
    _write_learning_svg(out_dir / "learning_curve.svg", train_rows)
    return payload


def train_sb3_policy(
    *,
    config: Sb3TrainingConfig,
    out_dir: Path,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    algorithm_cls, action_noise = _load_sb3_algorithm(config)
    train_env = ScenarioSamplingEnv(
        scenarios=config.scenarios,
        steps=config.steps,
        decision_interval_s=config.decision_interval_s,
        seed=config.seed,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=config.reward_config,
        shield_config=config.shield_config,
        action_mode=config.action_mode,
        delta_action_scale=config.delta_action_scale,
        delta_feed_scale=config.delta_feed_scale,
        delta_spindle_scale=config.delta_spindle_scale,
        delta_mapping=config.delta_mapping,
        candidate_guard=config.candidate_guard,
    )
    model_kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": train_env,
        "learning_rate": config.learning_rate,
        "buffer_size": config.buffer_size,
        "learning_starts": config.learning_starts,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "train_freq": config.train_freq,
        "gradient_steps": config.gradient_steps,
        "seed": config.seed,
        "device": "cpu",
        "verbose": 0,
    }
    if action_noise is not None:
        model_kwargs["action_noise"] = action_noise
    model = algorithm_cls(**model_kwargs)
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    eval_rows = evaluate_sb3_policy(
        model=model,
        controller_name=config.algorithm,
        scenarios=config.scenarios,
        episodes=config.eval_episodes,
        steps=config.steps,
        decision_interval_s=config.decision_interval_s,
        seed=config.seed + 1_000_000,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=config.reward_config,
        shield_config=config.shield_config,
        candidate_guard=config.candidate_guard,
        action_mode=config.action_mode,
        delta_action_scale=config.delta_action_scale,
        delta_feed_scale=config.delta_feed_scale,
        delta_spindle_scale=config.delta_spindle_scale,
        delta_mapping=config.delta_mapping,
    )
    action_rows = trace_sb3_policy_actions(
        model=model,
        controller_name=config.algorithm,
        scenarios=config.scenarios,
        episodes=config.eval_episodes,
        steps=config.steps,
        decision_interval_s=config.decision_interval_s,
        seed=config.seed + 1_000_000,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=config.reward_config,
        shield_config=config.shield_config,
        candidate_guard=config.candidate_guard,
        action_mode=config.action_mode,
        delta_action_scale=config.delta_action_scale,
        delta_feed_scale=config.delta_feed_scale,
        delta_spindle_scale=config.delta_spindle_scale,
        delta_mapping=config.delta_mapping,
    )
    eval_summary = _summarize(eval_rows)
    action_diagnostics = _summarize_action_trace(action_rows)
    baseline_payload = _run_baseline_comparison(
        config=config,
        margin_calibration=margin_calibration,
        randomization=randomization,
    )
    comparison_summary = [*eval_summary, *baseline_payload["summary"]]
    payload = {
        "algorithm": config.algorithm,
        "config": asdict(config),
        "evaluation": {
            "episodes": len(eval_rows),
            "summary": eval_summary,
        },
        "action_diagnostics": action_diagnostics,
        "baselines": {
            "controllers": list(config.baseline_controllers),
            "episodes": baseline_payload["episodes"],
            "summary": baseline_payload["summary"],
        },
        "comparison_summary": comparison_summary,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model.zip"))
    _write_csv(out_dir / "evaluation_episodes.csv", eval_rows)
    _write_csv(out_dir / "evaluation_actions.csv", action_rows)
    _write_csv(out_dir / "evaluation_summary.csv", eval_summary)
    _write_csv(out_dir / "action_diagnostics_summary.csv", action_diagnostics)
    _write_csv(out_dir / "baseline_episodes.csv", baseline_payload["episodes"])
    _write_csv(out_dir / "baseline_summary.csv", baseline_payload["summary"])
    _write_csv(out_dir / "comparison_summary.csv", comparison_summary)
    _write_json(out_dir / "metrics.json", payload)
    _write_report(out_dir / "report.md", payload)
    return payload


def train_multi_seed_policies(
    *,
    config: MultiSeedTrainingConfig,
    out_dir: Path,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Run shielded controller candidates across seeds and aggregate results."""

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    run_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    action_diagnostic_rows: list[dict[str, Any]] = []
    artifacts: list[str] = []

    for algorithm in config.algorithms:
        for seed in config.seeds:
            run_name = f"{algorithm}_seed_{seed}"
            run_out = runs_dir / run_name
            if algorithm == "q_learning":
                payload = train_q_learning(
                    config=QLearningConfig(
                        scenarios=config.scenarios,
                        episodes=config.episodes,
                        eval_episodes=config.eval_episodes,
                        steps=config.steps,
                        decision_interval_s=config.decision_interval_s,
                        seed=seed,
                        learning_rate=min(config.learning_rate, 1.0),
                        discount=config.gamma,
                        reward_config=config.reward_config,
                        shield_config=config.shield_config,
                    ),
                    out_dir=run_out,
                    margin_calibration=margin_calibration,
                    randomization=randomization,
                )
                summary_rows = payload["evaluation"]["summary"]
            else:
                payload = train_sb3_policy(
                    config=Sb3TrainingConfig(
                        algorithm=algorithm,
                        scenarios=config.scenarios,
                        total_timesteps=config.total_timesteps,
                        eval_episodes=config.eval_episodes,
                        steps=config.steps,
                        decision_interval_s=config.decision_interval_s,
                        seed=seed,
                        learning_rate=config.learning_rate,
                        buffer_size=config.buffer_size,
                        learning_starts=config.learning_starts,
                        batch_size=config.batch_size,
                        gamma=config.gamma,
                        train_freq=config.train_freq,
                        gradient_steps=config.gradient_steps,
                        baseline_controllers=config.baseline_controllers,
                        candidate_guard=config.candidate_guard,
                        reward_config=config.reward_config,
                        shield_config=config.shield_config,
                        action_mode=config.action_mode,
                        delta_action_scale=config.delta_action_scale,
                        delta_feed_scale=config.delta_feed_scale,
                        delta_spindle_scale=config.delta_spindle_scale,
                        delta_mapping=config.delta_mapping,
                    ),
                    out_dir=run_out,
                    margin_calibration=margin_calibration,
                    randomization=randomization,
                )
                summary_rows = payload["comparison_summary"]

            run_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "run_dir": str(run_out),
                    "evaluation_episodes": payload["evaluation"]["episodes"],
                    "guard_fallbacks": _diagnostic_total(payload, "guard_fallbacks"),
                    "shield_rejections": _diagnostic_total(payload, "shield_rejections"),
                }
            )
            artifacts.append(str(run_out))
            for row in summary_rows:
                comparison_rows.append(
                    {
                        "training_algorithm": algorithm,
                        "seed": seed,
                        "run_dir": str(run_out),
                        **row,
                    }
                )
            for row in payload.get("action_diagnostics", []):
                action_diagnostic_rows.append(
                    {
                        "training_algorithm": algorithm,
                        "seed": seed,
                        "run_dir": str(run_out),
                        **row,
                    }
                )

    aggregate_rows = _aggregate_seed_summaries(comparison_rows)
    action_diagnostic_aggregate_rows = _aggregate_seed_action_diagnostics(action_diagnostic_rows)
    payload = {
        "algorithm": "multi_seed_rl",
        "config": asdict(config),
        "runs": run_rows,
        "comparison_summary": comparison_rows,
        "aggregate_summary": aggregate_rows,
        "action_diagnostics": action_diagnostic_rows,
        "action_diagnostics_aggregate": action_diagnostic_aggregate_rows,
        "artifacts": {
            "runs": artifacts,
            "run_summary": "run_summary.csv",
            "comparison_summary": "comparison_summary.csv",
            "aggregate_summary": "aggregate_summary.csv",
            "action_diagnostics_summary": "action_diagnostics_summary.csv",
            "action_diagnostics_aggregate": "action_diagnostics_aggregate.csv",
            "metrics": "metrics.json",
            "report": "report.md",
        },
    }

    _write_csv(out_dir / "run_summary.csv", run_rows)
    _write_csv(out_dir / "comparison_summary.csv", comparison_rows)
    _write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    _write_csv(out_dir / "action_diagnostics_summary.csv", action_diagnostic_rows)
    _write_csv(out_dir / "action_diagnostics_aggregate.csv", action_diagnostic_aggregate_rows)
    _write_json(out_dir / "metrics.json", payload)
    _write_multi_seed_report(out_dir / "report.md", payload)
    return payload


def evaluate_saved_sb3_run(
    *,
    source_dir: Path,
    out_dir: Path,
    scenarios: tuple[str, ...],
    eval_episodes: int,
    steps: int,
    decision_interval_s: float,
    seed: int,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate saved SAC/TD3 checkpoints under a fresh scenario/randomization profile."""

    saved_runs = _discover_saved_sb3_runs(source_dir)
    if not saved_runs:
        raise ValueError(f"No saved SAC/TD3 model.zip artifacts found under {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    run_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    action_diagnostic_rows: list[dict[str, Any]] = []
    artifacts: list[str] = []
    first_config: Sb3TrainingConfig | None = None

    for saved_run in saved_runs:
        run_dir = saved_run["run_dir"]
        algorithm = str(saved_run["algorithm"])
        source_seed = int(saved_run["seed"])
        run_metrics = _read_json(run_dir / "metrics.json")
        config = _sb3_config_from_saved_metrics(
            run_metrics,
            algorithm=algorithm,
            scenarios=scenarios,
            eval_episodes=eval_episodes,
            steps=steps,
            decision_interval_s=decision_interval_s,
            seed=seed + source_seed,
        )
        if first_config is None:
            first_config = config
        algorithm_cls, _ = _load_sb3_algorithm(config)
        model = algorithm_cls.load(str(run_dir / "model.zip"), device="cpu")

        eval_rows = evaluate_sb3_policy(
            model=model,
            controller_name=config.algorithm,
            scenarios=config.scenarios,
            episodes=config.eval_episodes,
            steps=config.steps,
            decision_interval_s=config.decision_interval_s,
            seed=config.seed + 1_000_000,
            margin_calibration=margin_calibration,
            randomization=randomization,
            reward_config=config.reward_config,
            shield_config=config.shield_config,
            candidate_guard=config.candidate_guard,
            action_mode=config.action_mode,
            delta_action_scale=config.delta_action_scale,
            delta_feed_scale=config.delta_feed_scale,
            delta_spindle_scale=config.delta_spindle_scale,
            delta_mapping=config.delta_mapping,
        )
        action_rows = trace_sb3_policy_actions(
            model=model,
            controller_name=config.algorithm,
            scenarios=config.scenarios,
            episodes=config.eval_episodes,
            steps=config.steps,
            decision_interval_s=config.decision_interval_s,
            seed=config.seed + 1_000_000,
            margin_calibration=margin_calibration,
            randomization=randomization,
            reward_config=config.reward_config,
            shield_config=config.shield_config,
            candidate_guard=config.candidate_guard,
            action_mode=config.action_mode,
            delta_action_scale=config.delta_action_scale,
            delta_feed_scale=config.delta_feed_scale,
            delta_spindle_scale=config.delta_spindle_scale,
            delta_mapping=config.delta_mapping,
        )
        eval_summary = _summarize(eval_rows)
        action_diagnostics = _summarize_action_trace(action_rows)
        baseline_payload = _run_baseline_comparison(
            config=config,
            margin_calibration=margin_calibration,
            randomization=randomization,
        )
        comparison_summary = [*eval_summary, *baseline_payload["summary"]]

        run_out = runs_dir / f"{algorithm}_seed_{source_seed}"
        run_out.mkdir(parents=True, exist_ok=True)
        run_payload = {
            "algorithm": algorithm,
            "source_run": str(run_dir),
            "source_seed": source_seed,
            "config": asdict(config),
            "evaluation": {
                "episodes": len(eval_rows),
                "summary": eval_summary,
            },
            "action_diagnostics": action_diagnostics,
            "baselines": {
                "controllers": list(config.baseline_controllers),
                "episodes": baseline_payload["episodes"],
                "summary": baseline_payload["summary"],
            },
            "comparison_summary": comparison_summary,
        }
        _write_csv(run_out / "evaluation_episodes.csv", eval_rows)
        _write_csv(run_out / "evaluation_actions.csv", action_rows)
        _write_csv(run_out / "evaluation_summary.csv", eval_summary)
        _write_csv(run_out / "action_diagnostics_summary.csv", action_diagnostics)
        _write_csv(run_out / "baseline_episodes.csv", baseline_payload["episodes"])
        _write_csv(run_out / "baseline_summary.csv", baseline_payload["summary"])
        _write_csv(run_out / "comparison_summary.csv", comparison_summary)
        _write_json(run_out / "metrics.json", run_payload)
        _write_report(run_out / "report.md", run_payload)

        run_rows.append(
            {
                "algorithm": algorithm,
                "seed": source_seed,
                "source_run": str(run_dir),
                "run_dir": str(run_out),
                "evaluation_episodes": len(eval_rows),
                "guard_fallbacks": sum(int(row["guard_fallbacks"]) for row in action_diagnostics),
                "shield_rejections": sum(int(row["shield_rejections"]) for row in action_diagnostics),
            }
        )
        artifacts.append(str(run_out))
        for row in comparison_summary:
            comparison_rows.append(
                {
                    "training_algorithm": algorithm,
                    "seed": source_seed,
                    "run_dir": str(run_out),
                    **row,
                }
            )
        for row in action_diagnostics:
            action_diagnostic_rows.append(
                {
                    "training_algorithm": algorithm,
                    "seed": source_seed,
                    "run_dir": str(run_out),
                    **row,
                }
            )

    assert first_config is not None
    aggregate_rows = _aggregate_seed_summaries(comparison_rows)
    action_diagnostic_aggregate_rows = _aggregate_seed_action_diagnostics(action_diagnostic_rows)
    profile_config = _saved_eval_profile_config(
        config=first_config,
        source_dir=source_dir,
        saved_runs=saved_runs,
        randomization=randomization,
    )
    payload = {
        "algorithm": "saved_sb3_evaluation",
        "config": profile_config,
        "runs": run_rows,
        "comparison_summary": comparison_rows,
        "aggregate_summary": aggregate_rows,
        "action_diagnostics": action_diagnostic_rows,
        "action_diagnostics_aggregate": action_diagnostic_aggregate_rows,
        "artifacts": {
            "runs": artifacts,
            "run_summary": "run_summary.csv",
            "comparison_summary": "comparison_summary.csv",
            "aggregate_summary": "aggregate_summary.csv",
            "action_diagnostics_summary": "action_diagnostics_summary.csv",
            "action_diagnostics_aggregate": "action_diagnostics_aggregate.csv",
            "metrics": "metrics.json",
            "report": "report.md",
        },
    }

    _write_csv(out_dir / "run_summary.csv", run_rows)
    _write_csv(out_dir / "comparison_summary.csv", comparison_rows)
    _write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    _write_csv(out_dir / "action_diagnostics_summary.csv", action_diagnostic_rows)
    _write_csv(out_dir / "action_diagnostics_aggregate.csv", action_diagnostic_aggregate_rows)
    _write_json(out_dir / "metrics.json", payload)
    _write_saved_policy_eval_report(out_dir / "report.md", payload)
    return payload


def evaluate_sb3_policy(
    *,
    model: Any,
    controller_name: str,
    scenarios: tuple[str, ...],
    episodes: int,
    steps: int,
    decision_interval_s: float,
    seed: int,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
    candidate_guard: bool = True,
    action_mode: str = "absolute",
    delta_action_scale: float = 1.0,
    delta_feed_scale: float | None = None,
    delta_spindle_scale: float | None = None,
    delta_mapping: str = "fixed",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for episode in range(episodes):
            episode_seed = seed + 10_000 * episode + 100 * scenario_index
            env = _make_env(
                scenario=scenario,
                seed=episode_seed,
                steps=steps,
                decision_interval_s=decision_interval_s,
                margin_calibration=margin_calibration,
                randomization=randomization,
                reward_config=reward_config,
                shield_config=shield_config,
            )
            obs, _ = env.reset(seed=episode_seed)
            row = _new_episode_row(controller_name, scenario, episode, steps)
            for _ in range(steps):
                action, _ = model.predict(obs, deterministic=True)
                action = _policy_action_to_absolute(
                    env,
                    action,
                    action_mode,
                    delta_action_scale,
                    delta_feed_scale,
                    delta_spindle_scale,
                    delta_mapping,
                )
                guard_info = None
                if candidate_guard:
                    action, guard_info = _guard_continuous_action_with_info(env, action, enabled=True)
                obs, reward, _, truncated, info = env.step(np.array(action, dtype=np.float32))
                reward = _apply_guard_reward_adjustment(reward, info, guard_info, reward_config)
                _update_episode_row(row, reward, info)
                if truncated:
                    break
            rows.append(_finalize_episode_row(row))
    return rows


def trace_sb3_policy_actions(
    *,
    model: Any,
    controller_name: str,
    scenarios: tuple[str, ...],
    episodes: int,
    steps: int,
    decision_interval_s: float,
    seed: int,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
    candidate_guard: bool = True,
    action_mode: str = "absolute",
    delta_action_scale: float = 1.0,
    delta_feed_scale: float | None = None,
    delta_spindle_scale: float | None = None,
    delta_mapping: str = "fixed",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for episode in range(episodes):
            episode_seed = seed + 10_000 * episode + 100 * scenario_index
            env = _make_env(
                scenario=scenario,
                seed=episode_seed,
                steps=steps,
                decision_interval_s=decision_interval_s,
                margin_calibration=margin_calibration,
                randomization=randomization,
                reward_config=reward_config,
                shield_config=shield_config,
            )
            obs, _ = env.reset(seed=episode_seed)
            for step in range(steps):
                previous_feed = float(env._machine_state.last_feed_override)
                previous_spindle = float(env._machine_state.last_spindle_override)
                current_risk_now = float(obs[4])
                current_risk_horizon = float(obs[5])
                current_margin = float(obs[6])
                current_uncertainty = float(obs[8])
                raw_action, _ = model.predict(obs, deterministic=True)
                absolute_action = _policy_action_to_absolute(
                    env,
                    raw_action,
                    action_mode,
                    delta_action_scale,
                    delta_feed_scale,
                    delta_spindle_scale,
                    delta_mapping,
                )
                guarded_action, guard_info = _guard_continuous_action_with_info(
                    env,
                    absolute_action,
                    enabled=candidate_guard,
                )
                obs, reward, _, truncated, info = env.step(np.array(guarded_action, dtype=np.float32))
                reward = _apply_guard_reward_adjustment(reward, info, guard_info, reward_config)
                shield = info["shield"]
                risk = info["risk"]
                rows.append(
                    {
                        "controller": controller_name,
                        "scenario": scenario,
                        "episode": episode,
                        "step": step,
                        "previous_feed_override": previous_feed,
                        "previous_spindle_override": previous_spindle,
                        "current_risk_now": current_risk_now,
                        "current_risk_horizon": current_risk_horizon,
                        "current_margin": current_margin,
                        "current_uncertainty": current_uncertainty,
                        "action_mode": action_mode,
                        "policy_action_0": float(np.array(raw_action, dtype=np.float32).reshape(-1)[0]),
                        "policy_action_1": float(np.array(raw_action, dtype=np.float32).reshape(-1)[1]),
                        **guard_info,
                        "shield_feed_override": float(shield.feed_override),
                        "shield_spindle_override": float(shield.spindle_override),
                        "shield_rejected": int(shield.rejected),
                        "shield_reasons": ";".join(shield.reasons),
                        "reward": float(reward),
                        "risk_now": float(risk.risk_chatter_now),
                        "risk_horizon": float(risk.risk_chatter_horizon),
                        "risk_label": risk.label,
                        "risk_uncertainty": float(risk.uncertainty),
                    }
                )
                if truncated:
                    break
    return rows


def evaluate_q_policy(
    *,
    policy: TabularQPolicy,
    scenarios: tuple[str, ...],
    episodes: int,
    steps: int,
    decision_interval_s: float,
    seed: int,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for episode in range(episodes):
            episode_seed = seed + 10_000 * episode + 100 * scenario_index
            env = _make_env(
                scenario=scenario,
                seed=episode_seed,
                steps=steps,
                decision_interval_s=decision_interval_s,
                margin_calibration=margin_calibration,
                randomization=randomization,
                reward_config=reward_config,
                shield_config=shield_config or policy.shield_config,
            )
            obs, _ = env.reset(seed=episode_seed)
            row = _new_episode_row("q_learning", scenario, episode, steps)
            for _ in range(steps):
                action_index = policy.greedy_action(obs)
                feed, spindle = policy.action_to_overrides(obs, action_index)
                obs, reward, _, truncated, info = env.step(np.array([feed, spindle], dtype=np.float32))
                _update_episode_row(row, reward, info)
                if truncated:
                    break
            rows.append(_finalize_episode_row(row))
    return rows


def load_q_policy(path: Path) -> TabularQPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TabularQPolicy.from_dict(payload)


def _make_env(
    *,
    scenario: str,
    seed: int,
    steps: int,
    decision_interval_s: float,
    margin_calibration: MarginCalibration | None,
    randomization: DomainRandomizationConfig | None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> ChatterSuppressEnv:
    modal, tool, cut, sim_config = make_scenario(scenario)
    modal, cut, sim_config, _ = apply_domain_randomization(
        modal=modal,
        cut=cut,
        sim_config=sim_config,
        rng=np.random.default_rng(seed),
        config=randomization or DomainRandomizationConfig(),
    )
    sim_config = replace(
        sim_config,
        duration_s=decision_interval_s,
        random_seed=seed,
    )
    return ChatterSuppressEnv(
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
        reward_config=reward_config,
        shield_config=shield_config,
        max_steps=steps,
        decision_interval_s=decision_interval_s,
    )


def _load_sb3_algorithm(config: Sb3TrainingConfig):
    try:
        from stable_baselines3 import SAC, TD3
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
        raise RuntimeError(
            "SAC/TD3 training requires optional RL dependencies. "
            "Run with `rtk uv run --extra rl chatter-twin train-rl ...`."
        ) from exc

    match config.algorithm:
        case "sac":
            return SAC, None
        case "td3":
            noise = NormalActionNoise(mean=np.zeros(2), sigma=0.03 * np.ones(2))
            return TD3, noise
        case _:
            raise ValueError(f"Unsupported SB3 algorithm {config.algorithm!r}")


def _policy_action_space(
    env: ChatterSuppressEnv,
    action_mode: str,
    delta_action_scale: float = 1.0,
    delta_feed_scale: float | None = None,
    delta_spindle_scale: float | None = None,
):
    if action_mode == "absolute":
        return env.action_space
    if action_mode == "delta":
        feed_scale, spindle_scale = _effective_delta_scales(delta_action_scale, delta_feed_scale, delta_spindle_scale)
        return gym.spaces.Box(
            low=np.array([-feed_scale, -spindle_scale], dtype=np.float32),
            high=np.array([feed_scale, spindle_scale], dtype=np.float32),
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported action_mode {action_mode!r}")


def _effective_delta_scales(
    delta_action_scale: float,
    delta_feed_scale: float | None,
    delta_spindle_scale: float | None,
) -> tuple[float, float]:
    return (
        delta_action_scale if delta_feed_scale is None else delta_feed_scale,
        delta_action_scale if delta_spindle_scale is None else delta_spindle_scale,
    )


def _validate_optional_delta_scale(value: float | None, name: str) -> None:
    if value is not None and not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")


def _policy_action_to_absolute(
    env: ChatterSuppressEnv,
    action,
    action_mode: str,
    delta_action_scale: float = 1.0,
    delta_feed_scale: float | None = None,
    delta_spindle_scale: float | None = None,
    delta_mapping: str = "fixed",
) -> np.ndarray:
    action_array = np.array(action, dtype=np.float32).reshape(-1)
    if action_mode == "absolute":
        return action_array.astype(np.float32)
    if action_mode == "delta":
        feed_scale, spindle_scale = _effective_delta_scales(delta_action_scale, delta_feed_scale, delta_spindle_scale)
        if delta_mapping == "fixed":
            feed = env._machine_state.last_feed_override + float(action_array[0]) * env.shield_config.max_feed_delta
            spindle = (
                env._machine_state.last_spindle_override
                + float(action_array[1]) * env.shield_config.max_spindle_delta
            )
        elif delta_mapping == "headroom":
            feed = _headroom_delta_axis_to_absolute(
                previous=env._machine_state.last_feed_override,
                raw_action=float(action_array[0]),
                action_scale=feed_scale,
                lower=env.shield_config.min_feed_override,
                upper=env.shield_config.max_feed_override,
                max_delta=env.shield_config.max_feed_delta,
            )
            spindle = _headroom_delta_axis_to_absolute(
                previous=env._machine_state.last_spindle_override,
                raw_action=float(action_array[1]),
                action_scale=spindle_scale,
                lower=env.shield_config.min_spindle_override,
                upper=env.shield_config.max_spindle_override,
                max_delta=env.shield_config.max_spindle_delta,
            )
        else:
            raise ValueError(f"Unsupported delta_mapping {delta_mapping!r}")
        return np.array([feed, spindle], dtype=np.float32)
    raise ValueError(f"Unsupported action_mode {action_mode!r}")


def _headroom_delta_axis_to_absolute(
    *,
    previous: float,
    raw_action: float,
    action_scale: float,
    lower: float,
    upper: float,
    max_delta: float,
) -> float:
    if action_scale <= 0:
        raise ValueError("action_scale must be positive")
    step_fraction = _clip(raw_action / action_scale, -1.0, 1.0)
    if step_fraction >= 0:
        allowed_step = min(max_delta * action_scale, max(0.0, upper - previous))
    else:
        allowed_step = min(max_delta * action_scale, max(0.0, previous - lower))
    return previous + step_fraction * allowed_step


def _guard_continuous_action(env: ChatterSuppressEnv, action) -> np.ndarray:
    guarded_action, _ = _guard_continuous_action_with_info(env, action, enabled=True)
    return guarded_action


def _guard_continuous_action_with_info(env: ChatterSuppressEnv, action, *, enabled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    action_array = np.array(action, dtype=np.float32).reshape(-1)
    raw_feed = float(action_array[0])
    raw_spindle = float(action_array[1])
    feed = _clip(raw_feed, env.shield_config.min_feed_override, env.shield_config.max_feed_override)
    spindle = _clip(raw_spindle, env.shield_config.min_spindle_override, env.shield_config.max_spindle_override)
    info: dict[str, Any] = {
        "candidate_guard_enabled": int(enabled),
        "raw_feed_override": raw_feed,
        "raw_spindle_override": raw_spindle,
        "clipped_feed_override": feed,
        "clipped_spindle_override": spindle,
        "guard_feed_override": feed,
        "guard_spindle_override": spindle,
        "guard_fallback": 0,
        "guard_reasons": "",
        "guard_candidate_risk": "",
        "guard_candidate_horizon_risk": "",
        "guard_candidate_label": "",
        "guard_candidate_uncertainty": "",
        "guard_candidate_margin": "",
    }
    if not enabled:
        info["guard_feed_override"] = raw_feed
        info["guard_spindle_override"] = raw_spindle
        return np.array([raw_feed, raw_spindle], dtype=np.float32), info

    _, info = env._evaluate_window(feed, spindle)
    risk = info["risk"]
    guard_info = {
        "candidate_guard_enabled": 1,
        "raw_feed_override": raw_feed,
        "raw_spindle_override": raw_spindle,
        "clipped_feed_override": feed,
        "clipped_spindle_override": spindle,
        "guard_feed_override": feed,
        "guard_spindle_override": spindle,
        "guard_fallback": 0,
        "guard_reasons": "",
        "guard_candidate_risk": float(risk.risk_chatter_now),
        "guard_candidate_horizon_risk": float(risk.risk_chatter_horizon),
        "guard_candidate_label": risk.label,
        "guard_candidate_uncertainty": float(risk.uncertainty),
        "guard_candidate_margin": float(risk.margin_physics),
    }
    fallback_reasons: list[str] = []
    if risk.uncertainty > env.shield_config.max_uncertainty:
        fallback_reasons.append("high_uncertainty")
    if risk.label == "unknown":
        fallback_reasons.append("unknown_risk")
    if raw_feed != feed:
        fallback_reasons.append("feed_clipped")
    if raw_spindle != spindle:
        fallback_reasons.append("spindle_clipped")
    if risk.uncertainty > env.shield_config.max_uncertainty or risk.label == "unknown":
        guard_info["guard_fallback"] = 1
        guard_info["guard_reasons"] = ";".join(fallback_reasons)
        guard_info["guard_feed_override"] = float(env._machine_state.last_feed_override)
        guard_info["guard_spindle_override"] = float(env._machine_state.last_spindle_override)
        return np.array(
            [
                guard_info["guard_feed_override"],
                guard_info["guard_spindle_override"],
            ],
            dtype=np.float32,
        ), guard_info
    guard_info["guard_reasons"] = ";".join(fallback_reasons)
    return np.array([feed, spindle], dtype=np.float32), guard_info


def _apply_guard_reward_adjustment(
    reward: float,
    info: dict[str, Any],
    guard_info: dict[str, Any] | None,
    reward_config: RewardConfig | None,
) -> float:
    if guard_info is None:
        return float(reward)
    penalty, terms = _guard_reward_penalty(guard_info, reward_config or RewardConfig())
    if not terms:
        return float(reward)
    info.setdefault("reward_terms", {}).update(terms)
    return float(reward) - penalty


def _guard_reward_penalty(guard_info: dict[str, Any], config: RewardConfig) -> tuple[float, dict[str, float]]:
    if int(guard_info.get("candidate_guard_enabled", 0)) != 1:
        return 0.0, {}
    reasons = [reason for reason in str(guard_info.get("guard_reasons", "")).split(";") if reason]
    clip_count = sum(reason in {"feed_clipped", "spindle_clipped"} for reason in reasons)
    penalty = config.clip_penalty * clip_count
    return float(penalty), {
        "guard_clip_count": float(clip_count),
        "guard_clip_penalty": float(penalty),
    }


def _run_baseline_comparison(
    *,
    config: Sb3TrainingConfig,
    margin_calibration: MarginCalibration | None,
    randomization: DomainRandomizationConfig | None,
) -> dict[str, list[dict[str, Any]]]:
    from chatter_twin.benchmark import run_closed_loop_benchmark

    return run_closed_loop_benchmark(
        controllers=list(config.baseline_controllers),
        scenarios=list(config.scenarios),
        episodes=config.eval_episodes,
        steps=config.steps,
        out_dir=Path("/tmp/chatter_twin_rl_baseline_discard"),
        seed=config.seed + 1_000_000,
        decision_interval_s=config.decision_interval_s,
        margin_calibration=margin_calibration,
        randomization=randomization,
        reward_config=config.reward_config,
        shield_config=config.shield_config,
    )


def _discover_saved_sb3_runs(source_dir: Path) -> list[dict[str, Any]]:
    if (source_dir / "model.zip").exists():
        metrics = _read_json(source_dir / "metrics.json")
        algorithm = str(metrics.get("algorithm", ""))
        if algorithm in {"sac", "td3"}:
            config = metrics.get("config", {})
            return [{"run_dir": source_dir, "algorithm": algorithm, "seed": int(config.get("seed", 0))}]

    metrics = _read_json(source_dir / "metrics.json")
    discovered: list[dict[str, Any]] = []
    for row in metrics.get("runs", []):
        algorithm = str(row.get("algorithm", ""))
        if algorithm not in {"sac", "td3"}:
            continue
        run_dir = _resolve_saved_run_dir(source_dir, Path(str(row.get("run_dir", ""))))
        if (run_dir / "model.zip").exists():
            discovered.append(
                {
                    "run_dir": run_dir,
                    "algorithm": algorithm,
                    "seed": int(row.get("seed", 0)),
                }
            )
    if discovered:
        return discovered

    runs_dir = source_dir / "runs"
    if not runs_dir.exists():
        return []
    for model_path in sorted(runs_dir.glob("*_seed_*/model.zip")):
        run_dir = model_path.parent
        metrics = _read_json(run_dir / "metrics.json")
        algorithm = str(metrics.get("algorithm", ""))
        if algorithm not in {"sac", "td3"}:
            continue
        config = metrics.get("config", {})
        discovered.append({"run_dir": run_dir, "algorithm": algorithm, "seed": int(config.get("seed", 0))})
    return discovered


def _resolve_saved_run_dir(source_dir: Path, run_dir: Path) -> Path:
    if run_dir.exists():
        return run_dir
    if run_dir.is_absolute():
        return run_dir
    candidate = source_dir / run_dir
    if candidate.exists():
        return candidate
    candidate = source_dir.parent / run_dir
    if candidate.exists():
        return candidate
    return run_dir


def _sb3_config_from_saved_metrics(
    payload: dict[str, Any],
    *,
    algorithm: str,
    scenarios: tuple[str, ...],
    eval_episodes: int,
    steps: int,
    decision_interval_s: float,
    seed: int,
) -> Sb3TrainingConfig:
    config = payload.get("config", {})
    return Sb3TrainingConfig(
        algorithm=algorithm,
        scenarios=scenarios,
        total_timesteps=max(1, int(config.get("total_timesteps", 1))),
        eval_episodes=eval_episodes,
        steps=steps,
        decision_interval_s=decision_interval_s,
        seed=seed,
        learning_rate=float(config.get("learning_rate", 3.0e-4)),
        buffer_size=max(1, int(config.get("buffer_size", 1))),
        learning_starts=max(0, int(config.get("learning_starts", 0))),
        batch_size=max(1, int(config.get("batch_size", 1))),
        gamma=float(config.get("gamma", 0.92)),
        train_freq=max(1, int(config.get("train_freq", 1))),
        gradient_steps=max(0, int(config.get("gradient_steps", 0))),
        baseline_controllers=tuple(config.get("baseline_controllers", ("sld", "mpc"))),
        candidate_guard=bool(config.get("candidate_guard", True)),
        reward_config=RewardConfig(**config.get("reward_config", {})),
        shield_config=ShieldConfig(**config.get("shield_config", {})),
        action_mode=str(config.get("action_mode", "absolute")),
        delta_action_scale=float(config.get("delta_action_scale", 1.0)),
        delta_feed_scale=_optional_float(config.get("delta_feed_scale")),
        delta_spindle_scale=_optional_float(config.get("delta_spindle_scale")),
        delta_mapping=str(config.get("delta_mapping", "fixed")),
    )


def _saved_eval_profile_config(
    *,
    config: Sb3TrainingConfig,
    source_dir: Path,
    saved_runs: list[dict[str, Any]],
    randomization: DomainRandomizationConfig | None,
) -> dict[str, Any]:
    return {
        "source_dir": str(source_dir),
        "algorithms": tuple(sorted({str(row["algorithm"]) for row in saved_runs})),
        "seeds": tuple(int(row["seed"]) for row in saved_runs),
        "scenarios": config.scenarios,
        "eval_episodes": config.eval_episodes,
        "steps": config.steps,
        "decision_interval_s": config.decision_interval_s,
        "total_timesteps": config.total_timesteps,
        "baseline_controllers": config.baseline_controllers,
        "candidate_guard": config.candidate_guard,
        "reward_config": asdict(config.reward_config),
        "shield_config": asdict(config.shield_config),
        "action_mode": config.action_mode,
        "delta_action_scale": config.delta_action_scale,
        "delta_feed_scale": config.delta_feed_scale,
        "delta_spindle_scale": config.delta_spindle_scale,
        "delta_mapping": config.delta_mapping,
        "stress_randomization": asdict(randomization or DomainRandomizationConfig()),
    }


def _new_episode_row(controller: str, scenario: str, episode: int, max_steps: int) -> dict[str, Any]:
    return {
        "controller": controller,
        "scenario": scenario,
        "episode": episode,
        "steps": 0,
        "max_steps": max_steps,
        "total_reward": 0.0,
        "risks": [],
        "labels": [],
        "feed_overrides": [],
        "spindle_overrides": [],
        "shield_rejections": 0,
        "epsilon": 0.0,
    }


def _update_episode_row(row: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
    shield = info["shield"]
    risk = info["risk"]
    row["steps"] += 1
    row["total_reward"] += float(reward)
    row["risks"].append(float(risk.risk_chatter_now))
    row["labels"].append(risk.label)
    row["feed_overrides"].append(float(shield.feed_override))
    row["spindle_overrides"].append(float(shield.spindle_override))
    row["shield_rejections"] += int(shield.rejected)


def _finalize_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    risks = row.pop("risks")
    labels = row.pop("labels")
    feeds = row.pop("feed_overrides")
    spindles = row.pop("spindle_overrides")
    steps = int(row["steps"])
    severe_steps = sum(label == "severe" for label in labels)
    mean_feed = _mean(feeds)
    mean_spindle = _mean(spindles)
    return {
        **row,
        "total_reward": float(row["total_reward"]),
        "mean_risk": _mean(risks),
        "max_risk": max(risks) if risks else 0.0,
        "final_risk": risks[-1] if risks else 0.0,
        "final_label": labels[-1] if labels else "unknown",
        "severe_steps": severe_steps,
        "severe_fraction": severe_steps / max(steps, 1),
        "mean_feed_override": mean_feed,
        "mean_spindle_override": mean_spindle,
        "relative_mrr_proxy": mean_feed * mean_spindle,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), str(row["scenario"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (controller, scenario), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        summary.append(
            {
                "controller": controller,
                "scenario": scenario,
                "episodes": len(group),
                "mean_total_reward": _mean([row["total_reward"] for row in group]),
                "mean_risk": _mean([row["mean_risk"] for row in group]),
                "max_risk": max(float(row["max_risk"]) for row in group),
                "mean_final_risk": _mean([row["final_risk"] for row in group]),
                "severe_fraction": _mean([row["severe_fraction"] for row in group]),
                "mean_feed_override": _mean([row["mean_feed_override"] for row in group]),
                "mean_spindle_override": _mean([row["mean_spindle_override"] for row in group]),
                "shield_rejections": sum(int(row["shield_rejections"]) for row in group),
                "relative_mrr_proxy": _mean([row["relative_mrr_proxy"] for row in group]),
            }
        )
    return summary


def _aggregate_seed_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["training_algorithm"]), str(row["controller"]), str(row["scenario"]))
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (training_algorithm, controller, scenario), group in sorted(groups.items(), key=lambda item: item[0]):
        aggregate_rows.append(
            {
                "training_algorithm": training_algorithm,
                "controller": controller,
                "scenario": scenario,
                "seeds": len({int(row["seed"]) for row in group}),
                "episodes": sum(int(row["episodes"]) for row in group),
                "mean_total_reward_mean": _mean(row["mean_total_reward"] for row in group),
                "mean_total_reward_std": _std(row["mean_total_reward"] for row in group),
                "mean_risk_mean": _mean(row["mean_risk"] for row in group),
                "mean_risk_std": _std(row["mean_risk"] for row in group),
                "mean_final_risk_mean": _mean(row["mean_final_risk"] for row in group),
                "mean_final_risk_std": _std(row["mean_final_risk"] for row in group),
                "severe_fraction_mean": _mean(row["severe_fraction"] for row in group),
                "severe_fraction_std": _std(row["severe_fraction"] for row in group),
                "relative_mrr_proxy_mean": _mean(row["relative_mrr_proxy"] for row in group),
                "relative_mrr_proxy_std": _std(row["relative_mrr_proxy"] for row in group),
                "shield_rejections_sum": sum(int(row["shield_rejections"]) for row in group),
                "shield_rejections_mean": _mean(row["shield_rejections"] for row in group),
            }
        )
    return aggregate_rows


def _summarize_action_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["controller"]), str(row["scenario"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (controller, scenario), group in sorted(groups.items(), key=lambda item: item[0]):
        guard_fallbacks = sum(int(row["guard_fallback"]) for row in group)
        shield_rejections = sum(int(row["shield_rejected"]) for row in group)
        candidate_risks = [_maybe_float(row["guard_candidate_risk"]) for row in group]
        candidate_uncertainties = [_maybe_float(row["guard_candidate_uncertainty"]) for row in group]
        candidate_risks = [value for value in candidate_risks if value is not None]
        candidate_uncertainties = [value for value in candidate_uncertainties if value is not None]
        summary.append(
            {
                "controller": controller,
                "scenario": scenario,
                "steps": len(group),
                "guard_fallbacks": guard_fallbacks,
                "guard_fallback_fraction": guard_fallbacks / max(len(group), 1),
                "shield_rejections": shield_rejections,
                "shield_rejection_fraction": shield_rejections / max(len(group), 1),
                "mean_raw_feed_override": _mean(row["raw_feed_override"] for row in group),
                "mean_raw_spindle_override": _mean(row["raw_spindle_override"] for row in group),
                "mean_guard_feed_override": _mean(row["guard_feed_override"] for row in group),
                "mean_guard_spindle_override": _mean(row["guard_spindle_override"] for row in group),
                "mean_shield_feed_override": _mean(row["shield_feed_override"] for row in group),
                "mean_shield_spindle_override": _mean(row["shield_spindle_override"] for row in group),
                "mean_abs_guard_feed_delta": _mean(
                    abs(float(row["guard_feed_override"]) - float(row["raw_feed_override"])) for row in group
                ),
                "mean_abs_guard_spindle_delta": _mean(
                    abs(float(row["guard_spindle_override"]) - float(row["raw_spindle_override"])) for row in group
                ),
                "mean_abs_shield_feed_delta": _mean(
                    abs(float(row["shield_feed_override"]) - float(row["guard_feed_override"])) for row in group
                ),
                "mean_abs_shield_spindle_delta": _mean(
                    abs(float(row["shield_spindle_override"]) - float(row["guard_spindle_override"])) for row in group
                ),
                "mean_guard_candidate_risk": _mean(candidate_risks),
                "mean_guard_candidate_uncertainty": _mean(candidate_uncertainties),
                "mean_current_risk": _mean(row["current_risk_now"] for row in group),
                "mean_current_uncertainty": _mean(row["current_uncertainty"] for row in group),
                "guard_reason_counts": _count_joined(row["guard_reasons"] for row in group),
                "shield_reason_counts": _count_joined(row["shield_reasons"] for row in group),
            }
        )
    return summary


def _aggregate_seed_action_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["training_algorithm"]), str(row["controller"]), str(row["scenario"]))
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (training_algorithm, controller, scenario), group in sorted(groups.items(), key=lambda item: item[0]):
        steps = sum(int(row["steps"]) for row in group)
        guard_fallbacks = sum(int(row["guard_fallbacks"]) for row in group)
        shield_rejections = sum(int(row["shield_rejections"]) for row in group)
        aggregate_rows.append(
            {
                "training_algorithm": training_algorithm,
                "controller": controller,
                "scenario": scenario,
                "seeds": len({int(row["seed"]) for row in group}),
                "steps": steps,
                "guard_fallbacks": guard_fallbacks,
                "guard_fallback_fraction": guard_fallbacks / max(steps, 1),
                "shield_rejections": shield_rejections,
                "shield_rejection_fraction": shield_rejections / max(steps, 1),
                "mean_raw_feed_override": _mean(row["mean_raw_feed_override"] for row in group),
                "mean_guard_feed_override": _mean(row["mean_guard_feed_override"] for row in group),
                "mean_shield_feed_override": _mean(row["mean_shield_feed_override"] for row in group),
                "mean_raw_spindle_override": _mean(row["mean_raw_spindle_override"] for row in group),
                "mean_guard_spindle_override": _mean(row["mean_guard_spindle_override"] for row in group),
                "mean_shield_spindle_override": _mean(row["mean_shield_spindle_override"] for row in group),
                "mean_abs_guard_feed_delta": _mean(row["mean_abs_guard_feed_delta"] for row in group),
                "mean_abs_guard_spindle_delta": _mean(row["mean_abs_guard_spindle_delta"] for row in group),
                "mean_abs_shield_feed_delta": _mean(row["mean_abs_shield_feed_delta"] for row in group),
                "mean_abs_shield_spindle_delta": _mean(row["mean_abs_shield_spindle_delta"] for row in group),
                "mean_guard_candidate_risk": _mean(row["mean_guard_candidate_risk"] for row in group),
                "mean_guard_candidate_uncertainty": _mean(row["mean_guard_candidate_uncertainty"] for row in group),
                "mean_current_risk": _mean(row["mean_current_risk"] for row in group),
                "mean_current_uncertainty": _mean(row["mean_current_uncertainty"] for row in group),
                "guard_reason_counts": _merge_count_strings(row["guard_reason_counts"] for row in group),
                "shield_reason_counts": _merge_count_strings(row["shield_reason_counts"] for row in group),
            }
        )
    return aggregate_rows


def _diagnostic_total(payload: dict[str, Any], key: str) -> int:
    return sum(int(row.get(key, 0)) for row in payload.get("action_diagnostics", []))


def _ensure_state(q_table: dict[str, list[float]], state_key: str, action_count: int) -> None:
    q_table.setdefault(state_key, [0.0 for _ in range(action_count)])


def _argmax_stable(values: list[float]) -> int:
    best = max(values)
    return int(next(index for index, value in enumerate(values) if value == best))


def _bin(value: float, bins: tuple[float, ...]) -> int:
    return int(np.digitize([value], bins)[0])


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _mean(values) -> float:
    values = list(values)
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _std(values) -> float:
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(np.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _maybe_float(value: Any) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _count_joined(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        for item in str(value).split(";"):
            item = item.strip()
            if item:
                counts[item] = counts.get(item, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _merge_count_strings(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        for item in str(value).split(";"):
            item = item.strip()
            if not item or ":" not in item:
                continue
            key, count = item.rsplit(":", 1)
            try:
                counts[key] = counts.get(key, 0) + int(count)
            except ValueError:
                continue
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    if payload["algorithm"] in {"sac", "td3"}:
        _write_sb3_report(path, payload)
        return
    config = payload["config"]
    lines = [
        "# CPU RL Baseline Report",
        "",
        f"Algorithm: `{payload['algorithm']}`",
        f"Training episodes: `{config['episodes']}`",
        f"Evaluation episodes per scenario: `{config['eval_episodes']}`",
        f"State count: `{payload['state_count']}`",
        f"Mean reward last 10 training episodes: `{payload['training']['mean_reward_last_10']:.3f}`",
        f"Mean shield rejections last 10 training episodes: `{payload['training']['mean_shield_rejections_last_10']:.3f}`",
        "",
        "## Evaluation Summary",
        "",
        "| Scenario | Episodes | Mean risk | Final risk | Relative MRR | Mean reward | Shield rejects |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["evaluation"]["summary"]:
        lines.append(
            "| "
            f"{row['scenario']} | {row['episodes']} | "
            f"{row['mean_risk']:.3f} | {row['mean_final_risk']:.3f} | "
            f"{row['relative_mrr_proxy']:.3f} | {row['mean_total_reward']:.3f} | "
            f"{row['shield_rejections']} |"
        )
    lines.extend(
        [
            "",
            "This is a CPU-only discrete Q-learning baseline. It is useful for validating the RL pipeline, not as a replacement for SAC/TD3.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_saved_policy_eval_report(path: Path, payload: dict[str, Any]) -> None:
    config = payload["config"]
    randomization = config.get("stress_randomization", {})
    lines = [
        "# Saved RL Policy Evaluation Report",
        "",
        f"Source run: `{config['source_dir']}`",
        f"Algorithms: `{', '.join(config['algorithms'])}`",
        f"Seeds: `{', '.join(str(seed) for seed in config['seeds'])}`",
        f"Scenarios: `{', '.join(config['scenarios'])}`",
        f"Evaluation episodes per scenario per seed: `{config['eval_episodes']}`",
        f"Steps per episode: `{config['steps']}`",
        f"Randomization enabled: `{randomization.get('enabled', False)}`",
        "",
        "## Stress Randomization",
        "",
        "| Parameter | Range |",
        "|---|---|",
    ]
    for key in (
        "spindle_scale",
        "feed_scale",
        "axial_depth_scale",
        "radial_depth_scale",
        "stiffness_scale",
        "damping_scale",
        "cutting_coeff_scale",
        "noise_scale",
    ):
        value = randomization.get(key, "")
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| Training algo | Controller | Scenario | Seeds | Mean risk | Risk std | Final risk | MRR | Reward | Shield rejects |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["aggregate_summary"]:
        lines.append(
            "| "
            f"{row['training_algorithm']} | {row['controller']} | {row['scenario']} | "
            f"{row['seeds']} | {row['mean_risk_mean']:.3f} | {row['mean_risk_std']:.3f} | "
            f"{row['mean_final_risk_mean']:.3f} | {row['relative_mrr_proxy_mean']:.3f} | "
            f"{row['mean_total_reward_mean']:.3f} | {row['shield_rejections_sum']} |"
        )
    if payload.get("action_diagnostics_aggregate"):
        lines.extend(
            [
                "",
                "## Action Diagnostics",
                "",
                "| Training algo | Controller | Scenario | Steps | Guard fallbacks | Shield rejects | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in payload["action_diagnostics_aggregate"]:
            lines.append(
                "| "
                f"{row['training_algorithm']} | {row['controller']} | {row['scenario']} | "
                f"{row['steps']} | {row['guard_fallbacks']} | {row['shield_rejections']} | "
                f"{row['mean_raw_feed_override']:.3f} | {row['mean_shield_feed_override']:.3f} | "
                f"{row['mean_raw_spindle_override']:.3f} | {row['mean_shield_spindle_override']:.3f} | "
                f"{row['shield_reason_counts']} |"
            )
    lines.extend(
        [
            "",
            "This report evaluates saved SAC/TD3 checkpoints under a fresh scenario/randomization profile; it does not retrain the policies.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sb3_report(path: Path, payload: dict[str, Any]) -> None:
    config = payload["config"]
    lines = [
        "# Shielded RL Controller Report",
        "",
        f"Algorithm: `{payload['algorithm']}`",
        f"Training timesteps: `{config['total_timesteps']}`",
        f"Evaluation episodes per scenario: `{config['eval_episodes']}`",
        f"Compared baselines: `{', '.join(payload['baselines']['controllers'])}`",
        "",
        "## Comparison Summary",
        "",
        "| Controller | Scenario | Episodes | Mean risk | Final risk | Relative MRR | Mean reward | Shield rejects |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_summary"]:
        lines.append(
            "| "
            f"{row['controller']} | {row['scenario']} | {row['episodes']} | "
            f"{row['mean_risk']:.3f} | {row['mean_final_risk']:.3f} | "
            f"{row['relative_mrr_proxy']:.3f} | {row['mean_total_reward']:.3f} | "
            f"{row['shield_rejections']} |"
        )
    if payload.get("action_diagnostics"):
        lines.extend(
            [
                "",
                "## Action Diagnostics",
                "",
                "| Scenario | Steps | Guard fallbacks | Shield rejects | Current unc | Candidate unc | Raw feed | Guard feed | Shield feed | Raw spindle | Guard spindle | Shield spindle | Guard reasons | Shield reasons |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in payload["action_diagnostics"]:
            lines.append(
                "| "
                f"{row['scenario']} | {row['steps']} | "
                f"{row['guard_fallbacks']} | {row['shield_rejections']} | "
                f"{row['mean_current_uncertainty']:.3f} | {row['mean_guard_candidate_uncertainty']:.3f} | "
                f"{row['mean_raw_feed_override']:.3f} | {row['mean_guard_feed_override']:.3f} | "
                f"{row['mean_shield_feed_override']:.3f} | "
                f"{row['mean_raw_spindle_override']:.3f} | {row['mean_guard_spindle_override']:.3f} | "
                f"{row['mean_shield_spindle_override']:.3f} | "
                f"{row['guard_reason_counts']} | {row['shield_reason_counts']} |"
            )
    lines.extend(
        [
            "",
            "The learned policy is still shielded by the environment safety layer. This CPU run is a functional baseline, not a final tuned controller.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_multi_seed_report(path: Path, payload: dict[str, Any]) -> None:
    config = payload["config"]
    lines = [
        "# Multi-Seed Shielded RL Report",
        "",
        f"Algorithms: `{', '.join(config['algorithms'])}`",
        f"Seeds: `{', '.join(str(seed) for seed in config['seeds'])}`",
        f"Scenarios: `{', '.join(config['scenarios'])}`",
        f"Evaluation episodes per scenario per seed: `{config['eval_episodes']}`",
        f"Training timesteps per SAC/TD3 seed: `{config['total_timesteps']}`",
        "",
        "## Aggregate Summary",
        "",
        "| Training algo | Controller | Scenario | Seeds | Mean risk | Risk std | Final risk | MRR | Reward | Shield rejects |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_summary"]:
        lines.append(
            "| "
            f"{row['training_algorithm']} | {row['controller']} | {row['scenario']} | "
            f"{row['seeds']} | {row['mean_risk_mean']:.3f} | {row['mean_risk_std']:.3f} | "
            f"{row['mean_final_risk_mean']:.3f} | {row['relative_mrr_proxy_mean']:.3f} | "
            f"{row['mean_total_reward_mean']:.3f} | {row['shield_rejections_sum']} |"
        )
    if payload.get("action_diagnostics_aggregate"):
        lines.extend(
            [
                "",
                "## Action Diagnostics",
                "",
                "| Training algo | Controller | Scenario | Steps | Guard fallbacks | Shield rejects | Current unc | Candidate unc | Raw feed | Guard feed | Shield feed | Raw spindle | Guard spindle | Shield spindle | Guard reasons | Shield reasons |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in payload["action_diagnostics_aggregate"]:
            lines.append(
                "| "
                f"{row['training_algorithm']} | {row['controller']} | {row['scenario']} | "
                f"{row['steps']} | {row['guard_fallbacks']} | {row['shield_rejections']} | "
                f"{row['mean_current_uncertainty']:.3f} | {row['mean_guard_candidate_uncertainty']:.3f} | "
                f"{row['mean_raw_feed_override']:.3f} | {row['mean_guard_feed_override']:.3f} | "
                f"{row['mean_shield_feed_override']:.3f} | "
                f"{row['mean_raw_spindle_override']:.3f} | {row['mean_guard_spindle_override']:.3f} | "
                f"{row['mean_shield_spindle_override']:.3f} | "
                f"{row['guard_reason_counts']} | {row['shield_reason_counts']} |"
            )
    lines.extend(
        [
            "",
            "Each learned policy is still evaluated through the same safety guard and compared against matched SLD/MPC baselines for the same seed set.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_learning_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 920
    height = 260
    left = 54
    right = 24
    top = 34
    bottom = 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    rewards = [float(row["total_reward"]) for row in rows]
    if not rewards:
        path.write_text("", encoding="utf-8")
        return
    y_min = min(rewards)
    y_max = max(rewards)
    if abs(y_max - y_min) < 1.0e-9:
        y_min -= 1.0
        y_max += 1.0

    def x_pos(index: int) -> float:
        return left + plot_w * index / max(len(rewards) - 1, 1)

    def y_pos(value: float) -> float:
        return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    points = " ".join(f"{x_pos(index):.1f},{y_pos(value):.1f}" for index, value in enumerate(rewards))
    rows_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="sans-serif" font-size="18" font-weight="600">Q-learning training reward</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444"/>',
        f'<polyline points="{points}" fill="none" stroke="#2f6fbb" stroke-width="2"/>',
        f'<text x="16" y="{top + 4}" font-family="sans-serif" font-size="11">{y_max:.1f}</text>',
        f'<text x="16" y="{top + plot_h}" font-family="sans-serif" font-size="11">{y_min:.1f}</text>',
        f'<text x="{left + plot_w - 58}" y="{height - 10}" font-family="sans-serif" font-size="12">episode</text>',
        "</svg>",
    ]
    path.write_text("\n".join(rows_svg), encoding="utf-8")
