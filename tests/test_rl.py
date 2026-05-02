import csv
import json

import pytest

from chatter_twin.cli import main
from chatter_twin.env import ChatterSuppressEnv
from chatter_twin.models import MachineState
from chatter_twin.rl import (
    MultiSeedTrainingConfig,
    QLearningConfig,
    Sb3TrainingConfig,
    _effective_delta_scales,
    _policy_action_to_absolute,
    train_multi_seed_policies,
    train_q_learning,
)
from chatter_twin.rl_compare import RlRunRef, compare_rl_runs, parse_run_ref
from chatter_twin.replay import DomainRandomizationConfig


def test_train_q_learning_writes_artifacts(tmp_path):
    payload = train_q_learning(
        config=QLearningConfig(
            scenarios=("stable",),
            episodes=3,
            eval_episodes=1,
            steps=2,
            decision_interval_s=0.05,
            seed=42,
        ),
        out_dir=tmp_path,
        randomization=DomainRandomizationConfig(enabled=True),
    )

    assert payload["algorithm"] == "tabular_q_learning"
    assert payload["evaluation"]["episodes"] == 1
    assert (tmp_path / "policy.json").exists()
    assert (tmp_path / "learning_curve.svg").exists()

    with (tmp_path / "evaluation_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["controller"] == "q_learning"


def test_cli_train_rl_q_learning_runs(tmp_path):
    out_dir = tmp_path / "rl"
    status = main(
        [
            "train-rl",
            "--algorithm",
            "q_learning",
            "--scenarios",
            "stable",
            "--episodes",
            "2",
            "--eval-episodes",
            "1",
            "--steps",
            "2",
            "--decision-interval",
            "0.05",
            "--randomize",
            "--out",
            str(out_dir),
        ]
    )

    assert status == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["algorithm"] == "tabular_q_learning"


def test_sb3_config_accepts_sac_and_td3():
    assert Sb3TrainingConfig(algorithm="sac").algorithm == "sac"
    assert Sb3TrainingConfig(algorithm="td3").algorithm == "td3"
    assert Sb3TrainingConfig(algorithm="sac", action_mode="delta").action_mode == "delta"
    assert Sb3TrainingConfig(algorithm="sac", action_mode="delta", delta_action_scale=0.5).delta_action_scale == 0.5
    config = Sb3TrainingConfig(algorithm="sac", action_mode="delta", delta_feed_scale=0.35, delta_spindle_scale=0.5)
    assert config.delta_feed_scale == 0.35
    assert config.delta_spindle_scale == 0.5
    assert Sb3TrainingConfig(algorithm="sac", action_mode="delta", delta_mapping="headroom").delta_mapping == "headroom"
    assert _effective_delta_scales(0.8, None, 0.4) == (0.8, 0.4)
    with pytest.raises(ValueError):
        Sb3TrainingConfig(algorithm="sac", action_mode="bad")
    with pytest.raises(ValueError):
        Sb3TrainingConfig(algorithm="sac", delta_action_scale=0.0)
    with pytest.raises(ValueError):
        Sb3TrainingConfig(algorithm="sac", delta_feed_scale=1.5)
    with pytest.raises(ValueError):
        Sb3TrainingConfig(algorithm="sac", delta_mapping="bad")


def test_headroom_delta_mapping_uses_remaining_override_range():
    env = ChatterSuppressEnv(max_steps=1)
    env._machine_state = MachineState(last_feed_override=1.04, last_spindle_override=1.09)

    fixed = _policy_action_to_absolute(env, [0.5, 0.5], "delta", 0.5, None, None, "fixed")
    headroom = _policy_action_to_absolute(env, [0.5, 0.5], "delta", 0.5, None, None, "headroom")

    assert fixed[0] > env.shield_config.max_feed_override
    assert fixed[1] > env.shield_config.max_spindle_override
    assert headroom[0] == pytest.approx(env.shield_config.max_feed_override)
    assert headroom[1] == pytest.approx(env.shield_config.max_spindle_override)


def test_train_rl_multiseed_q_learning_aggregates(tmp_path):
    payload = train_multi_seed_policies(
        config=MultiSeedTrainingConfig(
            algorithms=("q_learning",),
            seeds=(11, 22),
            scenarios=("stable",),
            episodes=2,
            eval_episodes=1,
            steps=2,
            decision_interval_s=0.05,
            learning_rate=0.2,
        ),
        out_dir=tmp_path,
        randomization=DomainRandomizationConfig(enabled=True),
    )

    assert len(payload["runs"]) == 2
    assert len(payload["aggregate_summary"]) == 1
    assert payload["aggregate_summary"][0]["training_algorithm"] == "q_learning"
    assert payload["aggregate_summary"][0]["seeds"] == 2
    assert (tmp_path / "runs" / "q_learning_seed_11" / "policy.json").exists()
    assert (tmp_path / "aggregate_summary.csv").exists()


def test_compare_rl_runs_writes_report(tmp_path):
    strict_dir = tmp_path / "strict"
    hold_dir = tmp_path / "hold"
    _write_compare_fixture(strict_dir, "sac", "reject", risk=0.37, mrr=1.02, reward=2.0, rejects=8)
    _write_compare_fixture(hold_dir, "sac", "hold", risk=0.36, mrr=0.99, reward=2.3, rejects=0)

    payload = compare_rl_runs(
        runs=(
            RlRunRef("strict", strict_dir),
            RlRunRef("hold", hold_dir),
        ),
        baseline_label="strict",
        out_dir=tmp_path / "comparison",
    )

    assert len(payload["profiles"]) == 2
    assert len(payload["learned_policy_summary"]) == 2
    assert len(payload["candidate_summary"]) == 2
    assert payload["delta_summary"][0]["delta_shield_rejections"] == -8.0
    report = (tmp_path / "comparison" / "report.md").read_text(encoding="utf-8")
    assert "RL Run Comparison" in report
    assert "Candidate Summary" in report
    assert "hold" in report


def test_cli_compare_rl_runs(tmp_path):
    run_dir = tmp_path / "run"
    _write_compare_fixture(run_dir, "td3", "hold", risk=0.46, mrr=1.05, reward=-0.9, rejects=0)
    out_dir = tmp_path / "comparison"

    status = main(
        [
            "compare-rl-runs",
            "--run",
            f"td3_hold={run_dir}",
            "--out",
            str(out_dir),
        ]
    )

    assert status == 0
    assert (out_dir / "combined_summary.csv").exists()
    assert (out_dir / "candidate_summary.csv").exists()
    assert (out_dir / "report.md").exists()
    assert parse_run_ref(f"td3_hold={run_dir}").label == "td3_hold"


def _write_compare_fixture(run_dir, algorithm: str, uncertainty_mode: str, *, risk: float, mrr: float, reward: float, rejects: int):
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": {
                    "action_mode": "delta",
                    "delta_action_scale": 1.0,
                    "delta_feed_scale": None,
                    "delta_spindle_scale": None,
                    "delta_mapping": "fixed",
                    "shield_config": {"uncertainty_mode": uncertainty_mode},
                    "reward_config": {
                        "productivity_mode": "mrr",
                        "productivity_weight": 2.0,
                        "smoothness_weight": 0.1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        run_dir / "aggregate_summary.csv",
        [
            {
                "training_algorithm": algorithm,
                "controller": algorithm,
                "scenario": "near_boundary",
                "seeds": 3,
                "episodes": 12,
                "mean_total_reward_mean": reward,
                "mean_total_reward_std": 0.1,
                "mean_risk_mean": risk,
                "mean_risk_std": 0.02,
                "mean_final_risk_mean": risk,
                "mean_final_risk_std": 0.02,
                "severe_fraction_mean": 0.0,
                "severe_fraction_std": 0.0,
                "relative_mrr_proxy_mean": mrr,
                "relative_mrr_proxy_std": 0.01,
                "shield_rejections_sum": rejects,
                "shield_rejections_mean": rejects / 3,
            }
        ],
    )
    _write_csv(
        run_dir / "action_diagnostics_aggregate.csv",
        [
            {
                "training_algorithm": algorithm,
                "controller": algorithm,
                "scenario": "near_boundary",
                "seeds": 3,
                "steps": 96,
                "guard_fallbacks": 1,
                "guard_fallback_fraction": 0.01,
                "shield_rejections": rejects,
                "shield_rejection_fraction": 0.01,
                "mean_raw_feed_override": 1.0,
                "mean_guard_feed_override": 1.0,
                "mean_shield_feed_override": 1.0,
                "mean_raw_spindle_override": 1.0,
                "mean_guard_spindle_override": 1.0,
                "mean_shield_spindle_override": 1.0,
                "mean_abs_guard_feed_delta": 0.0,
                "mean_abs_guard_spindle_delta": 0.0,
                "mean_abs_shield_feed_delta": 0.0,
                "mean_abs_shield_spindle_delta": 0.0,
                "mean_guard_candidate_risk": risk,
                "mean_guard_candidate_uncertainty": 0.56,
                "mean_current_risk": risk,
                "mean_current_uncertainty": 0.56,
                "guard_reason_counts": "",
                "shield_reason_counts": "high_uncertainty:8" if rejects else "",
            }
        ],
    )


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
