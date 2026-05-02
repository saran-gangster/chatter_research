import csv
import json

from chatter_twin.cli import main
from chatter_twin.policy_selection import PolicySelectionConfig, select_rl_policy


def test_select_rl_policy_penalizes_mrr_shortfall(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_policy_eval_fixture(eval_dir)

    payload = select_rl_policy(
        eval_dir=eval_dir,
        out_dir=tmp_path / "selected",
        config=PolicySelectionConfig(mrr_shortfall_weight=5.0),
    )

    assert payload["selected"]["seed"] == 22
    assert payload["deployment_boundary"]["mode"] == "shadow_only"
    assert (tmp_path / "selected" / "selected_policy.json").exists()
    assert (tmp_path / "selected" / "candidate_ranking.csv").exists()
    assert "shadow-mode" in (tmp_path / "selected" / "policy_card.md").read_text(encoding="utf-8")


def test_cli_select_rl_policy_runs(tmp_path):
    eval_dir = tmp_path / "eval"
    out_dir = tmp_path / "selected"
    _write_policy_eval_fixture(eval_dir)

    status = main(
        [
            "select-rl-policy",
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out_dir),
            "--mrr-shortfall-weight",
            "5.0",
        ]
    )

    assert status == 0
    payload = json.loads((out_dir / "selected_policy.json").read_text(encoding="utf-8"))
    assert payload["selected"]["seed"] == 22


def _write_policy_eval_fixture(eval_dir):
    eval_dir.mkdir(parents=True)
    _write_csv(
        eval_dir / "run_summary.csv",
        [
            {
                "algorithm": "td3",
                "seed": 11,
                "source_run": "source/td3_seed_11",
                "run_dir": str(eval_dir / "runs" / "td3_seed_11"),
                "evaluation_episodes": 4,
                "guard_fallbacks": 0,
                "shield_rejections": 0,
            },
            {
                "algorithm": "td3",
                "seed": 22,
                "source_run": "source/td3_seed_22",
                "run_dir": str(eval_dir / "runs" / "td3_seed_22"),
                "evaluation_episodes": 4,
                "guard_fallbacks": 4,
                "shield_rejections": 0,
            },
        ],
    )
    _write_run(eval_dir / "runs" / "td3_seed_11", risk=0.25, unstable_risk=0.35, mrr=0.85, fallbacks=0)
    _write_run(eval_dir / "runs" / "td3_seed_22", risk=0.32, unstable_risk=0.38, mrr=1.05, fallbacks=4)


def _write_run(run_dir, *, risk: float, unstable_risk: float, mrr: float, fallbacks: int):
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "evaluation_summary.csv",
        [
            {
                "controller": "td3",
                "scenario": "stable",
                "episodes": 2,
                "mean_total_reward": 1.0,
                "mean_risk": risk,
                "max_risk": risk + 0.1,
                "mean_final_risk": risk,
                "severe_fraction": 0.0,
                "mean_feed_override": 1.0,
                "mean_spindle_override": mrr,
                "shield_rejections": 0,
                "relative_mrr_proxy": mrr,
            },
            {
                "controller": "td3",
                "scenario": "unstable",
                "episodes": 2,
                "mean_total_reward": -1.0,
                "mean_risk": unstable_risk,
                "max_risk": unstable_risk + 0.1,
                "mean_final_risk": unstable_risk,
                "severe_fraction": 0.0,
                "mean_feed_override": 1.0,
                "mean_spindle_override": mrr,
                "shield_rejections": 0,
                "relative_mrr_proxy": mrr,
            },
        ],
    )
    _write_csv(
        run_dir / "action_diagnostics_summary.csv",
        [
            {
                "controller": "td3",
                "scenario": "stable",
                "steps": 10,
                "guard_fallbacks": fallbacks,
            }
        ],
    )


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
