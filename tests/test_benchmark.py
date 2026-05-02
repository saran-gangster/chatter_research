import csv
import json
from pathlib import Path

from chatter_twin.benchmark import run_benchmark, run_closed_loop_benchmark, run_closed_loop_episode, run_episode, summarize_metrics
from chatter_twin.cli import main
from chatter_twin.replay import DomainRandomizationConfig


def test_run_episode_returns_core_metrics():
    metrics = run_episode(
        controller_name="fixed",
        scenario_name="stable",
        episode=0,
        steps=3,
        seed=123,
    )
    assert metrics.steps == 3
    assert 0.0 <= metrics.mean_risk <= 1.0
    assert 0.0 <= metrics.relative_mrr_proxy
    assert metrics.final_label in {"stable", "transition", "slight", "severe", "unknown"}


def test_summarize_metrics_groups_by_controller_and_scenario():
    rows = [
        run_episode(controller_name="fixed", scenario_name="stable", episode=0, steps=2, seed=1),
        run_episode(controller_name="fixed", scenario_name="stable", episode=1, steps=2, seed=2),
        run_episode(controller_name="rule", scenario_name="unstable", episode=0, steps=2, seed=3),
    ]
    summary = summarize_metrics(rows)
    assert len(summary) == 2
    assert {row.controller for row in summary} == {"fixed", "rule"}


def test_run_benchmark_writes_artifacts(tmp_path: Path):
    payload = run_benchmark(
        controllers=["fixed", "rule", "sld", "mpc"],
        scenarios=["stable"],
        episodes=2,
        steps=3,
        out_dir=tmp_path,
    )
    assert len(payload["episodes"]) == 8
    assert len(payload["summary"]) == 4
    for name in ["episodes.csv", "summary.csv", "summary.json", "summary.md", "risk_vs_mrr.svg", "pareto.csv", "pareto.svg"]:
        assert (tmp_path / name).exists()

    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8

    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(data) == {"episodes", "summary"}
    assert "Relative MRR" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "<svg" in (tmp_path / "risk_vs_mrr.svg").read_text(encoding="utf-8")
    assert "<svg" in (tmp_path / "pareto.svg").read_text(encoding="utf-8")


def test_closed_loop_episode_returns_core_metrics():
    metrics = run_closed_loop_episode(
        controller_name="fixed",
        scenario_name="stable",
        episode=0,
        steps=3,
        seed=123,
        decision_interval_s=0.05,
    )
    assert metrics.steps == 3
    assert 0.0 <= metrics.mean_risk <= 1.0
    assert metrics.final_label in {"stable", "transition", "slight", "severe", "unknown"}


def test_closed_loop_benchmark_writes_artifacts(tmp_path: Path):
    payload = run_closed_loop_benchmark(
        controllers=["fixed", "hybrid"],
        scenarios=["stable"],
        episodes=1,
        steps=2,
        decision_interval_s=0.05,
        out_dir=tmp_path,
    )
    assert len(payload["episodes"]) == 2
    assert len(payload["summary"]) == 2
    for name in ["episodes.csv", "summary.csv", "summary.json", "summary.md", "risk_vs_mrr.svg", "pareto.csv", "pareto.svg"]:
        assert (tmp_path / name).exists()


def test_closed_loop_benchmark_supports_domain_randomization(tmp_path: Path):
    payload = run_closed_loop_benchmark(
        controllers=["fixed", "mpc"],
        scenarios=["stable"],
        episodes=2,
        steps=2,
        decision_interval_s=0.05,
        out_dir=tmp_path,
        randomization=DomainRandomizationConfig(enabled=True),
    )

    assert len(payload["episodes"]) == 4
    assert all(row["randomized"] for row in payload["episodes"])
    assert any(abs(row["axial_depth_scale"] - 1.0) > 1.0e-6 for row in payload["episodes"])

    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["randomized"] == "True"
    assert "cutting_coeff_scale" in rows[0]


def test_cli_benchmark_runs(tmp_path: Path):
    out_dir = tmp_path / "bench"
    status = main(
        [
            "benchmark",
            "--controllers",
            "fixed,rule,sld,mpc",
            "--scenarios",
            "stable",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--out",
            str(out_dir),
        ]
    )
    assert status == 0
    assert (out_dir / "summary.csv").exists()


def test_cli_closed_loop_benchmark_runs(tmp_path: Path):
    out_dir = tmp_path / "closed"
    status = main(
        [
            "closed-loop-benchmark",
            "--controllers",
            "fixed,hybrid",
            "--scenarios",
            "stable",
            "--episodes",
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
    assert (out_dir / "pareto.csv").exists()

    with (out_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["randomized"] == "True"
