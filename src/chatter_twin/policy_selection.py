from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicySelectionConfig:
    profile_label: str = "safety_first"
    min_relative_mrr: float = 1.0
    risk_weight: float = 1.0
    worst_risk_weight: float = 0.5
    unstable_risk_weight: float = 0.5
    mrr_weight: float = 0.05
    mrr_shortfall_weight: float = 3.0
    guard_fallback_weight: float = 0.15
    shield_rejection_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.min_relative_mrr < 0:
            raise ValueError("min_relative_mrr cannot be negative")
        for name, value in asdict(self).items():
            if name == "profile_label":
                continue
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


def select_rl_policy(
    *,
    eval_dir: Path,
    out_dir: Path,
    config: PolicySelectionConfig = PolicySelectionConfig(),
) -> dict[str, Any]:
    run_rows = _read_csv(eval_dir / "run_summary.csv")
    if not run_rows:
        raise ValueError(f"No run_summary.csv rows found in {eval_dir}")

    candidates: list[dict[str, Any]] = []
    for run in run_rows:
        run_dir = Path(str(run["run_dir"]))
        summary_rows = _read_csv(run_dir / "evaluation_summary.csv")
        diagnostic_rows = _read_csv(run_dir / "action_diagnostics_summary.csv")
        if not summary_rows:
            continue
        candidates.append(_candidate_row(eval_dir, run, summary_rows, diagnostic_rows, config))

    if not candidates:
        raise ValueError(f"No policy candidates could be read from {eval_dir}")

    candidates = sorted(candidates, key=lambda row: (float(row["selection_score"]), int(row["seed"])))
    selected = {**candidates[0], "selected": True}
    candidate_rows = [{**row, "selected": row["seed"] == selected["seed"]} for row in candidates]
    payload = {
        "profile_label": config.profile_label,
        "eval_dir": str(eval_dir),
        "selection_config": asdict(config),
        "selected": selected,
        "candidates": candidate_rows,
        "deployment_boundary": {
            "mode": "shadow_only",
            "cnc_writes_enabled": False,
            "requires_safety_shield": True,
            "requires_human_review": True,
        },
        "artifacts": {
            "candidate_ranking": "candidate_ranking.csv",
            "selected_policy": "selected_policy.json",
            "policy_card": "policy_card.md",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "candidate_ranking.csv", candidate_rows)
    _write_json(out_dir / "selected_policy.json", payload)
    _write_policy_card(out_dir / "policy_card.md", payload)
    return payload


def _candidate_row(
    eval_dir: Path,
    run: dict[str, str],
    summary_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    config: PolicySelectionConfig,
) -> dict[str, Any]:
    mean_risk_avg = _mean(_float(row, "mean_risk") for row in summary_rows)
    worst_scenario_risk = max((_float(row, "mean_risk") for row in summary_rows), default=0.0)
    max_window_risk = max((_float(row, "max_risk") for row in summary_rows), default=0.0)
    unstable_risk = _scenario_float(summary_rows, "unstable", "mean_risk")
    near_boundary_risk = _scenario_float(summary_rows, "near_boundary", "mean_risk")
    relative_mrr_avg = _mean(_float(row, "relative_mrr_proxy") for row in summary_rows)
    relative_mrr_min = min((_float(row, "relative_mrr_proxy") for row in summary_rows), default=0.0)
    unstable_relative_mrr = _scenario_float(summary_rows, "unstable", "relative_mrr_proxy")
    shield_rejections = sum(_int(row, "shield_rejections") for row in summary_rows)
    guard_fallbacks = sum(_int(row, "guard_fallbacks") for row in diagnostic_rows)
    guard_steps = sum(_int(row, "steps") for row in diagnostic_rows)
    guard_fallback_fraction = guard_fallbacks / max(guard_steps, 1)
    mrr_shortfall = max(0.0, config.min_relative_mrr - relative_mrr_min)
    score = (
        config.risk_weight * mean_risk_avg
        + config.worst_risk_weight * worst_scenario_risk
        + config.unstable_risk_weight * unstable_risk
        + config.mrr_shortfall_weight * mrr_shortfall
        + config.guard_fallback_weight * guard_fallback_fraction
        + config.shield_rejection_weight * shield_rejections
        - config.mrr_weight * relative_mrr_avg
    )
    source_run = str(run.get("source_run", ""))
    source_model_path = str(Path(source_run) / "model.zip") if source_run else ""
    return {
        "profile_label": config.profile_label,
        "seed": int(run["seed"]),
        "algorithm": run["algorithm"],
        "eval_dir": str(eval_dir),
        "eval_run_dir": run["run_dir"],
        "source_run": source_run,
        "source_model_path": source_model_path,
        "selection_score": score,
        "mean_risk_avg": mean_risk_avg,
        "worst_scenario_risk": worst_scenario_risk,
        "max_window_risk": max_window_risk,
        "unstable_risk": unstable_risk,
        "near_boundary_risk": near_boundary_risk,
        "relative_mrr_avg": relative_mrr_avg,
        "relative_mrr_min": relative_mrr_min,
        "unstable_relative_mrr": unstable_relative_mrr,
        "mrr_shortfall": mrr_shortfall,
        "guard_fallbacks": guard_fallbacks,
        "guard_fallback_fraction": guard_fallback_fraction,
        "shield_rejections": shield_rejections,
    }


def _write_policy_card(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected"]
    config = payload["selection_config"]
    lines = [
        "# RL Shadow Policy Card",
        "",
        f"Profile: `{payload['profile_label']}`",
        f"Selected seed: `{selected['seed']}`",
        f"Algorithm: `{selected['algorithm']}`",
        f"Source model: `{selected['source_model_path']}`",
        f"Evaluation source: `{payload['eval_dir']}`",
        "",
        "## Deployment Boundary",
        "",
        "| Boundary | Value |",
        "|---|---|",
        "| Mode | `shadow_only` |",
        "| CNC writes enabled | `false` |",
        "| Safety shield required | `true` |",
        "| Human review required | `true` |",
        "",
        "## Selection Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Selection score | {float(selected['selection_score']):.3f} |",
        f"| Average risk | {float(selected['mean_risk_avg']):.3f} |",
        f"| Worst scenario risk | {float(selected['worst_scenario_risk']):.3f} |",
        f"| Max window risk | {float(selected['max_window_risk']):.3f} |",
        f"| Unstable risk | {float(selected['unstable_risk']):.3f} |",
        f"| Near-boundary risk | {float(selected['near_boundary_risk']):.3f} |",
        f"| Average relative MRR | {float(selected['relative_mrr_avg']):.3f} |",
        f"| Minimum relative MRR | {float(selected['relative_mrr_min']):.3f} |",
        f"| Unstable relative MRR | {float(selected['unstable_relative_mrr']):.3f} |",
        f"| Guard fallbacks | {int(selected['guard_fallbacks'])} |",
        f"| Shield rejections | {int(selected['shield_rejections'])} |",
        "",
        "## Selection Objective",
        "",
        "| Term | Weight |",
        "|---|---:|",
        f"| Average risk | {float(config['risk_weight']):.3f} |",
        f"| Worst scenario risk | {float(config['worst_risk_weight']):.3f} |",
        f"| Unstable risk | {float(config['unstable_risk_weight']):.3f} |",
        f"| MRR reward | {float(config['mrr_weight']):.3f} |",
        f"| Minimum MRR shortfall | {float(config['mrr_shortfall_weight']):.3f} |",
        f"| Guard fallback fraction | {float(config['guard_fallback_weight']):.3f} |",
        f"| Shield rejections | {float(config['shield_rejection_weight']):.3f} |",
        f"| Minimum relative MRR target | {float(config['min_relative_mrr']):.3f} |",
        "",
        "This card is for shadow-mode review only. It is not a hardware actuation approval.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _scenario_float(rows: list[dict[str, str]], scenario: str, key: str) -> float:
    return _mean(_float(row, key) for row in rows if row.get("scenario") == scenario)
