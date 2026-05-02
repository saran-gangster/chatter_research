from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RlRunRef:
    label: str
    path: Path


def compare_rl_runs(
    *,
    runs: tuple[RlRunRef, ...],
    out_dir: Path,
    baseline_label: str | None = None,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("runs must be non-empty")
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    baseline_label = baseline_label or runs[0].label
    if baseline_label not in labels:
        raise ValueError(f"baseline label {baseline_label!r} is not in the run list")

    out_dir.mkdir(parents=True, exist_ok=True)

    profiles: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    for run in runs:
        metrics = _read_json(run.path / "metrics.json")
        aggregate_rows = _read_csv(run.path / "aggregate_summary.csv")
        diagnostics_rows = _read_csv(run.path / "action_diagnostics_aggregate.csv")
        profile = _profile_row(run, metrics, aggregate_rows, diagnostics_rows)
        profiles.append(profile)
        for row in aggregate_rows:
            summary_rows.append({"profile": run.label, **row})
        for row in diagnostics_rows:
            action_rows.append({"profile": run.label, **row})

    learned_rows = [
        row for row in summary_rows if str(row.get("controller")) == str(row.get("training_algorithm"))
    ]
    candidate_rows = _candidate_summary(learned_rows, action_rows)
    delta_rows = _delta_rows(summary_rows, baseline_label)

    payload = {
        "profiles": profiles,
        "combined_summary": summary_rows,
        "learned_policy_summary": learned_rows,
        "candidate_summary": candidate_rows,
        "combined_action_diagnostics": action_rows,
        "delta_summary": delta_rows,
        "artifacts": {
            "profiles": "profiles.csv",
            "combined_summary": "combined_summary.csv",
            "learned_policy_summary": "learned_policy_summary.csv",
            "candidate_summary": "candidate_summary.csv",
            "combined_action_diagnostics": "combined_action_diagnostics.csv",
            "delta_summary": "delta_summary.csv",
            "metrics": "metrics.json",
            "report": "report.md",
        },
    }

    _write_csv(out_dir / "profiles.csv", profiles)
    _write_csv(out_dir / "combined_summary.csv", summary_rows)
    _write_csv(out_dir / "learned_policy_summary.csv", learned_rows)
    _write_csv(out_dir / "candidate_summary.csv", candidate_rows)
    _write_csv(out_dir / "combined_action_diagnostics.csv", action_rows)
    _write_csv(out_dir / "delta_summary.csv", delta_rows)
    _write_json(out_dir / "metrics.json", payload)
    _write_report(out_dir / "report.md", payload, baseline_label)
    return payload


def parse_run_ref(value: str) -> RlRunRef:
    if "=" not in value:
        raise ValueError("run references must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError("run label cannot be empty")
    if not path:
        raise ValueError("run path cannot be empty")
    return RlRunRef(label=label, path=Path(path))


def _profile_row(
    run: RlRunRef,
    metrics: dict[str, Any],
    aggregate_rows: list[dict[str, str]],
    diagnostics_rows: list[dict[str, str]],
) -> dict[str, Any]:
    config = metrics.get("config", {})
    shield_config = config.get("shield_config", {})
    reward_config = config.get("reward_config", {})
    algorithms = sorted({str(row.get("training_algorithm", "")) for row in aggregate_rows if row.get("training_algorithm")})
    learned_rows = [row for row in aggregate_rows if str(row.get("controller")) == str(row.get("training_algorithm"))]
    return {
        "profile": run.label,
        "path": str(run.path),
        "algorithms": ",".join(algorithms),
        "action_mode": config.get("action_mode", ""),
        "delta_action_scale": config.get("delta_action_scale", 1.0),
        "delta_feed_scale": config.get("delta_feed_scale", ""),
        "delta_spindle_scale": config.get("delta_spindle_scale", ""),
        "delta_mapping": config.get("delta_mapping", "fixed"),
        "uncertainty_mode": shield_config.get("uncertainty_mode", "reject"),
        "productivity_mode": reward_config.get("productivity_mode", ""),
        "productivity_weight": reward_config.get("productivity_weight", ""),
        "smoothness_weight": reward_config.get("smoothness_weight", ""),
        "clip_penalty": reward_config.get("clip_penalty", 0.0),
        "rate_limit_penalty": reward_config.get("rate_limit_penalty", 0.0),
        "summary_rows": len(aggregate_rows),
        "learned_rows": len(learned_rows),
        "action_diagnostic_rows": len(diagnostics_rows),
    }


def _delta_rows(summary_rows: list[dict[str, Any]], baseline_label: str) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        if row["profile"] != baseline_label:
            continue
        key = (str(row["training_algorithm"]), str(row["controller"]), str(row["scenario"]))
        baseline[key] = row

    deltas: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["profile"] == baseline_label:
            continue
        key = (str(row["training_algorithm"]), str(row["controller"]), str(row["scenario"]))
        base = baseline.get(key)
        if base is None:
            continue
        deltas.append(
            {
                "profile": row["profile"],
                "baseline_profile": baseline_label,
                "training_algorithm": row["training_algorithm"],
                "controller": row["controller"],
                "scenario": row["scenario"],
                "delta_mean_risk": _float(row, "mean_risk_mean") - _float(base, "mean_risk_mean"),
                "delta_relative_mrr_proxy": _float(row, "relative_mrr_proxy_mean") - _float(base, "relative_mrr_proxy_mean"),
                "delta_mean_reward": _float(row, "mean_total_reward_mean") - _float(base, "mean_total_reward_mean"),
                "delta_shield_rejections": _float(row, "shield_rejections_sum") - _float(base, "shield_rejections_sum"),
            }
        )
    return deltas


def _candidate_summary(
    learned_rows: list[dict[str, Any]],
    diagnostics_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in learned_rows:
        key = (str(row["profile"]), str(row["training_algorithm"]), str(row["controller"]))
        groups.setdefault(key, []).append(row)

    diagnostics_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in diagnostics_rows:
        key = (str(row["profile"]), str(row["training_algorithm"]), str(row["controller"]))
        diagnostics_by_key.setdefault(key, []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    for (profile, training_algorithm, controller), rows in groups.items():
        diagnostics = diagnostics_by_key.get((profile, training_algorithm, controller), [])
        scenario_names = sorted({str(row["scenario"]) for row in rows})
        candidate_rows.append(
            {
                "profile": profile,
                "training_algorithm": training_algorithm,
                "controller": controller,
                "scenarios": ",".join(scenario_names),
                "scenario_count": len(scenario_names),
                "mean_risk_avg": _mean(_float(row, "mean_risk_mean") for row in rows),
                "mean_risk_worst": max((_float(row, "mean_risk_mean") for row in rows), default=0.0),
                "unstable_risk": _scenario_float(rows, "unstable", "mean_risk_mean"),
                "near_boundary_risk": _scenario_float(rows, "near_boundary", "mean_risk_mean"),
                "relative_mrr_avg": _mean(_float(row, "relative_mrr_proxy_mean") for row in rows),
                "relative_mrr_min": min((_float(row, "relative_mrr_proxy_mean") for row in rows), default=0.0),
                "unstable_relative_mrr": _scenario_float(rows, "unstable", "relative_mrr_proxy_mean"),
                "mean_reward_avg": _mean(_float(row, "mean_total_reward_mean") for row in rows),
                "shield_rejections_sum": sum(_float(row, "shield_rejections_sum") for row in rows),
                "guard_fallbacks_sum": sum(_float(row, "guard_fallbacks") for row in diagnostics),
                "mean_raw_feed_override": _mean(_float(row, "mean_raw_feed_override") for row in diagnostics),
                "mean_shield_feed_override": _mean(_float(row, "mean_shield_feed_override") for row in diagnostics),
                "mean_raw_spindle_override": _mean(_float(row, "mean_raw_spindle_override") for row in diagnostics),
                "mean_shield_spindle_override": _mean(_float(row, "mean_shield_spindle_override") for row in diagnostics),
                "is_pareto_frontier": 0,
            }
        )

    _mark_pareto_frontier(candidate_rows)
    return candidate_rows


def _mark_pareto_frontier(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                _float(other, "mean_risk_avg") <= _float(row, "mean_risk_avg")
                and _float(other, "relative_mrr_avg") >= _float(row, "relative_mrr_avg")
                and _float(other, "shield_rejections_sum") <= _float(row, "shield_rejections_sum")
            )
            strictly_better = (
                _float(other, "mean_risk_avg") < _float(row, "mean_risk_avg")
                or _float(other, "relative_mrr_avg") > _float(row, "relative_mrr_avg")
                or _float(other, "shield_rejections_sum") < _float(row, "shield_rejections_sum")
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row["is_pareto_frontier"] = int(not dominated)


def _write_report(path: Path, payload: dict[str, Any], baseline_label: str) -> None:
    lines: list[str] = [
        "# RL Run Comparison",
        "",
        f"Baseline profile for deltas: `{baseline_label}`.",
        "",
        "## Profiles",
        "",
        "| Profile | Algorithms | Action mode | Delta scale | Feed scale | Spindle scale | Delta mapping | Uncertainty mode | Productivity mode | Productivity weight | Clip penalty | Rate-limit penalty | Summary rows |",
        "|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["profiles"]:
        lines.append(
            "| {profile} | {algorithms} | {action_mode} | {delta_action_scale} | {delta_feed_scale} | {delta_spindle_scale} | {delta_mapping} | {uncertainty_mode} | {productivity_mode} | {productivity_weight} | {clip_penalty} | {rate_limit_penalty} | {summary_rows} |".format(
                **row
            )
        )
    reward_signatures = {
        (
            str(row["productivity_mode"]),
            str(row["productivity_weight"]),
            str(row["clip_penalty"]),
            str(row["rate_limit_penalty"]),
        )
        for row in payload["profiles"]
    }
    if len(reward_signatures) > 1:
        lines.extend(
            [
                "",
                "Note: reward settings differ across profiles, so compare risk, MRR, and actions first. Reward deltas are not normalized across these profiles.",
            ]
        )

    lines.extend(
        [
            "",
            "## Learned Policies",
            "",
            "| Profile | Algorithm | Scenario | Mean risk | Risk std | Relative MRR | Mean reward | Shield rejects |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _sort_rows(payload["learned_policy_summary"]):
        lines.append(
            "| {profile} | {training_algorithm} | {scenario} | {risk:.3f} | {risk_std:.3f} | {mrr:.3f} | {reward:.3f} | {rejects:g} |".format(
                profile=row["profile"],
                training_algorithm=row["training_algorithm"],
                scenario=row["scenario"],
                risk=_float(row, "mean_risk_mean"),
                risk_std=_float(row, "mean_risk_std"),
                mrr=_float(row, "relative_mrr_proxy_mean"),
                reward=_float(row, "mean_total_reward_mean"),
                rejects=_float(row, "shield_rejections_sum"),
            )
        )

    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "| Profile | Algorithm | Avg risk | Worst risk | Unstable risk | Avg MRR | Min MRR | Unstable MRR | Guard fallbacks | Pareto |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        payload["candidate_summary"],
        key=lambda item: (_float(item, "mean_risk_avg"), -_float(item, "relative_mrr_avg"), str(item.get("profile", ""))),
    ):
        lines.append(
            "| {profile} | {training_algorithm} | {risk:.3f} | {worst:.3f} | {unstable_risk:.3f} | {mrr:.3f} | {min_mrr:.3f} | {unstable_mrr:.3f} | {fallbacks:g} | {pareto:g} |".format(
                profile=row["profile"],
                training_algorithm=row["training_algorithm"],
                risk=_float(row, "mean_risk_avg"),
                worst=_float(row, "mean_risk_worst"),
                unstable_risk=_float(row, "unstable_risk"),
                mrr=_float(row, "relative_mrr_avg"),
                min_mrr=_float(row, "relative_mrr_min"),
                unstable_mrr=_float(row, "unstable_relative_mrr"),
                fallbacks=_float(row, "guard_fallbacks_sum"),
                pareto=_float(row, "is_pareto_frontier"),
            )
        )

    lines.extend(
        [
            "",
            "## Learned-Policy Action Diagnostics",
            "",
            "| Profile | Algorithm | Scenario | Guard fallbacks | Shield rejects | Raw feed | Shield feed | Raw spindle | Shield spindle | Shield reasons |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    learned_keys = {
        (row["profile"], row["training_algorithm"], row["controller"], row["scenario"])
        for row in payload["learned_policy_summary"]
    }
    diagnostics = [
        row
        for row in payload["combined_action_diagnostics"]
        if (row["profile"], row["training_algorithm"], row["controller"], row["scenario"]) in learned_keys
    ]
    for row in _sort_rows(diagnostics):
        lines.append(
            "| {profile} | {training_algorithm} | {scenario} | {fallbacks:g} | {rejects:g} | {raw_feed:.3f} | {shield_feed:.3f} | {raw_spindle:.3f} | {shield_spindle:.3f} | {reasons} |".format(
                profile=row["profile"],
                training_algorithm=row["training_algorithm"],
                scenario=row["scenario"],
                fallbacks=_float(row, "guard_fallbacks"),
                rejects=_float(row, "shield_rejections"),
                raw_feed=_float(row, "mean_raw_feed_override"),
                shield_feed=_float(row, "mean_shield_feed_override"),
                raw_spindle=_float(row, "mean_raw_spindle_override"),
                shield_spindle=_float(row, "mean_shield_spindle_override"),
                reasons=row.get("shield_reason_counts", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Deltas Against Baseline",
            "",
            "| Profile | Controller | Scenario | Delta risk | Delta MRR | Delta reward | Delta rejects |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in _sort_rows(payload["delta_summary"]):
        lines.append(
            "| {profile} | {controller} | {scenario} | {risk:.3f} | {mrr:.3f} | {reward:.3f} | {rejects:g} |".format(
                profile=row["profile"],
                controller=row["controller"],
                scenario=row["scenario"],
                risk=_float(row, "delta_mean_risk"),
                mrr=_float(row, "delta_relative_mrr_proxy"),
                reward=_float(row, "delta_mean_reward"),
                rejects=_float(row, "delta_shield_rejections"),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if value in (None, ""):
        return 0.0
    return float(value)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _scenario_float(rows: list[dict[str, Any]], scenario: str, key: str) -> float:
    matches = [_float(row, key) for row in rows if str(row.get("scenario")) == scenario]
    return _mean(matches)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("profile", "")),
            str(row.get("training_algorithm", "")),
            str(row.get("controller", "")),
            str(row.get("scenario", "")),
        ),
    )
