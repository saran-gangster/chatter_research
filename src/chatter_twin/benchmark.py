from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from chatter_twin.calibration import MarginCalibration, apply_margin_calibration, calibrated_margin_uncertainty
from chatter_twin.controllers import make_controller
from chatter_twin.dynamics import simulate_milling
from chatter_twin.env import ChatterSuppressEnv, compute_control_reward
from chatter_twin.features import extract_signal_features
from chatter_twin.models import MachineState, RewardConfig, ShieldConfig, SimulationConfig
from chatter_twin.replay import DomainRandomizationConfig, apply_domain_randomization
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.scenarios import make_scenario
from chatter_twin.shield import apply_safety_shield
from chatter_twin.stability import signed_stability_margin


@dataclass(frozen=True)
class EpisodeMetrics:
    controller: str
    scenario: str
    episode: int
    steps: int
    total_reward: float
    mean_risk: float
    max_risk: float
    final_risk: float
    final_label: str
    severe_steps: int
    severe_fraction: float
    mean_feed_override: float
    mean_spindle_override: float
    shield_rejections: int
    relative_mrr_proxy: float
    randomized: bool = False
    spindle_scale: float = 1.0
    feed_scale: float = 1.0
    axial_depth_scale: float = 1.0
    radial_depth_scale: float = 1.0
    stiffness_scale: float = 1.0
    damping_scale: float = 1.0
    cutting_coeff_scale: float = 1.0
    noise_scale: float = 1.0


@dataclass(frozen=True)
class SummaryMetrics:
    controller: str
    scenario: str
    episodes: int
    mean_total_reward: float
    mean_risk: float
    max_risk: float
    mean_final_risk: float
    severe_fraction: float
    mean_feed_override: float
    mean_spindle_override: float
    shield_rejections: int
    relative_mrr_proxy: float


def run_benchmark(
    *,
    controllers: list[str],
    scenarios: list[str],
    episodes: int,
    steps: int,
    out_dir: Path,
    seed: int = 101,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> dict[str, list[dict]]:
    """Run deterministic controller benchmarks and write report artifacts."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    episode_metrics: list[EpisodeMetrics] = []
    for scenario in scenarios:
        for controller in controllers:
            for episode in range(episodes):
                episode_metrics.append(
                    run_episode(
                        controller_name=controller,
                        scenario_name=scenario,
                        episode=episode,
                        steps=steps,
                        seed=seed + 10_000 * episode + 100 * scenarios.index(scenario),
                        margin_calibration=margin_calibration,
                        randomization=randomization,
                        reward_config=reward_config,
                        shield_config=shield_config,
                    )
                )

    summary = summarize_metrics(episode_metrics)
    payload = {
        "episodes": [asdict(metric) for metric in episode_metrics],
        "summary": [asdict(metric) for metric in summary],
    }
    _write_csv(out_dir / "episodes.csv", payload["episodes"])
    _write_csv(out_dir / "summary.csv", payload["summary"])
    _write_json(out_dir / "summary.json", payload)
    _write_pareto_csv(out_dir / "pareto.csv", summary)
    _write_markdown(out_dir / "summary.md", summary)
    _write_svg(out_dir / "risk_vs_mrr.svg", summary)
    _write_pareto_svg(out_dir / "pareto.svg", summary)
    return payload


def run_closed_loop_benchmark(
    *,
    controllers: list[str],
    scenarios: list[str],
    episodes: int,
    steps: int,
    out_dir: Path,
    seed: int = 101,
    decision_interval_s: float = 0.12,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> dict[str, list[dict]]:
    """Run controller benchmarks while carrying dynamic state between decisions."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if decision_interval_s <= 0:
        raise ValueError("decision_interval_s must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    episode_metrics: list[EpisodeMetrics] = []
    for scenario in scenarios:
        for controller in controllers:
            for episode in range(episodes):
                episode_metrics.append(
                    run_closed_loop_episode(
                        controller_name=controller,
                        scenario_name=scenario,
                        episode=episode,
                        steps=steps,
                        seed=seed + 10_000 * episode + 100 * scenarios.index(scenario),
                        decision_interval_s=decision_interval_s,
                        margin_calibration=margin_calibration,
                        randomization=randomization,
                        reward_config=reward_config,
                        shield_config=shield_config,
                    )
                )

    summary = summarize_metrics(episode_metrics)
    payload = {
        "episodes": [asdict(metric) for metric in episode_metrics],
        "summary": [asdict(metric) for metric in summary],
    }
    _write_csv(out_dir / "episodes.csv", payload["episodes"])
    _write_csv(out_dir / "summary.csv", payload["summary"])
    _write_json(out_dir / "summary.json", payload)
    _write_pareto_csv(out_dir / "pareto.csv", summary)
    _write_markdown(out_dir / "summary.md", summary)
    _write_svg(out_dir / "risk_vs_mrr.svg", summary)
    _write_pareto_svg(out_dir / "pareto.svg", summary)
    return payload


def run_episode(
    *,
    controller_name: str,
    scenario_name: str,
    episode: int,
    steps: int,
    seed: int,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> EpisodeMetrics:
    modal, tool, cut, sim_config = make_scenario(scenario_name)
    modal, cut, sim_config, randomization_metadata = apply_domain_randomization(
        modal=modal,
        cut=cut,
        sim_config=sim_config,
        rng=np.random.default_rng(seed),
        config=randomization or DomainRandomizationConfig(),
    )
    sim_config = replace(sim_config, random_seed=seed)
    env = ChatterSuppressEnv(
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
        reward_config=reward_config,
        shield_config=shield_config,
        max_steps=steps,
    )
    controller = make_controller(
        controller_name,
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
    )
    _, info = env.reset(seed=seed)
    machine_state = MachineState()

    total_reward = 0.0
    risks: list[float] = []
    labels: list[str] = []
    feed_overrides: list[float] = []
    spindle_overrides: list[float] = []
    shield_rejections = 0

    for _ in range(steps):
        proposal = controller.propose(info["risk"], machine_state)
        obs, reward, _, truncated, info = env.step([proposal.feed_override, proposal.spindle_override])
        shield = info["shield"]
        risk = info["risk"]
        total_reward += reward
        risks.append(risk.risk_chatter_now)
        labels.append(risk.label)
        feed_overrides.append(shield.feed_override)
        spindle_overrides.append(shield.spindle_override)
        shield_rejections += int(shield.rejected)
        machine_state = MachineState(
            last_feed_override=shield.feed_override,
            last_spindle_override=shield.spindle_override,
        )
        if truncated:
            break

    steps_run = len(risks)
    severe_steps = sum(label == "severe" for label in labels)
    mean_feed = _mean(feed_overrides)
    mean_spindle = _mean(spindle_overrides)
    return EpisodeMetrics(
        controller=controller_name,
        scenario=scenario_name,
        episode=episode,
        steps=steps_run,
        total_reward=total_reward,
        mean_risk=_mean(risks),
        max_risk=max(risks) if risks else 0.0,
        final_risk=risks[-1] if risks else 0.0,
        final_label=labels[-1] if labels else "unknown",
        severe_steps=severe_steps,
        severe_fraction=severe_steps / max(steps_run, 1),
        mean_feed_override=mean_feed,
        mean_spindle_override=mean_spindle,
        shield_rejections=shield_rejections,
        relative_mrr_proxy=mean_feed * mean_spindle,
        **_episode_randomization_fields(randomization_metadata),
    )


def run_closed_loop_episode(
    *,
    controller_name: str,
    scenario_name: str,
    episode: int,
    steps: int,
    seed: int,
    decision_interval_s: float = 0.12,
    margin_calibration: MarginCalibration | None = None,
    randomization: DomainRandomizationConfig | None = None,
    reward_config: RewardConfig | None = None,
    shield_config: ShieldConfig | None = None,
) -> EpisodeMetrics:
    modal, tool, cut, sim_config = make_scenario(scenario_name)
    modal, cut, sim_config, randomization_metadata = apply_domain_randomization(
        modal=modal,
        cut=cut,
        sim_config=sim_config,
        rng=np.random.default_rng(seed),
        config=randomization or DomainRandomizationConfig(),
    )
    sim_config = replace(sim_config, duration_s=decision_interval_s, random_seed=seed)
    controller = make_controller(
        controller_name,
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
    )
    machine_state = MachineState()
    displacement = sim_config.initial_displacement_m
    velocity = sim_config.initial_velocity_m_s

    total_reward = 0.0
    risks: list[float] = []
    labels: list[str] = []
    feed_overrides: list[float] = []
    spindle_overrides: list[float] = []
    shield_rejections = 0

    for step_idx in range(steps):
        active_cut = replace(
            cut,
            feed_per_tooth_m=cut.feed_per_tooth_m * machine_state.last_feed_override,
            spindle_rpm=cut.spindle_rpm * machine_state.last_spindle_override,
        )
        active_config = replace(
            sim_config,
            random_seed=seed + step_idx,
            initial_displacement_m=(float(displacement[0]), float(displacement[1])),
            initial_velocity_m_s=(float(velocity[0]), float(velocity[1])),
        )
        result = simulate_milling(modal, tool, active_cut, active_config)
        risk = _risk_from_result(result, active_config, modal, tool, active_cut, margin_calibration)
        proposal = controller.propose(risk, machine_state)
        shielded = apply_safety_shield(proposal, risk, machine_state, shield_config)

        reward, _ = compute_control_reward(
            risk=risk,
            shielded=shielded,
            previous_state=machine_state,
            config=reward_config,
        )

        total_reward += reward
        risks.append(risk.risk_chatter_now)
        labels.append(risk.label)
        feed_overrides.append(machine_state.last_feed_override)
        spindle_overrides.append(machine_state.last_spindle_override)
        shield_rejections += int(shielded.rejected)
        machine_state = MachineState(
            last_feed_override=shielded.feed_override,
            last_spindle_override=shielded.spindle_override,
        )
        displacement = result.displacement_m[-1]
        velocity = result.velocity_m_s[-1]

    steps_run = len(risks)
    severe_steps = sum(label == "severe" for label in labels)
    mean_feed = _mean(feed_overrides)
    mean_spindle = _mean(spindle_overrides)
    return EpisodeMetrics(
        controller=controller_name,
        scenario=scenario_name,
        episode=episode,
        steps=steps_run,
        total_reward=total_reward,
        mean_risk=_mean(risks),
        max_risk=max(risks) if risks else 0.0,
        final_risk=risks[-1] if risks else 0.0,
        final_label=labels[-1] if labels else "unknown",
        severe_steps=severe_steps,
        severe_fraction=severe_steps / max(steps_run, 1),
        mean_feed_override=mean_feed,
        mean_spindle_override=mean_spindle,
        shield_rejections=shield_rejections,
        relative_mrr_proxy=mean_feed * mean_spindle,
        **_episode_randomization_fields(randomization_metadata),
    )


def _risk_from_result(result, sim_config: SimulationConfig, modal, tool, cut, margin_calibration: MarginCalibration | None = None):
    features = extract_signal_features(
        result.sensor_signal,
        sim_config.sample_rate_hz,
        cut.spindle_rpm,
        tool.flute_count,
        modal.natural_frequency_hz,
    )
    raw_margin = signed_stability_margin(modal, tool, cut)
    margin = apply_margin_calibration(raw_margin, margin_calibration, modal, tool, cut)
    risk = estimate_chatter_risk(features, margin)
    calibration_uncertainty = calibrated_margin_uncertainty(raw_margin, margin_calibration, modal, tool, cut)
    if calibration_uncertainty is not None and calibration_uncertainty > risk.uncertainty:
        risk = replace(
            risk,
            uncertainty=calibration_uncertainty,
            reason_codes=tuple(dict.fromkeys((*risk.reason_codes, "calibration_uncertainty"))),
        )
    return risk


def _episode_randomization_fields(metadata: dict[str, float | bool]) -> dict[str, float | bool]:
    return {
        "randomized": bool(metadata.get("randomized", False)),
        "spindle_scale": float(metadata.get("spindle_scale", 1.0)),
        "feed_scale": float(metadata.get("feed_scale", 1.0)),
        "axial_depth_scale": float(metadata.get("axial_depth_scale", 1.0)),
        "radial_depth_scale": float(metadata.get("radial_depth_scale", 1.0)),
        "stiffness_scale": float(metadata.get("stiffness_scale", 1.0)),
        "damping_scale": float(metadata.get("damping_scale", 1.0)),
        "cutting_coeff_scale": float(metadata.get("cutting_coeff_scale", 1.0)),
        "noise_scale": float(metadata.get("noise_scale", 1.0)),
    }


def summarize_metrics(metrics: list[EpisodeMetrics]) -> list[SummaryMetrics]:
    groups: dict[tuple[str, str], list[EpisodeMetrics]] = {}
    for metric in metrics:
        groups.setdefault((metric.controller, metric.scenario), []).append(metric)

    summary: list[SummaryMetrics] = []
    for (controller, scenario), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        summary.append(
            SummaryMetrics(
                controller=controller,
                scenario=scenario,
                episodes=len(rows),
                mean_total_reward=_mean([row.total_reward for row in rows]),
                mean_risk=_mean([row.mean_risk for row in rows]),
                max_risk=max(row.max_risk for row in rows),
                mean_final_risk=_mean([row.final_risk for row in rows]),
                severe_fraction=_mean([row.severe_fraction for row in rows]),
                mean_feed_override=_mean([row.mean_feed_override for row in rows]),
                mean_spindle_override=_mean([row.mean_spindle_override for row in rows]),
                shield_rejections=sum(row.shield_rejections for row in rows),
                relative_mrr_proxy=_mean([row.relative_mrr_proxy for row in rows]),
            )
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pareto_csv(path: Path, summary: list[SummaryMetrics]) -> None:
    rows: list[dict] = []
    for metric in summary:
        rows.append(
            {
                **asdict(metric),
                "pareto_efficient": _is_pareto_efficient(metric, summary),
                "risk_mrr_score": metric.relative_mrr_proxy / max(metric.mean_risk, 1.0e-9),
            }
        )
    _write_csv(path, rows)


def _write_markdown(path: Path, summary: list[SummaryMetrics]) -> None:
    lines = [
        "# Chatter Twin Benchmark Summary",
        "",
        "| Scenario | Controller | Episodes | Mean risk | Severe fraction | Relative MRR | Mean reward | Shield rejects |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row.scenario} | {row.controller} | {row.episodes} | "
            f"{row.mean_risk:.3f} | {row.severe_fraction:.3f} | "
            f"{row.relative_mrr_proxy:.3f} | {row.mean_total_reward:.3f} | "
            f"{row.shield_rejections} |"
        )
    lines.extend(
        [
            "",
            "Relative MRR is a proxy: mean feed override multiplied by mean spindle override.",
            "Lower risk is better; higher relative MRR is better when risk remains controlled.",
            "Pareto rows are written to `pareto.csv` and `pareto.svg`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg(path: Path, summary: list[SummaryMetrics]) -> None:
    width = 960
    height = max(220, 90 + 38 * len(summary))
    left = 190
    right = 40
    top = 40
    scale_width = width - left - right
    rows: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="26" font-family="sans-serif" font-size="18" font-weight="600">Mean chatter risk vs relative MRR proxy</text>',
    ]
    for index, row in enumerate(summary):
        y = top + 34 + index * 38
        risk_width = max(1.0, row.mean_risk * scale_width)
        mrr_width = max(1.0, min(row.relative_mrr_proxy, 1.25) / 1.25 * scale_width)
        label = f"{row.scenario} / {row.controller}"
        rows.append(f'<text x="20" y="{y + 12}" font-family="sans-serif" font-size="12">{_escape(label)}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{risk_width:.1f}" height="12" fill="#c94f4f"/>')
        rows.append(f'<rect x="{left}" y="{y + 16}" width="{mrr_width:.1f}" height="12" fill="#3f7fbf"/>')
        rows.append(
            f'<text x="{left + scale_width + 8}" y="{y + 12}" font-family="sans-serif" font-size="11">'
            f"risk {row.mean_risk:.2f}</text>"
        )
        rows.append(
            f'<text x="{left + scale_width + 8}" y="{y + 28}" font-family="sans-serif" font-size="11">'
            f"mrr {row.relative_mrr_proxy:.2f}</text>"
        )
    legend_y = height - 28
    rows.append(f'<rect x="20" y="{legend_y - 10}" width="14" height="10" fill="#c94f4f"/>')
    rows.append(f'<text x="40" y="{legend_y}" font-family="sans-serif" font-size="12">mean risk</text>')
    rows.append(f'<rect x="130" y="{legend_y - 10}" width="14" height="10" fill="#3f7fbf"/>')
    rows.append(f'<text x="150" y="{legend_y}" font-family="sans-serif" font-size="12">relative MRR proxy</text>')
    rows.append("</svg>")
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_pareto_svg(path: Path, summary: list[SummaryMetrics]) -> None:
    width = 860
    height = max(320, 120 + 48 * len({row.scenario for row in summary}))
    left = 80
    right = 40
    top = 48
    bottom = 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_risk = max((row.mean_risk for row in summary), default=1.0)
    max_mrr = max((row.relative_mrr_proxy for row in summary), default=1.0)
    min_mrr = min((row.relative_mrr_proxy for row in summary), default=0.0)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="600">Controller Pareto: risk vs MRR proxy</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w - 110}" y="{height - 18}" font-family="sans-serif" font-size="12">mean risk</text>',
        f'<text x="18" y="{top + 12}" font-family="sans-serif" font-size="12">MRR</text>',
    ]
    palette = ["#2b7a78", "#c94f4f", "#3f7fbf", "#9a6fb0", "#d18b2f", "#4f8f3a"]
    scenarios = sorted({row.scenario for row in summary})
    scenario_color = {scenario: palette[idx % len(palette)] for idx, scenario in enumerate(scenarios)}
    for metric in summary:
        x = left + (metric.mean_risk / max(max_risk, 1.0e-9)) * plot_w
        y = top + (1.0 - (metric.relative_mrr_proxy - min_mrr) / max(max_mrr - min_mrr, 1.0e-9)) * plot_h
        efficient = _is_pareto_efficient(metric, summary)
        radius = 6 if efficient else 4
        color = scenario_color[metric.scenario]
        opacity = "1.0" if efficient else "0.45"
        label = f"{metric.scenario}/{metric.controller}"
        rows.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"/>')
        rows.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-family="sans-serif" font-size="10">{_escape(label)}</text>')
    rows.append("</svg>")
    path.write_text("\n".join(rows), encoding="utf-8")


def _is_pareto_efficient(metric: SummaryMetrics, all_metrics: list[SummaryMetrics]) -> bool:
    peers = [row for row in all_metrics if row.scenario == metric.scenario]
    for other in peers:
        if other is metric:
            continue
        no_worse = other.mean_risk <= metric.mean_risk and other.relative_mrr_proxy >= metric.relative_mrr_proxy
        strictly_better = other.mean_risk < metric.mean_risk or other.relative_mrr_proxy > metric.relative_mrr_proxy
        if no_worse and strictly_better:
            return False
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
