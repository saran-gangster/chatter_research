from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chatter_twin.benchmark import run_benchmark, run_closed_loop_benchmark
from chatter_twin.calibration import CALIBRATION_MODELS, calibrate_margin_surrogate, load_margin_calibration
from chatter_twin.counterfactual import (
    CounterfactualConfig,
    CounterfactualPolicyConfig,
    StabilityPolicyConfig,
    run_counterfactual_risk_shadow_policy,
    run_shadow_action_sweep,
    run_shadow_episode_counterfactual,
    run_shadow_counterfactual,
    run_stability_margin_shadow_policy,
)
from chatter_twin.datasets import (
    BoschCNCIngestConfig,
    ICNC_FILENAME,
    ICNCIngestConfig,
    KITIndustrialIngestConfig,
    KITMatIngestConfig,
    KIT_INDUSTRIAL_FILENAME,
    MachineRunIngestConfig,
    MTCuttingIngestConfig,
    download_icnc_dataset,
    ingest_bosch_cnc_dataset,
    ingest_machine_run_dataset,
    ingest_mt_cutting_dataset,
    ingest_kit_industrial_dataset,
    ingest_kit_mat_dataset,
    inspect_mt_cutting_dataset,
    inspect_bosch_cnc_dataset,
    inspect_machine_run,
    inspect_kit_industrial_dataset,
    inspect_kit_synchronized_mat,
    ingest_icnc_dataset,
    write_machine_run_template,
    write_icnc_source_manifest,
    write_kit_industrial_source_manifest,
    write_mt_cutting_source_manifest,
)
from chatter_twin.controllers import make_controller
from chatter_twin.demo import InternalDemoConfig, write_internal_demo_report
from chatter_twin.dynamics import simulate_milling
from chatter_twin.env import ChatterSuppressEnv
from chatter_twin.features import extract_signal_features
from chatter_twin.models import MachineState
from chatter_twin.models import RewardConfig, ShieldConfig
from chatter_twin.offline import (
    CALIBRATION_METHODS,
    FEATURE_SETS,
    HOLDOUT_TAILS,
    MODEL_TYPES,
    SPLIT_MODES,
    TARGETS,
    RiskTrainingConfig,
    train_risk_model,
)
from chatter_twin.policy_selection import PolicySelectionConfig, select_rl_policy
from chatter_twin.pseudo_label import (
    DEFAULT_PSEUDO_SCORE_COLUMNS,
    POSITIVE_MODES,
    PseudoLabelConfig,
    pseudo_label_replay_dataset,
)
from chatter_twin.realdata import parse_real_data_run_spec, write_real_data_benchmark, write_risk_error_analysis
from chatter_twin.risk import estimate_chatter_risk
from chatter_twin.replay import LABEL_TO_ID, DomainRandomizationConfig, HorizonConfig, TransitionFocusConfig, WindowSpec, export_synthetic_dataset
from chatter_twin.rl import (
    MultiSeedTrainingConfig,
    QLearningConfig,
    Sb3TrainingConfig,
    evaluate_saved_sb3_run,
    train_multi_seed_policies,
    train_q_learning,
    train_sb3_policy,
)
from chatter_twin.rl_shadow import (
    GATE_PROFILE_CHOICES,
    RLShadowGateConfig,
    RLShadowReplayConfig,
    gate_rl_shadow_replay,
    run_selected_rl_shadow_policy,
)
from chatter_twin.rl_compare import compare_rl_runs, parse_run_ref
from chatter_twin.scenarios import make_scenario
from chatter_twin.shadow import ShadowPolicyConfig, run_shadow_evaluation
from chatter_twin.shield import apply_safety_shield
from chatter_twin.stability import signed_stability_margin

CONTROLLER_CHOICES = ["fixed", "rule", "sld", "mpc", "cf", "hybrid"]
SCENARIO_CHOICES = ["stable", "near_boundary", "unstable", "onset"]
DEFAULT_RANDOMIZATION = DomainRandomizationConfig()
DEFAULT_INTERNAL_DEMO = InternalDemoConfig()


def _add_randomization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--spindle-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.spindle_scale)
    parser.add_argument("--feed-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.feed_scale)
    parser.add_argument("--axial-depth-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.axial_depth_scale)
    parser.add_argument("--radial-depth-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.radial_depth_scale)
    parser.add_argument("--stiffness-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.stiffness_scale)
    parser.add_argument("--damping-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.damping_scale)
    parser.add_argument("--cutting-coeff-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.cutting_coeff_scale)
    parser.add_argument("--noise-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.noise_scale)


def _randomization_from_args(args: argparse.Namespace) -> DomainRandomizationConfig:
    return DomainRandomizationConfig(
        enabled=bool(args.randomize),
        spindle_scale=tuple(args.spindle_scale),
        feed_scale=tuple(args.feed_scale),
        axial_depth_scale=tuple(args.axial_depth_scale),
        radial_depth_scale=tuple(args.radial_depth_scale),
        stiffness_scale=tuple(args.stiffness_scale),
        damping_scale=tuple(args.damping_scale),
        cutting_coeff_scale=tuple(args.cutting_coeff_scale),
        noise_scale=tuple(args.noise_scale),
    )


def _add_reward_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--productivity-mode", choices=["feed", "mrr"], default="feed")
    parser.add_argument("--productivity-weight", type=float, default=1.0)
    parser.add_argument("--risk-now-weight", type=float, default=3.0)
    parser.add_argument("--risk-horizon-weight", type=float, default=1.5)
    parser.add_argument("--severe-penalty", type=float, default=2.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.5)
    parser.add_argument("--rejection-penalty", type=float, default=0.75)
    parser.add_argument("--clip-penalty", type=float, default=0.0)
    parser.add_argument("--rate-limit-penalty", type=float, default=0.0)


def _reward_from_args(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        productivity_mode=args.productivity_mode,
        productivity_weight=args.productivity_weight,
        risk_now_weight=args.risk_now_weight,
        risk_horizon_weight=args.risk_horizon_weight,
        severe_penalty=args.severe_penalty,
        smoothness_weight=args.smoothness_weight,
        rejection_penalty=args.rejection_penalty,
        clip_penalty=args.clip_penalty,
        rate_limit_penalty=args.rate_limit_penalty,
    )


def _add_shield_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uncertainty-mode", choices=["reject", "hold", "advisory"], default="reject")


def _shield_from_args(args: argparse.Namespace) -> ShieldConfig:
    return ShieldConfig(uncertainty_mode=args.uncertainty_mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatter-twin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--scenario", choices=SCENARIO_CHOICES, default="stable")
    simulate.add_argument("--duration", type=float, default=None)
    simulate.add_argument("--out", type=Path, required=True)

    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--controller", choices=CONTROLLER_CHOICES, default="fixed")
    rollout.add_argument("--steps", type=int, default=20)
    rollout.add_argument("--scenario", choices=SCENARIO_CHOICES, default="near_boundary")
    rollout.add_argument("--margin-calibration", type=Path, default=None)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--controllers", default="fixed,rule,sld,mpc")
    benchmark.add_argument("--scenarios", default="stable,near_boundary,unstable")
    benchmark.add_argument("--episodes", type=int, default=20)
    benchmark.add_argument("--steps", type=int, default=20)
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.add_argument("--seed", type=int, default=101)
    benchmark.add_argument("--margin-calibration", type=Path, default=None)
    _add_randomization_args(benchmark)
    _add_shield_args(benchmark)

    closed_loop = subparsers.add_parser("closed-loop-benchmark")
    closed_loop.add_argument("--controllers", default="fixed,rule,sld,mpc,cf,hybrid")
    closed_loop.add_argument("--scenarios", default="stable,near_boundary,unstable,onset")
    closed_loop.add_argument("--episodes", type=int, default=5)
    closed_loop.add_argument("--steps", type=int, default=20)
    closed_loop.add_argument("--decision-interval", type=float, default=0.12)
    closed_loop.add_argument("--out", type=Path, required=True)
    closed_loop.add_argument("--seed", type=int, default=101)
    closed_loop.add_argument("--margin-calibration", type=Path, default=None)
    _add_randomization_args(closed_loop)
    _add_shield_args(closed_loop)

    calibrate_margin = subparsers.add_parser("calibrate-margin")
    calibrate_margin.add_argument("--scenarios", default="stable,near_boundary,onset,unstable")
    calibrate_margin.add_argument("--axial-depth-scales", default="0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.50,3.00")
    calibrate_margin.add_argument("--spindle-scales", default="0.86,0.90,0.94,0.98,1.02,1.06,1.10,1.14")
    calibrate_margin.add_argument("--duration", type=float, default=0.25)
    calibrate_margin.add_argument("--sample-rate", type=float, default=None)
    calibrate_margin.add_argument("--sensor-noise", type=float, default=0.0)
    calibrate_margin.add_argument("--target-threshold", type=float, default=0.50)
    calibrate_margin.add_argument("--calibration-model", choices=CALIBRATION_MODELS, default="raw")
    calibrate_margin.add_argument("--family-count", type=int, default=1)
    calibrate_margin.add_argument("--holdout-family", type=int, default=None)
    calibrate_margin.add_argument("--stiffness-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.75, 1.25))
    calibrate_margin.add_argument("--damping-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.70, 1.30))
    calibrate_margin.add_argument("--cutting-coeff-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.75, 1.35))
    calibrate_margin.add_argument("--radial-depth-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.70, 1.25))
    calibrate_margin.add_argument("--feed-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.80, 1.20))
    calibrate_margin.add_argument("--runout-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.50, 2.00))
    calibrate_margin.add_argument("--out", type=Path, required=True)
    calibrate_margin.add_argument("--seed", type=int, default=909)

    export_synthetic = subparsers.add_parser("export-synthetic")
    export_synthetic.add_argument("--scenarios", default="stable,near_boundary,unstable")
    export_synthetic.add_argument("--episodes", type=int, default=3)
    export_synthetic.add_argument("--duration", type=float, default=1.0)
    export_synthetic.add_argument("--window", type=float, default=0.10)
    export_synthetic.add_argument("--stride", type=float, default=0.05)
    export_synthetic.add_argument("--out", type=Path, required=True)
    export_synthetic.add_argument("--seed", type=int, default=202)
    export_synthetic.add_argument("--randomize", action="store_true")
    export_synthetic.add_argument("--spindle-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.spindle_scale)
    export_synthetic.add_argument("--feed-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.feed_scale)
    export_synthetic.add_argument("--axial-depth-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.axial_depth_scale)
    export_synthetic.add_argument("--radial-depth-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.radial_depth_scale)
    export_synthetic.add_argument("--stiffness-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.stiffness_scale)
    export_synthetic.add_argument("--damping-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.damping_scale)
    export_synthetic.add_argument("--cutting-coeff-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.cutting_coeff_scale)
    export_synthetic.add_argument("--noise-scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=DEFAULT_RANDOMIZATION.noise_scale)
    export_synthetic.add_argument("--focus-transitions", action="store_true")
    export_synthetic.add_argument("--transition-candidates", type=int, default=8)
    export_synthetic.add_argument("--min-transition-windows", type=int, default=1)
    export_synthetic.add_argument("--horizon", type=float, default=0.20)

    train_risk = subparsers.add_parser("train-risk")
    train_risk.add_argument("--dataset", type=Path, required=True)
    train_risk.add_argument("--out", type=Path, required=True)
    train_risk.add_argument("--model", choices=MODEL_TYPES, default="softmax", dest="model_type")
    train_risk.add_argument("--calibration", choices=CALIBRATION_METHODS, default="none")
    train_risk.add_argument("--feature-set", choices=FEATURE_SETS, default="base")
    train_risk.add_argument("--target", choices=TARGETS, default="current")
    train_risk.add_argument("--epochs", type=int, default=800)
    train_risk.add_argument("--learning-rate", type=float, default=0.08)
    train_risk.add_argument("--l2", type=float, default=1.0e-3)
    train_risk.add_argument("--test-fraction", type=float, default=0.30)
    train_risk.add_argument("--validation-fraction", type=float, default=0.0)
    train_risk.add_argument("--seed", type=int, default=303)
    train_risk.add_argument("--split-mode", choices=SPLIT_MODES, default="row")
    train_risk.add_argument("--holdout-column", default="axial_depth_scale")
    train_risk.add_argument("--holdout-tail", choices=HOLDOUT_TAILS, default="high")

    pseudo_label = subparsers.add_parser("pseudo-label-replay")
    pseudo_label.add_argument("--dataset", type=Path, required=True)
    pseudo_label.add_argument("--out", type=Path, required=True)
    pseudo_label.add_argument("--score-columns", default=",".join(DEFAULT_PSEUDO_SCORE_COLUMNS))
    pseudo_label.add_argument("--positive-scenarios", default="")
    pseudo_label.add_argument("--positive-mode", choices=sorted(POSITIVE_MODES), default="scenario")
    pseudo_label.add_argument("--horizon", type=float, default=None)
    pseudo_label.add_argument("--transition-quantile", type=float, default=0.95)
    pseudo_label.add_argument("--slight-quantile", type=float, default=0.99)
    pseudo_label.add_argument("--severe-quantile", type=float, default=0.997)
    pseudo_label.add_argument("--transition-floor", type=float, default=1.0)
    pseudo_label.add_argument("--slight-floor", type=float, default=2.0)
    pseudo_label.add_argument("--severe-floor", type=float, default=3.5)

    train_rl = subparsers.add_parser("train-rl")
    train_rl.add_argument("--algorithm", choices=["q_learning", "sac", "td3"], default="sac")
    train_rl.add_argument("--scenarios", default="stable,near_boundary,onset,unstable")
    train_rl.add_argument("--episodes", type=int, default=200)
    train_rl.add_argument("--total-timesteps", type=int, default=5_000)
    train_rl.add_argument("--eval-episodes", type=int, default=8)
    train_rl.add_argument("--steps", type=int, default=20)
    train_rl.add_argument("--decision-interval", type=float, default=0.10)
    train_rl.add_argument("--learning-rate", type=float, default=3.0e-4)
    train_rl.add_argument("--buffer-size", type=int, default=50_000)
    train_rl.add_argument("--learning-starts", type=int, default=100)
    train_rl.add_argument("--batch-size", type=int, default=64)
    train_rl.add_argument("--gamma", type=float, default=0.92)
    train_rl.add_argument("--train-freq", type=int, default=1)
    train_rl.add_argument("--gradient-steps", type=int, default=1)
    train_rl.add_argument("--baseline-controllers", default="sld,mpc")
    train_rl.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    train_rl.add_argument("--delta-action-scale", type=float, default=1.0)
    train_rl.add_argument("--delta-feed-scale", type=float, default=None)
    train_rl.add_argument("--delta-spindle-scale", type=float, default=None)
    train_rl.add_argument("--delta-mapping", choices=["fixed", "headroom"], default="fixed")
    train_rl.add_argument("--no-candidate-guard", action="store_true")
    train_rl.add_argument("--margin-calibration", type=Path, default=None)
    train_rl.add_argument("--out", type=Path, required=True)
    train_rl.add_argument("--seed", type=int, default=616)
    _add_randomization_args(train_rl)
    _add_reward_args(train_rl)
    _add_shield_args(train_rl)

    train_rl_multiseed = subparsers.add_parser("train-rl-multiseed")
    train_rl_multiseed.add_argument("--algorithms", default="sac,td3")
    train_rl_multiseed.add_argument("--seeds", default="616,717,818")
    train_rl_multiseed.add_argument("--scenarios", default="stable,near_boundary,onset,unstable")
    train_rl_multiseed.add_argument("--episodes", type=int, default=200)
    train_rl_multiseed.add_argument("--total-timesteps", type=int, default=5_000)
    train_rl_multiseed.add_argument("--eval-episodes", type=int, default=8)
    train_rl_multiseed.add_argument("--steps", type=int, default=20)
    train_rl_multiseed.add_argument("--decision-interval", type=float, default=0.10)
    train_rl_multiseed.add_argument("--learning-rate", type=float, default=3.0e-4)
    train_rl_multiseed.add_argument("--buffer-size", type=int, default=50_000)
    train_rl_multiseed.add_argument("--learning-starts", type=int, default=100)
    train_rl_multiseed.add_argument("--batch-size", type=int, default=64)
    train_rl_multiseed.add_argument("--gamma", type=float, default=0.92)
    train_rl_multiseed.add_argument("--train-freq", type=int, default=1)
    train_rl_multiseed.add_argument("--gradient-steps", type=int, default=1)
    train_rl_multiseed.add_argument("--baseline-controllers", default="sld,mpc")
    train_rl_multiseed.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    train_rl_multiseed.add_argument("--delta-action-scale", type=float, default=1.0)
    train_rl_multiseed.add_argument("--delta-feed-scale", type=float, default=None)
    train_rl_multiseed.add_argument("--delta-spindle-scale", type=float, default=None)
    train_rl_multiseed.add_argument("--delta-mapping", choices=["fixed", "headroom"], default="fixed")
    train_rl_multiseed.add_argument("--no-candidate-guard", action="store_true")
    train_rl_multiseed.add_argument("--margin-calibration", type=Path, default=None)
    train_rl_multiseed.add_argument("--out", type=Path, required=True)
    _add_randomization_args(train_rl_multiseed)
    _add_reward_args(train_rl_multiseed)
    _add_shield_args(train_rl_multiseed)

    compare_rl = subparsers.add_parser("compare-rl-runs")
    compare_rl.add_argument("--run", action="append", required=True, help="Run reference as LABEL=PATH")
    compare_rl.add_argument("--baseline-label", default=None)
    compare_rl.add_argument("--out", type=Path, required=True)

    select_rl = subparsers.add_parser("select-rl-policy")
    select_rl.add_argument("--eval-dir", type=Path, required=True)
    select_rl.add_argument("--out", type=Path, required=True)
    select_rl.add_argument("--profile-label", default="safety_first")
    select_rl.add_argument("--min-relative-mrr", type=float, default=1.0)
    select_rl.add_argument("--risk-weight", type=float, default=1.0)
    select_rl.add_argument("--worst-risk-weight", type=float, default=0.5)
    select_rl.add_argument("--unstable-risk-weight", type=float, default=0.5)
    select_rl.add_argument("--mrr-weight", type=float, default=0.05)
    select_rl.add_argument("--mrr-shortfall-weight", type=float, default=3.0)
    select_rl.add_argument("--guard-fallback-weight", type=float, default=0.15)
    select_rl.add_argument("--shield-rejection-weight", type=float, default=10.0)

    rl_shadow = subparsers.add_parser("shadow-rl-policy")
    rl_shadow.add_argument("--selection", type=Path, required=True)
    rl_shadow.add_argument("--scenarios", default="stable,near_boundary,onset,unstable")
    rl_shadow.add_argument("--episodes", type=int, default=4)
    rl_shadow.add_argument("--steps", type=int, default=10)
    rl_shadow.add_argument("--decision-interval", type=float, default=0.08)
    rl_shadow.add_argument("--seed", type=int, default=3_000_000)
    rl_shadow.add_argument("--margin-calibration", type=Path, default=None)
    rl_shadow.add_argument("--out", type=Path, required=True)
    _add_randomization_args(rl_shadow)

    gate_rl_shadow = subparsers.add_parser("gate-rl-shadow")
    gate_rl_shadow.add_argument("--shadow-dir", type=Path, required=True)
    gate_rl_shadow.add_argument("--out", type=Path, required=True)
    gate_rl_shadow.add_argument("--profile", choices=GATE_PROFILE_CHOICES, default="shadow_review")
    gate_rl_shadow.add_argument("--min-recommendation-windows", type=int, default=None)
    gate_rl_shadow.add_argument("--max-mean-risk", type=float, default=None)
    gate_rl_shadow.add_argument("--max-max-risk", type=float, default=None)
    gate_rl_shadow.add_argument("--max-unstable-mean-risk", type=float, default=None)
    gate_rl_shadow.add_argument("--min-relative-mrr", type=float, default=None)
    gate_rl_shadow.add_argument("--max-guard-fallback-fraction", type=float, default=None)
    gate_rl_shadow.add_argument("--max-action-fraction", type=float, default=None)
    gate_rl_shadow.add_argument("--max-shield-rejections", type=int, default=None)
    gate_rl_shadow.add_argument("--allow-cnc-writes", action="store_true")

    internal_demo = subparsers.add_parser("internal-demo-report")
    internal_demo.add_argument("--out", type=Path, required=True)
    internal_demo.add_argument("--summary-out", type=Path, default=None)
    internal_demo.add_argument("--calibration-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.calibration_dir)
    internal_demo.add_argument("--risk-model-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.risk_model_dir)
    internal_demo.add_argument("--closed-loop-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.closed_loop_dir)
    internal_demo.add_argument("--rl-comparison-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.rl_comparison_dir)
    internal_demo.add_argument("--champion-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.champion_dir)
    internal_demo.add_argument("--rl-shadow-replay-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.rl_shadow_replay_dir)
    internal_demo.add_argument("--shadow-review-gate-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.shadow_review_gate_dir)
    internal_demo.add_argument("--live-shadow-gate-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.live_shadow_gate_dir)
    internal_demo.add_argument("--hardware-gate-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.hardware_gate_dir)
    internal_demo.add_argument("--shadow-eval-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.shadow_eval_dir)
    internal_demo.add_argument("--counterfactual-dir", type=Path, default=DEFAULT_INTERNAL_DEMO.counterfactual_dir)
    internal_demo.add_argument("--test-status", default="")

    real_data_benchmark = subparsers.add_parser("real-data-benchmark")
    real_data_benchmark.add_argument("--out", type=Path, required=True)
    real_data_benchmark.add_argument(
        "--run",
        action="append",
        default=None,
        help="Optional run spec: dataset=...,modality=...,path=...,claim=...,note=...",
    )

    error_analysis = subparsers.add_parser("risk-error-analysis")
    error_analysis.add_argument("--model-dir", type=Path, required=True)
    error_analysis.add_argument("--out", type=Path, required=True)
    error_analysis.add_argument("--group-column", default="scenario")

    icnc_manifest = subparsers.add_parser("icnc-manifest")
    icnc_manifest.add_argument("--out", type=Path, required=True)

    kit_manifest = subparsers.add_parser("kit-industrial-manifest")
    kit_manifest.add_argument("--out", type=Path, default=Path("data/raw/kit_industrial/source_manifest.json"))

    mt_manifest = subparsers.add_parser("mt-cutting-manifest")
    mt_manifest.add_argument("--out", type=Path, default=Path("data/raw/mt_cutting_dataset/source_manifest.json"))

    inspect_kit = subparsers.add_parser("inspect-kit-industrial")
    inspect_kit.add_argument("--source", type=Path, required=True)
    inspect_kit.add_argument("--out", type=Path, default=None)

    inspect_mt = subparsers.add_parser("inspect-mt-cutting")
    inspect_mt.add_argument("--source", type=Path, required=True)
    inspect_mt.add_argument("--out", type=Path, default=None)

    inspect_kit_mat = subparsers.add_parser("inspect-kit-mat")
    inspect_kit_mat.add_argument("--source", type=Path, required=True)
    inspect_kit_mat.add_argument("--trial", default="IM-02F-A01")
    inspect_kit_mat.add_argument("--max-datasets", type=int, default=200)
    inspect_kit_mat.add_argument("--out", type=Path, default=None)

    download_icnc = subparsers.add_parser("download-icnc")
    download_icnc.add_argument("--out", type=Path, default=Path("data/raw/icnc") / ICNC_FILENAME)
    download_icnc.add_argument("--manifest-out", type=Path, default=Path("data/raw/icnc/source_manifest.json"))
    download_icnc.add_argument("--force", action="store_true")

    ingest_icnc = subparsers.add_parser("ingest-icnc")
    ingest_icnc.add_argument("--source", type=Path, required=True)
    ingest_icnc.add_argument("--out", type=Path, required=True)
    ingest_icnc.add_argument("--window", type=float, default=0.10)
    ingest_icnc.add_argument("--stride", type=float, default=0.05)
    ingest_icnc.add_argument("--horizon", type=float, default=0.25)
    ingest_icnc.add_argument("--flutes", type=int, default=4)
    ingest_icnc.add_argument("--modal-frequency", type=float, default=None)
    ingest_icnc.add_argument("--default-sample-rate", type=float, default=10_000.0)
    ingest_icnc.add_argument("--default-spindle-rpm", type=float, default=9_000.0)
    ingest_icnc.add_argument("--default-feed-per-tooth", type=float, default=45.0e-6)
    ingest_icnc.add_argument("--include-unknown", action="store_true")
    ingest_icnc.add_argument("--max-packages-per-file", type=int, default=None)
    ingest_icnc.add_argument("--max-windows", type=int, default=None)

    ingest_kit = subparsers.add_parser("ingest-kit-industrial")
    ingest_kit.add_argument("--source", type=Path, required=True)
    ingest_kit.add_argument("--out", type=Path, required=True)
    ingest_kit.add_argument("--trials", default="IM-01F,IM-02F-A01")
    ingest_kit.add_argument("--window", type=float, default=1.0)
    ingest_kit.add_argument("--stride", type=float, default=0.5)
    ingest_kit.add_argument("--horizon", type=float, default=2.0)
    ingest_kit.add_argument("--sample-rate", type=float, default=500.0)
    ingest_kit.add_argument("--signal-columns", default="LOAD|6,CURRENT|6")
    ingest_kit.add_argument("--include-other-anomalies", action="store_true")
    ingest_kit.add_argument("--max-windows", type=int, default=None)

    ingest_kit_mat = subparsers.add_parser("ingest-kit-mat")
    ingest_kit_mat.add_argument("--source", type=Path, required=True)
    ingest_kit_mat.add_argument("--out", type=Path, required=True)
    ingest_kit_mat.add_argument("--trials", default="IM-01F,IM-02F-A01")
    ingest_kit_mat.add_argument("--window", type=float, default=0.10)
    ingest_kit_mat.add_argument("--stride", type=float, default=0.05)
    ingest_kit_mat.add_argument("--horizon", type=float, default=0.25)
    ingest_kit_mat.add_argument("--signal-names", default="xAcceleration,yAcceleration")
    ingest_kit_mat.add_argument("--standardize-signals", action="store_true")
    ingest_kit_mat.add_argument("--include-other-anomalies", action="store_true")
    ingest_kit_mat.add_argument("--max-windows", type=int, default=None)
    ingest_kit_mat.add_argument("--max-samples-per-trial", type=int, default=None)

    ingest_mt = subparsers.add_parser("ingest-mt-cutting")
    ingest_mt.add_argument("--source", type=Path, required=True)
    ingest_mt.add_argument("--out", type=Path, required=True)
    ingest_mt.add_argument("--experiments", default="")
    ingest_mt.add_argument("--window", type=float, default=0.10)
    ingest_mt.add_argument("--stride", type=float, default=0.05)
    ingest_mt.add_argument("--horizon", type=float, default=0.25)
    ingest_mt.add_argument("--sensors", default="0,1,2")
    ingest_mt.add_argument("--include-unknown", action="store_true")
    ingest_mt.add_argument("--max-experiments", type=int, default=None)
    ingest_mt.add_argument("--max-windows", type=int, default=None)

    inspect_bosch = subparsers.add_parser("inspect-bosch-cnc")
    inspect_bosch.add_argument("--source", type=Path, required=True)
    inspect_bosch.add_argument("--out", type=Path, default=None)

    machine_template = subparsers.add_parser("machine-run-template")
    machine_template.add_argument("--out", type=Path, required=True)
    machine_template.add_argument("--overwrite", action="store_true")

    validate_machine = subparsers.add_parser("validate-machine-run")
    validate_machine.add_argument("--source", type=Path, required=True)
    validate_machine.add_argument("--out", type=Path, default=None)
    validate_machine.add_argument("--sensor-columns", default="accel_x,accel_y,accel_z")
    validate_machine.add_argument("--default-label", choices=list(LABEL_TO_ID), default="unknown")

    ingest_machine = subparsers.add_parser("ingest-machine-run")
    ingest_machine.add_argument("--source", type=Path, required=True)
    ingest_machine.add_argument("--out", type=Path, required=True)
    ingest_machine.add_argument("--sensor-columns", default="accel_x,accel_y,accel_z")
    ingest_machine.add_argument("--window", type=float, default=0.10)
    ingest_machine.add_argument("--stride", type=float, default=0.05)
    ingest_machine.add_argument("--horizon", type=float, default=0.25)
    ingest_machine.add_argument("--default-label", choices=list(LABEL_TO_ID), default="unknown")
    ingest_machine.add_argument("--max-windows", type=int, default=None)

    ingest_bosch = subparsers.add_parser("ingest-bosch-cnc")
    ingest_bosch.add_argument("--source", type=Path, required=True)
    ingest_bosch.add_argument("--out", type=Path, required=True)
    ingest_bosch.add_argument("--machines", default="")
    ingest_bosch.add_argument("--operations", default="")
    ingest_bosch.add_argument("--window", type=float, default=0.25)
    ingest_bosch.add_argument("--stride", type=float, default=0.125)
    ingest_bosch.add_argument("--horizon", type=float, default=0.5)
    ingest_bosch.add_argument("--sample-rate", type=float, default=2_000.0)
    ingest_bosch.add_argument("--spindle-rpm", type=float, default=9_000.0)
    ingest_bosch.add_argument("--flute-count", type=int, default=4)
    ingest_bosch.add_argument("--max-files", type=int, default=None)
    ingest_bosch.add_argument("--max-files-per-quality", type=int, default=None)
    ingest_bosch.add_argument("--max-windows", type=int, default=None)

    eval_rl = subparsers.add_parser("eval-rl-run")
    eval_rl.add_argument("--run-dir", type=Path, required=True)
    eval_rl.add_argument("--scenarios", default="stable,near_boundary,onset,unstable")
    eval_rl.add_argument("--eval-episodes", type=int, default=6)
    eval_rl.add_argument("--steps", type=int, default=10)
    eval_rl.add_argument("--decision-interval", type=float, default=0.08)
    eval_rl.add_argument("--seed", type=int, default=2_000_000)
    eval_rl.add_argument("--margin-calibration", type=Path, default=None)
    eval_rl.add_argument("--out", type=Path, required=True)
    _add_randomization_args(eval_rl)

    shadow_eval = subparsers.add_parser("shadow-eval")
    shadow_eval.add_argument("--model-dir", type=Path, required=True)
    shadow_eval.add_argument("--out", type=Path, required=True)
    shadow_eval.add_argument("--threshold-source", choices=["event", "lead", "default", "manual"], default="event")
    shadow_eval.add_argument("--warning-threshold", type=float, default=None)
    shadow_eval.add_argument("--clear-threshold", type=float, default=None)
    shadow_eval.add_argument("--warning-feed", type=float, default=0.92)
    shadow_eval.add_argument("--warning-spindle", type=float, default=1.04)
    shadow_eval.add_argument("--max-feed-delta", type=float, default=0.05)
    shadow_eval.add_argument("--max-spindle-delta", type=float, default=0.05)

    shadow_counterfactual = subparsers.add_parser("shadow-counterfactual")
    shadow_counterfactual.add_argument("--shadow-dir", type=Path, required=True)
    shadow_counterfactual.add_argument("--dataset-dir", type=Path, default=None)
    shadow_counterfactual.add_argument("--out", type=Path, required=True)
    shadow_counterfactual.add_argument("--sample-rate", type=float, default=None)
    shadow_counterfactual.add_argument("--sensor-noise", type=float, default=None)
    shadow_counterfactual.add_argument("--seed", type=int, default=707)
    shadow_counterfactual.add_argument("--max-windows", type=int, default=None)

    shadow_episode_counterfactual = subparsers.add_parser("shadow-episode-counterfactual")
    shadow_episode_counterfactual.add_argument("--shadow-dir", type=Path, required=True)
    shadow_episode_counterfactual.add_argument("--dataset-dir", type=Path, default=None)
    shadow_episode_counterfactual.add_argument("--out", type=Path, required=True)
    shadow_episode_counterfactual.add_argument("--sample-rate", type=float, default=None)
    shadow_episode_counterfactual.add_argument("--sensor-noise", type=float, default=None)
    shadow_episode_counterfactual.add_argument("--seed", type=int, default=707)
    shadow_episode_counterfactual.add_argument("--max-windows", type=int, default=None)

    shadow_sweep = subparsers.add_parser("shadow-action-sweep")
    shadow_sweep.add_argument("--shadow-dir", type=Path, required=True)
    shadow_sweep.add_argument("--dataset-dir", type=Path, default=None)
    shadow_sweep.add_argument("--out", type=Path, required=True)
    shadow_sweep.add_argument("--feed-values", default="0.88,0.92,0.96,1.0")
    shadow_sweep.add_argument("--spindle-values", default="0.96,1.0,1.04,1.08")
    shadow_sweep.add_argument("--sample-rate", type=float, default=None)
    shadow_sweep.add_argument("--sensor-noise", type=float, default=None)
    shadow_sweep.add_argument("--seed", type=int, default=707)
    shadow_sweep.add_argument("--max-windows", type=int, default=None)

    stability_policy = subparsers.add_parser("shadow-stability-policy")
    stability_policy.add_argument("--shadow-dir", type=Path, required=True)
    stability_policy.add_argument("--dataset-dir", type=Path, default=None)
    stability_policy.add_argument("--out", type=Path, required=True)
    stability_policy.add_argument("--min-spindle", type=float, default=0.90)
    stability_policy.add_argument("--max-spindle", type=float, default=1.10)
    stability_policy.add_argument("--candidates", type=int, default=21)
    stability_policy.add_argument("--feed", type=float, default=1.0)
    stability_policy.add_argument("--min-margin-improvement", type=float, default=0.05)
    stability_policy.add_argument("--margin-weight", type=float, default=1.0)
    stability_policy.add_argument("--productivity-weight", type=float, default=0.03)
    stability_policy.add_argument("--move-penalty", type=float, default=0.08)
    stability_policy.add_argument("--max-windows", type=int, default=None)

    counterfactual_policy = subparsers.add_parser("shadow-counterfactual-policy")
    counterfactual_policy.add_argument("--shadow-dir", type=Path, required=True)
    counterfactual_policy.add_argument("--dataset-dir", type=Path, default=None)
    counterfactual_policy.add_argument("--out", type=Path, required=True)
    counterfactual_policy.add_argument("--feed-values", default="1.0")
    counterfactual_policy.add_argument("--spindle-values", default="0.96,1.0,1.04,1.08")
    counterfactual_policy.add_argument("--min-risk-reduction", type=float, default=0.005)
    counterfactual_policy.add_argument("--risk-weight", type=float, default=1.0)
    counterfactual_policy.add_argument("--productivity-weight", type=float, default=0.02)
    counterfactual_policy.add_argument("--move-penalty", type=float, default=0.04)
    counterfactual_policy.add_argument("--sample-rate", type=float, default=None)
    counterfactual_policy.add_argument("--sensor-noise", type=float, default=None)
    counterfactual_policy.add_argument("--seed", type=int, default=707)
    counterfactual_policy.add_argument("--max-windows", type=int, default=None)

    subparsers.add_parser("env-smoke")

    args = parser.parse_args(argv)
    match args.command:
        case "simulate":
            return _cmd_simulate(args)
        case "rollout":
            return _cmd_rollout(args)
        case "benchmark":
            return _cmd_benchmark(args)
        case "closed-loop-benchmark":
            return _cmd_closed_loop_benchmark(args)
        case "calibrate-margin":
            return _cmd_calibrate_margin(args)
        case "export-synthetic":
            return _cmd_export_synthetic(args)
        case "train-risk":
            return _cmd_train_risk(args)
        case "pseudo-label-replay":
            return _cmd_pseudo_label_replay(args)
        case "train-rl":
            return _cmd_train_rl(args)
        case "train-rl-multiseed":
            return _cmd_train_rl_multiseed(args)
        case "compare-rl-runs":
            return _cmd_compare_rl_runs(args)
        case "select-rl-policy":
            return _cmd_select_rl_policy(args)
        case "shadow-rl-policy":
            return _cmd_shadow_rl_policy(args)
        case "gate-rl-shadow":
            return _cmd_gate_rl_shadow(args)
        case "internal-demo-report":
            return _cmd_internal_demo_report(args)
        case "real-data-benchmark":
            return _cmd_real_data_benchmark(args)
        case "risk-error-analysis":
            return _cmd_risk_error_analysis(args)
        case "icnc-manifest":
            return _cmd_icnc_manifest(args)
        case "kit-industrial-manifest":
            return _cmd_kit_industrial_manifest(args)
        case "mt-cutting-manifest":
            return _cmd_mt_cutting_manifest(args)
        case "inspect-kit-industrial":
            return _cmd_inspect_kit_industrial(args)
        case "inspect-mt-cutting":
            return _cmd_inspect_mt_cutting(args)
        case "inspect-bosch-cnc":
            return _cmd_inspect_bosch_cnc(args)
        case "machine-run-template":
            return _cmd_machine_run_template(args)
        case "validate-machine-run":
            return _cmd_validate_machine_run(args)
        case "ingest-machine-run":
            return _cmd_ingest_machine_run(args)
        case "inspect-kit-mat":
            return _cmd_inspect_kit_mat(args)
        case "download-icnc":
            return _cmd_download_icnc(args)
        case "ingest-icnc":
            return _cmd_ingest_icnc(args)
        case "ingest-kit-industrial":
            return _cmd_ingest_kit_industrial(args)
        case "ingest-kit-mat":
            return _cmd_ingest_kit_mat(args)
        case "ingest-mt-cutting":
            return _cmd_ingest_mt_cutting(args)
        case "ingest-bosch-cnc":
            return _cmd_ingest_bosch_cnc(args)
        case "eval-rl-run":
            return _cmd_eval_rl_run(args)
        case "shadow-eval":
            return _cmd_shadow_eval(args)
        case "shadow-counterfactual":
            return _cmd_shadow_counterfactual(args)
        case "shadow-episode-counterfactual":
            return _cmd_shadow_episode_counterfactual(args)
        case "shadow-action-sweep":
            return _cmd_shadow_action_sweep(args)
        case "shadow-stability-policy":
            return _cmd_shadow_stability_policy(args)
        case "shadow-counterfactual-policy":
            return _cmd_shadow_counterfactual_policy(args)
        case "env-smoke":
            return _cmd_env_smoke()
    raise AssertionError("unreachable")


def _cmd_simulate(args: argparse.Namespace) -> int:
    modal, tool, cut, config = make_scenario(args.scenario)
    if args.duration is not None:
        from dataclasses import replace

        config = replace(config, duration_s=args.duration)
    result = simulate_milling(modal, tool, cut, config)
    features = extract_signal_features(
        result.sensor_signal,
        config.sample_rate_hz,
        cut.spindle_rpm,
        tool.flute_count,
        modal.natural_frequency_hz,
    )
    margin = signed_stability_margin(modal, tool, cut)
    risk = estimate_chatter_risk(features, margin)
    np.savez(
        args.out,
        time_s=result.time_s,
        displacement_m=result.displacement_m,
        velocity_m_s=result.velocity_m_s,
        acceleration_m_s2=result.acceleration_m_s2,
        cutting_force_n=result.cutting_force_n,
        sensor_signal=result.sensor_signal,
        risk_chatter_now=risk.risk_chatter_now,
        margin_physics=risk.margin_physics,
        label=risk.label,
    )
    print(json.dumps(_risk_summary(risk), indent=2, sort_keys=True))
    return 0


def _cmd_rollout(args: argparse.Namespace) -> int:
    modal, tool, cut, sim_config = make_scenario(args.scenario)
    margin_calibration = _load_optional_margin_calibration(args.margin_calibration)
    env = ChatterSuppressEnv(
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
        max_steps=args.steps,
    )
    controller = make_controller(
        args.controller,
        modal=modal,
        tool=tool,
        cut=cut,
        sim_config=sim_config,
        margin_calibration=margin_calibration,
    )
    obs, info = env.reset(seed=0)
    state = MachineState()
    total_reward = 0.0
    labels: list[str] = []
    for _ in range(args.steps):
        proposal = controller.propose(info["risk"], state)
        shielded = apply_safety_shield(proposal, info["risk"], state)
        obs, reward, _, truncated, info = env.step([shielded.feed_override, shielded.spindle_override])
        total_reward += reward
        labels.append(info["risk"].label)
        state = MachineState(
            last_feed_override=shielded.feed_override,
            last_spindle_override=shielded.spindle_override,
        )
        if truncated:
            break
    print(
        json.dumps(
            {
                "controller": args.controller,
                "steps": len(labels),
                "total_reward": total_reward,
                "final_label": labels[-1] if labels else "unknown",
                "final_observation": obs.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_env_smoke() -> int:
    env = ChatterSuppressEnv(max_steps=5)
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(5):
        obs, reward, _, truncated, info = env.step([1.0, 1.0])
        total_reward += reward
        if truncated:
            break
    print(
        json.dumps(
            {
                "ok": True,
                "total_reward": total_reward,
                "final_label": info["risk"].label,
                "final_observation": obs.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    controllers = _parse_csv_arg(args.controllers)
    scenarios = _parse_csv_arg(args.scenarios)
    margin_calibration = _load_optional_margin_calibration(args.margin_calibration)
    shield_config = _shield_from_args(args)
    payload = run_benchmark(
        controllers=controllers,
        scenarios=scenarios,
        episodes=args.episodes,
        steps=args.steps,
        out_dir=args.out,
        seed=args.seed,
        margin_calibration=margin_calibration,
        randomization=_randomization_from_args(args),
        shield_config=shield_config,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "episodes": len(payload["episodes"]),
                "summary_rows": len(payload["summary"]),
                "artifacts": [
                    "episodes.csv",
                    "summary.csv",
                    "summary.json",
                    "summary.md",
                    "risk_vs_mrr.svg",
                    "pareto.csv",
                    "pareto.svg",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_closed_loop_benchmark(args: argparse.Namespace) -> int:
    controllers = _parse_csv_arg(args.controllers)
    scenarios = _parse_csv_arg(args.scenarios)
    margin_calibration = _load_optional_margin_calibration(args.margin_calibration)
    shield_config = _shield_from_args(args)
    payload = run_closed_loop_benchmark(
        controllers=controllers,
        scenarios=scenarios,
        episodes=args.episodes,
        steps=args.steps,
        decision_interval_s=args.decision_interval,
        out_dir=args.out,
        seed=args.seed,
        margin_calibration=margin_calibration,
        randomization=_randomization_from_args(args),
        shield_config=shield_config,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "episodes": len(payload["episodes"]),
                "summary_rows": len(payload["summary"]),
                "artifacts": [
                    "episodes.csv",
                    "summary.csv",
                    "summary.json",
                    "summary.md",
                    "risk_vs_mrr.svg",
                    "pareto.csv",
                    "pareto.svg",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_calibrate_margin(args: argparse.Namespace) -> int:
    payload = calibrate_margin_surrogate(
        scenarios=_parse_csv_arg(args.scenarios),
        axial_depth_scales=tuple(_parse_float_csv_arg(args.axial_depth_scales)),
        spindle_scales=tuple(_parse_float_csv_arg(args.spindle_scales)),
        duration_s=args.duration,
        sample_rate_hz=args.sample_rate,
        sensor_noise_std=args.sensor_noise,
        target_threshold=args.target_threshold,
        calibration_model=args.calibration_model,
        family_count=args.family_count,
        holdout_family=args.holdout_family,
        stiffness_scale=tuple(args.stiffness_scale),
        damping_scale=tuple(args.damping_scale),
        cutting_coeff_scale=tuple(args.cutting_coeff_scale),
        radial_depth_scale=tuple(args.radial_depth_scale),
        feed_scale=tuple(args.feed_scale),
        runout_scale=tuple(args.runout_scale),
        out_dir=args.out,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "samples": payload["metrics"]["sample_count"],
                "model": payload["calibration"]["model_type"],
                "positive_fraction": payload["metrics"]["positive_fraction"],
                "roc_auc": payload["metrics"]["roc_auc"],
                "brier_score": payload["metrics"]["brier_score"],
                "train_roc_auc": payload["metrics"]["train"]["roc_auc"],
                "holdout_roc_auc": payload["metrics"]["holdout"]["roc_auc"] if payload["metrics"]["holdout"] else None,
                "holdout_brier_score": payload["metrics"]["holdout"]["brier_score"] if payload["metrics"]["holdout"] else None,
                "boundary_raw_margin_at_p_0_5": payload["metrics"]["boundary_raw_margin_at_p_0_5"],
                "risk_at_raw_margin_zero": payload["metrics"]["risk_at_raw_margin_zero"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_export_synthetic(args: argparse.Namespace) -> int:
    scenarios = _parse_csv_arg(args.scenarios)
    randomization = DomainRandomizationConfig(
        enabled=args.randomize,
        spindle_scale=tuple(args.spindle_scale),
        feed_scale=tuple(args.feed_scale),
        axial_depth_scale=tuple(args.axial_depth_scale),
        radial_depth_scale=tuple(args.radial_depth_scale),
        stiffness_scale=tuple(args.stiffness_scale),
        damping_scale=tuple(args.damping_scale),
        cutting_coeff_scale=tuple(args.cutting_coeff_scale),
        noise_scale=tuple(args.noise_scale),
    )
    transition_focus = TransitionFocusConfig(
        enabled=args.focus_transitions,
        candidates_per_episode=args.transition_candidates,
        min_transition_windows=args.min_transition_windows,
    )
    horizon = HorizonConfig(horizon_s=args.horizon)
    manifest = export_synthetic_dataset(
        scenarios=scenarios,
        episodes=args.episodes,
        duration_s=args.duration,
        window_spec=WindowSpec(window_s=args.window, stride_s=args.stride),
        out_dir=args.out,
        seed=args.seed,
        randomization=randomization,
        transition_focus=transition_focus,
        horizon=horizon,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest.schema_version,
                "total_windows": manifest.total_windows,
                "label_counts": manifest.label_counts,
                "domain_randomization": manifest.domain_randomization,
                "sampling_strategy": manifest.sampling_strategy,
                "horizon": manifest.horizon,
                "artifacts": list(manifest.artifacts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_risk(args: argparse.Namespace) -> int:
    payload = train_risk_model(
        dataset_dir=args.dataset,
        out_dir=args.out,
        config=RiskTrainingConfig(
            model_type=args.model_type,
            calibration=args.calibration,
            feature_set=args.feature_set,
            target=args.target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            split_mode=args.split_mode,
            holdout_column=args.holdout_column,
            holdout_tail=args.holdout_tail,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "model": payload["model"]["type"],
                "target": payload["target"]["mode"],
                "train_accuracy": payload["train"]["accuracy"],
                "validation_accuracy": payload["validation"]["accuracy"] if payload["validation"] else None,
                "test_accuracy": payload["test"]["accuracy"],
                "test_chatter_f1": payload["test"]["binary_chatter_f1"],
                "test_intervention_f1": payload["test"]["intervention_f1"],
                "test_lead_time_f1": payload["lead_time"]["test"]["f1"],
                "test_lead_time_recall": payload["lead_time"]["test"]["recall"],
                "selected_warning_threshold": payload["lead_time"]["threshold_selection"]["selected_threshold"],
                "test_lead_time_selected_threshold_f1": payload["lead_time"]["threshold_selection"]["test_at_selected_threshold"]["f1"],
                "test_lead_time_oracle_f1": payload["lead_time"]["threshold_selection"]["test_oracle_best_f1"]["f1"],
                "threshold_selection_source": payload["lead_time"]["threshold_selection"]["source"],
                "test_event_warning_f1": payload["event_warning"]["test"]["f1"],
                "test_event_warning_recall": payload["event_warning"]["test"]["recall"],
                "selected_event_warning_threshold": payload["event_warning"]["threshold_selection"]["selected_threshold"],
                "event_threshold_selection_source": payload["event_warning"]["threshold_selection"]["source"],
                "split": payload["split"],
                "validation_split": payload["validation_split"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_pseudo_label_replay(args: argparse.Namespace) -> int:
    payload = pseudo_label_replay_dataset(
        dataset_dir=args.dataset,
        out_dir=args.out,
        config=PseudoLabelConfig(
            score_columns=tuple(column.strip() for column in args.score_columns.split(",") if column.strip()),
            positive_scenarios=tuple(scenario.strip() for scenario in args.positive_scenarios.split(",") if scenario.strip()),
            positive_mode=args.positive_mode,
            horizon_s=args.horizon,
            transition_quantile=args.transition_quantile,
            slight_quantile=args.slight_quantile,
            severe_quantile=args.severe_quantile,
            transition_floor=args.transition_floor,
            slight_floor=args.slight_floor,
            severe_floor=args.severe_floor,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "source_dataset": payload["source_dataset"],
                "total_windows": payload["total_windows"],
                "changed_windows": payload["changed_windows"],
                "candidate_windows": payload["candidate_windows"],
                "label_counts_before": payload["label_counts_before"],
                "label_counts_after": payload["label_counts_after"],
                "positive_scenarios": payload["positive_scenarios"],
                "positive_mode": payload["positive_mode"],
                "thresholds": payload["thresholds"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_rl(args: argparse.Namespace) -> int:
    scenarios = tuple(_parse_csv_arg(args.scenarios))
    margin_calibration = _load_optional_margin_calibration(args.margin_calibration)
    randomization = _randomization_from_args(args)
    shield_config = _shield_from_args(args)
    if args.algorithm == "q_learning":
        payload = train_q_learning(
            config=QLearningConfig(
                scenarios=scenarios,
                episodes=args.episodes,
                eval_episodes=args.eval_episodes,
                steps=args.steps,
                decision_interval_s=args.decision_interval,
                seed=args.seed,
                learning_rate=args.learning_rate,
                discount=args.gamma,
                reward_config=_reward_from_args(args),
                shield_config=shield_config,
            ),
            out_dir=args.out,
            margin_calibration=margin_calibration,
            randomization=randomization,
        )
    else:
        payload = train_sb3_policy(
            config=Sb3TrainingConfig(
                algorithm=args.algorithm,
                scenarios=scenarios,
                total_timesteps=args.total_timesteps,
                eval_episodes=args.eval_episodes,
                steps=args.steps,
                decision_interval_s=args.decision_interval,
                seed=args.seed,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                baseline_controllers=tuple(_parse_csv_arg(args.baseline_controllers)),
                candidate_guard=not args.no_candidate_guard,
                reward_config=_reward_from_args(args),
                shield_config=shield_config,
                action_mode=args.action_mode,
                delta_action_scale=args.delta_action_scale,
                delta_feed_scale=args.delta_feed_scale,
                delta_spindle_scale=args.delta_spindle_scale,
                delta_mapping=args.delta_mapping,
            ),
            out_dir=args.out,
            margin_calibration=margin_calibration,
            randomization=randomization,
        )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "algorithm": payload["algorithm"],
                "evaluation_episodes": payload["evaluation"]["episodes"],
                "artifacts": _rl_artifacts(args.algorithm),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_train_rl_multiseed(args: argparse.Namespace) -> int:
    payload = train_multi_seed_policies(
        config=MultiSeedTrainingConfig(
            algorithms=tuple(_parse_csv_arg(args.algorithms)),
            seeds=tuple(_parse_int_csv_arg(args.seeds)),
            scenarios=tuple(_parse_csv_arg(args.scenarios)),
            episodes=args.episodes,
            total_timesteps=args.total_timesteps,
            eval_episodes=args.eval_episodes,
            steps=args.steps,
            decision_interval_s=args.decision_interval,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            baseline_controllers=tuple(_parse_csv_arg(args.baseline_controllers)),
            candidate_guard=not args.no_candidate_guard,
            reward_config=_reward_from_args(args),
            shield_config=_shield_from_args(args),
            action_mode=args.action_mode,
            delta_action_scale=args.delta_action_scale,
            delta_feed_scale=args.delta_feed_scale,
            delta_spindle_scale=args.delta_spindle_scale,
            delta_mapping=args.delta_mapping,
        ),
        out_dir=args.out,
        margin_calibration=_load_optional_margin_calibration(args.margin_calibration),
        randomization=_randomization_from_args(args),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "runs": len(payload["runs"]),
                "aggregate_rows": len(payload["aggregate_summary"]),
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_compare_rl_runs(args: argparse.Namespace) -> int:
    payload = compare_rl_runs(
        runs=tuple(parse_run_ref(value) for value in args.run),
        baseline_label=args.baseline_label,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "profiles": len(payload["profiles"]),
                "learned_policy_rows": len(payload["learned_policy_summary"]),
                "delta_rows": len(payload["delta_summary"]),
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_select_rl_policy(args: argparse.Namespace) -> int:
    payload = select_rl_policy(
        eval_dir=args.eval_dir,
        out_dir=args.out,
        config=PolicySelectionConfig(
            profile_label=args.profile_label,
            min_relative_mrr=args.min_relative_mrr,
            risk_weight=args.risk_weight,
            worst_risk_weight=args.worst_risk_weight,
            unstable_risk_weight=args.unstable_risk_weight,
            mrr_weight=args.mrr_weight,
            mrr_shortfall_weight=args.mrr_shortfall_weight,
            guard_fallback_weight=args.guard_fallback_weight,
            shield_rejection_weight=args.shield_rejection_weight,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "selected_seed": payload["selected"]["seed"],
                "selection_score": payload["selected"]["selection_score"],
                "source_model_path": payload["selected"]["source_model_path"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_rl_policy(args: argparse.Namespace) -> int:
    payload = run_selected_rl_shadow_policy(
        selection_path=args.selection,
        out_dir=args.out,
        config=RLShadowReplayConfig(
            scenarios=tuple(_parse_csv_arg(args.scenarios)),
            episodes=args.episodes,
            steps=args.steps,
            decision_interval_s=args.decision_interval,
            seed=args.seed,
        ),
        margin_calibration=_load_optional_margin_calibration(args.margin_calibration),
        randomization=_randomization_from_args(args),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "mode": payload["deployment_boundary"]["mode"],
                "cnc_writes_enabled": payload["deployment_boundary"]["cnc_writes_enabled"],
                "selected_seed": payload["selected_policy"]["seed"],
                "recommendation_windows": payload["window_metrics"]["recommendation_windows"],
                "action_fraction": payload["window_metrics"]["action_fraction"],
                "guard_fallbacks": payload["window_metrics"]["guard_fallbacks"],
                "relative_mrr_proxy": payload["window_metrics"]["relative_mrr_proxy"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_gate_rl_shadow(args: argparse.Namespace) -> int:
    payload = gate_rl_shadow_replay(
        shadow_dir=args.shadow_dir,
        out_dir=args.out,
        config=RLShadowGateConfig.from_profile(
            args.profile,
            min_recommendation_windows=args.min_recommendation_windows,
            max_mean_risk=args.max_mean_risk,
            max_max_risk=args.max_max_risk,
            max_unstable_mean_risk=args.max_unstable_mean_risk,
            min_relative_mrr=args.min_relative_mrr,
            max_guard_fallback_fraction=args.max_guard_fallback_fraction,
            max_action_fraction=args.max_action_fraction,
            max_shield_rejections=args.max_shield_rejections,
            require_no_cnc_writes=not args.allow_cnc_writes,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "profile": payload["gate_config"]["profile"],
                "status": payload["decision"]["status"],
                "promotion_level": payload["decision"]["promotion_level"],
                "hardware_actuation_allowed": payload["decision"]["hardware_actuation_allowed"],
                "failed_checks": payload["decision"]["failed_checks"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["decision"]["passed"] else 2


def _cmd_internal_demo_report(args: argparse.Namespace) -> int:
    payload = write_internal_demo_report(
        out_path=args.out,
        summary_path=args.summary_out,
        config=InternalDemoConfig(
            calibration_dir=args.calibration_dir,
            risk_model_dir=args.risk_model_dir,
            closed_loop_dir=args.closed_loop_dir,
            rl_comparison_dir=args.rl_comparison_dir,
            champion_dir=args.champion_dir,
            rl_shadow_replay_dir=args.rl_shadow_replay_dir,
            shadow_review_gate_dir=args.shadow_review_gate_dir,
            live_shadow_gate_dir=args.live_shadow_gate_dir,
            hardware_gate_dir=args.hardware_gate_dir,
            shadow_eval_dir=args.shadow_eval_dir,
            counterfactual_dir=args.counterfactual_dir,
            test_status=args.test_status,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "summary_out": str(args.summary_out) if args.summary_out is not None else None,
                "stage": payload["conclusion"]["current_stage"],
                "readiness": payload["readiness"]["status"],
                "hardware_ready": payload["conclusion"]["hardware_ready"],
                "gate_statuses": {row["profile"]: row["status"] for row in payload["promotion_gates"]},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_real_data_benchmark(args: argparse.Namespace) -> int:
    runs = tuple(parse_real_data_run_spec(value) for value in args.run) if args.run else None
    payload = write_real_data_benchmark(out_dir=args.out, runs=runs)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(payload["rows"]),
                "skipped": len(payload["skipped"]),
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_risk_error_analysis(args: argparse.Namespace) -> int:
    payload = write_risk_error_analysis(
        model_dir=args.model_dir,
        out_dir=args.out,
        group_column=args.group_column,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "windows": payload["windows"],
                "groups": len(payload["group_metrics"]),
                "worst_groups": payload["worst_groups"][:5],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_icnc_manifest(args: argparse.Namespace) -> int:
    manifest = write_icnc_source_manifest(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "record_url": manifest["record_url"],
                "filename": manifest["filename"],
                "size_bytes": manifest["size_bytes"],
                "md5": manifest["md5"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_kit_industrial_manifest(args: argparse.Namespace) -> int:
    manifest = write_kit_industrial_source_manifest(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "record_url": manifest["record_url"],
                "download_url": manifest["download_url"],
                "filename": manifest["filename"],
                "size_bytes": manifest["size_bytes"],
                "resumable_curl": (
                    "curl -L -C - --fail "
                    f"-o data/raw/kit_industrial/{KIT_INDUSTRIAL_FILENAME} "
                    f"'{manifest['download_url']}'"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_mt_cutting_manifest(args: argparse.Namespace) -> int:
    manifest = write_mt_cutting_source_manifest(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "repository_url": manifest["repository_url"],
                "dataset": manifest["dataset"],
                "bridge_value": manifest["bridge_value"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_inspect_kit_industrial(args: argparse.Namespace) -> int:
    payload = inspect_kit_industrial_dataset(args.source)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_mt_cutting(args: argparse.Namespace) -> int:
    payload = inspect_mt_cutting_dataset(args.source)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_bosch_cnc(args: argparse.Namespace) -> int:
    payload = inspect_bosch_cnc_dataset(args.source)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _machine_run_config_from_args(args: argparse.Namespace) -> MachineRunIngestConfig:
    defaults = MachineRunIngestConfig()
    return MachineRunIngestConfig(
        sensor_columns=tuple(column.strip() for column in args.sensor_columns.split(",") if column.strip()),
        window_s=getattr(args, "window", defaults.window_s),
        stride_s=getattr(args, "stride", defaults.stride_s),
        horizon_s=getattr(args, "horizon", defaults.horizon_s),
        default_label=args.default_label,
        max_windows=getattr(args, "max_windows", None),
    )


def _cmd_machine_run_template(args: argparse.Namespace) -> int:
    payload = write_machine_run_template(args.out, overwrite=args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_validate_machine_run(args: argparse.Namespace) -> int:
    payload = inspect_machine_run(args.source, _machine_run_config_from_args(args))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 2


def _cmd_ingest_machine_run(args: argparse.Namespace) -> int:
    payload = ingest_machine_run_dataset(
        source=args.source,
        out_dir=args.out,
        config=_machine_run_config_from_args(args),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_inspect_kit_mat(args: argparse.Namespace) -> int:
    payload = inspect_kit_synchronized_mat(
        source=args.source,
        trial=args.trial,
        max_datasets=args.max_datasets,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_download_icnc(args: argparse.Namespace) -> int:
    payload = download_icnc_dataset(
        out_path=args.out,
        manifest_path=args.manifest_out,
        skip_existing=not args.force,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_ingest_kit_industrial(args: argparse.Namespace) -> int:
    signal_columns = tuple(column.strip() for column in args.signal_columns.split(",") if column.strip())
    payload = ingest_kit_industrial_dataset(
        source=args.source,
        out_dir=args.out,
        trials=[trial.strip() for trial in args.trials.split(",") if trial.strip()],
        config=KITIndustrialIngestConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            sample_rate_hz=args.sample_rate,
            signal_columns=signal_columns,  # type: ignore[arg-type]
            include_other_anomalies=args.include_other_anomalies,
            max_windows=args.max_windows,
        ),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "sources": payload["sources"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ingest_kit_mat(args: argparse.Namespace) -> int:
    signal_names = tuple(name.strip() for name in args.signal_names.split(",") if name.strip())
    payload = ingest_kit_mat_dataset(
        source=args.source,
        out_dir=args.out,
        trials=[trial.strip() for trial in args.trials.split(",") if trial.strip()],
        config=KITMatIngestConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            signal_names=signal_names,  # type: ignore[arg-type]
            standardize_signals=args.standardize_signals,
            include_other_anomalies=args.include_other_anomalies,
            max_windows=args.max_windows,
            max_samples_per_trial=args.max_samples_per_trial,
        ),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "sources": payload["sources"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ingest_mt_cutting(args: argparse.Namespace) -> int:
    payload = ingest_mt_cutting_dataset(
        source=args.source,
        out_dir=args.out,
        experiments=[experiment.strip() for experiment in args.experiments.split(",") if experiment.strip()],
        config=MTCuttingIngestConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            sensors=tuple(_parse_int_csv_arg(args.sensors)),
            include_unknown=args.include_unknown,
            max_experiments=args.max_experiments,
            max_windows=args.max_windows,
        ),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "sources": payload["sources"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ingest_icnc(args: argparse.Namespace) -> int:
    payload = ingest_icnc_dataset(
        source=args.source,
        out_dir=args.out,
        config=ICNCIngestConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            flute_count=args.flutes,
            modal_frequency_hz=args.modal_frequency,
            default_sample_rate_hz=args.default_sample_rate,
            default_spindle_rpm=args.default_spindle_rpm,
            default_feed_per_tooth_m=args.default_feed_per_tooth,
            include_unknown=args.include_unknown,
            max_packages_per_file=args.max_packages_per_file,
            max_windows=args.max_windows,
        ),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "sources": payload["sources"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ingest_bosch_cnc(args: argparse.Namespace) -> int:
    payload = ingest_bosch_cnc_dataset(
        source=args.source,
        out_dir=args.out,
        machines=[machine.strip() for machine in args.machines.split(",") if machine.strip()],
        operations=[operation.strip() for operation in args.operations.split(",") if operation.strip()],
        config=BoschCNCIngestConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            sample_rate_hz=args.sample_rate,
            spindle_rpm=args.spindle_rpm,
            flute_count=args.flute_count,
            max_files=args.max_files,
            max_files_per_quality=args.max_files_per_quality,
            max_windows=args.max_windows,
        ),
    )
    manifest = payload["manifest"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "schema_version": manifest["schema_version"],
                "total_windows": manifest["total_windows"],
                "label_counts": manifest["label_counts"],
                "sources": payload["sources"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_eval_rl_run(args: argparse.Namespace) -> int:
    payload = evaluate_saved_sb3_run(
        source_dir=args.run_dir,
        out_dir=args.out,
        scenarios=tuple(_parse_csv_arg(args.scenarios)),
        eval_episodes=args.eval_episodes,
        steps=args.steps,
        decision_interval_s=args.decision_interval,
        seed=args.seed,
        margin_calibration=_load_optional_margin_calibration(args.margin_calibration),
        randomization=_randomization_from_args(args),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "runs": len(payload["runs"]),
                "aggregate_rows": len(payload["aggregate_summary"]),
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_eval(args: argparse.Namespace) -> int:
    payload = run_shadow_evaluation(
        model_dir=args.model_dir,
        out_dir=args.out,
        config=ShadowPolicyConfig(
            threshold_source=args.threshold_source,
            warning_threshold=args.warning_threshold,
            clear_threshold=args.clear_threshold,
            warning_feed_override=args.warning_feed,
            warning_spindle_override=args.warning_spindle,
            max_feed_delta=args.max_feed_delta,
            max_spindle_delta=args.max_spindle_delta,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "warning_threshold": payload["policy"]["resolved_warning_threshold"],
                "warning_fraction": payload["window_metrics"]["warning_fraction"],
                "action_fraction": payload["window_metrics"]["action_fraction"],
                "relative_mrr_proxy": payload["window_metrics"]["relative_mrr_proxy"],
                "event_f1": payload["event_metrics"]["f1"],
                "event_recall": payload["event_metrics"]["recall"],
                "false_warning_episodes": payload["event_metrics"]["false_warning_episodes"],
                "mean_event_lead_time_s": payload["event_metrics"]["mean_detected_lead_time_s"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_counterfactual(args: argparse.Namespace) -> int:
    payload = run_shadow_counterfactual(
        shadow_dir=args.shadow_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        config=CounterfactualConfig(
            sample_rate_hz=args.sample_rate,
            sensor_noise_std=args.sensor_noise,
            seed=args.seed,
            max_windows=args.max_windows,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "windows": payload["windows"]["count"],
                "episodes": payload["episodes"]["episode_count"],
                "action_fraction": payload["windows"]["action_fraction"],
                "relative_mrr_proxy": payload["windows"]["mean_relative_mrr_proxy"],
                "baseline_risk": payload["windows"]["mean_baseline_risk"],
                "shadow_risk": payload["windows"]["mean_shadow_risk"],
                "risk_reduction": payload["windows"]["mean_risk_reduction"],
                "mitigated_event_episodes": payload["episodes"]["mitigated_event_episodes"],
                "worsened_event_episodes": payload["episodes"]["worsened_event_episodes"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_episode_counterfactual(args: argparse.Namespace) -> int:
    payload = run_shadow_episode_counterfactual(
        shadow_dir=args.shadow_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        config=CounterfactualConfig(
            sample_rate_hz=args.sample_rate,
            sensor_noise_std=args.sensor_noise,
            seed=args.seed,
            max_windows=args.max_windows,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "windows": payload["windows"]["count"],
                "episodes": payload["episodes"]["episode_count"],
                "action_fraction": payload["windows"]["action_fraction"],
                "relative_mrr_proxy": payload["windows"]["mean_relative_mrr_proxy"],
                "baseline_risk": payload["windows"]["mean_baseline_risk"],
                "shadow_risk": payload["windows"]["mean_shadow_risk"],
                "risk_reduction": payload["windows"]["mean_risk_reduction"],
                "mitigated_event_episodes": payload["episodes"]["mitigated_event_episodes"],
                "worsened_event_episodes": payload["episodes"]["worsened_event_episodes"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_action_sweep(args: argparse.Namespace) -> int:
    payload = run_shadow_action_sweep(
        shadow_dir=args.shadow_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        feed_values=tuple(_parse_float_csv_arg(args.feed_values)),
        spindle_values=tuple(_parse_float_csv_arg(args.spindle_values)),
        config=CounterfactualConfig(
            sample_rate_hz=args.sample_rate,
            sensor_noise_std=args.sensor_noise,
            seed=args.seed,
            max_windows=args.max_windows,
        ),
    )
    best = payload["best_policy"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidate_count": payload["candidate_count"],
                "best_feed_override": best["feed_override"],
                "best_spindle_override": best["spindle_override"],
                "best_relative_mrr_proxy": best["relative_mrr_proxy"],
                "best_risk_reduction": best["mean_risk_reduction"],
                "best_shadow_risk": best["mean_shadow_risk"],
                "mitigated_event_episodes": best["mitigated_event_episodes"],
                "worsened_event_episodes": best["worsened_event_episodes"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_stability_policy(args: argparse.Namespace) -> int:
    payload = run_stability_margin_shadow_policy(
        shadow_dir=args.shadow_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        max_windows=args.max_windows,
        config=StabilityPolicyConfig(
            min_spindle_override=args.min_spindle,
            max_spindle_override=args.max_spindle,
            candidate_count=args.candidates,
            feed_override=args.feed,
            min_margin_improvement=args.min_margin_improvement,
            margin_weight=args.margin_weight,
            productivity_weight=args.productivity_weight,
            move_penalty=args.move_penalty,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "warning_fraction": payload["window_metrics"]["warning_fraction"],
                "action_fraction": payload["window_metrics"]["action_fraction"],
                "relative_mrr_proxy": payload["window_metrics"]["mean_relative_mrr_proxy"],
                "mean_selected_margin_improvement": payload["window_metrics"]["mean_selected_margin_improvement"],
                "event_f1": payload["event_metrics"]["f1"],
                "event_recall": payload["event_metrics"]["recall"],
                "false_warning_episodes": payload["event_metrics"]["false_warning_episodes"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_shadow_counterfactual_policy(args: argparse.Namespace) -> int:
    payload = run_counterfactual_risk_shadow_policy(
        shadow_dir=args.shadow_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        counterfactual_config=CounterfactualConfig(
            sample_rate_hz=args.sample_rate,
            sensor_noise_std=args.sensor_noise,
            seed=args.seed,
            max_windows=args.max_windows,
        ),
        policy_config=CounterfactualPolicyConfig(
            feed_values=tuple(_parse_float_csv_arg(args.feed_values)),
            spindle_values=tuple(_parse_float_csv_arg(args.spindle_values)),
            min_risk_reduction=args.min_risk_reduction,
            risk_weight=args.risk_weight,
            productivity_weight=args.productivity_weight,
            move_penalty=args.move_penalty,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "warning_fraction": payload["window_metrics"]["warning_fraction"],
                "action_fraction": payload["window_metrics"]["action_fraction"],
                "relative_mrr_proxy": payload["window_metrics"]["mean_relative_mrr_proxy"],
                "mean_selected_risk_reduction": payload["window_metrics"]["mean_selected_risk_reduction"],
                "event_f1": payload["event_metrics"]["f1"],
                "event_recall": payload["event_metrics"]["recall"],
                "false_warning_episodes": payload["event_metrics"]["false_warning_episodes"],
                "artifacts": payload["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _risk_summary(risk) -> dict:
    return {
        "label": risk.label,
        "margin_physics": risk.margin_physics,
        "margin_signal": risk.margin_signal,
        "reason_codes": list(risk.reason_codes),
        "risk_chatter_horizon": risk.risk_chatter_horizon,
        "risk_chatter_now": risk.risk_chatter_now,
        "uncertainty": risk.uncertainty,
    }


def _parse_csv_arg(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one comma-separated value")
    return parsed


def _parse_float_csv_arg(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one comma-separated float")
    return parsed


def _parse_int_csv_arg(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one comma-separated integer")
    return parsed


def _rl_artifacts(algorithm: str) -> list[str]:
    if algorithm == "q_learning":
        return [
            "training_episodes.csv",
            "evaluation_episodes.csv",
            "evaluation_summary.csv",
            "policy.json",
            "metrics.json",
            "report.md",
            "learning_curve.svg",
        ]
    return [
        "model.zip",
        "evaluation_actions.csv",
        "evaluation_episodes.csv",
        "action_diagnostics_summary.csv",
        "evaluation_summary.csv",
        "baseline_episodes.csv",
        "baseline_summary.csv",
        "comparison_summary.csv",
        "metrics.json",
        "report.md",
    ]


def _load_optional_margin_calibration(path: Path | None):
    if path is None:
        return None
    return load_margin_calibration(path)


if __name__ == "__main__":
    raise SystemExit(main())
