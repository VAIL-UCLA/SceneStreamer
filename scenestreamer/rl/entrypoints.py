from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import torch

from scenestreamer._startup import configure_startup_noise_filters, quiet_native_startup_noise
from scenestreamer.model_hub import DEFAULT_HF_REPO

configure_startup_noise_filters()


def _print_json_block(prefix: str, title: str, payload: dict[str, Any]) -> None:
    print(f"[{prefix}] {title}:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def require_rl_deps() -> None:
    try:
        with quiet_native_startup_noise():
            import gymnasium  # noqa: F401
            import stable_baselines3  # noqa: F401
            from metadrive.envs.scenario_env import ScenarioEnv  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing RL dependencies. Install them via:\n"
            "  uv sync --extra rl\n"
        ) from e


def _generator_backend_name(name: str) -> str:
    mapping = {
        "scenestreamer": "SceneStreamer",
    }
    return mapping[name]


def add_table3_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mode",
        choices=["open-loop", "closed-loop"],
        default="closed-loop",
        help=(
            "RL training mode. 'closed-loop' regenerates the scenario once before each episode reset "
            "using the current learning agent trajectory history; it does not run SceneStreamer at every simulator step."
        ),
    )
    p.add_argument("--train-data-dir", required=True, help="ScenarioNet database directory for RL training episodes")
    p.add_argument("--eval-data-dir", required=True, help="ScenarioNet database directory for RL evaluation episodes")
    p.add_argument("--save-dir", required=True, help="Directory to save TD3 checkpoints and logs")
    p.add_argument("--exp-name", default="scenestreamer-td3", help="Experiment name")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--training-steps", type=int, default=2_000_000, help="Total TD3 environment steps")
    p.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation frequency in environment steps")
    p.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    p.add_argument("--num-eval-envs", type=int, default=1, help="Number of parallel MetaDrive eval environments")
    p.add_argument("--horizon", type=int, default=100, help="Training horizon in MetaDrive")
    p.add_argument("--eval-horizon", type=int, default=100, help="Evaluation horizon in MetaDrive")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ckpt-path", default=None, help="Optional TD3 checkpoint to resume from")
    p.add_argument(
        "--generator",
        choices=["scenestreamer"],
        default="scenestreamer",
        help="Closed-loop scenario generator backend",
    )
    p.add_argument(
        "--generator-model",
        default=None,
        help=(
            "Closed-loop SceneStreamer generator model name, e.g. scenestreamer-base-large or scenestreamer-full-xl. "
            "Required when --mode closed-loop."
        ),
    )
    p.add_argument(
        "--generator-hf-repo",
        default=DEFAULT_HF_REPO,
        help="HuggingFace repo for the closed-loop SceneStreamer generator checkpoint",
    )
    p.add_argument(
        "--generator-hf-file",
        default=None,
        help=(
            "Optional HuggingFace checkpoint filename override for the closed-loop generator. "
            "If omitted, the selected generator model config provides the default checkpoint."
        ),
    )
    p.add_argument(
        "--generator-ckpt",
        default=None,
        help="Optional local checkpoint path override for the closed-loop generator",
    )
    p.add_argument("--no-adaptive", action="store_true", help="Use GT ego trajectory instead of adaptive closed-loop edits")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="scgen")
    p.add_argument("--wandb-team", default="drivingforce")


def add_table3_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", required=True, help="ScenarioNet database directory for RL policy evaluation")
    p.add_argument("--ckpt-dir", required=True, help="Directory containing TD3 checkpoints named seed_*_<steps>_steps.zip")
    p.add_argument("--ckpt-steps", required=True, type=int, help="Checkpoint step number to evaluate across seeds")
    p.add_argument("--eval-horizon", type=int, default=100, help="Evaluation horizon in MetaDrive")
    p.add_argument("--num-scenarios", type=int, default=100, help="Number of evaluation scenarios")


def run_table3_train(args: argparse.Namespace) -> None:
    require_rl_deps()
    if args.mode == "closed-loop" and not args.generator_model:
        raise SystemExit("--generator-model is required when --mode closed-loop")
    if args.generator_ckpt and args.generator_hf_file:
        raise SystemExit("Provide either --generator-ckpt or --generator-hf-file, not both.")
    if not torch.cuda.is_available():
        raise SystemExit("RL training currently requires CUDA. Please run on a CUDA-enabled machine.")

    generator_hf_repo = None
    if args.mode == "closed-loop" and not args.generator_ckpt and args.generator_hf_file is not None:
        generator_hf_repo = args.generator_hf_repo

    with quiet_native_startup_noise():
        from metadrive.policy.env_input_policy import EnvInputPolicy
        from metadrive.scenario.utils import get_number_of_scenarios
        from scenestreamer.rl.training import train_wrapper

    train_num_scenarios = get_number_of_scenarios(args.train_data_dir)
    eval_num_scenarios = get_number_of_scenarios(args.eval_data_dir)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_train = dict(
        store_map=False,
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=args.train_data_dir,
        agent_policy=EnvInputPolicy,
        start_scenario_index=0,
        num_scenarios=train_num_scenarios,
        sequential_seed=False,
        horizon=args.horizon,
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,
    )
    config_eval = dict(
        store_map=False,
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=args.eval_data_dir,
        agent_policy=EnvInputPolicy,
        start_scenario_index=0,
        num_scenarios=eval_num_scenarios,
        sequential_seed=True,
        horizon=args.eval_horizon,
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,
        crash_vehicle_done=False,
        out_of_route_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=False,
    )
    if args.mode == "closed-loop":
        config_train["total_timesteps"] = args.training_steps

    if args.mode == "closed-loop":
        print(
            "[table3] Closed-loop means SceneStreamer regenerates the scenario once at episode reset using the "
            "current learning agent trajectory history. It is not running SceneStreamer at every simulator step."
        )

    _print_json_block(
        "table3",
        "Training setup",
        {
            "mode": args.mode,
            "train_data_dir": args.train_data_dir,
            "eval_data_dir": args.eval_data_dir,
            "save_dir": str(save_dir),
            "seed": args.seed,
            "training_steps": args.training_steps,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "horizon": args.horizon,
            "eval_horizon": args.eval_horizon,
            "num_train_scenarios": train_num_scenarios,
            "num_eval_scenarios": eval_num_scenarios,
            "num_eval_envs": args.num_eval_envs,
            "learning_rate": args.learning_rate,
            "generator": args.generator if args.mode == "closed-loop" else None,
            "generator_model": args.generator_model if args.mode == "closed-loop" else None,
            "generator_hf_repo": generator_hf_repo if args.mode == "closed-loop" else None,
            "generator_hf_file": args.generator_hf_file if args.mode == "closed-loop" else None,
            "generator_ckpt": args.generator_ckpt if args.mode == "closed-loop" else None,
            "no_adaptive": args.no_adaptive if args.mode == "closed-loop" else None,
        },
    )

    train_wrapper(
        config_train=config_train,
        config_eval=config_eval,
        exp_name=args.exp_name,
        seed=args.seed,
        save_path=str(save_dir),
        ckpt_path=args.ckpt_path,
        training_steps=args.training_steps,
        eval_freq=args.eval_freq,
        lr=args.learning_rate,
        wandb_config={
            "use_wandb": args.wandb,
            "wandb_project": args.wandb_project,
            "wandb_team": args.wandb_team,
        },
        closed_loop=args.mode == "closed-loop",
        closed_loop_dir=args.train_data_dir,
        closed_loop_generator=_generator_backend_name(args.generator),
        model_name=args.generator_model,
        num_eval_envs=args.num_eval_envs,
        eval_ep=args.eval_episodes,
        no_adaptive=args.no_adaptive,
        generator_hf_repo=generator_hf_repo if args.mode == "closed-loop" else None,
        generator_hf_file=args.generator_hf_file if args.mode == "closed-loop" else None,
        generator_ckpt_path=args.generator_ckpt if args.mode == "closed-loop" else None,
    )


def run_table3_eval(args: argparse.Namespace) -> None:
    require_rl_deps()
    with quiet_native_startup_noise():
        from metadrive.policy.env_input_policy import EnvInputPolicy
        from scenestreamer.rl.evaluation import eval_ckpt_for_seeds

    config_eval = dict(
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=args.data_dir,
        start_scenario_index=0,
        num_scenarios=args.num_scenarios,
        agent_policy=EnvInputPolicy,
        force_render_fps=10,
        reactive_traffic=False,
        sequential_seed=True,
        horizon=args.eval_horizon,
        out_of_route_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=False,
    )

    _print_json_block(
        "table3",
        "Evaluation setup",
        {
            "data_dir": args.data_dir,
            "ckpt_dir": args.ckpt_dir,
            "ckpt_steps": args.ckpt_steps,
            "eval_horizon": args.eval_horizon,
            "num_scenarios": args.num_scenarios,
        },
    )
    results = eval_ckpt_for_seeds(config_eval, args.ckpt_dir, args.ckpt_steps)
    _print_json_block("table3", "Evaluation summary", results)


def build_table3_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/table3_train.py",
        description="Table 3: TD3 training with open-loop or closed-loop MetaDrive scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_table3_train_args(parser)
    return parser


def build_table3_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/table3_eval.py",
        description="Table 3: evaluate TD3 checkpoints in MetaDrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_table3_eval_args(parser)
    return parser


def main_table3_train(argv: list[str] | None = None) -> None:
    parser = build_table3_train_parser()
    args = parser.parse_args(argv)
    run_table3_train(args)


def main_table3_eval(argv: list[str] | None = None) -> None:
    parser = build_table3_eval_parser()
    args = parser.parse_args(argv)
    run_table3_eval(args)
