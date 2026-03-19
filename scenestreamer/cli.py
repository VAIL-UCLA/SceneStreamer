from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import yaml
from scenestreamer.model_hub import DEFAULT_HF_REPO
from scenestreamer._startup import configure_startup_noise_filters, quiet_native_startup_noise

configure_startup_noise_filters()


def _to_plain(obj: Any) -> Any:
    if hasattr(obj, "items"):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _to_easydict(obj: Any):
    from easydict import EasyDict

    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def load_yaml_config(path: str | os.PathLike[str]):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return _to_easydict(data)


def apply_overrides(cfg, overrides: list[str]) -> None:
    """
    Apply overrides of form KEY=VALUE where KEY is dot-delimited.
    VALUE is parsed using yaml.safe_load (so numbers/bools/lists work).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected KEY=VALUE): {item}")
        key, raw_val = item.split("=", 1)
        value = yaml.safe_load(raw_val)

        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not hasattr(cur, p):
                setattr(cur, p, _to_easydict({}))
            cur = getattr(cur, p)
        setattr(cur, parts[-1], value)


@dataclass(frozen=True)
class RunPaths:
    run_dir: pathlib.Path
    config_path: pathlib.Path
    metrics_path: pathlib.Path


def make_run_dir(base_dir: str | os.PathLike[str], run_id: str | None) -> RunPaths:
    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        metrics_path=run_dir / "metrics.json",
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    # Prefer failing fast on missing ScenarioNet without importing heavy deps.
    try:
        with quiet_native_startup_noise():
            import scenarionet  # noqa: F401
    except ModuleNotFoundError as e:
        raise e

    with quiet_native_startup_noise():
        from scenestreamer.dataset.dataset import SceneStreamerDataset

    cfg = load_yaml_config(args.config)
    apply_overrides(cfg, args.set or [])

    # Paths: prefer CLI args, but allow config overrides.
    if args.train_dir:
        cfg.DATA.TRAINING_DATA_DIR = args.train_dir
    if args.test_dir:
        cfg.DATA.TEST_DATA_DIR = args.test_dir

    cfg.DATA.USE_CACHE = True

    run = make_run_dir(args.artifacts_dir, args.run_id)
    with open(run.config_path, "w") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False)

    mode = args.split
    ds = SceneStreamerDataset(cfg, mode)

    # Iterate to materialize cache files.
    for i in range(len(ds)):
        _ = ds[i]
        if args.limit is not None and (i + 1) >= args.limit:
            break

    metrics = {
        "status": "ok",
        "mode": mode,
        "train_dir": getattr(cfg.DATA, "TRAINING_DATA_DIR", None),
        "test_dir": getattr(cfg.DATA, "TEST_DATA_DIR", None),
        "limit": args.limit,
    }
    with open(run.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(str(run.run_dir))


def _load_model_from_args(args: argparse.Namespace):
    with quiet_native_startup_noise():
        from scenestreamer.utils import utils

    if args.ckpt:
        return utils.get_model(checkpoint_path=args.ckpt, device=args.device)
    if args.hf_repo:
        return utils.get_model(huggingface_repo=args.hf_repo, huggingface_file=args.hf_file, device=args.device)
    raise ValueError("Must provide either --hf-repo/--hf-file or --ckpt")


def cmd_table1(args: argparse.Namespace) -> None:
    with quiet_native_startup_noise():
        from scenestreamer.paper.table1_mmd import run_table1_mmd

    pl_model = _load_model_from_args(args)
    run_dir = run_table1_mmd(
        pl_model=pl_model,
        dataset_dir=args.dataset_dir,
        split=args.split,
        limit=args.limit,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
    print(str(run_dir))


def cmd_table2(args: argparse.Namespace) -> None:
    with quiet_native_startup_noise():
        from scenestreamer.paper.table2_motion import run_table2_motion

    pl_model = _load_model_from_args(args)
    run_dir = run_table2_motion(
        pl_model=pl_model,
        dataset_dir=args.dataset_dir,
        split=args.split,
        mode=args.mode,
        num_modes=args.num_modes,
        limit=args.limit,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
    print(str(run_dir))


def cmd_densify_demo(args: argparse.Namespace) -> None:
    with quiet_native_startup_noise():
        from scenestreamer.paper.densify_demo import run_densify_demo

    pl_model = _load_model_from_args(args)
    run_dir = run_densify_demo(
        pl_model=pl_model,
        dataset_dir=args.dataset_dir,
        split=args.split,
        scenario_index=args.scenario_index,
        max_agents=args.max_agents,
        force_no_end=args.force_no_end,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
    print(str(run_dir))

def cmd_table3_train(args: argparse.Namespace) -> None:
    from scenestreamer.rl.entrypoints import run_table3_train

    run_table3_train(args)


def cmd_table3_eval(args: argparse.Namespace) -> None:
    from scenestreamer.rl.entrypoints import run_table3_eval

    run_table3_eval(args)


def build_parser() -> argparse.ArgumentParser:
    from scenestreamer.rl.entrypoints import add_table3_eval_args, add_table3_train_args

    parser = argparse.ArgumentParser(prog="scenestreamer", description="SceneStreamer paper reproduction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--artifacts-dir", default="artifacts", help="Directory to write run artifacts")
        p.add_argument("--run-id", default=None, help="Run ID (default: timestamp)")
        p.add_argument("--seed", type=int, default=0)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="auto", help="torch device string: auto, cuda, mps, or cpu")
        p.add_argument("--ckpt", default=None, help="Path to a .ckpt checkpoint")
        p.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HuggingFace repo id, e.g. user/repo")
        p.add_argument("--hf-file", default="scenestreamer-full-large.ckpt", help="HuggingFace filename, e.g. model.ckpt")

    # preprocess
    p = sub.add_parser("preprocess", help="Preprocess ScenarioNet SD dataset and build cache")
    add_run_args(p)
    p.add_argument("--config", default="cfgs/motion_default.yaml")
    p.add_argument("--set", action="append", default=[], help="Override config KEY=VALUE (repeatable)")
    p.add_argument("--train-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--split", choices=["training", "test"], default="training")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_preprocess)

    # table1
    p = sub.add_parser("table1", help="Table 1: initial state MMD (strict + relaxed)")
    add_run_args(p)
    add_model_args(p)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", choices=["training", "test"], default="test")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_table1)

    # table2
    p = sub.add_parser("table2", help="Table 2: motion prediction (ADE/FDE + ADD/FDD)")
    add_run_args(p)
    add_model_args(p)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", choices=["training", "test"], default="test")
    p.add_argument("--mode", choices=["motion", "full"], default="motion")
    p.add_argument("--num-modes", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_table2)

    # demo
    p = sub.add_parser("densify-demo", help="Qualitative densification demo (generate to max agents)")
    add_run_args(p)
    add_model_args(p)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", choices=["training", "test"], default="test")
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--max-agents", type=int, default=128)
    p.add_argument("--force-no-end", action="store_true", help="Disable end token so it keeps generating agents")
    p.set_defaults(func=cmd_densify_demo)

    p = sub.add_parser("table3-train", help="Table 3: TD3 RL training in MetaDrive")
    add_table3_train_args(p)
    p.set_defaults(func=cmd_table3_train)

    p = sub.add_parser("table3-eval", help="Table 3: TD3 RL evaluation in MetaDrive")
    add_table3_eval_args(p)
    p.set_defaults(func=cmd_table3_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ModuleNotFoundError as e:
        # Most common in a fresh environment: scenarionet / waymo-open-dataset missing.
        msg = str(e)
        if "scenarionet" in msg:
            raise SystemExit(
                "Missing dependency 'scenarionet'. Install it via:\n"
                "  pip install git+https://github.com/metadriverse/scenarionet.git\n"
            ) from e
        raise


if __name__ == "__main__":
    main()
