from __future__ import annotations

import argparse

from scenestreamer._startup import quiet_native_startup_noise
from _common import add_model_args, add_run_args, load_model_from_args, print_stage, require_scenarionet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/table1_mmd.py",
        description="Table 1: initial state MMD (strict + relaxed).",
    )
    add_run_args(parser)
    add_model_args(parser)
    parser.add_argument("--dataset-dir", required=True, help="ScenarioNet database directory")
    parser.add_argument("--split", choices=["training", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    require_scenarionet()

    with quiet_native_startup_noise():
        from scenestreamer.paper.table1_mmd import run_table1_mmd

    print_stage("table1", 1, 3, "loading model")
    pl_model = load_model_from_args(args)
    print_stage("table1", 2, 3, "running evaluation")
    out_dir = run_table1_mmd(
        pl_model=pl_model,
        dataset_dir=args.dataset_dir,
        split=args.split,
        limit=args.limit,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
    print_stage("table1", 3, 3, f"finished; artifacts saved to {out_dir}")
    print(str(out_dir))


if __name__ == "__main__":
    main()

