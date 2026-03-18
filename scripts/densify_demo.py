from __future__ import annotations

import argparse

from scenestreamer._startup import quiet_native_startup_noise
from _common import add_model_args, add_run_args, load_model_from_args, print_stage, require_scenarionet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/densify_demo.py",
        description="Qualitative densification demo (generate to max agents).",
    )
    add_run_args(parser)
    add_model_args(parser)
    parser.add_argument("--dataset-dir", required=True, help="ScenarioNet database directory")
    parser.add_argument("--split", choices=["training", "test"], default="test")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--max-agents", type=int, default=128)
    parser.add_argument("--force-no-end", action="store_true", help="Disable end token so it keeps generating agents")
    args = parser.parse_args(argv)

    require_scenarionet()

    with quiet_native_startup_noise():
        from scenestreamer.paper.densify_demo import run_densify_demo

    print_stage("densify", 1, 3, "loading model")
    pl_model = load_model_from_args(args)
    print_stage("densify", 2, 3, "running densification demo")
    out_dir = run_densify_demo(
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
    print_stage("densify", 3, 3, f"finished; artifacts saved to {out_dir}")
    print(str(out_dir))


if __name__ == "__main__":
    main()

