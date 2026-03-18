from __future__ import annotations

import argparse
import json
import pathlib
import time

from scenestreamer._startup import quiet_native_startup_noise
from _common import add_config_args, add_run_args, apply_overrides, load_yaml_config, require_scenarionet
from tqdm import tqdm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/preprocess.py",
        description="Preprocess a ScenarioNet database directory (build per-scenario cache).",
    )
    add_run_args(parser)
    add_config_args(parser)
    parser.add_argument("--dataset-dir", required=True, help="ScenarioNet database directory")
    parser.add_argument(
        "--split",
        choices=["training", "test"],
        default="training",
        help="Select which sampling interval to use. This only matters if training/test config settings differ.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    require_scenarionet()

    with quiet_native_startup_noise():
        from scenestreamer.dataset.dataset import SceneStreamerDataset

    cfg = load_yaml_config(args.config)
    apply_overrides(cfg, args.set or [])

    cfg.DATA.TRAINING_DATA_DIR = args.dataset_dir
    cfg.DATA.TEST_DATA_DIR = args.dataset_dir
    cfg.DATA.USE_CACHE = True

    ds = SceneStreamerDataset(cfg, args.split)

    total_scenarios = len(ds)
    target_scenarios = total_scenarios if args.limit is None else min(args.limit, total_scenarios)
    selected_interval = cfg.DATA.SAMPLE_INTERVAL_TRAINING if args.split == "training" else cfg.DATA.SAMPLE_INTERVAL_TEST

    print(f"[preprocess] Dataset directory: {args.dataset_dir}")
    print(f"[preprocess] Split: {args.split} (sample interval={selected_interval})")
    if cfg.DATA.SAMPLE_INTERVAL_TRAINING == cfg.DATA.SAMPLE_INTERVAL_TEST:
        print(
            "[preprocess] Note: with the current config, training/test use the same sampling interval, "
            "so --split will not change which scenarios are processed."
        )
    else:
        print(
            "[preprocess] Note: --split selects which sampling interval to use "
            f"(training={cfg.DATA.SAMPLE_INTERVAL_TRAINING}, test={cfg.DATA.SAMPLE_INTERVAL_TEST})."
        )
    print(f"[preprocess] Target: {target_scenarios}/{total_scenarios} scenario(s)")

    start_time = time.time()
    processed_scenarios = 0

    for i in tqdm(range(target_scenarios), desc="Preprocessing", unit="scenario"):
        _ = ds[i]
        processed_scenarios = i + 1

    elapsed_sec = time.time() - start_time
    cache_dir = pathlib.Path(args.dataset_dir) / "cache"
    cache_files = len(list(cache_dir.glob("*"))) if cache_dir.is_dir() else 0

    out_dir = pathlib.Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"preprocess-{args.split}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "status": "ok",
                "dataset_dir": args.dataset_dir,
                "split": args.split,
                "limit": args.limit,
                "processed_scenarios": processed_scenarios,
                "dataset_size": total_scenarios,
                "elapsed_sec": elapsed_sec,
                "cache_dir": str(cache_dir),
                "cache_files": cache_files,
            },
            f,
            indent=2,
        )

    print(f"[preprocess] Finished. Preprocessed {processed_scenarios} scenario(s) in {elapsed_sec:.1f}s.")
    print(f"[preprocess] Cache directory: {cache_dir} ({cache_files} file(s))")
    print(f"[preprocess] Metrics: {run_dir / 'metrics.json'}")
    print(f"[preprocess] Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()

