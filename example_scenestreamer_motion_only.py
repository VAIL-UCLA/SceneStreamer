"""
Example: Generate motion trajectories for existing agents.

This example demonstrates how to:
1. Load a pretrained SceneStreamer model from HuggingFace
2. Load a scenario dataset
3. Generate motion predictions for agents in the scenario
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tqdm

from scenestreamer._startup import configure_startup_noise_filters, quiet_native_startup_noise
from scenestreamer.model_hub import DEFAULT_HF_REPO

configure_startup_noise_filters()

DEFAULT_HF_FILE = "scenestreamer-base-large.ckpt"

with quiet_native_startup_noise():
    from scenestreamer.dataset.dataset import SceneStreamerDataset
    from scenestreamer.gradio_ui.artifact import render_asset
    from scenestreamer.infer.motion import generate_motion
    from scenestreamer.utils import utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SceneStreamer motion-only predictions from a HuggingFace checkpoint."
    )
    parser.add_argument("--dataset-dir", default="data/20scenarios")
    parser.add_argument("--split", choices=["training", "test"], default="test")
    parser.add_argument("--num-scenarios", type=int, default=1, help="Use <= 0 to run the full split")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--ckpt", default=None, help="Optional local checkpoint path; overrides HuggingFace defaults")
    parser.add_argument("--device", default="auto", help="torch device string: auto, cuda, mps, or cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.ckpt:
        pl_model = utils.get_model(checkpoint_path=args.ckpt, device=args.device)
    else:
        pl_model = utils.get_model(
            huggingface_repo=args.hf_repo,
            huggingface_file=args.hf_file,
            device=args.device,
        )

    device = pl_model.device
    config = pl_model.config
    config.DATA.TRAINING_DATA_DIR = args.dataset_dir
    config.DATA.TEST_DATA_DIR = args.dataset_dir
    config.DATA.USE_CACHE = True

    test_dataset = SceneStreamerDataset(config, args.split)
    num_scenarios = None if args.num_scenarios <= 0 else args.num_scenarios
    total = len(test_dataset) if num_scenarios is None else min(num_scenarios, len(test_dataset))
    print(f"[SceneStreamer] Device: {device}")
    print(f"[SceneStreamer] Model: {args.hf_repo}/{args.hf_file}" if not args.ckpt else f"[SceneStreamer] Model: {args.ckpt}")
    print(f"[SceneStreamer] Dataset: split={args.split!r} size={len(test_dataset)}. Running {total} scenario(s).")
    print("[SceneStreamer] Next: use --num-scenarios 0 to process the full split.")

    iterable = test_dataset if num_scenarios is None else itertools.islice(test_dataset, num_scenarios)

    for count, data_dict in enumerate(tqdm.tqdm(iterable, total=total)):
        batched_data_dict = utils.batch_data(utils.numpy_to_torch(data_dict, device=device))

        output_data_dict = generate_motion(
            data_dict=batched_data_dict,
            model=pl_model.model,
            autoregressive_start_step=0,
            teacher_forcing_sdc=True,
            num_decode_steps=19,
        )

        scenario_id = data_dict.get("scenario_id", count)
        print(f"[SceneStreamer] Generated motion for scenario_id={scenario_id!r}.")
        print("[SceneStreamer] Outputs are in the returned dict (e.g. keys like 'decoder/reconstructed_position').")

        out_dir = Path("artifacts/example_motion_only/latest")
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_pos = output_data_dict["decoder/reconstructed_position"][0].detach().cpu().numpy()
        pred_valid = output_data_dict["decoder/reconstructed_valid_mask"][0].detach().cpu().numpy().astype(bool)
        pred_heading = output_data_dict["decoder/reconstructed_heading"][0].detach().cpu().numpy()

        np.savez_compressed(
            out_dir / "output.npz",
            scenario_id=np.array(str(scenario_id)),
            pred_pos=pred_pos,
            pred_valid=pred_valid,
            pred_heading=pred_heading,
        )

        raw_map_feature = batched_data_dict["raw/map_feature"][0].detach().cpu().numpy()
        map_feature_valid_mask = batched_data_dict["encoder/map_feature_valid_mask"][0].detach().cpu().numpy().astype(bool)
        current_agent_shape = batched_data_dict["decoder/current_agent_shape"][0].detach().cpu().numpy()

        if "decoder/sdc_index" in batched_data_dict:
            sdc_index = int(batched_data_dict["decoder/sdc_index"][0].detach().cpu().item())
        else:
            sdc_index = 0
        if "decoder/object_of_interest_id" in batched_data_dict:
            ooi = batched_data_dict["decoder/object_of_interest_id"][0].detach().cpu().numpy()
        else:
            ooi = np.array([], dtype=np.int64)

        asset_path = out_dir / "asset.npz"
        fig_path = out_dir / "figure.png"

        traffic_light_feature = batched_data_dict["encoder/traffic_light_feature"][0].detach().cpu().numpy()
        traffic_light_position = batched_data_dict["encoder/traffic_light_position"][0].detach().cpu().numpy()
        traffic_light_valid_mask = batched_data_dict["encoder/traffic_light_valid_mask"][0].detach().cpu().numpy().astype(
            bool
        )

        np.savez_compressed(
            asset_path,
            scenario_id=np.array(str(scenario_id)),
            **{
                "raw/map_feature": raw_map_feature,
                "encoder/map_feature_valid_mask": map_feature_valid_mask,
                "encoder/traffic_light_feature": traffic_light_feature,
                "encoder/traffic_light_position": traffic_light_position,
                "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
                "decoder/reconstructed_position": pred_pos,
                "decoder/reconstructed_heading": pred_heading,
                "decoder/reconstructed_valid_mask": pred_valid,
                "decoder/current_agent_shape": current_agent_shape,
                "decoder/sdc_index": np.array(sdc_index, dtype=np.int64),
                "decoder/object_of_interest_id": ooi,
            },
        )

        render_asset(asset_path, fig_path, verbose=False)

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "scenario_id": str(scenario_id),
                    "device": str(device),
                    "model": args.ckpt or f"{args.hf_repo}/{args.hf_file}",
                    "asset_path": str(asset_path),
                    "figure_path": str(fig_path),
                    "rerender_command": f"uv run python scripts/plot_artifact.py --asset {asset_path} --out {fig_path}",
                    "gradio_demo_command": "uv run python scripts/demo_gradio.py",
                },
                f,
                indent=2,
            )

        print(f"[SceneStreamer] Saved artifacts to: {out_dir}")
        print(f"[SceneStreamer] Saved plot asset: {asset_path}")
        print(f"[SceneStreamer] Auto-rendered figure: {fig_path}")
        print("uv run python scripts/demo_gradio.py")


if __name__ == "__main__":
    main()
