from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import shutil
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from scenestreamer.paper import print_console_json
from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.infer.infinite import generate_scenestreamer_motion
from scenestreamer.utils import utils


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_densify_demo(
    *,
    pl_model,
    dataset_dir: str,
    split: str,
    scenario_index: int,
    max_agents: int,
    force_no_end: bool,
    artifacts_dir: str,
    run_id: str | None,
    seed: int,
) -> pathlib.Path:
    _seed_everything(seed)

    config = pl_model.config
    config.DATA.TRAINING_DATA_DIR = dataset_dir
    config.DATA.TEST_DATA_DIR = dataset_dir
    config.DATA.USE_CACHE = True
    config.PREPROCESSING.keep_all_data = True

    print(f"[densify] Device: {pl_model.device}")
    print(f"[densify] Loading dataset from {dataset_dir} (split='{split}')")

    progress = tqdm(total=4, desc="Densify demo", unit="stage")
    try:
        progress.set_postfix_str("load dataset")
        ds = SceneStreamerDataset(config, split)
        progress.update(1)
        print(f"[densify] Dataset size: {len(ds)} scenario(s)")

        progress.set_postfix_str(f"load scenario {scenario_index}")
        raw = ds[scenario_index]
        progress.update(1)

        device = pl_model.device
        batched = utils.batch_data(utils.numpy_to_torch(raw, device=device))

        # Densify + motion rollout. `force_add=True` disables the "end of agent states" token.
        progress.set_postfix_str("generate motion")
        out = generate_scenestreamer_motion(
            data_dict=batched,
            model=pl_model.model,
            force_add=force_no_end,
            max_agents=max_agents,
            num_decode_steps=19,
        )
        progress.update(1)

        # Save a lightweight artifact; visualization can be added later.
        pred_pos = out.get("decoder/reconstructed_position")
        pred_valid = out.get("decoder/reconstructed_valid_mask")
        if pred_pos is None or pred_valid is None:
            raise ValueError("Model output missing decoder/reconstructed_position or decoder/reconstructed_valid_mask")

        pred_pos_np = pred_pos.detach().cpu().numpy()
        pred_valid_np = pred_valid.detach().cpu().numpy().astype(bool)

        base = pathlib.Path(artifacts_dir)
        base.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = f"densify-{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        out_dir = base / run_id
        if out_dir.exists():
            print(f"[densify] Overwriting existing artifacts at {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save a small summary to avoid huge artifacts by default.
        num_agents_seen_anytime = int(pred_valid_np.any(axis=1).any(axis=0).sum()) if pred_valid_np.ndim == 3 else None
        num_agents_final = int(pred_valid_np[:, -1].any(axis=0).sum()) if pred_valid_np.ndim == 3 else None
        summary: dict[str, Any] = {
            "scenario_id": raw.get("scenario_id", None),
            "pred_shape": list(pred_pos_np.shape),
            "num_agents_final": num_agents_final,
            "num_agents_seen_anytime": num_agents_seen_anytime,
            "max_agents_target": max_agents,
            "force_no_end": force_no_end,
            "seed": seed,
        }

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        progress.set_postfix_str("write artifacts")
        progress.update(1)
        print_console_json("densify", "Summary", summary)
        print(f"[densify] Wrote summary to {out_dir / 'metrics.json'}")
        return out_dir
    finally:
        progress.close()
