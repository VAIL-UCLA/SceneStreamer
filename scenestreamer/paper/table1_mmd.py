from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from scenestreamer.paper import print_console_json
from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.eval.test_trafficgen_eval import TrafficGenEvaluator
from scenestreamer.infer.initial_state import generate_initial_state
from scenestreamer.utils import utils


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_jsonable(v: Any):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.detach().cpu().item()
        return v.detach().cpu().tolist()
    return v


@dataclass
class _MetricAgg:
    sums: dict[str, float]
    counts: dict[str, int]

    @classmethod
    def create(cls) -> "_MetricAgg":
        return cls(sums={}, counts={})

    def add(self, k: str, v: Any) -> None:
        vv = _to_jsonable(v)
        if isinstance(vv, list):
            return
        if vv is None:
            return
        self.sums[k] = self.sums.get(k, 0.0) + float(vv)
        self.counts[k] = self.counts.get(k, 0) + 1

    def mean(self) -> dict[str, float]:
        out = {}
        for k, s in self.sums.items():
            c = self.counts.get(k, 0)
            if c:
                out[k] = s / c
        return out


def run_table1_mmd(
    *,
    pl_model,
    dataset_dir: str,
    split: str,
    limit: int | None,
    artifacts_dir: str,
    run_id: str | None,
    seed: int,
) -> pathlib.Path:
    _seed_everything(seed)

    config = pl_model.config
    config.DATA.TRAINING_DATA_DIR = dataset_dir
    config.DATA.TEST_DATA_DIR = dataset_dir
    config.DATA.SD_PASSTHROUGH = True
    config.DATA.USE_CACHE = True
    config.PREPROCESSING.keep_all_data = True

    # Required by TrafficGenEvaluator (kept for backward-compat with existing evaluator code).
    config.EVALUATION.USE_TG_AS_GT = 1111

    ds = SceneStreamerDataset(config, split)
    target_scenarios = len(ds) if limit is None else min(limit, len(ds))

    print(f"[table1] Device: {pl_model.device}")
    print(f"[table1] Dataset: split='{split}' size={len(ds)}. Evaluating {target_scenarios} scenario(s).")

    evaluator = TrafficGenEvaluator(config)
    agg = _MetricAgg.create()

    device = pl_model.device

    for idx in tqdm(range(target_scenarios), desc="Table 1", unit="scenario"):
        raw = ds[idx]
        batched = utils.batch_data(utils.numpy_to_torch(raw, device=device))

        # Generate initial agent states (TrafficGen-style).
        densified, _ = generate_initial_state(
            data_dict=batched,
            model=pl_model.model,
            force_add=False,
        )
        if "raw_scenario_description" in raw:
            densified["raw_scenario_description"] = [raw["raw_scenario_description"]]

        def log_func(name: str, value: Any) -> None:
            agg.add(name, value)

        evaluator.validation_step(densified, stat={}, log_func=log_func)

    metrics = agg.mean()

    base = pathlib.Path(artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = f"table1-{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    out_dir = base / run_id
    if out_dir.exists():
        print(f"[table1] Overwriting existing artifacts at {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "table": "table1",
                "dataset_dir": dataset_dir,
                "split": split,
                "limit": limit,
                "seed": seed,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print_console_json("table1", "Metrics", metrics)
    print(f"[table1] Wrote metrics to {out_dir / 'metrics.json'}")
    return out_dir

