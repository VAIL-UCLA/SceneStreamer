from __future__ import annotations

import datetime as dt
import json
import math
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
from scenestreamer.infer.initial_state import convert_initial_states_as_motion_data, generate_initial_state
from scenestreamer.infer.motion import generate_motion
from scenestreamer.utils import utils


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _nanmean(x: np.ndarray) -> float:
    if np.isnan(x).all():
        return float("nan")
    return float(np.nanmean(x))


def _pick_ooi_index(sample: dict[str, Any]) -> int | None:
    # Common keys across variants; keep this conservative.
    for key in ("decoder/object_of_interest_id", "decoder/labeled_agent_id"):
        if key in sample:
            v = sample[key]
            if isinstance(v, np.ndarray):
                flat = v.reshape(-1)
                flat = flat[flat != -1]
                if flat.size:
                    return int(flat[0])
            elif np.isscalar(v):
                return int(v)
    return None


@dataclass
class MotionMetrics:
    ade_avg: float
    ade_min: float
    fde_avg: float
    fde_min: float
    add: float
    fdd: float


def _compute_metrics(
    *,
    gt_pos: np.ndarray,  # (T, N, 2)
    gt_valid: np.ndarray,  # (T, N)
    pred_pos: np.ndarray,  # (K, T, N, 2)
    pred_valid: np.ndarray,  # (K, T, N)
    agent_mask: np.ndarray,  # (N,)
) -> MotionMetrics:
    # Evaluate on the standard Waymo horizon: future steps [11:91] => 80 steps
    t0 = 11 if gt_pos.shape[0] >= 91 else 0
    t1 = min(gt_pos.shape[0], pred_pos.shape[1])

    gt_pos = gt_pos[t0:t1]
    gt_valid = gt_valid[t0:t1]
    pred_pos = pred_pos[:, t0:t1]
    pred_valid = pred_valid[:, t0:t1]

    T = gt_pos.shape[0]
    N = gt_pos.shape[1]
    K = pred_pos.shape[0]

    # Mask invalids.
    valid = (gt_valid[None] & pred_valid)  # (K, T, N)
    valid &= agent_mask[None, None, :]

    # L2 errors (K, T, N)
    err = np.linalg.norm(pred_pos - gt_pos[None], axis=-1)
    err[~valid] = np.nan

    # ADE per (K, N)
    ade_kn = np.nanmean(err, axis=1)
    # FDE per (K, N): last valid timestep per (K, N)
    fde_kn = np.full((K, N), np.nan, dtype=np.float64)
    for k in range(K):
        for n in range(N):
            vv = valid[k, :, n]
            if not vv.any():
                continue
            last = int(np.where(vv)[0][-1])
            fde_kn[k, n] = err[k, last, n]

    # Average over agents first, then modes.
    ade_k = np.nanmean(ade_kn, axis=1)
    fde_k = np.nanmean(fde_kn, axis=1)

    ade_avg = _nanmean(ade_k)
    fde_avg = _nanmean(fde_k)

    ade_min = _nanmean(np.nanmin(ade_kn, axis=0))
    fde_min = _nanmean(np.nanmin(fde_kn, axis=0))

    # FDD: max pairwise distance between mode final positions (per agent).
    final_pos = np.full((K, N, 2), np.nan, dtype=np.float64)
    for k in range(K):
        for n in range(N):
            vv = valid[k, :, n]
            if not vv.any():
                continue
            last = int(np.where(vv)[0][-1])
            final_pos[k, n] = pred_pos[k, last, n]

    fdd_n = np.full((N,), np.nan, dtype=np.float64)
    for n in range(N):
        if not agent_mask[n]:
            continue
        pts = final_pos[:, n, :]
        if np.isnan(pts).any():
            # require all modes to be valid for this agent for diversity metrics
            continue
        # max pairwise L2
        dmax = 0.0
        for i in range(K):
            for j in range(K):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                dmax = max(dmax, d)
        fdd_n[n] = dmax

    # ADD: for each (t,n), max pairwise distance across modes at that timestep; then mean over t per agent.
    add_n = np.full((N,), np.nan, dtype=np.float64)
    for n in range(N):
        if not agent_mask[n]:
            continue
        per_t = []
        for t in range(T):
            vv = valid[:, t, n]
            if not vv.all():
                continue
            pts = pred_pos[:, t, n, :]
            dmax = 0.0
            for i in range(K):
                for j in range(K):
                    d = float(np.linalg.norm(pts[i] - pts[j]))
                    dmax = max(dmax, d)
            per_t.append(dmax)
        if per_t:
            add_n[n] = float(np.mean(per_t))

    return MotionMetrics(
        ade_avg=float(ade_avg),
        ade_min=float(ade_min),
        fde_avg=float(fde_avg),
        fde_min=float(fde_min),
        add=_nanmean(add_n),
        fdd=_nanmean(fdd_n),
    )


def run_table2_motion(
    *,
    pl_model,
    dataset_dir: str,
    split: str,
    mode: str,
    num_modes: int,
    limit: int | None,
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

    ds = SceneStreamerDataset(config, split)
    target_scenarios = len(ds) if limit is None else min(limit, len(ds))

    device = pl_model.device

    print(f"[table2] Device: {device}")
    print(
        f"[table2] Dataset: split='{split}' size={len(ds)}. "
        f"Evaluating {target_scenarios} scenario(s) with mode='{mode}' and num_modes={num_modes}."
    )

    all_rows: list[dict[str, Any]] = []
    scenestreamer_generator = None

    for idx in tqdm(range(target_scenarios), desc="Table 2", unit="scenario"):
        raw = ds[idx]
        batched = utils.batch_data(utils.numpy_to_torch(raw, device=device))
        expanded = utils.expand_for_modes(batched, num_modes=num_modes)

        if pl_model.config.MODEL.NAME == "scenestreamer":
            if scenestreamer_generator is None:
                from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator

                scenestreamer_generator = SceneStreamerGenerator(
                    model=pl_model.model,
                    device=device,
                )
            scenestreamer_generator.reset(new_data_dict=expanded)
            if mode == "motion":
                out = scenestreamer_generator.generate_scenestreamer_motion(progress_bar=False)
            elif mode == "full":
                out = scenestreamer_generator.generate_scenestreamer_initial_state_and_motion(progress_bar=False)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            if mode == "motion":
                out = generate_motion(
                    data_dict=expanded,
                    model=pl_model.model,
                    autoregressive_start_step=0,
                    teacher_forcing_sdc=True,
                    num_decode_steps=19,
                )
            elif mode == "full":
                densified, _ = generate_initial_state(data_dict=expanded, model=pl_model.model)
                densified_motion_input = convert_initial_states_as_motion_data(densified)
                out = generate_motion(
                    data_dict=densified_motion_input,
                    model=pl_model.model,
                    autoregressive_start_step=0,
                    teacher_forcing_sdc=False,
                    num_decode_steps=19,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Prefer reconstructed outputs; fall back to agent_position if needed.
        pred_pos = out.get("decoder/reconstructed_position")
        pred_valid = out.get("decoder/reconstructed_valid_mask")
        if pred_pos is None or pred_valid is None:
            raise ValueError("Model output missing decoder/reconstructed_position or decoder/reconstructed_valid_mask")

        gt_pos = expanded.get("decoder/agent_position")
        gt_valid = expanded.get("decoder/agent_valid_mask")
        if gt_pos is None or gt_valid is None:
            raise ValueError("Input missing decoder/agent_position or decoder/agent_valid_mask")

        # Convert to numpy
        pred_pos_np = pred_pos.detach().cpu().numpy()
        pred_valid_np = pred_valid.detach().cpu().numpy().astype(bool)
        gt_pos_np = gt_pos[0].detach().cpu().numpy()[..., :2] if gt_pos.ndim == 4 else gt_pos.detach().cpu().numpy()
        gt_valid_np = gt_valid[0].detach().cpu().numpy().astype(bool) if gt_valid.ndim == 3 else gt_valid.detach().cpu().numpy().astype(bool)

        # Align shapes to (K,T,N,2) and (K,T,N)
        if pred_pos_np.ndim != 4:
            raise ValueError(f"Unexpected pred_pos shape: {pred_pos_np.shape}")
        K, T, N, _ = pred_pos_np.shape
        if pred_valid_np.shape != (K, T, N):
            raise ValueError(f"Unexpected pred_valid shape: {pred_valid_np.shape} vs {(K, T, N)}")

        if gt_pos_np.ndim != 3:
            raise ValueError(f"Unexpected gt_pos shape: {gt_pos_np.shape}")
        if gt_valid_np.shape != gt_pos_np.shape[:2]:
            raise ValueError(f"Unexpected gt_valid shape: {gt_valid_np.shape} vs {gt_pos_np.shape[:2]}")

        # Some generation paths may produce a different number of agents than the original sample.
        # Evaluate on the shared prefix so the motion benchmark remains defined.
        eval_agents = min(pred_pos_np.shape[2], gt_pos_np.shape[1])
        pred_pos_np = pred_pos_np[:, :, :eval_agents]
        pred_valid_np = pred_valid_np[:, :, :eval_agents]
        gt_pos_np = gt_pos_np[:, :eval_agents]
        gt_valid_np = gt_valid_np[:, :eval_agents]

        # Per-agent masks
        all_agent_mask = np.ones((eval_agents,), dtype=bool)
        ooi_idx = _pick_ooi_index(raw)
        ooi_mask = np.zeros((eval_agents,), dtype=bool)
        if ooi_idx is not None and 0 <= ooi_idx < eval_agents:
            ooi_mask[ooi_idx] = True

        metrics_all = _compute_metrics(
            gt_pos=gt_pos_np,
            gt_valid=gt_valid_np,
            pred_pos=pred_pos_np,
            pred_valid=pred_valid_np,
            agent_mask=all_agent_mask,
        )
        metrics_ooi = _compute_metrics(
            gt_pos=gt_pos_np,
            gt_valid=gt_valid_np,
            pred_pos=pred_pos_np,
            pred_valid=pred_valid_np,
            agent_mask=ooi_mask if ooi_mask.any() else all_agent_mask,
        )

        row = {
            "index": idx,
            "scenario_id": raw.get("scenario_id", None),
            "all": metrics_all.__dict__,
            "ooi": metrics_ooi.__dict__,
        }
        all_rows.append(row)

    def _avg(key: str, group: str) -> float:
        vals = [r[group][key] for r in all_rows if not math.isnan(r[group][key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "all": {k: _avg(k, "all") for k in MotionMetrics.__annotations__.keys()},
        "ooi": {k: _avg(k, "ooi") for k in MotionMetrics.__annotations__.keys()},
    }

    base = pathlib.Path(artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = f"table2-{mode}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    out_dir = base / run_id
    if out_dir.exists():
        print(f"[table2] Overwriting existing artifacts at {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "table": "table2",
                "dataset_dir": dataset_dir,
                "split": split,
                "mode": mode,
                "num_modes": num_modes,
                "limit": limit,
                "seed": seed,
                "summary": summary,
                "rows": all_rows,
            },
            f,
            indent=2,
        )

    print_console_json("table2", "Summary", summary)
    print(f"[table2] Wrote metrics to {out_dir / 'metrics.json'}")
    return out_dir

