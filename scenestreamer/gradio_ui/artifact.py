from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np


def npz_to_dict(npz: np.lib.npyio.NpzFile) -> dict:
    out: dict = {}
    for k in npz.files:
        v = npz[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        out[k] = v
    return out


def load_asset(asset_path: str | Path) -> dict:
    with np.load(asset_path, allow_pickle=False) as npz:
        return npz_to_dict(npz)


def render_asset(asset_path: str | Path, out_path: str | Path, verbose: bool = False) -> Path:
    from scenestreamer.gradio_ui.plot import plot_pred

    asset_path = Path(asset_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dict = load_asset(asset_path)

    if verbose:
        plot_pred(data_dict, show=False, save_path=str(out_path))
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_pred(data_dict, show=False, save_path=str(out_path))
    return out_path

