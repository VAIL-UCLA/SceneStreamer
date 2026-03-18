from __future__ import annotations

import pathlib

DEFAULT_HF_REPO = "pengzhenghao97/scenestreamer"

MODEL_CONFIG_TO_HF_FILE = {
    "scenestreamer-base-small": "scenestreamer-base-small.ckpt",
    "scenestreamer-base-large": "scenestreamer-base-large.ckpt",
    "scenestreamer-base-xl": "scenestreamer-base-xl.ckpt",
    "scenestreamer-full-small": "scenestreamer-full-small.ckpt",
    "scenestreamer-full-large": "scenestreamer-full-large.ckpt",
    "scenestreamer-full-large-nors": "scenestreamer-full-large-nors.ckpt",
    "scenestreamer-full-xl": "scenestreamer-full-xl.ckpt",
}


def get_default_hf_file(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    model_name = model_name.removesuffix(".yaml")
    return MODEL_CONFIG_TO_HF_FILE.get(model_name)


HF_FILE_TO_MODEL_CONFIG = {v: k for k, v in MODEL_CONFIG_TO_HF_FILE.items()}


def get_default_config_name_for_checkpoint(checkpoint_name: str | None) -> str | None:
    if checkpoint_name is None:
        return None
    checkpoint_name = pathlib.Path(checkpoint_name).name
    model_name = HF_FILE_TO_MODEL_CONFIG.get(checkpoint_name)
    if model_name is None:
        return None
    return f"{model_name}.yaml"
