from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from scenestreamer.gradio_ui.demo_app import DEFAULT_HF_FILE, DEFAULT_HF_REPO, build_demo


def _build_space_demo() -> gr.Blocks:
    dataset_dir = os.environ.get("SCENESTREAMER_DATASET_DIR", "data/20scenarios")
    hf_repo = os.environ.get("SCENESTREAMER_HF_REPO", DEFAULT_HF_REPO)
    hf_file = os.environ.get("SCENESTREAMER_HF_FILE", DEFAULT_HF_FILE)
    ckpt = os.environ.get("SCENESTREAMER_CKPT") or None
    device = os.environ.get("SCENESTREAMER_DEVICE", "cpu")

    if not Path(dataset_dir).exists():
        with gr.Blocks(title="SceneStreamer Space Setup") as demo:
            gr.Markdown("## SceneStreamer Space Setup Required")
            gr.Markdown(
                "This Space needs a local ScenarioNet dataset directory before the interactive demo can start.\n\n"
                f"Current `SCENESTREAMER_DATASET_DIR`: `{dataset_dir}`"
            )
            gr.Markdown(
                "Set Space variables or attach storage, then restart the Space:\n"
                "- `SCENESTREAMER_DATASET_DIR`\n"
                "- `SCENESTREAMER_HF_REPO` (optional)\n"
                "- `SCENESTREAMER_HF_FILE` (optional)\n"
                "- `SCENESTREAMER_CKPT` (optional local checkpoint)\n"
                "- `SCENESTREAMER_DEVICE` (default `cpu`)"
            )
        return demo

    return build_demo(
        dataset_dir=dataset_dir,
        hf_repo=hf_repo,
        hf_file=hf_file,
        ckpt=ckpt,
        device=device,
    )


demo = _build_space_demo()


if __name__ == "__main__":
    demo.launch()

