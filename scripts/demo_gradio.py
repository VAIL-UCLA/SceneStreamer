from __future__ import annotations

import argparse

try:
    import gradio as gr
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency `gradio`.\n"
        "You are likely using system/anaconda Python instead of the project's uv environment.\n"
        "Run: `uv run python scripts/demo_gradio.py`\n"
        "Or activate `.venv` first."
    ) from e

from scenestreamer._startup import configure_startup_noise_filters, quiet_native_startup_noise

configure_startup_noise_filters()

with quiet_native_startup_noise():
    from scenestreamer.gradio_ui.demo_app import DEFAULT_HF_FILE, DEFAULT_HF_REPO, build_demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="uv run python scripts/demo_gradio.py",
        description="Launch an interactive Gradio demo for SceneStreamer generation.",
    )
    parser.add_argument("--dataset-dir", default="data/20scenarios")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    parser.add_argument("--ckpt", default=None, help="Optional local checkpoint path; overrides HuggingFace defaults")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    demo = build_demo(
        dataset_dir=args.dataset_dir,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        ckpt=args.ckpt,
        device=args.device,
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

