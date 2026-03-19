from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch

from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.gradio_ui.plot import plot_gt, plot_pred
from scenestreamer.infer.initial_state import convert_initial_states_as_motion_data, generate_initial_state
from scenestreamer.infer.motion import generate_motion
from scenestreamer.model_hub import DEFAULT_HF_REPO
from scenestreamer.utils import utils

DEFAULT_HF_FILE = "scenestreamer-full-large.ckpt"


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_demo(
    *,
    dataset_dir: str = "data/20scenarios",
    hf_repo: str = DEFAULT_HF_REPO,
    hf_file: str = DEFAULT_HF_FILE,
    ckpt: str | None = None,
    device: str = "auto",
) -> gr.Blocks:
    device_obj = choose_device(device)
    if ckpt:
        pl_model = utils.get_model(checkpoint_path=ckpt, device=device_obj)
    else:
        pl_model = utils.get_model(huggingface_repo=hf_repo, huggingface_file=hf_file, device=device_obj)

    config = pl_model.config
    config.DATA.TRAINING_DATA_DIR = dataset_dir
    config.DATA.TEST_DATA_DIR = dataset_dir
    config.DATA.USE_CACHE = True

    dataset = SceneStreamerDataset(config, "test")
    scenestreamer_generator = None

    def load_ground_truth(scenario_index: int):
        raw = dataset[int(scenario_index)]
        gt_img = plot_gt(raw)
        scenario_id = raw.get("scenario_id", scenario_index)
        status = (
            f"Loaded scenario `{scenario_id}` on `{pl_model.device}`.\n\n"
            "Ground truth is fixed. Click `Generate` to refresh only the prediction panel."
        )
        return status, gt_img, None

    def run_demo(scenario_index: int, mode: str):
        nonlocal scenestreamer_generator
        raw = dataset[int(scenario_index)]

        batched = utils.batch_data(utils.numpy_to_torch(raw, device=pl_model.device))
        if pl_model.config.MODEL.NAME == "scenestreamer":
            if scenestreamer_generator is None:
                from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator

                scenestreamer_generator = SceneStreamerGenerator(
                    model=pl_model.model,
                    device=pl_model.device,
                )
            scenestreamer_generator.reset(new_data_dict=batched)
            if mode == "motion_only":
                output = scenestreamer_generator.generate_scenestreamer_motion(
                    progress_bar=False,
                    teacher_forcing_sdc=True,
                )
            else:
                output = scenestreamer_generator.generate_scenestreamer_initial_state_and_motion(progress_bar=False)
        else:
            if mode == "motion_only":
                output = generate_motion(
                    data_dict=batched,
                    model=pl_model.model,
                    autoregressive_start_step=0,
                    teacher_forcing_sdc=True,
                    num_decode_steps=19,
                )
            else:
                densified, _ = generate_initial_state(data_dict=batched, model=pl_model.model)
                densified_motion_input = convert_initial_states_as_motion_data(densified)
                output = generate_motion(
                    data_dict=densified_motion_input,
                    model=pl_model.model,
                    autoregressive_start_step=0,
                    teacher_forcing_sdc=False,
                    num_decode_steps=19,
                )

        pred = utils.unbatch_data(utils.torch_to_numpy(output))
        pred_img = plot_pred(pred)
        scenario_id = raw.get("scenario_id", scenario_index)
        status = (
            f"Generated prediction for scenario `{scenario_id}` on `{pl_model.device}`.\n\n"
            f"Mode: `{mode}`\n"
            f"Dataset size: `{len(dataset)}`\n"
            "Ground truth stays fixed while the prediction is regenerated."
        )
        return status, pred_img

    max_index = max(0, len(dataset) - 1)

    with gr.Blocks(title="SceneStreamer Interactive Demo") as demo:
        gr.Markdown("## SceneStreamer Interactive Demo")
        gr.Markdown("Pick a scenario, choose a generation mode, and inspect ground-truth vs generated results in the browser.")
        gr.Markdown(f"Loaded dataset: `{dataset_dir}` on device `{pl_model.device}`")

        with gr.Row():
            scenario_index = gr.Slider(
                minimum=0,
                maximum=max_index,
                value=0,
                step=1,
                label="Scenario Index",
            )
            mode = gr.Radio(
                choices=[
                    ("Motion Only", "motion_only"),
                    ("Densified Agents", "densified_agents"),
                ],
                value="motion_only",
                label="Generation Mode",
            )
            run_button = gr.Button("Generate")

        status = gr.Markdown()
        with gr.Row():
            gt_image = gr.Image(label="Ground Truth")
            pred_image = gr.Image(label="Generated Prediction")

        scenario_index.change(load_ground_truth, inputs=[scenario_index], outputs=[status, gt_image, pred_image])
        run_button.click(run_demo, inputs=[scenario_index, mode], outputs=[status, pred_image])
        demo.load(load_ground_truth, inputs=[scenario_index], outputs=[status, gt_image, pred_image])

    return demo

