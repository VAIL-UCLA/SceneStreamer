import argparse
import functools
import os
import pathlib
import pickle

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf

from scenestreamer.dataset.preprocess_action_label import TurnAction
from scenestreamer.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils
from scenestreamer.gradio_ui.plot_video import plot_gt_video, plot_pred_video

os.environ['GRADIO_TEMP_DIR'] = str(REPO_ROOT / "gradio_tmp")

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", help="Enable sharing")
parser.add_argument("--default_ckpt", "--default_model", type=str, default="data/20scenarios")
parser.add_argument("--default_data", type=str, default="")
args = parser.parse_args()

default_config = OmegaConf.load(REPO_ROOT / "cfgs/motion_default.yaml")

OmegaConf.set_struct(default_config, False)
default_config.MODEL.D_MODEL = 32
default_config.MODEL.NUM_DECODER_LAYERS = 1
default_config.MODEL.NUM_ATTN_LAYERS = 1
default_config.ACTION_LABEL.USE_SAFETY_LABEL = True
default_config.ACTION_LABEL.USE_ACTION_LABEL = True
default_config.ROOT_DIR = REPO_ROOT
OmegaConf.set_struct(default_config, True)

DEFAULT_DATA_PATH = args.default_data or "data/20scenarios"
DEFAULT_MODEL = args.default_ckpt or None

NUM_OF_MODES = 6
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LENGTH = 1000


class State:
    model = None
    model_path = None
    config: dict = default_config
    dataset_path: pathlib.Path = REPO_ROOT / DEFAULT_DATA_PATH

    scenario = None

    raw_data_files = None
    data_files = None

    raw_data_dict = None
    data_dict = None

    default_config: dict = default_config


state = State()


def ckpt_callback(ckpt_path):
    from scenestreamer.models.motionlm_lightning import MotionLMLightning

    msg = "Failed!"
    temperature = 1.0
    safe_agents = ""
    turn_agents = ""
    main_vis = None
    sampling_method = "topp"

    if ckpt_path.lower() == "debug":
        try:
            config = state.default_config
            OmegaConf.set_struct(config, False)
            config.MODEL.D_MODEL = 32
            config.MODEL.NUM_DECODER_LAYERS = 1
            config.MODEL.NUM_ATTN_LAYERS = 1
            config.ACTION_LABEL.USE_SAFETY_LABEL = True
            config.ACTION_LABEL.USE_ACTION_LABEL = True
            OmegaConf.set_struct(config, True)
            model = MotionLMLightning(config)
            model = model.to(device)
            msg = "DEBUG MODEL LOADED!"
            config = model.config
            temperature = config.SAMPLING.TEMPERATURE
            state.model = model
            state.config = config
            sampling_method = config.SAMPLING.SAMPLING_METHOD
        except Exception as e:
            print("Error: ", e)
            msg = "Failed to load DEBUG model!"

        return [msg, sampling_method, temperature, main_vis] + [""] * 7

    path = pathlib.Path(ckpt_path)
    path = REPO_ROOT / path

    print("Loading model from: ", path.absolute())
    if not path.exists():
        msg = "{} does not exist!".format(path)
        return [msg, sampling_method, temperature, main_vis] + [""] * 7

    try:
        model = utils.load_from_checkpoint(
            checkpoint_path=path, cls=MotionLMLightning, config=None, default_config=default_config, strict=False
        )
        model = model.to(device)
        msg = "Model loaded successfully!"
        config = model.config
        temperature = config.SAMPLING.TEMPERATURE
        state.model = model
        state.config = config
        sampling_method = config.SAMPLING.SAMPLING_METHOD
    except Exception as e:
        print("Error: ", e)
        msg = "Failed to load model!"

    return [msg, sampling_method, temperature, main_vis] + [""] * 7


def on_dataset_path_submit(path):

    print(state, type(state))

    FAILED_MSG = "Failed!"

    path = pathlib.Path(path)
    path = REPO_ROOT / path

    if not path.exists():
        return FAILED_MSG

    if not path.is_dir():
        return FAILED_MSG

    state.dataset_path = path
    print(state.dataset_path)

    files = os.listdir(path)
    files = [f for f in files if f.endswith(".pkl")]
    print("Files: ", files)

    if not hasattr(state, "count"):
        state.count = 0
    state.count += 1

    return [
        "Dataset with {} Scenarios Listed!".format(len(files)),
        gr.FileExplorer(
            file_count="single",
            root_dir=state.dataset_path,
            # root_dir=path,
            glob="**/*.pkl",
            scale=1
            # label="UPDATED={}".format(state.count),
            # interactive=True
        )
    ]


def on_data_file_name_search(search):
    return gr.FileExplorer(file_count="single", root_dir=state.dataset_path, glob=f"**/*{search}*.pkl", scale=1)


def on_data_file_select(file_path):
    if not file_path:
        return (None, ) + ("", ) * 7

    file_path = pathlib.Path(file_path)
    assert state.dataset_path is not None
    file_path = state.dataset_path / file_path

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    state.scenario = data
    scenario_data_dict = preprocess_scenario_description_for_motionlm(
        scenario=data, config=state.config, in_evaluation=True, keep_all_data=True
    )
    state.raw_data_dict = scenario_data_dict
    video_file = plot_gt_video(data)
    return (gr.Video(value=video_file.name, label=data["id"]), ) + ("", ) * 7


def on_generate_button_click(
    sampling_method, temperature, seed, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
    agents_turn_left, agents_turn_right, agents_turn_uturn, only_draw_gt
):

    # TODO: Seed is not respect!
    # TODO: Seed is not respect!
    # TODO: Seed is not respect!

    assert sampling_method in ["softmax", "topp"]

    if state.scenario is None:
        return (
            None,
            "Data is not loaded!",
        ) + (None, ) * 7

    model = state.model
    if model is None and not only_draw_gt:
        return (
            None,
            "Model is not loaded!",
        ) + (None, ) * 7

    data_dict = preprocess_scenario_description_for_motionlm(
        scenario=state.scenario, config=state.config, in_evaluation=True, keep_all_data=True
    )

    # ===== Overwrite the labels =====
    def _parse_agents(agents_str):
        is_raw = False
        if agents_str:
            if agents_str.startswith("[RAW]"):
                agents_str = agents_str[5:]
                is_raw = True
                if not agents_str:
                    return [], is_raw
            return [int(agent.strip()) for agent in agents_str.split(",")], is_raw
        else:
            is_raw = True
            return [], is_raw

    def _fill_label(data_dict, label_name, agents, label_value):
        if agents:
            label = data_dict['decoder/' + label_name]
            for aid in agents:
                assert 0 <= aid < label.shape[0], (aid, label.shape)
                label[aid] = label_value

    agents_safe_0, agents_safe_0_is_raw = _parse_agents(agents_safe_0)
    _fill_label(data_dict, 'label_safety', agents_safe_0, 0)
    agents_safe_1, agents_safe_1_is_raw = _parse_agents(agents_safe_1)
    _fill_label(data_dict, 'label_safety', agents_safe_1, 1)
    agents_turn_stop, agents_turn_stop_is_raw = _parse_agents(agents_turn_stop)
    _fill_label(data_dict, 'label_turning', agents_turn_stop, TurnAction.STOP)
    agents_turn_straight, agents_turn_straight_is_raw = _parse_agents(agents_turn_straight)
    _fill_label(data_dict, 'label_turning', agents_turn_straight, TurnAction.KEEP_STRAIGHT)
    agents_turn_left, agents_turn_left_is_raw = _parse_agents(agents_turn_left)
    _fill_label(data_dict, 'label_turning', agents_turn_left, TurnAction.TURN_LEFT)
    agents_turn_right, agents_turn_right_is_raw = _parse_agents(agents_turn_right)
    _fill_label(data_dict, 'label_turning', agents_turn_right, TurnAction.TURN_RIGHT)
    agents_turn_uturn, agents_turn_uturn_is_raw = _parse_agents(agents_turn_uturn)
    _fill_label(data_dict, 'label_turning', agents_turn_uturn, TurnAction.U_TURN)

    if only_draw_gt:
        output_dict = data_dict
        video_file = plot_gt_video(state.scenario)

    else:
        input_data_dict = {
            k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
            for k, v in data_dict.items()
        }
        # Extend the batch dim:
        input_data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in input_data_dict.items()}
        input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)

        with torch.no_grad():
            output_dict = model.model.autoregressive_rollout(
                input_data_dict, num_decode_steps=16, sampling_method=sampling_method, temperature=temperature
            )
            output_dict = get_tokenizer(model.config).detokenize(output_dict)
        output_dict = {
            k: (v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        video_file = plot_pred_video(
            state.scenario, output_dict, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
            agents_turn_left, agents_turn_right, agents_turn_uturn
        )

    # Postprocess
    if "decoder/label_safety" in output_dict:
        safety_label = output_dict["decoder/label_safety"]
        if agents_safe_0_is_raw:
            aid = [str(v) for v in (safety_label == 0).nonzero()[0]]
            agents_safe_0 = "[RAW]" + ",".join(aid)
        if agents_safe_1_is_raw:
            aid = [str(v) for v in (safety_label == 1).nonzero()[0]]
            agents_safe_1 = "[RAW]" + ",".join(aid)
    if "decoder/label_turning" in output_dict:
        turning_label = output_dict["decoder/label_turning"]
        if agents_turn_stop_is_raw:
            aid = [str(v) for v in (turning_label == TurnAction.STOP).nonzero()[0]]
            agents_turn_stop = "[RAW]" + ",".join(aid)
        if agents_turn_straight_is_raw:
            aid = [str(v) for v in (turning_label == TurnAction.KEEP_STRAIGHT).nonzero()[0]]
            agents_turn_straight = "[RAW]" + ",".join(aid)
        if agents_turn_left_is_raw:
            aid = [str(v) for v in (turning_label == TurnAction.TURN_LEFT).nonzero()[0]]
            agents_turn_left = "[RAW]" + ",".join(aid)
        if agents_turn_right_is_raw:
            aid = [str(v) for v in (turning_label == TurnAction.TURN_RIGHT).nonzero()[0]]
            agents_turn_right = "[RAW]" + ",".join(aid)
        if agents_turn_uturn_is_raw:
            aid = [str(v) for v in (turning_label == TurnAction.U_TURN).nonzero()[0]]
            agents_turn_uturn = "[RAW]" + ",".join(aid)

    return (
        gr.Video(value=video_file.name, label=state.scenario["id"]), "Scenario Generated!",
        ", ".join([str(v) for v in agents_safe_0]) if isinstance(agents_safe_0, list) else agents_safe_0,
        ", ".join([str(v) for v in agents_safe_1]) if isinstance(agents_safe_1, list) else agents_safe_1,
        ", ".join([str(v) for v in agents_turn_stop]) if isinstance(agents_turn_stop, list) else agents_turn_stop,
        ", ".join([str(v)
                   for v in agents_turn_straight]) if isinstance(agents_turn_straight, list) else agents_turn_straight,
        ", ".join([str(v) for v in agents_turn_left]) if isinstance(agents_turn_left, list) else agents_turn_left,
        ", ".join([str(v) for v in agents_turn_right]) if isinstance(agents_turn_right, list) else agents_turn_right,
        ", ".join([str(v) for v in agents_turn_uturn]) if isinstance(agents_turn_uturn, list) else agents_turn_uturn
    )


# ============================================================
# ======================== GRADIO UI =========================
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(text_size="lg")) as demo:
    with gr.Group():
        gr.Markdown("  ## Data")
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Path to Dataset Folder", value=DEFAULT_DATA_PATH)

            with gr.Column(scale=1):
                out = gr.Textbox(label="Status", placeholder="Enter to submit...")

        # gr.Markdown("## Visualization")
        with gr.Row(equal_height=True):  # Future release fix: https://github.com/gradio-app/gradio/pull/9577
            with gr.Column(scale=1):
                with gr.Group():
                    file_name_input = gr.Textbox(label="Search Scenario ID", max_lines=1)
                    file_explorer = gr.FileExplorer(
                        root_dir=state.dataset_path,
                        glob="**/*.pkl",
                        file_count="single",
                        interactive=True,
                        container=True,
                        max_height=900
                    )

            with gr.Column(scale=2):
                gt_vis = gr.Video(
                    label="Original Scenario",
                    show_download_button=False,
                    width=LENGTH,
                    height=LENGTH,
                    interactive=False
                )

    with gr.Group():
        gr.Markdown("## Model")
        with gr.Row():
            with gr.Column(scale=3):
                if DEFAULT_MODEL:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint", value=DEFAULT_MODEL, placeholder="/home/.../last.ckpt"
                    )
                else:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint",
                        placeholder="/home/.../last.ckpt (Type 'debug' for debug model!)"
                    )
            with gr.Column(scale=1):
                ckpt_output = gr.Textbox(label="Status", placeholder="Enter to load...")

        gr.Markdown("## Visualization")
        with gr.Row():
            with gr.Column(scale=1):
                sampling_method = gr.Radio(label="Sampling Method", choices=["softmax", "topp"], value="topp")
                temperature = gr.Slider(
                    label="Sampling Temperature", minimum=0.0, maximum=2.0, step=0.1, value=1.0, interactive=True
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                gr.Markdown(
                    "### Agents ID to assign labels:\n1. Split by comma ','\n2. If empty, original labels are used and printed as `[RAW]`"
                )
                agents_safe_0 = gr.Textbox(label="label_safety = 0 (NO COLL)", interactive=True, placeholder="0, 1")
                agents_safe_1 = gr.Textbox(label="label_safety = 1 (W/ COLL)", interactive=True, placeholder="0, 1")
                agents_turn_stop = gr.Textbox(label="label_turning = STOP", interactive=True)
                agents_turn_straight = gr.Textbox(label="label_turning = KEEP_STRAIGHT", interactive=True)
                agents_turn_left = gr.Textbox(label="label_turning = LEFT", interactive=True)
                agents_turn_right = gr.Textbox(label="label_turning = RIGHT", interactive=True)
                agents_turn_uturn = gr.Textbox(label="label_turning = U_TURN", interactive=True)
                generate_button = gr.Button(value="Generate")
                draw_gt_button = gr.Button(value="Draw Original Scenario")

            with gr.Column(scale=2):
                main_vis_text = gr.Textbox(label="Status", placeholder="", interactive=False)
                main_vis = gr.Video(label="Generated Scenario", show_download_button=False, width=LENGTH, height=LENGTH)

    inp.submit(on_dataset_path_submit, inputs=inp, outputs=[out, file_explorer])
    file_name_input.change(on_data_file_name_search, inputs=file_name_input, outputs=file_explorer)
    file_explorer.change(
        on_data_file_select,
        inputs=file_explorer,
        outputs=[
            gt_vis, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight, agents_turn_left,
            agents_turn_right, agents_turn_uturn
        ],
    )

    ckpt_input.submit(
        ckpt_callback,
        inputs=ckpt_input,
        outputs=[
            ckpt_output, sampling_method, temperature, main_vis, agents_safe_0, agents_safe_1, agents_turn_stop,
            agents_turn_straight, agents_turn_left, agents_turn_right, agents_turn_uturn
        ],
    )
    generate_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False),
        inputs=[
            sampling_method, temperature, seed, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
            agents_turn_left, agents_turn_right, agents_turn_uturn
        ],
        outputs=[
            main_vis, main_vis_text, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
            agents_turn_left, agents_turn_right, agents_turn_uturn
        ],
    )
    draw_gt_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=True),
        inputs=[
            sampling_method, temperature, seed, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
            agents_turn_left, agents_turn_right, agents_turn_uturn
        ],
        outputs=[
            main_vis, main_vis_text, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight,
            agents_turn_left, agents_turn_right, agents_turn_uturn
        ],
    )

if DEFAULT_MODEL:
    print("Loading default model from: ", DEFAULT_MODEL)
    ckpt_callback(DEFAULT_MODEL)

demo.queue().launch(server_port=7860, share=args.share)
