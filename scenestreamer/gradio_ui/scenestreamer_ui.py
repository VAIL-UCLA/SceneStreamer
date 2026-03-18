import os
import pathlib
import pickle
import gradio as gr

from scenestreamer.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from scenestreamer.tokenization import get_tokenizer

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from pathlib import Path
import uuid
from scenestreamer.utils import REPO_ROOT

# --- Configuration ---
DEFAULT_DATA_PATH = "data/20scenarios"  # Adjust as needed
DEFAULT_CONFIG_NAME = "0220_midgpt.yaml"
DEFAULT_MODEL = "/home/zhenghao/scenestreamer/lightning_logs/scenestreamer/0226_MidGPT_V19_WTG_2025-02-26/"

# Load config with Hydra
config_path = REPO_ROOT / "cfgs"
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    config_dict = compose(config_name=DEFAULT_CONFIG_NAME)
DEFAULT_CONFIG = config_dict


# --- Utility Functions ---
def list_map_files(dataset_path: str):
    """List all .pkl files in the dataset folder."""
    p = pathlib.Path(dataset_path)
    return sorted([str(f.relative_to(p)) for f in p.glob("*.pkl")])


def load_first_map(dataset_path: str):
    """Return the first map file in the folder, if available."""
    files = list_map_files(dataset_path)
    return files[0] if files else None


def load_and_visualize_map(selected_file, dataset_path, state):
    """
    Loads a selected map file (or auto-selects the first one if none is selected),
    updates the state, and visualizes the map.
    """
    # If no file is selected, auto-select the first one.
    if not selected_file:
        selected_file = load_first_map(dataset_path)
        if not selected_file:
            return "No map files found", state, None

    # Build the absolute file path.
    ds_path = pathlib.Path(dataset_path).resolve()
    file_path = ds_path / selected_file
    if not file_path.exists():
        return "File not found", state, None

    # Load the map data (assumed to be a pickle file).
    try:
        with open(file_path, "rb") as f:
            scenario = pickle.load(f)
    except Exception as e:
        return f"Failed to load file: {e}", state, None

    data_dict = preprocess_scenario_description_for_motionlm(
        scenario=scenario,
        config=DEFAULT_CONFIG,
        in_evaluation=True,
        keep_all_data=True,
        tokenizer=get_tokenizer(config=DEFAULT_CONFIG)
    )

    # Update the state with the loaded map.
    state["selected_map"] = {"path": str(file_path), "scenario": scenario}

    # Visualize the map using your existing plotting routine.
    from scenestreamer.gradio_ui.plot import plot_gt
    result = plot_gt(data_dict)
    img = result[0] if isinstance(result, tuple) else result

    return f"Loaded map: {file_path.name}", state, img


# --- Generation Functions ---
def generate_initial_states(state):
    """
    Generate initial states and return a status message along with the
    visualization image obtained via plot_gt (with get_info=True).
    """
    if "selected_map" not in state or not state["selected_map"]:
        return "No map loaded. Please select a map first.", state, None

    from scenestreamer.infer.initial_state import generate_initial_state, convert_initial_states_as_motion_data
    from scenestreamer.utils import utils

    force_add = False
    scenario = state["selected_map"]["scenario"]
    config = state["config"]
    pl_model = state["model"]

    data_dict = preprocess_scenario_description_for_motionlm(
        scenario=scenario,
        config=config,
        in_evaluation=True,
        keep_all_data=False,
        tokenizer=get_tokenizer(config=config)
    )
    data_dict = utils.batch_data(utils.numpy_to_torch(data_dict, device=pl_model.device))

    data_dict, _ = generate_initial_state(data_dict=data_dict, model=pl_model.model, force_add=force_add)

    data_dict = convert_initial_states_as_motion_data(data_dict)

    state["initial_state_output_data_dict"] = data_dict

    unbatched_data = utils.unbatch_data(utils.torch_to_numpy(data_dict))

    # Draw the image using your plot_gt function with get_info=True.
    from scenestreamer.gradio_ui.plot import plot_gt
    img, info_dict = plot_gt(unbatched_data, get_info=True)
    return f"Initial states generated for map {state['selected_map']['path']}", state, img


def generate_motions(state, num_decode_steps):
    """
    Generate motion predictions and return a status message along with a video.
    Here we use create_animation_from_pred as a placeholder for video generation.
    """
    if "selected_map" not in state or not state["selected_map"]:
        return "No map loaded. Please select a map first.", state, None

    from scenestreamer.infer.motion import generate_motion
    from scenestreamer.utils import utils

    pl_model = state["model"]
    config = state["config"]
    data_dict = state["initial_state_output_data_dict"]

    generated_data_dict = generate_motion(
        data_dict=data_dict,
        model=pl_model.model,
        autoregressive_start_step=0,
        num_decode_steps=num_decode_steps,
        remove_out_of_map_agent=True
    )

    unbatched_data = utils.unbatch_data(utils.torch_to_numpy(generated_data_dict))

    from scenestreamer.gradio_ui.plot import create_animation_from_pred
    video_path = str(REPO_ROOT / "gradio_tmp" / "gt_animation_{}.mp4".format(uuid.uuid4()))
    video_path = create_animation_from_pred(unbatched_data, save_path=video_path, dpi=100, fps=10)
    # TODO: 0.5 hardcoded.
    return f"Motions generated for map {state['selected_map']['path']} with {num_decode_steps * 0.5}s", state, video_path


def generate_scenestreamer_motions(state, num_decode_steps):
    if "selected_map" not in state or not state["selected_map"]:
        return "No map loaded. Please select a map first.", state, None

    from scenestreamer.infer.infinite import generate_scenestreamer_motion
    from scenestreamer.utils import utils

    pl_model = state["model"]
    config = state["config"]
    data_dict = state["initial_state_output_data_dict"]

    generated_data_dict = generate_scenestreamer_motion(
        data_dict=data_dict,
        model=pl_model.model,
        autoregressive_start_step=0,
        num_decode_steps=num_decode_steps,
        remove_out_of_map_agent=True
    )

    unbatched_data = utils.unbatch_data(utils.torch_to_numpy(generated_data_dict))

    from scenestreamer.gradio_ui.plot import create_animation_from_pred
    video_path = str(REPO_ROOT / "gradio_tmp" / "scenestreamer_animation_{}.mp4".format(uuid.uuid4()))
    video_path = create_animation_from_pred(unbatched_data, save_path=video_path, dpi=100, fps=10)
    return state, video_path


# --- Model Checkpoint Loading Function ---
def load_checkpoint(ckpt_path, state):
    from scenestreamer.models.motionlm_lightning import MotionLMLightning
    from scenestreamer.utils import utils
    import torch
    import copy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = ckpt_path.replace("\\", "")
    path = pathlib.Path(ckpt_path)
    path = REPO_ROOT / path

    if path.is_dir():
        path = path / "last.ckpt"

    print("Loading model from: ", path.absolute())
    if not path.exists():
        msg = f"{path} does not exist!"
        return msg

    try:
        model = utils.get_model(config=None, checkpoint_path=path, device=device).eval()
        msg = "Model loaded successfully!"
        config = model.config
        state["model"] = model
        state["config"] = config
    except Exception as e:
        print("Error: ", e)
        msg = "Failed to load model!"
    return msg


# --- Build the Gradio UI ---
with gr.Blocks(title="Map & Motion Generator") as demo:
    # Use Gradio state (a dictionary) to store our data.
    state = gr.State(value={})

    with gr.Group("Map Selection") as map_group:
        gr.Markdown("### Map Selection and Visualization")
        with gr.Row():
            with gr.Column():
                dataset_path_input = gr.Textbox(label="Dataset Folder", value=DEFAULT_DATA_PATH, interactive=True)
            with gr.Column():
                file_status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            with gr.Column():
                file_explorer = gr.FileExplorer(
                    label="Choose a map (.pkl)",
                    file_count="single",
                    root_dir=DEFAULT_DATA_PATH,
                    glob="**/*.pkl",
                    interactive=True
                )
            with gr.Column():
                map_image = gr.Image(label="Map Visualization")
        # Callback to load and visualize map when a file is selected.
        file_explorer.change(
            load_and_visualize_map,
            inputs=[file_explorer, dataset_path_input, state],
            outputs=[file_status, state, map_image]
        )
        # Automatically load the first map on startup.
        demo.load(
            load_and_visualize_map,
            inputs=[file_explorer, dataset_path_input, state],
            outputs=[file_status, state, map_image]
        )

    with gr.Group("Model Checkpoint") as ckpt_group:
        gr.Markdown("### Model Checkpoint Loading")
        with gr.Row():
            with gr.Column():
                ckpt_input = gr.Textbox(label="Checkpoint Path", value=DEFAULT_MODEL, interactive=True)
            with gr.Column():
                ckpt_status = gr.Textbox(label="Checkpoint Status", interactive=False)
        with gr.Row():
            ckpt_button = gr.Button("Load Checkpoint")
        ckpt_button.click(load_checkpoint, inputs=[ckpt_input, state], outputs=ckpt_status)
        # Automatically load the default checkpoint on startup.
        demo.load(load_checkpoint, inputs=[ckpt_input, state], outputs=ckpt_status)

    with gr.Group("Generation Controls") as gen_group:
        gr.Markdown("### Generation Controls")
        with gr.Row():
            gen_initial_button = gr.Button("Generate Initial States")
            gen_motions_button = gr.Button("Generate Motions")
        with gr.Row():
            # New numeric input for number of decoded steps.
            num_decode_steps = gr.Slider(label="Number of Decoded Steps", value=19, interactive=True)

        gen_status = gr.Textbox(label="Generation Status", interactive=False)

    with gr.Group("Output Visualization Canvas"):
        gr.Markdown("### Output Visualization")
        with gr.Row():
            canvas_image = gr.Image(label="Initial State Image", interactive=False, height=400, width=400)
            canvas_video = gr.Video(
                label="Motion Video", interactive=False, height=400, width=400, autoplay=True, loop=True
            )

    # Connect generation outputs to the visualization canvases.
    # For initial states, update the image canvas.
    gen_initial_button.click(generate_initial_states, inputs=state, outputs=[gen_status, state, canvas_image])
    # For motions, update the video canvas.
    gen_motions_button.click(
        generate_motions, inputs=[state, num_decode_steps], outputs=[gen_status, state, canvas_video]
    )

    with gr.Group("SceneStreamer") as gen_group:
        gr.Markdown("### SceneStreamer")
        with gr.Row():
            scenestreamer_button = gr.Button("Kickoff Continuous Generation")
        with gr.Row():
            # New numeric input for number of decoded steps.
            scenestreamer_num_decode_steps = gr.Slider(label="Number of Decoded Steps", value=100, interactive=True)

        gr.Markdown("### SceneStreamer Visualization")
        with gr.Row():
            scenestreamer_canvas_video = gr.Video(
                label="SceneStreamer Video", interactive=False, height=800, width=800, autoplay=True, loop=True
            )
    scenestreamer_button.click(
        generate_scenestreamer_motions,
        inputs=[state, scenestreamer_num_decode_steps],
        outputs=[state, scenestreamer_canvas_video],
    )

demo.launch()
