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
from scenestreamer.gradio_ui.plot import plot_gt, plot_pred
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils

os.environ['GRADIO_TEMP_DIR'] = str(REPO_ROOT / "gradio_tmp")

default_config = OmegaConf.load(REPO_ROOT / "cfgs/motion_default.yaml")

OmegaConf.set_struct(default_config, False)
default_config.MODEL.D_MODEL = 32
default_config.MODEL.NUM_DECODER_LAYERS = 1
default_config.MODEL.NUM_ATTN_LAYERS = 1
default_config.ACTION_LABEL.USE_SAFETY_LABEL = True
default_config.ACTION_LABEL.USE_ACTION_LABEL = True
default_config.ROOT_DIR = REPO_ROOT
OmegaConf.set_struct(default_config, True)

DEFAULT_DATA_PATH = "data/20scenarios"

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

from scenestreamer.models.motionlm_lightning import MotionLMLightning

ckpt_path = "/home/zhenghao/scenestreamer/lightning_logs/scenestreamer/1012_motionlm_joint_condition_2024-10-12_1149/last.ckpt"

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
        # print("Error: ", e)
        raise e
        msg = "Failed to load DEBUG model!"

path = pathlib.Path(ckpt_path)
path = REPO_ROOT / path

print("Loading model from: ", path.absolute())
if not path.exists():
    msg = "{} does not exist!".format(path)

try:
    model = utils.load_from_checkpoint(
        checkpoint_path=path, cls=MotionLMLightning, config=None, default_config=default_config
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
    raise e
    msg = "Failed to load model!"
