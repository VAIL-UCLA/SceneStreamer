import collections
import copy
import time

import numpy as np
import torch
from tqdm import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.models.motionlm import MotionLM
from scenestreamer.utils import debug_tools


def toy_test():
    cfg_file = "cfgs/motion_debug_2_local_train.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    config.MODEL.update(dict(
        D_MODEL=512,
        NUM_ATTN_LAYERS=6,
        NUM_ATTN_HEAD=8,
        NUM_DECODER_LAYERS=6,
    ))
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=1,
        train_num_workers=0,
        val_batch_size=2,
        val_num_workers=8,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    model = MotionLM(config)
    model.eval()
    model.cuda()

    time_no_cache = 0.0
    time_with_cache = 0.0
    stat_dict = collections.defaultdict(list)
    for input_dict in tqdm(dataloader):
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.cuda()

        num_modes_for_eval = 6

        def _repeat_for_modes(v):
            d = v.ndim
            if d > 1:
                v = v[:, None]
                v = v.repeat(1, num_modes_for_eval, *((1, ) * (d - 1)))
                v = v.flatten(0, 1)
            else:
                v = v.repeat(num_modes_for_eval)
            return v

        input_dict = {
            k: _repeat_for_modes(input_dict[k])
            for k in input_dict.keys() if (
                k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                or k.startswith("eval/")
            )
        }

        input_dict2 = copy.deepcopy(input_dict)
        input_dict1 = input_dict

        s = time.time()
        with torch.no_grad():
            input_dict1 = model.autoregressive_rollout(
                input_dict1, num_decode_steps=16, use_cache=False, sampling_method="argmax"
            )
        time_no_cache += time.time() - s

        s = time.time()
        with torch.no_grad():
            input_dict2 = model.autoregressive_rollout(
                input_dict2, num_decode_steps=16, use_cache=True, sampling_method="argmax"
            )
        time_with_cache += time.time() - s

        diff_dict = {
            k: (input_dict1[k].float() - input_dict2[k].float()).abs().mean().item()
            for k in input_dict1 if isinstance(input_dict1[k], torch.Tensor)
        }
        action_mismatch = (input_dict1["decoder/output_action"] !=
                           input_dict2["decoder/output_action"]).float().mean().item()
        diff_dict = {k: v for k, v in diff_dict.items() if v != 0.0}
        diff_dict["action_mismatch"] = action_mismatch

        for k, v in diff_dict.items():
            stat_dict[k].append(v)

    stat_dict = {k: np.mean(v) for k, v in stat_dict.items()}
    print(f"FINISHED. TIME without CACHE: {time_no_cache}, TIME with CACHE: {time_with_cache}.\nDIFF:{stat_dict}")


if __name__ == '__main__':
    toy_test()
