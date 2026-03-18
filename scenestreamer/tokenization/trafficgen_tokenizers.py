import logging
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from scenestreamer.utils import rotate, wrap_to_pi
from scenestreamer.utils import utils


class TrafficGenTokenizerBaseVer:
    limit = {
        "position_x": (-30, 30),
        "position_y": (-20, 20),
        "velocity_x": (0, 30),
        "velocity_y": (-10, 10),
        "heading": (-np.pi / 2, np.pi / 2),
        "length": (0.5, 10),
        "width": (0.5, 3),
        "height": (0.5, 4),
        "agent_type": (None, None),
    }
    num_bins = {
        "position_x": 121,
        "position_y": 81,
        "velocity_x": 61,
        "velocity_y": 41,
        "heading": 21,
        "length": 21,
        "width": 11,
        "height": 11,
        "agent_type": 3,
    }

    def __init__(self, config):
        self.config = config
        self.start_action_id = config.PREPROCESSING.MAX_MAP_FEATURES
        self.end_action_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1
        self.INIT_START_ACTION = self.start_action_id
        self.INIT_END_ACTION = self.end_action_id

    @classmethod
    def bucketize(cls, value, key):
        is_torch = isinstance(value, torch.Tensor)
        if not is_torch:
            value = torch.tensor(value)
        limit_min, limit_max = cls.limit[key]
        num_bins = cls.num_bins[key]
        if limit_min is None:
            return value
        value = torch.clamp(value, limit_min, limit_max)
        ret = torch.round((value - limit_min) / (limit_max - limit_min) * (num_bins - 1))
        if not is_torch:
            ret = ret.numpy()
        return ret

    @classmethod
    def de_bucketize(cls, value, key):
        is_torch = isinstance(value, torch.Tensor)
        if not is_torch:
            value = torch.tensor(value)
        limit_min, limit_max = cls.limit[key]
        num_bins = cls.num_bins[key]
        if limit_min is None:
            return value
        ret = value / (num_bins - 1) * (limit_max - limit_min) + limit_min
        if not is_torch:
            ret = ret.numpy()
        return ret

    def detokenize(self, data_dict, action, agent_type, offset_action):
        B, M, _ = data_dict["encoder/map_position"].shape
        action = action.clone().unsqueeze(-1)
        assert action.ndim == 3  # B, T, 1

        is_valid_action = action < self.start_action_id
        action[~is_valid_action] = 0
        map_pos = torch.gather(data_dict["encoder/map_position"], dim=1, index=action.expand(-1, -1, 3))[..., :2]
        map_head = torch.gather(data_dict["encoder/map_heading"][:, :M], dim=1, index=action.reshape(B, -1))

        offset_values = {}
        for k, a in offset_action.items():
            offset_values[k] = self.de_bucketize(a, k)

        pos = utils.rotate(x=offset_values["position_x"], y=offset_values["position_y"], angle=map_head)
        pos = pos + map_pos

        head = wrap_to_pi(offset_values["heading"] + map_head)

        vel = utils.rotate(x=offset_values["velocity_x"], y=offset_values["velocity_y"], angle=map_head)

        shape = torch.stack([offset_values["length"], offset_values["width"], offset_values["height"]], dim=-1)

        # feature is the relative pos/head/vel
        feature = torch.stack(
            [
                offset_values["position_x"], offset_values["position_y"], offset_values["heading"],
                offset_values["velocity_x"], offset_values["velocity_y"]
            ],
            dim=-1
        )

        # I am sorry that we change the key names here...
        return {
            "position": pos,
            "velocity": vel,
            "heading": head,
            "shape": shape,
            "agent_type": agent_type,
            "feature": feature,
            "offset_values": offset_values,
        }


class TrafficGenTokenizerSpecialVer(TrafficGenTokenizerBaseVer):
    limit = {
        "position_x": (-10, 10),
        "position_y": (-10, 10),
        "velocity_x": (0, 30),
        "velocity_y": (-10, 10),
        "heading": (-np.pi / 4, np.pi / 4),
        "length": (0.5, 10),
        "width": (0.5, 3),
        "height": (0.5, 4),
        "agent_type": (None, None),
    }
    num_bins = {
        "position_x": 81,
        "position_y": 81,
        "velocity_x": 61,
        "velocity_y": 41,
        "heading": 41,
        "length": 41,
        "width": 41,
        "height": 41,
        "agent_type": 3,
    }


# TODO: Hardcoded...
TrafficGenTokenizer = TrafficGenTokenizerSpecialVer

class TrafficGenTokenizerAutoregressive(TrafficGenTokenizerBaseVer):
    limit = {
        "position_x": (-10, 10),
        "position_y": (-10, 10),
        "velocity_x": (0, 30),
        "velocity_y": (-10, 10),
        "heading": (-np.pi / 2, np.pi / 2),
        "length": (0.5, 10),
        "width": (0.5, 3),
        "height": (0.5, 4),
        "agent_type": (None, None),
    }
    num_bins = {
        "position_x": 81,
        "position_y": 81,
        "velocity_x": 81,
        "velocity_y": 81,
        "heading": 81,
        "length": 81,
        "width": 81,
        "height": 81,
        "agent_type": 3,
    }
