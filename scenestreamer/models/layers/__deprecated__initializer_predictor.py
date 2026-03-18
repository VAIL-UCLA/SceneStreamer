import torch.nn as nn

from scenestreamer import utils
from scenestreamer.dataset import constants
from scenestreamer.models.layers import common_layers


class InitializerPredictor(nn.Module):
    def __init__(self, d_model, num_modes):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.vel_head = common_layers.build_mlps(
            c_in=self.d_model,
            # mlp_channels=[self.num_modes * 2],
            mlp_channels=[self.d_model, self.d_model, self.num_modes * 6 * 3],
            ret_before_act=True,
        )
        self.heading_head = common_layers.build_mlps(
            c_in=self.d_model,
            # mlp_channels=[self.num_modes * 1],
            mlp_channels=[self.d_model, self.d_model, self.num_modes * 3 * 3],
            ret_before_act=True,
        )
        self.pos_head = common_layers.build_mlps(
            c_in=self.d_model,
            # mlp_channels=[self.num_modes * 2],
            mlp_channels=[self.d_model, self.d_model, self.num_modes * 6 * 3],
            ret_before_act=True,
        )
        self.size_head = common_layers.build_mlps(
            c_in=self.d_model,
            # mlp_channels=[self.num_modes * 3],
            mlp_channels=[self.d_model, self.d_model, self.num_modes * 7 * 3],
            ret_before_act=True,
        )
        self.score_head = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model, self.d_model, 1 * 3],
            ret_before_act=True,
        )

        self.type_head = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model, self.d_model, 5],
            ret_before_act=True,
        )

    def _get_dist(self, p):
        return utils.get_distribution(p)

    def forward(self, start_tokens, start_token_valid_mask):
        B, N, token_dim = start_tokens.shape
        # feat = self.feat(start_tokens[start_token_valid_mask])
        feat = start_tokens[start_token_valid_mask]
        num_modes = self.num_modes

        pred_vel = utils.unwrap(self.vel_head(feat),
                                start_token_valid_mask).reshape(B, N, constants.NUM_TYPES, num_modes, 6)
        vel_dist = self._get_dist(pred_vel)

        pred_head = utils.unwrap(self.heading_head(feat),
                                 start_token_valid_mask).reshape(B, N, constants.NUM_TYPES, num_modes, 3)
        head_dist = self._get_dist(pred_head)

        pred_pos = utils.unwrap(self.pos_head(feat),
                                start_token_valid_mask).reshape(B, N, constants.NUM_TYPES, num_modes, 6)
        pos_dist = self._get_dist(pred_pos)

        pred_size = utils.unwrap(self.size_head(feat),
                                 start_token_valid_mask).reshape(B, N, constants.NUM_TYPES, num_modes, 7)
        size_dist = self._get_dist(pred_size)

        map_feat_score = utils.unwrap(
            self.score_head(feat), start_token_valid_mask, fill=float("-inf")
        ).reshape(B, N, constants.NUM_TYPES, 1)

        actor_type = utils.unwrap(self.type_head(feat), start_token_valid_mask, fill=float("-inf")).reshape(B, N, 5)

        return pred_pos, pred_vel, pred_head, pred_size, map_feat_score, actor_type, pos_dist, vel_dist, head_dist, size_dist
