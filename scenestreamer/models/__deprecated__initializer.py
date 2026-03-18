"""

"""
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scenestreamer.models.layers.initializer_predictor import InitializerPredictor
from torch import Tensor

from scenestreamer.dataset import constants
from scenestreamer.models.layers import polyline_encoder, common_layers
from scenestreamer.models.layers import position_encoding_utils
from scenestreamer.models.layers.multi_head_attention import MultiheadAttention
# from scenestreamer.models.layers.multi_head_attention_local import MultiheadAttentionLocal
from scenestreamer.models.ops.knn import knn_utils
from scenestreamer.utils import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        use_local_attn=False,
    ):
        super().__init__()
        self.use_local_attn = use_local_attn

        if self.use_local_attn:
            raise ValueError()
            self.self_attn = MultiheadAttentionLocal(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = common_layers.get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        index_pair=None,
        query_batch_cnt=None,
        key_batch_cnt=None,
        index_pair_batch=None
    ):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            index_pair=index_pair,
            query_batch_cnt=query_batch_cnt,
            key_batch_cnt=key_batch_cnt,
            index_pair_batch=index_pair_batch
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SceneEncoderFuser(nn.Module):
    """
    A stack of transformer layers to fuse multi-modal information of a scenario.
    Note that the embedding layer for each modality is not included.

    Input: The embedding of different modality.
    Output: A set of tokens of the scene.
    """
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.d_model = d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_ATTN_LAYERS
        self.num_modes = self.model_cfg.NUM_MOTION_MODES
        self.num_of_neighbors = self.model_cfg.NUM_NEIGHBORS

        nhead = self.model_cfg.NUM_ATTN_HEAD
        dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)

        # build transformer encoder layers
        self.use_local_attn = True
        self_attn_layers = []
        for _ in range(self.num_decoder_layers):
            self_attn_layers.append(
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    normalize_before=False,
                    use_local_attn=self.use_local_attn
                )
            )

        self.self_attn_layers = nn.ModuleList(self_attn_layers)

    def apply_local_attn(
        self, *, map_token, agent_token, map_position, agent_position, map_valid_mask, agent_valid_mask,
        num_of_neighbors, light_token, light_position, light_valid_mask
    ):

        B, M, D = map_token.shape

        if agent_token is not None:
            _, N, _ = agent_token.shape

            x = torch.cat([map_token, agent_token], axis=1)
            x_mask = torch.cat([map_valid_mask, agent_valid_mask], axis=1)
            x_pos = torch.cat([map_position, agent_position], axis=1)[..., :2]

        else:
            N = 0

            x = map_token
            x_mask = map_valid_mask
            x_pos = map_position[..., :2]

        if light_token is not None:
            _, L, _ = light_token.shape
            x = torch.cat([x, light_token], axis=1)
            x_mask = torch.cat([x_mask, light_valid_mask], axis=1)
            x_pos = torch.cat([x_pos, light_position[..., :2]], axis=1)
        else:
            L = 0

        assert torch.all(x_mask.sum(dim=-1) > 0)
        # batch_size, N, d_model = x.shape
        _, num_tokens, _ = x.shape

        x_stack_full = x.view(-1, D)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)

        x_pos_stack_full = x_pos.view(-1, 2)
        batch_idxs_full = torch.arange(B).type_as(x)[:, None].repeat(1, num_tokens).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]

        # It is in shape (BS * N,). It record the batch index of each selected "map feat".
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = utils.get_batch_offsets(batch_idxs=batch_idxs, bs=B, device=x.device)  # (batch_size + 1)

        # in shape (bs,)
        # how many map_feat's index==i
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        x_pos_stack_3d = F.pad(x_pos_stack, (0, 1))
        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack_3d, x_pos_stack_3d, batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = \
            position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=D)[0]

        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )

        ret_full_feature = utils.unwrap(output, x_mask)

        output_map_token = ret_full_feature[:, :M]

        if agent_token is not None:
            output_agent_token = ret_full_feature[:, M:M + N]
            assert output_agent_token.shape[1] == N

        else:
            output_agent_token = None

        if light_token is not None:
            output_light_token = ret_full_feature[:, M + N:]
            assert output_light_token.shape[1] == L
        else:
            output_light_token = None

        return output_map_token, output_agent_token, output_light_token

    def forward(
        self, *, map_token, agent_token, map_position, agent_position, map_valid_mask, agent_valid_mask, light_token,
        light_position, light_valid_mask
    ):
        if self.use_local_attn:
            out = self.apply_local_attn(
                map_token=map_token,
                agent_token=agent_token,
                map_position=map_position,
                agent_position=agent_position,
                map_valid_mask=map_valid_mask,
                agent_valid_mask=agent_valid_mask,
                num_of_neighbors=self.num_of_neighbors,
                light_token=light_token,
                light_position=light_position,
                light_valid_mask=light_valid_mask
            )
        else:
            raise ValueError()
            # global_token_feature = self.apply_global_attn(
            #     x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            # )
        return out


def get_distributions_for_evaluation(data_dict, model_output, selected_map, actor_type):
    B, M, _ = data_dict["encoder/map_position"].shape

    actor_type_index = (actor_type - 1).reshape(B, 1, 1).expand(B, M, 1)

    selected_map_pos = torch.gather(
        data_dict["encoder/map_position"],  # [B, M, 2]
        index=selected_map.reshape(B, 1, 1).expand(B, 1, 3),  # [B, N]
        dim=1
    ).squeeze(1)  # [B, 3]
    selected_map_heading = torch.gather(
        data_dict["encoder/map_heading"], dim=1, index=selected_map.reshape(B, 1)
    ).squeeze(1)  # [B]
    # Recenter the map position here. This is a little troublesome.

    # ===== Size =====
    selected_map = selected_map.reshape(B, 1, 1, 1)
    actor_type_index = actor_type_index.reshape(B, M, 1, 1, 1)
    nearest_size_logit = torch.gather(
        model_output["fake_size"], dim=1, index=actor_type_index.expand(B, M, 1, *model_output["fake_size"].shape[3:])
    ).squeeze(2)
    nearest_size_logit = torch.gather(
        nearest_size_logit, dim=1, index=selected_map.expand(B, 1, *model_output["fake_size"].shape[3:])
    ).squeeze(1)
    size_dist = utils.get_distribution(nearest_size_logit)

    nearest_position_logit = torch.gather(
        model_output["fake_position"],
        dim=1,
        index=actor_type_index.expand(B, M, 1, *model_output["fake_position"].shape[3:])
    ).squeeze(2)
    nearest_position_logit = torch.gather(
        nearest_position_logit, dim=1, index=selected_map.expand(B, 1, *model_output["fake_position"].shape[3:])
    ).squeeze(1)
    pos_dist = utils.get_distribution(nearest_position_logit)

    nearest_heading_logit = torch.gather(
        model_output["fake_heading"],
        dim=1,
        index=actor_type_index.expand(B, M, 1, *model_output["fake_heading"].shape[3:])
    ).squeeze(2)
    nearest_heading_logit = torch.gather(
        nearest_heading_logit, dim=1, index=selected_map.expand(B, 1, *model_output["fake_heading"].shape[3:])
    ).squeeze(1)
    head_dist = utils.get_distribution(nearest_heading_logit)

    nearest_velocity_logit = torch.gather(
        model_output["fake_velocity"],
        dim=1,
        index=actor_type_index.expand(B, M, 1, *model_output["fake_velocity"].shape[3:])
    ).squeeze(2)
    nearest_velocity_logit = torch.gather(
        nearest_velocity_logit, dim=1, index=selected_map.expand(B, 1, *model_output["fake_velocity"].shape[3:])
    ).squeeze(1)
    vel_dist = utils.get_distribution(nearest_velocity_logit)

    return selected_map_pos, selected_map_heading, pos_dist, vel_dist, head_dist, size_dist


def get_distributions_for_training(data_dict, model_output, selected_map, actor_type):
    B, M, _ = data_dict["encoder/map_position"].shape

    _, _, N, _ = data_dict["encoder/agent_feature"].shape

    selected_map_pos = torch.gather(
        data_dict["encoder/map_position"],  # [B, M, 2]
        index=selected_map.reshape(B, N, 1).expand(B, N, 3),  # [B, N]
        dim=1
    )  # [B, N, 3]
    selected_map_heading = torch.gather(
        data_dict["encoder/map_heading"], dim=1, index=selected_map.reshape(B, N)
    )  # [B, N]
    # Recenter the map position here. This is a little troublesome.

    # ===== Size =====
    selected_map = selected_map.reshape(B, N, 1, 1, 1)

    actor_type = actor_type.clone()
    actor_type[(actor_type < 1) | (actor_type > 3)] = 1
    actor_type = actor_type - 1

    actor_type_index = actor_type.reshape(B, N, 1, 1, 1)
    nearest_size_logit = torch.gather(
        model_output["fake_size"], dim=1, index=selected_map.expand(B, N, *model_output["fake_size"].shape[2:])
    )
    nearest_size_logit = torch.gather(
        nearest_size_logit, dim=2, index=actor_type_index.expand(B, N, 1, *model_output["fake_size"].shape[3:])
    ).squeeze(2)
    size_dist = utils.get_distribution(nearest_size_logit)

    nearest_position_logit = torch.gather(
        model_output["fake_position"], dim=1, index=selected_map.expand(B, N, *model_output["fake_position"].shape[2:])
    )
    nearest_position_logit = torch.gather(
        nearest_position_logit, dim=2, index=actor_type_index.expand(B, N, 1, *model_output["fake_position"].shape[3:])
    ).squeeze(2)
    pos_dist = utils.get_distribution(nearest_position_logit)

    nearest_heading_logit = torch.gather(
        model_output["fake_heading"], dim=1, index=selected_map.expand(B, N, *model_output["fake_heading"].shape[2:])
    ).squeeze(1)
    nearest_heading_logit = torch.gather(
        nearest_heading_logit, dim=2, index=actor_type_index.expand(B, N, 1, *model_output["fake_heading"].shape[3:])
    ).squeeze(2)
    head_dist = utils.get_distribution(nearest_heading_logit)

    nearest_velocity_logit = torch.gather(
        model_output["fake_velocity"], dim=1, index=selected_map.expand(B, N, *model_output["fake_velocity"].shape[2:])
    )
    nearest_velocity_logit = torch.gather(
        nearest_velocity_logit, dim=2, index=actor_type_index.expand(B, N, 1, *model_output["fake_velocity"].shape[3:])
    ).squeeze(2)
    vel_dist = utils.get_distribution(nearest_velocity_logit)

    return selected_map_pos, selected_map_heading, pos_dist, vel_dist, head_dist, size_dist


@torch.no_grad()
def sample_from_distributions(
    pos_dist, vel_dist, head_dist, size_dist, deterministic_state, selected_map_pos, selected_map_heading
):
    if deterministic_state:
        sampled_size = size_dist.mean  # * constants.SIZE_RANGE
    else:
        sampled_size = size_dist.sample()  # * constants.SIZE_RANGE
    sampled_size = sampled_size.clamp(0.1)

    if deterministic_state:
        sampled_pos = pos_dist.mean  # * constants.LOCAL_POSITION_XY_RANGE
    else:
        sampled_pos = pos_dist.sample()  # * constants.LOCAL_POSITION_XY_RANGE
    # sampled_pos = sampled_pos.clamp(-LOCAL_POSITION_XY_RANGE, constants.LOCAL_POSITION_XY_RANGE)
    sampled_pos = torch.cat([sampled_pos, sampled_size[:, 2:] / 2], dim=-1)  # Add Z axis
    sampled_pos = utils.relative_to_absolute(sampled_pos, selected_map_heading)
    sampled_pos = sampled_pos + selected_map_pos

    if deterministic_state:
        sampled_head = head_dist.mean  # * constants.HEADING_RANGE
    else:
        sampled_head = head_dist.sample()  # * constants.HEADING_RANGE
    # sampled_head = sampled_head.clamp(-np.pi/2, np.pi/2)
    sampled_head = utils.wrap_to_pi(sampled_head + selected_map_heading)

    if deterministic_state:
        sampled_vel = vel_dist.mean  # * constants.VELOCITY_XY_RANGE
    else:
        sampled_vel = vel_dist.sample()  # * constants.VELOCITY_XY_RANGE
    sampled_vel = utils.relative_to_absolute(sampled_vel, selected_map_heading)

    return sampled_pos, sampled_vel, sampled_head, sampled_size


def if_intersection(new, actor_position, actor_feature, actor_valid_mask):
    if actor_position is None:
        return False
    pos = actor_position[:, 0]  # [B, N, 3]
    size = actor_feature[:, 0, :, 6:9]  # * constants.SIZE_RANGE
    max_size = size.max(-1)[0]  # [B, N]
    dist = torch.cdist(new["sampled_pos"].unsqueeze(1), pos).squeeze(1)  # [B, N]
    ret = (dist < max_size).any(-1)  # [B, ]
    return ret


@torch.no_grad()
def sample_new_actor(
    data_dict,
    model_output,
    sampling_method,
    actor_type,
    temperature=1.0,
    topk=10,
    topp=0.9,
    deterministic_state=False,
    use_nature_probability=False
):
    # [B, N, num_modes]
    B, M, _, num_modes, _ = model_output["fake_position"].shape

    # actor_type is in [B,] in range {0, 1, 2, 3, 4}
    # We can use it to select map feature.
    actor_type = actor_type.clone()
    actor_type[(actor_type < 1) | (actor_type > 3)] = 1
    actor_type_index = (actor_type - 1).reshape(B, 1, 1).expand(B, M, 1)

    if use_nature_probability:
        map_prob = F.sigmoid(model_output["fake_map_feat_score"]).reshape(B, M, constants.NUM_TYPES)  # [B, M, 3]

        pos_log_prob = model_output["fake_position_dist"].log_prob(model_output["fake_position_dist"].mean)  # [B, M, 3]
        vel_log_prob = model_output["fake_velocity_dist"].log_prob(model_output["fake_velocity_dist"].mean)  # [B, M, 3]
        head_log_prob = model_output["fake_heading_dist"].log_prob(model_output["fake_heading_dist"].mean)  # [B, M, 3]
        size_log_prob = model_output["fake_size_dist"].log_prob(model_output["fake_size_dist"].mean)  # [B, M, 3]

        prob = map_prob * torch.exp(pos_log_prob + vel_log_prob + head_log_prob + size_log_prob)

        # mask out invalid map feature
        prob[~model_output["encoder/map_valid_mask"]] = 0

        if temperature is not None:
            prob = prob**(1.0 / temperature)
        score = prob
    else:
        score = model_output["fake_map_feat_score"].reshape(B, M, constants.NUM_TYPES)
        score[~model_output["encoder/map_valid_mask"]] = float("-inf")
        score = score / temperature  # [B, M, 3]

    score = torch.gather(input=score, index=actor_type_index, dim=2).reshape(B, M)

    if sampling_method == "softmax":
        if use_nature_probability:
            selected_map = torch.distributions.Categorical(probs=score.clamp(-100, 100)).sample()
        else:
            selected_map = torch.distributions.Categorical(logits=score.clamp(-100, 100)).sample()

    elif sampling_method == "topk":
        indices_to_remove = score < score.topk(topk, dim=-1)[0][..., -1, None]
        if use_nature_probability:
            score[indices_to_remove] = 0.0
            selected_map = torch.distributions.Categorical(probs=score.clamp(-100, 100)).sample()
        else:
            score[indices_to_remove] = float("-inf")
            selected_map = torch.distributions.Categorical(logits=score.clamp(-100, 100)).sample()

    elif sampling_method == "topp":
        sorted_logits, sorted_indices = torch.sort(score, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create a mask for all tokens whose cumulative probability exceeds the threshold
        sorted_indices_to_remove = cumulative_probs > topp

        # Since we want to keep at least one token, shift the mask to the right
        # This way the first token (with the highest probability) will always be kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter the sorted tensor to match the original indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        # Set all logits that are masked out to a very large negative number,
        # so that they become zero after applying softmax
        if use_nature_probability:
            score[indices_to_remove] = 0.0
            selected_map = torch.distributions.Categorical(probs=score.clamp(-100, 100)).sample()
        else:
            score[indices_to_remove] = float("-inf")
            selected_map = torch.distributions.Categorical(logits=score.clamp(-100, 100)).sample()

    elif sampling_method == "argmax":
        selected_map = score.argmax(-1)

    else:
        raise ValueError(f"Unknown {sampling_method}")
    assert selected_map.shape == (B, )

    selected_map_pos, selected_map_heading, pos_dist, vel_dist, head_dist, size_dist = get_distributions_for_evaluation(
        data_dict, model_output, selected_map, actor_type
    )

    new_actor_feature = score.new_zeros([B, constants.AGENT_STATE_DIM])

    sampled_pos, sampled_vel, sampled_head, sampled_size = sample_from_distributions(
        size_dist=size_dist,
        pos_dist=pos_dist,
        head_dist=head_dist,
        vel_dist=vel_dist,
        deterministic_state=deterministic_state,
        selected_map_pos=selected_map_pos,
        selected_map_heading=selected_map_heading
    )

    new_actor_feature[:, 6:9] = sampled_size  # / constants.SIZE_RANGE
    new_actor_feature[:, :3] = sampled_pos  # / constants.POSITION_XY_RANGE
    new_actor_feature[:, 3] = sampled_head  # / constants.HEADING_RANGE
    new_actor_feature[:, 9] = torch.sin(sampled_head)
    new_actor_feature[:, 10] = torch.cos(sampled_head)
    new_actor_feature[:, 4:6] = sampled_vel  # / constants.VELOCITY_XY_RANGE
    new_actor_feature[:, 11] = sampled_vel.norm(dim=-1)  # / constants.VELOCITY_XY_RANGE

    # actor type
    new_actor_feature[actor_type == 1, 12] = 1
    new_actor_feature[actor_type == 2, 13] = 1
    new_actor_feature[actor_type == 3, 14] = 1

    new_actor_feature[:, 15] = 1

    return new_actor_feature, {
        "sampled_pos": sampled_pos,
        "sampled_vel": sampled_vel,
        "sampled_head": sampled_head,
        "sampled_size": sampled_size
    }


class SceneStreamerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_ATTN_LAYERS
        self.num_modes = self.model_cfg.NUM_MOTION_MODES

        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=constants.MAP_FEATURE_STATE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_pre_layers=1,
            out_channels=self.d_model
        )
        self.actor_mlps = common_layers.build_mlps(
            c_in=constants.AGENT_STATE_DIM,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )
        self.light_mlps = common_layers.build_mlps(
            c_in=constants.TRAFFIC_LIGHT_STATE_DIM,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )

        self.start_token_predictor = InitializerPredictor(
            d_model=self.d_model,
            num_modes=self.model_cfg.NUM_MOTION_MODES,
            # max_map_feat=self.config.MAX_MAP_FEATURES
        )

        self.encoder = SceneEncoderFuser(self.model_cfg)

    def get_loss(self, data_dict, gt_dict, forward_ret_dict, tb_pre_tag=''):
        actor_position = data_dict["encoder/agent_position"][:, 0]  # [B, N, 2]
        raw_actor_valid_mask = data_dict["encoder/agent_valid_mask"][:, 0]

        map_mask = forward_ret_dict["map_mask_for_initializer"]

        map_position = data_dict["encoder/map_position"]
        map_dist = torch.cdist(actor_position, map_position, compute_mode="donot_use_mm_for_euclid_dist")  # [B, N, M]
        B, N, M = map_dist.shape

        map_mask = map_mask.unsqueeze(1).expand(B, N, M)
        map_dist[~map_mask] = +100000
        selected_map_distance, selected_map = map_dist.min(-1)
        selected_map = selected_map.reshape(B, N, 1, 1)

        # Remove objects that are too far from any existing map feature.
        actor_is_closed = selected_map_distance <= 5
        actor_valid_mask = torch.logical_and(raw_actor_valid_mask, actor_is_closed)

        actor_type = data_dict["decoder/actor_type"]

        selected_map_pos, selected_map_heading, pos_dist, vel_dist, head_dist, size_dist = \
            get_distributions_for_training(data_dict, forward_ret_dict, selected_map, actor_type)

        local_actor_position = actor_position - selected_map_pos
        local_actor_position = utils.absolute_to_relative(local_actor_position, selected_map_heading)
        # local_actor_position = local_actor_position / constants.LOCAL_POSITION_XY_RANGE
        # local_actor_position = local_actor_position
        init_pos_loss = -pos_dist.log_prob(local_actor_position[..., :2])  # THE TARGET IS NORMALIZED
        init_pos_loss = (torch.sum(init_pos_loss * actor_valid_mask, dim=1) / actor_valid_mask.sum(-1).clamp(1)).mean()

        local_actor_velocity = data_dict["encoder/agent_feature"][:, 0, :, 4:6]
        # local_actor_velocity = data_dict["encoder/agent_feature"][:, 0, :, 4:6] * constants.VELOCITY_XY_RANGE

        local_actor_velocity = utils.absolute_to_relative(local_actor_velocity, selected_map_heading)

        # local_actor_velocity = local_actor_velocity / constants.VELOCITY_XY_RANGE
        local_actor_velocity = local_actor_velocity

        init_vel_loss = -vel_dist.log_prob(local_actor_velocity)  # THE TARGET IS NORMALIZED
        init_vel_loss = (torch.sum(init_vel_loss * actor_valid_mask, dim=1) / actor_valid_mask.sum(-1).clamp(1)).mean()

        # head_tar = data_dict["encoder/agent_feature"][:, 0, :, 3] * constants.HEADING_RANGE
        head_tar = data_dict["encoder/agent_feature"][:, 0, :, 3]

        # Pred Head + Map Head = GT Head -> Pred Head = GT Head - Map Head
        head_tar = utils.wrap_to_pi(head_tar - selected_map_heading)

        # head_tar = head_tar / constants.HEADING_RANGE
        head_tar = head_tar

        init_head_loss = -head_dist.log_prob(head_tar)  # THE TARGET IS NORMALIZED
        init_head_loss = (torch.sum(init_head_loss * actor_valid_mask, dim=1) /
                          actor_valid_mask.sum(-1).clamp(1)).mean()

        size_tar = (data_dict["encoder/agent_feature"][:, 0, :, 6:9])
        init_size_loss = -size_dist.log_prob(size_tar)  # THE TARGET IS NORMALIZED
        init_size_loss = (torch.sum(init_size_loss * actor_valid_mask, dim=1) /
                          actor_valid_mask.sum(-1).clamp(1)).mean()

        # ===== Score Loss =====
        pred_score = forward_ret_dict["fake_map_feat_score"].squeeze(-1)  # [B, M]

        # map_feat_score_loss = F.binary_cross_entropy_with_logits(input=pred_score, target=is_lane_mask.float(), reduction="none")

        map_feat_score_loss_total = []
        for type_int in range(constants.NUM_TYPES):

            gt_score = pred_score.new_zeros([B, M])
            for i in range(B):
                m = actor_valid_mask[i]
                m = torch.logical_and(m, actor_type[i] == type_int + 1)
                gt_score[i].index_fill_(dim=0, index=selected_map[i].reshape(-1)[m], value=1)

            map_feat_score_loss = F.binary_cross_entropy_with_logits(
                input=pred_score[..., type_int], target=gt_score, reduction='none'
            )
            original_map_mask = forward_ret_dict["encoder/map_valid_mask"]
            map_feat_score_loss = (
                torch.sum(map_feat_score_loss.nan_to_num() * original_map_mask, dim=1) /
                original_map_mask.sum(-1).clamp(1)
            ).mean()
            map_feat_score_loss_total.append(map_feat_score_loss)

        # ===== Actor Type Loss =====
        # type_mask = forward_ret_dict["compress_actor_valid_mask"].unsqueeze(-1).expand(B, compress_T, N, num_modes)
        nearest_type_logit = torch.gather(
            forward_ret_dict["fake_actor_type"],
            dim=1,
            index=selected_map.squeeze(-1).expand(B, N, *forward_ret_dict["fake_actor_type"].shape[2:])
        )
        gt_actor_type = gt_dict["decoder/actor_type"].reshape(B, N)
        type_loss = F.cross_entropy(
            input=nearest_type_logit.reshape(-1, 5), target=gt_actor_type.reshape(-1), reduction="none"
        )
        type_loss = type_loss.reshape_as(gt_actor_type)
        type_loss = (torch.sum(type_loss * actor_valid_mask, dim=1) / actor_valid_mask.sum(-1).clamp(1)).mean()

        # ===== Total Loss =====
        total_loss = init_pos_loss + init_vel_loss + init_head_loss + init_size_loss + sum(
            map_feat_score_loss_total
        ) + type_loss
        tb_dict = dict(
            total_loss=total_loss.item(),
            # init_actor_loss=total_loss.item(),
            init_pos_loss=init_pos_loss.item(),
            init_vel_loss=init_vel_loss.item(),
            init_head_loss=init_head_loss.item(),
            init_size_loss=init_size_loss.item(),
            map_feat_score_loss=sum(map_feat_score_loss_total).item(),
            **{"map_feat_score_loss_type{}".format(i): v.item()
               for i, v in enumerate(map_feat_score_loss_total)},
            init_actor_type_loss=type_loss.item(),
            actor_valid_mask_raw=raw_actor_valid_mask.sum(-1).float().mean().item(),
            actor_valid_mask_to_model=forward_ret_dict["first_step_actor_valid_mask"].sum(-1).float().mean().item(),
            actor_valid_mask_to_loss=actor_valid_mask.sum(-1).float().mean().item(),
        )
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()
        return total_loss, tb_dict, tb_dict

    def forward(self, input_dict):
        actor_feature = input_dict["encoder/agent_feature"]
        actor_valid_mask = input_dict["encoder/agent_valid_mask"]
        actor_position = input_dict["encoder/agent_position"]
        # B, T, N, D_actor = actor_feature.shape
        # assert actor_feature.shape[:3] == actor_position.shape[:3]

        in_evaluation = input_dict.get("in_evaluation", False)

        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_token_valid_mask = map_valid_mask.sum(axis=-1) != 0
        map_position = input_dict["encoder/map_position"]
        B = map_position.shape[0]

        if actor_feature is not None:
            first_step_actor_valid_mask = actor_valid_mask[:, 0]
            first_step_actor_feature = actor_feature[:, 0]
            first_step_actor_position = actor_position[:, 0]
            _, N, _ = first_step_actor_feature.shape

            if not in_evaluation:  # TODO: Use a config to control this
                selected_num = torch.minimum(
                    (first_step_actor_valid_mask.sum(-1) * torch.rand(B, device=actor_valid_mask.device)).int(),
                    first_step_actor_valid_mask.sum(-1)
                ).clamp(0)  # [B, ]
                keep_mask = map_valid_mask.new_zeros((B, N))
                for i in range(B):
                    st_valids = first_step_actor_valid_mask[i].nonzero()[:, 0]
                    st_ind = torch.randperm(len(st_valids))[:selected_num[i]]
                    st_selected_ind = st_valids[st_ind]
                    keep_mask[i, st_selected_ind] = 1

                before = first_step_actor_valid_mask.sum()
                first_step_actor_valid_mask = torch.logical_and(first_step_actor_valid_mask, keep_mask)
                after = first_step_actor_valid_mask.sum()

            # TODO: In future, we can try to add some "empty token" to tell layers where there is a car but masked out.

            agent_enc = self.actor_mlps(first_step_actor_feature[first_step_actor_valid_mask])
            # agent_enc += self.actor_type_pe(input_dict["decoder/actor_type"][first_step_actor_valid_mask])
            agent_enc = utils.unwrap(agent_enc, first_step_actor_valid_mask)

        else:
            agent_enc = None
            first_step_actor_valid_mask = None
            first_step_actor_position = None

        map_token = self.map_polyline_encoder(map_feature, map_valid_mask)
        # map_type = input_dict["map_feature_type"]
        # map_type[map_type == -1] = 0
        # map_token += self.map_feature_type_pe(map_type)

        traffic_light_feature = input_dict["encoder/traffic_light_feature"][:, 0]
        traffic_light_position = input_dict["encoder/traffic_light_position"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"][:, 0]
        _, L, D_light = traffic_light_feature.shape
        # [B, T, L] -> [B, compress_T, L]
        # compress_light_mask = traffic_light_valid_mask[:, :compress_T * self.compress_step].reshape(
        #     B, compress_T, self.compress_step, L).any(dim=2).clone()

        # [B, T, num light, token dim]
        if L != 0:
            light_token = self.light_mlps(traffic_light_feature[traffic_light_valid_mask])
            # light_token += PE  # TODO: Can add PE for traffic light type.
            light_token = utils.unwrap(light_token, traffic_light_valid_mask)
        else:
            light_token = None

        output_map_token, output_actor_token, output_light_token = self.encoder(
            map_token=map_token,
            actor_token=agent_enc,
            map_position=map_position,
            actor_position=first_step_actor_position,
            map_valid_mask=map_token_valid_mask,
            actor_valid_mask=first_step_actor_valid_mask,
            light_token=light_token if L > 0 else None,
            light_position=traffic_light_position if L > 0 else None,
            light_valid_mask=traffic_light_valid_mask if L > 0 else None
        )
        feature = output_map_token

        # Note: The map feature here includes all types. But when computing prediction, we will only select the lane
        # map feature.
        pred_pos, pred_vel, pred_head, pred_size, map_feat_score, actor_type, pos_dist, vel_dist, head_dist, size_dist = \
            self.start_token_predictor(feature, map_token_valid_mask)

        original_map_mask = input_dict["encoder/map_feature_valid_mask"].sum(axis=-1) != 0
        map_mask = original_map_mask  # TODO: We can filter out some useless map feature avoid them be the anchor.

        return {
            "fake_position": pred_pos,
            "fake_velocity": pred_vel,
            "fake_heading": pred_head,
            "fake_size": pred_size,
            "fake_map_feat_score": map_feat_score,
            "fake_actor_type": actor_type,
            "fake_position_dist": pos_dist,
            "fake_velocity_dist": vel_dist,
            "fake_heading_dist": head_dist,
            "fake_size_dist": size_dist,
            "map_mask_for_initializer": map_mask,
            "encoder/map_valid_mask": original_map_mask,
            "first_step_actor_valid_mask": first_step_actor_valid_mask,
        }

    def autoregressive_generate(
        self,
        data_dict,
        num_v,
        temperature=1.0,
        sampling_method="softmax",
        topk=10,
        topp=0.9,
        deterministic_state=False,
        use_nature_probability=False,
        record_intermediate_model_output=False,
        condition_on_sdc=False
    ):
        input_dict, gt_dict = data_dict
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        B = map_valid_mask.shape[0]

        if condition_on_sdc:
            sdc_index = gt_dict["sdc_index"]  # [B, ]
            sdc_index = sdc_index.reshape(B, 1, 1)

            actor_feature = torch.gather(
                input_dict["encoder/agent_feature"][:, 0],  # [B, N, D]
                index=sdc_index.expand(B, 1, input_dict["encoder/agent_feature"].shape[-1]),  # [B, 1, D]
                dim=1
            ).unsqueeze(1)  # -> [B, 1, 1, D]
            actor_feature_list = [actor_feature.reshape(B, input_dict["encoder/agent_feature"].shape[-1])]

            actor_position = torch.gather(
                input_dict["encoder/agent_position"][:, 0],  # [B, N, 3]
                index=sdc_index.expand(B, 1, input_dict["encoder/agent_position"].shape[-1]),  # [B, 1, 3]
                dim=1
            ).unsqueeze(1)  # -> [B, 1, 1, 3]
            actor_position_list = [actor_position.reshape(B, 3)]

            actor_valid_mask = torch.gather(
                input_dict["encoder/agent_valid_mask"][:, 0],  # [B, N]
                index=sdc_index.reshape(B, 1),  # [B, 1]
                dim=1
            ).unsqueeze(1)  # [B, 1, 1]
            actor_valid_mask_list = [actor_valid_mask.reshape(B, )]

            assert num_v > 1, num_v
            num_v = num_v - 1

        else:
            actor_feature = None
            actor_position = None
            actor_valid_mask = None
            actor_valid_mask_list = []
            actor_feature_list = []
            actor_position_list = []

        intermediate_model_output = []

        return_sampled_dict = defaultdict(list)

        # TODO: In future we should support user-specified different actor type
        gt_actor_type = input_dict["decoder/actor_type"].clone()
        if condition_on_sdc:
            # The actor type of known actors
            actor_type = torch.gather(
                gt_actor_type,  # [B, N, 3]
                index=sdc_index.reshape(B, 1),  # [B, 1, 3]
                dim=1
            )  # [B, 1]
            actor_type_list = [actor_type.reshape(B, )]

            new_gt_actor_type = []
            for i in range(B):
                N = len(gt_actor_type[i])
                ind = gt_actor_type.new_ones(N, dtype=bool)
                ind[sdc_index[i]] = 0
                new_gt_actor_type.append(gt_actor_type[i][ind])
            new_gt_actor_type = torch.stack(new_gt_actor_type, dim=0)
            gt_actor_type = new_gt_actor_type

        else:
            actor_type = None
            actor_type_list = []

        new_gt_actor_type = torch.zeros_like(gt_actor_type)
        for i in range(B):
            valid_actor_type = gt_actor_type[i][gt_actor_type[i] != -1]
            new_gt_actor_type[i, :len(valid_actor_type)] = valid_actor_type
        gt_actor_type = new_gt_actor_type

        actor_valid_sum = input_dict["encoder/agent_valid_mask"][:, 0].sum(-1)  # [B, ]
        intersection_count = 0

        for i in range(num_v):
            input_dict_tmp = {
                "encoder/map_feature": input_dict["encoder/map_feature"],
                "encoder/map_feature_valid_mask": input_dict["encoder/map_feature_valid_mask"],
                "encoder/map_position": input_dict["encoder/map_position"],
                "encoder/map_heading": input_dict["encoder/map_heading"],
                # "map_feature_type": input_dict["map_feature_type"],
                "encoder/traffic_light_feature": input_dict["encoder/traffic_light_feature"],
                "encoder/traffic_light_position": input_dict["encoder/traffic_light_position"],
                "encoder/traffic_light_valid_mask": input_dict["encoder/traffic_light_valid_mask"],

                # These data should be updated
                "encoder/agent_feature": actor_feature,
                "encoder/agent_valid_mask": actor_valid_mask,
                "encoder/agent_position": actor_position,
                "decoder/actor_type": actor_type,
                "in_evaluation": True,
            }
            out = self(input_dict_tmp)

            if record_intermediate_model_output:
                intermediate_model_output.append(out)

            # Do the sampling and create new token here
            trial = 0
            while trial < 5:
                new_token, sampled_dict = sample_new_actor(
                    data_dict=input_dict_tmp,
                    model_output=out,
                    sampling_method=sampling_method,
                    actor_type=gt_actor_type[:, i],
                    temperature=temperature,
                    topk=topk,
                    topp=topp,
                    deterministic_state=deterministic_state,
                    use_nature_probability=use_nature_probability
                )
                intersect = if_intersection(sampled_dict, actor_position, actor_feature, actor_valid_mask)

                if actor_position is not None and intersect.any():
                    # print(f"Trial {trial} Find {intersect.sum()} intersection")
                    intersection_count += intersect.sum()
                    trial += 1
                else:
                    break

            for k, v in sampled_dict.items():
                return_sampled_dict[k].append(v)

            # Fill up information here
            actor_feature_list.append(new_token)  # [B, D]
            actor_feature = torch.stack(actor_feature_list, dim=1).unsqueeze(1)  # -> B, 1, N, 16
            actor_position_list.append(sampled_dict["sampled_pos"])
            actor_position = torch.stack(actor_position_list, dim=1).unsqueeze(1)  # -> B, 1, N, 3

            actor_type_list.append(gt_actor_type[:, i])
            actor_type = torch.stack(actor_type_list, dim=1)

            # actor_valid_mask = map_valid_mask.new_ones(actor_feature.shape[:3]) if actor_feature is not None else None
            actor_valid_mask_list.append(i < actor_valid_sum)  # [B, ]
            actor_valid_mask = torch.stack(actor_valid_mask_list, dim=1).unsqueeze(1)  # -> B, 1, N

        return_sampled_dict = {k: torch.stack(v, dim=1) for k, v in return_sampled_dict.items()}
        return_sampled_dict.update(
            {
                "encoder/agent_feature": actor_feature,
                "encoder/agent_valid_mask": actor_valid_mask,
                "encoder/agent_position": actor_position,
                "intersection_count": intersection_count / B,
                "total_count": actor_valid_sum.sum().item() / B
            }
        )

        if record_intermediate_model_output:
            return_sampled_dict["intermediate_model_output"] = intermediate_model_output
        return return_sampled_dict
