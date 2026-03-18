import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scenestreamer.dataset.constants import AGENT_STATE_DIM, MAP_FEATURE_STATE_DIM, TRAFFIC_LIGHT_STATE_DIM, \
    TRAFFIC_LIGHT_PREDICT_DIM, VELOCITY_XY_RANGE, HEADING_RANGE, POSITION_XY_RANGE
from scenestreamer.models.layers import common_layers, polyline_encoder, our_decoder_layer, position_encoding_utils
from scenestreamer.models.ops.search_knn_indices import search_k_nearest_object_indices, \
    search_k_nearest_map_feature_indicies, \
    search_k_nearest_map_feature_indicies_for_map
from scenestreamer.utils import wrap_to_pi, rotate

NUM_TYPES = 3


class ActorPredictor(nn.Module):
    """
    input: output token of a given actor from transformer (might from each layer or from the last layer)
    output: predicted state of each actor
    """
    def __init__(self, d_model, num_modes, step_per_token, small, use_gaussian=False):
        super().__init__()

        self.d_model = d_model
        self.use_gaussian = use_gaussian
        self.num_modes = num_modes
        self.step_per_token = step_per_token
        self.head_input_dim = self.d_model * (1 if small else 4)

        self.extra = not small

        if small:
            self.actor_mlps_decompress = common_layers.build_mlps(
                c_in=self.d_model,
                mlp_channels=[self.d_model, self.d_model],
                ret_before_act=False,
            )
            self.position_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[
                    6 * self.step_per_token * self.num_modes if use_gaussian else 3 * self.step_per_token *
                    self.num_modes * NUM_TYPES
                ],
                ret_before_act=True,
            )
            self.score_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[self.num_modes * NUM_TYPES],
                ret_before_act=True,
            )
        else:
            self.actor_mlps_decompress = common_layers.build_mlps(
                c_in=self.d_model,
                mlp_channels=[self.d_model * 4, self.d_model * 4],
                ret_before_act=False,
            )
            self.position_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[
                    6 * self.num_modes * self.step_per_token if use_gaussian else 3 * self.num_modes *
                    self.step_per_token * NUM_TYPES
                ],
                ret_before_act=True,
            )
            self.score_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[self.num_modes * NUM_TYPES],
                ret_before_act=True,
            )
        if self.extra:
            self.velocity_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[
                    4 * self.num_modes * self.step_per_token if use_gaussian else 2 * self.num_modes *
                    self.step_per_token * NUM_TYPES
                ],
                ret_before_act=True,
            )
            self.heading_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[
                    2 * self.num_modes * self.step_per_token if use_gaussian else 1 * self.num_modes *
                    self.step_per_token * NUM_TYPES
                ],
                ret_before_act=True,
            )

            # TODO: This is a little weird, remove this.
            self.actor_type_head = common_layers.build_mlps(
                c_in=self.d_model * (1 if small else 4),
                mlp_channels=[5 * self.num_modes],
                ret_before_act=True,
            )

    def forward(self, actor_tokens, actor_valid_mask, step_per_token, actor_type):
        B, compress_T, N, token_dim = actor_tokens.shape
        actor_prediction_feat = self.actor_mlps_decompress(actor_tokens[actor_valid_mask])

        num_modes = self.num_modes

        actor_type = actor_type.clone()
        actor_type[(actor_type < 1) | (actor_type > 3)] = 3
        actor_type = actor_type - 1

        # Get predicted position in shape: [B, compress_T, N, num_modes, step_per_token, 2/4]
        pred_pos = unwrap(self.position_head(actor_prediction_feat), actor_valid_mask)
        pred_pos = pred_pos.reshape(
            B, compress_T, N, NUM_TYPES, num_modes, step_per_token, (6 if self.use_gaussian else 3)
        )
        pred_pos = torch.gather(
            pred_pos,
            index=actor_type.reshape(B, 1, N, 1, 1, 1, 1).expand(
                B, compress_T, N, 1, num_modes, step_per_token, (6 if self.use_gaussian else 3)
            ),
            dim=3
        ).squeeze(3)

        # Get predicted trajectory score in shape: [B, compress_T, N, num_modes]
        score_prediction_logit = self.score_head(actor_prediction_feat)
        score_prediction_logit = score_prediction_logit.reshape(score_prediction_logit.shape[0], NUM_TYPES, -1)
        pred_score = score_prediction_logit.new_zeros(B, compress_T, N, NUM_TYPES, num_modes)
        pred_score.fill_(float("-inf"))
        pred_score[actor_valid_mask] = score_prediction_logit
        pred_score = torch.gather(
            pred_score, index=actor_type.reshape(B, 1, N, 1, 1).expand(B, compress_T, N, 1, num_modes), dim=3
        ).squeeze(3)
        # pred_score = F.log_softmax(pred_score, dim=-1)  # PZH 0531: Strange, why we have logsoftmax here?
        # Note that there are some nan in pred_score!

        if not self.extra:
            return pred_pos, pred_score

        # Get predicted heading in shape: [B, compress_T, N, num_modes, step_per_token, 1/2]
        pred_heading = unwrap(self.heading_head(actor_prediction_feat), actor_valid_mask)
        pred_heading = pred_heading.reshape(
            B, compress_T, N, NUM_TYPES, num_modes, step_per_token, (2 if self.use_gaussian else 1)
        )
        pred_heading = torch.gather(
            pred_heading,
            index=actor_type.reshape(B, 1, N, 1, 1, 1, 1).expand(
                B, compress_T, N, 1, num_modes, step_per_token, (2 if self.use_gaussian else 1)
            ),
            dim=3
        ).squeeze(3)

        # Get predicted velocity in shape: [B, compress_T, N, num_modes, step_per_token, 2/4]
        pred_velocity = unwrap(self.velocity_head(actor_prediction_feat), actor_valid_mask)
        pred_velocity = pred_velocity.reshape(
            B, compress_T, N, NUM_TYPES, num_modes, step_per_token, (4 if self.use_gaussian else 2)
        )
        pred_velocity = torch.gather(
            pred_velocity,
            index=actor_type.reshape(B, 1, N, 1, 1, 1, 1).expand(
                B, compress_T, N, 1, num_modes, step_per_token, (4 if self.use_gaussian else 2)
            ),
            dim=3
        ).squeeze(3)

        # Get predicted vehicle type in shape: [B, compress_T, N, step_per_token, 5]
        pred_actor_type = unwrap(self.actor_type_head(actor_prediction_feat), actor_valid_mask)
        pred_actor_type = pred_actor_type.reshape(B, compress_T, N, num_modes, 5)

        return pred_pos, pred_velocity, pred_heading, pred_actor_type, pred_score


class QueryPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.map_pe = nn.Embedding(2000, d_model)
        self.actor_pe = nn.Embedding(500, d_model)
        self.light_pe = nn.Embedding(500, d_model)
        self.time_pe = nn.Embedding(500, d_model)
        self.d_model = d_model

        max_seq_len = 2000
        # Initialize the positional encoding matrix
        pos_enc = torch.zeros(max_seq_len, d_model)

        # Compute the positional encodings
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000**((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000**((2 * (i + 1)) / d_model)))

        # pos_enc = pos_enc.unsqueeze(0)  # Add a batch dimension
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, map_token, actor_token, light_token):
        B, T, N, _ = actor_token.shape
        _, _, L, _ = light_token.shape

        # Apply position embeddings for map features
        map_pos = torch.arange(map_token.size(1), device=map_token.device)
        map_pos_emb = self.map_pe(map_pos)
        map_pos_emb += self.pos_enc[:map_token.size(1)]
        map_pos_emb = map_pos_emb.unsqueeze(0)

        map_token_emb = map_token + map_pos_emb

        # Apply position and time embeddings for actors and traffic lights
        time_pos = torch.arange(T, device=actor_token.device)
        time_pos_emb = self.time_pe(time_pos)
        time_pos_emb += self.pos_enc[:T]
        time_pos_emb = time_pos_emb.reshape(1, T, 1, self.d_model)

        # Actors (cars)
        actor_pos = torch.arange(N, device=actor_token.device)
        actor_pos_emb = self.actor_pe(actor_pos)
        actor_pos_emb += self.pos_enc[:N]
        actor_pos_emb = actor_pos_emb.reshape(1, 1, N, self.d_model)
        actor_token_emb = actor_token + actor_pos_emb + time_pos_emb

        # Traffic lights
        if L > 0:
            light_pos = torch.arange(L, device=light_token.device)
            light_pos_emb = self.light_pe(light_pos)
            light_pos_emb += self.pos_enc[:L]
            light_pos_emb = light_pos_emb.reshape(1, 1, L, self.d_model)
            light_token_emb = light_token + light_pos_emb + time_pos_emb
        else:
            light_token_emb = light_token

        return map_token_emb, actor_token_emb, light_token_emb


class MotionLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Set up config
        self.model_cfg = config
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_ATTN_LAYERS
        self.compress_step = self.model_cfg.INPUT_STEP_PER_TOKEN
        self.step_per_token = self.model_cfg.PREDICT_STEP_PER_TOKEN
        # self.num_modes = self.config.NUM_MOTION_MODES
        # use_gaussian = self.config.USE_GAUSSIAN
        # hidden_size = self.d_model
        # self.discrete = self.config.DISCRETE

        # ========== A general encoder of everything: map features & obj features ==========

        # Allow three types of starting token, so that user can control which actor type to create.
        # self.start_token = nn.Embedding(5, self.d_model)
        # self.start_token_pe = nn.Embedding(500, self.d_model)
        # self.start_token_pe_for_empty = nn.Embedding(1, self.d_model)

        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=MAP_FEATURE_STATE_DIM, hidden_dim=64, num_layers=5, num_pre_layers=3, out_channels=self.d_model
        )
        self.actor_mlps = common_layers.build_mlps(
            c_in=AGENT_STATE_DIM,
            mlp_channels=[self.d_model] * 2,
            ret_before_act=True,
        )
        self.light_mlps = common_layers.build_mlps(
            c_in=TRAFFIC_LIGHT_STATE_DIM, mlp_channels=[self.d_model] * 2, ret_before_act=True, without_norm=True
        )
        self.actor_mlps_compress = common_layers.build_mlps(
            c_in=self.d_model * self.compress_step,
            mlp_channels=[self.d_model * 4, self.d_model * 4, self.d_model],
            ret_before_act=True,
        )
        self.light_mlps_compress = common_layers.build_mlps(
            c_in=self.d_model * self.compress_step,
            mlp_channels=[self.d_model * 2, self.d_model * 2, self.d_model],
            ret_before_act=True,
            without_norm=True
        )
        self.decoder_tokenizer = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model, self.d_model],
            ret_before_act=True,
        )
        self.decoder_layers = self.build_transformer_decoder(
            d_model=self.d_model,  # 256
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.DROPOUT_OF_ATTN,
            num_decoder_layers=self.model_cfg.NUM_ATTN_LAYERS,
            use_local_attn=True
        )

        self.pe = QueryPE(d_model=self.d_model)

        self.actor_predictor = ActorPredictor(
            d_model=self.d_model,
            num_modes=self.num_modes,
            step_per_token=self.step_per_token,
            use_gaussian=use_gaussian,
            small=False
        )

        self.traffic_light_predictor = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model, self.d_model * 2, TRAFFIC_LIGHT_PREDICT_DIM * self.step_per_token],
            ret_before_act=True,
            without_norm=True
        )

    def build_transformer_decoder(self, d_model, nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        decoder_layer_1 = our_decoder_layer.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            relative_pe=self.model_cfg.RELATIVE_PE,
            dim_feedforward=d_model * 4,
            # dim_feedforward=d_model,
            dropout=dropout,
            activation="gelu",
            normalize_before=False,
            keep_query_pos=False,  # Not using Query Position (but use for the first decoder layer)
            rm_self_attn_decoder=False,
            use_local_attn=use_local_attn,
            is_first=True
        )
        decoder_layer = our_decoder_layer.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            relative_pe=self.model_cfg.RELATIVE_PE,
            dim_feedforward=d_model * 4,
            # dim_feedforward=d_model,
            dropout=dropout,
            activation="gelu",
            normalize_before=False,
            keep_query_pos=False,  # Not using Query Position (but use for the first decoder layer)
            rm_self_attn_decoder=False,
            use_local_attn=use_local_attn,
        )
        decoder_layers = nn.ModuleList(
            [decoder_layer_1] + [copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers - 1)]
        )
        return decoder_layers

    def apply_transformer_decoder(
        self,
        *,
        all_token,
        all_token_valid_mask,
        all_position,
        all_token_valid_mask_without_map,
        # all_token_valid_mask_without_start_token,
        all_position_without_map,
        actor_token,
        actor_position,
        actor_valid_mask,
        stacked_actor_valid_mask,
        # start_token_pe,
        # st_drop_mask,
        # start_token_valid_mask,
        traffic_light_token,
        traffic_light_position,
        traffic_light_valid_mask,
        map_token,
        map_position,
        map_valid_mask,
        anchor_heading,
        anchor_velocity,
        anchor_position,
        query_cache=None,
        in_evaluation=False,
    ):
        """
        Notations:

        B - Batch size
        T - Total time steps of the trajectory, does not consider which are history and which are for prediction.
        M - Number of map features
        N - Number of actors
        L - number of traffic lights
        """
        B, num_total_tokens, d_model = all_token.shape
        _, compress_T, N, _ = actor_token.shape
        _, _, L, _ = traffic_light_token.shape
        _, M, _ = map_token.shape

        T = compress_T

        H = self.model_cfg.HISTORY_TOKENS

        num_of_neighbors_traffic_light = 10
        num_of_neighbors_actor = 20
        num_of_neighbors_map = 128
        num_of_neighbors_map_for_map = 128

        # New output size: [num valid in all_position_without_map, K]
        # Fall in range [0, N]
        neighbor_index_actor = search_k_nearest_object_indices(
            ego_position_full=all_position_without_map,  # [B, T, N+L, 2]
            ego_valid_mask=all_token_valid_mask_without_map,
            neighbor_position_full=actor_position,  # [B, T, N, 2]
            neighbor_valid_mask=actor_valid_mask,
            num_neighbors=num_of_neighbors_actor
        )

        # New output size: [B, T, N+L, K]
        # Fall in range [0, L]
        if L == 0:
            neighbor_index_traffic_light = neighbor_index_actor.new_zeros([B, T, N + L, num_of_neighbors_traffic_light])
            neighbor_index_traffic_light.fill_(-1)
        else:
            neighbor_index_traffic_light = search_k_nearest_object_indices(
                ego_position_full=all_position_without_map,  # [B, T, N+L, 2]
                ego_valid_mask=all_token_valid_mask_without_map,
                neighbor_position_full=traffic_light_position,  # [B, T, L, 2]
                neighbor_valid_mask=traffic_light_valid_mask,
                num_neighbors=num_of_neighbors_traffic_light
            )
            neighbor_index_traffic_light[neighbor_index_traffic_light != -1] += N

        # Fall in range [0, num valid map feat]
        neighbor_index_map = search_k_nearest_map_feature_indicies(
            ego_position_full=all_position_without_map,  # [B, T, N+L, 2]
            ego_valid_mask=all_token_valid_mask_without_map,
            neighbor_position_full=map_position,  # [B, M, 2]
            neighbor_valid_mask=map_valid_mask,  # [B, M]
            num_neighbors=num_of_neighbors_map
        )

        # Fall in range [0, num valid map feat]
        map_neighbor = search_k_nearest_map_feature_indicies_for_map(
            ego_position_full=map_position,  # [B, M, 2]
            ego_valid_mask=map_valid_mask,
            num_neighbors=num_of_neighbors_map_for_map
        )

        # PZH NOTE: At this moment, we already collected 4 set of neighbor indices.
        # Each of them are fallen into different domain, which is: the number of valid "neighbors" at given time step.
        # Now, we need to convert the domain from "per-step neighbors" to the "whole input sequence".

        # Number of valid map feats in each batch
        num_valid_maps = map_valid_mask.sum(-1, keepdims=True).repeat(1, T)  # [B, T]
        num_valid_actors = F.pad(actor_valid_mask.sum(-1).cumsum(-1), (1, 0))  # [B, T+1]
        num_valid_lights = F.pad(traffic_light_valid_mask.sum(-1).cumsum(-1), (1, 0))  # [B, T+1]

        # ST: Offset the number of valid start token
        # num_valid_start_tokens = start_token_valid_mask.sum(-1, keepdims=True).repeat(1, T)

        # We can build a mapping in shape [B, T, N+L] mapping every box-index to flatten-index.
        mapping_to_flatten = num_valid_maps.new_zeros([B, T, N + L])
        mapping_to_flatten += (num_valid_maps + num_valid_actors[:, :T] + num_valid_lights[:, :T])[..., None]
        # Till now, the row (b, t) is filled with the "number of tokens" before (b, t).
        # But for the token in the row [b, t, :], we still need to offset them one by one.

        # Now add the "in row offset":
        mapping_to_flatten[..., :N] += F.pad(actor_valid_mask.cumsum(-1), (1, 0))[..., :N]

        mapping_to_flatten[..., :N][~actor_valid_mask] = -1
        mapping_to_flatten[...,
                           N:] += actor_valid_mask.sum(-1)[...,
                                                           None] + F.pad(traffic_light_valid_mask.cumsum(-1),
                                                                         (1, 0))[..., :L]
        mapping_to_flatten[..., N:][~traffic_light_valid_mask] = -1

        neighbor_index_object = torch.cat([neighbor_index_actor, neighbor_index_traffic_light], dim=-1)

        max_token_to_attend = H * neighbor_index_object.shape[-1]

        neighbor_index_per_obj = neighbor_index_object.new_empty([B, T, N + L, max_token_to_attend])
        neighbor_index_per_obj.fill_(-1)
        history_ind = torch.arange(H)
        for t in range(T):
            selected_t_dim = history_ind[:t + 1] + max(0, t - H)
            selected_length = min(t + 1, H)

            # Shape: [B, selected_length, N+L, N+L]
            key = mapping_to_flatten[:, selected_t_dim].reshape(B, selected_length, 1, N + L).repeat(1, 1, N + L, 1)

            query = neighbor_index_object[:, t].reshape(B, 1, N + L, -1).repeat(1, selected_length, 1, 1).long()

            query_mask = query == -1

            query[query_mask] = 0

            flatten_index = torch.gather(key, index=query, dim=3)

            flatten_index[query_mask] = -1

            # [B, selected_length, N, K] -> [B, N, K * selected_length]
            flatten_index = flatten_index.permute(0, 2, 3, 1).flatten(2, 3).reshape(B, N + L, -1)

            neighbor_index_per_obj[:, t, :, :flatten_index.shape[-1]] = flatten_index

        assert (neighbor_index_map.max(-1)[0] < map_valid_mask.sum(-1)[:, None, None]).all()
        assert neighbor_index_per_obj[neighbor_index_per_obj != -1].min() >= map_valid_mask.sum(-1).min()
        assert (map_neighbor.max(-1)[0].max(-1)[0] < map_valid_mask.sum(-1)).all()

        # -> [B, T, N, K_map + K_obj]
        neighbor_index_per_obj = torch.cat([neighbor_index_map, neighbor_index_per_obj], dim=-1)

        max_token_to_attend = max(max_token_to_attend, neighbor_index_per_obj.shape[-1])
        max_token_to_attend = max(max_token_to_attend, N + num_of_neighbors_map)

        neighbor_index_per_obj = neighbor_index_per_obj.reshape(B, T * (N + L), -1)
        if neighbor_index_per_obj.shape[-1] < max_token_to_attend:
            neighbor_index_per_obj = F.pad(
                neighbor_index_per_obj, (0, max_token_to_attend - neighbor_index_per_obj.shape[-1]),
                mode="constant",
                value=-1
            )

        if map_neighbor.shape[-1] < max_token_to_attend:
            # now neighbor_index_per_step is in shape [B, T*(N+L), sum of K]
            map_neighbor = F.pad(
                map_neighbor, (0, max_token_to_attend - map_neighbor.shape[-1]), mode="constant", value=-1
            )

        # valid_map_counts = map_valid_mask.sum(-1).cpu().numpy()
        # neighbor_index_start_token = np.zeros([B, N, max_token_to_attend], dtype=int) - 1

        # causal_mask = np.triu(np.full((N, N), -1), k=1)
        # causal_indices = np.tril(np.arange(N))
        # causal_mask += causal_indices
        # causal_mask = np.repeat(causal_mask[np.newaxis, :, :], B, axis=0)

        # start_token_valid_mask_np = start_token_valid_mask.cpu().numpy()
        # start_token_valid_mask_cumsum_np = F.pad(start_token_valid_mask.cumsum(-1), (1, 0)).cpu().numpy()[:, :-1]
        # start_token_valid_mask_cumsum_np[~start_token_valid_mask_np] = -1
        # start_token_valid_mask_cumsum_np = np.repeat(start_token_valid_mask_cumsum_np[:, np.newaxis, :], N, axis=1)
        # start_token_valid_mask_cumsum_np[~start_token_valid_mask_np] = -1
        # for i in range(B):
        #     m = start_token_valid_mask_cumsum_np[i]
        #     m[m != -1] += valid_map_counts[i]
        #     neighbor_index_start_token[i, :, :N] = m
        #     if valid_map_counts[i] > max_token_to_attend - N:
        #
        #         tar = map_valid_mask[i].nonzero().cpu().numpy().reshape(-1)
        #         for j in range(N):
        #             neighbor_index_start_token[i, j, N:] = np.random.choice(tar, size=(max_token_to_attend - N), replace=False)
        #
        #     else:
        #         neighbor_index_start_token[i, :, N:N+valid_map_counts[i]] = map_valid_mask[i].nonzero().cpu().numpy()[None, :, 0]
        #
        # neighbor_index_start_token = torch.from_numpy(neighbor_index_start_token).to(map_neighbor)

        # rand_map_neighbor = torch.rand((B, N, max_token_to_attend), device=map_valid_mask.device)
        # max_map_feat = map_valid_mask.sum(-1)[..., None, None].expand(B, N, max_token_to_attend)
        # rand_map_neighbor = (rand_map_neighbor * max_map_feat).floor().int()
        # rand_map_neighbor = torch.minimum(rand_map_neighbor, max_map_feat - 1).clamp(0).int()
        # assert (rand_map_neighbor.max(-1)[0].max(-1)[0] < map_valid_mask.sum(-1)).all()

        all_neighbor_index_full = torch.cat([map_neighbor, neighbor_index_per_obj], dim=1)  # [B, M+N+T*(N+L), sum of K]

        assert all_neighbor_index_full.max() < M + N + T * (N + L)
        assert (all_neighbor_index_full.max(dim=-1)[0].max(dim=-1)[0] < all_token_valid_mask.sum(-1)).all()

        batch_index = torch.arange(0, B, device=all_token.device, dtype=torch.int)  # [B,]
        batch_index = batch_index.reshape(B, 1, 1)  # [B, 1, 1]
        batch_index = batch_index.repeat(1, T, N + L)  # [B, T, max_ego_objects]
        batch_index = batch_index.reshape(B, -1)  # [B, T*(N+L)]

        batch_index_map = torch.arange(0, B, device=all_token.device, dtype=torch.int).reshape(B, 1)
        batch_index_map = batch_index_map.repeat(1, M)  # [B, M]

        # batch_index_start_token = torch.arange(0, B, device=all_token.device, dtype=torch.int).reshape(B, 1)
        # batch_index_start_token = batch_index_start_token.repeat(1, N)  # [B, M]

        batch_index = torch.cat([batch_index_map, batch_index], dim=1)  # [B, M+T*(N+L)]

        _, num_keys, _ = all_token.shape
        query_sine_embed = position_encoding_utils.gen_sineembed_for_position(all_position[..., :2], hidden_dim=d_model)

        # query_sine_embed[:, M: M + N][st_drop_mask] = self.start_token_pe_for_empty(
        #     batch_index.new_zeros(query_sine_embed[:, M: M + N][st_drop_mask].shape[:1])
        # )

        # query_sine_embed = torch.cat([
        #     query_sine_embed[:, :M],
        #     # start_token_pe,
        #     query_sine_embed[:, M:]
        # ], dim=1)

        kv_pos_embed_stack = query_sine_embed[all_token_valid_mask]  # [num valid tokens, DIM]

        key_batch_cnt = all_token_valid_mask.sum(-1).int()

        kv_pos_raw = all_position[..., :2][all_token_valid_mask]

        if query_cache:
            last_token_num = query_cache["last_token_num"]  # [B, num total tokens]

        query_cache_list = []
        if in_evaluation:
            # pre-allocate space
            for i in range(self.num_decoder_layers):
                v = torch.zeros_like(all_token)
                if query_cache:
                    v[:, :last_token_num] = query_cache[f"query_cache_{i}"]
                query_cache_list.append(v)

        prediction_list = []

        # all_position_with_start_token = torch.cat([
        #     all_position[:, :M],
        #     all_position.new_zeros([B, N, 2]),
        #     all_position[:, M:]
        # ], dim=1)

        if query_cache:  # Need to have full shape query_feature since we need to slice it later.
            assert in_evaluation
            diff_all_token_valid_mask = all_token_valid_mask[:, last_token_num:]
            query_feature = all_token
            all_neighbor_index = all_neighbor_index_full[:, last_token_num:][diff_all_token_valid_mask]
            index_pair_batch = batch_index[:, last_token_num:][diff_all_token_valid_mask]
            query_pos = all_position[:, last_token_num:][diff_all_token_valid_mask]
            query_sine_embed_stack = query_sine_embed[:, last_token_num:][diff_all_token_valid_mask]

        else:
            query_feature = all_token[all_token_valid_mask]
            # they are share the same first dim size = num valid objects (across time steps and batch)
            all_neighbor_index = all_neighbor_index_full[all_token_valid_mask]
            # (num valid) -> the batch index of each valid object
            index_pair_batch = batch_index[all_token_valid_mask]
            assert len(all_neighbor_index) == len(index_pair_batch)
            assert (all_neighbor_index_full.max(-1)[0].max(-1)[0] < key_batch_cnt).all()
            query_sine_embed_stack = query_sine_embed[all_token_valid_mask]
            query_pos = all_position[all_token_valid_mask]

        for layer_idx in range(self.num_decoder_layers):

            if query_cache:
                kv_feature_stack = query_feature[all_token_valid_mask]
                query_feature = query_feature[:, last_token_num:][diff_all_token_valid_mask]
            else:
                kv_feature_stack = query_feature

            query_feature = self.decoder_layers[layer_idx](
                tgt=query_feature,
                # tgt_valid_mask=diff_all_token_valid_mask,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed_stack,
                memory=kv_feature_stack,
                memory_pos_emb=kv_pos_embed_stack,
                memory_pos=kv_pos_raw,
                is_first=(layer_idx == 0),
                key_batch_cnt=key_batch_cnt,
                index_pair=all_neighbor_index,
                index_pair_batch=index_pair_batch,
            )
            assert query_feature.ndim == 2

            # If using query_cache from previous forward pass, we now need to fill the new query into existing queries.
            # This should be done even if we need to add future embedding into actors' tokens.
            # Everything in the query cache is not added with future embedding.
            if query_cache:
                query_cache_list[layer_idx][:, last_token_num:][diff_all_token_valid_mask] = query_feature.clone()
                query_feature = query_cache_list[layer_idx]
            else:
                if in_evaluation:
                    query_cache_list[layer_idx][all_token_valid_mask] = query_feature.clone()

            if self.model_cfg.LOSS_EACH_LAYER:
                if query_cache:
                    all_output_tokens = query_feature
                else:
                    all_output_tokens = unwrap(query_feature, all_token_valid_mask)
                object_output_tokens = all_output_tokens[:, M + N:]
                object_output_tokens = object_output_tokens.reshape(B, compress_T, N + L, self.d_model)
                actor_output_tokens = object_output_tokens[:, :, :N]
                ret = self.get_prediction_for_actor(
                    anchor_heading=anchor_heading,
                    anchor_velocity=anchor_velocity,
                    anchor_position=anchor_position,
                    actor_output_tokens=actor_output_tokens,
                    actor_valid_mask=actor_valid_mask,
                    stacked_actor_valid_mask=stacked_actor_valid_mask,
                    in_evaluation=in_evaluation if (layer_idx == self.num_decoder_layers - 1) else False,
                    layer_index=layer_idx,
                    actor_type=actor_type
                )
                prediction_list.append(ret)

                if layer_idx < self.num_decoder_layers - 1:
                    embedding = self.get_internal_future_embedding(ret, actor_valid_mask, layer_index=layer_idx)
                    actor_output_tokens += embedding

                all_output_tokens = torch.cat(
                    [
                        all_output_tokens[:, :M + N],
                        torch.cat([actor_output_tokens, object_output_tokens[:, :, N:]], dim=2).flatten(1, 2)
                    ],
                    dim=1
                )
                if query_cache:
                    query_feature = all_output_tokens
                else:
                    query_feature = all_output_tokens[all_token_valid_mask]

        if query_feature.ndim == 3:
            ret = query_feature
        else:
            ret = unwrap(query_feature, all_token_valid_mask)

        ret_dict = {"last_token_num": all_token_valid_mask.shape[1]}
        for i in range(len(query_cache_list)):
            ret_dict[f"query_cache_{i}"] = query_cache_list[i]
        return ret, ret_dict, prediction_list

    def build_our_predict_head(self, hidden_size, num_modes, actor_state_dim, light_state_dim):
        actor_predict_heads = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[hidden_size, hidden_size, actor_state_dim * num_modes],
            ret_before_act=True,
        )
        light_predict_head = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[hidden_size, hidden_size, light_state_dim],
            ret_before_act=True,
        )
        return actor_predict_heads, light_predict_head

    def get_position_loss(self, data_dict, model_output, step, start_time, future_end):
        B, T, N, num_modes, _ = model_output["sampled_position"].shape
        predicted_traj = model_output["sampled_position"][:, start_time:future_end]
        gt_traj = data_dict["encoder/agent_position"][:, start_time:future_end].unsqueeze(3).expand(
            B, future_end - start_time, N, num_modes, 3
        )
        assert predicted_traj.shape == gt_traj.shape

        original_mask = data_dict["encoder/agent_valid_mask"][:, start_time:future_end]
        mask = original_mask.unsqueeze(3).expand(B, future_end - start_time, N, num_modes)
        assert mask.shape == predicted_traj.shape[:4]

        assert gt_traj.shape == predicted_traj.shape
        diff = (gt_traj - predicted_traj).norm(dim=-1)  # [B, T, N, num_modes]

        mode_diff = (diff.sum(1) / mask.sum(1).clamp(1))  # [B, N, num_modes]

        gt_selected = (diff * mask).sum(1).argmin(-1)

        selected = model_output["log_probability"][:, start_time:future_end].sum(1).argmax(-1)  # [B, N]

        best_diff = torch.gather(mode_diff, index=selected.unsqueeze(-1), dim=2).squeeze(-1)  # [B, N]

        acc = (gt_selected == selected)[original_mask.any(1)].float().mean().item()

        return {
            f"eval/best_diff_{step}": best_diff.mean().item(),
            f"eval/avg_diff_{step}": diff[mask].mean().item(),
            f"eval/score_acc_{step}": acc
        }

    def get_actor_loss(
        self,
        *,
        stacked_actor_feat,
        stacked_actor_valid_mask,
        gt_position,
        gt_dict,
        forward_ret_dict,
        anchor_velocity,
        anchor_position,
        anchor_heading,
        layer_index=None
    ):

        stacked_actor_feat = stacked_actor_feat
        gt_position = gt_position

        B, compress_T, N, num_modes, step_per_token, _ = forward_ret_dict["position_logit"].shape

        if num_modes > 1:
            nearest_predicted_dict, per_step_distance = generate_predicted_trajectory_for_training(
                gt_position, forward_ret_dict, use_ade=self.model_cfg.USE_ADE
            )

        # ========== Step 2: Compute loss using the selected trajectory ==========
        pos_target = (stacked_actor_feat[:, 1:, ..., :3] * POSITION_XY_RANGE - anchor_position[:, :-1])
        if self.model_cfg.RELATIVE_POSITION_HEADING:
            pos_target = rotate(
                pos_target[..., 0], pos_target[..., 1], -anchor_heading[:, :-1].squeeze(-1), z=pos_target[..., 2]
            )
        stacked_actor_valid_mask = torch.logical_and(stacked_actor_valid_mask[:, 1:], stacked_actor_valid_mask[:, :-1])

        before_count = stacked_actor_valid_mask.sum()
        if self.model_cfg.USE_SLOW_MASK:
            # rule out those samples where the actor is static.
            # current_actor_valid_mask = torch.logical_and(current_actor_valid_mask, (pos_target != 0).any(-1))
            offset_norm = pos_target.norm(dim=-1)
            slow_mask = offset_norm > 0.01
            stacked_actor_valid_mask = torch.logical_and(stacked_actor_valid_mask, slow_mask)
        after_count = stacked_actor_valid_mask.sum()
        assert stacked_actor_valid_mask.sum() > 0, stacked_actor_valid_mask.sum()

        # ===== Position Loss =====
        if self.model_cfg.USE_NEAREST_LOSS and num_modes > 1:
            position_logit = nearest_predicted_dict["nearest_position_logit"]
        else:
            position_logit = forward_ret_dict["position_logit"]
        assert position_logit.shape[:5] == (
            B, compress_T, N, 1 if self.model_cfg.USE_NEAREST_LOSS else num_modes, step_per_token
        )

        if self.model_cfg.USE_CUMSUM:
            position_logit = position_logit.cumsum(4)

        # Fast actor loss:
        position_logit1 = position_logit[:, :-1][stacked_actor_valid_mask]
        pos_target1 = pos_target[stacked_actor_valid_mask]
        assert position_logit1.shape == pos_target1.shape
        if self.model_cfg.USE_HUBER_LOSS:
            pos_loss = F.huber_loss(input=position_logit1, target=pos_target1)
        else:
            pos_loss = F.mse_loss(input=position_logit1, target=pos_target1)

        # Static actor loss
        # position_logit2 = position_logit[:, :-1][torch.logical_and(stacked_actor_valid_mask, ~slow_mask)]
        # pos_target2 = pos_target[torch.logical_and(stacked_actor_valid_mask, ~slow_mask)]
        # assert position_logit2.shape == pos_target2.shape
        # pos_loss2 = F.huber_loss(input=position_logit2, target=pos_target2)
        #
        # pos_loss = pos_loss1 + pos_loss2 / 5

        if self.model_cfg.SCORE_FORM == "class" and num_modes > 1:
            score_input = forward_ret_dict["score_logit"][forward_ret_dict["compress_actor_valid_mask"]]
            score_target = nearest_predicted_dict["nearest_index_target"][forward_ret_dict["compress_actor_valid_mask"]]
            assert score_input.shape[:-1] == score_target.shape
            score_loss = F.cross_entropy(input=score_input, target=score_target)

        elif self.model_cfg.SCORE_FORM == "reward" and num_modes > 1:
            raise ValueError()
            # with torch.no_grad():
            #     gt_score = 1 / per_step_distance.clamp(1e-3, 1000).detach()
            #     gt_score = gt_score[forward_ret_dict["compress_actor_valid_mask"]]
            #
            # score_input = forward_ret_dict["score_logit"][forward_ret_dict["compress_actor_valid_mask"]]
            #
            # assert score_input.shape == gt_score.shape
            # score_loss = F.huber_loss(
            #     input=score_input,
            #     target=gt_score
            # )

        elif num_modes == 1:
            score_loss = 0.0

        else:
            raise ValueError()

        if self.model_cfg.LOSS_EACH_LAYER and layer_index < self.num_decoder_layers - 1:
            # return since we don't regress other values.
            return dict(
                position_loss=pos_loss,
                velocity_loss=0.0,
                heading_loss=0.0,
                actor_type_loss=0.0,
                score_loss=score_loss,
                slow_mask_remove=(before_count - after_count) / before_count,
            )

        if self.model_cfg.USE_NEAREST_LOSS and num_modes > 1:
            velocity_logit = nearest_predicted_dict["nearest_velocity_logit"]
        else:
            velocity_logit = forward_ret_dict["velocity_logit"]
        assert velocity_logit.ndim == 6
        if self.model_cfg.USE_CUMSUM:
            velocity_logit = velocity_logit.cumsum(4)

        if self.model_cfg.RELATIVE_VELOCITY:
            # Relative velocity
            vel_target = stacked_actor_feat[:, 1:, ..., 4:6] * VELOCITY_XY_RANGE - anchor_velocity[:, :-1]

            if self.model_cfg.RELATIVE_POSITION_HEADING:
                vel_target = rotate(vel_target[..., 0], vel_target[..., 1], -anchor_heading[:, :-1].squeeze(-1))

        else:
            vel_target = stacked_actor_feat[:, 1:, ..., 4:6]

        velocity_logit = velocity_logit[:, :-1][stacked_actor_valid_mask]
        vel_target = vel_target[stacked_actor_valid_mask]

        assert velocity_logit.shape == vel_target.shape
        if self.model_cfg.USE_HUBER_LOSS:
            vel_loss = F.huber_loss(input=velocity_logit, target=vel_target)
        else:
            vel_loss = F.mse_loss(input=velocity_logit, target=vel_target)

        if self.model_cfg.RELATIVE_HEADING:
            # Absolute heading
            heading_target = wrap_to_pi(stacked_actor_feat[:, 1:, ..., 3:4] * HEADING_RANGE - anchor_heading[:, :-1])
        else:
            heading_target = wrap_to_pi(stacked_actor_feat[:, 1:, ..., 3:4] * HEADING_RANGE)

        if self.model_cfg.USE_NEAREST_LOSS and num_modes > 1:
            heading_logit = nearest_predicted_dict["nearest_heading_logit"][:, :-1]
        else:
            heading_logit = forward_ret_dict["heading_logit"][:, :-1]
        assert heading_logit.shape == heading_target.shape
        assert heading_logit.ndim == 6
        if self.model_cfg.USE_CUMSUM:
            heading_logit = heading_logit.cumsum(4)

        heading_logit = heading_logit[stacked_actor_valid_mask]
        heading_target = heading_target[stacked_actor_valid_mask]

        assert heading_logit.shape == heading_target.shape, (heading_logit.shape, heading_target.shape)
        if self.model_cfg.USE_HUBER_LOSS:
            head_loss = F.huber_loss(input=wrap_to_pi(heading_logit - heading_target), target=heading_target * 0)
        else:
            head_loss = F.mse_loss(input=wrap_to_pi(heading_logit - heading_target), target=heading_target * 0)

        type_mask = forward_ret_dict["compress_actor_valid_mask"].unsqueeze(-1).expand(B, compress_T, N, num_modes)

        actor_type_logit = forward_ret_dict["actor_type_logit"]  # [B, compress_T, N, num_modes, 5]
        actor_type_logit = actor_type_logit[type_mask]

        gt_actor_type = gt_dict["decoder/actor_type"].reshape(B, 1, N, 1).expand(B, compress_T, N, num_modes)
        gt_actor_type = gt_actor_type[type_mask]

        type_loss = F.cross_entropy(input=actor_type_logit, target=gt_actor_type)

        return dict(
            position_loss=pos_loss,
            velocity_loss=vel_loss,
            heading_loss=head_loss,
            actor_type_loss=type_loss,
            score_loss=score_loss,
            slow_mask_remove=(before_count - after_count) / before_count,
        )

    def get_loss(self, data_dict, gt_dict, forward_ret_dict, tb_pre_tag=''):
        anchor_position = forward_ret_dict["anchor_position"]
        anchor_heading = forward_ret_dict["anchor_heading"]
        anchor_velocity = forward_ret_dict["anchor_velocity"]
        _, _, L, _ = data_dict["encoder/traffic_light_feature"].shape
        B, compress_T, N, num_modes, step_per_token, _ = forward_ret_dict["sampled_position"].shape

        weight_pos = self.model_cfg.LOSS_WEIGHTS.get('position', 1.0)
        weight_vel = self.model_cfg.LOSS_WEIGHTS.get('velocity', 0.2)
        weight_heading = self.model_cfg.LOSS_WEIGHTS.get('heading', 0.5)
        weight_type = self.model_cfg.LOSS_WEIGHTS.get('actor_type', 0.5)
        weight_light = self.model_cfg.LOSS_WEIGHTS.get('traffic_light_state', 0.5)
        weight_score = self.model_cfg.LOSS_WEIGHTS.get('score', 1.0)
        weight_token = self.model_cfg.LOSS_WEIGHTS.get('token', 1.0)
        weight_start = self.model_cfg.LOSS_WEIGHTS.get('start', 1.0)

        total_loss = 0.0
        layer_loss_list = []

        stacked_actor_feat = roll_and_stack(
            data_dict["encoder/agent_feature"],
            step_per_token=self.step_per_token,
            num_modes=1 if self.model_cfg.USE_NEAREST_LOSS else num_modes,
            compress_T=compress_T,
            compress_step=self.compress_step
        )
        stacked_actor_valid_mask = forward_ret_dict["stacked_actor_valid_mask"]

        # [B, T, N, num_modes, 2]
        gt_position = roll_and_stack(
            data_dict["encoder/agent_position"],
            compress_T=compress_T,
            step_per_token=step_per_token,
            num_modes=num_modes,
            compress_step=self.compress_step
        )

        # ========== Loss for start token predictor ==========
        init_actor_loss = torch.zeros(1)
        init_actor_loss_dict = {}
        if self.model_cfg.ENABLE_START_TOKEN:
            raise ValueError()
        #     init_actor_loss_dict = self.get_init_actor_loss(
        #         data_dict=data_dict,
        #         stacked_actor_valid_mask=stacked_actor_valid_mask[:, 0, :, 0],
        #         forward_ret_dict=forward_ret_dict,
        #     )
        #     init_actor_loss = (
        #         init_actor_loss_dict["init_pos_loss"] +
        #         init_actor_loss_dict["init_vel_loss"] +
        #         init_actor_loss_dict["init_head_loss"] +
        #         init_actor_loss_dict["init_size_loss"] +
        #         # init_actor_loss_dict["init_score_loss"] +
        #         init_actor_loss_dict["init_map_feat_score_loss"]
        #     )
        # if self.config.ENABLE_START_TOKEN_ACTOR_LOSS:
        #     init_actor_loss += (
        #         weight_pos * init_actor_loss_dict["init_position_loss"] +
        #         weight_vel * init_actor_loss_dict["init_velocity_loss"] +
        #         weight_heading * init_actor_loss_dict["init_heading_loss"]
        #     )
        # total_loss += weight_start * init_actor_loss

        # ========== Loss for actor predictor ==========
        if self.model_cfg.USE_NEAREST_LOSS:
            anchor_velocity = anchor_velocity[:, :, :, :1]
            anchor_position = anchor_position[:, :, :, :1]
            anchor_heading = anchor_heading[:, :, :, :1]
            stacked_actor_valid_mask = stacked_actor_valid_mask[:, :, :, :1]
        if self.model_cfg.LOSS_EACH_LAYER:
            for layer_index in range(self.num_decoder_layers):
                actor_loss = self.get_actor_loss(
                    stacked_actor_feat=stacked_actor_feat,
                    stacked_actor_valid_mask=stacked_actor_valid_mask,
                    gt_position=gt_position,
                    gt_dict=gt_dict,
                    forward_ret_dict=forward_ret_dict[f"prediction_list_{layer_index}"],
                    anchor_heading=anchor_heading,
                    anchor_velocity=anchor_velocity,
                    anchor_position=anchor_position,
                    layer_index=layer_index
                )
                layer_loss = (
                    weight_pos * actor_loss["position_loss"] + weight_vel * actor_loss["velocity_loss"] +
                    weight_heading * actor_loss["heading_loss"] + weight_type * actor_loss["actor_type_loss"] +
                    weight_score * actor_loss["score_loss"]
                )
                total_loss += layer_loss
                layer_loss_list.append(layer_loss)
        else:
            actor_loss = self.get_actor_loss(
                stacked_actor_feat=stacked_actor_feat,
                stacked_actor_valid_mask=stacked_actor_valid_mask,
                gt_dict=gt_dict,
                gt_position=gt_position,
                forward_ret_dict=forward_ret_dict,
                anchor_heading=anchor_heading,
                anchor_velocity=anchor_velocity,
                anchor_position=anchor_position,
            )
            total_loss += (
                weight_pos * actor_loss["position_loss"] + weight_vel * actor_loss["velocity_loss"] +
                weight_heading * actor_loss["heading_loss"] + weight_type * actor_loss["actor_type_loss"] +
                weight_score * actor_loss["score_loss"]
            )

        # ========== Loss for traffic light predictor ==========
        # TODO: Add traffic light loss back.
        # if L > 0:
        #     light_gt = roll_and_stack(
        #         gt_dict["traffic_light_state"].unsqueeze(-1),
        #         step_per_token=self.step_per_token,
        #         num_modes=1,
        #         compress_T=compress_T,
        #         compress_step=self.compress_step
        #     )
        #     light_mask = roll_and_stack_for_mask(
        #         data_dict["encoder/traffic_light_valid_mask"],
        #         step_per_token=self.step_per_token,
        #         num_modes=1,
        #         compress_T=compress_T,
        #         compress_step=self.compress_step
        #     )
        #     light_gt = light_gt.squeeze(3)  # Squeeze the "modes" dim since we don't output multi-mode traffic lights.
        #     light_mask = light_mask.squeeze(3)
        #     compress_traffic_light_mask = forward_ret_dict["compress_traffic_light_mask"]
        #     compress_traffic_light_mask = compress_traffic_light_mask.unsqueeze(-1).expand(
        #         B, compress_T, L, step_per_token
        #     )
        #     light_mask = torch.logical_and(light_mask, compress_traffic_light_mask)
        #     light_gt = light_gt.squeeze(-1)
        #     light_input = forward_ret_dict["traffic_light_state_logit"]
        #     light_input = light_input[light_mask]
        #     light_gt = light_gt[light_mask]
        #     light_loss = F.cross_entropy(
        #         input=light_input,
        #         target=light_gt
        #     )
        # else:
        light_loss = 0.0

        # ========== Loss for token-evolution ==========
        token_loss = 0.0
        if self.model_cfg.TOKEN_EVOLUTION:
            token_mask = forward_ret_dict["input_token_valid_mask"][:, 1:]
            out_t = forward_ret_dict["output_token"][:, :-1]
            next_in_t = forward_ret_dict["input_token"].detach()[:, 1:]
            token_loss = F.mse_loss(input=out_t[token_mask], target=next_in_t[token_mask])

        total_loss += weight_token * token_loss + weight_light * light_loss

        tb_dict = {k: v.mean().item() if torch.is_tensor(v) else float(v) for k, v in actor_loss.items()}
        tb_dict.update(
            total_loss=total_loss.item(),
            token_loss=token_loss.item() if self.model_cfg.TOKEN_EVOLUTION else float("nan"),
            traffic_light_loss=light_loss.item() if isinstance(light_loss, torch.Tensor) else float("nan"),
            init_actor_loss=init_actor_loss.item(),
            **init_actor_loss_dict
        )
        if layer_loss_list:
            for i in range(len(layer_loss_list)):
                tb_dict["layer{}_loss".format(i)] = layer_loss_list[i].item()
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, tb_dict

    def forward(self, input_dict):
        in_evaluation = input_dict.get("in_evaluation", False)

        actor_feature = input_dict["encoder/agent_feature"]
        actor_valid_mask = input_dict["encoder/agent_valid_mask"]
        actor_position = input_dict["encoder/agent_position"]
        B, T, N, D_actor = actor_feature.shape
        assert actor_feature.shape[:3] == actor_position.shape[:3]

        assert (T - 1) % self.compress_step == 0
        compress_T = (T - 1) // self.compress_step

        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_position = input_dict["encoder/map_position"]
        _, M, num_vector, D_vector = map_feature.shape

        traffic_light_feature = input_dict["encoder/traffic_light_feature"]
        traffic_light_position = input_dict["encoder/traffic_light_position"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]
        _, _, L, D_light = traffic_light_feature.shape

        # ========== Tokenize all objects (actor & map feat) ==========
        # [B, M, token dim]
        if "map_token" in input_dict:
            map_token = input_dict["map_token"]
        else:
            map_token = self.map_polyline_encoder(map_feature, map_valid_mask)

        # [B, M]
        map_token_valid_mask = map_valid_mask.sum(axis=-1) != 0

        actor_token = unwrap(self.actor_mlps(actor_feature[actor_valid_mask]), actor_valid_mask)

        # ===== Code fragment that stack tokens from [B, T, N, D] to [B, T/5, N, 5*D] =====
        token_dim = actor_token.shape[-1]
        # [B, T, N, token dim] -> [B, N, T, token dim]
        actor_token = actor_token.permute(0, 2, 1, 3)
        # -> [B, N, compress_T, token dim*compress_step]
        actor_token = actor_token[:, :, :compress_T *
                                  self.compress_step].reshape(B, N, compress_T, self.compress_step * token_dim)
        # -> [B, compress_T, N, token_dim*compress_step]
        actor_token = actor_token.permute(0, 2, 1, 3)

        decompress_actor_valid_mask = actor_valid_mask.clone()  # [:, :compress_T * self.compress_step]

        # [B, T, N+L] -> [B, N+L, T] -> [B, N+L, compress_T * compress_step]
        compress_actor_valid_mask = actor_valid_mask[:, :compress_T *
                                                     self.compress_step].reshape(B, compress_T, self.compress_step,
                                                                                 N).any(dim=2).clone()

        # -> [B, compress_T, N, token dim]
        actor_token = unwrap(
            self.actor_mlps_compress(actor_token[compress_actor_valid_mask]), compress_actor_valid_mask
        )

        displacement, last_pos = get_displacement(
            input_dict["encoder/agent_position"][:, :10, ..., :2], input_dict["encoder/agent_valid_mask"][:, :10]
        )
        # start_token = actor_token[:, 0]
        # start_token_valid_mask = compress_actor_valid_mask[:, 0].clone()
        # start_token_valid_mask = torch.logical_and(start_token_valid_mask, displacement.squeeze(1) > 0.01)
        # start_token_pe = self.start_token_pe(torch.arange(N, device=actor_token.device).unsqueeze(0).expand(B, N))

        # prepare the starting token. in shape [B, N, d_model]
        # actor_type_valid_mask = input_dict["decoder/actor_type"] != -1
        # empty_start_token = unwrap(
        #     self.start_token(input_dict["decoder/actor_type"][actor_type_valid_mask]),
        #     actor_type_valid_mask
        # )
        # assert start_token.shape == empty_start_token.shape

        # selected_num = torch.minimum(
        #     (start_token_valid_mask.sum(-1) * torch.rand(B, device=start_token_valid_mask.device)).int(),
        #     start_token_valid_mask.sum(-1)
        # ).clamp(1)  # [B, ]
        #
        # st_drop_mask = actor_type_valid_mask.new_zeros((B, N))
        # for i in range(B):
        #     st_valids = start_token_valid_mask[i].nonzero()[:, 0]
        #     st_ind = torch.randperm(len(st_valids))[:selected_num[i]]
        #     st_selected_ind = st_valids[st_ind]
        #     st_drop_mask[i, st_selected_ind] = 1
        # # assert (st_drop_mask.sum(-1) == selected_num).all(), (st_drop_mask.sum(-1), selected_num)
        # start_token[st_drop_mask] = empty_start_token[st_drop_mask]
        # start_token += start_token_pe

        compress_actor_position = find_last_valid_in_compress_step(
            actor_position, actor_valid_mask, compress_step=self.compress_step
        )

        # start_token_position = compress_actor_position[:, 0].clone()
        # start_token_position[st_drop_mask] = 0.0

        # st_training_mask = torch.logical_and(st_drop_mask, start_token_valid_mask)
        # st_training_mask = start_token_valid_mask

        # [B, T, L] -> [B, compress_T, L]
        compress_light_mask = traffic_light_valid_mask[:, :compress_T * self.compress_step].reshape(
            B, compress_T, self.compress_step, L
        ).any(dim=2).clone()

        # [B, T, num light, token dim]
        if L != 0:
            light_token = unwrap(
                self.light_mlps(traffic_light_feature[traffic_light_valid_mask]), traffic_light_valid_mask
            )

            # [B, T, L, token dim] -> [B, L, T, token dim]
            light_token = light_token.permute(0, 2, 1, 3)
            # [B, N, T, token dim] -> [B, L, compress_T, token dim*compress_step]
            light_token = light_token[:, :, :compress_T *
                                      self.compress_step].reshape(B, L, compress_T, self.compress_step * token_dim)
            # [B, N, compress_T, token dim*compress_step] -> [B, compress_T,LN, token dim*compress_step]
            light_token = light_token.permute(0, 2, 1, 3)

            light_token = unwrap(self.light_mlps_compress(light_token[compress_light_mask]), compress_light_mask)

        else:
            light_token = traffic_light_feature.new_zeros([B, compress_T, L, self.d_model])

        map_token, actor_token, light_token = self.pe(map_token, actor_token, light_token)

        cat_token = torch.concatenate([actor_token, light_token], dim=2)  # [B, T, N+L, token dim]

        cat_mask = torch.concatenate([compress_actor_valid_mask, compress_light_mask], dim=2)  # [B, T, N+L]
        all_token_valid_mask = torch.concatenate(
            [
                map_token_valid_mask,  # [B, M, token dim]
                # start_token_valid_mask,  # [B, N, token dim]
                cat_mask.reshape(B, compress_T * (L + N))  # [B, T*(N+L), token dim]
            ],
            dim=1
        )
        all_token_valid_mask_without_map = cat_mask
        # all_token_valid_mask_without_start_token = torch.concatenate([
        #     map_token_valid_mask,  # [B, M, token dim]
        #     cat_mask.reshape(B, compress_T * (L + N))  # [B, T*(N+L), token dim]
        # ], dim=1)

        # all_token is in shape [B, M+N+T*(N+L), token dim]
        all_token = torch.concatenate(
            [
                map_token,  # [B, M, token dim]
                # start_token,
                cat_token.reshape(B, compress_T * (L + N), -1),  # [B, T*(N+L), token dim]
            ],
            dim=1
        )

        all_token = unwrap(self.decoder_tokenizer(all_token[all_token_valid_mask]), all_token_valid_mask)

        compress_light_position = find_last_valid_in_compress_step(
            traffic_light_position.unsqueeze(1).repeat(1, T, 1, 1),
            traffic_light_valid_mask,
            compress_step=self.compress_step
        )

        compress_actor_velocity = find_last_valid_in_compress_step(
            input_dict["encoder/agent_feature"][..., 4:6] * VELOCITY_XY_RANGE,
            actor_valid_mask,
            compress_step=self.compress_step
        )

        compress_actor_heading = find_last_valid_in_compress_step(
            wrap_to_pi(input_dict["encoder/agent_feature"][..., 3:4] * HEADING_RANGE),
            actor_valid_mask,
            compress_step=self.compress_step
        )

        cat_position = torch.concatenate(
            [compress_actor_position[..., :2], compress_light_position[..., :2]], dim=2
        )  # [B, T, N+L]
        all_position = torch.concatenate(
            [
                map_position[..., :2],  # [B, M, token dim]
                # start_token_position[..., :2],
                cat_position.reshape(B, compress_T * (L + N), 2),  # [B, T*(N+L), token dim]
            ],
            dim=1
        )
        all_position_without_map = cat_position

        stacked_actor_valid_mask = roll_and_stack_for_mask(
            decompress_actor_valid_mask,
            compress_T=compress_T,
            step_per_token=self.step_per_token,
            num_modes=self.num_modes,
            compress_step=self.compress_step,
            set_unknown_to_true=False
        )
        # Also mask out the prediction of the
        input_token_valid = compress_actor_valid_mask.reshape(B, compress_T, N, 1,
                                                              1).expand(*stacked_actor_valid_mask.shape)
        stacked_actor_valid_mask = torch.logical_and(stacked_actor_valid_mask, input_token_valid)

        output_tokens, query_cache, pred_list = self.apply_transformer_decoder(
            all_token=all_token,
            all_token_valid_mask=all_token_valid_mask,
            all_position=all_position,
            all_token_valid_mask_without_map=all_token_valid_mask_without_map,
            # all_token_valid_mask_without_start_token=all_token_valid_mask_without_start_token,
            all_position_without_map=all_position_without_map,
            actor_token=actor_token,
            actor_position=compress_actor_position,
            actor_valid_mask=compress_actor_valid_mask,
            stacked_actor_valid_mask=stacked_actor_valid_mask,
            # start_token_pe=start_token_pe,
            # st_drop_mask=st_drop_mask,
            # start_token_valid_mask=start_token_valid_mask,
            traffic_light_token=light_token,
            traffic_light_position=compress_light_position,
            traffic_light_valid_mask=compress_light_mask,
            map_token=map_token,
            map_position=map_position,
            map_valid_mask=map_token_valid_mask,
            query_cache=input_dict["query_cache"] if "query_cache" in input_dict else None,
            in_evaluation=in_evaluation,
            anchor_heading=compress_actor_heading,
            anchor_velocity=compress_actor_velocity,
            anchor_position=compress_actor_position
        )
        # output_tokens = all_token

        assert output_tokens.shape == (B, compress_T * (N + L) + M, self.d_model)

        # output_start_tokens = output_tokens[:, M: M + N]
        object_output_tokens = output_tokens[:, M:]
        object_output_tokens = object_output_tokens.reshape(B, compress_T, N + L, self.d_model)

        actor_output_tokens = object_output_tokens[:, :, :N]  # (B, compress_T, N)
        traffic_light_output_tokens = object_output_tokens[:, :, N:]  # (B, compress_T, L)

        # ==================== Translate output tokens to prediction ====================
        if self.model_cfg.LOSS_EACH_LAYER:
            ret = pred_list[-1]
            if not in_evaluation:
                for layer_index in range(len(pred_list)):
                    ret[f"prediction_list_{layer_index}"] = pred_list[layer_index]

        else:
            ret = self.get_prediction_for_actor(
                anchor_position=compress_actor_position,
                anchor_velocity=compress_actor_velocity,
                anchor_heading=compress_actor_heading,
                actor_output_tokens=actor_output_tokens,
                actor_valid_mask=compress_actor_valid_mask,
                in_evaluation=in_evaluation,
                stacked_actor_valid_mask=stacked_actor_valid_mask,
                actor_type=input_dict["decoder/actor_type"]
            )

        # ret.update(self.get_prediction_for_start_token(
        #     output_start_tokens=output_start_tokens,
        #     start_token_valid_mask=start_token_valid_mask,
        # ))

        ret.update(
            self.get_prediction_for_traffic_light(
                traffic_light_output_tokens=traffic_light_output_tokens,
                traffic_light_valid_mask=compress_light_mask,
                in_evaluation=in_evaluation,
            )
        )

        # ret["st_training_mask"] = st_training_mask
        ret["encoder/map_valid_mask"] = map_token_valid_mask
        ret["compress_actor_valid_mask"] = compress_actor_valid_mask
        ret["compress_light_mask"] = compress_light_mask
        if self.model_cfg.TOKEN_EVOLUTION:
            ret["input_token"] = cat_token  # [B, compress_T, N+L, d_model]
            ret["input_token_valid_mask"] = all_token_valid_mask_without_map  # [B, comT, N+L]
            ret["output_token"] = object_output_tokens  # [B, compress_T, N+L, d_model]

        if in_evaluation:
            ret["map_token"] = map_token
            query_cache["last_T"] = T
            ret["query_cache"] = query_cache

        return ret

    def get_prediction_for_actor(
        self,
        *,
        anchor_position,
        anchor_velocity,
        anchor_heading,
        actor_output_tokens,
        actor_valid_mask,
        stacked_actor_valid_mask,
        in_evaluation,
        actor_type,
        layer_index=None,
        force_loss_at_final_layer=False
    ):
        B, compress_T, N, _ = actor_output_tokens.shape
        step_per_token = self.step_per_token
        num_modes = self.num_modes

        pred_heading = pred_velocity = pred_actor_type = None
        sampled_heading = sampled_actor_type = sampled_velocity = None
        if self.model_cfg.LOSS_EACH_LAYER and (not force_loss_at_final_layer):
            assert layer_index is not None
            if layer_index == self.num_decoder_layers - 1:  # Last layer:
                pred_pos, pred_velocity, pred_heading, pred_actor_type, pred_score = self.actor_predictor[layer_index](
                    actor_output_tokens, actor_valid_mask, step_per_token
                )
            else:
                # Only return 1 mode if in internal layers
                pred_pos, pred_score = self.actor_predictor[layer_index](
                    actor_output_tokens, actor_valid_mask, step_per_token
                )

        elif self.model_cfg.LOSS_EACH_LAYER and force_loss_at_final_layer:
            raise NotImplementedError()

        else:
            assert layer_index is None
            pred_pos, pred_velocity, pred_heading, pred_actor_type, pred_score = self.actor_predictor(
                actor_output_tokens, actor_valid_mask, step_per_token, actor_type
            )

        ret = {"score_logit": pred_score}
        if in_evaluation:
            ret.update(
                {
                    # "score": F.softmax(pred_score, dim=-1),
                    "anchor_position": anchor_position,
                    "anchor_velocity": anchor_velocity,
                    "anchor_heading": anchor_heading,
                }
            )

        # Transform from [B, compress_T, N, 2] to [B, compress_T, N, num_modes, step_per_token, 2]
        anchor_position = anchor_position.reshape(B, compress_T, N, 1, 1,
                                                  3).repeat(1, 1, 1, num_modes, step_per_token, 1)
        anchor_velocity = anchor_velocity.reshape(B, compress_T, N, 1, 1,
                                                  2).repeat(1, 1, 1, num_modes, step_per_token, 1)
        anchor_heading = anchor_heading.reshape(B, compress_T, N, 1, 1, 1).repeat(1, 1, 1, num_modes, step_per_token, 1)

        sampled_position = pred_pos.clone()
        assert sampled_position.ndim == 6
        if self.model_cfg.USE_CUMSUM:
            sampled_position = sampled_position.cumsum(4)
        assert anchor_position.shape == sampled_position.shape
        if self.model_cfg.RELATIVE_POSITION_HEADING:
            sampled_position = rotate(
                sampled_position[..., 0],
                sampled_position[..., 1],
                angle=anchor_heading.squeeze(-1),
                z=sampled_position[..., 2]
            )
            sampled_position += anchor_position
        else:
            sampled_position += anchor_position

        if pred_heading is not None:
            sampled_heading = pred_heading.clone()
            assert sampled_heading.ndim == 6
            if self.model_cfg.USE_CUMSUM:
                sampled_heading = sampled_heading.cumsum(4)
            if self.model_cfg.RELATIVE_HEADING:
                sampled_heading = sampled_heading + anchor_heading
            else:
                pass
            sampled_heading = wrap_to_pi(sampled_heading)

        if pred_velocity is not None:
            sampled_velocity = pred_velocity.clone()
            assert sampled_velocity.ndim == 6
            if self.model_cfg.USE_CUMSUM:
                sampled_velocity = sampled_velocity.cumsum(4)
            if self.model_cfg.RELATIVE_VELOCITY:
                if self.model_cfg.RELATIVE_POSITION_HEADING:
                    sampled_velocity = rotate(
                        sampled_velocity[..., 0], sampled_velocity[..., 1], angle=anchor_heading.squeeze(-1)
                    )
                assert anchor_velocity.shape == sampled_velocity.shape
                sampled_velocity = sampled_velocity + anchor_velocity
            else:
                sampled_velocity = sampled_velocity * VELOCITY_XY_RANGE

        if pred_actor_type is not None:
            actor_type_dist = torch.distributions.Categorical(logits=pred_actor_type)
            sampled_actor_type = actor_type_dist.sample()

        # sampled_position[~stacked_actor_valid_mask222] = 0
        # sampled_velocity[~stacked_actor_valid_mask222] = 0
        # sampled_heading[~stacked_actor_valid_mask222] = 0

        if in_evaluation:
            ret.update(
                {
                    # "sampled_position_before_offset": sampled_position_before_offset,
                    "sampled_position": sampled_position,
                    "sampled_heading": sampled_heading,
                    "sampled_velocity": sampled_velocity,
                    "sampled_actor_type": sampled_actor_type,
                    "position_logit": pred_pos,
                }
            )
        else:
            ret.update(
                {
                    "sampled_position": sampled_position,
                    "position_logit": pred_pos,
                    "heading_logit": pred_heading,
                    "velocity_logit": pred_velocity,
                    "actor_type_logit": pred_actor_type,
                }
            )
            if layer_index == self.num_decoder_layers - 1:
                ret.update(
                    {
                        "sampled_heading": sampled_heading,
                        "sampled_velocity": sampled_velocity,
                        "sampled_actor_type": sampled_actor_type,
                    }
                )

        if not in_evaluation:
            if self.model_cfg.LOSS_EACH_LAYER and layer_index < self.num_decoder_layers - 1:
                ret.update(
                    {
                        "stacked_actor_valid_mask": stacked_actor_valid_mask,
                        "compress_actor_valid_mask": actor_valid_mask,  # [B, compress_T, N]
                    }
                )
            else:
                ret.update(
                    {
                        "anchor_position": anchor_position,
                        "anchor_velocity": anchor_velocity,
                        "anchor_heading": anchor_heading,
                        "stacked_actor_valid_mask": stacked_actor_valid_mask,
                        "compress_actor_valid_mask": actor_valid_mask,  # [B, compress_T, N]
                    }
                )
        return ret

    def get_prediction_for_traffic_light(self, *, traffic_light_output_tokens, traffic_light_valid_mask, in_evaluation):
        step_per_token = self.step_per_token
        B, compress_T, L = traffic_light_valid_mask.shape

        # Get predicted vehicle type in shape: [B, compress_T, L, step_per_token, TRAFFIC_LIGHT_PREDICT_DIM]
        pred_traffic_light = traffic_light_output_tokens.new_zeros(
            B, compress_T, L, step_per_token, TRAFFIC_LIGHT_PREDICT_DIM
        )
        sampled_state = torch.zeros(
            [B, compress_T, L, step_per_token], dtype=torch.long, device=pred_traffic_light.device
        )
        # -> [B, compressed_T, L, compress_step, token_dim]
        if L > 0:
            traffic_light_prediction = self.traffic_light_predictor(
                traffic_light_output_tokens[traffic_light_valid_mask]
            )
            traffic_light_prediction = traffic_light_prediction.reshape(-1, step_per_token, TRAFFIC_LIGHT_PREDICT_DIM)
            if traffic_light_prediction.shape[0] > 0:
                state_dist = torch.distributions.Categorical(logits=traffic_light_prediction)
                sampled_state_valid = state_dist.sample()
                sampled_state[traffic_light_valid_mask] = sampled_state_valid
            pred_traffic_light[traffic_light_valid_mask] = traffic_light_prediction

        assert pred_traffic_light.shape[-1] == TRAFFIC_LIGHT_PREDICT_DIM
        B, compress_T, L, step_per_token, _ = pred_traffic_light.shape

        if in_evaluation:
            return {
                "sampled_traffic_light_state": sampled_state,
            }
        else:
            return {
                "traffic_light_state_logit": pred_traffic_light,
                "sampled_traffic_light_state": sampled_state,
                "compress_traffic_light_mask": traffic_light_valid_mask
            }

    # def get_internal_future_embedding(self, pred_dict, actor_valid_mask, layer_index):
    #     """
    #     This function process predict futures to the embedding.
    #     The embedding will be used to added to the tokens for next self-attention layer.
    #     """
    #     raise ValueError()
    #     B, T, N, num_modes, step_per_token, _ = pred_dict["position_logit"].shape
    #
    #     # [B, T, N, num_modes, step_per_token, 6]
    #     # with torch.no_grad():
    #     # x = pred_dict["position_logit"][actor_valid_mask].flatten(1, -1)
    #     x = torch.cat([
    #         pred_dict["position_logit"][actor_valid_mask],
    #         pred_dict["score_logit"][actor_valid_mask].reshape(-1, num_modes, 1, 1).repeat(1, 1, step_per_token, 1)
    #     ], dim=-1).flatten(1, -1)
    #
    #     # -> [B, T, N, d_model]
    #     x = unwrap(self.future_mlp[layer_index](x), actor_valid_mask)
    #
    #     return x
