import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

from scenestreamer.dataset import constants
from scenestreamer.models import relation
from scenestreamer.models.layers import common_layers
from scenestreamer.models.layers import fourier_embedding
from scenestreamer.models.layers.gpt_decoder_layer import MultiCrossAttTransformerDecoder
from scenestreamer.models.layers.gpt_decoder_layer import MultiheadAttentionLayer
from scenestreamer.models.motion_decoder import create_causal_mask
from scenestreamer.models.motion_decoder_gpt import get_edge_info_new
from scenestreamer.tokenization.trafficgen_tokenizers import TrafficGenTokenizer
from scenestreamer.utils import utils


class MultiCrossAttTransformerDecoderLayerForTrafficGen(Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, use_adaln=False) -> None:
        super().__init__()
        assert dropout == 0.0
        self.cross_a2a = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=True,
            simple_relation_factor=1,
            is_v7=True,
            update_relation=False,
            add_relation_to_v=False,
        )
        self.cross_a2s = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=True,
            simple_relation_factor=1,
            is_v7=True,
            update_relation=False,
            add_relation_to_v=False,
        )

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = common_layers.Mlp(in_features=d_model, hidden_features=4 * d_model, act_layer=approx_gelu, drop=0)

        # self.cross_a2s = MultiheadAttentionLayer(d_model=d_model, n_heads=nhead, dropout=dropout, simple_relation=False)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = common_layers.Mlp(in_features=d_model, hidden_features=4 * d_model, act_layer=approx_gelu, drop=0)

        self.a2s_norm = nn.LayerNorm(d_model)
        self.a2a_norm = nn.LayerNorm(d_model)
        self.mlp_prenorm = nn.LayerNorm(d_model)

        self.a2a_norm_rel = nn.LayerNorm(d_model)
        self.a2s_norm_rel = nn.LayerNorm(d_model)

    def forward(self, *, agent_token, scene_token, a2a_info, a2s_info, use_cache=False, past_key_value=None, **kwargs):
        B, N, D = agent_token.shape
        x = agent_token

        # === agent-agent attention ===
        out = x
        out = self.a2a_norm(out)
        out, past_key_value_a2t, _ = self.cross_a2a(
            q=out,
            k=out,
            edge_features=self.a2a_norm_rel(a2a_info['edge_features']),
            edge_index=a2a_info['edge_index'],
            use_cache=use_cache,
            cache=past_key_value,
        )
        x = x + out

        # === agent-scene attention ===
        out = x
        out = self.a2s_norm(out)
        out, _, _ = self.cross_a2s(
            q=out,
            k=scene_token,
            edge_features=self.a2s_norm_rel(a2s_info['edge_features']),
            edge_index=a2s_info['edge_index'],
        )
        x = x + out

        # === Feed-forward layer ===
        out = x
        out = self.mlp_prenorm(out)
        out = self.mlp(out)
        x = x + out
        return x, past_key_value_a2t


class OffsetHead(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.prenorm = nn.LayerNorm(input_dim)
        self.mlp = common_layers.build_mlps(c_in=input_dim, mlp_channels=[4 * d_model, d_model], ret_before_act=True)
        self.norm = nn.LayerNorm(d_model)
        self.position_x = nn.Linear(d_model, TrafficGenTokenizer.num_bins["position_x"])
        self.position_y = nn.Linear(d_model, TrafficGenTokenizer.num_bins["position_y"])
        self.velocity_x = nn.Linear(d_model, TrafficGenTokenizer.num_bins["velocity_x"])
        self.velocity_y = nn.Linear(d_model, TrafficGenTokenizer.num_bins["velocity_y"])
        self.heading = nn.Linear(d_model, TrafficGenTokenizer.num_bins["heading"])
        self.length = nn.Linear(d_model, TrafficGenTokenizer.num_bins["length"])
        self.width = nn.Linear(d_model, TrafficGenTokenizer.num_bins["width"])
        self.height = nn.Linear(d_model, TrafficGenTokenizer.num_bins["height"])

    def forward(self, x):
        x = self.prenorm(x)
        x = self.mlp(x)
        x = self.norm(x)
        return {
            "position_x": self.position_x(x),
            "position_y": self.position_y(x),
            "velocity_x": self.velocity_x(x),
            "velocity_y": self.velocity_y(x),
            "heading": self.heading(x),
            "length": self.length(x),
            "width": self.width(x),
            "height": self.height(x),
        }


class TrafficGenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        self.max_agents = self.config.PREPROCESSING.MAX_AGENTS
        assert self.config.MODEL.NAME in ['gpt']
        num_tg_actions = self.config.PREPROCESSING.MAX_MAP_FEATURES + 2
        self.start_action_id = config.PREPROCESSING.MAX_MAP_FEATURES
        self.end_action_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1

        # === Input embedding ===
        self.action_embed = fourier_embedding.FourierEmbedding(input_dim=5, hidden_dim=d_model, num_freq_bands=64)
        self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
            input_dim=11, hidden_dim=d_model, num_freq_bands=64
        )
        self.relation_embed_a2s = fourier_embedding.FourierEmbedding(input_dim=3, hidden_dim=d_model, num_freq_bands=64)
        self.type_embed = common_layers.Tokenizer(
            num_actions=constants.NUM_TYPES, d_model=d_model, add_one_more_action=False
        )
        self.shape_embed = common_layers.build_mlps(c_in=3, mlp_channels=[d_model, d_model], ret_before_act=True)
        self.map_embed = common_layers.Tokenizer(num_actions=num_tg_actions, d_model=d_model, add_one_more_action=False)
        self.step_embed = common_layers.Tokenizer(
            num_actions=self.max_agents + 2, d_model=d_model, add_one_more_action=False
        )

        # === Transformer ===
        self.decoder = MultiCrossAttTransformerDecoder(
            decoder_layer=MultiCrossAttTransformerDecoderLayerForTrafficGen(
                d_model=d_model, nhead=self.num_heads, dropout=0.0, use_adaln=False
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
        )

        # === Action head ===
        self.action_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, num_tg_actions], ret_before_act=True
        )
        self.action_prenorm = nn.LayerNorm(d_model)

        # === Offset heads ===
        offset_head_input = 2 * d_model
        self.agent_type_head = common_layers.build_mlps(
            c_in=offset_head_input,
            mlp_channels=[d_model, TrafficGenTokenizer.num_bins['agent_type']],
            ret_before_act=True
        )
        self.offset_head = OffsetHead(input_dim=offset_head_input, d_model=d_model)
        self.trafficgen_tokenizer = TrafficGenTokenizer(self.config)

    def autoregressive_rollout_trafficgen(self, data_dict):

        START_ACTION = self.config.PREPROCESSING.MAX_MAP_FEATURES
        END_ACTION = self.config.PREPROCESSING.MAX_MAP_FEATURES + 1

        # Prepare running variables
        current_input_action = data_dict["decoder/input_action_for_trafficgen"].clone()[:, :1]
        current_input_action_valid_mask = data_dict["decoder/input_action_valid_mask_for_trafficgen"].clone()[:, :1]
        num_valid_gt = data_dict["decoder/input_action_valid_mask_for_trafficgen"].sum().item() - 1

        accumulative_action_valid_mask = current_input_action_valid_mask.clone()

        current_agent_position = data_dict["decoder/modeled_agent_position_for_trafficgen"].clone()[:, :1]
        current_agent_velocity = data_dict["decoder/modeled_agent_velocity_for_trafficgen"].clone()[:, :1]
        current_agent_heading = data_dict["decoder/modeled_agent_heading_for_trafficgen"].clone()[:, :1]
        current_agent_type = data_dict["decoder/agent_type_for_trafficgen"].clone()[:, :1]
        current_agent_shape = data_dict["decoder/current_agent_shape_for_trafficgen"].clone()[:, :1]
        current_agent_feature = data_dict["decoder/input_action_feature_for_trafficgen"].clone()[:, :1]

        B = current_input_action.shape[0]

        use_gt = False

        # data_dict = model.model.encode_scene(data_dict)

        assert "encoder/scenario_token" in data_dict, "You must call encode_scene first."

        num_decode_steps = 128
        num_collisions = 0
        num_violations = 0
        assert B == 1
        decode_step = 0
        # for decode_step in range(num_decode_steps):
        for _ in range(500):
            input_dict = {
                # Static features
                "encoder/scenario_token": data_dict["encoder/scenario_token"],
                "encoder/scenario_heading": data_dict["encoder/scenario_heading"],
                "encoder/scenario_position": data_dict["encoder/scenario_position"],
                "encoder/scenario_valid_mask": data_dict["encoder/scenario_valid_mask"],
                "encoder/map_position": data_dict["encoder/map_position"],
                "encoder/map_feature": data_dict["encoder/map_feature"],
                "encoder/map_valid_mask": data_dict["encoder/map_valid_mask"],
                "in_evaluation": torch.ones([
                    B,
                ], dtype=torch.bool),

                # Actions
                "decoder/input_action_for_trafficgen": current_input_action,
                "decoder/input_action_valid_mask_for_trafficgen": current_input_action_valid_mask,

                # Agent features
                "decoder/modeled_agent_position_for_trafficgen": current_agent_position,
                "decoder/modeled_agent_heading_for_trafficgen": current_agent_heading,
                "decoder/modeled_agent_velocity_for_trafficgen": current_agent_velocity,
                "decoder/agent_type_for_trafficgen": current_agent_type,
                "decoder/current_agent_shape_for_trafficgen": current_agent_shape,
                "decoder/input_action_feature_for_trafficgen": current_agent_feature,
            }

            temperature = 1.0

            input_dict = copy.deepcopy(input_dict)

            output_dict = self.forward(input_dict)

            # Force model to predict at least the same amount of agents in GT.
            if decode_step < num_decode_steps:
                force_no_end = True
            else:
                force_no_end = False

            sampled_action = self.sample_action(output_dict, force_no_end=force_no_end, temperature=temperature)
            sampled_action = sampled_action[:, -1:]

            if decode_step == 0 and self.config.FORCE_SDC_FOR_TRAFFICGEN:
                # In LCTGen, the map feature is cropped around SDC's position.
                # The agent will always pick the map feature that is closest to the center of the map.
                # That is the map feature whose (x, y) is closest to the (0, 0).
                # To ensure fair comparison, we need to do the same here and force the first selected
                # map feature to be the one closest to (0, 0).

                # TODO: Add a flag here.

                # TODO hardcode
                if data_dict["decoder/agent_position"].shape[1] > 150:
                    current_t = 0
                else:
                    current_t = 10
                assert B == 1
                sdc_index = data_dict["decoder/sdc_index"][0].item()
                sdc_center = data_dict["decoder/agent_position"][:, current_t, sdc_index]

                map_to_sdc_dist = (data_dict["encoder/map_position"][0][..., :2] - sdc_center[0, :2]).norm(dim=-1)

                map_to_sdc_dist_valid_mask = data_dict["encoder/map_valid_mask"].clone()
                map_to_sdc_dist_valid_mask = (
                    map_to_sdc_dist_valid_mask & (data_dict["encoder/map_feature"][:, :, 0, 13] == 1)
                )

                map_to_sdc_dist[~map_to_sdc_dist_valid_mask[0]] = 1e6

                map_argmin = map_to_sdc_dist.argmin()
                map_min = map_to_sdc_dist.min()

                # print("Original select action: {}, new action: {}, min dist: {}".format(
                #     sampled_action.item(), map_argmin.item(), map_min.item()
                # ))
                sampled_action = map_argmin.unsqueeze(0).unsqueeze(-1)

            if use_gt:
                sampled_action = data_dict["decoder/input_action_for_trafficgen"][:, decode_step + 1:decode_step + 2]
            new_current_input_action = torch.cat([current_input_action, sampled_action], dim=1)
            is_end = sampled_action == END_ACTION
            sampled_action[is_end] = 0
            accumulative_action_valid_mask[is_end] = False

            # Use last action to predict next position
            # The first action is START_ACTION so we need to skip it.
            agent_type_output = self.forward_agent_type(output_dict, action=new_current_input_action[:, 1:])
            agent_type = self.sample_agent_type(agent_type_output, temperature=temperature)
            offset_output = self.forward_offset(
                output_dict, action=new_current_input_action[:, 1:], agent_type=agent_type
            )
            offset_action = self.sample_offset(offset_output=offset_output, temperature=temperature)
            offset_action = {k: v[:, -1:] for k, v in offset_action.items()}
            agent_type = agent_type[:, -1:]

            if use_gt:
                offset_action = data_dict["decoder/target_offset_for_trafficgen"][:, decode_step:decode_step + 1]
                # gt_position_x, gt_position_y, gt_heading, gt_vel_x, gt_vel_y, gt_shape_l, gt_shape_w, gt_shape_h, gt_agent_type
                offset_action = {
                    "position_x": offset_action[..., 0],
                    "position_y": offset_action[..., 1],
                    "heading": offset_action[..., 2],
                    "velocity_x": offset_action[..., 3],
                    "velocity_y": offset_action[..., 4],
                    "length": offset_action[..., 5],
                    "width": offset_action[..., 6],
                    "height": offset_action[..., 7],
                    "agent_type": offset_action[..., 8],
                }

            predicted_values = self.trafficgen_tokenizer.detokenize(
                data_dict, new_current_input_action[:, -1:], agent_type=agent_type, offset_action=offset_action
            )
            pos = predicted_values["position"]
            head = predicted_values["heading"]
            vel = predicted_values["velocity"]
            agent_type = predicted_values["agent_type"]  # in 0,1,2
            agent_shape = predicted_values["shape"]
            agent_feature = predicted_values["feature"]
            pos = pos * accumulative_action_valid_mask.unsqueeze(-1)
            head = head * accumulative_action_valid_mask
            vel = vel * accumulative_action_valid_mask.unsqueeze(-1)
            agent_type = agent_type * accumulative_action_valid_mask
            agent_shape = agent_shape * accumulative_action_valid_mask.unsqueeze(-1)
            agent_feature = agent_feature * accumulative_action_valid_mask.unsqueeze(-1)

            # BID = 0
            # print("=== Step: {} ===".format(decode_step))
            # print(
            #     "New agent type: {}, length: {:.2f}, width: {:.2f}, height: {:.2f}".format(
            #         agent_type[BID, 0].item(), agent_shape[BID, 0, 0].item(), agent_shape[BID, 0, 1].item(),
            #         agent_shape[BID, 0, 2].item()
            #     )
            # )

            # Check if collision happens:
            from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
            assert current_agent_position.shape[0] == 1
            existing_contours = cal_polygon_contour(
                x=current_agent_position[0, :, 0].cpu().numpy(),
                y=current_agent_position[0, :, 1].cpu().numpy(),
                theta=current_agent_heading[0, :].cpu().numpy(),
                width=current_agent_shape[0, :, 1].cpu().numpy(),
                length=current_agent_shape[0, :, 0].cpu().numpy()
            )  # (N, 4, 2)
            new_contour = cal_polygon_contour(
                x=pos[0, :, 0].cpu().numpy(),
                y=pos[0, :, 1].cpu().numpy(),
                theta=head[0, :].cpu().numpy(),
                width=agent_shape[0, :, 1].cpu().numpy(),
                length=agent_shape[0, :, 0].cpu().numpy()
            )
            if existing_contours.shape[0] == 1:
                no_coll = True  # Skip first one (it's the START_ACTION)
            else:
                no_coll = True
                for existing_id in range(1, existing_contours.shape[0]):
                    collision_detected = detect_collision(
                        [existing_contours[existing_id]],  # (N, 4, 2)
                        [current_input_action_valid_mask[0][existing_id]],  # (N,)
                        new_contour,
                        accumulative_action_valid_mask[0]
                    )
                    if collision_detected[0]:
                        # print("Collision detected!")
                        num_collisions += 1
                        no_coll = False
                        break

            # ===== Additional postprocessing to comply with LCTGen =====
            offset_values = predicted_values["offset_values"]
            vel_valid_mask = abs(torch.atan2(offset_values["velocity_y"], offset_values["velocity_x"])) < np.pi / 6
            dir_valid_mask = abs(offset_values["heading"]) < np.pi / 4
            sdc_index = data_dict["decoder/sdc_index"][0].item()

            # TODO hardcode
            if data_dict["decoder/agent_position"].shape[1] > 150:
                current_t = 0
            else:
                current_t = 10

            sdc_center = data_dict["decoder/agent_position"][0, current_t, sdc_index]
            distance_mask = ((abs(pos[..., 0] - sdc_center[0]) < 50) & (abs(pos[..., 1] - sdc_center[1]) < 50))
            if existing_contours.shape[0] == 1:
                no_violation = True  # Skip first one (it's the START_ACTION)
            else:
                assert vel_valid_mask.numel() == 1
                assert dir_valid_mask.numel() == 1
                assert distance_mask.numel() == 1
                no_violation = (vel_valid_mask & dir_valid_mask & distance_mask).item()
                if not no_violation:
                    num_violations += 1

            if no_coll:
                # Overwrite
                current_agent_position = torch.cat([current_agent_position, pos], dim=1)
                current_agent_velocity = torch.cat([current_agent_velocity, vel], dim=1)
                current_agent_heading = torch.cat([current_agent_heading, head], dim=1)
                current_agent_type = torch.cat([current_agent_type, agent_type], dim=1)
                current_agent_shape = torch.cat([current_agent_shape, agent_shape], dim=1)
                current_agent_feature = torch.cat([current_agent_feature, agent_feature], dim=1)
                current_input_action_valid_mask = torch.cat(
                    [current_input_action_valid_mask, accumulative_action_valid_mask], dim=1
                )
                current_input_action = new_current_input_action.clone()

                decode_step += 1

            if not accumulative_action_valid_mask.any():
                break

            if decode_step > num_decode_steps:
                break

        # Remove batch dim

        data_dict.update(input_dict)
        return data_dict, {"num_collisions": num_collisions, "num_violations": num_violations}

    def forward(self, input_dict, use_cache=False):
        assert self.config.REMOVE_AGENT_FROM_SCENE_ENCODER
        assert self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE
        assert use_cache is False

        # TrafficGen decoder takes two inputs:
        # 1. The map features, which is the scene tokens below (from the SceneEncoder)
        # 2. The agent features of a frame, which is the agent tokens below (from the AgentEncoder)
        # Should note that the agent features will start with <start> token and end with <end> token,
        # just like in language task.
        # ===== Scene Tokens =====
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]
        B, M, _ = input_dict["encoder/map_position"].shape
        S = scene_token.shape[1]
        map_id = torch.zeros([B, S], dtype=torch.long, device=scene_token.device)
        map_id[:, :M] = torch.arange(M, device=map_id.device).unsqueeze(0)
        map_id[~scenario_valid_mask] = -1
        map_id_pe = self.map_embed(map_id)
        # We don't add map feat ID pe in SceneEncoder, so we add it here.
        scene_token = scene_token + map_id_pe

        # ===== Agent Tokens =====
        input_action = input_dict["decoder/input_action_for_trafficgen"]
        B, seq_len = input_action.shape
        input_action_valid_mask = input_dict["decoder/input_action_valid_mask_for_trafficgen"]
        agent_pos = input_dict["decoder/modeled_agent_position_for_trafficgen"]
        agent_heading = input_dict["decoder/modeled_agent_heading_for_trafficgen"]
        # agent_vel = input_dict["decoder/modeled_agent_velocity_for_trafficgen"]

        # Shape embedding and type embedding
        type_emb = self.type_embed(input_dict["decoder/agent_type_for_trafficgen"])
        shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape_for_trafficgen"])

        if "decoder/input_step_for_trafficgen" not in input_dict:
            input_step = torch.arange(seq_len).to(input_action.device).unsqueeze(0).repeat(B, 1)
            input_dict["decoder/input_step_for_trafficgen"] = input_step
        input_step = input_dict["decoder/input_step_for_trafficgen"]
        assert input_step.shape == (B, seq_len), (B, seq_len, input_step.shape)
        input_step[input_step >= self.max_agents] = self.max_agents
        step_emb = self.step_embed(input_step)

        # Here we reuse the map_embedding to embed the action!
        action_emb = self.map_embed(input_action)
        action_feat = input_dict["decoder/input_action_feature_for_trafficgen"]
        action_token = self.action_embed(
            continuous_inputs=action_feat[input_action_valid_mask],
            categorical_embs=[
                type_emb[input_action_valid_mask], shape_emb[input_action_valid_mask],
                action_emb[input_action_valid_mask], step_emb[input_action_valid_mask]
            ]
        )
        action_token = utils.unwrap(action_token, valid_mask=input_action_valid_mask)

        # The T here is the number of agents, not the real temporal length.
        causal_valid_mask = create_causal_mask(T=seq_len, N=1, is_valid_mask=True).to(action_token.device)

        # ===== Get agent-agent relation =====
        a2a_rel_feat, a2a_mask, _ = relation.compute_relation_simple_relation(
            query_pos=agent_pos,
            query_heading=agent_heading,
            query_valid_mask=input_action_valid_mask,
            query_step=None,
            key_pos=agent_pos,
            key_heading=agent_heading,
            key_valid_mask=input_action_valid_mask,
            key_step=None,
            hidden_dim=self.d_model,
            causal_valid_mask=causal_valid_mask,
            knn=self.config.MODEL.A2A_KNN,
            return_pe=False,
            query_width=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 1],
            query_length=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 0],
            key_width=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 1],
            key_length=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 0],
            non_agent_relation=False,
            per_contour_point_relation=False,
        )
        # a2a_rel_pe = utils.unwrap(self.relation_embed_a2a(a2a_rel_feat[a2a_mask]), a2a_mask)
        # a2a_info = get_edge_info(attn_valid_mask=a2a_mask, rel_pe_cross=a2a_rel_pe)
        a2a_info = get_edge_info_new(
            q_k_valid_mask=a2a_mask,
            q_k_relation=a2a_rel_feat,
            relation_model=self.relation_embed_a2a,
            relation_model_v=None
        )

        # ===== Get agent-scene relation =====
        a2s_rel_feat, a2s_mask, a2s_indices = relation.compute_relation_simple_relation(
            query_pos=agent_pos,
            query_heading=agent_heading,
            query_valid_mask=input_action_valid_mask,
            query_step=None,
            key_pos=input_dict["encoder/scenario_position"],  # [..., :2],
            key_heading=input_dict["encoder/scenario_heading"],
            key_valid_mask=scenario_valid_mask,
            key_step=None,
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=self.config.MODEL.A2S_KNN,
            gather=False,
            return_pe=False,
            query_width=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 1],
            query_length=input_dict["decoder/current_agent_shape_for_trafficgen"][:, :, 0],
            key_width=torch.zeros([B, S], device=agent_pos.device),
            key_length=torch.zeros([B, S], device=agent_pos.device),
            non_agent_relation=True,
            per_contour_point_relation=False,
        )
        # a2s_rel_pe = utils.unwrap(self.relation_embed_a2s(a2s_rel_feat[a2s_mask]), a2s_mask)
        # a2s_info = get_edge_info(attn_valid_mask=a2s_mask, rel_pe_cross=a2s_rel_pe)
        a2s_info = get_edge_info_new(
            q_k_valid_mask=a2s_mask,
            q_k_relation=a2s_rel_feat,
            relation_model=self.relation_embed_a2s,
            relation_model_v=None
        )

        # === Call models ===
        past_key_value_list = None
        if use_cache:
            # Cache from last rollout
            if "decoder/cache" in input_dict:
                past_key_value_list = input_dict["decoder/cache"]

        decoded_tokens = self.decoder(
            agent_token=action_token,
            scene_token=scene_token,
            a2a_info=a2a_info,
            # a2t_info=None,
            a2s_info=a2s_info,
            # condition_token=None,  #condition_token if self.use_adaln else None,
            use_cache=use_cache,  # We don't need decoder to take care cache.
            past_key_value_list=past_key_value_list
        )

        # if use_cache:
        #     decoded_tokens, past_key_value_list = decoded_tokens
        #     for l in past_key_value_list:
        #         if l:
        #             l.append((B * N, real_T))
        #     input_dict["decoder/cache"] = past_key_value_list

        output_token = self.action_head(self.action_prenorm(decoded_tokens[input_action_valid_mask]))
        output_token = utils.unwrap(output_token, valid_mask=input_action_valid_mask)

        input_dict["decoder/output_logit_for_trafficgen"] = output_token
        input_dict["decoder/output_token_for_trafficgen"] = decoded_tokens
        return input_dict

    def sample_action(self, data_dict, force_no_end=False, temperature=1.0):
        raw_output_logit = data_dict['decoder/output_logit_for_trafficgen']  # [:, -1, :]
        output_logit = raw_output_logit.new_full(raw_output_logit.shape, float('-inf'))

        # mask out invalid actions
        # scenario_valid_mask = data_dict["encoder/scenario_valid_mask"]
        B, M, _ = data_dict["encoder/map_position"].shape
        # map_mask = scenario_valid_mask[:, :M]
        # assert (map_mask == data_dict["encoder/map_valid_mask"]).all()
        map_mask = data_dict["encoder/map_valid_mask"]
        only_lane = self.config.ONLY_LANE_FOR_TRAFFICGEN
        if only_lane:
            map_mask = map_mask & (data_dict["encoder/map_feature"][:, :, 0, 13] == 1)

        T = raw_output_logit.shape[1]

        output_logit[:, :, :M] = torch.where(
            map_mask.unsqueeze(1).expand(-1, T, -1), raw_output_logit[:, :, :M], output_logit[:, :, :M]
        )
        if force_no_end:
            output_logit[:, :, -1] = float('-inf')
        else:
            output_logit[:, :, -1] = raw_output_logit[:, :, -1]  # The prob for "End Action"

        # Just do the softmax sampling
        sampled_action = torch.distributions.Categorical(logits=output_logit / temperature).sample()
        return sampled_action

    def forward_agent_type(self, input_dict, action):
        # Get scene token:
        in_evaluation = input_dict["in_evaluation"][0].item()
        scene_token = input_dict["encoder/scenario_token"]
        B, M, _ = input_dict["encoder/map_position"].shape
        action = action.clone()

        is_valid_action = (action < self.start_action_id) & (action >= 0)
        action[~is_valid_action] = 0

        selected_scene_token = torch.gather(
            scene_token, dim=1, index=action.unsqueeze(-1).expand(-1, -1, scene_token.shape[-1])
        )

        # Get the input
        output_token = input_dict["decoder/output_token_for_trafficgen"].clone()
        if in_evaluation:
            assert selected_scene_token.shape == output_token.shape
        else:
            # output token contains value for the END_ACTION, which is not in the selected_scene_token.
            B, T_minus_1, D = selected_scene_token.shape
            assert output_token.shape == (B, T_minus_1 + 1, D)
            output_token = output_token[:, :-1, :]

        agent_type_offset = self.agent_type_head(torch.cat([output_token, selected_scene_token], dim=-1))
        return agent_type_offset

    def sample_agent_type(self, agent_type_output, temperature=1.0):
        return torch.distributions.Categorical(logits=agent_type_output / temperature).sample()

    def forward_offset(self, input_dict, action, agent_type):

        # Get scene token:
        in_evaluation = input_dict["in_evaluation"][0].item()
        scene_token = input_dict["encoder/scenario_token"]
        B, M, _ = input_dict["encoder/map_position"].shape
        action = action.clone()

        is_valid_action = (action < self.start_action_id) & (action >= 0)
        action[~is_valid_action] = 0

        selected_scene_token = torch.gather(
            scene_token, dim=1, index=action.unsqueeze(-1).expand(-1, -1, scene_token.shape[-1])
        )

        # Get the input
        output_token = input_dict["decoder/output_token_for_trafficgen"].clone()
        if in_evaluation:
            assert selected_scene_token.shape == output_token.shape
        else:
            # output token contains value for the END_ACTION, which is not in the selected_scene_token.
            B, T_minus_1, D = selected_scene_token.shape
            assert output_token.shape == (B, T_minus_1 + 1, D)
            output_token = output_token[:, :-1, :]

        agent_type_emb = self.type_embed(agent_type)
        output_token = agent_type_emb + output_token

        offset_output = self.offset_head(torch.cat([output_token, selected_scene_token], dim=-1))

        return offset_output

    def sample_offset(self, offset_output, temperature=1.0):
        def _sample(v):
            return torch.distributions.Categorical(logits=v / temperature).sample()

        offset_action = {k: _sample(v) for k, v in offset_output.items()}

        return offset_action
