from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from scenestreamer.dataset import constants
from scenestreamer.dataset.preprocess_action_label import SafetyAction
from scenestreamer.models import relation
from scenestreamer.models.layers import common_layers, fourier_embedding
from scenestreamer.models.layers.gpt_decoder_layer import MultiCrossAttTransformerDecoderLayer, MultiCrossAttTransformerDecoder
from scenestreamer.models.motion_decoder import create_causal_mask
from scenestreamer.models.scene_encoder import mode_agent_id
from scenestreamer.tokenization import get_action_dim, get_tokenizer, START_ACTION, END_ACTION
from scenestreamer.utils import utils


def get_edge_info_new(*, q_k_valid_mask, q_k_relation, relation_model, relation_model_v):
    B, Lq, Lk = q_k_valid_mask.shape
    edge_index, _ = dense_to_sparse(q_k_valid_mask.swapaxes(1, 2).contiguous())
    assert edge_index.numel() > 0, (edge_index.shape, q_k_valid_mask.sum())
    assert edge_index[0].max() < B * Lk, f"{edge_index[0].max()} >= {B * Lk}"
    assert edge_index[1].max() < B * Lq, f"{edge_index[1].max()} >= {B * Lq}"

    batch_ind = edge_index[1] // Lq
    q_ind = edge_index[1] % Lq
    batch_ind_k = edge_index[0] // Lk
    k_ind = edge_index[0] % Lk
    assert torch.all(batch_ind == batch_ind_k)
    edge_features = q_k_relation[batch_ind, q_ind, k_ind]

    if relation_model_v is not None:
        edge_features_v = relation_model_v(edge_features)
    else:
        edge_features_v = None

    if relation_model is not None:
        edge_features = relation_model(edge_features)

    return {
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_features_v": edge_features_v,
    }


###############################################################################
# New Data Structures
###############################################################################

# @dataclass
# class EncoderOutput:
#     # Formerly EncoderInput
#     scenario_token: torch.Tensor
#     scenario_valid_mask: torch.Tensor
#     scenario_position: torch.Tensor
#     scenario_heading: torch.Tensor
#     modeled_agent_pe: Optional[torch.Tensor] = None


@dataclass
class DecoderInput:
    # Encoder's output
    map_token: torch.Tensor
    map_valid_mask: torch.Tensor
    map_position: torch.Tensor
    map_heading: torch.Tensor
    map_pe: torch.Tensor
    tl_token: torch.Tensor
    tl_position: torch.Tensor
    tl_valid_mask: torch.Tensor

    # Decoder-side inputs
    input_action: torch.Tensor  # (B, T, N)
    input_action_valid_mask: torch.Tensor  # (B, T, N)
    agent_delta: torch.Tensor  # (B, T, N, D_delta)
    agent_position: torch.Tensor  # (B, N, D_pos) or (B, T, N, D_pos)
    agent_heading: torch.Tensor  # (B, N) or (B, T, N)
    agent_velocity: torch.Tensor  # (B, N, D_vel) or (B, T, N, D_vel)  # TODO: Remove this?
    agent_type: torch.Tensor  # (B, N) or (B, T, N)

    agent_shape: torch.Tensor  # (B, N, D_shape)
    agent_pe: torch.Tensor  # (B, N, D_model)

    # Optional fields
    input_step: Optional[torch.Tensor] = None  # (B, T)
    in_backward_prediction: Optional[torch.Tensor] = None  # (B, )

    def sanity_check(self):
        # Check shapes
        B, T, N = self.input_action.shape
        assert self.input_action_valid_mask.shape == (
            B, T, N
        ), f"Expected shape {(B, T, N)}, got {self.input_action_valid_mask.shape}"
        assert self.agent_delta.shape == (
            B, T, N, self.agent_delta.shape[-1]
        ), f"Expected shape {(B, T, N, self.agent_delta.shape[-1])}, got {self.agent_delta.shape}"
        assert self.agent_position.shape in [
            (B, N, self.agent_position.shape[-1]), (B, T, N, self.agent_position.shape[-1])
        ], f"Unexpected shape {self.agent_position.shape}"
        assert self.agent_heading.shape in [(B, N), (B, T, N)], f"Unexpected shape {self.agent_heading.shape}"
        assert self.agent_velocity.shape in [
            (B, N, self.agent_velocity.shape[-1]), (B, T, N, self.agent_velocity.shape[-1])
        ], f"Unexpected shape {self.agent_velocity.shape}"
        assert self.agent_type.shape in [(B, N), (B, T, N)], f"Unexpected shape {self.agent_type.shape}"
        assert self.agent_shape.shape == (
            B, N, self.agent_shape.shape[-1]
        ), f"Expected shape {(B, N, self.agent_shape.shape[-1])}, got {self.agent_shape.shape}"
        assert self.agent_pe.shape == (
            B, N, self.agent_pe.shape[-1]
        ), f"Expected shape {(B, N, self.agent_pe.shape[-1])}, got {self.agent_pe.shape}"

        # Check optional fields
        if self.input_step is not None:
            assert self.input_step.shape == (B, T), f"Expected shape {(B, T)}, got {self.input_step.shape}"
        if self.in_backward_prediction is not None:
            assert self.in_backward_prediction.shape == (
                B,
            ), f"Expected shape {(B,)}, got {self.in_backward_prediction.shape}"

    def __post_init__(self):
        if self.input_step is None:
            B, T, _ = self.input_action.shape
            self.input_step = torch.arange(T, device=self.input_action.device).unsqueeze(0).expand(B, T)


# @dataclass
# class HistoryData:
#     # History/cache for autoregressive decoding
#     modeled_agent_position_history: Optional[torch.Tensor] = None
#     modeled_agent_velocity_history: Optional[torch.Tensor] = None
#     modeled_agent_heading_history: Optional[torch.Tensor] = None
#     modeled_agent_valid_mask_history: Optional[torch.Tensor] = None
#     modeled_agent_step_history: Optional[torch.Tensor] = None

# @dataclass
# class MotionDecoderData:
#     # Top-level container for all inputs to MotionDecoder
#     in_evaluation: bool
#     encoder: EncoderOutput
#     decoder: DecoderInput
#     history: Optional[HistoryData] = None
#     cache: Optional[Any] = None  # For past key/value caching or similar

###############################################################################
# MotionDecoderGPT with the New Data Structures
###############################################################################


class MotionDecoderGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        self.num_actions = get_action_dim(self.config)
        dropout = self.config.MODEL.DROPOUT
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD

        use_adaln = False
        self.use_adaln = use_adaln

        # TODO: Remove there:
        assert self.config.MODEL.IS_V7 is True
        assert self.config.SIMPLE_RELATION is True
        assert self.config.SIMPLE_RELATION_FACTOR == 1
        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        assert self.add_pe_for_token is True
        assert self.config.MODEL.NAME in ['gpt']
        assert self.config.ADD_CONTOUR_RELATION
        assert self.config.SIMPLE_RELATION is True
        assert self.config.MODEL.ADD_RELATION_TO_V is False
        assert self.config.REMOVE_AGENT_FROM_SCENE_ENCODER is True

        simple_relation = self.config.SIMPLE_RELATION
        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        self.decoder = MultiCrossAttTransformerDecoder(
            decoder_layer=MultiCrossAttTransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dropout=dropout,
                use_adaln=use_adaln,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor,
                is_v7=True,
                update_relation=self.config.UPDATE_RELATION,
                add_relation_to_v=self.config.MODEL.ADD_RELATION_TO_V,
                remove_rel_norm=self.config.REMOVE_REL_NORM
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
        )
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model,
            mlp_channels=[d_model, self.num_actions],
            ret_before_act=True,
        )
        if self.use_adaln:
            self.prediction_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        else:
            self.prediction_prenorm = nn.LayerNorm(d_model)

        relation_d_model = d_model // simple_relation_factor
        self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
            input_dim=12,
            hidden_dim=relation_d_model,
            num_freq_bands=64,
        )
        self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
            input_dim=12,
            hidden_dim=relation_d_model,
            num_freq_bands=64,
        )
        self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
            input_dim=3,
            hidden_dim=relation_d_model,
            num_freq_bands=64,
        )

        self.type_embed = common_layers.Tokenizer(
            num_actions=constants.NUM_TYPES, d_model=d_model, add_one_more_action=False
        )
        self.action_embed = common_layers.Tokenizer(
            num_actions=self.num_actions, d_model=d_model, add_one_more_action=True
        )
        self.shape_embed = common_layers.build_mlps(
            c_in=3,
            mlp_channels=[d_model, d_model],
            ret_before_act=True,
        )
        self.agent_id_embed = common_layers.Tokenizer(
            num_actions=self.config.PREPROCESSING.MAX_AGENTS, d_model=self.d_model, add_one_more_action=False
        )

        self.motion_embed = fourier_embedding.FourierEmbedding(
            input_dim=6,
            hidden_dim=d_model,
            num_freq_bands=64,
        )

        tokenizer = get_tokenizer(self.config)
        motion_features = tokenizer.get_motion_feature()
        if tokenizer.use_type_specific_bins:
            motion_features = torch.cat([motion_features, torch.zeros(1, 3, 4)], dim=0)
        else:
            motion_features = torch.cat([motion_features, torch.zeros(1, 4)], dim=0)
        self.tokenizer = tokenizer
        self.register_buffer("motion_features", motion_features)

        self.special_token_embed = common_layers.Tokenizer(
            num_actions=4, d_model=self.d_model, add_one_more_action=False
        )

        if self.config.BACKWARD_PREDICTION:
            self.in_backward_prediction_embed = common_layers.Tokenizer(
                num_actions=2, d_model=self.d_model, add_one_more_action=False
            )
        if self.use_adaln:
            self.initialize_weights_for_adaln()

    def initialize_weights_for_adaln(self):
        for block in self.decoder.layers:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def randomize_modeled_agent_id(self, data: MotionDecoderData, clip_agent_id=False):
        modeled_agent_id = data.decoder.agent_id  # Was: input_dict["decoder/agent_id"]
        if not self.config.MODEL.RANDOMIZE_AGENT_ID:
            if clip_agent_id:
                modeled_agent_id = mode_agent_id(
                    modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True
                )
            return modeled_agent_id

        if clip_agent_id:
            modeled_agent_id = mode_agent_id(
                modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True
            )
        B, N = modeled_agent_id.shape
        weights = torch.ones(self.config.PREPROCESSING.MAX_AGENTS).expand(B, -1)
        if N > self.config.PREPROCESSING.MAX_AGENTS:
            num_samples = self.config.PREPROCESSING.MAX_AGENTS
            new_modeled_agent_id = torch.full_like(modeled_agent_id, num_samples - 1)
            new_modeled_agent_id[:, :num_samples] = torch.multinomial(
                weights, num_samples=num_samples, replacement=False
            ).to(modeled_agent_id)
            new_modeled_agent_id[modeled_agent_id == -1] = -1
        else:
            num_samples = N
            new_modeled_agent_id = torch.multinomial(
                weights, num_samples=num_samples, replacement=False
            ).to(modeled_agent_id)
            new_modeled_agent_id[modeled_agent_id == -1] = -1
        return new_modeled_agent_id

    def _legacy_data_dict_to_new_data_structure(self, data_dict):
        encoder = EncoderOutput(
            scenario_token=data_dict["encoder/scenario_token"],
            scenario_valid_mask=data_dict["encoder/scenario_valid_mask"],
            scenario_position=data_dict["encoder/scenario_position"],
            scenario_heading=data_dict["encoder/scenario_heading"],
            modeled_agent_pe=data_dict["encoder/modeled_agent_pe"] if "encoder/modeled_agent_pe" in data_dict else None,
        )
        decoder = DecoderInput(
            input_action=data_dict["decoder/input_action"],
            modeled_agent_delta=data_dict["decoder/modeled_agent_delta"],
            input_action_valid_mask=data_dict["decoder/input_action_valid_mask"],
            modeled_agent_position=data_dict["decoder/modeled_agent_position"],
            modeled_agent_heading=data_dict["decoder/modeled_agent_heading"],
            modeled_agent_velocity=data_dict["decoder/modeled_agent_velocity"],
            agent_type=data_dict["decoder/agent_type"],
            current_agent_shape=data_dict["decoder/current_agent_shape"],
            agent_id=data_dict["decoder/agent_id"],
            input_step=data_dict["decoder/input_step"] if "decoder/input_step" in data_dict else None,
            # label_safety=data_dict["decoder/label_safety"],
            randomized_modeled_agent_id=data_dict["decoder/randomized_modeled_agent_id"]
            if "decoder/randomized_modeled_agent_id" in data_dict else None,
            in_backward_prediction=data_dict["decoder/in_backward_prediction"]
            if "decoder/in_backward_prediction" in data_dict else None,
        )
        # TODO: Better handle history and cache.
        history = HistoryData(
            modeled_agent_position_history=data_dict["history/modeled_agent_position_history"]
            if "history/modeled_agent_position_history" in data_dict else None,
            modeled_agent_velocity_history=data_dict["history/modeled_agent_velocity_history"]
            if "history/modeled_agent_velocity_history" in data_dict else None,
            modeled_agent_heading_history=data_dict["history/modeled_agent_heading_history"]
            if "history/modeled_agent_heading_history" in data_dict else None,
            modeled_agent_valid_mask_history=data_dict["history/modeled_agent_valid_mask_history"]
            if "history/modeled_agent_valid_mask_history" in data_dict else None,
            modeled_agent_step_history=data_dict["history/modeled_agent_step_history"]
            if "history/modeled_agent_step_history" in data_dict else None,
        )
        return MotionDecoderData(
            in_evaluation=data_dict["in_evaluation"],
            encoder=encoder,
            decoder=decoder,
            history=history,
            cache=data_dict["cache"] if "cache" in data_dict else None,
        )

    def forward(self, data: MotionDecoderData, use_cache=False, a2a_knn=None, a2t_knn=None, a2s_knn=None):

        if isinstance(data, dict):
            data = self._legacy_data_dict_to_new_data_structure(data)

        in_evaluation = data.in_evaluation

        # Process scene (encoder) embedding
        scene_token = data.encoder.scenario_token
        scenario_valid_mask = data.encoder.scenario_valid_mask

        # Process action (decoder) embedding
        input_action = data.decoder.input_action
        modeled_agent_delta = data.decoder.modeled_agent_delta
        B, T_skipped, N = input_action.shape

        if in_evaluation:
            assert data.decoder.randomized_modeled_agent_id is not None, \
                "Need to provide randomized modeled agent id for evaluation! Please call randomize_modeled_agent_id()"
            new_modeled_agent_id = data.decoder.randomized_modeled_agent_id
        else:
            new_modeled_agent_id = self.randomize_modeled_agent_id(data, clip_agent_id=False)
        modeled_agent_pe = self.agent_id_embed(new_modeled_agent_id)

        assert modeled_agent_pe.shape == (B, N, self.d_model), modeled_agent_pe.shape
        modeled_agent_pe = modeled_agent_pe[:, None].expand(B, T_skipped, N, self.d_model)

        action_valid_mask = data.decoder.input_action_valid_mask
        agent_pos = data.decoder.modeled_agent_position
        agent_heading = data.decoder.modeled_agent_heading

        # input_step is set in __post_init__ if not provided.
        agent_step = data.decoder.input_step.reshape(1, T_skipped, 1).expand(B, T_skipped, N)

        # Shape and type embeddings
        type_emb = self.type_embed(data.decoder.agent_type)[:, None].expand(B, T_skipped, N, self.d_model)
        shape_emb = self.shape_embed(data.decoder.current_agent_shape)[:, None].expand(B, T_skipped, N, self.d_model)

        valid_actions = input_action[action_valid_mask]
        is_start_actions = valid_actions == START_ACTION
        special_tok = torch.full_like(valid_actions, 0).int()
        special_tok[is_start_actions] = 1
        valid_actions[is_start_actions] = -1
        if self.config.BACKWARD_PREDICTION:
            is_end_actions = valid_actions == END_ACTION
            special_tok[is_end_actions] = 2
            valid_actions[is_end_actions] = -1
        special_tok_emb = self.special_token_embed(special_tok)
        if self.config.BACKWARD_PREDICTION:
            if data.decoder.in_backward_prediction is None:
                data.decoder.in_backward_prediction = valid_actions.new_zeros(B, T_skipped, N)
            in_backward_full = data.decoder.in_backward_prediction.reshape(B, 1, 1).expand(B, T_skipped, N)
            in_backward = in_backward_full[action_valid_mask].int()
            in_backward_prediction_embed = self.in_backward_prediction_embed(in_backward)
            special_tok_emb += in_backward_prediction_embed
        action_emb = self.action_embed(valid_actions)

        if self.tokenizer.use_type_specific_bins:
            agent_type = data.decoder.agent_type
            agent_type = agent_type - 1
            agent_type[agent_type < 0] = 0
            agent_type = agent_type.reshape(B, 1, N).expand(B, T_skipped, N)
            agent_type = agent_type[action_valid_mask]
            agent_type = agent_type.reshape(-1, 1, 1, 1).expand(-1, self.motion_features.shape[0], 1, 4)
            motion_feat = self.motion_features.reshape(1, -1, 3, 4).expand(agent_type.shape[0], -1, 3, 4)
            motion_feat = torch.gather(motion_feat, dim=-2, index=agent_type).squeeze(-2)
        else:
            motion_feat = self.motion_features.reshape(1, -1, 4).expand(valid_actions.shape[0], -1, 4)
        valid_actions[valid_actions < 0] = self.num_actions
        valid_actions = valid_actions.reshape(-1, 1, 1).expand(-1, 1, 4)
        motion_feat = torch.gather(motion_feat, dim=-2, index=valid_actions).squeeze(-2)

        motion_feat = torch.cat([motion_feat, modeled_agent_delta[action_valid_mask]], dim=-1)

        action_token = self.motion_embed(
            continuous_inputs=motion_feat,
            categorical_embs=[
                special_tok_emb, modeled_agent_pe[action_valid_mask], type_emb[action_valid_mask],
                shape_emb[action_valid_mask], action_emb
            ]
        )
        action_token = utils.unwrap(action_token, action_valid_mask)
        assert action_token.shape == (B, T_skipped, N, self.d_model)
        assert action_valid_mask.shape == (B, T_skipped, N)

        condition_token = None
        if self.config.ACTION_LABEL.USE_SAFETY_LABEL:
            action_label_safety = self.action_label_tokenizer_safety(data.decoder.label_safety)
            condition_token = action_label_safety[:, None]
            if self.use_adaln:
                pass
            else:
                action_token += condition_token

        # Prepare agent-temporal relation data (permute BTND -> BNTD etc.)
        agent_pos_bntd = torch.permute(agent_pos, [0, 2, 1, 3])
        agent_heading_bnt = torch.permute(agent_heading, [0, 2, 1])
        agent_mask_bnt = torch.permute(action_valid_mask, [0, 2, 1])
        agent_step_bnt = torch.permute(agent_step, [0, 2, 1])

        if use_cache:
            self.update_cache(data)
            agent_pos_with_history = data.history.modeled_agent_position_history
            agent_heading_with_history = data.history.modeled_agent_heading_history
            agent_mask_with_history = data.history.modeled_agent_valid_mask_history
            agent_step_with_history = data.history.modeled_agent_step_history
            real_T = agent_mask_with_history.shape[1]
            key_pos = torch.permute(agent_pos_with_history, [0, 2, 1, 3]).flatten(0, 1)
            key_heading = torch.permute(agent_heading_with_history, [0, 2, 1]).flatten(0, 1)
            key_mask = torch.permute(agent_mask_with_history, [0, 2, 1]).flatten(0, 1)
            causal_valid_mask = None
            key_step = agent_step_with_history.reshape(1, 1, -1).expand(B, N, -1).flatten(0, 1)
        else:
            real_T = T_skipped
            key_pos = agent_pos_bntd.flatten(0, 1)
            key_heading = agent_heading_bnt.flatten(0, 1)
            key_mask = agent_mask_bnt.flatten(0, 1)
            key_step = agent_step_bnt.flatten(0, 1)
            causal_valid_mask = create_causal_mask(T=real_T, N=1, is_valid_mask=True).to(action_token.device)

        assert agent_pos_bntd.shape == (B, N, T_skipped, 2)

        a2t_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            agent_shape_no_time = data.decoder.current_agent_shape
            agent_length = agent_shape_no_time[..., 0]
            agent_width = agent_shape_no_time[..., 1]
            a2t_kwargs = dict(
                include_contour=True,
                query_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                query_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                key_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
                key_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
                non_agent_relation=False,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )

        if self.config.SIMPLE_RELATION:
            relation_func = relation.compute_relation_simple_relation
        else:
            relation_func = relation.compute_relation

        a2t_rel_feat, a2t_mask, _ = relation_func(
            query_pos=agent_pos_bntd.flatten(0, 1),
            query_heading=agent_heading_bnt.flatten(0, 1),
            query_valid_mask=agent_mask_bnt.flatten(0, 1),
            query_step=agent_step_bnt.flatten(0, 1),
            key_pos=key_pos,
            key_heading=key_heading,
            key_valid_mask=key_mask,
            key_step=key_step,
            hidden_dim=self.d_model,
            causal_valid_mask=causal_valid_mask,
            knn=None,
            max_distance=None,
            return_pe=False,
            **a2t_kwargs
        )
        a2t_info = get_edge_info_new(
            q_k_valid_mask=a2t_mask,
            q_k_relation=a2t_rel_feat,
            relation_model=self.relation_embed_a2t,
            relation_model_v=self.relation_embed_a2t_v if self.config.MODEL.ADD_RELATION_TO_V else None
        )

        a2a_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(0, 1)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(0, 1)
            a2a_kwargs = dict(
                include_contour=True,
                query_width=w,
                query_length=l,
                key_width=w,
                key_length=l,
                non_agent_relation=False,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )
        a2a_rel_feat, a2a_mask, _ = relation_func(
            query_pos=agent_pos.flatten(0, 1),
            query_heading=agent_heading.flatten(0, 1),
            query_valid_mask=action_valid_mask.flatten(0, 1),
            query_step=agent_step.flatten(0, 1),
            key_pos=agent_pos.flatten(0, 1),
            key_heading=agent_heading.flatten(0, 1),
            key_valid_mask=action_valid_mask.flatten(0, 1),
            key_step=agent_step.flatten(0, 1),
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=a2a_knn if a2a_knn is not None else self.config.MODEL.A2A_KNN,
            max_distance=self.config.MODEL.A2A_DISTANCE,
            return_pe=False,
            **a2a_kwargs
        )
        a2a_info = get_edge_info_new(
            q_k_valid_mask=a2a_mask,
            q_k_relation=a2a_rel_feat,
            relation_model=self.relation_embed_a2a,
            relation_model_v=self.relation_embed_a2a_v if self.config.MODEL.ADD_RELATION_TO_V else None
        )

        a2s_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            kw = torch.zeros_like(data.encoder.scenario_position[..., 0])
            a2s_kwargs = dict(
                include_contour=True,
                query_width=w,
                query_length=l,
                key_width=kw,
                key_length=kw,
                non_agent_relation=True,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )
        a2s_rel_feat, a2s_mask, a2s_indices = relation_func(
            query_pos=agent_pos.flatten(1, 2),
            query_heading=agent_heading.flatten(1, 2),
            query_valid_mask=action_valid_mask.flatten(1, 2),
            query_step=agent_step.flatten(1, 2),
            key_pos=data.encoder.scenario_position,
            key_heading=data.encoder.scenario_heading,
            key_valid_mask=scenario_valid_mask,
            key_step=agent_pos.new_zeros(B, data.encoder.scenario_position.shape[1]),
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=a2s_knn if a2s_knn is not None else self.config.MODEL.A2S_KNN,
            max_distance=self.config.MODEL.A2S_DISTANCE,
            gather=False,
            return_pe=False,
            **a2s_kwargs
        )
        a2s_info = get_edge_info_new(
            q_k_valid_mask=a2s_mask,
            q_k_relation=a2s_rel_feat,
            relation_model=self.relation_embed_a2s,
            relation_model_v=self.relation_embed_a2s_v if self.config.MODEL.ADD_RELATION_TO_V else None
        )

        past_key_value_list = None
        if use_cache:
            past_key_value_list = data.cache

        decoded_tokens = self.decoder(
            agent_token=action_token,
            scene_token=scene_token,
            a2a_info=a2a_info,
            a2t_info=a2t_info,
            a2s_info=a2s_info,
            condition_token=condition_token if self.use_adaln else None,
            use_cache=use_cache,
            past_key_value_list=past_key_value_list
        )

        if use_cache:
            decoded_tokens, past_key_value_list = decoded_tokens
            for l in past_key_value_list:
                if l:
                    l.append((B * N, real_T))
            data.cache = past_key_value_list

        if self.config.USE_DIFFUSION:
            # Attach decoded tokens to the decoder part of our data structure.
            data.decoder.decoded_tokens = decoded_tokens
            return data

        if self.use_adaln:
            output_tokens = self.prediction_adaln_norm(decoded_tokens[action_valid_mask])
            shift, scale = self.adaln_modulation(output_tokens).chunk(2, dim=-1)
            output_tokens = utils.modulate(output_tokens, shift, scale)
        else:
            output_tokens = self.prediction_prenorm(decoded_tokens[action_valid_mask])
        logits = utils.unwrap(self.prediction_head(output_tokens), action_valid_mask)

        assert logits.shape == (B, T_skipped, N, self.num_actions)
        data.decoder.output_logit = logits

        return data

    def update_cache(self, data: MotionDecoderData):
        assert self.config.EVALUATION.USE_CACHE
        if data.history is None:
            data.history = HistoryData(
                modeled_agent_position_history=data.decoder.modeled_agent_position.clone(),
                modeled_agent_velocity_history=data.decoder.modeled_agent_velocity.clone(),
                modeled_agent_heading_history=data.decoder.modeled_agent_heading.clone(),
                modeled_agent_valid_mask_history=data.decoder.input_action_valid_mask.clone(),
                modeled_agent_step_history=data.decoder.input_step.clone()
            )
        else:
            data.history.modeled_agent_position_history = torch.cat(
                [data.history.modeled_agent_position_history, data.decoder.modeled_agent_position], dim=1
            )
            data.history.modeled_agent_velocity_history = torch.cat(
                [data.history.modeled_agent_velocity_history, data.decoder.modeled_agent_velocity], dim=1
            )
            data.history.modeled_agent_heading_history = torch.cat(
                [data.history.modeled_agent_heading_history, data.decoder.modeled_agent_heading], dim=1
            )
            data.history.modeled_agent_valid_mask_history = torch.cat(
                [data.history.modeled_agent_valid_mask_history, data.decoder.input_action_valid_mask], dim=1
            )
            data.history.modeled_agent_step_history = torch.cat(
                [data.history.modeled_agent_step_history, data.decoder.input_step], dim=0
            )
