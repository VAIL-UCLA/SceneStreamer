from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import softmax

from scenestreamer.dataset import constants
from scenestreamer.dataset.preprocessor import TG_SKIP_STEP, NUM_TG_MULTI
from scenestreamer.models import relation
from scenestreamer.models.layers import common_layers, fourier_embedding
from scenestreamer.models.layers import polyline_encoder
from scenestreamer.models.layers.decoder_layer import _get_clones
from scenestreamer.models.layers.gpt_encoder_layer import SelfAttTransformerEncoder, SelfAttTransformerEncoderLayer
from scenestreamer.models.motion_decoder import create_causal_mask
from scenestreamer.tokenization import get_action_dim, get_tokenizer, START_ACTION
from scenestreamer.tokenization.trafficgen_tokenizers import TrafficGenTokenizerAutoregressive
from scenestreamer.utils import utils


def get_num_tg(N):
    return N * NUM_TG_MULTI + 2


def mode_agent_id(agent_id, max_agents, fill_negative_1=False):
    # As most of the "modeled agents" are in the first few agents, we want to remap those useless agents to latter
    # positions.
    agent_id = agent_id.clone()
    if fill_negative_1:
        agent_id[torch.logical_or(agent_id >= max_agents, agent_id < 0)] = -1
    else:
        agent_id[torch.logical_or(agent_id >= max_agents, agent_id < 0)] = max_agents - 1
    return agent_id


def get_edge_info_for_scenestreamer(*, q_k_valid_mask, q_k_relation, relation_model, relation_model_1d=None, require_relation_pairwise=None):
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
    edge_relation = q_k_relation[batch_ind, q_ind, k_ind]

    edge_features_v = None

    assert relation_model is not None
    if require_relation_pairwise is not None:
        require_relation = require_relation_pairwise[batch_ind, q_ind, k_ind]

        edge_feat = relation_model(edge_relation[require_relation])
        edge_features = utils.unwrap(edge_feat, require_relation)

        if relation_model_1d is not None:
            assert edge_relation.shape[-1] == 4
            edge_feat_1d = relation_model_1d(edge_relation[~require_relation][:, -1:])
            edge_features = utils.unwrap(edge_feat_1d, ~require_relation, existing=edge_features)

        # (edge_features[require_relation] == edge_feat_4d).all()
        # (edge_features[~require_relation] == edge_feat_1d).all()

    else:
        edge_features = relation_model(edge_relation)

    return {
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_features_v": edge_features_v,
    }


class MultiheadAttentionLayer(MessagePassing):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        simple_relation=False,
        simple_relation_factor=2,
        is_v7=False,
        update_relation=False,
        add_relation_to_v=None
    ):
        super(MultiheadAttentionLayer, self).__init__(aggr='add', node_dim=0)  # Aggregation method 'add'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert dropout == 0.0, "dropout is not supported"
        self.dropout = nn.Dropout(dropout)
        self.relation_head_dim = self.head_dim // simple_relation_factor
        self.to_q_relation = nn.Linear(d_model, d_model)
        self.to_k_r = nn.Linear(d_model // simple_relation_factor, d_model)
        self.to_v_r = nn.Linear(d_model // simple_relation_factor, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        q,
        k,
        edge_index,
        edge_features,
        edge_features_v=None,
        use_cache=False,
        cache=None,  #Relation=None
    ):
        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        # Compute linear projections
        x_dst = q
        x_src = k
        Q = self.to_q(x_dst).reshape(-1, self.n_heads * self.head_dim)
        K = self.to_k(x_src).reshape(-1, self.n_heads * self.head_dim)
        V = self.to_v(x_src).reshape(-1, self.n_heads * self.head_dim)

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            key_B, key_T = cache[2]

            K = K.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_key = past_key.reshape(key_B, key_T, self.n_heads * self.head_dim)
            K = torch.cat((past_key, K), dim=1)
            K = K.reshape(-1, self.n_heads * self.head_dim)

            V = V.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_value = past_value.reshape(key_B, key_T, self.n_heads * self.head_dim)
            V = torch.cat((past_value, V), dim=1)
            V = V.reshape(-1, self.n_heads * self.head_dim)

        assert edge_index[0].max() < K.shape[0], f"{edge_index[0].max()} >= {K.shape[0]}"
        assert edge_index[1].max() < Q.shape[0], f"{edge_index[1].max()} >= {Q.shape[0]}"

        if use_cache:
            new_cache = [K, V]
        else:
            new_cache = None

        Q_relation = self.to_q_relation(x_dst).reshape(-1, self.n_heads * self.head_dim)
        Q = torch.cat([Q, Q_relation], dim=-1)

        assert edge_features_v is None
        edge_features_v = edge_features
        edge_features = self.to_k_r(edge_features)
        edge_features_v = self.to_v_r(edge_features_v)

        # Propagate messages using edge_index
        out, new_edge_features = self.propagate(
            edge_index=edge_index,
            # x_dst=x_dst.reshape(-1, self.n_heads * self.head_dim),
            q=Q,
            k=K,
            v=V,
            edge_features=edge_features,
            edge_features_v=edge_features_v,
        )

        # Project the output back to original dimension
        out = out.reshape(B, Lq, D)
        if new_edge_features is not None:
            new_edge_features = new_edge_features.reshape(-1, D)
        out = self.out(out)
        return out, new_cache, new_edge_features  #, edge_features, edge_features_v

    def message(
        self, q_i, k_j, v_j, edge_features, edge_features_v, index, ptr, edge_index, edge_index_i, edge_index_j,
        relation
    ):
        k_j = k_j.reshape(-1, self.n_heads, self.head_dim)
        v_j = v_j.reshape(-1, self.n_heads, self.head_dim)
        q_i, q_relation = q_i[:, :self.n_heads * self.head_dim], q_i[:, self.n_heads * self.head_dim:]
        # Compute attention scores
        q_i = q_i.reshape(-1, self.n_heads, self.head_dim)
        q_relation = q_relation.reshape(-1, self.n_heads, self.head_dim)
        edge_features = edge_features.reshape(-1, self.n_heads, self.head_dim)
        attn_scores = (q_i * k_j).sum(dim=-1) / self.head_dim**0.5  # Scaled dot-product
        attn_scores_relation = (q_relation * edge_features).sum(dim=-1) / self.head_dim**0.5
        attn_scores = attn_scores + attn_scores_relation
        attn_weights = softmax(attn_scores, index=index, ptr=ptr)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights
        if edge_features_v is not None:
            edge_features_v = edge_features_v.reshape(-1, self.n_heads, self.head_dim)
            v_j = v_j + edge_features_v
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights
        return v_j * attn_weights.unsqueeze(-1), None

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        raw_inputs, new_edge_features = inputs
        inputs = super().aggregate(raw_inputs, index, ptr, dim_size)
        if new_edge_features is not None:
            new_edge_features = new_edge_features + raw_inputs
        return inputs, new_edge_features


class SceneEncoderGPT(nn.Module):
    def __init__(self, config, relation_embed):
        super().__init__()
        self.config = config
        self.d_model = self.config.MODEL.D_MODEL
        self.num_layers = self.config.MODEL.NUM_ATTN_LAYERS
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        dropout = self.config.MODEL.DROPOUT
        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=constants.MAP_FEATURE_STATE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_pre_layers=1,
            out_channels=self.d_model,
        )
        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        simple_relation = self.config.SIMPLE_RELATION
        self.relation_embed = relation_embed
        self.encoder = SelfAttTransformerEncoder(
            decoder_layer=SelfAttTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor,
                dropout=dropout,
                update_relation=self.config.UPDATE_RELATION,
                add_relation_to_v=self.config.MODEL.ADD_RELATION_TO_V,
                remove_rel_norm=self.config.REMOVE_REL_NORM,
            ),
            num_layers=self.num_layers,
        )
        self.out = common_layers.build_mlps(
            c_in=self.d_model, mlp_channels=[self.d_model], ret_before_act=True,
        )
        self.out_prenorm = nn.LayerNorm(self.d_model)

    def forward(self, input_dict):
        # ===== Get shape =====
        B, M, num_vector, D_vector = input_dict["encoder/map_feature"].shape
        # ===== Embed map feature =====
        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_position = input_dict["encoder/map_position"]
        map_heading = input_dict["encoder/map_heading"]
        map_token_valid_mask = input_dict["encoder/map_valid_mask"]
        map_token = self.map_polyline_encoder(map_feature, map_valid_mask)
        assert map_token.shape == (B, M, self.d_model)
        x = map_token  # [map_token, traffic_light_token]
        x_pos = map_position  # [map_position, traffic_light_position]
        x_heading = map_heading  # [map_heading, traffic_light_heading]
        x_mask = map_token_valid_mask  # [map_token_valid_mask, tlmask]
        assert torch.all(x_mask.sum(dim=-1) > 0)
        rel_feat, rel_mask, require_relation_pairwise = relation.compute_relation_for_scenestreamer(
            query_pos=x_pos,
            query_heading=x_heading,
            query_valid_mask=x_mask,
            key_pos=x_pos,
            key_heading=x_heading,
            key_valid_mask=x_mask,
            # hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=self.config.SCENESTREAMER_ATTENTION_KNN,
            max_distance=self.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE,
            gather=False,
            # return_pe=False,
            non_agent_relation=True,
            require_relation=None,
            # per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION,
        )
        scene_info = get_edge_info_for_scenestreamer(
            q_k_valid_mask=rel_mask,
            q_k_relation=rel_feat,
            relation_model=self.relation_embed,
        )
        x = self.encoder(
            scene_tokens=x,
            scene_info=scene_info,
            edge_features=scene_info["edge_features"],
            edge_features_v=scene_info["edge_features_v"]
        )
        x = self.out_prenorm(x[x_mask])
        x = self.out(x)  # .reshape(list(x.shape[:-1]) + [self.d_model])
        x = utils.unwrap(x, x_mask)
        input_dict["model/map_token"] = x
        return input_dict


class TransformerBlock(nn.Module):
    """
    A single transformer block that uses adaptive layer normalization.
    It includes a self-attention layer and a feed-forward network.
    """

    def __init__(self, hidden_size, num_heads, conditioning_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=False)
        self.adaln1 = common_layers.AdaLayerNorm(hidden_size, conditioning_dim, batch_first=False)
        self.adaln2 = common_layers.AdaLayerNorm(hidden_size, conditioning_dim, batch_first=False)

        # Simple feed-forward network.
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.ReLU(), nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, z, attn_mask=None, key_padding_mask=None):
        """
        x: Tensor of shape [seq_len, B, hidden_size]
        z: Conditioning tensor of shape [B, conditioning_dim]
        attn_mask: Optional attention mask for self-attention.
        key_padding_mask: Optional mask for padded positions.
        """
        # Self-attention with pre-normalization using AdaLN.
        # We apply AdaLN to x before attention.
        x_norm = self.adaln1(x, z)
        if key_padding_mask is not None:
            assert attn_mask.dtype == key_padding_mask.dtype
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=True
        )
        x = x + self.dropout(attn_output)

        # Feed-forward network with pre-normalization.
        x_norm = self.adaln2(x, z)
        ff_output = self.ff(x_norm)
        x = x + self.dropout(ff_output)
        return x


class TrafficgenPredictionHead(nn.Module):
    def __init__(
            self,
            vocab_size,
            type_size,
            hidden_size,
            map_id_size,
            num_heads,
            num_layers,
            conditioning_dim,
            max_seq_len=512,
            dropout=0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        # self.token_embedding = token_embedding
        # self.map_id_embedding = common_layers.Tokenizer(vocab_size, hidden_size, add_one_more_action=False)

        self.offset_token_embedding = common_layers.Tokenizer(vocab_size, hidden_size, add_one_more_action=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        # ===== agent state prediction head =====
        # Stack of transformer blocks.
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, conditioning_dim, dropout) for _ in range(num_layers)]
        )
        # Final normalization (can be standard LN).
        self.ln_final = nn.LayerNorm(hidden_size)
        # Project back to vocabulary logits.
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.type_output_layer = nn.Linear(hidden_size, type_size)

        # ===== map id prediction head =====
        d_model = hidden_size
        self.map_id_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, map_id_size], ret_before_act=True,
        )
        # self.dest_id_head = common_layers.build_mlps(
        #     c_in=d_model, mlp_channels=[d_model, map_id_size], ret_before_act=True,
        # )

    def forward(self, offset, trafficgen_token, key_padding_mask=None):
        B, T, G, D = trafficgen_token.shape
        assert offset.shape[0] == B
        assert offset.shape[1] == T
        assert offset.shape[3] == 9
        N = offset.shape[2]

        # Remove last agent dest id and sequence_eos:
        trafficgen_token = trafficgen_token[:, :, 1:-1]
        trafficgen_token = trafficgen_token.reshape(B, T, N, NUM_TG_MULTI, D)

        # The input tokens sequence: action_sos, map_id, agent_state, dest_map_id, action_eos

        # ===== sequence_sos/last_agent_dest_id -> agent_type =====
        agent_type_token = trafficgen_token[:, :, :, 0]
        agent_type_logits = self.type_output_layer(agent_type_token)
        assert agent_type_logits.shape == (B, T, N, 3)

        # ===== agent type -> map_id =====
        # To process the first token:
        map_id_token = trafficgen_token[:, :, :, 1]
        map_id_logit = self.map_id_head(map_id_token)
        assert map_id_logit.shape[:3] == (B, T, N)

        # ===== map_id -> agent_state =====
        # To process the second token whose output is agent_state:
        z = trafficgen_token[:, :, :, 2].flatten(0, 2)  # B*T*N, D
        offset_flattened = offset.flatten(0, 2)  # B*T*N, 8

        assert offset_flattened.dim() == 2, "Input tensor must have shape [B, seq_len]"
        assert z.dim() == 2, "Conditioning tensor must have shape [B, conditioning_dim]"

        # Compute token embeddings.
        emb = self.offset_token_embedding(offset_flattened)
        # emb = torch.cat([torch.zeros_like(type_emb), type_emb, emb], dim=1)
        seq_len = emb.size(1)
        assert seq_len == 9
        emb = emb + self.pos_embedding[:, :seq_len, :]
        h = emb.transpose(0, 1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)
        attn_mask = attn_mask < 0
        for layer in self.layers:
            h = layer(h, z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Final normalization.
        h = self.ln_final(h)
        # Transpose back to [B, seq_len, hidden_size]
        h = h.transpose(0, 1)
        agent_state_logits = self.output_layer(h)
        agent_state_logits = agent_state_logits.reshape(B, T, N, 9, -1)

        # ===== agent_state -> dest_id =====
        # To process the third token whose output is dest map id:
        # dest_id_token = trafficgen_token[:, :, :, 3]
        # dest_id_logit = self.dest_id_head(dest_id_token)
        # assert dest_id_logit.shape[:3] == (B, T, N)

        return map_id_logit, agent_type_logits, agent_state_logits, None

    @torch.no_grad()
    def generate(self, z, greedy=False):
        """
        Autoregressively generate a sequence conditioned on latent vector z.

        z: FloatTensor of shape [B, conditioning_dim]
        max_length: Maximum length to generate (including <s> and <e> tokens).
        greedy: If True, use argmax sampling; otherwise, sample from the distribution.

        Returns:
            generated: LongTensor of shape [B, generated_seq_len] (including the starting <s>).
        """
        assert z.ndim == 2

        B = z.size(0)
        max_length = self.max_seq_len
        device = z.device
        # Start each sequence with the <s> token.
        input_seq = torch.full((B, 1), -1, dtype=torch.long, device=device)

        # key_padding_mask = torch.zeros(B, 1, dtype=torch.float32, device=device)
        # key_padding_valid_mask_bool = torch.ones(B, 1, dtype=torch.bool, device=device)
        # key_padding_mask = (~key_padding_valid_mask_bool).clone()

        for step in range(max_length - 1):  # already have one token

            assert input_seq.dim() == 2, "Input tensor must have shape [B, seq_len]"

            # Compute token embeddings.
            emb = self.offset_token_embedding(input_seq)
            # emb = torch.cat([torch.zeros_like(type_emb), type_emb, emb], dim=1)
            seq_len = emb.size(1)
            # assert seq_len == 9
            emb = emb + self.pos_embedding[:, :seq_len, :]
            h = emb.transpose(0, 1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)
            attn_mask = attn_mask < 0
            for layer in self.layers:
                h = layer(h, z, attn_mask=attn_mask, key_padding_mask=None)
            # Final normalization.
            h = self.ln_final(h)
            # Transpose back to [B, seq_len, hidden_size]
            h = h.transpose(0, 1)
            agent_state_logits = self.output_layer(h)
            last_logits = agent_state_logits[:, -1:]

            from scenestreamer.infer.scenestreamer_motion import sample_action
            next_token, _ = sample_action(last_logits, sampling_method="softmax")

            # Append the predicted token.
            input_seq = torch.cat([input_seq, next_token], dim=1)
        return input_seq


class SceneStreamer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ===== A bunch of hyper-parameters and assertions =====
        self.config = config
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        self.num_actions = get_action_dim(self.config)
        dropout = self.config.MODEL.DROPOUT
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        assert self.config.MODEL.NAME == "scenestreamer"
        assert self.add_pe_for_token
        self.use_destination = self.config.USE_DESTINATION
        simple_relation = self.config.SIMPLE_RELATION
        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        is_v7 = self.config.MODEL.IS_V7
        self.is_v7 = is_v7
        assert is_v7 is True
        assert simple_relation is True
        assert self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE is False
        self.start_action_id = config.PREPROCESSING.MAX_MAP_FEATURES
        self.end_action_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1
        self.no_tg = config.get("SCENESTREAMER_NO_TG", False)

        # ===== Build tokenizer =====
        tokenizer = get_tokenizer(self.config)
        motion_features = tokenizer.get_motion_feature()
        if tokenizer.use_type_specific_bins:
            motion_features = torch.cat([motion_features, torch.zeros(1, 3, 4)], dim=0)
        else:
            motion_features = torch.cat([motion_features, torch.zeros(1, 4)], dim=0)
        self.motion_tokenizer = tokenizer

        # ===== Build the relative continuous embedding =====
        relation_d_model = d_model // simple_relation_factor
        self.relation_embed_4d = fourier_embedding.FourierEmbedding(
            input_dim=4, hidden_dim=relation_d_model, num_freq_bands=64,
        )
        self.relation_embed_3d = fourier_embedding.FourierEmbedding(
            input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64,
        )
        self.relation_embed_1d = fourier_embedding.FourierEmbedding(
            input_dim=1, hidden_dim=relation_d_model, num_freq_bands=64,
        )

        # ===== Build map features embedding =====
        self.map_encoder = SceneEncoderGPT(config=self.config, relation_embed=self.relation_embed_3d)

        # ===== Build the egocentric discrete embedding =====
        # Adding 2 tokens for trafficgen
        num_total_map_actions = self.config.PREPROCESSING.MAX_MAP_FEATURES + 7
        self.map_id_embed = common_layers.Tokenizer(
            num_actions=num_total_map_actions, d_model=d_model, add_one_more_action=True
        )

        trafficgen_sequence_sos_id = config.PREPROCESSING.MAX_MAP_FEATURES
        trafficgen_sequence_eos_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1
        trafficgen_sequence_pad_id = config.PREPROCESSING.MAX_MAP_FEATURES + 2
        veh_id = config.PREPROCESSING.MAX_MAP_FEATURES + 3
        ped_id = config.PREPROCESSING.MAX_MAP_FEATURES + 4
        cyc_id = config.PREPROCESSING.MAX_MAP_FEATURES + 5
        trafficgen_agent_sos_id = config.PREPROCESSING.MAX_MAP_FEATURES + 6
        self.trafficgen_sequence_sos_id = trafficgen_sequence_sos_id
        self.trafficgen_sequence_eos_id = trafficgen_sequence_eos_id
        self.trafficgen_sequence_pad_id = trafficgen_sequence_pad_id
        self.trafficgen_agent_sos_id = trafficgen_agent_sos_id
        self.veh_id = veh_id
        self.ped_id = ped_id
        self.cyc_id = cyc_id

        N = 128
        G = get_num_tg(N)

        self.traffic_light_id_embed = common_layers.Tokenizer(
            num_actions=self.config.PREPROCESSING.MAX_TRAFFIC_LIGHTS, d_model=self.d_model, add_one_more_action=True
        )
        self.agent_id_embed = common_layers.Tokenizer(
            num_actions=N, d_model=self.d_model, add_one_more_action=True
        )
        self.action_embed = common_layers.Tokenizer(
            num_actions=self.num_actions, d_model=d_model, add_one_more_action=True
        )
        self.traffic_light_state_embed = common_layers.Tokenizer(
            num_actions=4, d_model=self.d_model, add_one_more_action=True
        )

        # ===== Build the egocentric continuous embedding =====
        self.shape_embed = common_layers.build_mlps(
            c_in=3, mlp_channels=[d_model, d_model], ret_before_act=True,
        )
        self.motion_embed = fourier_embedding.FourierEmbedding(
            input_dim=6, hidden_dim=d_model, num_freq_bands=64,
        )
        self.register_buffer("motion_features", motion_features)

        # ===== Build the backbone transformer =====
        self.decoder = SceneStreamerDecoder(
            decoder_layer=SceneStreamerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
        )

        # ===== Build the output head for different modalities =====
        num_traffic_light_states = 4
        self.traffic_light_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, num_traffic_light_states], ret_before_act=True, is_v7=is_v7,
            zero_init=is_v7
        )
        self.traffic_light_prenorm = nn.LayerNorm(d_model)

        self.motion_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, self.num_actions], ret_before_act=True, is_v7=is_v7, zero_init=is_v7
        )
        if self.config.MODEL.USE_MOTION_HEAD_PRENORM:
            self.motion_prenorm = nn.LayerNorm(d_model)
        else:
            self.motion_prenorm = None

        if self.no_tg:
            pass

        else:
            self.trafficgen_intra_step = common_layers.Tokenizer(
                num_actions=G, d_model=self.d_model, add_one_more_action=False
            )
            self.trafficgen_feat_embed = common_layers.build_mlps(
                c_in=8, mlp_channels=[d_model, d_model], ret_before_act=True,
            )
            self.trafficgen_head = TrafficgenPredictionHead(
                vocab_size=TrafficGenTokenizerAutoregressive.num_bins["position_x"],
                type_size=3,
                hidden_size=self.d_model,
                map_id_size=num_total_map_actions,
                num_heads=4,
                num_layers=3,
                conditioning_dim=self.d_model,
                max_seq_len=8 + 1,
                dropout=0.0
            )
            self.trafficgen_prenorm = nn.LayerNorm(d_model)
            self.trafficgen_tokenizer = TrafficGenTokenizerAutoregressive(self.config)

    def forward(self, input_dict):

        # ===== Build up some variables =====
        # in_evaluation = input_dict["in_evaluation"][0].item()
        B, T, N = input_dict["decoder/input_action"].shape
        _, _, L = input_dict["encoder/traffic_light_state"].shape
        _, _, G = input_dict["decoder/input_action_for_trafficgen"].shape

        # ===== Prepare map tokens =====
        input_dict = self.prepare_map_tokens(input_dict)

        # ===== Prepare traffic light tokens =====
        input_dict = self.prepare_traffic_light_tokens(input_dict)


        if self.no_tg:
            # ===== Prepare trafficgen tokens =====
            # input_dict = self.prepare_trafficgen_tokens(input_dict)

            # ===== Prepare motion tokens =====
            input_dict = self.prepare_motion_tokens(input_dict)

            # ===== Prepare traffic light relation =====
            input_dict = self.prepare_dynamic_relation_notg(input_dict)

        else:

            # ===== Prepare trafficgen tokens =====
            input_dict = self.prepare_trafficgen_tokens(input_dict)

            # ===== Prepare motion tokens =====
            input_dict = self.prepare_motion_tokens(input_dict)

            # ===== Prepare traffic light relation =====
            input_dict = self.prepare_dynamic_relation(input_dict)

        # ===== Call the decoder =====
        # TODO: dest is not conditioning on anyone.
        output_dict = self.decoder(input_dict=input_dict)

        # ===== Deal with the output =====
        all_token = output_dict["model/all_token"]

        # ===== Traffic light head =====
        traffic_light_token = []

        if self.no_tg:
            total = N + L
            pointer = 0
            for t in range(T):
                traffic_light_token.append(all_token[:, pointer: pointer+L])
                pointer += total
            traffic_light_token = torch.stack(traffic_light_token, dim=1)
            debug_traffic_light_token = traffic_light_token

        else:
            total = N + G + L
            pointer = 0
            for t in range(T):
                traffic_light_token.append(all_token[:, pointer: pointer+L])
                if t % TG_SKIP_STEP == 0:
                    pointer += total
                else:
                    pointer += N + L
            assert pointer == all_token.shape[1]
            traffic_light_token = torch.stack(traffic_light_token, dim=1)
        traffic_light_token = self.traffic_light_prenorm(traffic_light_token)
        traffic_light_token = self.traffic_light_head(traffic_light_token)
        output_dict["model/traffic_light_logit"] = traffic_light_token

        # ===== Trafficgen head =====
        if self.no_tg:
            pass

        else:
            trafficgen_output_token = []
            pointer = 0
            for t in range(T):
                if t % TG_SKIP_STEP == 0:
                    trafficgen_output_token.append(all_token[:, pointer + L: pointer + L + G])
                    pointer += total
                else:
                    pointer += N + L
            assert pointer == all_token.shape[1]
            trafficgen_output_token = torch.stack(trafficgen_output_token, dim=1)

            trafficgen_output_token = self.trafficgen_prenorm(trafficgen_output_token)

            from scenestreamer.dataset.preprocessor import slice_trafficgen_data
            map_id_logit, agent_type_logits, agent_state_logits, dest_id_logit = self.trafficgen_head(
                offset=slice_trafficgen_data(input_dict["decoder/input_offset_for_trafficgen"], dim=1),
                trafficgen_token=trafficgen_output_token,
            )
            output_dict["model/trafficgen_map_id_logit"] = map_id_logit
            output_dict["model/trafficgen_agent_type_logit"] = agent_type_logits
            output_dict["model/trafficgen_agent_state_logit"] = agent_state_logits
            output_dict["model/trafficgen_dest_id_logit"] = dest_id_logit

        # output_dict["model/trafficgen_output_token"] = trafficgen_output_token

        # ===== Motion head =====
        if self.no_tg:
            motion_token = []
            pointer = 0
            for t in range(T):
                motion_token.append(all_token[:, pointer+L: pointer+L+N])
                pointer += N + L
            assert pointer == all_token.shape[1]
            motion_token = torch.stack(motion_token, dim=1)
        else:
            motion_token = []
            pointer = 0
            for t in range(T):
                motion_token.append(all_token[:, pointer+L+G: pointer+L+G+N])
                if t % TG_SKIP_STEP == 0:
                    pointer += total
                else:
                    pointer += N + L
            assert pointer == all_token.shape[1]
            motion_token = torch.stack(motion_token, dim=1)

        # assert (debug_traffic_light_token == all_token.reshape(B, T, N + L, -1)[:, :, :L]).all()
        # assert (motion_token == all_token.reshape(B, T, N + L, -1)[:, :, L:]).all()

        # TODO: dest is not conditioning on anyone.
        if self.motion_prenorm is not None:
            motion_token = self.motion_prenorm(motion_token)
        motion_token = self.motion_head(motion_token)
        output_dict["model/motion_logit"] = motion_token

        return input_dict

    def prepare_map_tokens(self, input_dict):

        # ===== Get shape =====
        B, M, num_vector, D_vector = input_dict["encoder/map_feature"].shape

        # ===== Embed map feature =====
        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_position = input_dict["encoder/map_position"]
        map_heading = input_dict["encoder/map_heading"]
        map_token_valid_mask = input_dict["encoder/map_valid_mask"]
        # map_token = self.map_polyline_encoder(map_feature, map_valid_mask)
        input_dict = self.map_encoder(input_dict=input_dict)
        map_token = input_dict["model/map_token"]
        assert map_token.shape == (B, M, self.d_model)
        map_id = torch.arange(M, device=map_feature.device).unsqueeze(0).expand(B, M).clone()
        map_id[~map_token_valid_mask] = -1
        map_id_pe = self.map_id_embed(map_id)

        # egocentric discrete embedding
        assert map_token.shape == (B, M, self.d_model), (map_token.shape, B, M, self.d_model)
        assert map_id_pe.shape == (B, M, self.d_model), (map_id_pe.shape, B, M, self.d_model)
        map_token = map_token + map_id_pe

        input_dict["model/map_token"] = map_token
        input_dict["model/map_token_position"] = map_position
        input_dict["model/map_token_heading"] = map_heading
        input_dict["model/map_token_valid_mask"] = map_token_valid_mask

        return input_dict

    def prepare_traffic_light_tokens(self, input_dict):

        traffic_light_state = input_dict["encoder/traffic_light_state"]
        traffic_light_map_id = input_dict["encoder/traffic_light_map_id"]
        traffic_light_position = input_dict["encoder/traffic_light_position"]
        traffic_light_heading = input_dict["encoder/traffic_light_heading"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]

        B, T, L = traffic_light_state.shape

        tl_id = torch.arange(L, device=traffic_light_state.device).reshape(1, 1, L).expand(B, T, L).clone()
        tl_id[~traffic_light_valid_mask] = -1
        tl_id_pe = self.traffic_light_id_embed(tl_id)

        tl_map_id_pe = self.map_id_embed(traffic_light_map_id)
        tl_map_id_pe = tl_map_id_pe.unsqueeze(1).expand(B, T, L, self.d_model)

        light_tokens = self.traffic_light_state_embed(traffic_light_state)
        light_tokens = light_tokens + tl_map_id_pe
        light_tokens = light_tokens + tl_id_pe

        input_dict["model/traffic_light_token"] = light_tokens
        input_dict["model/traffic_light_token_position"] = traffic_light_position.unsqueeze(1).expand(B, T, L, traffic_light_position.shape[-1])
        input_dict["model/traffic_light_token_heading"] = traffic_light_heading.unsqueeze(1).expand(B, T, L)
        input_dict["model/traffic_light_token_valid_mask"] = traffic_light_valid_mask

        require_relation = traffic_light_valid_mask.clone()
        input_dict["model/traffic_light_require_relation"] = require_relation

        return input_dict

    def _pad_sos_eos(self, tensor, dim, value):
        assert dim == 2
        return torch.cat([torch.full_like(tensor[:, :, :1], value), tensor, torch.full_like(tensor[:, :, :1], value)],
                         dim=dim)

    def prepare_trafficgen_tokens(self, input_dict):

        # ===== Agent Tokens =====
        B, T, N, _ = input_dict["decoder/modeled_agent_position"].shape

        # tg_start, agent_start, agent_map_id, agent_state, agent_dest, agent_end, ..., tg_end
        G = get_num_tg(N)

        tg_input_action = input_dict["decoder/input_action_for_trafficgen"]
        assert tg_input_action.shape == (B, T, G), (B, T, G, tg_input_action.shape)

        assert tg_input_action.max() <= self.trafficgen_agent_sos_id, (tg_input_action.max(), self.cyc_id)

        tg_intra_step = torch.arange(G, device=tg_input_action.device).reshape(1, 1, G).expand(B, T, G).clone()
        tg_intra_step[tg_intra_step > self.trafficgen_intra_step.num_actions] = -1
        tg_intra_step_emb = self.trafficgen_intra_step(tg_intra_step)

        if input_dict["decoder/agent_type_for_trafficgen"].max().item() not in [self.veh_id, self.cyc_id, self.ped_id]:
            print("WARNING: agent type is not veh, cyc, ped, it is: ", input_dict["decoder/agent_type_for_trafficgen"].max().item(), input_dict["scenario_id"], self.veh_id, self.cyc_id, self.ped_id)
        type_emb = self.map_id_embed(input_dict["decoder/agent_type_for_trafficgen"])
        assert type_emb.shape == (B, T, G, self.d_model)

        # shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape_for_trafficgen"])
        # assert shape_emb.shape == (B, G, self.d_model), (B, G, self.d_model, shape_emb.shape)
        # shape_emb = shape_emb.unsqueeze(1).expand(B, T, G, self.d_model)

        modeled_agent_id = mode_agent_id(input_dict["decoder/agent_id_for_trafficgen"], 128, fill_negative_1=True)
        agent_id_emb = self.agent_id_embed(modeled_agent_id)
        assert agent_id_emb.shape == (B, T, G, self.d_model), (B, G, self.d_model, agent_id_emb.shape)

        tg_action_emb = self.map_id_embed(tg_input_action)
        assert tg_action_emb.shape == (B, T, G, self.d_model), (B, T, G, self.d_model, tg_action_emb.shape)

        tg_feat_emb = self.trafficgen_feat_embed(input_dict["decoder/input_action_feature_for_trafficgen"])
        assert tg_action_emb.shape == (B, T, G, self.d_model), (B, T, G, self.d_model, tg_action_emb.shape)

        tg_tokens = tg_intra_step_emb + type_emb + agent_id_emb + tg_action_emb + tg_feat_emb

        input_dict["model/trafficgen_token"] = tg_tokens
        # pad 0 before and after the sequence

        # TODO: hardcoded 5, 6
        tg_length = input_dict["decoder/input_action_feature_for_trafficgen"][..., 5]
        tg_width = input_dict["decoder/input_action_feature_for_trafficgen"][..., 6]

        input_dict["model/trafficgen_position"] = input_dict["decoder/trafficgen_position"]
        input_dict["model/trafficgen_heading"] = input_dict["decoder/trafficgen_heading"]
        input_dict["model/trafficgen_valid_mask"] = input_dict["decoder/input_action_valid_mask_for_trafficgen"]
        input_dict["model/trafficgen_width"] = tg_width
        input_dict["model/trafficgen_length"] = tg_length

        require_relation = torch.ones(B, T, N, NUM_TG_MULTI, device=tg_input_action.device, dtype=torch.bool)
        require_relation[:, :, :, 0] = 0  # sos
        require_relation[:, :, :, 1] = 0  # agent type
        require_relation[:, :, :, 2] = 1  # map_id (map pos)
        require_relation[:, :, :, 3] = 1  # agent_state (agent pos)
        # require_relation[:, :, :, 4] = 1  # dest_id (dest pos)
        require_relation = torch.cat([
            torch.zeros(B, T, 1, device=tg_input_action.device, dtype=torch.bool),
            require_relation.flatten(2, 3),
            torch.zeros(B, T, 1, device=tg_input_action.device, dtype=torch.bool)
        ], dim=2)

        input_dict["model/trafficgen_require_relation"] = require_relation & input_dict["model/trafficgen_valid_mask"]
        return input_dict

    def prepare_trafficgen_single_token(
            self, *, tg_intra_step, tg_type, tg_agent_id, tg_action, tg_feat
    ):
        assert tg_intra_step.ndim == 2
        assert tg_type.ndim == 2
        assert tg_agent_id.ndim == 2
        assert tg_action.ndim == 2
        assert tg_feat.ndim == 3
        tg_intra_step = mode_agent_id(tg_intra_step, max_agents=self.trafficgen_intra_step.num_actions, fill_negative_1=True)
        tg_intra_step_emb = self.trafficgen_intra_step(tg_intra_step)
        if (tg_type!=-1).any():
            assert tg_type[tg_type!=-1].min() >= self.veh_id
        type_emb = self.map_id_embed(tg_type)
        tg_agent_id = mode_agent_id(tg_agent_id, max_agents=128, fill_negative_1=True)
        agent_id_emb = self.agent_id_embed(tg_agent_id)
        tg_action_emb = self.map_id_embed(tg_action)
        tg_feat_emb = self.trafficgen_feat_embed(tg_feat)
        tg_tokens = tg_intra_step_emb + type_emb + agent_id_emb + tg_action_emb + tg_feat_emb
        return tg_tokens

    def prepare_motion_tokens(self, input_dict):

        # === Process action embedding ===
        input_action = input_dict["decoder/input_action"]
        modeled_agent_delta = input_dict["decoder/modeled_agent_delta"]
        B, T_skipped, N = input_action.shape[:3]

        agent_id = input_dict["encoder/modeled_agent_id"].reshape(B, 1, N).expand(B, T_skipped, N)
        agent_id = mode_agent_id(agent_id, 128, fill_negative_1=True)
        agent_id_emb = self.agent_id_embed(agent_id)

        assert agent_id_emb.shape == (B, T_skipped, N, self.d_model), (
            B, T_skipped, N, self.d_model, agent_id_emb.shape)

        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        assert action_valid_mask.shape == (B, T_skipped, N), (action_valid_mask.shape, (B, T_skipped, N))
        agent_pos = input_dict["decoder/modeled_agent_position"]
        agent_heading = input_dict["decoder/modeled_agent_heading"]
        # agent_vel = input_dict["decoder/modeled_agent_velocity"]

        # ===== Prepare input tokens =====
        # assert input_dict["decoder/agent_type"].min() == self.veh_id, (input_dict["decoder/agent_type"].min(), self.veh_id)
        type_emb = self.map_id_embed(input_dict["decoder/agent_type"])[:, None].expand(B, T_skipped, N, self.d_model)
        shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape"])[:, None].expand(B, T_skipped, N,
                                                                                                self.d_model)

        valid_action = input_action[action_valid_mask]
        valid_action[valid_action == START_ACTION] = -1
        valid_action_emb = self.action_embed(valid_action)

        motion_feat = self.motion_features.reshape(1, -1, 4).expand(valid_action_emb.shape[0], -1, 4)

        valid_action[valid_action < 0] = self.num_actions
        valid_action = valid_action.reshape(-1, 1, 1).expand(-1, 1, 4)
        assert motion_feat.shape[-2] > valid_action.max()
        assert valid_action.min() >= 0
        motion_feat = torch.gather(motion_feat, dim=-2, index=valid_action).squeeze(-2)

        motion_feat = torch.cat([motion_feat, modeled_agent_delta[action_valid_mask]], dim=-1)

        action_token = self.motion_embed(
            continuous_inputs=motion_feat,
            categorical_embs=[
                agent_id_emb[action_valid_mask], type_emb[action_valid_mask],
                shape_emb[action_valid_mask], valid_action_emb
            ]
        )
        action_token = utils.unwrap(action_token, action_valid_mask)
        assert action_token.shape == (B, T_skipped, N, self.d_model)
        assert action_valid_mask.shape == (B, T_skipped, N)

        input_dict["model/motion_token"] = action_token
        input_dict["model/motion_token_valid_mask"] = action_valid_mask
        input_dict["model/motion_token_position"] = agent_pos
        input_dict["model/motion_token_heading"] = agent_heading

        T = input_dict["decoder/input_action"].shape[1]
        shape = input_dict["decoder/current_agent_shape"].unsqueeze(1).expand(B, T, N, 3)
        length = shape[:, :, :, 0]
        width = shape[:, :, :, 1]
        input_dict["model/motion_token_width"] = width
        input_dict["model/motion_token_length"] = length

        require_relation = action_valid_mask.clone()
        input_dict["model/motion_require_relation"] = require_relation
        return input_dict

    def _build_all_tokens_mask_for_tl(self, B, T, num_tl, num_tg, num_motion):
        """
        Recall that in our design, traffic light tokens attend to:
        1) map (not considered here),
        2) itself (self-attention),
        3) all previous traffic light tokens,
        4) last step motion tokens.

        The ultimate output should be a mask with shape: B, Q, K, where
        Q = T*num_tl, K = T*(num_tl + num_tg + num_motion).
        """
        total_tokens_per_step = num_tl + num_tg + num_motion
        tl_mask = []
        for t in range(T):
            mask = torch.zeros(B, num_tl, T, total_tokens_per_step, dtype=torch.bool)
            mask[:, :, t, :num_tl] = True  # self-attention

            # previous traffic light tokens
            mask[:, :, :t, :num_tl] = torch.diag(torch.ones(num_tl)).bool().unsqueeze(1)

            if t > 0:
                mask[:, :, t - 1, num_tl + num_tg:] = True  # last step motion tokens
            tl_mask.append(mask)
        return tl_mask  # .flatten(1, 2)

    def _build_all_tokens_mask_for_tg(self, B, T, num_tl, num_tg, num_motion):
        """
        Recall that in our design, traffic light tokens attend to:
        1) map (not considered here),
        2) itself (self-attention and WITH CASUAL MASK!!),
        3) current step traffic light tokens,
        4) last step motion tokens.
        5) all previous step trafficgen tokens.

        The ultimate output should be a mask with shape: B, Q, K, where
        Q = T*num_tg, K = T*(num_tl + num_tg + num_motion).
        """
        total_tokens_per_step = num_tl + num_tg + num_motion
        tg_mask = []
        intra_step_causal_mask = create_causal_mask(T=num_tg, N=1, is_valid_mask=True)
        diag = torch.diag(torch.ones(num_motion)).bool()
        diag_tg = torch.diag(torch.ones(num_tg)).bool()
        diag_rep = diag[:, None, :, None].repeat(1, NUM_TG_MULTI, 1, NUM_TG_MULTI).flatten(-2, -1).flatten(0, 1)
        for t in range(T):
            mask = torch.zeros(B, num_tg, T, total_tokens_per_step, dtype=torch.bool)
            mask[:, :, t, num_tl:num_tl + num_tg] = intra_step_causal_mask  # self-attention
            mask[:, :, t, :num_tl] = True  # current step traffic light tokens
            if t > 0:
                mask[:, :, t - 1, num_tl + num_tg:] = True  # last step motion tokens
                mask[:, 1:-1, :t, num_tl+1:num_tl + num_tg-1] = diag_rep[:, None]  # all previous step trafficgen tokens
                mask[:, :, :t, num_tl:num_tl + num_tg] = diag_tg.unsqueeze(1)
                mask[:, :, :t, num_tl + num_tg-1:num_tl + num_tg] = True  # all token attend to previous eos token
                mask[:, :, :t, num_tl:num_tl + 1] = True  # all token attend to previous sos token
            tg_mask.append(mask)
        return tg_mask  # .flatten(1, 2)

    def _build_all_tokens_mask_for_motion(self, B, T, num_tl, num_tg, num_motion):
        """
        Recall that in our design, traffic light tokens attend to:
        1) map (not considered here),
        2) itself (self-attention),
        3) current step traffic light tokens,
        4) current step trafficgen tokens,
        5) all previous step motion tokens.

        The ultimate output should be a mask with shape: B, Q, K, where
        Q = T*num_motion, K = T*(num_tl + num_tg + num_motion).
        """
        total_tokens_per_step = num_tl + num_tg + num_motion
        motion_mask = []
        diag = torch.diag(torch.ones(num_motion)).bool()
        diag_rep = diag[..., None].repeat(1, 1, NUM_TG_MULTI).flatten(1, 2)
        for t in range(T):
            mask = torch.zeros(B, num_motion, T, total_tokens_per_step, dtype=torch.bool)
            mask[:, :, t, num_tl + num_tg:] = True  # self-attention
            mask[:, :, t, :num_tl] = True  # current step traffic light tokens
            if num_tg > 0:
                mask[:, :, :t+1, num_tl + 1:num_tl + num_tg - 1] = diag_rep.unsqueeze(1)  # current step trafficgen tokens
                mask[:, :, :t+1, num_tl:num_tl + 1] = True  # current step trafficgen tokens
                mask[:, :, :t+1, num_tl+num_tg-1:num_tl + num_tg] = True  # current step trafficgen tokens
            # all previous step motion tokens FOR EACH AGENT
            mask[:, :, :t, num_tl + num_tg:] = diag.unsqueeze(1)
            motion_mask.append(mask)
        return motion_mask  # .flatten(1, 2)

    def _build_force_mask_for_tl(self, B, T, num_tl, num_tg, num_motion):
        """
        You must attend to your own history
        """
        total_tokens_per_step = num_tl + num_tg + num_motion
        tl_mask = []
        for t in range(T):
            mask = torch.zeros(B, num_tl, T, total_tokens_per_step, dtype=torch.bool)
            mask[:, :, :t, :num_tl] = torch.diag(torch.ones(num_tl)).bool().unsqueeze(1)
            tl_mask.append(mask)
        return tl_mask

    def _build_force_mask_for_tg(self, B, T, num_tl, num_tg, num_motion):
        assert num_tg > 0
        total_tokens_per_step = num_tl + num_tg + num_motion
        tg_mask = []
        diag = torch.diag(torch.ones(num_motion)).bool()
        diag_rep = diag[:, None, :, None].repeat(1, NUM_TG_MULTI, 1, NUM_TG_MULTI).flatten(-2, -1).flatten(0, 1)
        diag = diag[:, None, ].repeat(1, NUM_TG_MULTI, 1).flatten(0, 1)
        for t in range(T):
            mask = torch.zeros(B, num_tg, T, total_tokens_per_step, dtype=torch.bool)
            if t > 0:
                mask[:, 1:-1, :t, num_tl + num_tg:] = diag[:, None]  # history motion tokens
                mask[:, 1:-1, :t, num_tl + 1:num_tl + num_tg - 1] = diag_rep[:, None]  # history tg tokens (only the same agent)
                mask[:, :1, t - 1, num_tl + num_tg:] = True  # sos attends history motion token
            tg_mask.append(mask)
        return tg_mask

    def _build_force_mask_for_motion(self, B, T, num_tl, num_tg, num_motion):
        """
        You must attend to your own history.
        """
        total_tokens_per_step = num_tl + num_tg + num_motion
        motion_mask = []
        diag = torch.diag(torch.ones(num_motion)).bool()
        diag_rep = diag[..., None].repeat(1, 1, NUM_TG_MULTI).flatten(1, 2)
        for t in range(T):
            mask = torch.zeros(B, num_motion, T, total_tokens_per_step, dtype=torch.bool)
            if num_tg > 0:
                mask[:, :, :t+1, num_tl + 1:num_tl + num_tg - 1] = diag_rep[:, None]  # attend to all prev ego TG tokens.
                mask[:, :, :t+1, num_tl:num_tl + 1] = True  # current step trafficgen tokens sos
                mask[:, :, :t+1, num_tl+num_tg-1:num_tl + num_tg] = True  # current step trafficgen tokens eos
            mask[:, :, :t, num_tl + num_tg:] = diag.unsqueeze(1)
            motion_mask.append(mask)
        return motion_mask  # .flatten(1, 2)

    def _build_all_tokens_mask(self, B, T, num_tl, num_tg, num_motion):

        if self.no_tg:
            tl_mask = self._build_all_tokens_mask_for_tl(B, T, num_tl, 0, num_motion)
            tl_mask = torch.stack(tl_mask, dim=1).flatten(3, 4)
            motion_mask = self._build_all_tokens_mask_for_motion(B, T, num_tl, 0, num_motion)
            motion_mask = torch.stack(motion_mask, dim=1).flatten(3, 4)
            all_mask = torch.cat([tl_mask, motion_mask], dim=2)
            notg_all_mask = all_mask.flatten(1, 2)
            assert notg_all_mask.shape[1] == (num_tl + num_motion) * T, (notg_all_mask.shape[1], num_tl, num_motion, T)
            return notg_all_mask

        assert self.no_tg is False, "This function is only for no_tg = False"
        with_tg = num_tl + num_tg + num_motion
        without_tg = num_tl + num_motion

        total_tokens_so_far = with_tg

        tl_mask = self._build_all_tokens_mask_for_tl(B, T, num_tl, num_tg, num_motion)
        tg_mask = self._build_all_tokens_mask_for_tg(B, T, num_tl, num_tg, num_motion)
        motion_mask = self._build_all_tokens_mask_for_motion(B, T, num_tl, num_tg, num_motion)

        tl_mask = torch.stack(tl_mask, dim=1).flatten(3, 4)
        tg_mask = torch.stack(tg_mask, dim=1).flatten(3, 4)
        motion_mask = torch.stack(motion_mask, dim=1).flatten(3, 4)

        all_mask = torch.cat([tl_mask, tg_mask, motion_mask], dim=2)

        # all_mask shape is [B, T, L+G+N, T*(L+G+N)]

        new_mask = []

        # [B, T, L+G+N, T*(L+G+N)] -> [B, correct_num_of_tokens, T*(L+G+N)]
        for t in range(T):
            if t % TG_SKIP_STEP == 0:
                new_mask.append(all_mask[:, t, :, :])
            else:
                cat = torch.cat([all_mask[:, t, :num_tl, :], all_mask[:, t, num_tl + num_tg:, :]], dim=1)
                new_mask.append(cat)
        new_mask = torch.cat(new_mask, dim=1)


        full_mask = []
        for t in range(T):
            if t % TG_SKIP_STEP == 0:
                full_mask.append(new_mask[:, :, t*with_tg: (t+1)*with_tg])
            else:
                full_mask.append(new_mask[:, :, t*with_tg:t*with_tg+num_tl])
                full_mask.append(new_mask[:, :, t*with_tg+num_tl+num_tg:(t+1)*with_tg])
        full_mask = torch.cat(full_mask, dim=2)
        assert full_mask.shape[1] == full_mask.shape[2], (full_mask.shape[1], full_mask.shape[2])

        # import matplotlib.pyplot as plt
        # vis = full_mask[0].cpu().numpy()
        # plt.imshow(vis)

        return full_mask

    def _build_all_force_mask(self, B, T, num_tl, num_tg, num_motion):
        if self.no_tg:
            tl_mask = self._build_force_mask_for_tl(B, T, num_tl, 0, num_motion)
            tl_mask = torch.stack(tl_mask, dim=1).flatten(3, 4)
            motion_mask = self._build_force_mask_for_motion(B, T, num_tl, 0, num_motion)
            motion_mask = torch.stack(motion_mask, dim=1).flatten(3, 4)
            all_mask = torch.cat([tl_mask, motion_mask], dim=2)
            notg_all_mask = all_mask.flatten(1, 2)
            assert notg_all_mask.shape[1] == (num_tl + num_motion) * T, (notg_all_mask.shape[1], num_tl, num_motion, T)
            return notg_all_mask

        assert self.no_tg is False, "This function is only for no_tg = False"
        with_tg = num_tl + num_tg + num_motion
        tl_mask = self._build_force_mask_for_tl(B, T, num_tl, num_tg, num_motion)
        tg_mask = self._build_force_mask_for_tg(B, T, num_tl, num_tg, num_motion)
        motion_mask = self._build_force_mask_for_motion(B, T, num_tl, num_tg, num_motion)
        tl_mask = torch.stack(tl_mask, dim=1).flatten(3, 4)
        tg_mask = torch.stack(tg_mask, dim=1).flatten(3, 4)
        motion_mask = torch.stack(motion_mask, dim=1).flatten(3, 4)
        all_mask = torch.cat([tl_mask, tg_mask, motion_mask], dim=2)
        new_mask = []
        for t in range(T):
            if t % TG_SKIP_STEP == 0:
                new_mask.append(all_mask[:, t, :, :])
            else:
                cat = torch.cat([all_mask[:, t, :num_tl, :], all_mask[:, t, num_tl + num_tg:, :]], dim=1)
                new_mask.append(cat)
        new_mask = torch.cat(new_mask, dim=1)
        full_mask = []
        for t in range(T):
            if t % TG_SKIP_STEP == 0:
                full_mask.append(new_mask[:, :, t*with_tg: (t+1)*with_tg])
            else:
                full_mask.append(new_mask[:, :, t*with_tg:t*with_tg+num_tl])
                full_mask.append(new_mask[:, :, t*with_tg+num_tl+num_tg:(t+1)*with_tg])
        full_mask = torch.cat(full_mask, dim=2)
        assert full_mask.shape[1] == full_mask.shape[2], (full_mask.shape[1], full_mask.shape[2])
        return full_mask

    def prepare_dynamic_relation_notg(self, input_dict):

        map_position = input_dict["model/map_token_position"]
        map_heading = input_dict["model/map_token_heading"]
        map_token_valid_mask = input_dict["model/map_token_valid_mask"]

        # traffic light tokens
        traffic_light_position = input_dict["model/traffic_light_token_position"]
        traffic_light_heading = input_dict["model/traffic_light_token_heading"]
        traffic_light_token = input_dict["model/traffic_light_token"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]
        traffic_light_width = torch.zeros_like(traffic_light_position[..., 0])
        traffic_light_length = torch.zeros_like(traffic_light_position[..., 0])
        traffic_light_require_relation = input_dict["model/traffic_light_require_relation"]
        B, T, L, _ = traffic_light_token.shape

        # motion tokens
        motion_token = input_dict["model/motion_token"]
        motion_token_valid_mask = input_dict["model/motion_token_valid_mask"]
        motion_token_position = input_dict["model/motion_token_position"]
        motion_token_heading = input_dict["model/motion_token_heading"]
        motion_token_width = input_dict["model/motion_token_width"]
        motion_token_length = input_dict["model/motion_token_length"]
        motion_token_require_relation = input_dict["model/motion_require_relation"]
        N = motion_token.shape[2]

        # build giant tensors for all "dynamic" tokens to serve as the key/value
        all_tokens = torch.cat([
            traffic_light_token,
            motion_token,
        ], dim=2)
        all_positions = torch.cat([
            traffic_light_position[..., :2],
            motion_token_position[..., :2],
        ], dim=2)
        all_headings = torch.cat([
            traffic_light_heading,
            motion_token_heading,
        ], dim=2)
        all_valid_masks = torch.cat([
            traffic_light_valid_mask,
            motion_token_valid_mask,
        ], dim=2)
        all_widths = torch.cat([
            traffic_light_width,
            motion_token_width,
        ], dim=2)
        all_lengths = torch.cat([
            traffic_light_length,
            motion_token_length,
        ], dim=2)
        all_require_relation = torch.cat([
            traffic_light_require_relation,
            motion_token_require_relation,
        ], dim=2)
        # all_require_relation = all_require_relation & all_valid_masks

        giant_N = all_tokens.shape[2]
        all_steps = torch.arange(T).to(traffic_light_position.device).reshape(1, T, 1)  # .expand(B, T, giant_N)

        # ===== Build causal mask for traffic light tokens =====
        tl_causal_mask = torch.stack(
            self._build_all_tokens_mask_for_tl(B=B, T=T, num_tl=L, num_tg=0, num_motion=N), dim=1
        ).flatten(-2, -1).to(traffic_light_position.device)

        # ===== Build causal mask for motion tokens =====
        motion_causal_mask = torch.stack(
            self._build_all_tokens_mask_for_motion(B=B, T=T, num_tl=L, num_tg=0, num_motion=N), dim=1
        ).flatten(-2, -1).to(traffic_light_position.device)

        all_causal_mask = torch.cat([tl_causal_mask, motion_causal_mask], dim=2)

        # ===== Build causal mask for traffic light tokens =====
        tl_force_mask = torch.stack(
            self._build_force_mask_for_tl(B=B, T=T, num_tl=L, num_tg=0, num_motion=N), dim=1
        ).flatten(-2, -1).to(traffic_light_position.device)
        motion_force_mask = torch.stack(
            self._build_force_mask_for_motion(B=B, T=T, num_tl=L, num_tg=0, num_motion=N), dim=1
        ).flatten(-2, -1).to(traffic_light_position.device)
        all_force_mask = torch.cat([tl_force_mask, motion_force_mask], dim=2)

        # import matplotlib.pyplot as plt
        # vis = all_causal_mask[0].flatten(0, 1).cpu().numpy()
        # plt.imshow(vis)

        relation_all_to_all, relation_valid_mask, require_relation_pairwise = relation.compute_relation_for_scenestreamer(
            query_pos=all_positions.flatten(1, 2),
            query_heading=all_headings.flatten(1, 2),
            query_valid_mask=all_valid_masks.flatten(1, 2),
            query_step=all_steps.expand(B, T, L  + N).flatten(1, 2),
            key_pos=all_positions.flatten(1, 2),
            key_heading=all_headings.flatten(1, 2),
            key_valid_mask=all_valid_masks.flatten(1, 2),
            key_step=all_steps.expand(B, T, L  + N).flatten(1, 2),
            causal_valid_mask=all_causal_mask.flatten(1, 2),

            force_attention_mask=all_force_mask.flatten(1, 2),

            require_relation=all_require_relation.flatten(1, 2),

            knn=self.config.SCENESTREAMER_ATTENTION_KNN,
            max_distance=self.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE,

            gather=False,
            query_width=None,  # set query's w/l to 0 so that we get the rel of contour of key w.r.t. center of query
            query_length=None,
            key_width=all_widths.flatten(1, 2),
            key_length=all_lengths.flatten(1, 2),
            non_agent_relation=True,
        )
        relation_all_to_all = get_edge_info_for_scenestreamer(
            q_k_relation=relation_all_to_all,
            q_k_valid_mask=relation_valid_mask,
            relation_model=self.relation_embed_4d,
            relation_model_1d=self.relation_embed_1d,
            require_relation_pairwise=require_relation_pairwise,
        )
        relation_all_to_map = self._get_relation_for_4d_token_vs_map_token(
            token_4d_pos=all_positions,
            token_4d_heading=all_headings,
            token_4d_valid_mask=all_valid_masks,
            token_4d_step=all_steps.expand(B, T, L + N),
            # token4d_length=all_lengths,
            # token4d_width=all_widths,
            map_pos=map_position,
            map_heading=map_heading,
            map_valid_mask=map_token_valid_mask,
            knn=self.config.SCENESTREAMER_ATTENTION_KNN,
            max_distance=self.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE,
            require_relation=all_require_relation,
        )

        # import matplotlib.pyplot as plt
        # vis = all_causal_mask.flatten(1, 2)[0].cpu().numpy()
        # plt.imshow(vis)

        input_dict["model/all_token"] = all_tokens.flatten(1, 2)
        input_dict["model/all_to_all_info"] = relation_all_to_all
        input_dict["model/all_to_map_info"] = relation_all_to_map
        return input_dict



    def prepare_dynamic_relation(self, input_dict):

        map_position = input_dict["model/map_token_position"]
        map_heading = input_dict["model/map_token_heading"]
        map_token_valid_mask = input_dict["model/map_token_valid_mask"]

        # traffic light tokens
        traffic_light_position = input_dict["model/traffic_light_token_position"]
        traffic_light_heading = input_dict["model/traffic_light_token_heading"]
        traffic_light_token = input_dict["model/traffic_light_token"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]
        traffic_light_width = torch.zeros_like(traffic_light_position[..., 0])
        traffic_light_length = torch.zeros_like(traffic_light_position[..., 0])
        traffic_light_require_relation = input_dict["model/traffic_light_require_relation"]
        B, T, L, _ = traffic_light_token.shape

        # trafficgen tokens
        tg_tokens = input_dict["model/trafficgen_token"]
        tg_position = input_dict["model/trafficgen_position"]
        tg_heading = input_dict["model/trafficgen_heading"]
        tg_valid_mask = input_dict["model/trafficgen_valid_mask"]
        tg_width = input_dict["model/trafficgen_width"]
        tg_length = input_dict["model/trafficgen_length"]
        tg_require_relation = input_dict["model/trafficgen_require_relation"]
        _, _, G, _ = tg_tokens.shape

        # motion tokens
        motion_token = input_dict["model/motion_token"]
        motion_token_valid_mask = input_dict["model/motion_token_valid_mask"]
        motion_token_position = input_dict["model/motion_token_position"]
        motion_token_heading = input_dict["model/motion_token_heading"]
        motion_token_width = input_dict["model/motion_token_width"]
        motion_token_length = input_dict["model/motion_token_length"]
        motion_token_require_relation = input_dict["model/motion_require_relation"]
        N = motion_token.shape[2]

        # build giant tensors for all "dynamic" tokens to serve as the key/value
        def _concat(tl_tokens, tg_tokens, motion_tokens):
            assert tl_tokens.shape[:3] == (B, T, L), (tl_tokens.shape, (B, T, L))
            assert tg_tokens.shape[:3] == (B, T, G), (tg_tokens.shape, (B, T, G))
            assert motion_tokens.shape[:3] == (B, T, N), (motion_tokens.shape, (B, T, N))
            ret = []
            for t in range(T):
                ret.append(tl_tokens[:, t])
                if t % TG_SKIP_STEP == 0:
                    ret.append(tg_tokens[:, t])
                ret.append(motion_tokens[:, t])
            ret = torch.cat(ret, dim=1)
            return ret

        all_knn = self.config.SCENESTREAMER_ATTENTION_KNN
        # tl_knn = torch.full((B, T, L), knn // 2, device=traffic_light_position.device)
        # tg_knn = torch.full((B, T, G), knn // 2, device=traffic_light_position.device)
        # motion_knn = torch.full((B, T, N), knn, device=traffic_light_position.device)
        # all_knn = _concat(tl_knn, tg_knn, motion_knn)

        all_tokens = _concat(traffic_light_token, tg_tokens, motion_token)
        all_positions = _concat(traffic_light_position[..., :2], tg_position[..., :2], motion_token_position[..., :2])
        all_headings = _concat(traffic_light_heading, tg_heading, motion_token_heading)
        all_valid_masks = _concat(traffic_light_valid_mask, tg_valid_mask, motion_token_valid_mask)
        all_widths = _concat(traffic_light_width, tg_width, motion_token_width)
        all_lengths = _concat(traffic_light_length, tg_length, motion_token_length)
        all_require_relation = _concat(traffic_light_require_relation, tg_require_relation, motion_token_require_relation)
        # all_require_relation = all_require_relation & all_valid_masks

        all_steps = torch.arange(T).to(traffic_light_position.device).reshape(1, T, 1)  # .expand(B, T, giant_N)
        all_steps = _concat(all_steps.expand(B, T, L), all_steps.expand(B, T, G), all_steps.expand(B, T, N))

        all_causal_mask = self._build_all_tokens_mask(B=B, T=T, num_tl=L, num_tg=G, num_motion=N).to(
            all_require_relation.device)

        all_force_mask = self._build_all_force_mask(B=B, T=T, num_tl=L, num_tg=G, num_motion=N).to(
            all_require_relation.device)

        # import matplotlib.pyplot as plt
        # vis = all_causal_mask[0].cpu().numpy()
        # plt.imshow(vis)
        relation_all_to_all, relation_valid_mask, require_relation_pairwise = relation.compute_relation_for_scenestreamer(
            query_pos=all_positions,
            query_heading=all_headings,
            query_valid_mask=all_valid_masks,
            query_step=all_steps,
            key_pos=all_positions,
            key_heading=all_headings,
            key_valid_mask=all_valid_masks,
            key_step=all_steps,
            causal_valid_mask=all_causal_mask,
            force_attention_mask=all_force_mask,
            require_relation=all_require_relation,

            knn=all_knn,
            max_distance=self.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE,

            gather=False,
            query_width=None,  # set query's w/l to 0 so that we get the rel of contour of key w.r.t. center of query
            query_length=None,
            key_width=all_widths,
            key_length=all_lengths,
            non_agent_relation=True,
        )
        relation_all_to_all = get_edge_info_for_scenestreamer(
            q_k_relation=relation_all_to_all,
            q_k_valid_mask=relation_valid_mask,
            relation_model=self.relation_embed_4d,
            relation_model_1d=self.relation_embed_1d,
            require_relation_pairwise=require_relation_pairwise,
        )
        relation_all_to_map = self._get_relation_for_3d_token_vs_map_token(
            token_3d_pos=all_positions,
            token_3d_heading=all_headings,
            token_3d_valid_mask=all_valid_masks,
            token_3d_step=all_steps,
            # token4d_length=all_lengths,
            # token4d_width=all_widths,
            map_pos=map_position,
            map_heading=map_heading,
            map_valid_mask=map_token_valid_mask,
            knn=self.config.SCENESTREAMER_ATTENTION_KNN,
            max_distance=self.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE,
            require_relation=all_require_relation,
        )

        # import matplotlib.pyplot as plt
        # vis = all_causal_mask.flatten(1, 2)[0].cpu().numpy()
        # plt.imshow(vis)

        input_dict["model/all_token"] = all_tokens
        input_dict["model/all_to_all_info"] = relation_all_to_all
        input_dict["model/all_to_map_info"] = relation_all_to_map
        return input_dict

    def _get_relation_for_4d_token_vs_map_token(
            self, *, token_4d_pos, token_4d_heading, token_4d_valid_mask, token_4d_step,
            map_pos, map_heading, map_valid_mask,
            knn, max_distance, require_relation
    ):

        return self._get_relation_for_3d_token_vs_map_token(
            token_3d_pos=token_4d_pos.flatten(1, 2),
            token_3d_heading=token_4d_heading.flatten(1, 2),
            token_3d_valid_mask=token_4d_valid_mask.flatten(1, 2),
            token_3d_step=token_4d_step.flatten(1, 2),
            map_pos=map_pos,
            map_heading=map_heading,
            map_valid_mask=map_valid_mask,
            knn=knn,
            max_distance=max_distance,
            require_relation=require_relation.flatten(1, 2),
        )

    def _get_relation_for_3d_token_vs_map_token(
            self, *, token_3d_pos, token_3d_heading, token_3d_valid_mask, token_3d_step,
            map_pos, map_heading, map_valid_mask,
            knn, max_distance, require_relation,
            token3d_width=None, token3d_length=None
    ):
        if token3d_width is not None:
            token3d_width = token3d_width.flatten(1, 2)
            token3d_length = token3d_length.flatten(1, 2)
            non_agent_relation = False
            raise ValueError
        else:
            non_agent_relation = True
        a2m_3d = self.config.MODEL.ALL_TO_MAP_3D
        q_k_relation, q_k_valid_mask, require_relation_output = relation.compute_relation_for_scenestreamer(
            query_pos=token_3d_pos,  # B, TN, D
            query_heading=token_3d_heading,
            query_valid_mask=token_3d_valid_mask,
            query_step=None if a2m_3d else token_3d_step,
            query_width=token3d_width,
            query_length=token3d_length,
            key_pos=map_pos,
            key_heading=map_heading,
            key_valid_mask=map_valid_mask,
            key_step=None if a2m_3d else torch.zeros_like(map_heading, dtype=torch.int64),
            key_width=None,
            key_length=None,
            causal_valid_mask=None,
            knn=knn,
            max_distance=max_distance,
            gather=False,
            non_agent_relation=non_agent_relation,
            require_relation=require_relation,
            require_relation_for_key=map_valid_mask,
        )
        relation_info = get_edge_info_for_scenestreamer(
            q_k_valid_mask=q_k_valid_mask,
            q_k_relation=q_k_relation,
            relation_model=self.relation_embed_3d if a2m_3d else self.relation_embed_4d,
            relation_model_1d=self.relation_embed_1d,
            require_relation_pairwise=require_relation_output,
        )
        return relation_info


class SceneStreamerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, d_model, ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model

    def forward(self, *, input_dict, use_cache=None, cache=None):
        new_past_key_value_list = []
        output_dict = input_dict
        for layer_idx, mod in enumerate(self.layers):
            layer_input_cache = cache[layer_idx] if cache is not None else None
            output_dict, layer_cache = mod(input_dict=output_dict, use_cache=use_cache, cache=layer_input_cache)
            if use_cache:
                new_past_key_value_list.append(layer_cache)
        if use_cache:
            return output_dict, new_past_key_value_list
        return output_dict


class SceneStreamerDecoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, ):
        super().__init__()

        # ===== map tokens =====
        # self.map_to_map_attention = MultiheadAttentionLayer(
        #     d_model=d_model,
        #     n_heads=nhead,
        #     dropout=dropout,
        #     simple_relation=True,
        #     simple_relation_factor=1,
        #     is_v7=True,
        #     update_relation=False,
        #     add_relation_to_v=False,
        # )
        # self.map_norm = nn.LayerNorm(d_model)
        # self.map_to_map_rel_norm = nn.LayerNorm(d_model)

        # ===== all tokens =====
        self.all_to_map_attention = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=True,
            simple_relation_factor=1,
            is_v7=True,
            update_relation=False,
            add_relation_to_v=False,
        )
        self.all_to_map_norm = nn.LayerNorm(d_model)
        self.all_to_map_rel_norm = nn.LayerNorm(d_model)

        self.all_to_all_attention = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=True,
            simple_relation_factor=1,
            is_v7=True,
            update_relation=False,
            add_relation_to_v=False,
        )
        self.all_to_all_norm = nn.LayerNorm(d_model)
        self.all_to_all_rel_norm = nn.LayerNorm(d_model)

        # ===== feed forward =====
        self.mlp_prenorm = nn.LayerNorm(d_model)
        self.mlp = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[4 * d_model, d_model], ret_before_act=True, without_norm=True
        )

    def forward(self, *, input_dict, use_cache=None, cache=None):
        map_token = input_dict["model/map_token"]

        # ===== all token to map cross-attention =====
        input_all_token = input_dict["model/all_token"]
        output_all_token = self.all_to_map_norm(input_all_token)
        all_rel = input_dict["model/all_to_map_info"]["edge_features"]
        all_rel = self.all_to_map_rel_norm(all_rel)
        output_all_token, _, _ = self.all_to_map_attention(
            q=output_all_token,
            k=map_token,
            edge_features=all_rel,
            edge_features_v=None,
            edge_index=input_dict["model/all_to_map_info"]["edge_index"],
            use_cache=False,
            cache=None,
        )
        output_all_token = input_all_token + output_all_token

        # ===== all token self-attention =====
        output_all_token = self.all_to_all_norm(output_all_token)
        all_to_all_rel = input_dict["model/all_to_all_info"]["edge_features"]
        all_to_all_rel = self.all_to_all_rel_norm(all_to_all_rel)
        output_all_token, new_cache, _ = self.all_to_all_attention(
            q=output_all_token,
            k=output_all_token,
            edge_features=all_to_all_rel,
            edge_features_v=None,
            edge_index=input_dict["model/all_to_all_info"]["edge_index"],
            use_cache=use_cache,
            cache=cache,
        )
        output_all_token = input_all_token + output_all_token

        # === Feed-forward layer ===
        output_all_token = self.mlp_prenorm(output_all_token)
        output_all_token = self.mlp(output_all_token)
        all_token = input_all_token + output_all_token
        input_dict["model/all_token"] = all_token

        return input_dict, new_cache
