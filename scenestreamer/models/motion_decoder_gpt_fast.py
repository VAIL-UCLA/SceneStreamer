import torch
import torch.nn as nn
import torch.nn.functional as F

from scenestreamer.dataset import constants
from scenestreamer.models import relation
from scenestreamer.models.layers import common_layers, fourier_embedding
from scenestreamer.models.layers.gpt_decoder_layer import MultiCrossAttTransformerDecoderLayer, MultiCrossAttTransformerDecoder
from scenestreamer.models.motion_decoder import create_causal_mask
from scenestreamer.models.motion_decoder_gpt import MotionDecoderGPT as MotionDecoderGPTBase, get_edge_info_new
from scenestreamer.tokenization import get_action_dim, get_tokenizer, START_ACTION as MOTION_START_ACTION, END_ACTION as MOTION_END_ACTION
from scenestreamer.utils import utils


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.
    Applies standard layer normalization and then conditions the normalized output
    on a latent vector z via learned affine parameters.
    """
    def __init__(self, hidden_size, conditioning_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # These projections output a modulation (scale and bias) for each feature.
        self.gamma_proj = nn.Linear(conditioning_dim, hidden_size)
        self.beta_proj = nn.Linear(conditioning_dim, hidden_size)
        # We disable affine parameters inside the LayerNorm since they will be provided by z.
        self.ln = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, x, z):
        """
        x: Tensor of shape (..., hidden_size) to be normalized.
        z: Conditioning tensor of shape (B, conditioning_dim) if x is [B, seq_len, hidden_size]
           or shape (B, conditioning_dim) when x is [B, hidden_size].
        """
        normalized = self.ln(x)
        # If x is 3D (B, seq_len, hidden_size), unsqueeze z along seq_len dimension.
        if normalized.dim() == 3:
            assert z.dim() == 2
            gamma = self.gamma_proj(z).unsqueeze(0)  # Note that input x is NOT batch first.
            beta = self.beta_proj(z).unsqueeze(0)
        elif normalized.dim() == 2:
            gamma = self.gamma_proj(z)  # [B, hidden_size]
            beta = self.beta_proj(z)
        else:
            raise ValueError("Unsupported input tensor shape for AdaLayerNorm")
        # Modulate normalized activations.
        return normalized * (1 + gamma) + beta


class TransformerBlock(nn.Module):
    """
    A single transformer block that uses adaptive layer normalization.
    It includes a self-attention layer and a feed-forward network.
    """
    def __init__(self, hidden_size, num_heads, conditioning_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=False)
        self.adaln1 = AdaLayerNorm(hidden_size, conditioning_dim)
        self.adaln2 = AdaLayerNorm(hidden_size, conditioning_dim)

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


class TransformerPredictionHead(nn.Module):
    """
    Transformer-based prediction head with adaptive layer normalization.

    The forward function takes:
      - x: LongTensor of shape [B, seq_len]. It may contain -1, which will be replaced by the pad token.
      - z: FloatTensor of shape [B, conditioning_dim] used to condition the AdaLN layers.

    The autoregressive generate function handles generation with proper <s> and <e> tokens.
    During training x is assumed not to include these special tokens.
    """
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_heads,
        num_layers,
        conditioning_dim,
        max_seq_len=512,
        dropout=0.1,
        pad_token=0,
        sos_token=1,
        eos_token=2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.num_actions = self.vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding.
        self.token_embedding = common_layers.Tokenizer(vocab_size, hidden_size, add_one_more_action=False)
        # Positional encoding: learnable embeddings.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        # Stack of transformer blocks.
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, conditioning_dim, dropout) for _ in range(num_layers)]
        )
        # Final normalization (can be standard LN).
        self.ln_final = nn.LayerNorm(hidden_size)
        # Project back to vocabulary logits.
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def get_action_embedding(self, action):
        action_emb = None
        for i in range(action.shape[-1]):
            v = self.token_embedding(action[..., i])
            v = torch.where((action[..., i] == -1)[..., None], 0, v)
            if action_emb is None:
                action_emb = v
            else:
                action_emb += v
        mask = (action == -1).all(dim=-1)
        action_emb[mask] = 0
        return action_emb

    def prepare_for_training(self, x):
        """
        This function append start token before the sequence and it looks for the first -1 token and replace it with end token.
        """
        assert x.ndim == 2
        B, T = x.shape

        newx = x.clone()

        #### DEBUG CODE:
        # newx = torch.full_like(newx, -1)
        # newx[:, 0] = 777
        # x = torch.full_like(x, -1)
        # x[:, 0] = 777
        # newx = torch.where(newx != -1, 777, newx)

        first_neg1_ind = (x == -1).float().argmax(dim=-1)
        all_invalid_mask = (x == -1).all(dim=-1)
        all_valid_mask = (x != -1).all(dim=-1)
        newx[torch.arange(B), first_neg1_ind] = self.eos_token
        newx = torch.where(all_valid_mask[:, None], x, newx)

        start_token = torch.full((B, 1), self.sos_token, dtype=torch.long, device=x.device)
        end_token = torch.full((B, 1), -1, dtype=torch.long, device=x.device)
        newx = torch.cat([start_token, newx, end_token], dim=1)
        newx[all_valid_mask, -1] = self.eos_token

        key_padding_valid_mask = newx != -1
        key_padding_valid_mask[all_invalid_mask] = False
        newx[~key_padding_valid_mask] = self.pad_token

        # In padding_mask, True means the token will be ignored.
        key_padding_mask = ~key_padding_valid_mask

        # assert (x[:, 0] == 1025).all()
        assert (newx[newx[:, -1] != self.pad_token][:, -1] == self.eos_token).all()

        return newx, key_padding_mask

    def forward(self, x, z, key_padding_mask=None, prepare_for_training=True):
        """
        x: LongTensor of shape [B, seq_len]. May contain -1 (which will be replaced by pad_token).
        z: FloatTensor of shape [B, conditioning_dim].
        Returns logits of shape [B, seq_len, vocab_size].
        """
        assert x.dim() == 2, "Input tensor must have shape [B, seq_len]"
        assert z.dim() == 2, "Conditioning tensor must have shape [B, conditioning_dim]"

        info = {}
        if prepare_for_training:
            x, key_padding_mask = self.prepare_for_training(x)
            info["fast_input_token"] = x
            info["fast_sos_token"] = self.sos_token
            info["fast_eos_token"] = self.eos_token
            info["fast_pad_token"] = self.pad_token

        # Compute token embeddings.
        emb = self.token_embedding(x)  # [B, seq_len, hidden_size]
        seq_len = emb.size(1)

        # Add positional embeddings.
        emb = emb + self.pos_embedding[:, :seq_len, :]

        # Transpose to [seq_len, B, hidden_size] for the PyTorch attention module.
        h = emb.transpose(0, 1)

        # Optionally, one may create an attention mask to prevent attending to pad positions.
        # key_padding_mask is assumed to be provided (or could be computed here based on x == pad_token)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)
        attn_mask = attn_mask < 0
        for layer in self.layers:
            h = layer(h, z, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # Final normalization.
        h = self.ln_final(h)
        # Transpose back to [B, seq_len, hidden_size]
        h = h.transpose(0, 1)
        # Compute logits.
        logits = self.output_layer(h)


        # # Compute the probability for GT tokens
        # # TODO: remove this
        # if x.shape[1] > 1:
        #     tmp = torch.where(x[:, 1:] == -1, self.pad_token, x[:, 1:])
        #     log_prob = utils.masked_average(
        #     torch.distributions.Categorical(logits=logits[:, :-1]).log_prob(tmp),~key_padding_mask[:, :-1], dim=1).mean()
        #     print("log_prob:", log_prob.item())
        logits[(key_padding_mask == True).all(-1)] = 0
        return logits, info

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
        generated = torch.full((B, 1), self.sos_token, dtype=torch.long, device=device)

        # key_padding_mask = torch.zeros(B, 1, dtype=torch.float32, device=device)
        key_padding_valid_mask_bool = torch.ones(B, 1, dtype=torch.bool, device=device)
        key_padding_mask = (~key_padding_valid_mask_bool).clone()

        for step in range(max_length - 1):  # already have one token
            # Compute logits for the current sequence.
            logits, _ = self.forward(
                generated, z, prepare_for_training=False, key_padding_mask=key_padding_mask
            )  # [B, seq_len, vocab_size]
            # Focus on the last time step.
            last_logits = logits[:, -1, :]  # [B, vocab_size]

            # Can't select sos token..
            last_logits[:, self.sos_token] = -float("inf")

            if step == 0:
                # Don't allow you to select the end token...
                last_logits[:, self.eos_token] = -float("inf")

            if greedy:
                next_token = last_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            else:
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Mask out the padding tokens
            next_token[~key_padding_valid_mask_bool] = -1

            # Append the predicted token.
            generated = torch.cat([generated, next_token], dim=1)

            key_padding_valid_mask_bool = (next_token != self.pad_token) & key_padding_valid_mask_bool & (
                next_token != self.eos_token
            ) & (next_token != -1)

            key_padding_mask = torch.cat([key_padding_mask, ~key_padding_valid_mask_bool], dim=1)

            # Check if all sequences have produced an <e> token.
            if not key_padding_valid_mask_bool.any():
                break

        out = generated[:, 1:]
        assert (out != self.sos_token).all(), "Generated sequence should not contain start token"
        return out


class MotionDecoderGPT(MotionDecoderGPTBase):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS

        # self.num_actions = get_action_dim(self.config)

        dropout = self.config.MODEL.DROPOUT
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        assert self.config.MODEL.NAME in ['gpt']
        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        assert self.add_pe_for_token

        # TODO: Implement this
        use_adaln = False
        self.use_adaln = use_adaln

        simple_relation = self.config.SIMPLE_RELATION
        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        is_v7 = self.config.MODEL.IS_V7
        self.is_v7 = is_v7
        self.decoder = MultiCrossAttTransformerDecoder(
            decoder_layer=MultiCrossAttTransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dropout=dropout,
                use_adaln=use_adaln,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor,
                is_v7=is_v7,
                update_relation=self.config.UPDATE_RELATION,
                add_relation_to_v=self.config.MODEL.ADD_RELATION_TO_V,
                remove_rel_norm=self.config.REMOVE_REL_NORM
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
        )
        # self.prediction_head = common_layers.build_mlps(
        #     c_in=d_model, mlp_channels=[d_model, self.num_actions], ret_before_act=True, is_v7=is_v7, zero_init=is_v7
        # )
        # if self.use_adaln:
        #     self.prediction_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        #     self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        # else:
        self.prediction_prenorm = nn.LayerNorm(d_model)

        relation_d_model = d_model // simple_relation_factor
        self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
            input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
        )
        self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
            input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
        )
        self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
            input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
        )

        self.type_embed = common_layers.Tokenizer(
            num_actions=constants.NUM_TYPES, d_model=d_model, add_one_more_action=False
        )
        # self.action_embed = common_layers.Tokenizer(
        #     num_actions=self.num_actions, d_model=d_model, add_one_more_action=True
        # )
        self.shape_embed = common_layers.build_mlps(
            c_in=3, mlp_channels=[d_model, d_model], ret_before_act=True, is_v7=is_v7
        )

        # if self.config.REMOVE_AGENT_FROM_SCENE_ENCODER:
        self.agent_id_embed = common_layers.Tokenizer(
            num_actions=self.config.PREPROCESSING.MAX_AGENTS, d_model=self.d_model, add_one_more_action=False
        )

        self.motion_embed = fourier_embedding.FourierEmbedding(
            input_dim=2, hidden_dim=d_model, num_freq_bands=64, is_v7=is_v7
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
        self.prediction_head = TransformerPredictionHead(
            vocab_size=self.tokenizer.fast_tokenizer.vocab_size + 3,
            hidden_size=self.d_model,
            num_heads=4,
            num_layers=3,
            conditioning_dim=self.d_model,
            max_seq_len=20,
            pad_token=self.tokenizer.fast_tokenizer.vocab_size,
            sos_token=self.tokenizer.fast_tokenizer.vocab_size + 1,
            eos_token=self.tokenizer.fast_tokenizer.vocab_size + 2,
            dropout=0.0
        )

    def forward(self, input_dict, use_cache=False, a2a_knn=None, a2t_knn=None, a2s_knn=None):
        in_evaluation = input_dict["in_evaluation"][0].item()

        # num_heads = self.num_heads
        # === Process scene embedding ===
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]

        # === Process action embedding ===
        input_action = input_dict["decoder/input_action"]
        modeled_agent_delta = input_dict["decoder/modeled_agent_delta"]
        B, T_skipped, N = input_action.shape[:3]

        if in_evaluation:
            assert "decoder/randomized_modeled_agent_id" in input_dict, "Need to provide randomized modeled agent id for evaluation! Please call randomize_modeled_agent_id()"
            new_modeled_agent_id = input_dict["decoder/randomized_modeled_agent_id"]
        else:
            new_modeled_agent_id = self.randomize_modeled_agent_id(input_dict, clip_agent_id=False)
        modeled_agent_pe = self.agent_id_embed(new_modeled_agent_id)

        assert modeled_agent_pe.shape == (B, N, self.d_model), (B, N, self.d_model, modeled_agent_pe.shape)
        modeled_agent_pe = modeled_agent_pe[:, None].expand(B, T_skipped, N, self.d_model)

        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        assert action_valid_mask.shape == (B, T_skipped, N), (action_valid_mask.shape, (B, T_skipped, N))
        agent_pos = input_dict["decoder/modeled_agent_position"]
        agent_heading = input_dict["decoder/modeled_agent_heading"]

        # ===== Prepare input tokens =====
        if "decoder/input_step" not in input_dict:
            input_dict["decoder/input_step"] = torch.arange(T_skipped).to(input_action.device)
        agent_step = input_dict["decoder/input_step"].reshape(1, T_skipped, 1).expand(B, T_skipped, N)

        # Shape embedding and type embedding
        type_emb = self.type_embed(input_dict["decoder/agent_type"])[:, None].expand(B, T_skipped, N, self.d_model)
        shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape"]
                                     )[:, None].expand(B, T_skipped, N, self.d_model)

        valid_actions = input_action[action_valid_mask]

        is_start_actions = valid_actions[..., 0] == MOTION_START_ACTION

        special_tok = torch.full([
            valid_actions.shape[0],
        ], 0, device=valid_actions.device, dtype=torch.long)
        special_tok[is_start_actions] = 1
        if self.config.BACKWARD_PREDICTION:
            is_end_actions = valid_actions == MOTION_END_ACTION
            special_tok[is_end_actions] = 2
            valid_actions[is_end_actions] = -1
        special_tok_emb = self.special_token_embed(special_tok)
        # TODO: Can add more special tokens in future

        valid_actions[is_start_actions] = -1
        action_emb = self.prediction_head.get_action_embedding(valid_actions)

        motion_feat = torch.cat([modeled_agent_delta[action_valid_mask]], dim=-1)

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

        # ===== Get agent-temporal relation =====
        # BTND -> BNTD
        agent_pos_bntd = torch.permute(agent_pos, [0, 2, 1, 3])
        agent_heading_bnt = torch.permute(agent_heading, [0, 2, 1])
        agent_mask_bnt = torch.permute(action_valid_mask, [0, 2, 1])
        agent_step_bnt = torch.permute(agent_step, [0, 2, 1])
        # agent_vel_bnt = torch.permute(agent_vel, [0, 2, 1, 3])
        if use_cache:
            self.update_cache(input_dict)

            agent_pos_with_history = input_dict["decoder/modeled_agent_position_history"]
            agent_heading_with_history = input_dict["decoder/modeled_agent_heading_history"]
            agent_mask_with_history = input_dict["decoder/modeled_agent_valid_mask_history"]
            agent_step_with_history = input_dict["decoder/modeled_agent_step_history"]
            # agent_vel_with_history = input_dict["decoder/modeled_agent_velocity_history"]
            real_T = agent_mask_with_history.shape[1]
            key_pos = torch.permute(agent_pos_with_history, [0, 2, 1, 3]).flatten(0, 1)
            # key_vel = torch.permute(agent_vel_with_history, [0, 2, 1, 3]).flatten(0, 1)
            key_heading = torch.permute(agent_heading_with_history, [0, 2, 1]).flatten(0, 1)
            key_mask = torch.permute(agent_mask_with_history, [0, 2, 1]).flatten(0, 1)
            causal_valid_mask = None
            key_step = agent_step_with_history.reshape(1, 1, -1).expand(B, N, -1).flatten(0, 1)
        else:
            real_T = T_skipped
            # key_vel = agent_vel_bnt.flatten(0, 1)
            key_pos = agent_pos_bntd.flatten(0, 1)
            key_heading = agent_heading_bnt.flatten(0, 1)
            key_mask = agent_mask_bnt.flatten(0, 1)
            key_step = agent_step_bnt.flatten(0, 1)
            causal_valid_mask = create_causal_mask(T=real_T, N=1, is_valid_mask=True).to(action_token.device)

        assert agent_pos_bntd.shape == (B, N, T_skipped, 2)

        a2t_kwargs = {}
        agent_shape_no_time = input_dict["decoder/current_agent_shape"]  # .reshape(B, 1, N, 3).expand(B, real_T, N, 3)
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

        relation_func = relation.compute_relation_simple_relation
        a2t_rel_feat, a2t_mask, _ = relation_func(
            query_pos=agent_pos_bntd.flatten(0, 1),  # BN, T, D
            query_heading=agent_heading_bnt.flatten(0, 1),
            query_valid_mask=agent_mask_bnt.flatten(0, 1),
            query_step=agent_step_bnt.flatten(0, 1),
            key_pos=key_pos,  # BN, T_full, D
            key_heading=key_heading,
            key_valid_mask=key_mask,
            key_step=key_step,
            hidden_dim=self.d_model,
            causal_valid_mask=causal_valid_mask,
            knn=None,
            max_distance=None,
            return_pe=False,
            # key_vel=key_vel,
            # query_vel=agent_vel_bnt.flatten(0, 1),
            **a2t_kwargs
        )
        a2t_info = get_edge_info_new(
            q_k_valid_mask=a2t_mask,
            q_k_relation=a2t_rel_feat,
            relation_model=self.relation_embed_a2t,
            relation_model_v=self.relation_embed_a2t_v if self.config.MODEL.ADD_RELATION_TO_V else None
        )

        # ===== Get agent-agent relation =====
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
            query_pos=agent_pos.flatten(0, 1),  # BT, N, D
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
            # query_vel=agent_vel.flatten(0, 1),
            # key_vel=agent_vel.flatten(0, 1),
            **a2a_kwargs
        )
        a2a_info = get_edge_info_new(
            q_k_valid_mask=a2a_mask,
            q_k_relation=a2a_rel_feat,
            relation_model=self.relation_embed_a2a,
            relation_model_v=self.relation_embed_a2a_v if self.config.MODEL.ADD_RELATION_TO_V else None
        )

        # ===== Get agent-scene relation =====
        a2s_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            kw = torch.zeros_like(input_dict["encoder/scenario_position"][..., 0])
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
            query_pos=agent_pos.flatten(1, 2),  # B, TN, D
            query_heading=agent_heading.flatten(1, 2),
            query_valid_mask=action_valid_mask.flatten(1, 2),
            query_step=agent_step.flatten(1, 2),
            key_pos=input_dict["encoder/scenario_position"],  # [..., :2],
            key_heading=input_dict["encoder/scenario_heading"],
            key_valid_mask=scenario_valid_mask,
            key_step=agent_pos.new_zeros(B, input_dict["encoder/scenario_position"].shape[1]),
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
            a2t_info=a2t_info,
            a2s_info=a2s_info,
            condition_token=None,  # TODO: Add condition token
            use_cache=use_cache,  # We don't need decoder to take care cache.
            past_key_value_list=past_key_value_list
        )

        if use_cache:
            decoded_tokens, past_key_value_list = decoded_tokens
            for l in past_key_value_list:
                if l:
                    l.append((B * N, real_T))
            input_dict["decoder/cache"] = past_key_value_list

        output_tokens = self.prediction_prenorm(decoded_tokens[action_valid_mask])

        if in_evaluation:
            pred_out = self.prediction_head.generate(z=output_tokens)

            pred_out = utils.unwrap(pred_out, action_valid_mask, fill=-1)

            input_dict["decoder/output_token"] = pred_out
            input_dict["decoder/output_logit"] = None

        else:

            target_actions = input_dict["decoder/target_action"]
            masked_target_actions = target_actions[action_valid_mask]

            valid_target_mask = (masked_target_actions != -1).any(dim=-1)

            pred_out, fast_info = self.prediction_head(masked_target_actions[valid_target_mask], z=output_tokens[valid_target_mask])

            pred_out_new = pred_out.new_zeros(valid_target_mask.shape[0], pred_out.shape[1], pred_out.shape[2])
            pred_out_new[valid_target_mask] = pred_out
            pred_out = pred_out_new

            fast_tok = fast_info["fast_input_token"].new_zeros(valid_target_mask.shape[0], pred_out.shape[1])
            fast_tok[valid_target_mask] = fast_info["fast_input_token"]
            fast_info["fast_input_token"] = fast_tok

            input_dict.update(fast_info)

            logits = utils.unwrap(pred_out.flatten(1, 2),
                                  action_valid_mask).reshape(B, T_skipped, N, -1, pred_out.shape[-1])

            assert logits.shape == (B, T_skipped, N, pred_out.shape[1], self.prediction_head.num_actions), (
                logits.shape, (B, T_skipped, N, pred_out.shape[1], self.prediction_head.num_actions)
            )
            input_dict["decoder/output_logit"] = logits

        return input_dict
