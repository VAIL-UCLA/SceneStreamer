import torch
import torch.nn as nn

from scenestreamer.dataset import constants
from scenestreamer.models import relation
from scenestreamer.models.layers import common_layers, fourier_embedding
from scenestreamer.models.layers.gpt_decoder_layer import MultiCrossAttTransformerDecoderLayer, MultiCrossAttTransformerDecoder
from scenestreamer.models.motion_decoder import create_causal_mask
from scenestreamer.models.motion_decoder_gpt import get_edge_info
from scenestreamer.models.scene_encoder import mode_agent_id
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import utils


class MotionDecoderGPTDiffusion(nn.Module):
    def __init__(self, config):

        # TODO: ADD_RELATION_TO_V is not implemented!
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)
        print("config.MODEL.ADD_RELATION_TO_V", config.MODEL.ADD_RELATION_TO_V)

        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        # self.num_actions = get_action_dim(self.config)
        dropout = self.config.MODEL['DROPOUT_OF_ATTN']
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        # use_condition = self.config.ACTION_LABEL.USE_ACTION_LABEL or self.config.ACTION_LABEL.USE_SAFETY_LABEL
        # self.use_condition = use_condition
        assert self.config.MODEL.NAME in ['gpt']
        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        assert self.add_pe_for_token
        use_adaln = self.config.USE_ADALN
        self.use_adaln = use_adaln

        simple_relation = self.config.SIMPLE_RELATION
        simple_relation_factor = 1
        self.decoder = MultiCrossAttTransformerDecoder(
            decoder_layer=MultiCrossAttTransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dropout=dropout,
                use_adaln=use_adaln,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
            self_attention_knn=self.config.MODEL['SELF_ATTN_KNN'],
            cross_attention_knn=self.config.MODEL['CROSS_ATTN_KNN'],
        )

        assert self.config.BACKWARD_PREDICTION is False

        assert self.config.ADD_CONTOUR_RELATION is True

        assert self.config.SIMPLE_RELATION is True
        relation_d_model = d_model // simple_relation_factor

        self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
            input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64
        )
        self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
            input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64
        )
        self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
            input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64
        )

        self.type_embed = common_layers.Tokenizer(
            num_actions=constants.NUM_TYPES, d_model=d_model, add_one_more_action=False
        )
        # self.action_embed = common_layers.Tokenizer(
        #     num_actions=self.num_actions, d_model=d_model, add_one_more_action=True
        # )
        self.shape_embed = common_layers.build_mlps(c_in=3, mlp_channels=[d_model, d_model], ret_before_act=True)

        if self.config.REMOVE_AGENT_FROM_SCENE_ENCODER:
            self.agent_id_embed = common_layers.Tokenizer(
                num_actions=self.config.PREPROCESSING.MAX_AGENTS, d_model=self.d_model, add_one_more_action=False
            )

        tokenizer = get_tokenizer(self.config)
        # motion_features = tokenizer.get_motion_feature()
        # if tokenizer.use_type_specific_bins:
        #     motion_features = torch.cat([motion_features, torch.zeros(1, 3, 4)], dim=0)
        # else:
        #     motion_features = torch.cat([motion_features, torch.zeros(1, 4)], dim=0)
        self.tokenizer = tokenizer
        # self.register_buffer("motion_features", motion_features)

        if self.tokenizer.use_delta_delta:
            agent_motion_embed_dim = 5 * 2 + 2
        else:
            agent_motion_embed_dim = 5 * 3 + 2

        self.agent_motion_embed = fourier_embedding.FourierEmbedding(
            input_dim=agent_motion_embed_dim, hidden_dim=d_model, num_freq_bands=64
        )

        # Special tokens: Invalid, Valid, Start, Masked, Unused.
        self.special_token_embed = common_layers.Tokenizer(
            num_actions=5, d_model=self.d_model, add_one_more_action=False
        )

        # if self.config.BACKWARD_PREDICTION:
        #     self.in_backward_prediction_embed = common_layers.Tokenizer(
        #         num_actions=2, d_model=self.d_model, add_one_more_action=False
        #     )

        from scenestreamer.diffusion.diffusion_loss import DiffLoss
        # diffloss_w = 4 * self.config.MODEL.D_MODEL
        diffloss_w = self.config.MODEL.D_MODEL
        diffloss_d = 3
        grad_checkpointing = False
        decoder_embed_dim = self.config.MODEL.D_MODEL

        if self.tokenizer.use_delta_delta:
            token_embed_dim = 5 * 2
        else:
            token_embed_dim = 5 * 3

        predict_xstart = False
        diffusion_steps = 100
        num_sampling_steps = '100'
        self.diffusion_loss = DiffLoss(
            target_channels=token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
            predict_xstart=predict_xstart,
            diffusion_steps=diffusion_steps,
        )

    def randomize_modeled_agent_id(self, data_dict, clip_agent_id=False):
        modeled_agent_id = data_dict["decoder/agent_id"]
        # batch_index = data_dict.get("batch_idx", None)
        if not self.config.MODEL.RANDOMIZE_AGENT_ID:
            if clip_agent_id:
                modeled_agent_id = mode_agent_id(
                    modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True
                )
            return modeled_agent_id

        # assert batch_index is not None, "Need batch index to randomize agent id!"
        # batch_to_unique = {}
        # for i, b in enumerate(batch_index):
        #     b = b.item()
        #     if b not in batch_to_unique:
        #         batch_to_unique[b] = len(batch_to_unique)

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

        # Allocate same agent id to the same batch
        # return_modeled_agent_id = torch.full_like(modeled_agent_id, -1)
        # for i, b in enumerate(batch_index):
        #     b = b.item()
        #     return_modeled_agent_id[i] = new_modeled_agent_id[batch_to_unique[b]]
        # return return_modeled_agent_id
        return new_modeled_agent_id

    def forward(self, input_dict, use_cache=False, a2a_knn=None, a2t_knn=None, a2s_knn=None):
        in_evaluation = input_dict["in_evaluation"][0].item()

        # === Process scene embedding ===
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]

        # === Process action embedding ===
        # input_action = input_dict["decoder/input_action"]
        modeled_agent_delta = input_dict["decoder/modeled_agent_delta"]
        # B, T_skipped, N = input_action.shape
        input_special_token = input_dict["decoder/input_action"]
        B, T_skipped, N = input_special_token.shape

        if self.config.REMOVE_AGENT_FROM_SCENE_ENCODER:
            if in_evaluation:
                assert "decoder/randomized_modeled_agent_id" in input_dict, "Need to provide randomized modeled agent id for evaluation! Please call randomize_modeled_agent_id()"
                new_modeled_agent_id = input_dict["decoder/randomized_modeled_agent_id"]
            else:
                new_modeled_agent_id = self.randomize_modeled_agent_id(input_dict, clip_agent_id=False)
            modeled_agent_pe = self.agent_id_embed(new_modeled_agent_id)

            # print("modeled_agent_pe", new_modeled_agent_id[0])
        else:
            modeled_agent_pe = input_dict["encoder/modeled_agent_pe"]
        assert modeled_agent_pe.shape == (B, N, self.d_model), modeled_agent_pe.shape
        modeled_agent_pe = modeled_agent_pe[:, None].expand(B, T_skipped, N, self.d_model)

        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        assert action_valid_mask.shape == (B, T_skipped, N), (action_valid_mask.shape, (B, T_skipped, N))
        agent_pos = input_dict["decoder/modeled_agent_position"][..., :2]
        agent_heading = input_dict["decoder/modeled_agent_heading"]

        # ===== Prepare input tokens =====
        if "decoder/input_step" not in input_dict:
            input_dict["decoder/input_step"] = torch.arange(T_skipped).to(scene_token.device)
        agent_step = input_dict["decoder/input_step"].reshape(1, T_skipped, 1).expand(B, T_skipped, N)

        # Shape embedding and type embedding
        type_emb = self.type_embed(input_dict["decoder/agent_type"])[:, None].expand(B, T_skipped, N, self.d_model)
        shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape"]
                                     )[:, None].expand(B, T_skipped, N, self.d_model)
        special_tok_emb = self.special_token_embed(input_special_token)

        # The input token contains:
        # 1. Special token (start, end, padding, masked)
        # 2. Modeled agent id
        # 3. Type embedding
        # 4. Shape embedding
        # 5. Last action (15-dim)
        # 6. modeled_agent_delta
        # No need to add modeled_agent_delta as the model will take care.
        input_agent_motion = input_dict["decoder/input_agent_motion"]
        cont_input = torch.cat([input_agent_motion[action_valid_mask], modeled_agent_delta[action_valid_mask]], dim=-1)
        action_token = self.agent_motion_embed(
            continuous_inputs=cont_input,
            categorical_embs=[
                special_tok_emb[action_valid_mask],
                modeled_agent_pe[action_valid_mask],
                type_emb[action_valid_mask],
                shape_emb[action_valid_mask],
            ]
        )
        action_token = utils.unwrap(action_token, action_valid_mask)
        assert action_token.shape == (B, T_skipped, N, self.d_model)
        assert action_valid_mask.shape == (B, T_skipped, N)

        # ===== Get agent-condition relation =====
        condition_token = None
        # if self.config.ACTION_LABEL.USE_SAFETY_LABEL:
        #     action_label_safety = self.action_label_tokenizer_safety(input_dict["decoder/label_safety"])
        #     condition_token = action_label_safety[:, None]
        #     if self.use_adaln:
        #         pass
        #     else:
        #         action_token += condition_token

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
        if self.config.ADD_CONTOUR_RELATION:
            agent_shape_no_time = input_dict["decoder/current_agent_shape"
                                             ]  # .reshape(B, 1, N, 3).expand(B, real_T, N, 3)
            agent_length = agent_shape_no_time[..., 0]
            agent_width = agent_shape_no_time[..., 1]
            a2t_kwargs = dict(
                include_contour=True,
                query_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                query_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                key_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
                key_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
            )

        if self.config.SIMPLE_RELATION:
            relation_func = relation.compute_relation_simple_relation
        else:
            relation_func = relation.compute_relation

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
            return_pe=False,
            # key_vel=key_vel,
            # query_vel=agent_vel_bnt.flatten(0, 1),
            **a2t_kwargs
        )
        a2t_rel_pe = utils.unwrap(self.relation_embed_a2t(a2t_rel_feat[a2t_mask]), a2t_mask)
        a2t_info = get_edge_info(attn_valid_mask=a2t_mask, rel_pe_cross=a2t_rel_pe)

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
            return_pe=False,
            # query_vel=agent_vel.flatten(0, 1),
            # key_vel=agent_vel.flatten(0, 1),
            **a2a_kwargs
        )
        a2a_rel_pe = utils.unwrap(self.relation_embed_a2a(a2a_rel_feat[a2a_mask]), a2a_mask)
        a2a_info = get_edge_info(attn_valid_mask=a2a_mask, rel_pe_cross=a2a_rel_pe)

        # ===== Get agent-scene relation =====
        a2a_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            kw = torch.zeros_like(input_dict["encoder/scenario_position"][..., 0])
            a2a_kwargs = dict(
                include_contour=True,
                query_width=w,
                query_length=l,
                key_width=kw,
                key_length=kw,
                non_agent_relation=True
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
            gather=False,
            return_pe=False,
            **a2a_kwargs
        )
        a2s_rel_pe = utils.unwrap(self.relation_embed_a2s(a2s_rel_feat[a2s_mask]), a2s_mask)
        a2s_info = get_edge_info(attn_valid_mask=a2s_mask, rel_pe_cross=a2s_rel_pe)

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
            condition_token=condition_token if self.use_adaln else None,
            use_cache=use_cache,  # We don't need decoder to take care cache.
            past_key_value_list=past_key_value_list
        )

        if use_cache:
            decoded_tokens, past_key_value_list = decoded_tokens
            for l in past_key_value_list:
                if l:
                    l.append((B * N, real_T))
            input_dict["decoder/cache"] = past_key_value_list

        input_dict["decoder/decoded_tokens"] = decoded_tokens
        return input_dict

    def update_cache(self, input_dict):
        # TODO: Do we have cache for diffusion?

        assert self.config.EVALUATION.USE_CACHE
        if "decoder/modeled_agent_position_history" not in input_dict:
            input_dict["decoder/modeled_agent_position_history"] = input_dict["decoder/modeled_agent_position"].clone()
            input_dict["decoder/modeled_agent_velocity_history"] = input_dict["decoder/modeled_agent_velocity"].clone()
            input_dict["decoder/modeled_agent_heading_history"] = input_dict["decoder/modeled_agent_heading"].clone()
            input_dict["decoder/modeled_agent_valid_mask_history"] = input_dict["decoder/input_action_valid_mask"
                                                                                ].clone()
            input_dict["decoder/modeled_agent_step_history"] = input_dict["decoder/input_step"].clone()
        else:
            input_dict["decoder/modeled_agent_position_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_position_history"], input_dict["decoder/modeled_agent_position"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_velocity_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_velocity_history"], input_dict["decoder/modeled_agent_velocity"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_heading_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_heading_history"], input_dict["decoder/modeled_agent_heading"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_valid_mask_history"] = torch.cat(
                [
                    input_dict["decoder/modeled_agent_valid_mask_history"],
                    input_dict["decoder/input_action_valid_mask"],
                ],
                dim=1
            )
            input_dict["decoder/modeled_agent_step_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_step_history"], input_dict["decoder/input_step"]], dim=0
            )

    def get_diffusion_loss(self, data_dict):

        target = data_dict['decoder/target_agent_motion']
        target_valid_mask = data_dict['decoder/target_action_valid_mask']
        t = target[target_valid_mask]

        out = data_dict["decoder/decoded_tokens"]
        z = out[target_valid_mask]

        # TODO: Consider getting this back?
        # self.diffusion_batch_mul = 4
        # z = z.repeat(self.diffusion_batch_mul, 1)
        # t = t.repeat(self.diffusion_batch_mul, 1)

        loss_dict = self.diffusion_loss(z=z, target=t)

        mean = t.mean(0)
        std = t.std(0)
        # assert mean.shape[0] == 15
        for i in range(mean.shape[0]):
            loss_dict[f"motion_stat/target_mean_{i}"] = mean[i]  # .item()
            loss_dict[f"motion_stat/target_std_{i}"] = std[i]  # .item()
            loss_dict[f"motion_stat/pred_mean_{i}"] = loss_dict["model_output"][i]  # .item()

        loss_dict["toks"] = z.shape[0]

        # TODO: Remove stat in formal experiment.
        print("\n==== Diffusion Loss ====")
        print("TARG", [round(v.item(), 4) for v in mean.cpu().detach().numpy()])
        print("TARG_MAX", [round(v.item(), 4) for v in t.max(0).values.cpu().detach().numpy()])
        print("TARG_MIN", [round(v.item(), 4) for v in t.min(0).values.cpu().detach().numpy()])
        print("PRED", [round(v.item(), 4) for v in loss_dict["model_output"]])
        print("MSE", loss_dict["mse"].mean().item())
        print("==== Diffusion Loss END ====")

        loss_dict.pop("model_output")
        return loss_dict

    def sample_diffusion(self, input_dict, use_cache):

        # TODO: Do we need to introduce the EMA model here? Or we can make another copy.

        out = self.forward(input_dict, use_cache)
        tok = out["decoder/decoded_tokens"]
        m = out["decoder/input_action_valid_mask"]
        z = tok[m]

        temperature = 1.0
        cfg = 1.0
        predicted = self.diffusion_loss.sample(z, temperature, cfg)

        predicted = utils.unwrap(predicted, m)

        out["decoder/output_action"] = predicted
        return out
