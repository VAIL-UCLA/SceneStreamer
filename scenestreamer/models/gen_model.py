import logging

import torch.nn as nn

from scenestreamer.dataset import constants
from scenestreamer.models.layers import polyline_encoder, common_layers, position_encoding_utils
from scenestreamer.models.layers.decoder_layer import TransformerDecoder, TransformerDecoderLayer
# from scenestreamer.models.motion_decoder import MotionDecoder
from scenestreamer.models.layers.encoder_layer import TransformerEncoderLayer  # as NativeTransformerEncoderLayer
from scenestreamer.models.motionlm import MotionLM, nucleus_sampling
from scenestreamer.models.ops.collapse_time import collapse_time
# from torch.nn.modules.transformer import TransformerEncoderLayer as NativeTransformerEncoderLayer
from scenestreamer.models.scene_encoder import compute_relation
from scenestreamer.tokenization.gen_tokenizers import GenTokenizer, Tokens, SceneStreamerTokenizer
from scenestreamer.utils import calculate_trajectory_probabilities

logger = logging.getLogger(__file__)


def create_causal_mask(causal_mask_offset, num_heads=None):
    """ Create the causal mask for a flattened token sequence. Tokens will not attend to future ids. Tokens for the
    agents in the same step can attend to each other.

    row: a query
    col: a key

    So for mask[100] it should see more keys than mask[0].

    Note that all +1 positions will be filled -inf.
    """
    B, L = causal_mask_offset.shape

    causal_mask_offset.masked_fill_(causal_mask_offset == -1, L)

    i = causal_mask_offset.unsqueeze(2)  # Shape (B, N, 1)
    j = causal_mask_offset.unsqueeze(1)  # Shape (B, 1, N)
    causal_mask = (i >= j)  #.int()
    causal_mask = ~causal_mask
    if num_heads is not None:
        B, L, _ = causal_mask.shape
        causal_mask = causal_mask.unsqueeze(1).expand(B, num_heads, L, L).reshape(B * num_heads, L, L)
    return causal_mask


class Tokenizer(nn.Module):
    def __init__(self, num_actions, d_model):
        super(Tokenizer, self).__init__()
        self.tokens = nn.Embedding(num_actions, d_model)  # An extra useless dummy token is used for invalid input.
        self.num_actions = num_actions

    def forward(self, actions, allow_invalid=False):
        if allow_invalid:
            actions = actions.clone()
            actions[actions < 0] = self.num_actions - 1
        return self.tokens(actions)


class SceneEncoderSceneStreamer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # TODO: Pass this from config or datasource
        SCENE_INPUT_TIME_STEPS = 11
        self.total_time_steps = SCENE_INPUT_TIME_STEPS
        self.config = config
        self.d_model = self.config.MODEL.D_MODEL
        self.num_layers = self.config.MODEL.NUM_ATTN_LAYERS

        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=constants.MAP_FEATURE_STATE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_pre_layers=1,
            out_channels=self.d_model
        )
        self.agent_mlps = common_layers.build_mlps(
            c_in=constants.AGENT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )
        self.light_mlps = common_layers.build_mlps(
            c_in=constants.TRAFFIC_LIGHT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )

        dropout = self.config.MODEL.DROPOUT_OF_ATTN
        self_attn_layers = []
        for _ in range(self.num_layers):
            self_attn_layers.append(
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.config.MODEL.NUM_ATTN_HEAD,
                    dim_feedforward=self.d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    pre_projection=self.config.MODEL.get('PRE_PROJECTION', False),
                    relative_pe=self.config.MODEL.get('RELATIVE_PE', False),
                )
            )

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        # self.agent_pe = nn.Embedding(self.config.PREPROCESSING.MAX_AGENTS, self.d_model)

        self.out = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model],
            ret_before_act=True,
        )

        self.relative_pe = self.config.MODEL.get('RELATIVE_PE', False)

    def forward(self, input_dict):

        # ===== Get shape =====
        B, T, N, D_agent = input_dict["encoder/agent_feature"].shape
        _, M, num_vector, D_vector = input_dict["encoder/map_feature"].shape
        _, _, L, D_light = input_dict["encoder/traffic_light_feature"].shape

        # ===== Embed agent feature =====
        agent_feature = input_dict["encoder/agent_feature"]
        agent_valid_mask = input_dict["encoder/agent_valid_mask"]
        agent_position = input_dict["encoder/agent_position"]
        agent_heading = input_dict["encoder/agent_heading"]
        # agent_id = input_dict["encoder/agent_id"]
        assert agent_feature.shape[:3] == agent_position.shape[:3] == agent_valid_mask.shape[:3]
        agent_feature = (
            agent_feature[:, :self.total_time_steps] * agent_valid_mask[:, :self.total_time_steps, ..., None]
        )
        agent_feature = collapse_time(agent_feature)
        agent_token = self.agent_mlps(agent_feature)  # (B, N, D)

        # Add:
        # agent_pe = self.agent_pe(agent_id)  # (B, N, D)
        # agent_token += agent_pe
        agent_pe = input_dict["encoder/agent_pe"]
        agent_token += agent_pe

        assert agent_token.shape == (B, N, self.d_model)

        # ===== Embed map feature =====
        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_position = input_dict["encoder/map_position"]
        map_heading = input_dict["encoder/map_heading"]
        map_token_valid_mask = input_dict["encoder/map_valid_mask"]
        map_token = self.map_polyline_encoder(map_feature, map_valid_mask)

        # Add:
        map_pe = input_dict["encoder/map_pe"]
        map_token += map_pe

        assert map_token.shape == (B, M, self.d_model)

        # ===== Embed traffic light =====
        traffic_light_feature = input_dict["encoder/traffic_light_feature"]
        traffic_light_position = input_dict["encoder/traffic_light_position"]
        traffic_light_heading = input_dict["encoder/traffic_light_heading"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]
        if L != 0:
            traffic_light_feature = (
                traffic_light_feature[:, :self.total_time_steps] *
                traffic_light_valid_mask[:, :self.total_time_steps, ..., None]
            )
            traffic_light_feature = collapse_time(traffic_light_feature)
            traffic_light_token = self.light_mlps(traffic_light_feature)
        else:
            traffic_light_token = traffic_light_feature.new_zeros([B, L, self.d_model])
        assert traffic_light_token.shape == (B, L, self.d_model)

        # ===== Call transformer layers =====
        x = torch.concatenate([map_token, agent_token, traffic_light_token], dim=1)
        x_pos = torch.concatenate(
            [map_position, agent_position[:, self.total_time_steps], traffic_light_position], dim=1
        )

        x_mask = torch.concatenate(
            [
                map_token_valid_mask, agent_valid_mask[:, self.total_time_steps],
                traffic_light_valid_mask[:, self.total_time_steps]
            ],
            dim=1
        )
        assert torch.all(x_mask.sum(dim=-1) > 0)

        if self.relative_pe:
            x_heading = torch.concatenate(
                [map_heading, agent_heading[:, self.total_time_steps], traffic_light_heading], dim=1
            )
            relation, rel_mask, indices = compute_relation(
                pos=x_pos,
                heading=x_heading,
                mask=x_mask,
                hidden_dim=self.d_model,
                knn=self.config.MODEL.get('KNN', 128)
            )
            pos_embedding = None
        else:
            relation = None
            pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos[..., 0:2], hidden_dim=self.d_model)

        for k in range(len(self.self_attn_layers)):
            # inp = self._add_pe(x, pos_embedding)
            x = self.self_attn_layers[k](
                tgt=x,
                pos=pos_embedding,
                tgt_key_padding_mask=~x_mask,
                relation=relation,
                relation_mask=rel_mask,
                relation_indices=indices,
            )

        # x = torch.cat([x, pos_embedding], dim=-1)
        x = self.out(x.reshape(-1, x.shape[-1])).reshape(list(x.shape[:-1]) + [self.d_model])

        if pos_embedding is not None:
            x = x + pos_embedding

        input_dict["encoder/scenario_token"] = x
        if self.relative_pe:
            input_dict["encoder/scenario_position"] = x_pos
            input_dict["encoder/scenario_heading"] = x_heading
        input_dict["encoder/scenario_valid_mask"] = x_mask

        # Add:
        # input_dict["encoder/modeled_agent_pe"] = self.agent_pe(input_dict["encoder/modeled_agent_id"])
        input_dict["encoder/map_pe"] = map_pe

        return input_dict


class MotionDecoder(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS

        # TODO: Pass through config.
        # self.num_actions = 169
        # num_pred_steps = 16 + 1  # TODO: FIXME: How to change this to support scenestreamer????

        pre_projection = self.config.MODEL.get('PRE_PROJECTION', False)

        self.relative_pe = self.config.MODEL.get('RELATIVE_PE_DECODER', False)

        dropout = self.config.MODEL.get('DROPOUT_OF_ATTN', 0.1)
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                pre_projection=pre_projection
            ),
            num_layers=num_decoder_layers,
            relative_pe=self.relative_pe,
            d_model=d_model,
            self_attention_knn=self.config.MODEL['SELF_ATTN_KNN'],
            cross_attention_knn=self.config.MODEL['CROSS_ATTN_KNN']
        )
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model, num_actions], ret_before_act=True
        )

        num_pred_steps = 100  # TODO: Is it enough? What should we do for this?
        self.step_pe = nn.Embedding(num_pred_steps, d_model)

        intra_step_tokens = 1000
        self.intra_step_pe = nn.Embedding(intra_step_tokens, d_model)

    def forward(self, input_dict, use_cache=False):
        # === Process scene embedding ===
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]
        modeled_agent_pe = input_dict["encoder/modeled_agent_pe"]
        scene_padding_mask = ~scenario_valid_mask

        # === Process action embedding ===
        input_token = input_dict["decoder/input_token"]

        step_pe = self.step_pe(input_dict["decoder/input_step"])
        input_token += step_pe

        intra_step_pe = self.intra_step_pe(input_dict["decoder/input_intra_step"])
        input_token += intra_step_pe

        input_token_valid_mask = input_dict["decoder/input_token_valid_mask"]

        # assert action_token.shape == (B, T_skipped, N, self.d_model)
        # assert modeled_agent_pe.shape == (B, N, self.d_model), modeled_agent_pe.shape
        # action_token += modeled_agent_pe[:, None]

        casual_mask = create_causal_mask(input_dict["decoder/causal_mask_offset"], num_heads=self.num_heads)

        action_padding_mask = ~input_token_valid_mask  # (T_skipped, N)
        # Flatten action token from (B, T_skipped, N, D) to (B, T_skipped*N, D)
        # action_token = action_token.flatten(1, 2)
        # Flatten action token from (B, T_skipped, N) to (B, T_skipped*N)
        # action_padding_mask = action_padding_mask.flatten(1, 2)

        # Cache from last rollout
        past_key_value = None
        if "decoder/cache" in input_dict:
            past_key_value = input_dict["decoder/cache"]

        # === Call models ===
        decoded_tokens = self.decoder(
            tgt=input_token.swapaxes(0, 1),
            tgt_mask=casual_mask,  # swapaxes(0, 1),
            tgt_key_padding_mask=action_padding_mask,
            tgt_is_causal=True,
            memory=scene_token.swapaxes(0, 1),
            memory_mask=None,  # The casual mask for memory
            memory_key_padding_mask=scene_padding_mask,
            memory_is_causal=False,
            past_key_value=past_key_value,
            use_cache=use_cache
        )

        if use_cache:
            decoded_tokens, past_key_value = decoded_tokens
            input_dict["decoder/cache"] = past_key_value

        decoded_tokens = decoded_tokens.swapaxes(0, 1)
        logits = self.prediction_head(decoded_tokens)  # TODO: We can do a masking here to reduce the computation.
        # logits = logits.reshape(B, T_skipped, N, self.num_actions)

        input_dict["decoder/output_logit"] = logits

        return input_dict


class GenModel(MotionLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.d_model = self.config.MODEL.D_MODEL

        self.scene_encoder = SceneEncoderSceneStreamer(config=self.config)

        num_actions = SceneStreamerTokenizer.get_num_actions(config)

        self.tokenizer = Tokenizer(num_actions=num_actions, d_model=self.d_model)
        self.motion_decoder = MotionDecoder(config=self.config, num_actions=num_actions)

    def encode_scene(self, input_dict):
        B, M, _, _ = input_dict["encoder/map_feature"].shape

        map_id = torch.arange(M).to(input_dict["encoder/map_feature"].device).reshape(1, M).repeat(B, 1)
        map_id.masked_fill_(input_dict["encoder/map_valid_mask"], 0)
        map_id = SceneStreamerTokenizer.get_map_id(map_id, self.config)
        map_pe = self.tokenizer(map_id)
        input_dict["encoder/map_pe"] = map_pe

        agent_id = SceneStreamerTokenizer.get_agent_id(input_dict["encoder/agent_id"], self.config, allow_invalid=False)
        agent_pe = self.tokenizer(agent_id)
        input_dict["encoder/agent_pe"] = agent_pe

        modeled_agent_id = SceneStreamerTokenizer.get_agent_id(
            input_dict["encoder/modeled_agent_id"], self.config, allow_invalid=False
        )
        modeled_agent_pe = self.tokenizer(modeled_agent_id)
        input_dict["encoder/modeled_agent_pe"] = modeled_agent_pe

        return self.scene_encoder(input_dict)

    def decode_motion(self, data_dict, use_cache=False, in_evaluation=False):
        data_dict["decoder/input_token"] = self.tokenizer(data_dict["decoder/input_token_id"], allow_invalid=True)
        data_dict = self.motion_decoder(data_dict, use_cache=use_cache)

        is_agent_tokens = GenTokenizer.is_agent_tokens(data_dict["decoder/input_token_id"], self.config)
        motion_token_valid_mask = torch.logical_and(data_dict["decoder/input_token_valid_mask"], is_agent_tokens)

        B, L, D = data_dict["decoder/output_logit"].shape
        _, T_plus_1, N = data_dict["decoder/input_action"].shape

        motion_logits = data_dict["decoder/output_logit"].new_zeros(B, T_plus_1, N, D)

        # all valid actions in (B, 17, N).
        # motion_logits[:, :-1][data_dict["decoder/input_action_valid_mask"][:, :-1]] =

        if in_evaluation:
            motion_logits[data_dict["decoder/input_action_valid_mask"]] = \
                data_dict["decoder/output_logit"][motion_token_valid_mask]
            # motion_logits[:, :-1][data_dict["decoder/input_action_valid_mask"][:, :-1]] = \
            #     data_dict["decoder/output_logit"][motion_token_valid_mask]
        else:
            motion_logits[:, :-1][data_dict["decoder/input_action_valid_mask"][:, :-1]] = \
                data_dict["decoder/output_logit"][motion_token_valid_mask]

        # TODO: FIXME: Do we want to implement the "masking out invalid actions" here?
        # Should we implement the "masking out invalid actions" in loss?
        data_dict["decoder/output_logit"] = motion_logits

        return data_dict

    def autoregressive_rollout(
        self,
        input_dict,
        num_decode_steps,
        num_prev_steps=1,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None
    ):
        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        # B, T_input, N = input_dict["decoder/input_action"].shape
        assert num_decode_steps >= 1
        # assert input_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
        # assert T_input >= num_prev_steps

        # Record "current" valid mask of input actions, we'll repeat it for each decoding step.
        # input_action_valid_mask = torch.clone(
        #     input_dict["decoder/input_action_valid_mask"][:, num_prev_steps - 1:num_prev_steps]
        # )

        # Discard future actions / mask
        # input_dict["decoder/input_action"] = input_dict["decoder/input_action"][:, :num_prev_steps]
        # input_dict["decoder/input_action_valid_mask"] = \
        #     input_dict["decoder/input_action_valid_mask"][:, :num_prev_steps]

        B, _, N = input_dict["decoder/input_action"].shape

        # Get scene embedding
        input_dict = self.encode_scene(input_dict)
        output_logit_list = []
        output_logit_masked_list = []
        output_action_list = []

        device = input_dict["decoder/input_action"].device

        # Record "current" valid mask of input actions, we'll repeat it for each decoding step.
        # input_action_valid_mask = torch.clone(
        #     input_dict["decoder/input_action_valid_mask"][:, num_prev_steps - 1:num_prev_steps]
        # )
        # # Discard future actions / mask
        # input_dict["decoder/input_action"] = input_dict["decoder/input_action"][:, :num_prev_steps]
        # input_dict["decoder/input_action_valid_mask"] = \
        #     input_dict["decoder/input_action_valid_mask"][:, :num_prev_steps]

        # === prepare those reusable tokens that will be appended at the end of the sequence at each step ===
        # [STEP_START, UPDATE_START, (AGENT_ID * N), ]
        pre_action_tokens = Tokens.create(
            ids=input_dict["decoder/input_token_id"].clone(),
            mask=input_dict["decoder/input_token_valid_mask"].clone(),
            causal_mask_offset=input_dict["decoder/causal_mask_offset"].clone(),
            length=input_dict["decoder/input_token_id"].shape[1]
        )

        # [UPDATE_END, STEP_END, ]
        update_end_tokens = Tokens.concatenate(
            [
                GenTokenizer.get_update_end_tokens(),
                GenTokenizer.get_step_end_tokens(),
            ]
        ).to_tensor(
            batch_size=B, device=device
        )

        # [UPDATE_END, STEP_END, STEP_START, UPDATE_START, (AGENT_ID * N),]
        post_action_tokens = Tokens.concatenate([update_end_tokens, pre_action_tokens])

        pre_intra_steps = input_dict["decoder/input_token_id"].shape[1]  # 0, 1, ..., 129 (130 steps)

        # intra step for new tokens
        intra_steps = torch.cat(
            [
                torch.arange(pre_intra_steps, pre_intra_steps + N + update_end_tokens.length),
                torch.arange(pre_intra_steps),
            ]
        ).to(device).reshape(1, -1).expand(B, -1)

        action_id_min, action_id_max = GenTokenizer.get_action_id_range(self.config)
        # You can't select the noop action. So we force:
        action_id_max = action_id_max - 1

        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            if not use_cache:
                raise ValueError()
                input_dict["decoder/input_step"] = input_step[:decode_step + 1]

            # Decode motion ids
            input_dict = self.decode_motion(input_dict, use_cache=use_cache, in_evaluation=True)

            output_token = input_dict["decoder/output_logit"]

            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)

            output_logit_list.append(output_token.clone())

            # mask out invalid actions
            output_token[..., :action_id_min].fill_(-1e9)
            output_token[..., action_id_max:].fill_(-1e9)
            output_logit_masked_list.append(output_token)

            # Sample the action
            if sampling_method == "argmax":
                selected_action = output_token.argmax(-1)
            elif sampling_method == "softmax":
                selected_action = torch.distributions.Categorical(logits=output_token / temperature).sample()
            elif sampling_method == "topp":
                selected_action = nucleus_sampling(logits=output_token / temperature, p=topp)
            else:
                raise ValueError("Unknown sampling method: {}".format(sampling_method))

            assert selected_action.max() < action_id_max
            assert selected_action.min() >= action_id_min

            output_action_list.append(selected_action)

            action_tokens = Tokens.create(
                ids=selected_action.reshape(B, N),
                mask=input_dict["decoder/input_action_valid_mask"].reshape(B, N),
                causal_mask_offset=selected_action.new_ones(B, N).fill_(N).int(),
                length=N
            )
            input_tokens = Tokens.concatenate([action_tokens, post_action_tokens])

            if use_cache:
                # Discard the previous ids whose key/value are cached.
                input_dict["decoder/input_token_id"] = input_tokens.ids
                input_dict["decoder/input_token_valid_mask"] = input_tokens.mask
                input_dict["decoder/input_step"] = torch.ones_like(input_tokens.ids).fill_(decode_step + 1)
                input_dict["decoder/input_intra_step"] = intra_steps
                input_dict["decoder/causal_mask_offset"] = input_tokens.causal_mask_offset

            else:
                raise ValueError()
                input_dict["decoder/input_token_id"] = torch.cat(
                    [input_dict["decoder/input_token_id"], new_tokens], dim=1
                )
                input_dict["decoder/input_action_valid_mask"] = torch.cat(
                    [input_dict["decoder/input_action_valid_mask"], step_valid_mask], dim=1
                )

            assert input_dict["decoder/input_action"].shape == input_dict["decoder/input_action_valid_mask"].shape

        output_action_list = torch.concatenate(output_action_list, dim=1)
        assert output_action_list.shape == (B, num_decode_steps, N)

        output_logit_list = torch.concatenate(output_logit_list, dim=1)
        output_logit_masked_list = torch.concatenate(output_logit_masked_list, dim=1)
        input_dict["decoder/output_logit"] = output_logit_list

        # Need to translate back to normal action range
        input_dict["decoder/output_action"] = output_action_list - action_id_min
        assert input_dict["decoder/output_action"].min() >= 0
        assert input_dict["decoder/output_action"].max() < NUM_ACTIONS + 1  # There is also a noop action.

        # TODO: Study which one is better
        # input_dict["decoder/output_score"] = calculate_trajectory_probabilities(
        #     output_logit_list, output_action_list, mask=input_dict["decoder/input_action_valid_mask"]
        # )  # (B, N)
        input_dict["decoder/output_score"] = calculate_trajectory_probabilities(
            output_logit_masked_list, output_action_list, mask=input_dict["decoder/input_action_valid_mask"]
        )  # (B, N)

        return input_dict


class SceneStreamerModel(GenModel):
    def decode_motion(self, data_dict, use_cache=False, in_evaluation=False):
        """
        Do not do any postprocessing, just through away the logits.
        """
        data_dict["decoder/input_token"] = self.tokenizer(data_dict["decoder/input_token_id"], allow_invalid=True)
        data_dict = self.motion_decoder(data_dict, use_cache=use_cache)
        return data_dict

    def autoregressive_rollout(
        self,
        input_dict,
        num_decode_steps,
        num_prev_steps=1,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None
    ):
        """
        GenModel:
            step 0 input: STEP_START, UPDATE_START, (AGENT_ID * N)
            predict 0: (ACTION_ID * N)
            step 1 input: (ACTION_ID * N), UPDATE_END, STEP_END, STEP_START, UPDATE_START, (AGENT_ID * N)
            ...

        SceneStreamerModel:
            step 0 input: STEP_START, ADD_START, (some tokens for add), ADD_END, UPDATE_START, (AGENT_ID * N)
            predict 0: (ACTION_ID * N)
            step 1 input: (ACTION_ID * N), UPDATE_END, STEP_END, STEP_START, UPDATE_START, (AGENT_ID * N)
            (that is, we just pretent the model will never remove or add new agents)
        """
        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        assert num_decode_steps >= 1

        B, _, N = input_dict["decoder/input_action"].shape

        # Get scene embedding
        input_dict = self.encode_scene(input_dict)
        output_logit_list = []
        output_logit_masked_list = []
        output_action_list = []

        device = input_dict["decoder/input_action"].device

        # === prepare those reusable tokens that will be appended at the end of the sequence at each step ===

        B = input_dict["decoder/input_token_valid_mask"].shape[0]

        # (B,)
        seq_start_indices = input_dict["decoder/input_token_valid_mask"].new_zeros(B).long()
        seq_end_indices = input_dict["decoder/input_token_valid_mask"].sum(-1)
        # (B,)
        num_motions = input_dict["eval/should_predict_motion"].sum(-1)

        from scenestreamer.utils.autoregressive_rollout import ARRollout

        map_ids = [input_dict["encoder/map_valid_mask"][i].nonzero()[:, 0] for i in range(B)]
        agent_ids = [
            input_dict["encoder/agent_id"][i][input_dict["decoder/input_action_valid_mask"][i, 0]] for i in range(B)
        ]
        rollout = ARRollout(
            init_tokens=input_dict["decoder/input_token_id"],
            init_valid_mask=input_dict["decoder/input_token_valid_mask"],
            causal_mask_offset=input_dict["decoder/causal_mask_offset"],
            config=self.config,
            map_ids=map_ids,
            agent_ids=input_dict["encoder/agent_id"],
        )

        #
        # pre_action_tokens = Tokens.create(
        #     ids=input_dict["decoder/input_token_id"].clone(),
        #     mask=input_dict["decoder/input_token_valid_mask"].clone(),
        #     causal_mask_offset=input_dict["decoder/causal_mask_offset"].clone(),
        #     length=input_dict["decoder/input_token_id"].shape[1]
        # )
        #
        # # [UPDATE_END, STEP_END, ]
        # update_end_tokens = Tokens.concatenate(
        #     [
        #         GenTokenizer.get_update_end_tokens(),
        #         GenTokenizer.get_step_end_tokens(),
        #     ]
        # ).to_tensor(batch_size=B, device=device)
        #
        # # [UPDATE_END, STEP_END, STEP_START, UPDATE_START, (AGENT_ID * N),]
        # post_action_tokens = Tokens.concatenate([update_end_tokens, pre_action_tokens])
        #
        #
        # pre_intra_steps = input_dict["decoder/input_token_id"].shape[1]  # 0, 1, ..., 129 (130 steps)
        #
        # # intra step for new tokens
        # intra_steps = torch.cat([
        #     torch.arange(pre_intra_steps, pre_intra_steps + N + update_end_tokens.length),
        #     torch.arange(pre_intra_steps),
        # ]).to(device).reshape(1, -1).expand(B, -1)
        #
        action_id_min, action_id_max = SceneStreamerTokenizer.get_action_id_range(self.config)
        # # You can't select the noop action. So we force:
        # action_id_max = action_id_max - 1
        #
        # assert len(np.unique(input_dict["scenario_id"])) == 1

        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            if not use_cache:
                raise ValueError()
                input_dict["decoder/input_step"] = input_step[:decode_step + 1]

            # input_tokens = rollout.get_tokens()

            # Decode motion ids
            input_dict = self.decode_motion(input_dict, use_cache=use_cache, in_evaluation=True)

            output_token = input_dict["decoder/output_logit"]

            # No matter what is it, just treat the output tokens as they are motion tokens.
            # output_token[..., :action_id_min].fill_(-1e9)
            # output_token[..., action_id_max:].fill_(-1e9)
            # output_logit_masked_list.append(output_token)  # TODO: Deal with scores
            #
            # # Sample the action
            # if sampling_method == "argmax":
            #     selected_action = output_token.argmax(-1)
            # elif sampling_method == "softmax":
            #     selected_action = torch.distributions.Categorical(logits=output_token / temperature).sample()
            # elif sampling_method == "topp":
            #     selected_action = nucleus_sampling(logits=output_token / temperature, p=topp)
            # else:
            #     raise ValueError("Unknown sampling method: {}".format(sampling_method))
            #
            # assert selected_action.max() < action_id_max
            # assert selected_action.min() >= action_id_min

            # Have a for loop here....
            # new_tokens = []
            # for b in range(B):
            #     motion_start = seq_end_indices[b] - num_motions[b]
            #     motion_end = seq_end_indices[b]
            #     a = selected_action[b,motion_start: motion_end]
            #     new_tokens.append(a)

            rollout.update(output_token)
            input_tokens = rollout.get_tokens()
            # print(1111)

            # output_action_list.append(selected_action)
            #
            # action_tokens = Tokens.create(
            #     ids=selected_action.reshape(B, N),
            #     mask=input_dict["decoder/input_action_valid_mask"].reshape(B, N),
            #     causal_mask_offset=selected_action.new_ones(B, N).fill_(N).int(),
            #     length=N
            # )
            # input_tokens = Tokens.concatenate([action_tokens, post_action_tokens])

            if use_cache:
                # Discard the previous ids whose key/value are cached.
                input_dict["decoder/input_token_id"] = input_tokens.ids
                input_dict["decoder/input_token_valid_mask"] = input_tokens.mask
                input_dict["decoder/input_step"] = torch.ones_like(input_tokens.ids).fill_(decode_step + 1)
                input_dict["decoder/input_intra_step"] = intra_steps
                input_dict["decoder/causal_mask_offset"] = input_tokens.causal_mask_offset

            else:
                raise ValueError()
                input_dict["decoder/input_token_id"] = torch.cat(
                    [input_dict["decoder/input_token_id"], new_tokens], dim=1
                )
                input_dict["decoder/input_action_valid_mask"] = torch.cat(
                    [input_dict["decoder/input_action_valid_mask"], step_valid_mask], dim=1
                )

            assert input_dict["decoder/input_action"].shape == input_dict["decoder/input_action_valid_mask"].shape

        output_action_list = torch.concatenate(output_action_list, dim=1)
        assert output_action_list.shape == (B, num_decode_steps, N)

        output_logit_list = torch.concatenate(output_logit_list, dim=1)
        output_logit_masked_list = torch.concatenate(output_logit_masked_list, dim=1)
        input_dict["decoder/output_logit"] = output_logit_list

        # Need to translate back to normal action range
        input_dict["decoder/output_action"] = output_action_list - action_id_min
        assert input_dict["decoder/output_action"].min() >= 0
        assert input_dict["decoder/output_action"].max() < NUM_ACTIONS + 1  # There is also a noop action.

        # TODO: Study which one is better
        # input_dict["decoder/output_score"] = calculate_trajectory_probabilities(
        #     output_logit_list, output_action_list, mask=input_dict["decoder/input_action_valid_mask"]
        # )  # (B, N)
        input_dict["decoder/output_score"] = calculate_trajectory_probabilities(
            output_logit_masked_list, output_action_list, mask=input_dict["decoder/input_action_valid_mask"]
        )  # (B, N)

        return input_dict


if __name__ == '__main__':

    import torch
    from tqdm import tqdm

    from scenestreamer.dataset.datamodule import SceneStreamerDataModule
    from scenestreamer.utils import debug_tools

    cfg_file = "cfgs/motion_debug_2_local_train.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    config.MODEL.update(dict(
        D_MODEL=512,
        NUM_ATTN_LAYERS=6,
        NUM_ATTN_HEAD=8,
        NUM_DECODER_LAYERS=6,
    ))

    config.MODEL.NAME = "gen"

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=2,
        train_num_workers=0,
        val_batch_size=2,
        val_num_workers=0,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    model = GenModel(config)
    model.eval()
    # model.cuda()

    for data_dict in tqdm(dataloader):
        # GenTokenizer.get_token_names(data_dict["decoder/input_token_id"], config)

        # data_dict = model(data_dict)

        model.autoregressive_rollout(data_dict, num_decode_steps=16, sampling_method="topp")

        # gt = data_dict["decoder/target_action"]
        # gt_mask = data_dict["decoder/target_action_valid_mask"]

        # gt = data_dict["decoder/input_token_id"][is_motion_token]
        # logits = data_dict["decoder/output_logit"][is_motion_token]

        # we need to reconstruct the "output_logit" in shape (B, T, N, D) so we can match decoder/target_action

        print(1)
