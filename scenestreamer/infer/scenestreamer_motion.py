"""
This module reimplements the autoregressive motion generation process.
"""

import copy
import dataclasses

import numpy as np
import torch
import tqdm

from scenestreamer.dataset.preprocessor import slice_trafficgen_data, NUM_TG_MULTI, TG_SKIP_STEP
from scenestreamer.models import relation
from scenestreamer.models.scenestreamer_model import get_edge_info_for_scenestreamer, get_num_tg
from scenestreamer.tokenization.motion_tokenizers import interpolate, interpolate_heading, START_ACTION as MOTION_START_ACTION
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils


@dataclasses.dataclass
class SceneStreamerTokens:
    _cache = None
    _cache_length = 0

    token: torch.Tensor
    position: torch.Tensor
    heading: torch.Tensor
    valid_mask: torch.Tensor
    width: torch.Tensor
    length: torch.Tensor
    causal_mask: torch.Tensor
    force_mask: torch.Tensor
    step: torch.Tensor
    require_relation: torch.Tensor
    N: int  # Number of motion tokens (agents)
    G: int  # Number of trafficgen tokens
    L: int  # Number of traffic light tokens
    current_step: int

    model: torch.nn.Module = None
    data_dict: dict = None

    output_token: torch.Tensor = None

    def add(self, *, token, position, heading, valid_mask, width, length, causal_mask, force_mask, current_step, require_relation):
        assert token.ndim == 3, token.shape  # B, seq_len, D
        assert position.ndim == 3  # B, seq_len, 2
        assert heading.ndim == 2  # B, seq_len
        assert valid_mask.ndim == 2
        assert width.ndim == 2
        assert length.ndim == 2
        assert causal_mask.ndim == 3, causal_mask.shape
        assert force_mask.ndim == 3, force_mask.shape
        assert current_step >= self.current_step, "current step {} < {}".format(current_step, self.current_step)
        device = self.token.device
        self.token = torch.cat([self.token, token.to(device)], dim=1)
        self.position = torch.cat([self.position, position.to(device)], dim=1)
        self.heading = torch.cat([self.heading, heading.to(device)], dim=1)
        self.valid_mask = torch.cat([self.valid_mask, valid_mask.to(device)], dim=1)
        self.width = torch.cat([self.width, width.to(device)], dim=1)
        self.length = torch.cat([self.length, length.to(device)], dim=1)
        step = torch.full((token.shape[0], token.shape[1]), current_step, dtype=torch.long, device=token.device)
        self.step = torch.cat([self.step, step.to(device)], dim=1)
        self.current_step = current_step

        num_existing_keys = self.causal_mask.shape[2]
        assert self.causal_mask.shape == (self.B, num_existing_keys, num_existing_keys), self.causal_mask.shape
        num_new_keys = causal_mask.shape[2]
        assert num_new_keys > num_existing_keys, (num_new_keys, num_existing_keys)
        new_all_causal_mask = self.causal_mask.new_zeros(self.B, num_new_keys, num_new_keys)
        new_all_causal_mask[:, :num_existing_keys, :num_existing_keys] = self.causal_mask
        new_all_causal_mask[:, num_existing_keys:, :] = causal_mask.to(device)
        self.causal_mask = new_all_causal_mask
        assert self.token.shape[-2] == num_new_keys, self.token.shape

        num_existing_keys = self.force_mask.shape[2]
        assert self.force_mask.shape == (self.B, num_existing_keys, num_existing_keys), self.force_mask.shape
        num_new_keys = force_mask.shape[2]
        assert num_new_keys > num_existing_keys, (num_new_keys, num_existing_keys)
        new_all_force_mask = self.force_mask.new_zeros(self.B, num_new_keys, num_new_keys)
        new_all_force_mask[:, :num_existing_keys, :num_existing_keys] = self.force_mask
        new_all_force_mask[:, num_existing_keys:, :] = force_mask.to(device)
        self.force_mask = new_all_force_mask
        assert self.token.shape[-2] == num_new_keys, self.token.shape

        self.require_relation = torch.cat(
            [self.require_relation, require_relation.to(device)], dim=1
        )

        # print("\t\tadd token length {}, current step {}, valid {}, pos {}".format(token.shape[1], self.current_step, valid_mask[0].tolist(), position[0].tolist()))

    @property
    def B(self):
        return self.token.shape[0]

    @property
    def seq_len(self):
        return self.token.shape[1]

    def able_to_call_model(self):
        return self.valid_mask[:, self._cache_length:].any().item()

    def call_model_with_cache(self, knn=None, max_distance=None, use_cache=True, keep_output_token=False):
        # ===== prepare dynamic relation =====
        data_dict = self.data_dict
        map_position = data_dict["model/map_token_position"]
        map_heading = data_dict["model/map_token_heading"]
        map_token_valid_mask = data_dict["model/map_token_valid_mask"]
        if knn is None:
            knn = self.model.config.SCENESTREAMER_ATTENTION_KNN
        if max_distance is None:
            max_distance = self.model.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE

        relation_all_to_all, relation_valid_mask, require_relation_pairwise = relation.compute_relation_for_scenestreamer(
            query_pos=self.position[:, self._cache_length:],
            query_heading=self.heading[:, self._cache_length:],
            query_valid_mask=self.valid_mask[:, self._cache_length:],
            query_step=self.step[:, self._cache_length:],
            key_pos=self.position,
            key_heading=self.heading,
            key_valid_mask=self.valid_mask,
            key_step=self.step,
            causal_valid_mask=self.causal_mask[:, self._cache_length:],
            force_attention_mask=self.force_mask[:, self._cache_length:],

            knn=knn,
            max_distance=max_distance,

            gather=False,
            query_width=None,
            # set query's w/l to 0 so that we get the rel of contour of key w.r.t. center of query
            query_length=None,
            key_width=None,
            key_length=None,
            non_agent_relation=True,

            require_relation=self.require_relation[:, self._cache_length:],
            require_relation_for_key=self.require_relation,
        )
        relation_all_to_all = get_edge_info_for_scenestreamer(
            q_k_relation=relation_all_to_all,
            q_k_valid_mask=relation_valid_mask,
            relation_model=self.model.relation_embed_4d,
            relation_model_1d=self.model.relation_embed_1d,
            require_relation_pairwise=require_relation_pairwise,
        )
        a2m_3d = self.model.config.MODEL.ALL_TO_MAP_3D
        assert a2m_3d is False
        relation_all_to_map, relation_map_valid_mask, require_relation_pairwise_map = relation.compute_relation_for_scenestreamer(
            query_pos=self.position[:, self._cache_length:],
            query_heading=self.heading[:, self._cache_length:],
            query_valid_mask=self.valid_mask[:, self._cache_length:],
            query_step=None if a2m_3d else self.step[:, self._cache_length:],
            query_width=None,
            query_length=None,
            key_pos=map_position,
            key_heading=map_heading,
            key_valid_mask=map_token_valid_mask,
            key_step=None if a2m_3d else torch.zeros_like(map_heading, dtype=torch.int64),
            key_width=None,
            key_length=None,
            causal_valid_mask=None,
            knn=knn,
            max_distance=max_distance,
            gather=False,
            non_agent_relation=True,
            require_relation=self.require_relation[:, self._cache_length:],
            require_relation_for_key=map_token_valid_mask,
        )
        relation_all_to_map = get_edge_info_for_scenestreamer(
            q_k_relation=relation_all_to_map,
            q_k_valid_mask=relation_map_valid_mask,
            relation_model=self.model.relation_embed_3d if a2m_3d else self.model.relation_embed_4d,
            relation_model_1d=self.model.relation_embed_1d,
            require_relation_pairwise=require_relation_pairwise_map,
        )

        # if self._cache is None:
        #     cachesize = None
        # else:
        #     cachesize = self._cache[0][0].shape if self._cache is not None else None
        # print("model call at step {}, length {}, already cache size {}, alltoken len {}".format(self.current_step, self.step[:, self._cache_length:].shape, cachesize, self.seq_len))

        input_dict = {
            "model/map_token": data_dict["model/map_token"],
            "model/all_token": self.token[:, self._cache_length:],
            "model/all_to_map_info": relation_all_to_map,
            "model/all_to_all_info": relation_all_to_all,
        }

        # ===== Call Model =====
        if use_cache:
            output_dict, cache = self.model.decoder(input_dict=input_dict, use_cache=use_cache, cache=self._cache)
            new_cache = []
            for layer in range(len(cache)):
                new_cache.append(cache[layer] + [(self.B, self.token.shape[1])])
            self._cache = new_cache
            self._cache_length = self.seq_len
        else:
            output_dict = self.model.decoder(input_dict=input_dict, use_cache=use_cache)
            self._cache = None
            self._cache_length = 0

        if keep_output_token:
            if self.output_token is None:
                self.output_token = output_dict["model/all_token"]
            else:
                self.output_token = torch.cat([self.output_token, output_dict["model/all_token"]], dim=1)
        return output_dict

# def motion_prediction_task(
#         *,
#         data_dict,
#         model,
#         autoregressive_start_step=None,
#         allow_newly_added_agent_step=None,
#         temperature=None,
#         topp=None,
#         num_decode_steps=None,
#         sampling_method=None,
#         interpolation=True,
#         remove_out_of_map_agent=False,
#         remove_static_agent=False,
#         teacher_forcing_sdc=False,
#         use_cache=True,
#         progress_bar=True,
#         keep_output_token=False,
#         teacher_forcing_dest=None,
# ):
#     if num_decode_steps is None:
#         num_decode_steps = 19
#     else:
#         print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)
#     from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator
#     g = SceneStreamerGenerator(model=model)
#     g.reset(new_data_dict=data_dict)
#     output = g.generate_scenestreamer_motion(
#         progress_bar=progress_bar,
#         num_decode_steps=num_decode_steps
#     )
#     return output

# @torch.no_grad()
def motion_prediction_task(
        *,
        data_dict,
        model,
        autoregressive_start_step=None,
        allow_newly_added_agent_step=None,
        temperature=None,
        topp=None,
        num_decode_steps=None,
        sampling_method=None,
        interpolation=True,
        remove_out_of_map_agent=False,
        remove_static_agent=False,
        teacher_forcing_sdc=False,
        use_cache=True,
        progress_bar=True,
        keep_output_token=False,
        teacher_forcing_dest=None,
):
    assert teacher_forcing_dest is not None, "Please set teacher_forcing_dest to True or False"
    # ===== Some preprocessing =====
    if topp is None:
        topp = model.config.SAMPLING.TOPP
    if temperature is None:
        temperature = model.config.SAMPLING.TEMPERATURE
    if sampling_method is None:
        sampling_method = model.config.SAMPLING.SAMPLING_METHOD
    B, T_input, N = data_dict["decoder/input_action"].shape[:3]
    # assert model.training is False, "This function is only for evaluation!"
    # data_dict = copy.deepcopy(data_dict)
    if num_decode_steps is None:
        num_decode_steps = 19
        # assert start_action_step + T_input == num_decode_steps  # Might not be True in waymo test set.
        assert num_decode_steps == 19
        assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
    else:
        print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

    # ===== Encode scenes =====
    data_dict, _ = encode_scene(data_dict=data_dict, model=model)

    # ===== Create a temporary input_dict removing the future information =====
    _, _, L = data_dict["encoder/traffic_light_state"].shape

    scenestreamer_tokens = None
    step_info_dict = {}

    knn = model.config.SCENESTREAMER_ATTENTION_KNN
    max_distance = model.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE

    valid_mask = [data_dict["decoder/input_action_valid_mask"][:, :1].clone()]
    pos = [data_dict["decoder/modeled_agent_position"][:, :1].clone()]
    head = [data_dict["decoder/modeled_agent_heading"][:, :1].clone()]
    vel = [data_dict["decoder/modeled_agent_velocity"][:, :1].clone()]
    dest = []
    dest_pos = []
    tl_state = []
    log_prob = []
    action = []

    # for decoding_step in range(num_decode_steps):
    if progress_bar:
        pbar = tqdm.trange(num_decode_steps, desc="Decoding Step")
    else:
        pbar = range(num_decode_steps)

    G = get_num_tg(N)
    all_token_casual_mask = model._build_all_tokens_mask(
        B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
    ).to(data_dict["decoder/input_action"].device)

    all_force_mask = model._build_all_force_mask(
        B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
    ).to(data_dict["decoder/input_action"].device)

    no_tg = model.no_tg

    for decoding_step in pbar:

        # # TODO: FIXME: generate_all_agents
        # if decoding_step == 0:
        #     generate_all_agents = True
        # else:
        #     generate_all_agents = False
        generate_all_agents = False

        if model.no_tg is False:
            if decoding_step % TG_SKIP_STEP == 0:
                no_tg = False
            else:
                no_tg = True

        # TODO: not hardcoded.
        if decoding_step < 2:
            teacher_forcing_motion = True
            allow_newly_added = True
        else:
            teacher_forcing_motion = False
            allow_newly_added = False

        if decoding_step <= 2:
            teacher_forcing_tg = True
            teacher_forcing_tl = True
        else:
            teacher_forcing_tg = False
            teacher_forcing_tl = False

        # ===== Traffic light =====
        # print("Step {}, Calling Traffic Light".format(decoding_step))
        scenestreamer_tokens, step_info_dict = call_model_for_traffic_light(
            model=model, data_dict=data_dict, knn=knn, max_distance=max_distance, scenestreamer_tokens=scenestreamer_tokens,
            current_step=decoding_step, step_info_dict=step_info_dict, teacher_forcing=teacher_forcing_tl,
            use_cache=use_cache, all_token_casual_mask=all_token_casual_mask, all_force_mask=all_force_mask,
            keep_output_token=keep_output_token,
        )
        if step_info_dict["traffic_light_state"].shape[1] > 0:
            tl_state.append(step_info_dict["traffic_light_state"].reshape(B, 1, L))

        # ===== Trafficgen =====
        if no_tg:
            if teacher_forcing_tg:
                current_step = decoding_step
                step_info_dict["agent_valid_mask"] = data_dict["decoder/input_action_valid_mask"][:, current_step]
                step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:, current_step]
                step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:, current_step]
                step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:, current_step]
                step_info_dict["agent_type"] = data_dict["decoder/agent_type"]
                step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"]
                step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"]

        else:

            if generate_all_agents:
                scenestreamer_tokens, step_info_dict = call_model_for_trafficgen_generate_all_agents(
                    model=model, data_dict=data_dict,  scenestreamer_tokens=scenestreamer_tokens,
                    current_step=decoding_step, step_info_dict=step_info_dict,
                    use_cache=use_cache, all_token_casual_mask=all_token_casual_mask, all_force_mask=all_force_mask,
                    keep_output_token=keep_output_token,
                )
                raise ValueError

            else:
                # print("Step {}, Calling Trafficgen".format(decoding_step))
                scenestreamer_tokens, step_info_dict = call_model_for_trafficgen(
                    model=model, data_dict=data_dict, knn=knn, max_distance=max_distance, scenestreamer_tokens=scenestreamer_tokens,
                    current_step=decoding_step, step_info_dict=step_info_dict, teacher_forcing_from_gt=teacher_forcing_tg,
                    use_cache=use_cache, all_token_casual_mask=all_token_casual_mask, teacher_forcing_dest=teacher_forcing_dest,
                    all_force_mask=all_force_mask, keep_output_token=keep_output_token,
                )

            # dest.append(step_info_dict["agent_destination"].reshape(B, 1, N))
            # dest_pos.append(step_info_dict["agent_destination_position"].reshape(B, 1, N, 2))

        # ===== Motion =====
        # print("Step {}, Calling Motion".format(decoding_step))
        scenestreamer_tokens, step_info_dict = call_model_for_motion(
            model=model, data_dict=data_dict, knn=knn, max_distance=max_distance,
            scenestreamer_tokens=scenestreamer_tokens, current_step=decoding_step,
            sampling_method=sampling_method, temperature=temperature, topp=topp,
            step_info_dict=step_info_dict, teacher_forcing=teacher_forcing_motion,
            use_cache=use_cache, allow_newly_added=allow_newly_added,
            all_token_casual_mask=all_token_casual_mask, all_force_mask=all_force_mask,
            keep_output_token=keep_output_token,
        )
        pos.append(step_info_dict["agent_position"].reshape(B, 1, N, 2))
        head.append(step_info_dict["agent_heading"].reshape(B, 1, N))
        vel.append(step_info_dict["agent_velocity"].reshape(B, 1, N, 2))
        valid_mask.append(step_info_dict["agent_valid_mask"].reshape(B, 1, N))
        log_prob.append(step_info_dict["motion_input_action_log_prob"].reshape(B, 1, N))
        action.append(step_info_dict["motion_input_action"].reshape(B, 1, N))

    assert all_token_casual_mask.shape[1] == all_token_casual_mask.shape[2] == scenestreamer_tokens.seq_len, (
        "{} vs {}".format(
            all_token_casual_mask.shape, scenestreamer_tokens.seq_len
        )
    )
    assert all_force_mask.shape[1] == all_force_mask.shape[2] == scenestreamer_tokens.seq_len, (
        "{} vs {}".format(
            all_force_mask.shape, scenestreamer_tokens.seq_len
        )
    )

    pos = torch.cat(pos, dim=1)
    head = torch.cat(head, dim=1)
    vel = torch.cat(vel, dim=1)
    action = torch.cat(action, dim=1)
    if dest:
        dest = torch.cat(dest, dim=1)
        dest_pos = torch.cat(dest_pos, dim=1)
    else:
        dest = None
        dest_pos = None

    # Evict the last step's input_action_valid_mask_list as it is not used.
    valid_mask = valid_mask[:-1]
    valid_mask = torch.cat(valid_mask, dim=1)

    tl_state = torch.cat(tl_state, dim=1)

    log_prob = torch.cat(log_prob, dim=1)

    # ===== Interpolate the output =====
    output_dict = {}

    output_dict, _ = interpolate_autoregressive_output(
        data_dict=output_dict,
        agent_heading=head,
        agent_position=pos,
        agent_velocity=vel,
        agent_destination=dest,
        agent_destination_position=dest_pos,
        input_valid_mask=valid_mask,
        num_skipped_steps=model.motion_tokenizer.num_skipped_steps,
        num_decoded_steps=num_decode_steps,
        teacher_forcing_sdc=teacher_forcing_sdc,
    )

    assert log_prob.shape == (B, 19, N)
    scores = (log_prob * valid_mask)[:, 2:].sum(1)

    # relation_valid_mask = relation.compute_relation_for_scenestreamer(
    #     query_pos=scenestreamer_tokens.position[:, :],
    #     query_heading=scenestreamer_tokens.heading[:, :],
    #     query_valid_mask=scenestreamer_tokens.valid_mask[:, :],
    #     query_step=scenestreamer_tokens.step[:, :],
    #     key_pos=scenestreamer_tokens.position,
    #     key_heading=scenestreamer_tokens.heading,
    #     key_valid_mask=scenestreamer_tokens.valid_mask,
    #     key_step=scenestreamer_tokens.step,
    #     causal_valid_mask=scenestreamer_tokens.causal_mask[:, :],
    #     force_attention_mask=scenestreamer_tokens.force_mask[:, :],
    #
    #     knn=knn,
    #     max_distance=max_distance,
    #
    #     gather=False,
    #     query_width=None,
    #     # set query's w/l to 0 so that we get the rel of contour of key w.r.t. center of query
    #     query_length=None,
    #     key_width=scenestreamer_tokens.width,
    #     key_length=scenestreamer_tokens.length,
    #     non_agent_relation=True,
    #
    #     require_relation=scenestreamer_tokens.require_relation[:, :],
    #     require_relation_for_key=scenestreamer_tokens.require_relation,
    # )[1]
    # import matplotlib.pyplot as plt
    # vis = relation_valid_mask[0].cpu().numpy()
    # plt.imshow(vis)
    #
    # data_dict = scenestreamer_tokens.data_dict
    # map_position = data_dict["model/map_token_position"]
    # map_heading = data_dict["model/map_token_heading"]
    # map_token_valid_mask = data_dict["model/map_token_valid_mask"]
    # relation_valid_mask = relation.compute_relation_for_scenestreamer(
    #     query_pos=scenestreamer_tokens.position[:, :],
    #     query_heading=scenestreamer_tokens.heading[:, :],
    #     query_valid_mask=scenestreamer_tokens.valid_mask[:, :],
    #     query_step=scenestreamer_tokens.step[:, :],
    #
    #
    #     # ===========================
    #
    #     key_pos=map_position,
    #     key_heading=map_heading,
    #     key_valid_mask=map_token_valid_mask,
    #     key_step=torch.zeros_like(map_heading, dtype=torch.int64),
    #     key_width=None,
    #     key_length=None,
    #     causal_valid_mask=None,
    #     knn=knn,
    #     max_distance=max_distance,
    #     gather=False,
    #     non_agent_relation=True,
    #     require_relation_for_key=map_token_valid_mask,
    #
    #     require_relation=scenestreamer_tokens.require_relation,
    # )[1]
    # import matplotlib.pyplot as plt
    # vis = relation_valid_mask[0].cpu().numpy()
    # plt.imshow(vis)

    output_dict.update({

        # TODO: Not accumulated across steps? now is the last.
        "decoder/current_agent_shape": step_info_dict["agent_shape"],
        "model/traffic_light_state": tl_state,

        # feed forward
        "encoder/map_feature_valid_mask": data_dict["encoder/map_feature_valid_mask"],
        "encoder/traffic_light_position": data_dict["encoder/traffic_light_position"],
        "encoder/traffic_light_valid_mask": data_dict["encoder/traffic_light_valid_mask"],
        # "decoder/labeled_agent_id"
        # "decoder/object_of_interest_id"

        "decoder/output_score": scores,

        "model/output_action": action,
    })
    if "decoder/sdc_index" in data_dict:
        output_dict["decoder/sdc_index"] = data_dict["decoder/sdc_index"]
    if "raw/map_feature" in data_dict:
        output_dict["raw/map_feature"] = data_dict["raw/map_feature"]
    if "vis/map_feature" in data_dict:
        output_dict["vis/map_feature"] = data_dict["vis/map_feature"]
    if "decoder/object_of_interest_id" in data_dict:
        output_dict["decoder/object_of_interest_id"] = data_dict["decoder/object_of_interest_id"]


    # plot_dict = utils.unbatch_data(utils.torch_to_numpy(output_dict))
    # from scenestreamer.gradio_ui.plot import plot_pred
    # plot_pred(plot_dict, show=True)

    output_dict["scenestreamer_tokens"] = scenestreamer_tokens
    return output_dict

def call_model_for_traffic_light(
        *,
        model,
        data_dict,
        step_info_dict,
        scenestreamer_tokens,
        current_step,
        knn,
        max_distance,
        teacher_forcing,
        use_cache,
        all_token_casual_mask,
        all_force_mask,
keep_output_token,
):
    tl_input_dict = {
        # no time dim:
        "encoder/traffic_light_position": data_dict["encoder/traffic_light_position"][..., :2],
        "encoder/traffic_light_heading": data_dict["encoder/traffic_light_heading"],
        "encoder/traffic_light_map_id": data_dict["encoder/traffic_light_map_id"],
    }

    if teacher_forcing:
        tl_input_dict.update({
            "encoder/traffic_light_state": data_dict["encoder/traffic_light_state"][:, current_step:current_step + 1],
            "encoder/traffic_light_valid_mask": data_dict["encoder/traffic_light_valid_mask"][:,
                                                current_step:current_step + 1],
        })
    else:
        tl_input_dict.update({
            "encoder/traffic_light_state": step_info_dict["traffic_light_state"],
            "encoder/traffic_light_valid_mask": step_info_dict["traffic_light_valid_mask"],
        })

    B, _, L = tl_input_dict["encoder/traffic_light_state"].shape

    tl_input_dict = model.prepare_traffic_light_tokens(tl_input_dict)
    tl_token = tl_input_dict["model/traffic_light_token"]
    tl_position = tl_input_dict["model/traffic_light_token_position"]
    tl_heading = tl_input_dict["model/traffic_light_token_heading"]
    tl_valid_mask = tl_input_dict["model/traffic_light_token_valid_mask"]
    assert tl_token.shape == (B, 1, L, model.d_model)
    assert tl_position.shape == (B, 1, L, 2)
    assert tl_heading.shape == (B, 1, L)
    assert tl_valid_mask.shape == (B, 1, L)
    traffic_light_width = torch.zeros_like(tl_position[..., 0])
    traffic_light_length = torch.zeros_like(tl_position[..., 0])

    # ===== causal mask =====
    N = data_dict["decoder/agent_id"].shape[1]
    if model.no_tg:
        G = 0
    else:
        G = get_num_tg(N)
    # tl_causal_mask = model._build_all_tokens_mask_for_tl(
    #     B=B, T=current_step + 1, num_tl=L, num_tg=G, num_motion=N
    # ).to(tl_position.device)

    # import matplotlib.pyplot as plt
    # vis = tl_causal_mask[0][0].cpu().numpy()
    # plt.imshow(vis)

    # tl_causal_mask is in shape (B, T, N, L+G+N).
    # the final G+N tokens are the trafficgen and motion tokens, which is in future.
    # We need to remove them.
    # tl_causal_mask = tl_causal_mask[:, -1, :, :-G - N]

    # ===== token =====
    if scenestreamer_tokens is None:
        tl_causal_mask = all_token_casual_mask[:, :L, :L]
        steps = torch.full((B, L), current_step, dtype=torch.long, device=tl_position.device)
        scenestreamer_tokens = SceneStreamerTokens(
            token=tl_token.flatten(1, 2),
            position=tl_position.flatten(1, 2),
            heading=tl_heading.flatten(1, 2),
            valid_mask=tl_valid_mask.flatten(1, 2),
            width=traffic_light_width.flatten(1, 2),
            length=traffic_light_length.flatten(1, 2),
            causal_mask=tl_causal_mask,
            force_mask=all_force_mask[:, :L, :L],
            step=steps,
            current_step=current_step,
            L=L,
            N=N,
            G=G,
            require_relation=tl_valid_mask.flatten(1, 2),

            model=model,
            data_dict=data_dict,
        )
    else:
        tl_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len+L, :scenestreamer_tokens.seq_len+L]
        scenestreamer_tokens.add(
            token=tl_token.flatten(1, 2),
            position=tl_position.flatten(1, 2),
            heading=tl_heading.flatten(1, 2),
            valid_mask=tl_valid_mask.flatten(1, 2),
            width=traffic_light_width.flatten(1, 2),
            length=traffic_light_length.flatten(1, 2),
            causal_mask=tl_causal_mask,
            current_step=current_step,
            require_relation=tl_valid_mask.flatten(1, 2),
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len+L, :scenestreamer_tokens.seq_len+L],
        )

    # import matplotlib.pyplot as plt
    # vis=all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + L, :scenestreamer_tokens.seq_len + L][0].cpu().numpy()
    # plt.imshow(vis)


    # Note that if teacher_forcing is False while there is no traffic light,
    # we will have L=1 and there will be error when calling the model in the first step.
    # Because at that time num_Q = num_K = 0.
    # This won't be a problem if we use teacher_forcing or in any future step > 0.
    if teacher_forcing:
        step_info_dict["traffic_light_state"] = data_dict["encoder/traffic_light_state"][:,
                                                current_step + 1:current_step + 2]
        step_info_dict["traffic_light_valid_mask"] = data_dict["encoder/traffic_light_valid_mask"][:,
                                                     current_step + 1:current_step + 2]

    else:
        if tl_valid_mask.any():
            output_dict = scenestreamer_tokens.call_model_with_cache(use_cache=use_cache, keep_output_token=keep_output_token)

            # ===== Post-process the data =====
            traffic_light_token = output_dict["model/all_token"][:, -L:]
            traffic_light_token = model.traffic_light_prenorm(traffic_light_token)
            traffic_light_token = model.traffic_light_head(traffic_light_token)
            # output_dict["model/traffic_light_logit"] = traffic_light_token

            tl_state, _ = sample_action(traffic_light_token, sampling_method="softmax")  # TODO: other sampling methods?
            step_info_dict["traffic_light_state"] = tl_state.reshape(B, 1, L)
        step_info_dict["traffic_light_valid_mask"] = tl_valid_mask.reshape(B, 1, L)

    return scenestreamer_tokens, step_info_dict


def call_model_for_trafficgen(
        *,
        model,
        data_dict,
        scenestreamer_tokens: SceneStreamerTokens,
        step_info_dict,
        current_step,
        knn,
        max_distance,
        teacher_forcing_from_gt,
        teacher_forcing_dest,
        use_cache,
        all_token_casual_mask,
        all_force_mask,
        keep_output_token,
):

    if teacher_forcing_from_gt:
        step_info_dict["agent_valid_mask"] = data_dict["decoder/input_action_valid_mask"][:, current_step]
        step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:, current_step]
        step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:, current_step]
        step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:, current_step]
        step_info_dict["agent_type"] = data_dict["decoder/agent_type"]
        step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"]
        step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"]

    B, N, G = scenestreamer_tokens.B, scenestreamer_tokens.N, scenestreamer_tokens.G

    # ===== call trafficgen tokenizer =====
    from scenestreamer.dataset.preprocessor import prepare_trafficgen_data_for_scenestreamer_a_step
    # assert B == 1, "B should be 1 but got " + str(B)
    device = scenestreamer_tokens.token.device
    tg_map_id_list = []
    tg_valid_list = []
    tg_feat_list = []
    tg_target_offset_list = []
    tg_pos_list = []
    tg_head_list = []
    for b in range(B):
        tg_map_id, tg_valid, tg_feat, tg_target_offset, tg_pos, tg_head = prepare_trafficgen_data_for_scenestreamer_a_step(
            pos=step_info_dict["agent_position"].reshape(B, N, 2)[b].cpu().numpy(),
            heading=step_info_dict["agent_heading"].reshape(B, N)[b].cpu().numpy(),
            vel=step_info_dict["agent_velocity"].reshape(B, N, 2)[b].cpu().numpy(),
            agent_valid_mask=step_info_dict["agent_valid_mask"].reshape(B, N)[b].cpu().numpy(),
            agent_type=step_info_dict["agent_type"].reshape(B, N)[b].cpu().numpy(),
            current_agent_shape=step_info_dict["agent_shape"].reshape(B, N, 3)[b].cpu().numpy(),
            map_pos=data_dict["model/map_token_position"][0].cpu().numpy()[..., :2],
            map_heading=data_dict["model/map_token_heading"][0].cpu().numpy(),
            map_valid_mask=data_dict["model/map_token_valid_mask"][0].cpu().numpy(),
            # start_action_id=model.trafficgen_agent_sos_id,
            # end_action_id=model.trafficgen_agent_eos_id,
            start_sequence_id=model.trafficgen_sequence_sos_id,
            end_sequence_id=model.trafficgen_sequence_eos_id,
            dest=None,
            dest_pad_id=model.trafficgen_sequence_pad_id,
            veh_id=model.veh_id,
            ped_id=model.ped_id,
            cyc_id=model.cyc_id,
            start_agent_id=model.trafficgen_agent_sos_id,
        )
        tg_map_id_list.append(tg_map_id)
        tg_valid_list.append(tg_valid)
        tg_feat_list.append(tg_feat)
        tg_target_offset_list.append(tg_target_offset)
        tg_pos_list.append(tg_pos)
        tg_head_list.append(tg_head)
    # input_action_for_trafficgen = torch.from_numpy(tg_map_id).to(device=device).reshape(B, 1, G)
    # input_action_valid_mask_for_trafficgen = torch.from_numpy(tg_valid).to(device=device).reshape(B, 1, G)
    # agent_feature_for_trafficgen = torch.from_numpy(tg_feat).to(device=device).reshape(B, 1, G, 8).float()
    # trafficgen_position = torch.from_numpy(tg_pos).to(device=device).reshape(B, 1, G, 2).float()
    # trafficgen_heading = torch.from_numpy(tg_head).to(device=device).reshape(B, 1, G).float()
    input_action_for_trafficgen = torch.from_numpy(np.stack(tg_map_id_list, axis=0)).to(device=device).reshape(B, 1, G)
    input_action_valid_mask_for_trafficgen = torch.from_numpy(np.stack(tg_valid_list, axis=0)).to(
        device=device).reshape(B, 1, G)
    agent_feature_for_trafficgen = torch.from_numpy(np.stack(tg_feat_list, axis=0)).to(device=device).reshape(B, 1, G,
                                                                                                              8).float()
    trafficgen_position = torch.from_numpy(np.stack(tg_pos_list, axis=0)).to(device=device).reshape(B, 1, G, 2).float()
    trafficgen_heading = torch.from_numpy(np.stack(tg_head_list, axis=0)).to(device=device).reshape(B, 1, G).float()

    # ===== prepare input data for trafficgen =====
    # -1, -1 -1 TYPE -1 -1, ..., -1
    G = scenestreamer_tokens.G
    agent_type = step_info_dict["agent_type"]
    agent_type_for_trafficgen = torch.full((B, N, NUM_TG_MULTI), -1, device=agent_type.device)
    agent_type_for_trafficgen[..., 2:] = agent_type[:, :, None]
    agent_type_for_trafficgen = torch.cat(
        [
            torch.full((B, 1), -1, device=agent_type.device),
            agent_type_for_trafficgen.flatten(1, 2),
            torch.full((B, 1), -1, device=agent_type.device),
        ], dim=1
    ).reshape(B, 1, G)

    # ===== build input data for tg autoregressive =====
    # tg_causal_mask = model._build_all_tokens_mask_for_tg(
    #     B=scenestreamer_tokens.B,
    #     T=current_step + 1,
    #     num_tl=scenestreamer_tokens.L,
    #     num_tg=scenestreamer_tokens.G,
    #     num_motion=scenestreamer_tokens.N
    # )
    # tg_causal_mask = tg_causal_mask[:, -1, :, :-scenestreamer_tokens.N]

    # ===== call model for tg autoregressive =====
    # First, input the sequence_sos_id.
    intra_step = 0
    tg_token = model.prepare_trafficgen_single_token(
        tg_action=torch.full((B, 1), model.trafficgen_sequence_sos_id, device=agent_type.device),
        tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
        tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
        tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
        tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
    )
    tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len+1, :scenestreamer_tokens.seq_len+1]

    scenestreamer_tokens.add(
        token=tg_token,
        position=torch.full((B, 1, 2), 0, device=agent_type.device),
        heading=torch.full((B, 1), 0, device=agent_type.device),
        valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
        width=torch.full((B, 1), 0.0, device=agent_type.device),
        length=torch.full((B, 1), 0.0, device=agent_type.device),
        causal_mask=tg_causal_mask,
        force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len+1, :scenestreamer_tokens.seq_len+1],
        current_step=current_step,
        require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
    )

    agent_destination_list = []
    agent_destination_pos_list = []
    for agent_index in range(N):
        agent_id = step_info_dict["agent_id"][:, agent_index:agent_index + 1]
        this_agent_valid_mask = step_info_dict["agent_valid_mask"][:, agent_index:agent_index + 1]

        # Step 0, agent start token.
        intra_step += 1
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=torch.full((B, 1), model.trafficgen_agent_sos_id, device=agent_type.device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=this_agent_valid_mask,
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        )

        # Step 1: input is the agent type.
        intra_step += 1
        tg_token = model.prepare_trafficgen_single_token(
            # TODO(PZH): Should change in TF.
            tg_action=agent_type[:, agent_index][:, None],
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=this_agent_valid_mask,
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        )

        # debug_dict = scenestreamer_tokens.call_model_with_cache()
        # debug_token = debug_dict["model/all_token"][:, -1:]
        # model.trafficgen_head.dest_id_head(debug_token)

        # Step 2: input is the map id.
        intra_step += 1
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=trafficgen_position[:, 0, intra_step:intra_step + 1],
            heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
            valid_mask=this_agent_valid_mask,
            # TODO: hardcoded 5, 6
            width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
            length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=this_agent_valid_mask,
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        )

        # Step 3: input is the agent feat.
        intra_step += 1
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=trafficgen_position[:, 0, intra_step:intra_step + 1],
            heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
            valid_mask=this_agent_valid_mask,
            # TODO: hardcoded 5, 6
            width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
            length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=this_agent_valid_mask,
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        )

        # if this_agent_valid_mask.any():
        #     output_dict = scenestreamer_tokens.call_model_with_cache(use_cache=use_cache, keep_output_token=keep_output_token)
        #     output_token = output_dict["model/all_token"][:, -1:]
        #     # call pred head to get agent feat.
        #
        #     output_token = model.trafficgen_prenorm(output_token)
        #
        #     dest_id_logit = model.trafficgen_head.dest_id_head(output_token)
        #     # sample from the logits to get dest_id.  TODO: other sampling methods?
        #     # tiny masked out here
        #     M = data_dict["model/map_token_valid_mask"].shape[1]
        #     assert dest_id_logit.shape[1] == 1
        #
        #     dest_id_logit_mask = torch.full((B, 1, dest_id_logit.shape[-1]), False, device=agent_type.device, dtype=torch.bool)
        #     dest_id_logit_mask[:, :, :M] = data_dict["model/map_token_valid_mask"][:, None]
        #
        #     dest_pos_full = data_dict["model/map_token_position"][..., :2]  # (B, M, 2)
        #     agent_pos = step_info_dict["agent_position"][:, agent_index][:, None]  # (B, 2)
        #     dest_agent_dist = torch.cdist(dest_pos_full, agent_pos)[..., 0]  # (B, M)
        #
        #     speed = step_info_dict["agent_velocity"][:, agent_index].norm(dim=-1) # (B,)
        #     displacement = speed * 3
        #     tolerance = displacement + 20
        #     # print("Agent {} speed: {}, displacement: {}, tolerance: {}".format(
        #     #     agent_index, speed[0].item(), displacement[0].item(), tolerance[0].item()
        #     # ))
        #     assert dest_agent_dist.ndim == 2
        #     assert tolerance.ndim == 1
        #     assert dest_id_logit_mask.ndim == 3
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (dest_agent_dist < tolerance[:, None])[:, None]
        #
        #     agent_heading = step_info_dict["agent_heading"][:, agent_index]
        #
        #     # Only allow dest in front of the agent.
        #     rel_pos = (dest_pos_full - agent_pos)
        #     rel_pos = utils.rotate(x=rel_pos[..., 0], y=rel_pos[..., 1], angle=-agent_heading[:, None].expand(B, M))
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (rel_pos[..., 0] > 0)[:, None]
        #
        #     dest_heading_full = data_dict["model/map_token_heading"]  # (B, M)
        #     dest_agent_heading_dist = torch.abs(dest_heading_full - agent_heading[:, None])  # (B, M)
        #     dest_agent_heading_dist = utils.wrap_to_pi(dest_agent_heading_dist)
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (dest_agent_heading_dist < np.pi/2)[:, None]
        #
        #     dest_id_logit_mask[..., model.trafficgen_sequence_pad_id] = True
        #
        #     only_lane = True
        #     if only_lane:
        #         map_feature = data_dict["encoder/map_feature"]
        #         dest_id_logit_mask[:, :, :M] = (map_feature[:, :, 0, 13] == 1)[:, None] & dest_id_logit_mask[:, :, :M]
        #
        #     dest_id_logit[~dest_id_logit_mask] = float("-inf")
        #     # dest_id, _ = sample_action(dest_id_logit, sampling_method="softmax")
        #     dest_id, _ = sample_action(dest_id_logit, sampling_method="topp", topp=0.9)
        #
        #     if teacher_forcing_dest:
        #         gt_dest = data_dict["decoder/dest_map_index"][:, current_step, agent_index].clone()
        #         gt_dest[gt_dest == -1] = model.trafficgen_sequence_pad_id
        #         dest_id = gt_dest.reshape(B, 1)
        #
        #     dest_id_pad_mask = dest_id == model.trafficgen_sequence_pad_id
        #
        #     dest_id[dest_id_pad_mask] = 0
        #
        #     dest_position = torch.gather(
        #         data_dict["model/map_token_position"][..., :2],
        #         index=dest_id.reshape(B, 1, 1).expand(B, 1, 2),
        #         dim=1
        #     )
        #     dest_position[dest_id_pad_mask] = step_info_dict["agent_position"][:, agent_index][:, None][dest_id_pad_mask]
        #
        #     dest_heading = torch.gather(
        #         data_dict["model/map_token_heading"],
        #         index=dest_id.reshape(B, 1),
        #         dim=1
        #     )
        #     dest_heading[dest_id_pad_mask] = step_info_dict["agent_heading"][:, agent_index][:, None][dest_id_pad_mask]
        #
        #     dest_id[dest_id_pad_mask] = model.trafficgen_sequence_pad_id
        #
        #     # TODO: DEBUG
        #     # dest_dist = (step_info_dict["agent_position"][:, agent_index][0] - dest_position[0, 0]).norm(dim=-1)
        #     # print("agent {} dest id: {}, dest position: {}, dest heading: {}, dest dist: {}".format(
        #     #     agent_index, dest_id[0].item(), dest_position[0, 0].tolist(), dest_heading[0, 0].item(), dest_dist.item()
        #     # ))
        # else:
        #     dest_id = torch.full((B, 1), model.trafficgen_sequence_pad_id, device=agent_type.device)
        #     dest_position = torch.full((B, 1, 2), 0.0, device=agent_type.device)
        #     dest_heading = torch.full((B, 1), 0.0, device=agent_type.device)
        #
        # # print("Per agent index{} id{}, dest id: {}".format(agent_index, agent_id[0].item(), dest_id.tolist()))
        # dest_id[~this_agent_valid_mask] = -1
        #
        # agent_destination_list.append(dest_id)
        # agent_destination_pos_list.append(dest_position)

        # Step 4: prepare dest ID.
        # intra_step += 1
        # tg_token = model.prepare_trafficgen_single_token(
        #     tg_action=dest_id.reshape(B, 1),
        #     tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
        #     tg_agent_id=agent_id,
        #     tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
        #     tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
        # )
        # tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
        #                  :scenestreamer_tokens.seq_len + 1]
        # scenestreamer_tokens.add(
        #     token=tg_token,
        #     position=dest_position,
        #     heading=dest_heading,
        #     valid_mask=this_agent_valid_mask,
        #     width=torch.full((B, 1), 0.0, device=agent_type.device),
        #     length=torch.full((B, 1), 0.0, device=agent_type.device),
        #     causal_mask=tg_causal_mask,
        #     current_step=current_step,
        #     require_relation=this_agent_valid_mask,
        #     force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        # )

    # Finally, input the sequence_eos_id.
    intra_step += 1
    assert intra_step == G - 1, (intra_step, G, G - 1)
    tg_token = model.prepare_trafficgen_single_token(
        tg_action=torch.full((B, 1), model.trafficgen_sequence_eos_id, device=agent_type.device),
        tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
        tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
        tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
        tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
    )
    tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                     :scenestreamer_tokens.seq_len + 1]
    scenestreamer_tokens.add(
        token=tg_token,
        position=torch.full((B, 1, 2), 0, device=agent_type.device),
        heading=torch.full((B, 1), 0, device=agent_type.device),
        valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
        width=torch.full((B, 1), 0.0, device=agent_type.device),
        length=torch.full((B, 1), 0.0, device=agent_type.device),
        causal_mask=tg_causal_mask,
        current_step=current_step,
        require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
        force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
    )

    # The only thing need to be updated by non-teacher_forcing TG is the destination:
    # step_info_dict["agent_destination"] = torch.stack(agent_destination_list, dim=1)
    # step_info_dict["agent_destination_position"] = torch.stack(agent_destination_pos_list, dim=1)

    return scenestreamer_tokens, step_info_dict


def call_model_for_trafficgen_generate_all_agents(
        *,
        model,
        data_dict,
        scenestreamer_tokens: SceneStreamerTokens,
        step_info_dict,
        current_step,
        use_cache,
        all_token_casual_mask,
):
    raise ValueError



def call_model_for_motion(
        *,
        model,
        data_dict,
        scenestreamer_tokens,
        step_info_dict,
        current_step,
        knn,
        max_distance,
        sampling_method,
        temperature,
        topp,
        teacher_forcing,
        allow_newly_added,
        use_cache,
        all_token_casual_mask,
        all_force_mask,
        keep_output_token,
):
    B, N = scenestreamer_tokens.B, scenestreamer_tokens.N

    agent_delta = utils.get_relative_velocity(
        vel=step_info_dict["agent_velocity"].reshape(B, 1, N, 2),
        heading=step_info_dict["agent_heading"].reshape(B, 1, N)
    )
    motion_input_dict = {
        "decoder/input_action_valid_mask": step_info_dict["agent_valid_mask"].reshape(B, 1, N),
        "decoder/modeled_agent_position": step_info_dict["agent_position"].reshape(B, 1, N, 2),
        "decoder/modeled_agent_heading": step_info_dict["agent_heading"].reshape(B, 1, N),
        "decoder/modeled_agent_delta": agent_delta,
        "decoder/current_agent_shape": step_info_dict["agent_shape"].reshape(B, N, 3),
        "decoder/agent_type": step_info_dict["agent_type"].reshape(B, N),

        "encoder/modeled_agent_id": step_info_dict["agent_id"].reshape(B, N),
    }
    if teacher_forcing:
        motion_input_dict["decoder/input_action"] = data_dict["decoder/input_action"][:, current_step:current_step + 1]
    else:
        motion_input_dict["decoder/input_action"] = step_info_dict["motion_input_action"].reshape(B, 1, N)

    motion_input_dict = model.prepare_motion_tokens(motion_input_dict)
    motion_tokens = motion_input_dict["model/motion_token"]
    motion_position = motion_input_dict["model/motion_token_position"]
    motion_heading = motion_input_dict["model/motion_token_heading"]
    motion_valid_mask = motion_input_dict["model/motion_token_valid_mask"]
    motion_width = motion_input_dict["model/motion_token_width"]
    motion_length = motion_input_dict["model/motion_token_length"]
    B, _, N, _ = motion_tokens.shape

    # ===== causal mask =====
    # causal_mask = model._build_all_tokens_mask_for_motion(
    #     B=scenestreamer_tokens.B,
    #     T=current_step + 1,
    #     num_tl=scenestreamer_tokens.L,
    #     num_tg=scenestreamer_tokens.G,
    #     num_motion=scenestreamer_tokens.N
    # )
    # causal_mask = causal_mask[:, -1]

    scenestreamer_tokens.add(
        token=motion_tokens.flatten(1, 2),
        position=motion_position.flatten(1, 2),
        heading=motion_heading.flatten(1, 2),
        valid_mask=motion_valid_mask.flatten(1, 2),
        width=motion_width.flatten(1, 2),
        length=motion_length.flatten(1, 2),
        causal_mask=all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + N, :scenestreamer_tokens.seq_len + N],
        current_step=current_step,
        force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + N, :scenestreamer_tokens.seq_len + N],
        require_relation=motion_valid_mask.flatten(1, 2),
    )

    # print("Step {}: motion position: {}, heading: {}, valid_mask: {}".format(
    #     current_step,
    #     motion_position.flatten(1, 2)[0, 0].tolist(),
    #     motion_heading.flatten(1, 2)[0, 0].tolist(),
    #     motion_valid_mask.flatten(1, 2)[0, 0].tolist()
    # ))

    # debug code: save causal mask to files
    # import matplotlib.pyplot as plt
    # vis = scenestreamer_tokens.causal_mask[0].cpu().numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(vis)
    # plt.savefig("causal_mask_{}.png".format(current_step))

    # ===== prepare dynamic relation =====
    output_dict = scenestreamer_tokens.call_model_with_cache(use_cache=use_cache, keep_output_token=keep_output_token)
    all_token = output_dict["model/all_token"]
    motion_token = all_token[:, -scenestreamer_tokens.N:]
    # TODO: dest is not conditioning on anyone.
    if model.motion_prenorm is not None:
        motion_token = model.motion_prenorm(motion_token)
    output_token = model.motion_head(motion_token)

    # ===== Post-process the data =====
    selected_action, sampling_info = sample_action(
        logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
    )

    agent_valid_mask = step_info_dict["agent_valid_mask"]
    agent_position = step_info_dict["agent_position"]
    agent_heading = step_info_dict["agent_heading"]
    agent_velocity = step_info_dict["agent_velocity"]
    agent_type = step_info_dict["agent_type"]

    # Remove invalid actions
    # assert selected_action.shape == input_action.shape
    # correct_selected_action = torch.where(input_action_valid_mask, selected_action, -1)
    selected_action = torch.where(agent_valid_mask, selected_action, -1)

    # TODO: Teacher forcing a subset of agents here
    # if teacher_forcing_valid_mask is not None:
    #         assert teacher_forcing_action.shape == selected_action.shape
    #         selected_action = torch.where(teacher_forcing_valid_mask, teacher_forcing_action, selected_action)
    #         # correct_selected_action = torch.where(teacher_forcing_valid_mask, teacher_forcing_action, correct_selected_action)
    #         output_token[teacher_forcing_valid_mask] = 0

    # tokenizer = model.tokenizer
    res = model.motion_tokenizer.detokenize_step(
        current_pos=agent_position.reshape(B, 1, N, 2),
        current_heading=agent_heading.reshape(B, 1, N),
        current_valid_mask=agent_valid_mask.reshape(B, 1, N),
        current_vel=agent_velocity.reshape(B, 1, N, 2),
        action=selected_action.reshape(B, 1, N),
        # agent_type=agent_type.reshape(B, 1, N),
    )

    # B, _, N = input_action.shape[:3]
    new_agent_position = res["pos"].reshape(B, N, 2)
    new_agent_heading = res["heading"].reshape(B, N)
    new_agent_velocity = res["vel"].reshape(B, N, 2)

    step_info_dict["agent_position"] = new_agent_position
    step_info_dict["agent_heading"] = new_agent_heading
    step_info_dict["agent_velocity"] = new_agent_velocity
    step_info_dict["motion_input_action"] = selected_action.reshape(B, N)

    if allow_newly_added:
        new_agent_valid_mask = (
                data_dict["decoder/input_action_valid_mask"][:, current_step + 1] & (~step_info_dict["agent_valid_mask"])
        )

        if new_agent_valid_mask.any():
            new_agent_pos = data_dict["decoder/modeled_agent_position"][:, current_step + 1]
            new_agent_heading = data_dict["decoder/modeled_agent_heading"][:, current_step + 1]
            new_agent_velocity = data_dict["decoder/modeled_agent_velocity"][:, current_step + 1]
            new_action = data_dict["decoder/input_action"][:, current_step + 1]

            B, N = new_agent_valid_mask.shape
            assert new_agent_pos.shape == (B, N, 2)
            assert new_agent_heading.shape == (B, N)
            assert new_agent_velocity.shape == (B, N, 2)

            current_pos = step_info_dict["agent_position"]
            current_heading = step_info_dict["agent_heading"]
            current_vel = step_info_dict["agent_velocity"]
            current_valid_mask = step_info_dict["agent_valid_mask"]

            mask_2d = new_agent_valid_mask[..., None].expand_as(new_agent_pos)
            current_pos = torch.where(mask_2d, new_agent_pos, current_pos)
            current_heading = torch.where(new_agent_valid_mask, new_agent_heading, current_heading)
            current_vel = torch.where(mask_2d, new_agent_velocity, current_vel)
            current_valid_mask = torch.where(new_agent_valid_mask, new_agent_valid_mask, current_valid_mask)

            step_info_dict["agent_position"] = current_pos
            step_info_dict["agent_heading"] = current_heading
            step_info_dict["agent_velocity"] = current_vel
            step_info_dict["agent_valid_mask"] = current_valid_mask
            step_info_dict["motion_input_action"] = torch.where(new_agent_valid_mask, new_action, step_info_dict["motion_input_action"])

    # TODO: evict agents that moving out of the map (useful in SceneStreamer)
    # next_step_data_dict, info_dict = evict_agents(
    #     data_dict=data_dict,
    #     step_data_dict=next_step_data_dict,
    #     step_info_dict=info_dict,
    #     remove_static_agent=remove_static_agent,
    #     remove_out_of_map_agent=remove_out_of_map_agent
    # )

    tmp_action = step_info_dict["motion_input_action"].clone()
    tmp_valid_mask = agent_valid_mask.clone()
    tmp_valid_mask[tmp_action == -1] = False
    tmp_valid_mask[tmp_action == MOTION_START_ACTION] = False
    tmp_action[tmp_action == -1] = 0
    tmp_action[tmp_action == MOTION_START_ACTION] = 0
    log_prob = sampling_info["dist"].log_prob(tmp_action)
    step_info_dict["motion_input_action_log_prob"] = log_prob * tmp_valid_mask

    # print("Step {}, sdc position: {}".format(current_step, step_info_dict["agent_position"][0, 0].tolist()))

    return scenestreamer_tokens, step_info_dict


def encode_scene(*, data_dict, model):
    if "model/map_token" not in data_dict:
        data_dict = model.prepare_map_tokens(data_dict)
    return data_dict, {}


def sample_action(logits, sampling_method, temperature=1.0, topp=None):
    # Sample the action
    info = {}
    if sampling_method == "argmax":
        selected_action = logits.argmax(-1)
    elif sampling_method == "softmax":
        dist = torch.distributions.Categorical(logits=logits / temperature)
        selected_action = dist.sample()
        info["dist"] = dist
    elif sampling_method == "topp":
        selected_action, info = nucleus_sampling(logits=logits / temperature, p=topp)
    elif sampling_method == "topk":
        candidates = logits.topk(5, dim=-1).indices
        selected_action = torch.gather(
            candidates, index=torch.randint(0, 5, size=candidates.shape[:-1])[..., None].to(candidates), dim=-1
        ).squeeze(-1)
    else:
        raise ValueError("Unknown sampling method: {}".format(sampling_method))
    return selected_action, info


def nucleus_sampling(logits, p=None, epsilon=1e-8):
    p = p or 0.9

    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits).fill_(-1e9), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits).fill_(-1e9), logits)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Sort the probabilities to identify the top-p cutoff
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold p
    cutoff_index = cumulative_probs > p
    # Shift the mask to the right to keep the first token above the threshold
    cutoff_index[..., 1:] = cutoff_index[..., :-1].clone()
    cutoff_index[..., 0] = False

    # Zero out the probabilities for tokens not in the top-p set
    sorted_probs.masked_fill_(cutoff_index, 0)

    # Recover the original order of the probabilities
    original_probs = torch.zeros_like(probs)
    original_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    dist = torch.distributions.Categorical(probs=original_probs)
    sampled_token_index = dist.sample()
    return sampled_token_index, {"cutoff_index": cutoff_index, "dist": dist}


def add_new_agent(
        *, step_data_dict, step_info, new_agent_valid_mask, new_agent_pos, new_agent_heading, new_agent_velocity,
        new_agent_delta, new_action
):
    if new_agent_valid_mask is None or not new_agent_valid_mask.any():
        return step_data_dict, step_info

    B, T, N = new_agent_valid_mask.shape
    assert new_agent_pos.shape == (B, T, N, 2)
    assert new_agent_heading.shape == (B, T, N)
    assert new_agent_velocity.shape == (B, T, N, 2)
    assert new_agent_delta.shape == (B, T, N, 2)

    current_pos = step_data_dict["agent_position"]
    current_heading = step_data_dict["agent_heading"]
    current_vel = step_data_dict["agent_velocity"]
    current_valid_mask = step_data_dict["agent_valid_mask"]
    current_delta = step_data_dict["agent_delta"]

    mask_2d = new_agent_valid_mask[..., None].expand_as(new_agent_pos)
    current_pos = torch.where(mask_2d, new_agent_pos, current_pos)
    current_heading = torch.where(new_agent_valid_mask, new_agent_heading, current_heading)
    current_vel = torch.where(mask_2d, new_agent_velocity, current_vel)
    current_valid_mask = torch.where(new_agent_valid_mask, new_agent_valid_mask, current_valid_mask)
    current_delta = torch.where(mask_2d, new_agent_delta, current_delta)

    step_data_dict["agent_position"] = current_pos
    step_data_dict["agent_heading"] = current_heading
    step_data_dict["agent_velocity"] = current_vel
    step_data_dict["agent_valid_mask"] = current_valid_mask
    step_data_dict["agent_delta"] = current_delta

    if new_action.ndim == 4:
        # Variable length action
        new_action, old_action = pad_sequences(new_action, step_data_dict["input_action"], x_value=-1, y_value=-1)
        step_data_dict["input_action"] = torch.where(new_agent_valid_mask[..., None], new_action, old_action)
    elif new_action.ndim == 3:
        step_data_dict["input_action"] = torch.where(new_agent_valid_mask, new_action, step_data_dict["input_action"])
    else:
        raise ValueError("Invalid new_action shape: {}".format(new_action.shape))
    step_data_dict["input_action_valid_mask"] = current_valid_mask

    output_token = step_info["output_token"]
    if output_token is not None:
        output_token = torch.where(
            new_agent_valid_mask[..., None].expand_as(output_token), torch.zeros_like(output_token), output_token
        )
        step_info["output_token"] = output_token

    assert_motion_step_data_dict(step_data_dict=step_data_dict, step_info=step_info)

    return step_data_dict, step_info


def interpolate_autoregressive_output(
        *, data_dict, num_skipped_steps, num_decoded_steps, agent_position, agent_heading, agent_velocity,
        input_valid_mask, agent_destination, agent_destination_position, teacher_forcing_sdc, agent_shape=None,
        sdc_index=None
):
    B, _, N, _ = agent_position.shape
    T_generated_chunks = num_decoded_steps
    reconstructed_pos = interpolate(agent_position, num_skipped_steps, remove_first_step=False)
    assert (reconstructed_pos[:, ::5] == agent_position).all()
    reconstructed_heading = interpolate_heading(agent_heading, num_skipped_steps, remove_first_step=False)
    reconstructed_vel = interpolate(agent_velocity, num_skipped_steps, remove_first_step=False)

    assert input_valid_mask.shape[1] == T_generated_chunks + 1

    valid = input_valid_mask[:, :-1].reshape(B, -1, 1, N).expand(-1, -1, num_skipped_steps, -1).reshape(B, -1, N)
    valid = torch.cat([valid, input_valid_mask[:, -1:]], dim=1)

    if teacher_forcing_sdc:
        sdc_index = sdc_index[0].item()
        assert sdc_index == 0
        valid[:, 91:, sdc_index] = False

    reconstructed_valid_mask = valid

    if agent_destination is not None:
        step = TG_SKIP_STEP * 5
        agent_destination = agent_destination[:, :, None].repeat((1, 1, step, 1)).flatten(1, 2)[:, :96]
        agent_destination_position = agent_destination_position[:, :, None, :].repeat((1, 1, step, 1, 1)).flatten(1, 2)[:, :96]

    # Mask out:
    reconstructed_pos = reconstructed_pos * reconstructed_valid_mask.unsqueeze(-1)
    reconstructed_vel = reconstructed_vel * reconstructed_valid_mask.unsqueeze(-1)
    reconstructed_heading = reconstructed_heading * reconstructed_valid_mask

    # We ensure that the output must be 5*T_chunks+1
    assert reconstructed_pos.shape[1] == num_skipped_steps * T_generated_chunks + 1
    assert reconstructed_valid_mask.shape[1] == num_skipped_steps * T_generated_chunks + 1
    assert reconstructed_vel.shape[1] == num_skipped_steps * T_generated_chunks + 1
    assert reconstructed_heading.shape[1] == num_skipped_steps * T_generated_chunks + 1

    data_dict["decoder/reconstructed_position"] = reconstructed_pos
    data_dict["decoder/reconstructed_heading"] = reconstructed_heading
    data_dict["decoder/reconstructed_velocity"] = reconstructed_vel
    data_dict["decoder/reconstructed_valid_mask"] = reconstructed_valid_mask
    data_dict["decoder/reconstructed_agent_destination"] = agent_destination
    data_dict["decoder/reconstructed_agent_destination_position"] = agent_destination_position
    if agent_shape is not None:
        data_dict["decoder/reconstructed_shape"] = \
            torch.stack(agent_shape, dim=1).expand(-1, reconstructed_vel.shape[1], -1, -1)

    return data_dict, {}


def evict_agents(
        *,
        data_dict,
        step_data_dict,
        step_info_dict,
        max_distance=10,
        remove_static_agent=False,
        remove_out_of_map_agent=False
):
    # Get scene token:
    # in_evaluation = input_dict["in_evaluation"][0].item()
    # scene_token = input_dict["encoder/scenario_token"]
    # B, M, _ = input_dict["encoder/map_position"].shape
    # action = action.clone()

    should_evict = None

    if remove_out_of_map_agent:
        map_position = data_dict["encoder/map_position"][..., :2]
        agent_position = step_data_dict["agent_position"]
        assert agent_position.ndim == 4
        agent_position = agent_position[:, 0]

        dist = torch.cdist(agent_position, map_position)
        min_dist = dist.min(dim=-1).values

        should_evict = min_dist > max_distance

    if remove_static_agent:
        agent_speed = step_data_dict["agent_velocity"].norm(dim=-1)[:, 0]
        static_agent = agent_speed < 0.5
        if should_evict is None:
            should_evict = static_agent
        else:
            should_evict = torch.logical_or(should_evict, static_agent)

    if should_evict is None or should_evict.sum().item() == 0:
        step_info_dict["evicted_agents"] = 0
        step_info_dict["evicted_agent_mask"] = None
        return step_data_dict, step_info_dict

    num_evicted = should_evict.sum().item()

    # We should inform the autoregressive process not to generate action in next step.
    # However, current's step's action is still valid (because the input_action_valid_mask for this particular agent
    # is valid), hence the outer process is still waiting for the new states of the agents.
    # Therefore, we shouldn't mask out these information.
    new_mask = step_data_dict["input_action_valid_mask"] & (~should_evict)
    step_data_dict["input_action_valid_mask"] = new_mask

    # step_data_dict["input_action"] = torch.where(new_mask, step_data_dict["input_action"], -1)
    # step_data_dict["agent_position"] = torch.where(new_mask.unsqueeze(-1), agent_position, 0)
    # step_data_dict["agent_heading"] = torch.where(new_mask, step_data_dict["agent_heading"], 0)
    # step_data_dict["agent_velocity"] = torch.where(new_mask.unsqueeze(-1), step_data_dict["agent_velocity"], 0)
    step_data_dict["agent_valid_mask"] = new_mask
    # step_data_dict["agent_delta"] = torch.where(new_mask.unsqueeze(-1), step_data_dict["agent_delta"], 0)
    # step_info_dict["output_token"] = torch.where(new_mask.unsqueeze(-1), step_info_dict["output_token"], 0)

    step_info_dict["evicted_agents"] = num_evicted
    step_info_dict["evicted_agent_mask"] = should_evict
    assert_motion_step_data_dict(step_data_dict, step_info_dict)

    return step_data_dict, step_info_dict


def assert_motion_step_data_dict(*, step_data_dict, step_info):
    assert "input_step" in step_data_dict
    assert "input_action" in step_data_dict
    assert "input_action_valid_mask" in step_data_dict
    assert "agent_position" in step_data_dict
    assert "agent_heading" in step_data_dict
    assert "agent_velocity" in step_data_dict
    assert "agent_valid_mask" in step_data_dict
    assert "agent_delta" in step_data_dict
    assert "agent_id" in step_data_dict
    assert "agent_type" in step_data_dict
    assert "agent_shape" in step_data_dict

    m = step_data_dict["input_action_valid_mask"]
    assert (step_data_dict["input_action"][~m] == -1).all()
    assert (m == step_data_dict["agent_valid_mask"]).all()
    if step_info["output_token"] is not None:
        assert (step_info["output_token"][~m] == 0).all()


def pad_sequences(x, y, x_value=0, y_value=0):
    max_seq_len = max(x.shape[-1], y.shape[-1])
    x = torch.nn.functional.pad(x, (0, max_seq_len - x.shape[-1]), value=x_value)
    y = torch.nn.functional.pad(y, (0, max_seq_len - y.shape[-1]), value=y_value)
    return x, y

def test_moving_dist(data_dict):
    def _get_first_last_pos(pos, valid_mask):
        T, N = valid_mask.shape
        ind = np.arange(T).reshape(-1, 1).repeat(N, axis=1)  # T, N
        ind[~valid_mask] = 0
        ind = ind.max(axis=0)
        last = np.take_along_axis(pos, indices=ind.reshape(1, N, 1), axis=0)
        last = np.squeeze(last, axis=0)

        # Find the index of the first True (or 1) along axis 0 (time) for each agent
        # First, create a mask of where any True exists per column
        has_valid = valid_mask.any(axis=0)

        # Use argmax along time axis: this returns first occurrence of maximum (i.e. True)
        first_idx = valid_mask.argmax(axis=0)

        # Set result to -1 where there was no valid entry
        first_idx[~has_valid] = -1

        first = np.take_along_axis(pos, indices=first_idx.reshape(1, N, 1), axis=0)
        first = np.squeeze(first, axis=0)
        return first, last
    agent_valid_mask = data_dict["decoder/agent_valid_mask"]
    agent_position = data_dict["decoder/agent_position"]
    first_pos, last_pos = _get_first_last_pos(agent_position, agent_valid_mask)
    moving_dist = np.linalg.norm((last_pos-first_pos)[:, :2], axis=-1)
    return moving_dist


def animate_scenestreamer(
    save_path, data_dict, fps=10, dpi=300, draw_traffic=True
):
    from scenestreamer.gradio_ui.plot import FFMpegWriter, _plot_map, _plot_traffic_light
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    from matplotlib import transforms

    agent_pos=data_dict["decoder/reconstructed_position"]
    agent_mask=data_dict["decoder/reconstructed_valid_mask"]
    agent_heading=data_dict["decoder/reconstructed_heading"]
    agent_shape=data_dict["decoder/current_agent_shape"]

    agent_dest_pos = data_dict["decoder/reconstructed_agent_destination_position"]
    agent_dest = data_dict["decoder/reconstructed_agent_destination"]

    # all_agent_pos = data_dict["decoder/agent_position"][:91, :, :2]
    # all_agent_heading = data_dict["decoder/agent_heading"]
    # all_agent_shape = data_dict["decoder/agent_shape"][10]
    if "decoder/labeled_agent_id" in data_dict:
        ooi = data_dict["decoder/labeled_agent_id"]
    else:
        ooi = []

    if 'decoder/sdc_index' in data_dict:
        ego_agent_id = int(data_dict['decoder/sdc_index'])
    else:
        ego_agent_id = 0

    assert agent_pos.ndim == 3
    T = agent_pos.shape[0]  # Number of timesteps
    N = agent_pos.shape[1]  # Number of agents

    cmap = sns.color_palette("colorblind", n_colors=N)  # Color for each agent

    all_agent_positions = agent_pos[:, :, ...].reshape(-1, 2)
    xmin, ymin = all_agent_positions.min(axis=0)
    xmax, ymax = all_agent_positions.max(axis=0)
    xlim, ylim = (xmin - 10, xmax + 10), (ymin - 10, ymax + 10)  # Adjust `BOUNDARY` as needed

    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-preset', 'ultrafast', '-crf', '23', '-threads', '4'])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.set_aspect(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    _plot_map(data_dict, ax, dont_draw_lane=True)
    _plot_traffic_light(data_dict, ax)

    agent_patches = []
    agent_texts = []
    agent_arrows = []
    agent_stars = []

    for agent_ind in range(N):
        if not draw_traffic and agent_ind not in ooi:
            agent_patches.append(None)
            agent_texts.append(None)
            agent_arrows.append(None)
            agent_stars.append(None)
            continue
        face_color = cmap[0] if agent_ind == ego_agent_id else cmap[agent_ind]
        label = "{}-SDC".format(ego_agent_id) if agent_ind == ego_agent_id else \
            "{}-OOI".format(agent_ind) if agent_ind in ooi else str(agent_ind)

        # Create a rectangular patch for each agent with black edge
        length = agent_shape[agent_ind, 0]
        width = agent_shape[agent_ind, 1]

        rect = Rectangle(
            (-length / 2, -width / 2),  # Center it at origin for now
            width=length,
            height=width,
            facecolor=face_color,
            edgecolor='black',
            linewidth=0.6,
            zorder=10
        )

        agent_patches.append(rect)
        ax.add_patch(rect)

        text = ax.text(0, 0, label, color=face_color, fontsize=11, ha='center', va='center', zorder=15)
        agent_texts.append(text)

        # Arrow from agent to destination
        arrow = FancyArrowPatch((0, 0), (0, 0),
                                facecolor=face_color,
                                edgecolor='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=0.8,
                                zorder=5)
        arrow.set_visible(False)  # Initially set to invisible
        agent_arrows.append(arrow)
        ax.add_patch(arrow)

        star = ax.scatter(0, 0, s=30, c='green', marker='*', zorder=12, visible=False)
        agent_stars.append(star)

    with writer.saving(fig, save_path, dpi=dpi):
        for t in range(T):
            pos = agent_pos[t]  # update agent positions and labels for each frame
            heading = agent_heading[t]

            for agent_ind, (rect, text) in enumerate(zip(agent_patches, agent_texts)):
                x, y = pos[agent_ind]
                arrow = agent_arrows[agent_ind]
                star = agent_stars[agent_ind]
                if rect is None or text is None or arrow is None or star is None:
                    continue

                if not agent_mask[t, agent_ind]:
                    rect.set_visible(False)
                    text.set_visible(False)
                    arrow.set_visible(False)
                    continue

                # Show rect and label
                rect.set_visible(True)
                text.set_visible(True)
                arrow.set_visible(True)

                length = agent_shape[agent_ind, 0]
                width = agent_shape[agent_ind, 1]

                # Reset base rectangle (centered at origin)
                rect.set_xy((-agent_shape[agent_ind, 0] / 2, -agent_shape[agent_ind, 1] / 2))

                theta_deg = np.degrees(heading[agent_ind])
                # Create a rotation around the agent center
                trans = (
                        transforms.Affine2D()
                        .rotate_deg_around(0, 0, theta_deg)
                        .translate(x, y)
                        + ax.transData
                )
                rect.set_transform(trans)


                rect.set_edgecolor('black')
                rect.set_linewidth(0.8)

                text.set_position((x, y))
                text.set_text(text.get_text())  # forces the text to render

                # Update arrow
                if agent_dest_pos is not None:
                    dest_x, dest_y = agent_dest_pos[t, agent_ind]

                    dest = agent_dest[t, agent_ind]
                    # FIXME: hardcoded
                    if dest == 3002:
                        arrow.set_positions((x, y), (dest_x, dest_y))
                        arrow.set_color("red")
                        arrow.set_visible(True)
                    elif dest == -1:
                        arrow.set_visible(False)
                    else:
                        arrow.set_positions((x, y), (dest_x, dest_y))
                        arrow.set_color("black")
                        arrow.set_visible(True)

                else:
                    arrow.set_visible(False)

            writer.grab_frame()
