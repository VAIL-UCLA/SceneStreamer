"""
A newer version of generator. This time we will have a class that maintain necessary state for autoregressive rollout.
"""

import copy
import pathlib

import numpy as np
import torch
import tqdm
from shapely.geometry import Polygon

from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour
from scenestreamer.dataset.preprocessor import NUM_TG_MULTI, TG_SKIP_STEP
from scenestreamer.infer import scenestreamer_motion
from scenestreamer.models.scenestreamer_model import get_num_tg
from scenestreamer.tokenization.motion_tokenizers import START_ACTION as MOTION_START_ACTION
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils


def evict_agents_function(
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
    step_data_dict["agent_valid_mask"] = new_mask

    step_info_dict["evicted_agents"] = num_evicted
    step_info_dict["evicted_agent_mask"] = should_evict

    return step_data_dict, step_info_dict





# A coding trick here to accumulate for multiple TG tokens before calling "prepare_trafficgen_single_token"
class TGTokenBuffer:
    def __init__(self):
        self.tg_action = []
        self.tg_type = []
        self.tg_agent_id = []
        self.tg_intra_step = []
        self.tg_feat = []
        self.position = []
        self.heading = []
        self.valid_mask = []
        self.width = []
        self.length = []
        self.causal_mask = []
        self.force_mask = []
        self.current_step = []
        self.require_relation = []

    def add(
            self, *, tg_action, tg_type, tg_agent_id, tg_intra_step, tg_feat,
            position, heading, valid_mask, width, length, causal_mask,
            force_mask, current_step, require_relation
    ):
        self.tg_action.append(tg_action)
        self.tg_type.append(tg_type)
        self.tg_agent_id.append(tg_agent_id)
        self.tg_intra_step.append(tg_intra_step)
        self.tg_feat.append(tg_feat)
        self.position.append(position)
        self.heading.append(heading)
        self.valid_mask.append(valid_mask)
        self.width.append(width)
        self.length.append(length)
        self.causal_mask.append(causal_mask)
        self.force_mask.append(force_mask)
        self.current_step.append(current_step)
        self.require_relation.append(require_relation)

    def append_to_scenestreamer_tokens(self, *, model, scenestreamer_tokens):
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=torch.cat(self.tg_action, dim=1),
            tg_type=torch.cat(self.tg_type, dim=1),
            tg_agent_id=torch.cat(self.tg_agent_id, dim=1),
            tg_intra_step=torch.cat(self.tg_intra_step, dim=1),
            tg_feat=torch.cat(self.tg_feat, dim=1),
        )
        assert self.current_step[0] == self.current_step[-1]

        num_new_keys = self.causal_mask[-1].shape[-1]
        B = self.causal_mask[-1].shape[0]
        N = len(self.causal_mask)
        new_all_causal_mask = self.causal_mask[0].new_zeros(B, N, num_new_keys)
        for i in range(N):
            new_all_causal_mask[:, i:i+1, :self.causal_mask[i].shape[2]] = self.causal_mask[i]

        new_all_force_mask = self.force_mask[0].new_zeros(B, N, num_new_keys)
        for i in range(N):
            new_all_force_mask[:, i:i+1, :self.force_mask[i].shape[2]] = self.force_mask[i]

        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.cat(self.position, dim=1),
            heading=torch.cat(self.heading, dim=1),
            valid_mask=torch.cat(self.valid_mask, dim=1),
            width=torch.cat(self.width, dim=1),
            length=torch.cat(self.length, dim=1),
            causal_mask=new_all_causal_mask,
            current_step=self.current_step[0],
            require_relation=torch.cat(self.require_relation, dim=1),
            force_mask=new_all_force_mask
        )

        # import matplotlib.pyplot as plt
        # vis = new_all_force_mask[0].cpu().numpy()
        # plt.imshow(vis)
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # vis = (new_all_causal_mask|new_all_force_mask)[0].cpu().numpy()
        # plt.imshow(vis)




class SceneStreamerGenerator:
    STATE_START = 0
    STATE_TRAFFICLIGHT_DONE = 1
    STATE_TRAFFICGEN_DONE = 2
    STATE_MOTION_DONE = 3
    STATE_TRAFFICGEN_SKIPPED = 4

    def __init__(self, model, device):
        # self.env = env
        self.model = model
        self.config = model.config
        self.state = None
        self.device = device
        self.keep_output_token = False

    def reset(self, new_sd=None, new_data_dict=None):
        self.raw_data_dict = copy.deepcopy(new_data_dict)
        self.state = self.STATE_START
        self.current_step = 0

        model = self.model
        data_dict = self.raw_data_dict
        # assert teacher_forcing_dest is not None, "Please set teacher_forcing_dest to True or False"
        # ===== Some preprocessing =====
        self.topp = model.config.SAMPLING.TOPP
        self.temperature = model.config.SAMPLING.TEMPERATURE
        self.sampling_method = model.config.SAMPLING.SAMPLING_METHOD
        B, T_input, N = data_dict["decoder/input_action"].shape[:3]
        num_decode_steps = 19
        assert num_decode_steps == 19
        assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)

        # ===== Encode scenes =====
        data_dict, _ = scenestreamer_motion.encode_scene(data_dict=data_dict, model=model)

        # ===== Create a temporary input_dict removing the future information =====
        _, _, L = data_dict["encoder/traffic_light_state"].shape
        self.scenestreamer_tokens = None
        self.step_info_dict = {}

        # TODO: Are they still correct if we are using TG??
        G = get_num_tg(N)
        all_token_casual_mask = model._build_all_tokens_mask(
            B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
        ).to(data_dict["decoder/input_action"].device)
        self.all_token_casual_mask = all_token_casual_mask
        all_force_mask = model._build_all_force_mask(
            B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
        ).to(data_dict["decoder/input_action"].device)
        self.all_force_mask = all_force_mask

    def _tg_generate_agent_agent_state(
            self, *, agent_id, agent_type, tg_intra_step, tg_input_action, agent_valid_mask,
            teacher_forcing_agent_state,

            selected_map_pos,
            selected_map_heading,
    ):
        assert agent_id.shape == (self.B, 1)
        assert agent_type.shape == (self.B, 1)
        assert tg_input_action.shape == (self.B, 1)
        assert agent_valid_mask.shape == (self.B, 1)

        model = self.model
        B = self.B
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask
        scenestreamer_tokens = self.scenestreamer_tokens
        device = self.device
        current_step = self.current_step

        tg_token = model.prepare_trafficgen_single_token(
            tg_action=tg_input_action,
            tg_type=agent_type,
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), tg_intra_step, device=device),
            tg_feat=torch.zeros((B, 1, 8), device=device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=selected_map_pos,
            heading=selected_map_heading,
            valid_mask=agent_valid_mask,
            width=torch.full((B, 1), 0.0, device=device),
            length=torch.full((B, 1), 0.0, device=device),
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=agent_valid_mask,
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                       :scenestreamer_tokens.seq_len + 1]
        )

        # Call model if you don't want to teacher forcing:
        if teacher_forcing_agent_state:
            return None, {}
        else:
            assert agent_valid_mask.all()
            output_dict = scenestreamer_tokens.call_model_with_cache(keep_output_token=self.keep_output_token)
            output_token = output_dict["model/all_token"][:, -1:]
            # call pred head to get agent feat.

            output_token = model.trafficgen_prenorm(output_token)

            z = output_token.reshape(self.B, -1)
            offset_output = model.trafficgen_head.generate(z=z)

            assert offset_output.shape == (B, 9)

            offset_action = offset_output[:, 1:]
            assert offset_action.shape == (B, 8), "offset_action shape: {}".format(offset_action.shape)
            offset_action = {
                "length": offset_action[:, 0].reshape(B, 1),
                "width": offset_action[:, 1].reshape(B, 1),
                "height": offset_action[:, 2].reshape(B, 1),
                "position_x": offset_action[:, 3].reshape(B, 1),
                "position_y": offset_action[:, 4].reshape(B, 1),
                "heading": offset_action[:, 5].reshape(B, 1),
                "velocity_x": offset_action[:, 6].reshape(B, 1),
                "velocity_y": offset_action[:, 7].reshape(B, 1),
            }
            agent_state_output = self.model.trafficgen_tokenizer.detokenize(
                data_dict=self.raw_data_dict, action=tg_input_action, agent_type=agent_type, offset_action=offset_action
            )
            return agent_state_output, {}

    def _get_map_pos_head(self, index: torch.Tensor):
        assert index.numel() == self.B
        M = self.raw_data_dict["model/map_token_position"].shape[1]
        assert index.min() >= 0, "index: {}, M: {}".format(index, M)
        assert index.max() < M, "index: {}, M: {}".format(index, M)
        pos = torch.gather(
            self.raw_data_dict["model/map_token_position"][..., :2],
            index=index.reshape(self.B, 1, 1).expand(self.B, 1, 2),
            dim=1
        )
        pos = pos.reshape(self.B, 1, 2)
        heading = torch.gather(
            self.raw_data_dict["model/map_token_heading"],
            index=index.reshape(self.B, 1),
            dim=1
        )
        heading = heading.reshape(self.B, 1)
        return pos, heading

    def _tg_generate_agent_map_id(
            self, *, agent_id, agent_type, tg_intra_step, tg_input_action, agent_valid_mask, teacher_forcing_map_id,
            sdc_position,
    ):
        assert (agent_type == -1).all()
        assert agent_id.shape == (self.B, 1)
        assert agent_type.shape == (self.B, 1)
        assert tg_input_action.shape == (self.B, 1)
        assert agent_valid_mask.shape == (self.B, 1)
        assert tg_input_action.min() >= self.model.veh_id
        assert tg_input_action.max() <= self.model.cyc_id

        model = self.model
        B = self.B
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask
        scenestreamer_tokens = self.scenestreamer_tokens
        device = self.device
        current_step = self.current_step

        tg_token = model.prepare_trafficgen_single_token(
            tg_action=tg_input_action,
            tg_type=agent_type,
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), tg_intra_step, device=device),
            tg_feat=torch.zeros((B, 1, 8), device=device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.zeros((B, 1, 2), device=device),
            heading=torch.zeros((B, 1), device=device),
            valid_mask=agent_valid_mask,
            width=torch.full((B, 1), 0.0, device=device),
            length=torch.full((B, 1), 0.0, device=device),
            causal_mask=tg_causal_mask,
            current_step=current_step,
            # <<< When generating map id, input is agent type. so no relation is required.
            require_relation=torch.full((B, 1), False, device=device),
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                       :scenestreamer_tokens.seq_len + 1]
        )
        # Call model if you don't want to teacher forcing:
        if teacher_forcing_map_id:
            return None, {}

        else:
            assert agent_valid_mask.all()
            output_dict = scenestreamer_tokens.call_model_with_cache(keep_output_token=self.keep_output_token)
            output_token = output_dict["model/all_token"][:, -1:]
            # call pred head to get agent feat.

            output_token = model.trafficgen_prenorm(output_token)
            map_id_logit = model.trafficgen_head.map_id_head(output_token)
            assert map_id_logit.shape[1] == 1

            map_id_logit_mask = torch.full((B, 1, map_id_logit.shape[-1]), False, device=agent_type.device,
                                           dtype=torch.bool)

            M = self.raw_data_dict["model/map_token_position"].shape[1]
            map_id_logit_mask[:, :, :M] = self.raw_data_dict["model/map_token_valid_mask"][:, None]

            if self.config.EVALUATION.TG_SDC_DISTANCE_MASKING:
                map_pos = self.raw_data_dict["model/map_token_position"][..., :2]  # (B, M, 2)
                THRESHOLD = 50.0
                closed_to_sdc_mask = (map_pos - sdc_position).norm(dim=-1) < THRESHOLD
                map_id_logit_mask[:, :, :M] = map_id_logit_mask[:, :, :M] & closed_to_sdc_mask[:, None]

            # only_lane = True
            # if only_lane:
            #     map_feature = self.raw_data_dict["encoder/map_feature"]
            #     map_id_logit_mask[:, :, :M] = (map_feature[:, :, 0, 13] == 1)[:, None] & map_id_logit_mask[:, :, :M]

            map_id_logit[~map_id_logit_mask] = float("-inf")

            map_id, _ = scenestreamer_motion.sample_action(map_id_logit, sampling_method="softmax")

            map_id_pad_mask = map_id == model.trafficgen_sequence_pad_id
            map_id[map_id_pad_mask] = model.trafficgen_sequence_pad_id

            return map_id, {
                "map_id_logit": map_id_logit,
            }

    def _tg_generate_dest(
            self, *, agent_pos, agent_heading, current_step,
            agent_width, agent_length, agent_id, agent_type, tg_intra_step,
            tg_input_action, agent_feature, agent_valid_mask
    ):
        assert agent_width.shape == (self.B, 1)
        assert agent_length.shape == (self.B, 1)
        assert agent_pos.shape == (self.B, 1, 2)
        assert agent_heading.shape == (self.B, 1)
        assert agent_id.shape == (self.B, 1)
        assert agent_type.shape == (self.B, 1), "agent_type shape: {}".format(agent_type.shape)
        assert tg_input_action.shape == (self.B, 1)
        assert agent_feature.shape == (self.B, 1, 8)
        assert agent_valid_mask.shape == (self.B, 1)

        model = self.model
        B = self.B
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask
        scenestreamer_tokens = self.scenestreamer_tokens

        tg_token = model.prepare_trafficgen_single_token(
            tg_action=tg_input_action,
            tg_type=agent_type,
            tg_agent_id=agent_id,
            tg_intra_step=torch.full((B, 1), tg_intra_step, device=agent_type.device),
            tg_feat=agent_feature,
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=agent_pos,
            heading=agent_heading,
            valid_mask=agent_valid_mask,
            width=agent_width,
            length=agent_length,
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=agent_valid_mask,
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                       :scenestreamer_tokens.seq_len + 1]
        )

        # if agent_valid_mask.any():
        #     output_dict = scenestreamer_tokens.call_model_with_cache(keep_output_token=self.keep_output_token)
        #     output_token = output_dict["model/all_token"][:, -1:]
        #     # call pred head to get agent feat.
        #
        #     output_token = model.trafficgen_prenorm(output_token)
        #
        #     dest_id_logit = model.trafficgen_head.dest_id_head(output_token)
        #     # tiny masked out here
        #     M = data_dict["model/map_token_valid_mask"].shape[1]
        #     assert dest_id_logit.shape[1] == 1
        #
        #     dest_id_logit_mask = torch.full((B, 1, dest_id_logit.shape[-1]), False, device=agent_type.device,
        #                                     dtype=torch.bool)
        #     dest_id_logit_mask[:, :, :M] = data_dict["model/map_token_valid_mask"][:, None]
        #
        #     dest_pos_full = data_dict["model/map_token_position"][..., :2]  # (B, M, 2)
        #     agent_pos = step_info_dict["agent_position"][:, agent_index][:, None]  # (B, 2)
        #     dest_agent_dist = torch.cdist(dest_pos_full, agent_pos)[..., 0]  # (B, M)
        #
        #     speed = step_info_dict["agent_velocity"][:, agent_index].norm(dim=-1)  # (B,)
        #     displacement = speed * 3
        #     tolerance = displacement + 20
        #     # print("Agent {} speed: {}, displacement: {}, tolerance: {}".format(
        #     #     agent_index, speed[0].item(), displacement[0].item(), tolerance[0].item()
        #     # ))
        #     assert dest_agent_dist.ndim == 2
        #     assert tolerance.ndim == 1
        #     assert dest_id_logit_mask.ndim == 3
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (dest_agent_dist < tolerance[:, None])[:,
        #                                                                   None]
        #
        #     agent_heading = step_info_dict["agent_heading"][:, agent_index]
        #
        #     # Only allow dest in front of the agent.
        #     rel_pos = (dest_pos_full - agent_pos)
        #     rel_pos = utils.rotate(x=rel_pos[..., 0], y=rel_pos[..., 1], angle=-agent_heading[:, None].expand(B, M))
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (rel_pos[..., 0] > 0)[:, None]
        #
        #     dest_heading_full = data_dict["model/map_token_heading"]  # (B, M)
        #     dest_agent_heading_dist = dest_heading_full - agent_heading[:, None]  # (B, M)
        #     dest_agent_heading_dist = torch.abs(utils.wrap_to_pi(dest_agent_heading_dist))
        #     dest_id_logit_mask[:, :, :M] = dest_id_logit_mask[:, :, :M] & (dest_agent_heading_dist < np.pi / 2)[:,
        #                                                                   None]
        #
        #     dest_id_logit_mask[..., model.trafficgen_sequence_pad_id] = True
        #
        #     only_lane = True
        #     if only_lane:
        #         map_feature = data_dict["encoder/map_feature"]
        #         dest_id_logit_mask[:, :, :M] = (map_feature[:, :, 0, 13] == 1)[:, None] & dest_id_logit_mask[:, :,
        #                                                                                   :M]
        #
        #     dest_id_logit[~dest_id_logit_mask] = float("-inf")
        #
        #     # TODO: hardcoded
        #     # dest_id, _ = scenestreamer_motion.sample_action(dest_id_logit, sampling_method="softmax")
        #     dest_id, _ = scenestreamer_motion.sample_action(dest_id_logit, sampling_method="topp", topp=0.95)
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
        #     dest_position[dest_id_pad_mask] = step_info_dict["agent_position"][:, agent_index][:, None][
        #         dest_id_pad_mask]
        #
        #     dest_heading = torch.gather(
        #         data_dict["model/map_token_heading"],
        #         index=dest_id.reshape(B, 1),
        #         dim=1
        #     )
        #     dest_heading[dest_id_pad_mask] = step_info_dict["agent_heading"][:, agent_index][:, None][
        #         dest_id_pad_mask]
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
        # dest_id[~agent_valid_mask] = -1
        #
        # return {
        #     "dest_id": dest_id,
        #     "dest_position": dest_position,
        #     "dest_heading": dest_heading,
        # }

    def _step_generate_trafficgen_no_agent_state(self, *, teacher_forcing_from_gt, teacher_forcing_dest=None,
                                                 generate_agent_states=False):
        assert self.state == self.STATE_TRAFFICLIGHT_DONE

        if generate_agent_states:
            return self._step_generate_trafficgen_with_agent_state(
                teacher_forcing_from_gt=teacher_forcing_from_gt,
                teacher_forcing_dest=teacher_forcing_dest
            )

        model = self.model
        data_dict = self.raw_data_dict
        scenestreamer_tokens = self.scenestreamer_tokens
        current_step = self.current_step
        step_info_dict = self.step_info_dict
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask

        if teacher_forcing_from_gt:
            step_info_dict["agent_valid_mask"] = data_dict["decoder/input_action_valid_mask"][:, current_step].clone()
            step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:, current_step].clone()
            step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:, current_step].clone()
            step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:, current_step].clone()
            step_info_dict["agent_type"] = data_dict["decoder/agent_type"].clone()
            step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"].clone()
            step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"].clone()

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
        input_action_for_trafficgen = torch.from_numpy(np.stack(tg_map_id_list, axis=0)).to(device=device)
        input_action_for_trafficgen = input_action_for_trafficgen.reshape(B, 1, G)
        # input_action_valid_mask_for_trafficgen = torch.from_numpy(np.stack(tg_valid_list, axis=0)).to(
        #     device=device).reshape(B, 1, G)
        agent_feature_for_trafficgen = (torch.from_numpy(np.stack(tg_feat_list, axis=0)).to(device=device)
                                        .reshape(B, 1, G, 8).float())
        trafficgen_position = torch.from_numpy(np.stack(tg_pos_list, axis=0)).to(device=device).reshape(B, 1, G,
                                                                                                        2).float()
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

        # ===== call model for tg autoregressive =====

        token_buffer = TGTokenBuffer()

        initial_seq_len = scenestreamer_tokens.seq_len

        # First, input the sequence_sos_id.
        intra_step = 0
        token_buffer.add(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_sos_id, device=agent_type.device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
            causal_mask=all_token_casual_mask[:, initial_seq_len+intra_step:initial_seq_len + intra_step + 1,
                         :initial_seq_len + intra_step + 1],
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            force_mask=all_force_mask[:, initial_seq_len+intra_step:initial_seq_len + intra_step + 1,
                       :initial_seq_len + intra_step + 1],
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
            token_buffer.add(
                tg_action=torch.full((B, 1), model.trafficgen_agent_sos_id, device=agent_type.device),
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
                position=torch.full((B, 1, 2), 0, device=agent_type.device),
                heading=torch.full((B, 1), 0, device=agent_type.device),
                valid_mask=this_agent_valid_mask,
                width=torch.full((B, 1), 0.0, device=agent_type.device),
                length=torch.full((B, 1), 0.0, device=agent_type.device),
                causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1],
                current_step=current_step,
                require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
                force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1]
            )

            # Step 1: input is the agent type.
            intra_step += 1
            token_buffer.add(
                tg_action=agent_type[:, agent_index][:, None],
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
                position=torch.full((B, 1, 2), 0.0, device=agent_type.device),
                heading=torch.full((B, 1), 0.0, device=agent_type.device),
                valid_mask=this_agent_valid_mask,
                width=torch.full((B, 1), 0.0, device=agent_type.device),
                length=torch.full((B, 1), 0.0, device=agent_type.device),
                causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1],
                current_step=current_step,
                require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
                force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1]
            )


            # Step 2: input is the map id.
            intra_step += 1
            token_buffer.add(
                tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
                position=trafficgen_position[:, 0, intra_step:intra_step + 1],
                heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
                valid_mask=this_agent_valid_mask,
                # TODO: hardcoded 5, 6
                width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
                length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
                causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1],
                current_step=current_step,
                require_relation=this_agent_valid_mask,
                force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1]
            )


            # Step 3: input is the agent feat.
            intra_step += 1
            token_buffer.add(
                tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
                position=trafficgen_position[:, 0, intra_step:intra_step + 1],
                heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
                valid_mask=this_agent_valid_mask,
                causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1],
                force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                           :initial_seq_len + intra_step + 1],
                require_relation=this_agent_valid_mask,
                width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
                length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
                current_step=current_step,
            )

            # quick fix a stupid bug. if we buffer too much it might OOM the GPU...............
            if len(token_buffer.current_step) > self.config.TOKEN_BUFFER_CACHE_LENGTH:
                if scenestreamer_tokens.able_to_call_model():
                    token_buffer.append_to_scenestreamer_tokens(
                        model=model,
                        scenestreamer_tokens=scenestreamer_tokens,
                    )
                    scenestreamer_tokens.call_model_with_cache()
                    token_buffer = TGTokenBuffer()

        # Finally, input the sequence_eos_id.
        intra_step += 1
        assert intra_step == G - 1, (intra_step, G, G - 1)
        token_buffer.add(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_eos_id, device=agent_type.device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                        :initial_seq_len + intra_step + 1],
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
            force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                       :initial_seq_len + intra_step + 1]
        )

        # The only thing need to be updated by non-teacher_forcing TG is the destination:
        # step_info_dict["agent_destination"] = torch.stack(agent_destination_list, dim=1)
        # step_info_dict["agent_destination_position"] = torch.stack(agent_destination_pos_list, dim=1)
        token_buffer.append_to_scenestreamer_tokens(
            model=model,
            scenestreamer_tokens=scenestreamer_tokens,
        )
        self.step_info_dict = step_info_dict
        self.state = self.STATE_TRAFFICGEN_DONE

    def _step_generate_trafficgen_densified_agent_state(self, *, teacher_forcing_from_gt,
                                                        veh_ratio, ped_ratio, num_new_agents,
                                                        teacher_forcing_dest=None,
                                                        generate_agent_states=False):
        assert self.state == self.STATE_TRAFFICLIGHT_DONE

        if generate_agent_states:
            return self._step_generate_trafficgen_with_agent_state(
                teacher_forcing_from_gt=teacher_forcing_from_gt,
                teacher_forcing_dest=teacher_forcing_dest
            )

        model = self.model
        data_dict = self.raw_data_dict
        scenestreamer_tokens = self.scenestreamer_tokens
        current_step = self.current_step
        step_info_dict = self.step_info_dict
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask

        if teacher_forcing_from_gt:

            atype = data_dict["decoder/agent_type"].clone()
            old_N = atype.shape[1]
            num_veh = (atype == model.veh_id).sum(dim=1)
            num_ped = (atype == model.ped_id).sum(dim=1)
            num_cyc = (atype == model.cyc_id).sum(dim=1)

            # For now just trying to fill vehicle.
            new_N = max(num_new_agents, old_N)

            # num_veh_to_add = new_N - old_N

            quota = new_N - old_N - 3  # At least 3 guys left

            if quota < 0:
                quota = 0

            num_veh_to_add = int(veh_ratio * quota)
            new_num_veh = num_veh + num_veh_to_add

            num_ped_to_add = int(ped_ratio * quota)
            new_num_ped = num_ped + num_ped_to_add

            num_cyc_to_add = new_N - new_num_veh - new_num_ped
            new_num_cyc = num_cyc + num_cyc_to_add

            print("num_veh: {}, num_ped: {}, num_cyc: {}".format(
                new_num_veh, new_num_ped, new_num_cyc
            ))
            new_atype = torch.full((data_dict["decoder/agent_type"].shape[0], new_N), -1, device=atype.device)
            new_atype[:, :new_num_veh] = model.veh_id
            new_atype[:, new_num_veh:new_num_veh + new_num_ped] = model.ped_id
            new_atype[:, new_num_veh + new_num_ped:new_N] = model.cyc_id

            # Assume all valid:
            new_valid_mask = torch.full((data_dict["decoder/input_action_valid_mask"].shape[0], new_N), True,
                                        device=atype.device)

            def _fill(arr):
                if arr.ndim == 2:
                    new_arr = arr.new_zeros((arr.shape[0], new_N,))
                elif arr.ndim == 3:
                    new_arr = arr.new_zeros((arr.shape[0], new_N, arr.shape[-1]))
                else:
                    print("arr shape: ", arr.shape)
                    raise ValueError
                new_arr[:, :num_veh] = arr[:, :num_veh]
                new_arr[:, new_num_veh:new_num_veh + num_ped] = arr[:, num_veh:num_veh + num_ped]
                if num_veh + num_ped < old_N:
                    l = arr[:, num_veh + num_ped:].shape[1]
                    if l > 0:
                        new_arr[:, new_num_veh + new_num_ped:new_num_veh + new_num_ped + l] = arr[:, num_veh + num_ped:]
                return new_arr

            step_info_dict["agent_type"] = new_atype
            step_info_dict["agent_valid_mask"] = new_valid_mask

            step_info_dict["agent_position"] = _fill(
                data_dict["decoder/modeled_agent_position"][:, current_step].clone())
            step_info_dict["agent_heading"] = _fill(data_dict["decoder/modeled_agent_heading"][:, current_step].clone())
            step_info_dict["agent_velocity"] = _fill(
                data_dict["decoder/modeled_agent_velocity"][:, current_step].clone())
            step_info_dict["agent_shape"] = _fill(data_dict["decoder/current_agent_shape"].clone())
            step_info_dict["agent_id"] = torch.arange(0, new_N, device=atype.device).expand(
                data_dict["decoder/agent_type"].shape[0], -1)

            # TODO FIXME
            # TODO FIXME
            # TODO FIXME
            # TODO FIXME
            should_create_new_agent = torch.logical_not(_fill(
                data_dict["decoder/input_action_valid_mask"][:, current_step].clone()
            ))

            # B  = data_dict["decoder/input_action_valid_mask"].shape[0]
            # should_create_new_agent = torch.ones((B, new_N))
            # should_create_new_agent[:, 0] = True









            scenestreamer_tokens.N = new_N
            scenestreamer_tokens.G = get_num_tg(new_N)

            new_input_action = _fill(data_dict["decoder/input_action"][:, current_step].clone())
            new_input_action.fill_(MOTION_START_ACTION)

            step_info_dict["motion_input_action"] = new_input_action

        else:
            should_create_new_agent = torch.zeros(
                (scenestreamer_tokens.B, scenestreamer_tokens.N),
                device=step_info_dict["agent_valid_mask"].device,
                dtype=torch.bool
            )

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
        input_action_for_trafficgen = torch.from_numpy(np.stack(tg_map_id_list, axis=0)).to(device=device)
        input_action_for_trafficgen = input_action_for_trafficgen.reshape(B, 1, G)
        # input_action_valid_mask_for_trafficgen = torch.from_numpy(np.stack(tg_valid_list, axis=0)).to(
        #     device=device).reshape(B, 1, G)
        agent_feature_for_trafficgen = (torch.from_numpy(np.stack(tg_feat_list, axis=0)).to(device=device)
                                        .reshape(B, 1, G, 8).float())
        trafficgen_position = torch.from_numpy(np.stack(tg_pos_list, axis=0)).to(device=device).reshape(B, 1, G,
                                                                                                        2).float()
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

        # ===== call model for tg autoregressive =====

        token_buffer = TGTokenBuffer()

        initial_seq_len = scenestreamer_tokens.seq_len

        # First, input the sequence_sos_id.
        intra_step = 0
        token_buffer.add(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_sos_id, device=agent_type.device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
            causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                        :initial_seq_len + intra_step + 1],
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                       :initial_seq_len + intra_step + 1],
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
        )

        agent_destination_list = []
        agent_destination_pos_list = []
        for agent_index in range(N):

            assert B == 1
            should_create_new = should_create_new_agent[0, agent_index].item()

            agent_id = step_info_dict["agent_id"][:, agent_index:agent_index + 1]
            this_agent_valid_mask = step_info_dict["agent_valid_mask"][:, agent_index:agent_index + 1]

            # Step 0, agent start token.
            intra_step += 1
            token_buffer.add(
                tg_action=torch.full((B, 1), model.trafficgen_agent_sos_id, device=agent_type.device),
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
                position=torch.full((B, 1, 2), 0, device=agent_type.device),
                heading=torch.full((B, 1), 0, device=agent_type.device),
                valid_mask=this_agent_valid_mask,
                width=torch.full((B, 1), 0.0, device=agent_type.device),
                length=torch.full((B, 1), 0.0, device=agent_type.device),
                causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                            :initial_seq_len + intra_step + 1],
                current_step=current_step,
                require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
                force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                           :initial_seq_len + intra_step + 1]
            )

            if not should_create_new:

                # Step 1: input is the agent type.
                intra_step += 1
                token_buffer.add(
                    tg_action=agent_type[:, agent_index][:, None],
                    tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_agent_id=agent_id,
                    tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                    tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
                    position=torch.full((B, 1, 2), 0.0, device=agent_type.device),
                    heading=torch.full((B, 1), 0.0, device=agent_type.device),
                    valid_mask=this_agent_valid_mask,
                    width=torch.full((B, 1), 0.0, device=agent_type.device),
                    length=torch.full((B, 1), 0.0, device=agent_type.device),
                    causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                                :initial_seq_len + intra_step + 1],
                    current_step=current_step,
                    require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
                    force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                               :initial_seq_len + intra_step + 1]
                )

                # Step 2: input is the map id.
                intra_step += 1
                token_buffer.add(
                    tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_agent_id=agent_id,
                    tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                    tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    position=trafficgen_position[:, 0, intra_step:intra_step + 1],
                    heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
                    valid_mask=this_agent_valid_mask,
                    # TODO: hardcoded 5, 6
                    width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
                    length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
                    causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                                :initial_seq_len + intra_step + 1],
                    current_step=current_step,
                    require_relation=this_agent_valid_mask,
                    force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                               :initial_seq_len + intra_step + 1]
                )

                # Step 3: input is the agent feat.
                intra_step += 1
                token_buffer.add(
                    tg_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_agent_id=agent_id,
                    tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
                    tg_feat=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    position=trafficgen_position[:, 0, intra_step:intra_step + 1],
                    heading=trafficgen_heading[:, 0, intra_step:intra_step + 1],
                    valid_mask=this_agent_valid_mask,
                    causal_mask=all_token_casual_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                                :initial_seq_len + intra_step + 1],
                    force_mask=all_force_mask[:, initial_seq_len + intra_step:initial_seq_len + intra_step + 1,
                               :initial_seq_len + intra_step + 1],
                    require_relation=this_agent_valid_mask,
                    width=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 6],
                    length=agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1][..., 5],
                    current_step=current_step,
                )



            else:
                # save current buffer first..
                token_buffer.append_to_scenestreamer_tokens(
                    model=model,
                    scenestreamer_tokens=scenestreamer_tokens,
                )
                if scenestreamer_tokens.able_to_call_model():
                    scenestreamer_tokens.call_model_with_cache()
                token_buffer = TGTokenBuffer()

                # Adding new agent!

                if self.config.EVALUATION.TG_REJECT_SAMPLING:
                    self.scenestreamer_tokens = scenestreamer_tokens
                    tmp_scenestreamer_tokens = copy.deepcopy(scenestreamer_tokens)
                    tmp_step_info_dict = copy.deepcopy(self.step_info_dict)
                    tmp_intra_step = copy.deepcopy(intra_step)

                this_agent_reject_count = 0
                while True:
                    # Step 1: input is the agent type.
                    teacher_forcing_this_agent = torch.full((B, 1), False, device=agent_type.device)
                    intra_step += 1
                    assert data_dict["decoder/sdc_index"][0].item() == 0, data_dict["decoder/sdc_index"]
                    selected_map_id, map_id_info = self._tg_generate_agent_map_id(
                        agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                        agent_id=agent_id,
                        tg_intra_step=intra_step,
                        agent_valid_mask=this_agent_valid_mask,
                        teacher_forcing_map_id=teacher_forcing_this_agent,
                        tg_input_action=agent_type[:, agent_index][:, None],
                        sdc_position=step_info_dict["agent_position"][:, 0:1],
                    )

                    # Step 2: input is the map id.
                    intra_step += 1
                    assert selected_map_id is not None
                    selected_map_id = selected_map_id.reshape(B, 1)
                    selected_map_pos, selected_map_heading = self._get_map_pos_head(index=selected_map_id)
                    selected_agent_state, as_info = self._tg_generate_agent_agent_state(
                        agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                        agent_id=agent_id,
                        tg_intra_step=intra_step,
                        agent_valid_mask=this_agent_valid_mask,
                        teacher_forcing_agent_state=teacher_forcing_this_agent,
                        tg_input_action=selected_map_id,
                        selected_map_pos=selected_map_pos,
                        selected_map_heading=selected_map_heading,
                    )

                    # Step 3: input is the agent feat.
                    intra_step += 1
                    as_position = selected_agent_state["position"]
                    as_heading = selected_agent_state["heading"]
                    as_feat = torch.zeros((B, 1, 8), device=device)
                    as_feat[:, :, 0] = selected_agent_state["offset_values"]["position_x"]
                    as_feat[:, :, 1] = selected_agent_state["offset_values"]["position_y"]
                    as_feat[:, :, 2] = selected_agent_state["offset_values"]["heading"]
                    as_feat[:, :, 3] = selected_agent_state["offset_values"]["velocity_x"]  # original_relative_vel
                    as_feat[:, :, 4] = selected_agent_state["offset_values"]["velocity_y"]  # original_relative_vel
                    as_feat[:, :, 5] = selected_agent_state["offset_values"]["length"]
                    as_feat[:, :, 6] = selected_agent_state["offset_values"]["width"]
                    as_feat[:, :, 7] = selected_agent_state["offset_values"]["height"]
                    # Overwrite agent data.
                    step_info_dict["agent_position"][:, agent_index] = as_position.clone().reshape(B, 2)
                    step_info_dict["agent_heading"][:, agent_index] = as_heading.clone().reshape(B, )
                    step_info_dict["agent_velocity"][:, agent_index] = selected_agent_state[
                        "velocity"].clone().reshape(B, 2)
                    step_info_dict["agent_shape"][:, agent_index] = selected_agent_state["shape"].clone().reshape(B,
                                                                                                                  3)
                    step_info_dict["agent_valid_mask"][:, agent_index] = this_agent_valid_mask.clone().reshape(B)
                    dest_out = self._tg_generate_dest(
                        agent_pos=as_position,
                        agent_heading=as_heading,
                        current_step=current_step,
                        agent_width=as_feat[..., 6],
                        agent_length=as_feat[..., 5],
                        agent_id=agent_id,
                        agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                        tg_intra_step=intra_step,
                        tg_input_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                        agent_feature=as_feat,
                        agent_valid_mask=this_agent_valid_mask
                    )

                    # Detect whether collision happens.
                    if agent_index == 0:
                        break

                    if not self.config.EVALUATION.TG_REJECT_SAMPLING:
                        # Skip the collision check.
                        break

                    if this_agent_reject_count > 5:
                        break

                    pos = self.step_info_dict["agent_position"][:, :agent_index + 1]
                    head = self.step_info_dict["agent_heading"][:, :agent_index + 1]
                    shape = self.step_info_dict["agent_shape"][:, :agent_index + 1]
                    # assert B == 1
                    for b in range(B):
                        poly = cal_polygon_contour(
                            x=pos[b, :, 0].cpu().numpy(),
                            y=pos[b, :, 1].cpu().numpy(),
                            theta=head[b].cpu().numpy(),
                            width=shape[b, :, 1].cpu().numpy(),
                            length=shape[b, :, 0].cpu().numpy()
                        )
                        last_poly = poly[-1]
                        last_poly = Polygon(last_poly)
                        coll = False
                        for i in range(len(poly) - 1):
                            poly2 = Polygon(poly[i])
                            if last_poly.intersects(poly2):
                                coll = True
                                coll_b = b
                                break
                    if coll:
                        print("Collision happens at batch {}, repeat the generation.".format(coll_b))
                        should_repeat = True
                    else:
                        should_repeat = False
                        break

                    self.scenestreamer_tokens = copy.deepcopy(tmp_scenestreamer_tokens)
                    self.step_info_dict = copy.deepcopy(tmp_step_info_dict)
                    intra_step = copy.deepcopy(tmp_intra_step)
                    step_info_dict = self.step_info_dict
                    scenestreamer_tokens = self.scenestreamer_tokens

            # quick fix a stupid bug. if we buffer too much it might OOM the GPU...............
            if len(token_buffer.current_step) > 100:
                if scenestreamer_tokens.able_to_call_model():
                    token_buffer.append_to_scenestreamer_tokens(
                        model=model,
                        scenestreamer_tokens=scenestreamer_tokens,
                    )
                    scenestreamer_tokens.call_model_with_cache()
                    token_buffer = TGTokenBuffer()

        # Finally, input the sequence_eos_id.
        intra_step += 1
        assert intra_step == G - 1, (intra_step, G, G - 1)
        token_buffer.add(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_eos_id, device=agent_type.device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=torch.full((B, 1), -1, device=agent_type.device),
            tg_intra_step=torch.full((B, 1), intra_step, device=agent_type.device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=agent_type.device),
            position=torch.full((B, 1, 2), 0, device=agent_type.device),
            heading=torch.full((B, 1), 0, device=agent_type.device),
            valid_mask=torch.full((B, 1), True, device=agent_type.device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=agent_type.device),
            length=torch.full((B, 1), 0.0, device=agent_type.device),
            causal_mask=all_token_casual_mask[:, initial_seq_len+intra_step:initial_seq_len + intra_step + 1,
                         :initial_seq_len + intra_step + 1],
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=agent_type.device, dtype=torch.bool),
            force_mask=all_force_mask[:, initial_seq_len+intra_step:initial_seq_len + intra_step + 1,
                        :initial_seq_len + intra_step + 1]
        )

        # The only thing need to be updated by non-teacher_forcing TG is the destination:
        # step_info_dict["agent_destination"] = torch.stack(agent_destination_list, dim=1)
        # step_info_dict["agent_destination_position"] = torch.stack(agent_destination_pos_list, dim=1)
        token_buffer.append_to_scenestreamer_tokens(
            model=model,
            scenestreamer_tokens=scenestreamer_tokens,
        )
        self.step_info_dict = step_info_dict
        self.state = self.STATE_TRAFFICGEN_DONE

    def _step_generate_trafficgen_with_agent_state(self, *, teacher_forcing_from_gt, teacher_forcing_dest=None):
        assert self.state == self.STATE_TRAFFICLIGHT_DONE

        model = self.model
        data_dict = self.raw_data_dict
        scenestreamer_tokens = self.scenestreamer_tokens
        current_step = self.current_step
        step_info_dict = self.step_info_dict
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask

        # assert teacher_forcing_from_gt is False, "teacher_forcing_from_gt should be False for trafficgen with agent state"
        if teacher_forcing_from_gt:
            step_info_dict["agent_valid_mask"] = data_dict["decoder/input_action_valid_mask"][:, current_step].clone()
            step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:, current_step].clone()
            step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:, current_step].clone()
            step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:, current_step].clone()
            step_info_dict["agent_type"] = data_dict["decoder/agent_type"].clone()
            step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"].clone()
            step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"].clone()

        B, N, G = scenestreamer_tokens.B, scenestreamer_tokens.N, scenestreamer_tokens.G

        device = scenestreamer_tokens.token.device

        # ===== call trafficgen tokenizer =====
        from scenestreamer.dataset.preprocessor import prepare_trafficgen_data_for_scenestreamer_a_step
        tg_map_id_list = []
        tg_feat_list = []
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
            tg_feat_list.append(tg_feat)
        input_action_for_trafficgen = torch.from_numpy(np.stack(tg_map_id_list, axis=0)).to(device=device)
        input_action_for_trafficgen = input_action_for_trafficgen.reshape(B, 1, G)
        agent_feature_for_trafficgen = (torch.from_numpy(np.stack(tg_feat_list, axis=0)).to(device=device)
                                        .reshape(B, 1, G, 8).float())

        # ===== prepare input data for trafficgen =====
        # -1, -1 -1 TYPE -1 -1, ..., -1
        # G = scenestreamer_tokens.G
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

        # ===== call model for tg autoregressive =====
        # First, input the sequence_sos_id.
        intra_step = 0
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_sos_id, device=device),
            tg_type=torch.full((B, 1), -1, device=device),
            tg_agent_id=torch.full((B, 1), -1, device=device),
            tg_intra_step=torch.full((B, 1), intra_step, device=device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]

        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.full((B, 1, 2), 0, device=device),
            heading=torch.full((B, 1), 0, device=device),
            valid_mask=torch.full((B, 1), True, device=device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=device),
            length=torch.full((B, 1), 0.0, device=device),
            causal_mask=tg_causal_mask,
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1],
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=device, dtype=torch.bool),
        )
        # print("BEFORE STEP {}: agent {}, scenestreamer len {}".format(
        #     current_step, 0, scenestreamer_tokens.seq_len
        # ))

        # agent_destination_list = []
        # agent_destination_pos_list = []
        for agent_index in range(N):
            agent_id = step_info_dict["agent_id"][:, agent_index:agent_index + 1]
            assert (agent_id >= 0).all(), agent_id
            # this_agent_valid_mask = step_info_dict["agent_valid_mask"][:, agent_index:agent_index + 1]
            # Teacher forcing the SDC agent!!!
            if agent_index == 0:
                teacher_forcing_this_agent = True
            else:
                teacher_forcing_this_agent = False

            # Because we are generating new agents, we manually set them to be all True.
            this_agent_valid_mask = torch.full((B, 1), True, device=device, dtype=torch.bool)
            assert this_agent_valid_mask.all()

            # Step 0, agent start token.
            intra_step += 1
            assert (agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1] == -1).all()
            tg_token = model.prepare_trafficgen_single_token(
                tg_action=torch.full((B, 1), model.trafficgen_agent_sos_id, device=device),
                tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                tg_agent_id=agent_id,
                tg_intra_step=torch.full((B, 1), intra_step, device=device),
                tg_feat=torch.full((B, 1, 8), 0.0, device=device),
            )
            tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                             :scenestreamer_tokens.seq_len + 1]
            scenestreamer_tokens.add(
                token=tg_token,
                position=torch.full((B, 1, 2), 0, device=device),
                heading=torch.full((B, 1), 0, device=device),
                valid_mask=this_agent_valid_mask,
                width=torch.full((B, 1), 0.0, device=device),
                length=torch.full((B, 1), 0.0, device=device),
                causal_mask=tg_causal_mask,
                current_step=current_step,
                require_relation=torch.full((B, 1), False, device=device, dtype=torch.bool),
                force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                           :scenestreamer_tokens.seq_len + 1]
            )


            if self.config.EVALUATION.TG_REJECT_SAMPLING:
                self.scenestreamer_tokens = scenestreamer_tokens
                tmp_scenestreamer_tokens = copy.deepcopy(scenestreamer_tokens)
                tmp_step_info_dict = copy.deepcopy(self.step_info_dict)
                tmp_intra_step = copy.deepcopy(intra_step)

            this_agent_reject_count = 0
            while True:
                # Step 1: input is the agent type.
                intra_step += 1
                assert data_dict["decoder/sdc_index"][0].item() == 0, data_dict["decoder/sdc_index"]
                selected_map_id, map_id_info = self._tg_generate_agent_map_id(
                    agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    agent_id=agent_id,
                    tg_intra_step=intra_step,
                    agent_valid_mask=this_agent_valid_mask,
                    teacher_forcing_map_id=teacher_forcing_this_agent,
                    tg_input_action=agent_type[:, agent_index][:, None],
                    sdc_position=step_info_dict["agent_position"][:, 0:1],
                )

                # Step 2: input is the map id.
                intra_step += 1
                if teacher_forcing_this_agent:
                    assert selected_map_id is None
                    selected_map_id = input_action_for_trafficgen[:, 0, intra_step:intra_step + 1]
                else:
                    assert selected_map_id is not None
                    selected_map_id = selected_map_id.reshape(B, 1)
                selected_map_pos, selected_map_heading = self._get_map_pos_head(index=selected_map_id)
                selected_agent_state, as_info = self._tg_generate_agent_agent_state(
                    agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    agent_id=agent_id,
                    tg_intra_step=intra_step,
                    agent_valid_mask=this_agent_valid_mask,
                    teacher_forcing_agent_state=teacher_forcing_this_agent,
                    tg_input_action=selected_map_id,
                    selected_map_pos=selected_map_pos,
                    selected_map_heading=selected_map_heading,
                )

                # Step 3: input is the agent feat.
                intra_step += 1
                if teacher_forcing_this_agent:
                    as_position = step_info_dict["agent_position"][:, agent_index].unsqueeze(1)
                    as_heading = step_info_dict["agent_heading"][:, agent_index].unsqueeze(1)
                    as_feat = agent_feature_for_trafficgen[:, 0, intra_step:intra_step + 1]
                else:
                    as_position = selected_agent_state["position"]
                    as_heading = selected_agent_state["heading"]
                    as_feat = torch.zeros((B, 1, 8), device=device)
                    as_feat[:, :, 0] = selected_agent_state["offset_values"]["position_x"]
                    as_feat[:, :, 1] = selected_agent_state["offset_values"]["position_y"]
                    as_feat[:, :, 2] = selected_agent_state["offset_values"]["heading"]
                    as_feat[:, :, 3] = selected_agent_state["offset_values"]["velocity_x"]  # original_relative_vel
                    as_feat[:, :, 4] = selected_agent_state["offset_values"]["velocity_y"]  # original_relative_vel
                    as_feat[:, :, 5] = selected_agent_state["offset_values"]["length"]
                    as_feat[:, :, 6] = selected_agent_state["offset_values"]["width"]
                    as_feat[:, :, 7] = selected_agent_state["offset_values"]["height"]
                    # Overwrite agent data.
                    step_info_dict["agent_position"][:, agent_index] = as_position.clone().reshape(B, 2)
                    step_info_dict["agent_heading"][:, agent_index] = as_heading.clone().reshape(B, )
                    step_info_dict["agent_velocity"][:, agent_index] = selected_agent_state["velocity"].clone().reshape(B, 2)
                    step_info_dict["agent_shape"][:, agent_index] = selected_agent_state["shape"].clone().reshape(B, 3)
                    step_info_dict["agent_valid_mask"][:, agent_index] = this_agent_valid_mask.clone().reshape(B)
                dest_out = self._tg_generate_dest(
                    agent_pos=as_position,
                    agent_heading=as_heading,
                    current_step=current_step,
                    agent_width=as_feat[..., 6],
                    agent_length=as_feat[..., 5],
                    agent_id=agent_id,
                    agent_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    tg_intra_step=intra_step,
                    tg_input_action=input_action_for_trafficgen[:, 0, intra_step:intra_step + 1],
                    agent_feature=as_feat,
                    agent_valid_mask=this_agent_valid_mask
                )

                this_agent_reject_count += 1

                # Detect whether collision happens.
                if agent_index == 0:
                    break

                if not self.config.EVALUATION.TG_REJECT_SAMPLING:
                    # Skip the collision check.
                    break

                if this_agent_reject_count > 5:
                    break

                pos = self.step_info_dict["agent_position"][:, :agent_index + 1]
                head = self.step_info_dict["agent_heading"][:, :agent_index + 1]
                shape = self.step_info_dict["agent_shape"][:, :agent_index + 1]
                # assert B == 1
                for b in range(B):
                    poly = cal_polygon_contour(
                        x=pos[b, :, 0].cpu().numpy(),
                        y=pos[b, :, 1].cpu().numpy(),
                        theta=head[b].cpu().numpy(),
                        width=shape[b, :, 1].cpu().numpy(),
                        length=shape[b, :, 0].cpu().numpy()
                    )
                    last_poly = poly[-1]
                    last_poly = Polygon(last_poly)
                    coll = False
                    for i in range(len(poly) - 1):
                        poly2 = Polygon(poly[i])
                        if last_poly.intersects(poly2):
                            coll = True
                            coll_b = b
                            break
                if coll:
                    print("Collision happens at batch {}, repeat the generation.".format(coll_b))
                    should_repeat = True
                else:
                    should_repeat = False
                    break

                self.scenestreamer_tokens = copy.deepcopy(tmp_scenestreamer_tokens)
                self.step_info_dict = copy.deepcopy(tmp_step_info_dict)
                intra_step = copy.deepcopy(tmp_intra_step)
                step_info_dict = self.step_info_dict
                scenestreamer_tokens = self.scenestreamer_tokens

            # print("STEP {}, Generating agent {} with teacher forcing: {}. Agent type {}, position: {}, shape: {}".format(
            #     self.current_step, agent_index, teacher_forcing_this_agent, agent_type[0, agent_index],
            #     as_position[0].cpu().numpy(), step_info_dict["agent_shape"][0, agent_index].cpu().numpy()
            # ))
        # print("STEP {}, Generating agent {}. SceneStreamer len {}.".format(
        #     self.current_step, agent_index, self.scenestreamer_tokens.seq_len
        # ))

        # Finally, input the sequence_eos_id.
        intra_step += 1
        assert intra_step == G - 1, (intra_step, G, G - 1)
        tg_token = model.prepare_trafficgen_single_token(
            tg_action=torch.full((B, 1), model.trafficgen_sequence_eos_id, device=device),
            tg_type=agent_type_for_trafficgen[:, 0, intra_step:intra_step + 1],
            tg_agent_id=torch.full((B, 1), -1, device=device),
            tg_intra_step=torch.full((B, 1), intra_step, device=device),
            tg_feat=torch.full((B, 1, 8), 0.0, device=device),
        )
        tg_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1,
                         :scenestreamer_tokens.seq_len + 1]
        scenestreamer_tokens.add(
            token=tg_token,
            position=torch.full((B, 1, 2), 0, device=device),
            heading=torch.full((B, 1), 0, device=device),
            valid_mask=torch.full((B, 1), True, device=device, dtype=torch.bool),
            width=torch.full((B, 1), 0.0, device=device),
            length=torch.full((B, 1), 0.0, device=device),
            causal_mask=tg_causal_mask,
            current_step=current_step,
            require_relation=torch.full((B, 1), False, device=device, dtype=torch.bool),
            force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + 1, :scenestreamer_tokens.seq_len + 1]
        )

        # The only thing need to be updated by non-teacher_forcing TG is the destination:
        # step_info_dict["agent_destination"] = torch.stack(agent_destination_list, dim=1)
        # step_info_dict["agent_destination_position"] = torch.stack(agent_destination_pos_list, dim=1)

        self.step_info_dict = step_info_dict
        self.state = self.STATE_TRAFFICGEN_DONE

    def _step_generate_motion(self, *, teacher_forcing, allow_newly_added, teacher_forcing_sdc):
        assert self.state in [self.STATE_TRAFFICGEN_DONE, self.STATE_TRAFFICGEN_SKIPPED]

        model = self.model
        data_dict = self.raw_data_dict
        scenestreamer_tokens = self.scenestreamer_tokens
        step_info_dict = self.step_info_dict
        current_step = self.current_step
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask
        sampling_method = self.sampling_method
        temperature = self.temperature
        topp = self.topp
        keep_output_token = self.keep_output_token

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
            motion_input_dict["decoder/input_action"] = data_dict["decoder/input_action"][:,
                                                        current_step:current_step + 1]
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
            causal_mask=all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + N,
                        :scenestreamer_tokens.seq_len + N],
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
        output_dict = scenestreamer_tokens.call_model_with_cache(keep_output_token=keep_output_token)
        all_token = output_dict["model/all_token"]
        motion_token = all_token[:, -scenestreamer_tokens.N:]
        if model.motion_prenorm is not None:
            motion_token = model.motion_prenorm(motion_token)
        output_token = model.motion_head(motion_token)

        # ===== Post-process the data =====
        selected_action, sampling_info = scenestreamer_motion.sample_action(
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

        if teacher_forcing_sdc:
            teacher_forcing_action = data_dict["decoder/target_action"][:, self.current_step].clone().reshape(B, N)
            sdc_index = data_dict["decoder/sdc_index"]
            assert (sdc_index == 0).all()
            # if not (data_dict["decoder/target_action_valid_mask"][:, self.current_step, 0] == True).all():
            #     print(111111)
            #     "teacher forcing SDC should always be valid in data."
            selected_action[:, 0] = teacher_forcing_action[:, 0]
            step_info_dict["agent_valid_mask"][:, 0] = data_dict["decoder/target_action_valid_mask"][:,
                                                       self.current_step, 0].clone()
            agent_position[:, 0] = data_dict["decoder/modeled_agent_position"][:, self.current_step, 0].clone()
            agent_heading[:, 0] = data_dict["decoder/modeled_agent_heading"][:, self.current_step, 0].clone()
            agent_velocity[:, 0] = data_dict["decoder/modeled_agent_velocity"][:, self.current_step, 0].clone()

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

        step_info_dict["agent_position"] = new_agent_position.clone()
        step_info_dict["agent_heading"] = new_agent_heading.clone()
        step_info_dict["agent_velocity"] = new_agent_velocity.clone()
        step_info_dict["motion_input_action"] = selected_action.reshape(B, N).clone()

        if allow_newly_added:
            new_agent_valid_mask = (
                    data_dict["decoder/input_action_valid_mask"][:, current_step + 1] & (
                ~step_info_dict["agent_valid_mask"])
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

                step_info_dict["agent_position"] = current_pos.clone()
                step_info_dict["agent_heading"] = current_heading.clone()
                step_info_dict["agent_velocity"] = current_vel.clone()
                step_info_dict["agent_valid_mask"] = current_valid_mask.clone()
                step_info_dict["motion_input_action"] = torch.where(new_agent_valid_mask, new_action,
                                                                    step_info_dict["motion_input_action"]).clone()

        # # TODO: evict agents that moving out of the map (useful in SceneStreamer)
        # if evict_agent:
        #     next_step_data_dict, info_dict = evict_agents_function(
        #         data_dict=data_dict,
        #         step_data_dict=next_step_data_dict,
        #         step_info_dict=info_dict,
        #         remove_static_agent=remove_static_agent,
        #         remove_out_of_map_agent=remove_out_of_map_agent
        #     )

        tmp_action = step_info_dict["motion_input_action"].clone()
        tmp_valid_mask = agent_valid_mask.clone()
        tmp_valid_mask[tmp_action == -1] = False
        tmp_valid_mask[tmp_action == MOTION_START_ACTION] = False
        tmp_action[tmp_action == -1] = 0
        tmp_action[tmp_action == MOTION_START_ACTION] = 0
        log_prob = sampling_info["dist"].log_prob(tmp_action)
        step_info_dict["motion_input_action_log_prob"] = (log_prob * tmp_valid_mask).clone()

        self.step_info_dict = step_info_dict
        self.state = self.STATE_MOTION_DONE

    def _step_generate_trafficlight(self, teacher_forcing=False):
        step_info_dict = self.step_info_dict
        data_dict = self.raw_data_dict
        current_step = self.current_step
        model = self.model
        scenestreamer_tokens = self.scenestreamer_tokens
        keep_output_token = False
        all_token_casual_mask = self.all_token_casual_mask
        all_force_mask = self.all_force_mask

        assert self.state in [self.STATE_START, self.STATE_MOTION_DONE], "State should be either start or motion done"

        tl_input_dict = {
            # no time dim:
            "encoder/traffic_light_position": data_dict["encoder/traffic_light_position"][..., :2],
            "encoder/traffic_light_heading": data_dict["encoder/traffic_light_heading"],
            "encoder/traffic_light_map_id": data_dict["encoder/traffic_light_map_id"],
        }

        if teacher_forcing:
            tl_input_dict.update({
                "encoder/traffic_light_state": data_dict["encoder/traffic_light_state"][:,
                                               current_step:current_step + 1],
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
        N = self.N
        if model.no_tg:
            G = 0
        else:
            G = get_num_tg(N)

        # ===== token =====
        if scenestreamer_tokens is None:
            tl_causal_mask = all_token_casual_mask[:, :L, :L]
            steps = torch.full((B, L), current_step, dtype=torch.long, device=tl_position.device)
            scenestreamer_tokens = scenestreamer_motion.SceneStreamerTokens(
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
            tl_causal_mask = all_token_casual_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + L,
                             :scenestreamer_tokens.seq_len + L]
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
                force_mask=all_force_mask[:, scenestreamer_tokens.seq_len:scenestreamer_tokens.seq_len + L,
                           :scenestreamer_tokens.seq_len + L],
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
                                                    current_step + 1:current_step + 2].clone()
            step_info_dict["traffic_light_valid_mask"] = data_dict["encoder/traffic_light_valid_mask"][:,
                                                         current_step + 1:current_step + 2].clone()

        else:
            if tl_valid_mask.any():
                output_dict = scenestreamer_tokens.call_model_with_cache(keep_output_token=keep_output_token)

                # ===== Post-process the data =====
                traffic_light_token = output_dict["model/all_token"][:, -L:]
                traffic_light_token = model.traffic_light_prenorm(traffic_light_token)
                traffic_light_token = model.traffic_light_head(traffic_light_token)
                # output_dict["model/traffic_light_logit"] = traffic_light_token

                tl_state, _ = scenestreamer_motion.sample_action(traffic_light_token, sampling_method="softmax")
                step_info_dict["traffic_light_state"] = tl_state.reshape(B, 1, L).clone()
            step_info_dict["traffic_light_valid_mask"] = tl_valid_mask.reshape(B, 1, L).clone()

        self.step_info_dict = step_info_dict
        self.scenestreamer_tokens = scenestreamer_tokens
        self.state = self.STATE_TRAFFICLIGHT_DONE

    @property
    def B(self):
        return self.raw_data_dict["decoder/input_action"].shape[0]

    @property
    def N(self):
        return self.raw_data_dict["decoder/input_action"].shape[2]

    @property
    def L(self):
        return self.raw_data_dict["encoder/traffic_light_state"].shape[2]

    @property
    def G(self):
        return get_num_tg(self.N)

    def generate_scenestreamer_motion(self, *, progress_bar=False, num_decode_steps=19, teacher_forcing_sdc=False):
        """
        This is the WOSAC generate where no initial state is generated.
        """
        model = self.model
        if progress_bar:
            pbar = tqdm.trange(num_decode_steps, desc="Decoding Step")
        else:
            pbar = range(num_decode_steps)

        data_dict = self.raw_data_dict
        valid_mask = [data_dict["decoder/input_action_valid_mask"][:, :1].clone()]
        pos = [data_dict["decoder/modeled_agent_position"][:, :1].clone()]
        head = [data_dict["decoder/modeled_agent_heading"][:, :1].clone()]
        vel = [data_dict["decoder/modeled_agent_velocity"][:, :1].clone()]
        dest = []
        dest_pos = []
        tl_state = []
        log_prob = []
        action = []
        B, N, G, L = self.B, self.N, self.G, self.L

        # TODO =========================================
        # TODO =========================================
        # TODO =========================================
        # TODO =========================================
        teacher_forcing_dest = True

        no_tg = model.no_tg
        for decoding_step in pbar:
            self.current_step = decoding_step
            if model.no_tg is False:
                if decoding_step % TG_SKIP_STEP == 0:
                    no_tg = False
                else:
                    no_tg = True
            # TODO: not hardcoded.
            if decoding_step < 2:
                teacher_forcing_motion = True
                allow_newly_added = True
                teacher_forcing_tl = True
            else:
                teacher_forcing_motion = False
                allow_newly_added = False
                teacher_forcing_tl = False
            if decoding_step <= 2:
                teacher_forcing_tg = True
            else:
                teacher_forcing_tg = False

            # ===== Traffic light =====
            self._step_generate_trafficlight(teacher_forcing=teacher_forcing_tl)
            if self.step_info_dict["traffic_light_state"].shape[1] > 0:
                tl_state.append(self.step_info_dict["traffic_light_state"].reshape(B, 1, L))

            # ===== Trafficgen =====
            if no_tg:
                if teacher_forcing_tg:
                    current_step = decoding_step
                    self.step_info_dict["agent_valid_mask"] = \
                        data_dict["decoder/input_action_valid_mask"][:, current_step].clone()
                    self.step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:,
                                                            current_step].clone()
                    self.step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:,
                                                           current_step].clone()
                    self.step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:,
                                                            current_step].clone()
                    self.step_info_dict["agent_type"] = data_dict["decoder/agent_type"].clone()
                    self.step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"].clone()
                    self.step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"].clone()
                self.state = self.STATE_TRAFFICGEN_SKIPPED
            else:
                self._step_generate_trafficgen_no_agent_state(
                    teacher_forcing_from_gt=teacher_forcing_tg, teacher_forcing_dest=teacher_forcing_dest
                )
                # dest.append(self.step_info_dict["agent_destination"].reshape(B, 1, N))
                # dest_pos.append(self.step_info_dict["agent_destination_position"].reshape(B, 1, N, 2))

            # ===== Motion =====
            self._step_generate_motion(
                teacher_forcing=teacher_forcing_motion,
                allow_newly_added=allow_newly_added,
                teacher_forcing_sdc=teacher_forcing_sdc
            )
            pos.append(self.step_info_dict["agent_position"].reshape(B, 1, N, 2).clone())
            head.append(self.step_info_dict["agent_heading"].reshape(B, 1, N).clone())
            vel.append(self.step_info_dict["agent_velocity"].reshape(B, 1, N, 2).clone())
            valid_mask.append(self.step_info_dict["agent_valid_mask"].reshape(B, 1, N).clone())
            log_prob.append(self.step_info_dict["motion_input_action_log_prob"].reshape(B, 1, N).clone())
            action.append(self.step_info_dict["motion_input_action"].reshape(B, 1, N).clone())

        assert self.all_token_casual_mask.shape[1] == self.all_token_casual_mask.shape[
            2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_token_casual_mask.shape, self.scenestreamer_tokens.seq_len
            )
        )
        assert self.all_force_mask.shape[1] == self.all_force_mask.shape[2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_force_mask.shape, self.scenestreamer_tokens.seq_len
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
        # valid_mask = valid_mask[:-1]
        valid_mask = torch.cat(valid_mask, dim=1)

        tl_state = torch.cat(tl_state, dim=1)

        log_prob = torch.cat(log_prob, dim=1)

        # ===== Interpolate the output =====
        output_dict = {}

        output_dict, _ = scenestreamer_motion.interpolate_autoregressive_output(
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
        scores = (log_prob * valid_mask[:, :-1])[:, 2:].sum(1)

        # from scenestreamer.models import relation
        # scenestreamer_tokens = self.scenestreamer_tokens
        # knn = self.model.config.SCENESTREAMER_ATTENTION_KNN
        # max_distance = self.model.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE
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
            "decoder/current_agent_shape": self.step_info_dict["agent_shape"],
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

        output_dict["scenestreamer_tokens"] = self.scenestreamer_tokens
        return output_dict

    def generate_scenestreamer_motion_with_densified_scenario(self, *,
                                                       veh_ratio,
                                                       ped_ratio,
                                                       num_new_agents,
                                                       progress_bar=False, num_decode_steps=19,
                                                       teacher_forcing_sdc=False):
        """
        This is the WOSAC generate where no initial state is generated.
        """
        model = self.model
        if progress_bar:
            pbar = tqdm.trange(num_decode_steps, desc="Decoding Step")
        else:
            pbar = range(num_decode_steps)

        data_dict = self.raw_data_dict

        # Should not prepare the data_dict here.
        # valid_mask = [data_dict["decoder/input_action_valid_mask"][:, :1].clone()]
        # pos = [data_dict["decoder/modeled_agent_position"][:, :1].clone()]
        # head = [data_dict["decoder/modeled_agent_heading"][:, :1].clone()]
        # vel = [data_dict["decoder/modeled_agent_velocity"][:, :1].clone()]
        valid_mask = []
        pos = []
        head = []
        vel = []
        agent_shape = []

        dest = []
        dest_pos = []
        tl_state = []
        log_prob = []
        action = []
        B, N, G, L = self.B, self.N, self.G, self.L

        # TODO: reset the scenestreamer tokens
        N = num_new_agents
        G = get_num_tg(N)
        all_token_casual_mask = model._build_all_tokens_mask(
            B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
        ).to(data_dict["decoder/input_action"].device)
        self.all_token_casual_mask = all_token_casual_mask
        all_force_mask = model._build_all_force_mask(
            B=B, T=num_decode_steps, num_tl=L, num_tg=G, num_motion=N
        ).to(data_dict["decoder/input_action"].device)
        self.all_force_mask = all_force_mask

        # TODO =========================================
        # TODO =========================================
        # TODO =========================================
        # TODO =========================================
        teacher_forcing_dest = True

        no_tg = model.no_tg
        for decoding_step in pbar:
            self.current_step = decoding_step
            if model.no_tg is False:
                if decoding_step % TG_SKIP_STEP == 0:
                    no_tg = False
                else:
                    no_tg = True

            # TODO: just disable TF for now.

            # if decoding_step < 2:
            # if decoding_step < 1:
            #     teacher_forcing_motion = True
            #     allow_newly_added = True
            # else:
            teacher_forcing_motion = False
            allow_newly_added = False

            if decoding_step <= 0:
                teacher_forcing_tg = True
                teacher_forcing_tl = True
            else:
                teacher_forcing_tg = False
                teacher_forcing_tl = False

            # ===== Traffic light =====
            self._step_generate_trafficlight(teacher_forcing=teacher_forcing_tl)
            if self.step_info_dict["traffic_light_state"].shape[1] > 0:
                tl_state.append(self.step_info_dict["traffic_light_state"].reshape(B, 1, L))

            # ===== Trafficgen =====
            if no_tg:
                if teacher_forcing_tg:
                    current_step = decoding_step
                    self.step_info_dict["agent_valid_mask"] = \
                        data_dict["decoder/input_action_valid_mask"][:, current_step].clone()
                    self.step_info_dict["agent_position"] = data_dict["decoder/modeled_agent_position"][:,
                                                            current_step].clone()
                    self.step_info_dict["agent_heading"] = data_dict["decoder/modeled_agent_heading"][:,
                                                           current_step].clone()
                    self.step_info_dict["agent_velocity"] = data_dict["decoder/modeled_agent_velocity"][:,
                                                            current_step].clone()
                    self.step_info_dict["agent_type"] = data_dict["decoder/agent_type"].clone()
                    self.step_info_dict["agent_shape"] = data_dict["decoder/current_agent_shape"].clone()
                    self.step_info_dict["agent_id"] = data_dict["encoder/modeled_agent_id"].clone()
                self.state = self.STATE_TRAFFICGEN_SKIPPED
            else:
                self._step_generate_trafficgen_densified_agent_state(
                    teacher_forcing_from_gt=teacher_forcing_tg, teacher_forcing_dest=teacher_forcing_dest,
                    veh_ratio=veh_ratio,
                    ped_ratio=ped_ratio,
                    num_new_agents=num_new_agents,
                )
                # dest.append(self.step_info_dict["agent_destination"].reshape(B, 1, N))
                # dest_pos.append(self.step_info_dict["agent_destination_position"].reshape(B, 1, N, 2))

                if self.current_step == 0:
                    pos.append(self.step_info_dict["agent_position"].reshape(B, 1, N, 2).clone())
                    head.append(self.step_info_dict["agent_heading"].reshape(B, 1, N).clone())
                    vel.append(self.step_info_dict["agent_velocity"].reshape(B, 1, N, 2).clone())
                    valid_mask.append(self.step_info_dict["agent_valid_mask"].reshape(B, 1, N).clone())
                    agent_shape.append(self.step_info_dict["agent_shape"].reshape(B, N, 3).clone())

            # ===== Motion =====
            assert teacher_forcing_sdc is False
            self._step_generate_motion(
                teacher_forcing=teacher_forcing_motion,
                allow_newly_added=allow_newly_added,
                teacher_forcing_sdc=teacher_forcing_sdc
            )
            pos.append(self.step_info_dict["agent_position"].reshape(B, 1, N, 2).clone())
            head.append(self.step_info_dict["agent_heading"].reshape(B, 1, N).clone())
            vel.append(self.step_info_dict["agent_velocity"].reshape(B, 1, N, 2).clone())
            valid_mask.append(self.step_info_dict["agent_valid_mask"].reshape(B, 1, N).clone())
            log_prob.append(self.step_info_dict["motion_input_action_log_prob"].reshape(B, 1, N).clone())
            action.append(self.step_info_dict["motion_input_action"].reshape(B, 1, N).clone())

        assert self.all_token_casual_mask.shape[1] == self.all_token_casual_mask.shape[
            2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_token_casual_mask.shape, self.scenestreamer_tokens.seq_len
            )
        )
        assert self.all_force_mask.shape[1] == self.all_force_mask.shape[2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_force_mask.shape, self.scenestreamer_tokens.seq_len
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
        # valid_mask = valid_mask[:-1]
        valid_mask = torch.cat(valid_mask, dim=1)

        tl_state = torch.cat(tl_state, dim=1)

        log_prob = torch.cat(log_prob, dim=1)

        # ===== Interpolate the output =====
        output_dict = {}

        output_dict, _ = scenestreamer_motion.interpolate_autoregressive_output(
            data_dict=output_dict,
            agent_heading=head,
            agent_position=pos,
            agent_velocity=vel,
            agent_destination=dest,
            agent_destination_position=dest_pos,
            input_valid_mask=valid_mask,
            num_skipped_steps=model.motion_tokenizer.num_skipped_steps,
            num_decoded_steps=num_decode_steps,
            agent_shape=agent_shape,
            teacher_forcing_sdc=teacher_forcing_sdc
        )

        assert log_prob.shape == (B, 19, N)
        scores = (log_prob * valid_mask[:, :-1])[:, 2:].sum(1)

        # from scenestreamer.models import relation
        # scenestreamer_tokens = self.scenestreamer_tokens
        # knn = self.model.config.SCENESTREAMER_ATTENTION_KNN
        # max_distance = self.model.config.SCENESTREAMER_ATTENTION_MAX_DISTANCE
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
            "decoder/current_agent_shape": self.step_info_dict["agent_shape"],
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

        output_dict["scenestreamer_tokens"] = self.scenestreamer_tokens
        return output_dict

    def generate_scenestreamer_initial_state_and_motion(self, *, progress_bar=False, num_decode_steps=19,
                                                 teacher_forcing_sdc=False):
        """
        This is the WOSAC generate where no initial state is generated.
        """
        model = self.model
        if progress_bar:
            pbar = tqdm.trange(num_decode_steps, desc="Decoding Step")
        else:
            pbar = range(num_decode_steps)
        data_dict = self.raw_data_dict

        # Don't do this in this task...
        # valid_mask = [data_dict["decoder/input_action_valid_mask"][:, :1].clone()]
        # pos = [data_dict["decoder/modeled_agent_position"][:, :1].clone()]
        # head = [data_dict["decoder/modeled_agent_heading"][:, :1].clone()]
        # vel = [data_dict["decoder/modeled_agent_velocity"][:, :1].clone()]
        valid_mask = []
        pos = []
        head = []
        vel = []

        dest = []
        dest_pos = []
        tl_state = []
        log_prob = []
        action = []
        agent_shape = []
        B, N, G, L = self.B, self.N, self.G, self.L
        no_tg = model.no_tg
        assert no_tg is False
        for decoding_step in pbar:
            self.current_step = decoding_step
            if decoding_step % TG_SKIP_STEP == 0:
                no_tg = False
            else:
                no_tg = True
            if decoding_step < 2:
                teacher_forcing_tl = True
            else:
                teacher_forcing_tl = False
            allow_newly_added = False
            teacher_forcing_motion = False

            # ===== Traffic light =====
            self._step_generate_trafficlight(teacher_forcing=teacher_forcing_tl)
            if self.step_info_dict["traffic_light_state"].shape[1] > 0:
                tl_state.append(self.step_info_dict["traffic_light_state"].reshape(B, 1, L))

            # ===== Trafficgen =====
            if no_tg:
                self.state = self.STATE_TRAFFICGEN_SKIPPED
            else:
                if self.current_step == 0:
                    self._step_generate_trafficgen_with_agent_state(teacher_forcing_from_gt=True)
                    assert self.step_info_dict["agent_valid_mask"].shape == (B, N)
                    assert (self.step_info_dict["agent_valid_mask"]).all()
                    self.step_info_dict["motion_input_action"] = torch.full(
                        (B, N), MOTION_START_ACTION, device=self.device
                    )
                    pos.append(self.step_info_dict["agent_position"].reshape(B, 1, N, 2).clone())
                    head.append(self.step_info_dict["agent_heading"].reshape(B, 1, N).clone())
                    vel.append(self.step_info_dict["agent_velocity"].reshape(B, 1, N, 2).clone())
                    valid_mask.append(self.step_info_dict["agent_valid_mask"].reshape(B, 1, N).clone())

                    agent_shape.append(self.step_info_dict["agent_shape"].reshape(B, N, 3).clone())
                else:
                    self._step_generate_trafficgen_no_agent_state(
                        teacher_forcing_from_gt=False,
                        generate_agent_states=False
                    )
                # dest.append(self.step_info_dict["agent_destination"].reshape(B, 1, N))
                # dest_pos.append(self.step_info_dict["agent_destination_position"].reshape(B, 1, N, 2))

            # ===== Motion =====
            self._step_generate_motion(
                teacher_forcing=teacher_forcing_motion,
                allow_newly_added=allow_newly_added,
                teacher_forcing_sdc=teacher_forcing_sdc,
            )
            # print("MOTION Step {}, scenestreamer len {}".format(
            #     self.current_step, self.scenestreamer_tokens.seq_len
            # ))
            pos.append(self.step_info_dict["agent_position"].reshape(B, 1, N, 2).clone())
            head.append(self.step_info_dict["agent_heading"].reshape(B, 1, N).clone())
            vel.append(self.step_info_dict["agent_velocity"].reshape(B, 1, N, 2).clone())
            valid_mask.append(self.step_info_dict["agent_valid_mask"].reshape(B, 1, N).clone())
            log_prob.append(self.step_info_dict["motion_input_action_log_prob"].reshape(B, 1, N).clone())
            action.append(self.step_info_dict["motion_input_action"].reshape(B, 1, N).clone())

        assert self.all_token_casual_mask.shape[1] == self.all_token_casual_mask.shape[
            2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_token_casual_mask.shape, self.scenestreamer_tokens.seq_len
            )
        )
        assert self.all_force_mask.shape[1] == self.all_force_mask.shape[2] == self.scenestreamer_tokens.seq_len, (
            "{} vs {}".format(
                self.all_force_mask.shape, self.scenestreamer_tokens.seq_len
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
        # valid_mask = valid_mask[:-1]
        valid_mask = torch.cat(valid_mask, dim=1)
        tl_state = torch.cat(tl_state, dim=1)
        log_prob = torch.cat(log_prob, dim=1)

        # ===== Interpolate the output =====
        output_dict = {}
        output_dict, _ = scenestreamer_motion.interpolate_autoregressive_output(
            data_dict=output_dict,
            agent_heading=head,
            agent_position=pos,
            agent_velocity=vel,
            agent_destination=dest,
            agent_destination_position=dest_pos,
            input_valid_mask=valid_mask,
            num_skipped_steps=model.motion_tokenizer.num_skipped_steps,
            num_decoded_steps=num_decode_steps,
            agent_shape=agent_shape,
            teacher_forcing_sdc=teacher_forcing_sdc,
            sdc_index=data_dict["decoder/sdc_index"],
        )

        assert log_prob.shape == (B, 19, N)
        scores = (log_prob * valid_mask[:, :-1])[:, 2:].sum(1)

        output_dict.update({

            # TODO: Not accumulated across steps? now is the last.
            "decoder/current_agent_shape": self.step_info_dict["agent_shape"],
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

        output_dict["scenestreamer_tokens"] = self.scenestreamer_tokens
        return output_dict

    def generate_scenestreamer_initial_state(self, *, progress_bar=False, num_decode_steps=19):
        """
        This is the WOSAC generate where no initial state is generated.
        """
        data_dict = self.raw_data_dict

        # Hardcode here, process the data and set "current_step" to 10.
        assert data_dict['metadata/current_time_index'].item() == 10
        new_data_dict = {}
        for k in data_dict:
            if ("encoder/traffic_light_" in k) or (k == "decoder/input_action_valid_mask") or \
                    ("decoder/modeled_agent_" in k):
                new_data_dict[k] = data_dict[k][:, 10:11]
        data_dict = new_data_dict

        tl_state = []
        B, N, G, L = self.B, self.N, self.G, self.L
        teacher_forcing_tl = True
        assert self.model.no_tg is False, "SceneStreamer should be used with trafficgen"
        # ===== Traffic light =====
        self._step_generate_trafficlight(teacher_forcing=teacher_forcing_tl)
        if self.step_info_dict["traffic_light_state"].shape[1] > 0:
            tl_state.append(self.step_info_dict["traffic_light_state"].reshape(B, 1, L))
        # ===== Trafficgen =====
        self._step_generate_trafficgen_with_agent_state(
            teacher_forcing_from_gt=True, teacher_forcing_dest=111111,
        )
        # Do some postprocessing
        step_data_dict = self.step_info_dict
        data_dict.update(
            {
                "decoder/modeled_agent_position_for_trafficgen": step_data_dict["agent_position"].clone(),
                "decoder/modeled_agent_heading_for_trafficgen": step_data_dict["agent_heading"].clone(),
                "decoder/modeled_agent_velocity_for_trafficgen": step_data_dict["agent_velocity"].clone(),
                "decoder/current_agent_shape_for_trafficgen": step_data_dict["agent_shape"].clone(),
                "decoder/agent_type_for_trafficgen": step_data_dict["agent_type"].clone(),
                "decoder/input_action_valid_mask_for_trafficgen": step_data_dict["agent_valid_mask"].clone(),
            }
        )
        assert step_data_dict["agent_valid_mask"].all()

        # from scenestreamer.infer.initial_state import convert_initial_states_as_motion_data
        # data_dict = convert_initial_states_as_motion_data(data_dict)

        return data_dict


def plot_initial_state(data_dict, save_path, draw_line=False, draw_text=True):
    from scenestreamer.gradio_ui.plot import (
        BOUNDARY,
        EGO_FONT_SIZE,
        MODELED_FONT_SIZE,
        NON_EGO_FONT_SIZE,
        _plot_map,
        _plot_traffic_light,
        draw_trajectory,
        get_limit,
    )
    import seaborn as sns
    import matplotlib.pyplot as plt
    import PIL

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    agent_pos = data_dict["decoder/agent_position"][:, :, :2]  # (91, N, 2)
    agent_heading = data_dict["decoder/agent_heading"]  # (91, N, 2)
    agent_velocity = data_dict["decoder/agent_velocity"]  # (91, N, 2)
    agent_shape = data_dict["decoder/agent_shape"]  # (91, N, 2)
    agent_mask = data_dict["decoder/agent_valid_mask"]
    ego_agent_id = data_dict['decoder/sdc_index']

    _plot_map(data_dict, ax, dont_draw_lane=True)

    _plot_traffic_light(data_dict, ax)

    T, N, _ = agent_pos.shape

    modeled_agents_indicies = np.concatenate([data_dict["decoder/object_of_interest_id"], np.atleast_1d(ego_agent_id)])

    # cmap = sns.color_palette("colorblind", n_colors=N)
    cmap = sns.color_palette("crest_r", as_cmap=False, n_colors=N)
    cmap_cbar = sns.color_palette("crest_r", as_cmap=True, n_colors=N)

    plotted_count = 0
    draw_trajectory(
        ax=ax,
        pos=agent_pos[:, ego_agent_id],
        heading=agent_heading[:, ego_agent_id],
        width=agent_shape[:, ego_agent_id, 1],
        length=agent_shape[:, ego_agent_id, 0],
        mask=agent_mask[:, ego_agent_id],
        fill_color=cmap[0],
        traj_kwargs=dict(),
        contour_kwargs=dict(
            edgecolor="k",
            linewidth=0.1,
            fill=False,
        ),
        text="{}-SDC".format(str(ego_agent_id)),
        fontsize=EGO_FONT_SIZE,
        draw_line=draw_line,
        draw_text=True,
    )
    plotted_count += 1

    for agent_ind in range(N):
        if agent_ind == ego_agent_id:
            continue
        if agent_ind in modeled_agents_indicies:
            text = "{}-OOI".format(str(agent_ind))
            fontsize = MODELED_FONT_SIZE
        else:
            text = str(agent_ind)
            fontsize = NON_EGO_FONT_SIZE
        draw_trajectory(
            ax=ax,
            pos=agent_pos[:, agent_ind],
            heading=agent_heading[:, agent_ind],
            width=agent_shape[:, agent_ind, 1],
            length=agent_shape[:, agent_ind, 0],
            mask=agent_mask[:, agent_ind],
            fill_color=cmap[plotted_count],
            traj_kwargs=dict(),
            contour_kwargs=dict(
                edgecolor="k",
                linewidth=0.1,
                fill=False,
            ),
            text=text,
            fontsize=fontsize,
            draw_line=draw_line,
            draw_text=False
        )
        plotted_count += 1

    if "vis/map_feature" in data_dict:
        map_pos = data_dict["vis/map_feature"][:, :, :2][data_dict["encoder/map_feature_valid_mask"]]
    else:
        map_pos = data_dict["encoder/map_position"][..., :2][data_dict["encoder/map_valid_mask"]]
    ret = get_limit(agent_pos=agent_pos[agent_mask], map_pos=map_pos)

    xmin, xmax, ymin, ymax = ret["xmin"], ret["xmax"], ret["ymin"], ret["ymax"]

    ax.set_xlim(xmin - BOUNDARY, xmax + BOUNDARY)
    ax.set_ylim(ymin - BOUNDARY, ymax + BOUNDARY)
    ax.set_aspect(1)

    # turn on color bar
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    norm = colors.Normalize(vmin=0, vmax=N - 1)
    sm = cm.ScalarMappable(cmap=cmap_cbar, norm=norm)
    sm.set_array([])  # dummy array for compatibility
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.invert_yaxis()  # ⬅️ Flip the colorbar
    # cbar = plt.colorbar(cmap_cbar, ax=ax)

    fig.tight_layout(pad=0.05)
    fig.canvas.draw()

    # plt.show()
    ret = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    fig.savefig(save_path)

    plt.close(fig)
