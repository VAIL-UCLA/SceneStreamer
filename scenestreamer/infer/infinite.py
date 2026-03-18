"""
This module reimplements the autoregressive motion generation process.
"""

import copy
import os
import numpy as np

import torch

from scenestreamer.tokenization.motion_tokenizers import interpolate, interpolate_heading
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils
from scenestreamer.infer.motion import encode_scene
from scenestreamer.tokenization.trafficgen_tokenizers import TrafficGenTokenizer
from scenestreamer.tokenization.motion_tokenizers import START_ACTION as MOTION_START_ACTION
from scenestreamer.tokenization.biycle_tokenizer import get_relative_velocity

from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
import torch

from scenestreamer.infer.initial_state import (
    generate_initial_state,
    convert_initial_states_as_motion_data,
    decode_one_step_initial_state,
)
from scenestreamer.infer.motion import (
    decode_one_step,
    randomize_agent_id,
    encode_scene,
    interpolate_autoregressive_output,
)


def _verbose_densify_logs() -> bool:
    return os.environ.get("SCENESTREAMER_VERBOSE_DENSIFY") == "1"


@torch.no_grad()
def generate_densified_scenario(
    *,
    data_dict,
    model,
    force_add=False,
    max_agents=128,
    num_decode_steps=None,
    discard_low_speed_agent=False,
    remove_static_agent=False,
    exclude_sdc_neighborhood=False
):
    # Motion generation
    agent_unique_id = data_dict['decoder/agent_id'][0]  # (N,)
    interpolation = True
    remove_out_of_map_agent = True
    tokenizer = model.tokenizer
    temperature = model.config.SAMPLING.TEMPERATURE
    topp = model.config.SAMPLING.TOPP
    sampling_method = model.config.SAMPLING.SAMPLING_METHOD
    B, T_input, N = data_dict["decoder/input_action"].shape
    agent_pos = data_dict["decoder/agent_position"][:, ::tokenizer.num_skipped_steps]
    agent_heading = data_dict["decoder/agent_heading"][:, ::tokenizer.num_skipped_steps]
    agent_valid_mask = data_dict["decoder/agent_valid_mask"][:, ::tokenizer.num_skipped_steps]
    agent_velocity = data_dict["decoder/agent_velocity"][:, ::tokenizer.num_skipped_steps]
    B, T_full, N, _ = agent_pos.shape
    gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
    assert agent_pos.ndim == 4
    gt_input_action = data_dict["decoder/input_action"].clone()
    data_dict, _ = randomize_agent_id(data_dict=data_dict, model=model)
    step_data_dict = dict(
        input_step=torch.arange(1).to(gt_input_action.device),
        input_action=gt_input_action[:, :1].clone(),
        input_action_valid_mask=data_dict["decoder/input_action_valid_mask"][:, :1].clone(),
        agent_position=data_dict["decoder/modeled_agent_position"][:, :1].clone(),
        agent_heading=data_dict["decoder/modeled_agent_heading"][:, :1].clone(),
        agent_velocity=data_dict["decoder/modeled_agent_velocity"][:, :1].clone(),  # TODO: Remove this?
        agent_valid_mask=data_dict["decoder/input_action_valid_mask"][:, :1].clone(),
        agent_delta=data_dict["decoder/modeled_agent_delta"][:, :1].clone(),
        cache=None,
        agent_id=data_dict["decoder/randomized_modeled_agent_id"],
        agent_type=data_dict["decoder/agent_type"],
        agent_shape=data_dict["decoder/current_agent_shape"],
        decode_step=0,
    )
    max_unique_id = agent_unique_id.max().item()
    data_dict, _ = encode_scene(data_dict=data_dict, model=model)

    # Densify the scenario
    num_agents_being_added = max(0, max_agents - (max_unique_id + 1))
    step_data_dict, _, agent_unique_id, max_unique_id = scenestreamer_step(
        data_dict=data_dict,
        motion_step_data_dict=step_data_dict,
        motion_decode_one_step_info={},
        evicted_agent_mask=None,
        model=model,
        agent_unique_id=agent_unique_id,
        num_agents_being_added=num_agents_being_added,
        max_agents=max_agents,
        discard_low_speed_agent=discard_low_speed_agent,
        exclude_sdc_neighborhood=exclude_sdc_neighborhood,
        max_unique_id=max_unique_id,
    )

    output_logit_list = []
    output_action_list = []
    agent_unique_id_list = [agent_unique_id.clone()]
    input_action_valid_mask_list = [step_data_dict["input_action_valid_mask"]]
    pos = [step_data_dict["agent_position"]]
    head = [step_data_dict["agent_heading"]]
    vel = [step_data_dict["agent_velocity"]]
    agent_type = [step_data_dict["agent_type"]]
    agent_shape = [step_data_dict["agent_shape"]]
    for decode_step in range(num_decode_steps):
        teacher_forcing_valid_mask = None
        teacher_forcing_action = None
        next_state_data_dict, decode_one_step_info = decode_one_step(
            data_dict=data_dict,
            model=model,
            sampling_method=sampling_method,
            temperature=temperature,
            topp=topp,
            teacher_forcing_valid_mask=teacher_forcing_valid_mask,
            teacher_forcing_action=teacher_forcing_action,
            remove_out_of_map_agent=remove_out_of_map_agent,
            **step_data_dict,
            remove_static_agent=remove_static_agent,
        )

        # if "evicted_agent_mask" in decode_one_step_info:
        if force_add:
            num_agents_being_added = max(0, max_agents - next_state_data_dict["agent_valid_mask"].sum().item())
            assert num_agents_being_added >= 0
        else:
            num_agents_being_added = None

        next_state_data_dict, decode_one_step_info, agent_unique_id, max_unique_id = scenestreamer_step(
            data_dict=data_dict,
            motion_step_data_dict=next_state_data_dict,
            motion_decode_one_step_info=decode_one_step_info,
            evicted_agent_mask=decode_one_step_info["evicted_agent_mask"],
            model=model,
            agent_unique_id=agent_unique_id,
            num_agents_being_added=num_agents_being_added,
            max_agents=max_agents,
            discard_low_speed_agent=discard_low_speed_agent,
            exclude_sdc_neighborhood=exclude_sdc_neighborhood,
            max_unique_id=max_unique_id,
        )
        # There is a very tricky bug. At step T, say agent A is evicted.
        # We know the agent A is valid at T (input_valid_mask is valid for A at T),
        # so we will generate the position at T+1.
        # However, in scenestreamer_step, all evicted agents are removed from the motion_step_data_dict.
        # Therefore the later interpolation process can't access to A's position at T+1.
        # So there will be an issue in the interpolation process.
        # To solve this, a workaround below is to remove agent A at T. (we might lost some information though)
        if "evicted_agent_mask" in decode_one_step_info and decode_one_step_info["evicted_agent_mask"] is not None:
            input_action_valid_mask_list[
                -1] = input_action_valid_mask_list[-1] * (~decode_one_step_info["evicted_agent_mask"].unsqueeze(1))

        pos.append(next_state_data_dict["agent_position"].clone())
        head.append(next_state_data_dict["agent_heading"].clone())
        vel.append(next_state_data_dict["agent_velocity"].clone())
        agent_type.append(next_state_data_dict["agent_type"].clone())
        agent_shape.append(next_state_data_dict["agent_shape"].clone())
        output_logit_list.append(decode_one_step_info["output_token"].clone())
        output_action_list.append(next_state_data_dict["input_action"].clone())
        agent_unique_id_list.append(agent_unique_id.clone())
        input_action_valid_mask_list.append(next_state_data_dict["input_action_valid_mask"].clone())
        step_data_dict = next_state_data_dict

    # ===== Post-process the data =====
    num_total_agents = agent_unique_id_list[-1].max().item() + 1

    def _scatter(data_list):
        ret = []
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list[1:], data_list)):
            s = list(data.shape)
            s[2] = num_total_agents
            new_data = data.new_zeros(s)
            assert len(unique_id) == data.shape[2]
            assert unique_id.max() < num_total_agents, (unique_id.max(), num_total_agents)
            new_data[:, :, unique_id] = data
            ret.append(new_data)
        return torch.cat(ret, dim=1)

    def _scatter_with_first_step(data_list):
        """This function is used for the data where the first step is the initial states."""
        ret = []
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list, data_list)):
            s = list(data.shape)
            s[2] = num_total_agents
            new_data = data.new_zeros(s)
            assert len(unique_id) == data.shape[2], (len(unique_id), data.shape[2])
            assert unique_id.max() < num_total_agents, (unique_id.max(), num_total_agents)
            new_data[:, :, unique_id] = data
            ret.append(new_data)
        return torch.cat(ret, dim=1)

    def _scatter_for_non_temporal(data_list):
        s = list(data_list[0].shape)[2:]
        ret = data_list[0].new_zeros([
            B,
            num_total_agents,
        ] + s)
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list, data_list)):
            assert len(unique_id) == data.shape[1], (len(unique_id), data.shape[1])
            assert unique_id.max() < num_total_agents, (unique_id.max(), num_total_agents)
            ret[:, unique_id] = data
        return ret

    output_action_list = _scatter(output_action_list)
    assert output_action_list.shape == (B, num_decode_steps, num_total_agents)
    assert len(input_action_valid_mask_list) == num_decode_steps + 1
    input_action_valid_mask = _scatter_with_first_step(input_action_valid_mask_list)
    # Evict the last step's input_action_valid_mask_list as it is not used.
    input_action_valid_mask = input_action_valid_mask[:, :-1]

    output_logit_list = _scatter(output_logit_list)
    traj_log_prob, traj_prob = utils.calculate_trajectory_probabilities_new(
        output_logit_list, output_action_list, mask=input_action_valid_mask
    )  # (B, N)
    pos = _scatter_with_first_step(pos)
    head = _scatter_with_first_step(head)
    vel = _scatter_with_first_step(vel)
    agent_type = _scatter_for_non_temporal(agent_type)
    agent_shape = _scatter_for_non_temporal(agent_shape)

    # ===== Interpolate the output =====
    if interpolation:
        data_dict, _ = interpolate_autoregressive_output(
            data_dict=data_dict,
            agent_heading=head,
            agent_position=pos,
            agent_velocity=vel,
            input_valid_mask=input_action_valid_mask,
            num_skipped_steps=tokenizer.num_skipped_steps,
            num_decoded_steps=num_decode_steps,
        )

    # ===== Save the data =====
    data_dict["decoder/output_logit"] = output_logit_list
    data_dict["decoder/output_action"] = output_action_list
    data_dict["decoder/output_score"] = traj_log_prob
    data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
    # data_dict["decoder/debug_ar_pos"] = pos
    # data_dict["decoder/debug_ar_head"] = head
    # data_dict["decoder/debug_ar_vel"] = vel

    data_dict["decoder/agent_type"] = agent_type
    data_dict["decoder/current_agent_shape"] = agent_shape
    data_dict.pop("decoder/agent_shape")
    data_dict.pop("decoder/object_of_interest_id")

    valid_output_action = output_action_list[input_action_valid_mask]
    assert valid_output_action.max() < tokenizer.num_actions
    assert valid_output_action.min() >= 0

    return data_dict


@torch.no_grad()
def generate_scenestreamer_motion(
    *,
    data_dict,
    model,
    force_add=False,
    max_agents=128,
    num_decode_steps=None,
    discard_low_speed_agent=False,
    remove_static_agent=False,
    exclude_sdc_neighborhood=False
):

    # Initial state generation
    data_dict, initial_state_info = generate_initial_state(
        data_dict=data_dict, model=model, force_add=force_add, discard_low_speed_agent=discard_low_speed_agent
    )
    data_dict = convert_initial_states_as_motion_data(data_dict)

    # Motion generation
    agent_unique_id = data_dict['decoder/agent_id'][0]  # (N,)
    interpolation = True
    remove_out_of_map_agent = True
    tokenizer = model.tokenizer
    temperature = model.config.SAMPLING.TEMPERATURE
    topp = model.config.SAMPLING.TOPP
    sampling_method = model.config.SAMPLING.SAMPLING_METHOD
    B, T_input, N = data_dict["decoder/input_action"].shape
    agent_pos = data_dict["decoder/agent_position"][:, ::tokenizer.num_skipped_steps]
    agent_heading = data_dict["decoder/agent_heading"][:, ::tokenizer.num_skipped_steps]
    agent_valid_mask = data_dict["decoder/agent_valid_mask"][:, ::tokenizer.num_skipped_steps]
    agent_velocity = data_dict["decoder/agent_velocity"][:, ::tokenizer.num_skipped_steps]
    B, T_full, N, _ = agent_pos.shape
    gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
    assert agent_pos.ndim == 4
    gt_input_action = data_dict["decoder/input_action"].clone()
    data_dict, _ = randomize_agent_id(data_dict=data_dict, model=model)
    step_data_dict = dict(
        input_step=torch.arange(1).to(gt_input_action.device),
        input_action=gt_input_action[:, :1].clone(),
        input_action_valid_mask=data_dict["decoder/input_action_valid_mask"][:, :1].clone(),
        agent_position=data_dict["decoder/modeled_agent_position"][:, :1].clone(),
        agent_heading=data_dict["decoder/modeled_agent_heading"][:, :1].clone(),
        agent_velocity=data_dict["decoder/modeled_agent_velocity"][:, :1].clone(),  # TODO: Remove this?
        agent_valid_mask=data_dict["decoder/input_action_valid_mask"][:, :1].clone(),
        agent_delta=data_dict["decoder/modeled_agent_delta"][:, :1].clone(),
        cache=None,
        agent_id=data_dict["decoder/randomized_modeled_agent_id"],
        agent_type=data_dict["decoder/agent_type"],
        agent_shape=data_dict["decoder/current_agent_shape"],
        decode_step=0,
    )
    max_unique_id = agent_unique_id.max().item()
    agent_unique_id_list = [agent_unique_id.clone()]
    output_logit_list = []
    output_action_list = []
    input_action_valid_mask_list = [step_data_dict["input_action_valid_mask"]]
    pos = [step_data_dict["agent_position"]]
    head = [step_data_dict["agent_heading"]]
    vel = [step_data_dict["agent_velocity"]]
    agent_type = [step_data_dict["agent_type"]]
    agent_shape = [step_data_dict["agent_shape"]]
    data_dict, _ = encode_scene(data_dict=data_dict, model=model)
    for decode_step in range(num_decode_steps):
        teacher_forcing_valid_mask = None
        teacher_forcing_action = None
        next_state_data_dict, decode_one_step_info = decode_one_step(
            data_dict=data_dict,
            model=model,
            sampling_method=sampling_method,
            temperature=temperature,
            topp=topp,
            teacher_forcing_valid_mask=teacher_forcing_valid_mask,
            teacher_forcing_action=teacher_forcing_action,
            remove_out_of_map_agent=remove_out_of_map_agent,
            **step_data_dict,
            remove_static_agent=remove_static_agent,
        )

        # if "evicted_agent_mask" in decode_one_step_info:
        if force_add:
            num_agents_being_added = max(0, max_agents - next_state_data_dict["agent_valid_mask"].sum().item())
            assert num_agents_being_added >= 0
        else:
            num_agents_being_added = None

        next_state_data_dict, decode_one_step_info, agent_unique_id, max_unique_id = scenestreamer_step(
            data_dict=data_dict,
            motion_step_data_dict=next_state_data_dict,
            motion_decode_one_step_info=decode_one_step_info,
            evicted_agent_mask=decode_one_step_info["evicted_agent_mask"],
            model=model,
            agent_unique_id=agent_unique_id,
            num_agents_being_added=num_agents_being_added,
            max_agents=max_agents,
            discard_low_speed_agent=discard_low_speed_agent,
            exclude_sdc_neighborhood=exclude_sdc_neighborhood,
            max_unique_id=max_unique_id,
        )
        # There is a very tricky bug. At step T, say agent A is evicted.
        # We know the agent A is valid at T (input_valid_mask is valid for A at T),
        # so we will generate the position at T+1.
        # However, in scenestreamer_step, all evicted agents are removed from the motion_step_data_dict.
        # Therefore the later interpolation process can't access to A's position at T+1.
        # So there will be an issue in the interpolation process.
        # To solve this, a workaround below is to remove agent A at T. (we might lost some information though)
        if "evicted_agent_mask" in decode_one_step_info and decode_one_step_info["evicted_agent_mask"] is not None:
            input_action_valid_mask_list[
                -1] = input_action_valid_mask_list[-1] * (~decode_one_step_info["evicted_agent_mask"].unsqueeze(1))

        pos.append(next_state_data_dict["agent_position"].clone())
        head.append(next_state_data_dict["agent_heading"].clone())
        vel.append(next_state_data_dict["agent_velocity"].clone())
        agent_type.append(next_state_data_dict["agent_type"].clone())
        agent_shape.append(next_state_data_dict["agent_shape"].clone())
        output_logit_list.append(decode_one_step_info["output_token"].clone())
        output_action_list.append(next_state_data_dict["input_action"].clone())
        agent_unique_id_list.append(agent_unique_id.clone())
        input_action_valid_mask_list.append(next_state_data_dict["input_action_valid_mask"].clone())
        step_data_dict = next_state_data_dict

    # ===== Post-process the data =====
    num_total_agents = agent_unique_id_list[-1].max().item() + 1

    def _scatter(data_list):
        ret = []
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list[1:], data_list)):
            s = list(data.shape)
            s[2] = num_total_agents
            new_data = data.new_zeros(s)
            assert len(unique_id) == data.shape[2]
            assert unique_id.max() < num_total_agents
            new_data[:, :, unique_id] = data
            ret.append(new_data)
        return torch.cat(ret, dim=1)

    def _scatter_with_first_step(data_list):
        """This function is used for the data where the first step is the initial states."""
        ret = []
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list, data_list)):
            s = list(data.shape)
            s[2] = num_total_agents
            new_data = data.new_zeros(s)
            assert len(unique_id) == data.shape[2]
            assert unique_id.max() < num_total_agents
            new_data[:, :, unique_id] = data
            ret.append(new_data)
        return torch.cat(ret, dim=1)

    def _scatter_for_non_temporal(data_list):
        s = list(data_list[0].shape)[2:]
        ret = data_list[0].new_zeros([
            B,
            num_total_agents,
        ] + s)
        for i, (unique_id, data) in enumerate(zip(agent_unique_id_list, data_list)):
            assert len(unique_id) == data.shape[1]
            assert unique_id.max() < num_total_agents
            ret[:, unique_id] = data
        return ret

    output_action_list = _scatter(output_action_list)
    assert output_action_list.shape == (B, num_decode_steps, num_total_agents)
    assert len(input_action_valid_mask_list) == num_decode_steps + 1
    input_action_valid_mask = _scatter_with_first_step(input_action_valid_mask_list)
    # Evict the last step's input_action_valid_mask_list as it is not used.
    input_action_valid_mask = input_action_valid_mask[:, :-1]

    output_logit_list = _scatter(output_logit_list)
    traj_log_prob, traj_prob = utils.calculate_trajectory_probabilities_new(
        output_logit_list, output_action_list, mask=input_action_valid_mask
    )  # (B, N)
    pos = _scatter_with_first_step(pos)
    head = _scatter_with_first_step(head)
    vel = _scatter_with_first_step(vel)
    agent_type = _scatter_for_non_temporal(agent_type)
    agent_shape = _scatter_for_non_temporal(agent_shape)

    # ===== Interpolate the output =====
    if interpolation:
        data_dict, _ = interpolate_autoregressive_output(
            data_dict=data_dict,
            agent_heading=head,
            agent_position=pos,
            agent_velocity=vel,
            input_valid_mask=input_action_valid_mask,
            num_skipped_steps=tokenizer.num_skipped_steps,
            num_decoded_steps=num_decode_steps,
        )

    # ===== Save the data =====
    data_dict["decoder/output_logit"] = output_logit_list
    data_dict["decoder/output_action"] = output_action_list
    data_dict["decoder/output_score"] = traj_log_prob
    data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
    # data_dict["decoder/debug_ar_pos"] = pos
    # data_dict["decoder/debug_ar_head"] = head
    # data_dict["decoder/debug_ar_vel"] = vel

    data_dict["decoder/agent_type"] = agent_type
    data_dict["decoder/current_agent_shape"] = agent_shape
    data_dict.pop("decoder/agent_shape")
    data_dict.pop("decoder/object_of_interest_id")

    valid_output_action = output_action_list[input_action_valid_mask]
    assert valid_output_action.max() < tokenizer.num_actions
    assert valid_output_action.min() >= 0

    return data_dict


def scenestreamer_step(
    *,
    data_dict,
    motion_step_data_dict,
    motion_decode_one_step_info,
    evicted_agent_mask,
    model,
    agent_unique_id,
    max_unique_id,
    num_agents_being_added=None,
    max_agents=128,
    discard_low_speed_agent=False,
    exclude_sdc_neighborhood=False
):
    scenestreamer_step_info = {}

    B = data_dict["encoder/scenario_token"].shape[0]

    # Step 1: Remove the evicted agent info from motion_data_dict.
    # Just do nothing here as we don't want to mess up the KV cache.
    if evicted_agent_mask is not None and _verbose_densify_logs():
        print("{} agents are evicted.".format(evicted_agent_mask.sum().item()))

    # Step 2: Run initial state generation.
    init_step_data_dict = build_initial_data_dict(
        data_dict=data_dict,
        agent_position=motion_step_data_dict["agent_position"][:, -1],
        agent_heading=motion_step_data_dict["agent_heading"][:, -1],
        agent_velocity=motion_step_data_dict["agent_velocity"][:, -1],
        agent_type=motion_step_data_dict["agent_type"],
        agent_shape=motion_step_data_dict["agent_shape"],
        agent_valid_mask=motion_step_data_dict["agent_valid_mask"][:, -1],
        evicted_agent_mask=evicted_agent_mask,
        start_action_id=model.trafficgen_decoder.trafficgen_tokenizer.start_action_id,
    )
    num_agents_before_adding = init_step_data_dict["input_action"].shape[-1] - 1
    num_agents_after_adding = num_agents_before_adding

    if num_agents_being_added is None:
        num_decode_steps = max(0, max_agents - num_agents_before_adding)
        force_add = False
    else:
        if _verbose_densify_logs():
            print("We will force add {} agents.".format(num_agents_being_added))
        num_decode_steps = num_agents_being_added
        force_add = True
    # for _ in tqdm.tqdm(range(num_decode_steps), desc="Generating initial state"):
    for decode_step in range(num_decode_steps):
        next_init_step_data_dict, step_info = decode_one_step_initial_state(
            data_dict=data_dict,
            model=model,
            force_add=force_add,
            **init_step_data_dict,
            discard_low_speed_agent=discard_low_speed_agent,
            exclude_sdc_neighborhood=exclude_sdc_neighborhood,
            decode_step=decode_step,
        )
        if step_info["terminated"]:
            break
        init_step_data_dict = next_init_step_data_dict
        num_agents_after_adding = init_step_data_dict["input_action"].shape[-1] - 1

        if num_agents_after_adding > max_agents:
            break

    # Step 4: Translate the initial state to motion_data_dict.
    # Note that we should fill in the "MOTION_START_TOKEN" here.
    # The idea is to just concat new agents' data into existing motion data dict.
    num_newly_added_agents = num_agents_after_adding - num_agents_before_adding
    if _verbose_densify_logs():
        print("{} agents are newly added.".format(num_newly_added_agents))
    # The tokens after init input tokens are the newly added agents' tokens.
    # init_input_token_num = motion_step_data_dict['input_action'].shape[-1] + 1

    new_pos = init_step_data_dict['agent_position'][:, num_agents_before_adding + 1:]
    new_head = init_step_data_dict['agent_heading'][:, num_agents_before_adding + 1:]
    new_vel = init_step_data_dict['agent_velocity'][:, num_agents_before_adding + 1:]
    new_shape = init_step_data_dict['agent_shape'][:, num_agents_before_adding + 1:]
    new_type = init_step_data_dict['agent_type'][:, num_agents_before_adding + 1:]
    new_valid_mask = init_step_data_dict['input_action_valid_mask'][:, num_agents_before_adding + 1:]
    assert new_valid_mask.shape == (B, num_newly_added_agents)

    if evicted_agent_mask is not None:
        leftover_indices = (~evicted_agent_mask)[0].nonzero()[:, 0]
    else:
        leftover_indices = torch.arange(num_agents_before_adding).to(new_pos.device)

    old_agent_id = motion_step_data_dict["agent_id"][:, leftover_indices]
    assert motion_step_data_dict["agent_id"].shape[0] == 1
    agent_id_candidates = list(range(max_agents))
    agent_id_candidates = [x for x in agent_id_candidates if x not in old_agent_id[0].tolist()]
    if len(agent_id_candidates) < num_newly_added_agents:
        print(
            "Not enough agent ids to assign! We have {} agents already and {} agents are newly added.".format(
                num_agents_before_adding, num_newly_added_agents
            )
        )
        return motion_step_data_dict, motion_decode_one_step_info, agent_unique_id, max_unique_id

    agent_id_candidates = np.random.choice(agent_id_candidates, num_newly_added_agents, replace=False)
    new_id = torch.tensor(agent_id_candidates).to(old_agent_id.device).long()
    motion_step_data_dict["agent_id"] = torch.cat([old_agent_id, new_id.unsqueeze(0)], dim=-1).long()

    motion_step_data_dict["input_action"] = torch.cat(
        [
            motion_step_data_dict["input_action"][:, :, leftover_indices],
            motion_step_data_dict["input_action"].new_full([B, 1, num_newly_added_agents], MOTION_START_ACTION)
        ],
        dim=-1
    )
    motion_step_data_dict["input_action_valid_mask"] = torch.cat(
        [motion_step_data_dict["input_action_valid_mask"][:, :, leftover_indices],
         new_valid_mask.unsqueeze(1)], dim=-1
    )
    motion_step_data_dict["agent_position"] = torch.cat(
        [motion_step_data_dict["agent_position"][:, :, leftover_indices],
         new_pos.unsqueeze(1)], dim=-2
    )
    motion_step_data_dict["agent_heading"] = torch.cat(
        [motion_step_data_dict["agent_heading"][:, :, leftover_indices],
         new_head.unsqueeze(1)], dim=-1
    )
    motion_step_data_dict["agent_velocity"] = torch.cat(
        [motion_step_data_dict["agent_velocity"][:, :, leftover_indices],
         new_vel.unsqueeze(1)], dim=-2
    )
    motion_step_data_dict["agent_shape"] = torch.cat(
        [motion_step_data_dict["agent_shape"][:, leftover_indices], new_shape], dim=-2
    )
    motion_step_data_dict["agent_type"] = torch.cat(
        [motion_step_data_dict["agent_type"][:, leftover_indices], new_type], dim=-1
    )
    motion_step_data_dict["agent_valid_mask"] = torch.cat(
        [motion_step_data_dict["agent_valid_mask"][:, :, leftover_indices],
         new_valid_mask.unsqueeze(1)], dim=-1
    )

    assert motion_step_data_dict["input_action"].shape == motion_step_data_dict["input_action_valid_mask"].shape

    new_delta = get_relative_velocity(new_vel.unsqueeze(1), new_head.unsqueeze(1))
    motion_step_data_dict["agent_delta"] = torch.cat(
        [motion_step_data_dict["agent_delta"][:, :, leftover_indices], new_delta], dim=-2
    )

    if _verbose_densify_logs():
        print("Totally {} agents exist.".format(motion_step_data_dict["agent_id"].shape[-1]))

    # The cache and agent history should be updated as well...
    if "output_token" in motion_decode_one_step_info:
        motion_decode_one_step_info["output_token"] = torch.cat(
            [
                motion_decode_one_step_info["output_token"][:, :, leftover_indices],
                motion_decode_one_step_info["output_token"].new_zeros(
                    [B, 1, num_newly_added_agents, motion_decode_one_step_info["output_token"].shape[-1]]
                )
            ],
            dim=-2
        )

    if motion_step_data_dict.get("agent_position_history") is not None:
        _, T_history, num_existing_agents, _ = motion_step_data_dict["agent_position_history"].shape
        motion_step_data_dict["agent_position_history"] = torch.cat(
            [
                motion_step_data_dict["agent_position_history"][:, :, leftover_indices],
                motion_step_data_dict["agent_position_history"].new_zeros([B, T_history, num_newly_added_agents, 2])
            ],
            dim=-2
        )

        motion_step_data_dict["agent_heading_history"] = torch.cat(
            [
                motion_step_data_dict["agent_heading_history"][:, :, leftover_indices],
                motion_step_data_dict["agent_heading_history"].new_zeros([B, T_history, num_newly_added_agents])
            ],
            dim=-1
        )
        motion_step_data_dict["agent_valid_mask_history"] = torch.cat(
            [
                motion_step_data_dict["agent_valid_mask_history"][:, :, leftover_indices],
                motion_step_data_dict["agent_valid_mask_history"].new_zeros([B, T_history, num_newly_added_agents])
            ],
            dim=-1
        )
        # No need to update this:
        # motion_step_data_dict["agent_step_history"] = torch.cat(

    if "cache" in motion_step_data_dict and motion_step_data_dict["cache"] is not None:
        # Need to update cache:
        new_cache = []
        old_cache = motion_step_data_dict["cache"]
        for layer_cache in old_cache:
            k, v, (batch_size, seq_len) = layer_cache
            assert seq_len == T_history
            assert batch_size == B * num_existing_agents
            k = k.reshape(B * num_existing_agents, seq_len, -1)[leftover_indices]
            v = v.reshape(B * num_existing_agents, seq_len, -1)[leftover_indices]
            k = torch.cat([k, k.new_zeros([B * num_newly_added_agents, seq_len, k.shape[-1]])], dim=0)
            v = torch.cat([v, v.new_zeros([B * num_newly_added_agents, seq_len, v.shape[-1]])], dim=0)
            k = k.reshape(B * (len(leftover_indices) + num_newly_added_agents), seq_len, -1)
            v = v.reshape(B * (len(leftover_indices) + num_newly_added_agents), seq_len, -1)
            new_cache.append((k, v, (B * (len(leftover_indices) + num_newly_added_agents), seq_len)))
        motion_step_data_dict["cache"] = new_cache

    agent_unique_id = torch.cat(
        [
            agent_unique_id[leftover_indices],
            max_unique_id + 1 + torch.arange(num_newly_added_agents).to(agent_unique_id.device)
        ]
    ).long()
    max_unique_id = agent_unique_id.max().item()

    if _verbose_densify_logs():
        print("agent_unique_id: ", list(agent_unique_id.cpu().numpy()))

    return motion_step_data_dict, motion_decode_one_step_info, agent_unique_id, max_unique_id


def build_initial_data_dict(
    *, data_dict, start_action_id, agent_position, agent_heading, agent_velocity, agent_type, agent_shape,
    agent_valid_mask, evicted_agent_mask
):

    # TODO
    only_lane = True

    if evicted_agent_mask is not None:
        agent_valid_mask = agent_valid_mask & (~evicted_agent_mask)

    leftover_indices = agent_valid_mask[0].nonzero()[:, 0]

    agent_position = agent_position[:, leftover_indices]
    agent_heading = agent_heading[:, leftover_indices]
    agent_velocity = agent_velocity[:, leftover_indices]
    agent_type = agent_type[:, leftover_indices]
    agent_shape = agent_shape[:, leftover_indices]
    agent_valid_mask = agent_valid_mask[:, leftover_indices]

    B, N, _ = agent_position.shape

    map_pos = data_dict["encoder/map_position"][..., :2]
    map_heading = data_dict["encoder/map_heading"]

    # Get map feature valid mask
    valid_map_feat = data_dict["encoder/map_valid_mask"]
    heading_diff = utils.wrap_to_pi(agent_heading[:, :, None] - map_heading[:, None])
    valid_heading = torch.abs(heading_diff) < np.deg2rad(90)
    valid_map_feat = valid_map_feat & valid_heading

    if only_lane:
        map_feature = data_dict["encoder/map_feature"]
        is_lane = map_feature[:, :, 0, 13] == 1
        is_lane = is_lane[:, None]
        valid_map_feat = is_lane & valid_map_feat

    # Find the closest map feature
    dist = torch.cdist(agent_position, map_pos)
    dist[~valid_map_feat] = torch.inf
    closest_map_feat = torch.argmin(dist, dim=-1)

    # Get the selected map feature
    selected_map_pos = torch.gather(map_pos, dim=1, index=closest_map_feat[:, :, None].expand(-1, -1, 2))
    selected_map_heading = torch.gather(map_heading, dim=1, index=closest_map_feat)

    # Get relative information
    relative_pos = agent_position - selected_map_pos
    relative_pos = utils.rotate(x=relative_pos[..., 0], y=relative_pos[..., 1], angle=-selected_map_heading)
    relative_heading = utils.wrap_to_pi(agent_heading - selected_map_heading)
    relative_vel = utils.rotate(x=agent_velocity[..., 0], y=agent_velocity[..., 1], angle=-selected_map_heading)

    # Get the discretized relative position
    gt_position_x = TrafficGenTokenizer.bucketize(relative_pos[..., 0], "position_x")
    gt_position_y = TrafficGenTokenizer.bucketize(relative_pos[..., 1], "position_y")
    recon_pos_x = TrafficGenTokenizer.de_bucketize(gt_position_x, "position_x")
    recon_pos_y = TrafficGenTokenizer.de_bucketize(gt_position_y, "position_y")

    # Reconstruct the position with the bucketized value
    recon_pos = torch.stack([recon_pos_x, recon_pos_y], dim=-1)
    recon_pos_abs = utils.rotate(x=recon_pos_x, y=recon_pos_y, angle=selected_map_heading) + selected_map_pos
    # pad
    recon_pos_abs = torch.cat([recon_pos_abs.new_zeros([B, 1, 2]), recon_pos_abs], dim=1)

    # Reconstruct the heading and velocity
    gt_heading = TrafficGenTokenizer.bucketize(relative_heading, "heading")
    recon_heading = TrafficGenTokenizer.de_bucketize(gt_heading, "heading")
    recon_heading_abs = utils.wrap_to_pi(recon_heading + selected_map_heading)
    # pad
    recon_heading_abs = torch.cat([recon_heading_abs.new_zeros([B, 1]), recon_heading_abs], dim=1)

    # Reconstruct the velocity
    gt_vel_x = TrafficGenTokenizer.bucketize(relative_vel[..., 0], "velocity_x")
    gt_vel_y = TrafficGenTokenizer.bucketize(relative_vel[..., 1], "velocity_y")
    recon_vel_x = TrafficGenTokenizer.de_bucketize(gt_vel_x, "velocity_x")
    recon_vel_y = TrafficGenTokenizer.de_bucketize(gt_vel_y, "velocity_y")
    recon_vel = torch.stack([recon_vel_x, recon_vel_y], dim=-1)
    recon_vel_abs = utils.rotate(x=recon_vel_x, y=recon_vel_y, angle=selected_map_heading)
    # pad
    recon_vel_abs = torch.cat([recon_vel_abs.new_zeros([B, 1, 2]), recon_vel_abs], dim=1)

    # Reconstruct shape
    gt_shape_l = TrafficGenTokenizer.bucketize(agent_shape[..., 0], "length")
    gt_shape_w = TrafficGenTokenizer.bucketize(agent_shape[..., 1], "width")
    gt_shape_h = TrafficGenTokenizer.bucketize(agent_shape[..., 2], "height")
    recon_shape = torch.stack(
        [
            TrafficGenTokenizer.de_bucketize(gt_shape_l, "length"),
            TrafficGenTokenizer.de_bucketize(gt_shape_w, "width"),
            TrafficGenTokenizer.de_bucketize(gt_shape_h, "height")
        ],
        dim=-1
    )
    # Pad recon shape in 1st dimension
    recon_shape = torch.cat([recon_shape.new_zeros([B, 1, 3]), recon_shape], dim=1)

    feat = recon_shape.new_zeros((B, recon_pos.shape[1] + 1, 5))
    feat[:, 1:, :2] = recon_pos
    feat[:, 1:, 2] = recon_heading
    feat[:, 1:, 3:5] = recon_vel

    input_action = torch.cat([closest_map_feat.new_full([B, 1], start_action_id), closest_map_feat], dim=1)

    input_action_valid_mask = torch.cat([agent_valid_mask.new_ones([B, 1]), agent_valid_mask], dim=1)

    # We are not building only the first input step, but instead the whole input sequence.
    step_data_dict = dict(
        input_step=torch.zeros(
            [B, recon_pos.shape[1] + 1], dtype=torch.long, device=data_dict["encoder/scenario_token"].device
        ),
        input_action=input_action,
        input_action_valid_mask=input_action_valid_mask,
        agent_position=recon_pos_abs,
        agent_heading=recon_heading_abs,
        agent_velocity=recon_vel_abs,
        agent_type=torch.cat([agent_type.new_zeros([B, 1]), agent_type], dim=1),
        agent_shape=recon_shape,
        agent_feature=feat,
    )
    return step_data_dict
