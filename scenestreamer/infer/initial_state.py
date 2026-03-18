"""
This module reimplements the autoregressive motion generation process.
"""

import copy

import torch

from scenestreamer.tokenization.motion_tokenizers import interpolate, interpolate_heading
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils
from scenestreamer.infer.motion import encode_scene

from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
import torch


@torch.no_grad()
def generate_initial_state(
    *, data_dict, model, force_add=False, discard_low_speed_agent=False, exclude_sdc_neighborhood=False
):
    num_collisions = 0
    num_violations = 0
    N = data_dict["decoder/modeled_agent_position"].shape[2]

    # ===== Call Model =====
    data_dict, _ = encode_scene(data_dict=data_dict, model=model)
    B = data_dict["encoder/scenario_token"].shape[0]
    step_data_dict = dict(
        input_step=torch.zeros([B, 1], dtype=torch.long, device=data_dict["encoder/scenario_token"].device),
        input_action=data_dict["decoder/input_action_for_trafficgen"][:, :1].clone(),
        input_action_valid_mask=data_dict["decoder/input_action_valid_mask_for_trafficgen"][:, :1].clone(),
        agent_position=data_dict["decoder/modeled_agent_position_for_trafficgen"][:, :1].clone(),
        agent_heading=data_dict["decoder/modeled_agent_heading_for_trafficgen"][:, :1].clone(),
        agent_velocity=data_dict["decoder/modeled_agent_velocity_for_trafficgen"][:, :1].clone(),
        agent_type=data_dict["decoder/agent_type_for_trafficgen"][:, :1].clone(),
        agent_shape=data_dict["decoder/current_agent_shape_for_trafficgen"][:, :1].clone(),
        agent_feature=data_dict["decoder/input_action_feature_for_trafficgen"][:, :1].clone(),
    )

    num_decode_steps = min(N, 128)

    # num_decode_steps = 128
    # for _ in tqdm.tqdm(range(num_decode_steps), desc="Generating initial state"):
    for decode_step in range(num_decode_steps):
        next_state_data_dict, step_info = decode_one_step_initial_state(
            data_dict=data_dict,
            model=model,
            force_add=force_add,
            discard_low_speed_agent=discard_low_speed_agent,
            **step_data_dict,
            exclude_sdc_neighborhood=exclude_sdc_neighborhood,
            decode_step=decode_step,
        )
        if step_info["terminated"]:
            break
        step_data_dict = next_state_data_dict

    # Filter out the first step because it's (0,0) the START ACTION.
    data_dict.update(
        {
            "decoder/modeled_agent_position_for_trafficgen": step_data_dict["agent_position"].clone()[:, 1:],
            "decoder/modeled_agent_heading_for_trafficgen": step_data_dict["agent_heading"].clone()[:, 1:],
            "decoder/modeled_agent_velocity_for_trafficgen": step_data_dict["agent_velocity"].clone()[:, 1:],
            "decoder/current_agent_shape_for_trafficgen": step_data_dict["agent_shape"].clone()[:, 1:],
            "decoder/agent_type_for_trafficgen": step_data_dict["agent_type"].clone()[:, 1:],
            "decoder/input_action_valid_mask_for_trafficgen": step_data_dict["input_action_valid_mask"].clone()[:, 1:],
        }
    )
    return data_dict, {
        # "num_collisions": step_info["num_collisions"],
        # "num_violations": num_violations,
        # "num_low_speed": step_info["num_low_speed"]
    }


def decode_one_step_initial_state(
    *,
    data_dict,
    model,
    input_action,
    input_action_valid_mask,
    input_step,
    agent_position,
    agent_heading,
    agent_velocity,
    agent_type,
    agent_shape,
    agent_feature,
    decode_step,
    num_collisions=None,
    force_add=False,
    discard_low_speed_agent=False,
    num_low_speed=None,
    exclude_sdc_neighborhood=False,
):
    if num_collisions is None:
        num_collisions = 0
    if num_low_speed is None:
        num_low_speed = 0
    B = data_dict["encoder/scenario_token"].shape[0]
    raw_input_dict = {
        # Static features
        "encoder/scenario_token": data_dict["encoder/scenario_token"],
        "encoder/scenario_heading": data_dict["encoder/scenario_heading"],
        "encoder/scenario_position": data_dict["encoder/scenario_position"],
        "encoder/scenario_valid_mask": data_dict["encoder/scenario_valid_mask"],
        "encoder/map_position": data_dict["encoder/map_position"],
        "encoder/map_feature": data_dict["encoder/map_feature"],
        "encoder/map_valid_mask": data_dict["encoder/map_valid_mask"],
        "in_evaluation": torch.ones([B], dtype=torch.bool, device=data_dict["encoder/scenario_token"].device),

        # Actions
        "decoder/input_step_for_trafficgen": input_step,
        "decoder/input_action_for_trafficgen": input_action,
        "decoder/input_action_valid_mask_for_trafficgen": input_action_valid_mask,

        # Agent features
        "decoder/modeled_agent_position_for_trafficgen": agent_position,
        "decoder/modeled_agent_heading_for_trafficgen": agent_heading,
        "decoder/modeled_agent_velocity_for_trafficgen": agent_velocity,
        "decoder/agent_type_for_trafficgen": agent_type,
        "decoder/current_agent_shape_for_trafficgen": agent_shape,
        "decoder/input_action_feature_for_trafficgen": agent_feature,
    }
    while True:
        raw_output_dict = model.trafficgen_decoder(copy.deepcopy(raw_input_dict))
        output_dict = {}
        for k, v in raw_output_dict.items():
            if "encoder" in k or k == "in_evaluation":
                output_dict[k] = v
            elif "for_trafficgen" in k and "input_step" not in k:
                output_dict[k] = v[:, -1:]

        if exclude_sdc_neighborhood:
            # Do surgery here to mask out SDC neighborhood in output logits.
            output_logit = output_dict['decoder/output_logit_for_trafficgen']
            vocab_size = output_logit.shape[-1]
            # We assume that the SDC is the first agent (the 2nd tokens).
            assert agent_position.shape[1] >= 2
            sdc_pos = agent_position[0, 1]
            map_pos = data_dict["encoder/map_position"][0, :, :2]
            dist = torch.cdist(map_pos, sdc_pos[None])
            invalid_map_mask = dist < 50.0
            invalid_map_mask_full = invalid_map_mask.new_zeros((1, 1, vocab_size))
            invalid_map_mask_full[0, 0, :invalid_map_mask.shape[0]] = invalid_map_mask[:, 0]
            output_logit = torch.where(invalid_map_mask_full, -1e9 * torch.ones_like(output_logit), output_logit)
            output_dict['decoder/output_logit_for_trafficgen'] = output_logit

        else:
            assert data_dict["encoder/map_position"].shape[0] == 1, "Batch size should be 1"

            # Do surgery here to mask IN SDC neighborhood in output logits.
            output_logit = output_dict['decoder/output_logit_for_trafficgen']
            vocab_size = output_logit.shape[-1]
            # We assume that the SDC is the first agent (the 2nd tokens).
            if agent_position.shape[1] >= 2:
                sdc_pos = agent_position[0, 1]
                map_pos = data_dict["encoder/map_position"][0, :, :2]
                dist = torch.cdist(map_pos, sdc_pos[None])
                valid_map_mask = dist < 50.0
                valid_map_mask_full = valid_map_mask.new_zeros((1, 1, vocab_size))
                valid_map_mask_full[0, 0, :valid_map_mask.shape[0]] = valid_map_mask[:, 0]
                output_logit = torch.where(valid_map_mask_full, output_logit, -1e9 * torch.ones_like(output_logit))
                output_dict['decoder/output_logit_for_trafficgen'] = output_logit

        temperature = 1.0
        sampled_action = model.trafficgen_decoder.sample_action(
            output_dict, force_no_end=force_add, temperature=temperature
        )
        is_end = sampled_action == model.trafficgen_decoder.trafficgen_tokenizer.INIT_END_ACTION
        new_agent_type_output = model.trafficgen_decoder.forward_agent_type(output_dict, action=sampled_action)
        new_agent_type = model.trafficgen_decoder.sample_agent_type(new_agent_type_output, temperature=temperature)
        new_offset_output = model.trafficgen_decoder.forward_offset(
            output_dict, action=sampled_action, agent_type=new_agent_type
        )
        new_offset_action = model.trafficgen_decoder.sample_offset(
            offset_output=new_offset_output, temperature=temperature
        )
        predicted_values = model.trafficgen_decoder.trafficgen_tokenizer.detokenize(
            data_dict, sampled_action, agent_type=new_agent_type, offset_action=new_offset_action
        )
        new_pos = predicted_values["position"]
        new_head = predicted_values["heading"]
        new_vel = predicted_values["velocity"]
        new_type = predicted_values["agent_type"]  # in 0,1,2
        new_shape = predicted_values["shape"]
        new_feature = predicted_values["feature"]

        # sdc_index = data_dict["decoder/sdc_index"].item()
        # sdc_speed = data_dict["decoder/agent_velocity"][0, :, sdc_index].norm(dim=1).max()
        # speed = new_vel.norm(dim=-1).item()
        # SPEED_THRESHOLD = max(sdc_speed / 2, 1.0)
        #
        # if speed < SPEED_THRESHOLD:
        #     num_low_speed += 1
        #     if discard_low_speed_agent:
        #         continue

        if decode_step == 0 and model.config.FORCE_SDC_FOR_TRAFFICGEN:
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
            sampled_action = map_argmin.unsqueeze(0).unsqueeze(-1)

            new_pos = sdc_center[:, :2].reshape(B, 1, 2)
            new_head = data_dict["decoder/agent_heading"][:, current_t, sdc_index].unsqueeze(1)
            new_vel = data_dict["decoder/agent_velocity"][:, current_t, sdc_index].unsqueeze(1)
            new_type = data_dict["decoder/agent_type"][:, sdc_index].unsqueeze(1)
            new_shape = data_dict["decoder/current_agent_shape"][:, sdc_index].unsqueeze(1)
            new_feature = data_dict["decoder/input_action_feature_for_trafficgen"][:, sdc_index].unsqueeze(1)

        # print("SPEED: {}, Threshold: {}".format(speed, SPEED_THRESHOLD))
        no_coll = detect_collision_for_new_agent(
            agent_position=agent_position,
            agent_shape=agent_shape,
            agent_heading=agent_heading,
            new_pos=new_pos,
            new_head=new_head,
            new_shape=new_shape,
            input_action_valid_mask=input_action_valid_mask,
            is_end=is_end,
        )
        if not no_coll:
            num_collisions += 1

        if no_coll:
            step_data_dict = dict(
                input_action=torch.cat([input_action, sampled_action], dim=1),
                input_action_valid_mask=torch.cat([input_action_valid_mask, ~is_end], dim=1),
                agent_position=torch.cat([agent_position, new_pos], dim=1),
                agent_heading=torch.cat([agent_heading, new_head], dim=1),
                agent_velocity=torch.cat([agent_velocity, new_vel], dim=1),
                agent_type=torch.cat([agent_type, new_type], dim=1),
                agent_shape=torch.cat([agent_shape, new_shape], dim=1),
                agent_feature=torch.cat([agent_feature, new_feature], dim=1),
                # num_collisions=num_collisions,
                # num_low_speed=num_low_speed,
                input_step=torch.cat([input_step, input_step[:, -1:] + 1], dim=1),
            )
            step_info = dict(
                # num_collisions=num_collisions,
                terminated=not step_data_dict["input_action_valid_mask"][0, -1].item(),
                # num_low_speed=num_low_speed,
            )
            return step_data_dict, step_info


def detect_collision_for_new_agent(
    *, agent_position, agent_heading, agent_shape, new_pos, new_head, new_shape, input_action_valid_mask, is_end
):
    assert agent_position.ndim == 3
    assert agent_position.shape[0] == 1
    # Check if collision happens:
    existing_contours = cal_polygon_contour(
        x=agent_position[0, :, 0].cpu().numpy(),
        y=agent_position[0, :, 1].cpu().numpy(),
        theta=agent_heading[0, :].cpu().numpy(),
        width=agent_shape[0, :, 1].cpu().numpy(),
        length=agent_shape[0, :, 0].cpu().numpy()
    )  # (N, 4, 2)
    new_contour = cal_polygon_contour(
        x=new_pos[0, :, 0].cpu().numpy(),
        y=new_pos[0, :, 1].cpu().numpy(),
        theta=new_head[0, :].cpu().numpy(),
        width=new_shape[0, :, 1].cpu().numpy(),
        length=new_shape[0, :, 0].cpu().numpy()
    )
    if existing_contours.shape[0] == 1:
        no_coll = True  # Skip first one (it's the START_ACTION)
    else:
        no_coll = True
        for existing_id in range(1, existing_contours.shape[0]):
            collision_detected = detect_collision(
                [existing_contours[existing_id]],  # (N, 4, 2)
                [input_action_valid_mask[0][existing_id]],  # (N,)
                new_contour,
                ~is_end[0],
            )
            if collision_detected[0]:
                no_coll = False
                break
    return no_coll


def convert_initial_states_as_motion_data(data_dict):
    num_tg_agents = data_dict["decoder/modeled_agent_position_for_trafficgen"].shape[1] - 1
    from scenestreamer.tokenization.motion_tokenizers import START_ACTION as MOTION_START_ACTION, get_relative_velocity
    B = data_dict["decoder/agent_position"].shape[0]
    device = data_dict["decoder/agent_position"].device
    data_dict.update(
        {
            # Agent features
            "decoder/agent_position": data_dict["decoder/modeled_agent_position_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents, 2),
            "decoder/modeled_agent_position": data_dict["decoder/modeled_agent_position_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents, 2),
            "decoder/agent_heading": data_dict["decoder/modeled_agent_heading_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents),
            "decoder/modeled_agent_heading": data_dict["decoder/modeled_agent_heading_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents),
            "decoder/agent_velocity": data_dict["decoder/modeled_agent_velocity_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents, 2),
            "decoder/modeled_agent_velocity": data_dict["decoder/modeled_agent_velocity_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents, 2),
            "decoder/agent_valid_mask": data_dict["decoder/input_action_valid_mask_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents),
            "decoder/current_agent_shape": data_dict["decoder/current_agent_shape_for_trafficgen"]
            [:, 1:].reshape(B, num_tg_agents, 3),
            "decoder/modeled_agent_delta": get_relative_velocity(
                vel=data_dict["decoder/modeled_agent_velocity_for_trafficgen"][:, 1:].reshape(B, 1, num_tg_agents, 2),
                heading=data_dict["decoder/modeled_agent_heading_for_trafficgen"][:, 1:].reshape(B, 1, num_tg_agents),
            ),
            "decoder/agent_type": data_dict["decoder/agent_type_for_trafficgen"][:, 1:].reshape(B, num_tg_agents),
            "decoder/agent_id": torch.arange(num_tg_agents, dtype=torch.long).unsqueeze(0).repeat(B, 1).to(device),
            "decoder/agent_shape": data_dict["decoder/current_agent_shape_for_trafficgen"]
            [:, 1:].reshape(B, 1, num_tg_agents, 3),  # This data has temporal information

            # Action
            "decoder/input_action": torch.full([B, 1, num_tg_agents], MOTION_START_ACTION, dtype=torch.long).to(device),
            "decoder/current_agent_valid_mask": torch.full([B, num_tg_agents], True, dtype=torch.bool).to(device),
            "decoder/input_action_valid_mask": torch.full([B, 1, num_tg_agents], True, dtype=torch.bool).to(device),
        }
    )
    return data_dict
