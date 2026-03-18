"""
This module reimplements the autoregressive motion generation process.
"""

import copy

import torch
from scenestreamer.tokenization.motion_tokenizers import END_ACTION
from scenestreamer.tokenization.motion_tokenizers import interpolate, interpolate_heading
from scenestreamer.utils import utils

import numpy as np


def get_motion_tokenizer(model):
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "motion_tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(model, "_tokenizer", None)
    if tokenizer is None:
        raise AttributeError(f"Model of type {type(model).__name__} does not expose a motion tokenizer.")
    return tokenizer

def generate_motion(
    *,
    data_dict,
    model,
    autoregressive_start_step,
    allow_newly_added_agent_step=None,
    temperature=None,
    topp=None,
    num_decode_steps=None,
    sampling_method=None,
    interpolation=True,
    remove_out_of_map_agent=False,
    remove_static_agent=False,
    teacher_forcing_sdc=False,
):
    assert model.training is False, "This function is only for evaluation!"
    data_dict = copy.deepcopy(data_dict)

    if allow_newly_added_agent_step is None:
        allow_newly_added_agent_step = autoregressive_start_step
    assert allow_newly_added_agent_step >= autoregressive_start_step
    tokenizer = get_motion_tokenizer(model)
    if temperature is None:
        temperature = model.config.SAMPLING.TEMPERATURE
    if topp is None:
        topp = model.config.SAMPLING.TOPP
    if sampling_method is None:
        sampling_method = model.config.SAMPLING.SAMPLING_METHOD
    B, T_input, N = data_dict["decoder/input_action"].shape[:3]
    if num_decode_steps is None:
        num_decode_steps = 19
        # assert start_action_step + T_input == num_decode_steps  # Might not be True in waymo test set.
        assert num_decode_steps == 19
        assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
    elif num_decode_steps != 19:
        print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

    # ===== Get initial data =====
    agent_pos = data_dict["decoder/agent_position"][:, ::tokenizer.num_skipped_steps]
    agent_heading = data_dict["decoder/agent_heading"][:, ::tokenizer.num_skipped_steps]
    agent_valid_mask = data_dict["decoder/agent_valid_mask"][:, ::tokenizer.num_skipped_steps]
    agent_velocity = data_dict["decoder/agent_velocity"][:, ::tokenizer.num_skipped_steps]
    B, T_full, N, _ = agent_pos.shape
    gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
    assert agent_pos.ndim == 4
    gt_input_action = data_dict["decoder/input_action"].clone()
    if autoregressive_start_step > 0 or teacher_forcing_sdc:
        gt_target_action = data_dict["decoder/target_action"].clone()
        gt_target_valid_mask = data_dict["decoder/target_action_valid_mask"].clone()

    # ===== Initialize the state =====
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
    if model.config.USE_DESTINATION:
        step_data_dict["agent_destination"] = data_dict["decoder/dest_map_index"][:, :1].clone()
    output_logit_list = []
    output_action_list = []
    input_action_valid_mask_list = [step_data_dict["input_action_valid_mask"]]
    pos = [step_data_dict["agent_position"]]
    head = [step_data_dict["agent_heading"]]
    vel = [step_data_dict["agent_velocity"]]
    decode_error_rate = []

    # ===== Run motion generation =====
    data_dict, _ = encode_scene(data_dict=data_dict, model=model)
    for decode_step in range(num_decode_steps):
        has_gt_step = (autoregressive_start_step > 0 or teacher_forcing_sdc) and decode_step < gt_target_valid_mask.shape[1]

        if decode_step < autoregressive_start_step:
            # Overwrite the action by GT action
            # teacher_forcing_valid_mask = torch.ones_like(step_data_dict["input_action_valid_mask"])
            if has_gt_step:
                teacher_forcing_valid_mask = (
                    step_data_dict["input_action_valid_mask"].clone() & gt_target_valid_mask[:, decode_step:decode_step + 1]
                )
                teacher_forcing_action = gt_target_action[:, decode_step:decode_step + 1]

                assert gt_target_valid_mask[:, decode_step:decode_step + 1][teacher_forcing_valid_mask].all()
            else:
                teacher_forcing_valid_mask = None
                teacher_forcing_action = None

        else:
            teacher_forcing_valid_mask = None
            teacher_forcing_action = None

        if teacher_forcing_sdc:
            assert data_dict["decoder/sdc_index"][0] == 0
            if not has_gt_step:
                teacher_forcing_valid_mask = None
                teacher_forcing_action = None
            else:
                if teacher_forcing_valid_mask is None:
                    teacher_forcing_valid_mask = torch.zeros_like(step_data_dict["input_action_valid_mask"])
                assert teacher_forcing_valid_mask.shape == (B, 1, N)
                teacher_forcing_valid_mask[:, :, 0] = 1
                teacher_forcing_valid_mask = teacher_forcing_valid_mask & step_data_dict["input_action_valid_mask"]
                teacher_forcing_valid_mask = teacher_forcing_valid_mask & gt_target_valid_mask[:,
                                                                                               decode_step:decode_step + 1]
                step_data_dict["agent_valid_mask"] = torch.where(
                    teacher_forcing_valid_mask,
                    step_data_dict["agent_valid_mask"] & gt_target_valid_mask[:, decode_step:decode_step + 1],
                    step_data_dict["agent_valid_mask"]
                )
                step_data_dict["input_action_valid_mask"] = step_data_dict["agent_valid_mask"]
                if teacher_forcing_action is None:
                    teacher_forcing_action = gt_target_action[:, decode_step:decode_step + 1]


        assert step_data_dict["decode_step"] == decode_step

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

        if decode_step < allow_newly_added_agent_step:
            new_agent_valid_mask = agent_valid_mask[:, decode_step + 1:decode_step +
                                                    2] & (~step_data_dict["agent_valid_mask"])
            next_state_data_dict, decode_one_step_info = add_new_agent(
                step_data_dict=next_state_data_dict,
                step_info=decode_one_step_info,
                new_agent_valid_mask=new_agent_valid_mask,
                new_agent_pos=agent_pos[:, decode_step + 1:decode_step + 2, ..., :2],
                new_agent_heading=agent_heading[:, decode_step + 1:decode_step + 2],
                new_agent_velocity=agent_velocity[:, decode_step + 1:decode_step + 2],
                new_agent_delta=gt_agent_delta[:, decode_step + 1:decode_step + 2],
                new_action=gt_input_action[:, decode_step + 1:decode_step + 2],
            )

        pos.append(next_state_data_dict["agent_position"].clone())
        head.append(next_state_data_dict["agent_heading"].clone())
        vel.append(next_state_data_dict["agent_velocity"].clone())
        if decode_one_step_info["output_token"] is not None:
            output_logit_list.append(decode_one_step_info["output_token"].clone())
        output_action_list.append(next_state_data_dict["input_action"].clone())
        input_action_valid_mask_list.append(next_state_data_dict["input_action_valid_mask"].clone())
        step_data_dict = next_state_data_dict
        if "error_rate" in decode_one_step_info:
            decode_error_rate.append(decode_one_step_info["error_rate"])

    # ===== Post-process the data =====

    if output_action_list[0].ndim == 4:
        max_seq_len = max([x.shape[-1] for x in output_action_list])
        output_action_list = [
            torch.nn.functional.pad(output_action_list[i], (0, max_seq_len - output_action_list[i].shape[-1]), value=-1)
            for i in range(len(output_action_list))
        ]
        output_action_list = torch.concatenate(output_action_list, dim=1)

    elif output_action_list[0].ndim == 3:
        output_action_list = torch.concatenate(output_action_list, dim=1)

    else:
        raise ValueError("Invalid output_action_list shape: {}".format(output_action_list[0].shape))

    assert output_action_list.shape[:3] == (B, num_decode_steps, N)
    assert len(input_action_valid_mask_list) == num_decode_steps + 1
    # Evict the last step's input_action_valid_mask_list as it is not used.
    input_action_valid_mask_list = input_action_valid_mask_list[:-1]
    input_action_valid_mask = torch.cat(input_action_valid_mask_list, dim=1)

    if output_logit_list:
        output_logit_list = torch.concatenate(output_logit_list, dim=1)
        traj_log_prob, traj_prob = utils.calculate_trajectory_probabilities_new(
            output_logit_list, output_action_list, mask=input_action_valid_mask
        )  # (B, N)
        data_dict["decoder/output_score"] = traj_log_prob

    else:
        data_dict["decoder/output_score"] = torch.zeros((B, N), dtype=torch.float32, device=output_action_list.device)

    pos = torch.cat(pos, dim=1)
    head = torch.cat(head, dim=1)
    vel = torch.cat(vel, dim=1)

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
    data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
    # data_dict["decoder/debug_ar_pos"] = pos
    # data_dict["decoder/debug_ar_head"] = head
    # data_dict["decoder/debug_ar_vel"] = vel

    if decode_error_rate:
        print("ERROR RATE:", np.mean(decode_error_rate))

    valid_output_action = output_action_list[input_action_valid_mask]
    if valid_output_action.ndim == 1:
        assert valid_output_action.max() < tokenizer.num_actions
        assert valid_output_action.min() >= 0
    elif valid_output_action.ndim == 2:
        assert valid_output_action.max() < tokenizer.num_actions
        if valid_output_action.amax(dim=-1).min() < 0:
            print("WARNING: Invalid action detected in valid_output_action", valid_output_action.amax(dim=-1))

    return data_dict


def encode_scene(*, data_dict, model):
    if "encoder/scenario_token" not in data_dict:
        data_dict = model.encode_scene(data_dict)
    return data_dict, {}


def randomize_agent_id(*, data_dict, model, clip_agent_id=True):
    if "decoder/randomized_modeled_agent_id" not in data_dict:
        data_dict["decoder/randomized_modeled_agent_id"] = model.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=clip_agent_id
        )
    return data_dict, {}


def decode_one_step(
    *,
    data_dict,
    model,
    input_step,
    input_action,
    input_action_valid_mask,
    agent_position,
    agent_heading,
    agent_velocity,
    agent_valid_mask,
    agent_delta,
    agent_shape,
    agent_type,
    agent_id,
    sampling_method,
    temperature,
    topp,
    teacher_forcing_valid_mask,
    teacher_forcing_action,
    agent_destination=None,
    cache=None,
    agent_position_history=None,
    agent_heading_history=None,
    agent_valid_mask_history=None,
    agent_step_history=None,
    remove_out_of_map_agent=False,
    remove_static_agent=False,
    decode_step=None
):
    B = data_dict["decoder/modeled_agent_position"].shape[0]
    if decode_step is None:
        # Older infinite-generation callers only carry `input_step`; recover the decode index from it.
        decode_step = int(input_step.max().item())
    input_dict = {
        # Static encoder features
        "encoder/scenario_token": data_dict["encoder/scenario_token"],
        "encoder/scenario_heading": data_dict["encoder/scenario_heading"],
        "encoder/scenario_position": data_dict["encoder/scenario_position"],
        "encoder/scenario_valid_mask": data_dict["encoder/scenario_valid_mask"],
        "encoder/map_position": data_dict["encoder/map_position"],
        "in_evaluation": torch.ones([B], dtype=torch.bool),

        # Actions
        "decoder/input_step": input_step,
        "decoder/input_action": input_action,
        "decoder/input_action_valid_mask": input_action_valid_mask,

        # Agent features
        "decoder/modeled_agent_position": agent_position,
        "decoder/modeled_agent_heading": agent_heading,
        "decoder/modeled_agent_velocity": agent_velocity,
        "decoder/modeled_agent_valid_mask": agent_valid_mask,
        "decoder/modeled_agent_delta": agent_delta,
        "decoder/current_agent_shape": agent_shape,
        "decoder/agent_type": agent_type,
        "decoder/randomized_modeled_agent_id": agent_id,
    }

    if agent_destination is not None:
        assert decode_step is not None
        # TODO: This is a workaround to update the destination following GT data.
        agent_destination = data_dict["decoder/dest_map_index"][:, decode_step:decode_step + 1]

        input_dict["decoder/dest_map_index"] = agent_destination

    assert (agent_valid_mask == input_action_valid_mask).all()

    if cache is not None:
        input_dict.update(
            {
                "decoder/cache": cache,
                "decoder/modeled_agent_position_history": agent_position_history,
                "decoder/modeled_agent_heading_history": agent_heading_history,
                "decoder/modeled_agent_valid_mask_history": agent_valid_mask_history,
                "decoder/modeled_agent_step_history": agent_step_history,
            }
        )
    assert not (input_action == END_ACTION).any()

    # Some released checkpoints are incompatible with the incremental cache path when decoding
    # the first autoregressive step on CPU/MPS. Disable decoder KV caching here for robustness.
    output_dict = model.decode_motion(input_dict, use_cache=False)

    if model.config.TOKENIZATION.TOKENIZATION_METHOD == "fast":
        selected_action = output_dict["decoder/output_token"]
        output_token = None

        selected_action = selected_action.masked_fill(selected_action >= tokenizer.fast_tokenizer.vocab_size, -1)

        selected_action = torch.where(input_action_valid_mask.unsqueeze(-1), selected_action, -1)

    else:
        output_token = output_dict["decoder/output_logit"]
        selected_action, sampling_info = sample_action(
            logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
        )

        # Remove invalid actions
        # assert selected_action.shape == input_action.shape
        # correct_selected_action = torch.where(input_action_valid_mask, selected_action, -1)
        selected_action = torch.where(input_action_valid_mask, selected_action, -1)

    if teacher_forcing_valid_mask is not None:

        if model.config.TOKENIZATION.TOKENIZATION_METHOD == "fast":

            assert teacher_forcing_action.shape[:3] == selected_action.shape[:3]

            teacher_forcing_action, selected_action = pad_sequences(
                teacher_forcing_action, selected_action, x_value=-1, y_value=-1
            )
            selected_action = torch.where(
                teacher_forcing_valid_mask[..., None], teacher_forcing_action, selected_action
            )

        else:
            assert teacher_forcing_action.shape == selected_action.shape
            selected_action = torch.where(teacher_forcing_valid_mask, teacher_forcing_action, selected_action)
            # correct_selected_action = torch.where(teacher_forcing_valid_mask, teacher_forcing_action, correct_selected_action)
            output_token[teacher_forcing_valid_mask] = 0

    tokenizer = get_motion_tokenizer(model)
    res = tokenizer.detokenize_step(
        current_pos=agent_position,
        current_heading=agent_heading,
        current_valid_mask=agent_valid_mask,
        current_vel=agent_velocity,
        action=selected_action,
        agent_type=agent_type,
    )

    # debug_err = (tokenizer.detokenize_step(
    #     current_pos=agent_position,
    #     current_heading=agent_heading,
    #     current_valid_mask=agent_valid_mask,
    #     current_vel=agent_velocity,
    #     action=selected_action,
    # )['pos'] - tokenizer.detokenize_step(
    #     current_pos=agent_position,
    #     current_heading=agent_heading,
    #     current_valid_mask=agent_valid_mask,
    #     current_vel=agent_velocity,
    #     action=correct_selected_action,
    # )['pos']).norm(dim=-1)
    #
    # assert (debug_err==0).all()

    B, _, N = input_action.shape[:3]
    current_pos = res["pos"].reshape(B, 1, N, 2)
    current_heading = res["heading"].reshape(B, 1, N)
    current_vel = res["vel"].reshape(B, 1, N, 2)
    current_delta = res["delta_pos"].reshape(B, 1, N, 2)
    current_model_step = input_step + 1
    current_input_action = selected_action

    current_valid_mask = agent_valid_mask

    next_step_data_dict = dict(
        input_step=current_model_step,
        input_action=current_input_action,
        input_action_valid_mask=current_valid_mask,
        agent_position=current_pos,
        agent_heading=current_heading,
        agent_velocity=current_vel,
        agent_valid_mask=current_valid_mask,
        agent_delta=current_delta,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_shape=agent_shape,
        cache=output_dict.get("decoder/cache"),
        agent_position_history=output_dict.get("decoder/modeled_agent_position_history"),
        agent_heading_history=output_dict.get("decoder/modeled_agent_heading_history"),
        agent_valid_mask_history=output_dict.get("decoder/modeled_agent_valid_mask_history"),
        agent_step_history=output_dict.get("decoder/modeled_agent_step_history"),
        decode_step=decode_step+1
    )
    if agent_destination is not None:
        next_step_data_dict["agent_destination"] = agent_destination
    info_dict = dict(output_token=output_token)
    if "error_rate_full" in res:
        info_dict["error_rate"] = res["error_rate_full"].mean()

    next_step_data_dict, info_dict = evict_agents(
        data_dict=data_dict,
        step_data_dict=next_step_data_dict,
        step_info_dict=info_dict,
        remove_static_agent=remove_static_agent,
        remove_out_of_map_agent=remove_out_of_map_agent
    )

    assert_motion_step_data_dict(step_data_dict=next_step_data_dict, step_info=info_dict)
    return next_step_data_dict, info_dict


def sample_action(logits, sampling_method, temperature, topp):
    # Sample the action
    info = {}
    if sampling_method == "argmax":
        selected_action = logits.argmax(-1)
    elif sampling_method == "softmax":
        selected_action = torch.distributions.Categorical(logits=logits / temperature).sample()
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
    sampled_token_index = torch.distributions.Categorical(probs=original_probs).sample()
    return sampled_token_index, {"cutoff_index": cutoff_index}


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
    *, data_dict, num_skipped_steps, num_decoded_steps, agent_position, agent_heading, agent_velocity, input_valid_mask
):
    B, _, N, _ = agent_position.shape
    T_generated_chunks = num_decoded_steps
    reconstructed_pos = interpolate(agent_position, num_skipped_steps, remove_first_step=False)
    assert (reconstructed_pos[:, ::5] == agent_position).all()
    reconstructed_heading = interpolate_heading(agent_heading, num_skipped_steps, remove_first_step=False)
    reconstructed_vel = interpolate(agent_velocity, num_skipped_steps, remove_first_step=False)

    valid = input_valid_mask.reshape(B, -1, 1, N).expand(-1, -1, num_skipped_steps, -1).reshape(B, -1, N)
    valid = torch.cat([valid, input_valid_mask[:, -1:]], dim=1)
    reconstructed_valid_mask = valid

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

    step_data_dict["input_action"] = torch.where(new_mask, step_data_dict["input_action"], -1)
    # step_data_dict["agent_position"] = torch.where(new_mask.unsqueeze(-1), agent_position, 0)
    # step_data_dict["agent_heading"] = torch.where(new_mask, step_data_dict["agent_heading"], 0)
    # step_data_dict["agent_velocity"] = torch.where(new_mask.unsqueeze(-1), step_data_dict["agent_velocity"], 0)
    step_data_dict["agent_valid_mask"] = new_mask
    # step_data_dict["agent_delta"] = torch.where(new_mask.unsqueeze(-1), step_data_dict["agent_delta"], 0)
    if step_info_dict.get("output_token") is not None:
        step_info_dict["output_token"] = torch.where(new_mask.unsqueeze(-1), step_info_dict["output_token"], 0)

    step_info_dict["evicted_agents"] = num_evicted
    step_info_dict["evicted_agent_mask"] = should_evict
    assert_motion_step_data_dict(step_data_dict=step_data_dict, step_info=step_info_dict)

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
