import hydra
import copy

import hydra
import numpy as np
import omegaconf
import torch

from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.gradio_ui.plot import plot_pred
from scenestreamer.models.motionlm import sample_action, calculate_trajectory_probabilities
from scenestreamer.tokenization.motion_tokenizers import END_ACTION, START_ACTION
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils


def _reconstruct_delta_pos_from_abs_vel(vel, heading, dt):
    vel = utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)
    pos = vel * dt
    return pos


def mcts_search(
    model,
    data_dict,
    config,
    start_steps,
    num_search_steps,
    num_search_width,
    bin_centers,
):
    """
    This function runs model forward search for a number of steps.
    Then use backpropagation to evaluate the trajectories.
    """
    backward_run_full_length = False
    per_agent_argmax = False
    backward_inference_horizon = 10

    # Do some tricks here to remove useless data.
    for pattern in [
            "eval/",
            "encoder/current_",
            "encoder/future_",
    ]:
        data_dict = {k: v for k, v in data_dict.items() if not k.startswith(pattern)}
    data_dict = {k: v for k, v in data_dict.items() if not k.startswith("encoder/agent_")}
    new_data_dict = {}
    for pattern in ["decoder/agent_id", "decoder/agent_type", "decoder/cache", "decoder/current_",
                    "decoder/modeled_agent_", "decoder/input_", "decoder/target_", "encoder/", "in_evaluation",
                    "batch_idx", "in_backward_prediction", "decoder/randomized_modeled_agent_id"]:
        new_data_dict.update({k: v for k, v in data_dict.items() if k.startswith(pattern)})
    data_dict = new_data_dict

    # To avoid those overwriting operation.
    data_dict = copy.deepcopy(data_dict)

    original_B, original_T, original_N, _ = data_dict["decoder/modeled_agent_position"].shape
    data_dict = {
        k: (
            utils.expand_for_modes(data_dict[k], num_modes=num_search_width)
            if k not in ["decoder/cache", "decoder/input_step", "decoder/modeled_agent_step_history"] else data_dict[k]
        )
        for k in data_dict.keys() if (
            k.startswith("encoder/") or k.startswith("decoder/") or k == "in_evaluation"
            or k == "in_backward_prediction" or k == "batch_idx"
        )
    }
    bin_centers = utils.expand_for_modes(bin_centers, num_modes=num_search_width)
    if "decoder/cache" in data_dict:

        def _expand_cache(tensor, shape):
            # expanding cache is not a easy task.
            # our tensor is only used in A2T attention, where the cache (K, V) is in shape:
            # (BN, T, D)
            # In the first dim the order is: b1n1, b1n2, b2n1, b2n2, ...
            # After expanding, say expanded to W, the shape should be:
            # (BWN, T, D)
            # In the first dim the order should be: b1w1n1, b1w1n2, b1w2n1, b1w2n2, ...
            # However, if we simply repeat dim, will make the shape be:
            # (BNW, T, D), but the order is wrong:
            # b1n1w1, b1n1w2, b1n2w1, b1n2w2, ...
            # So we need to do some reshaping.
            tensor = tensor.reshape(original_B, -1, *tensor.shape[1:])
            tensor = utils.expand_for_modes(tensor, num_modes=num_search_width)
            tensor = tensor.reshape(shape[0] * num_search_width, shape[1], -1)
            return tensor

        def _new_cache(c):
            return [
                _expand_cache(c[0], c[2]),
                _expand_cache(c[1], c[2]),
                (c[2][0] * num_search_width, c[2][1]),
            ]

        data_dict["decoder/cache"] = [_new_cache(v) for v in data_dict["decoder/cache"]]

    # Another trick is to re-randomize the modeled agent id to further improve diversity.
    # data_dict["decoder/randomized_modeled_agent_id"] = model.motion_decoder.randomize_modeled_agent_id(
    #     data_dict["decoder/agent_id"], clip_agent_id=True
    # )
    # The above is wrong. We can't do this because the cache is for original randomized agent id.

    sampling_method = config.SAMPLING.SAMPLING_METHOD
    temperature = config.SAMPLING.TEMPERATURE

    topp = config.SAMPLING.TOPP
    tokenizer = get_tokenizer(config=config)
    assert "encoder/scenario_token" in data_dict

    current_pos = data_dict["decoder/modeled_agent_position"].clone()
    current_heading = data_dict["decoder/modeled_agent_heading"].clone()
    current_vel = data_dict["decoder/modeled_agent_velocity"].clone()
    current_valid_mask = data_dict["decoder/modeled_agent_valid_mask"].clone()
    current_delta = data_dict["decoder/modeled_agent_delta"].clone()
    current_model_step = data_dict["decoder/input_step"].clone()
    assert (current_model_step == start_steps).all()
    current_input_action = data_dict["decoder/input_action"].clone()
    agent_shape = data_dict["decoder/current_agent_shape"].clone()
    agent_type = data_dict["decoder/agent_type"].clone()
    B, T, N, _ = current_pos.shape

    # ===== Run forward prediction to get forward trajectory =====
    if "decoder/modeled_agent_position_history" in data_dict:
        pos = [data_dict["decoder/modeled_agent_position_history"]]
        head = [data_dict["decoder/modeled_agent_heading_history"]]
        vel = [data_dict["decoder/modeled_agent_velocity_history"]]
        # delta = data_dict["decoder/modeled_agent_delta_history"].clone()
    else:
        pos = [current_pos.clone()]
        head = [current_heading.clone()]
        vel = [current_vel.clone()]
        # delta = [current_delta.clone()]
    output_logit_list = []
    output_action_list = [current_input_action.clone()]
    input_action_valid_mask_list = []
    for decode_step in range(num_search_steps):
        # Overwrite all necessary data:
        data_dict["decoder/modeled_agent_position"] = current_pos
        data_dict["decoder/modeled_agent_heading"] = current_heading
        data_dict["decoder/modeled_agent_velocity"] = current_vel
        data_dict["decoder/modeled_agent_valid_mask"] = current_valid_mask
        data_dict["decoder/modeled_agent_delta"] = current_delta
        data_dict["decoder/input_step"] = current_model_step + decode_step
        data_dict["decoder/input_action"] = current_input_action
        data_dict["decoder/input_action_valid_mask"] = current_valid_mask
        assert not (current_input_action == END_ACTION).any()
        assert (data_dict["in_backward_prediction"] == False).all()
        with torch.no_grad():
            data_dict = model.decode_motion(data_dict, use_cache=True)
        output_token = data_dict["decoder/output_logit"]
        selected_action = sample_action(
            logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
        )
        res = tokenizer.detokenize_step(
            current_pos=current_pos,
            current_heading=current_heading,
            current_valid_mask=current_valid_mask,
            current_vel=current_vel,
            action=selected_action,
            agent_shape=data_dict["decoder/current_agent_shape"],
            bin_centers=bin_centers,
            dt=tokenizer.dt,
        )
        recon_next_pos, recon_next_heading, recon_next_vel = res["pos"], res["heading"], res["vel"]

        # TODO: delta_pos computing is updated.
        raise ValueError()
        relative_delta_pos = recon_next_pos.reshape(B, 1, N, 2) - current_pos
        relative_delta_pos = utils.rotate(
            relative_delta_pos[..., 0], relative_delta_pos[..., 1], angle=-recon_next_heading.reshape(B, 1, N)
        )
        current_pos = recon_next_pos.reshape(B, 1, N, 2)
        current_heading = recon_next_heading.reshape(B, 1, N)
        current_vel = recon_next_vel.reshape(B, 1, N, 2)
        current_delta = relative_delta_pos.reshape(B, 1, N, 2)
        current_input_action = selected_action
        pos.append(current_pos.clone())
        head.append(current_heading.clone())
        vel.append(current_vel.clone())
        # delta.append(current_delta.clone())
        output_logit_list.append(output_token.clone())
        output_action_list.append(current_input_action.clone())
    data_dict.pop("decoder/cache")
    output_action_list = torch.concatenate(output_action_list, dim=1)
    output_logit_list = torch.concatenate(output_logit_list, dim=1)
    pos = torch.cat(pos, dim=1)
    head = torch.cat(head, dim=1)
    vel = torch.cat(vel, dim=1)

    # # ===== Backward-tokenize the forward trajectory =====
    current_pos = pos[:, -1:]
    current_heading = head[:, -1:]
    current_vel = vel[:, -1:]
    backward_first_action = torch.full_like(current_input_action, -1)
    backward_first_action[current_valid_mask] = END_ACTION
    backward_actions = [backward_first_action.reshape(B, N)]
    init_delta = _reconstruct_delta_pos_from_abs_vel(current_vel, current_heading + np.pi, dt=tokenizer.dt)
    backward_pos = [current_pos.clone()]
    backward_head = [current_heading.clone()]
    backward_vel = [current_vel.clone()]
    backward_delta = [init_delta.clone()]
    total_forward_steps = pos.shape[1] - 1  # minus one because the first step is already in the pos

    if backward_run_full_length:
        backward_tokenize_steps = min(total_forward_steps, backward_inference_horizon + num_search_steps)
    else:
        backward_tokenize_steps = num_search_steps
    for backward_step in range(backward_tokenize_steps):
        # backward_step = 0, ..., D-1
        forward_next_step = total_forward_steps - backward_step - 1
        # forward_next_step = D-1, ..., 0
        res = tokenizer._tokenize_a_step(
            current_pos=current_pos,
            current_heading=current_heading,
            current_vel=current_vel,
            current_valid_mask=current_valid_mask,
            next_pos=pos[:, forward_next_step:forward_next_step + 1],
            next_heading=head[:, forward_next_step:forward_next_step + 1],
            next_valid_mask=current_valid_mask,
            next_velocity=vel[:, forward_next_step:forward_next_step + 1],
            bin_centers=bin_centers,
            add_noise=False,
            topk=0,
            agent_shape=agent_shape,
            agent_type=agent_type,
            dt=-tokenizer.dt,
        )
        backward_actions.append(res["action"])
        current_pos = res["pos"]
        current_heading = res["heading"]
        current_vel = res["vel"]
        backward_pos.append(current_pos)
        backward_head.append(current_heading)
        backward_vel.append(current_vel)
        backward_delta.append(res["delta_pos"])
    backward_pos = torch.cat(backward_pos, dim=1)
    backward_head = torch.cat(backward_head, dim=1)
    backward_vel = torch.cat(backward_vel, dim=1)
    backward_delta = torch.cat(backward_delta, dim=1)

    # ===== Run backward prediction with teacher forcing =====
    backward_input_action = torch.stack(backward_actions, dim=1)
    backward_input_valid_mask = current_valid_mask.expand(-1, backward_tokenize_steps + 1, -1)
    backward_input_dict = {
        "decoder/modeled_agent_position": backward_pos,
        "decoder/modeled_agent_heading": backward_head,
        "decoder/modeled_agent_velocity": backward_vel,
        # "decoder/modeled_agent_valid_mask": current_valid_mask,
        "decoder/modeled_agent_delta": backward_delta,
        "decoder/input_step": torch.arange(backward_tokenize_steps + 1).to(current_pos.device),
        "decoder/input_action": backward_input_action,
        "decoder/input_action_valid_mask": backward_input_valid_mask,
        "encoder/scenario_token": data_dict["encoder/scenario_token"],
        "encoder/scenario_valid_mask": data_dict["encoder/scenario_valid_mask"],
        "encoder/scenario_position": data_dict["encoder/scenario_position"],
        "encoder/scenario_heading": data_dict["encoder/scenario_heading"],
        "in_backward_prediction": torch.ones_like(data_dict["in_backward_prediction"]),
        "in_evaluation": torch.zeros_like(data_dict["in_evaluation"]),
        "decoder/agent_id": data_dict["decoder/agent_id"],
        "decoder/agent_type": data_dict["decoder/agent_type"],
        "decoder/current_agent_shape": data_dict["decoder/current_agent_shape"],
        "batch_idx": data_dict["batch_idx"],
        # "decoder/randomized_modeled_agent_id": data_dict["decoder/randomized_modeled_agent_id"],
    }
    del data_dict
    with torch.no_grad():
        backward_output_dict = model.decode_motion(backward_input_dict, use_cache=False)

    # ===== Calculate the scores =====
    backward_logit = backward_output_dict["decoder/output_logit"][:, :-1]
    dist = torch.distributions.Categorical(logits=backward_logit / temperature)
    backward_target_action = backward_input_action[:, 1:].clone()
    backward_input_action_mask = backward_input_valid_mask.clone()[:, :-1]
    backward_input_action_mask = backward_input_action_mask & (backward_target_action !=
                                                               END_ACTION) & (backward_target_action != START_ACTION)
    del backward_input_dict
    del backward_output_dict

    # === Use log_prob as the score ===
    backward_target_action[~backward_input_action_mask] = 0
    backward_log_prob = dist.log_prob(backward_target_action)
    backward_log_prob[~backward_input_action_mask] = 0
    assert backward_log_prob.ndim == 3

    backward_entropy = dist.entropy()
    backward_entropy[~backward_input_action_mask] = 0

    # === Forward log_prob ===
    forward_dist = torch.distributions.Categorical(logits=output_logit_list / temperature)
    forward_input_action_mask = current_valid_mask.clone().expand(-1, num_search_steps, -1)
    forward_input_action_mask = forward_input_action_mask & (output_action_list[:, 1:] !=
                                                             END_ACTION) & (output_action_list[:, 1:] != START_ACTION)
    forward_log_prob = forward_dist.log_prob(output_action_list[:, 1:])
    forward_log_prob[~forward_input_action_mask] = 0

    forward_entropy = forward_dist.entropy()
    forward_entropy[~forward_input_action_mask] = 0

    # === Combine forward and backward log_prob ===
    # backward_scores = forward_log_prob.sum(1) + backward_log_prob.sum(1)  # Sum over time

    # # === Normalized scores ===
    # forward_mean = forward_log_prob.mean(dim=1)
    # forward_variance = forward_log_prob.var(dim=1)
    # backward_mean = backward_log_prob.mean(dim=1)
    # backward_variance = backward_log_prob.var(dim=1)
    # backward_scores = (forward_mean + backward_mean) / (1 + forward_variance + backward_variance)

    # === Entropy-regularized scores ===
    backward_scores = (forward_log_prob * forward_entropy).mean(1) + (backward_log_prob * backward_entropy).mean(1)
    # Sum of probs:
    # backward_scores = (forward_log_prob).mean(1) + (backward_log_prob).mean(1)
    # backward_scores = (forward_log_prob).mean(1) * 0

    # # # ===== Use GT data to evaluate the trajectories =====
    # agent_pos = data_dict["decoder/agent_position"][..., :2][:, ::5]
    # if start_steps + num_search_steps >= agent_pos.shape[1]:
    #     gt_final_pos = agent_pos[:, -1]
    #     final_pos = pos[:,  - start_steps + agent_pos.shape[1]]
    #     gt_mask = data_dict["decoder/agent_valid_mask"][:, ::5][:, -1]
    # else:
    #     gt_final_pos = agent_pos[:, start_steps + num_search_steps]
    #     final_pos = pos[:, -1]
    #     gt_mask = data_dict["decoder/agent_valid_mask"][:, ::5][:, start_steps + num_search_steps]
    # error = torch.norm(gt_final_pos - final_pos, dim=-1)
    # gt_mask = gt_mask & current_valid_mask.squeeze(1)
    # error = error * gt_mask
    # backward_scores = -error.sum(-1)
    # backward_scores = backward_scores.reshape(-1, num_search_width)

    # # ===== Another Option, Run backward prediction to get backward ADE =====
    # backward_input_action = torch.stack(backward_actions, dim=1)
    # backward_input_valid_mask = current_valid_mask.expand(-1, num_search_steps+1, -1)
    # backward_input_dict = {
    #     "decoder/modeled_agent_position": backward_pos,
    #     "decoder/modeled_agent_heading": backward_head,
    #     "decoder/modeled_agent_velocity": backward_vel,
    #     "decoder/modeled_agent_valid_mask": current_valid_mask,
    #     "decoder/modeled_agent_delta": backward_delta,
    #     "decoder/input_step": torch.arange(num_search_steps+1).to(current_pos.device),
    #     "decoder/input_action": backward_input_action,
    #     "decoder/input_action_valid_mask": backward_input_valid_mask,
    #     "encoder/scenario_token": data_dict["encoder/scenario_token"],
    #     "encoder/scenario_valid_mask": data_dict["encoder/scenario_valid_mask"],
    #     "encoder/scenario_position": data_dict["encoder/scenario_position"],
    #     "encoder/scenario_heading": data_dict["encoder/scenario_heading"],
    #     "in_backward_prediction": torch.ones_like(data_dict["in_backward_prediction"]),
    #     "in_evaluation": torch.zeros_like(data_dict["in_evaluation"]),
    #     "decoder/agent_id": data_dict["decoder/agent_id"],
    #     "decoder/agent_type": data_dict["decoder/agent_type"],
    #     "decoder/current_agent_shape": data_dict["decoder/current_agent_shape"],
    #     "decoder/randomized_modeled_agent_id": data_dict["decoder/randomized_modeled_agent_id"],
    # }
    # backward_current_pos = backward_pos[:, :1]
    # backward_current_heading = backward_head[:, :1]
    # backward_current_vel = backward_vel[:, :1]
    # backward_current_delta = backward_delta[:, :1]
    # backward_input_action = backward_input_action[:, :1]
    # backward_pos = [current_pos.clone()]
    # backward_head = [current_heading.clone()]
    # backward_vel = [current_vel.clone()]
    # backward_delta = [current_delta.clone()]
    # backward_output_logit_list = []
    # backward_output_action_list = [current_input_action.clone()]
    # backward_input_action_valid_mask_list = []
    # for decode_step in range(num_search_steps):
    #     # Overwrite all necessary data:
    #     backward_input_dict["decoder/modeled_agent_position"] = backward_current_pos
    #     backward_input_dict["decoder/modeled_agent_heading"] = backward_current_heading
    #     backward_input_dict["decoder/modeled_agent_velocity"] = backward_current_vel
    #     backward_input_dict["decoder/modeled_agent_valid_mask"] = current_valid_mask
    #     backward_input_dict["decoder/modeled_agent_delta"] = backward_current_delta
    #     backward_input_dict["decoder/input_step"] = torch.full_like(current_model_step, decode_step)
    #     backward_input_dict["decoder/input_action"] = backward_input_action
    #     backward_input_dict["decoder/input_action_valid_mask"] = current_valid_mask
    #     assert not (current_input_action == START_ACTION).any()
    #     assert (backward_input_dict["in_backward_prediction"] == True).all()
    #     backward_input_dict = model.decode_motion(backward_input_dict, use_cache=True)
    #     output_token = backward_input_dict["decoder/output_logit"]
    #     selected_action = sample_action(
    #         logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
    #     )
    #     res = tokenizer.detokenize_step(
    #         current_pos=current_pos,
    #         current_heading=current_heading,
    #         current_valid_mask=current_valid_mask,
    #         current_vel=current_vel,
    #         action=selected_action,
    #         agent_shape=backward_input_dict["decoder/current_agent_shape"],
    #         bin_centers=bin_centers,
    #         dt=tokenizer.dt,
    #     )
    #     recon_next_pos, recon_next_heading, recon_next_vel = res["pos"], res["heading"], res["vel"]
    #     relative_delta_pos = recon_next_pos.reshape(B, 1, N, 2) - current_pos
    #     relative_delta_pos = utils.rotate(
    #         relative_delta_pos[..., 0], relative_delta_pos[..., 1], angle=-recon_next_heading.reshape(B, 1, N)
    #     )
    #     backward_current_pos = recon_next_pos.reshape(B, 1, N, 2)
    #     backward_current_heading = recon_next_heading.reshape(B, 1, N)
    #     backward_current_vel = recon_next_vel.reshape(B, 1, N, 2)
    #     backward_current_delta = relative_delta_pos.reshape(B, 1, N, 2)
    #     #current_model_step.fill_(decode_step + 1 - start_steps)
    #     backward_input_action = selected_action
    #     backward_pos.append(backward_current_pos.clone())
    #     backward_head.append(backward_current_heading.clone())
    #     backward_vel.append(backward_current_vel.clone())
    #     backward_delta.append(backward_current_delta.clone())
    #     backward_output_logit_list.append(output_token.clone())
    #     backward_output_action_list.append(backward_input_action.clone())
    # backward_output_action_list = torch.concatenate(backward_output_action_list, dim=1)
    # backward_output_logit_list = torch.concatenate(backward_output_logit_list, dim=1)
    # backward_pos = torch.cat(backward_pos, dim=1)
    # backward_head = torch.cat(backward_head, dim=1)
    # backward_vel = torch.cat(backward_vel, dim=1)
    # backward_delta = torch.cat(backward_delta, dim=1)
    #
    # backward_final_pos = backward_pos[:, -1]
    # gt_current_pos = data_dict["decoder/modeled_agent_position"].clone().squeeze(1)
    #
    # # TODO: Can compute contour error here!
    # error = torch.norm(gt_current_pos - backward_final_pos, dim=-1)
    # error = error * current_valid_mask.squeeze(1)
    # scenario_error = error.sum(-1) / current_valid_mask.squeeze(1).sum(-1)
    # scenario_error = scenario_error.reshape(-1, num_search_width)
    # backward_scores = -scenario_error

    if per_agent_argmax:
        backward_scores = backward_scores.reshape(-1, num_search_width, N)

    else:
        score_mask = current_valid_mask.squeeze(1)
        backward_scores = backward_scores.sum(1) / score_mask.sum(-1)  # Avg over agent
        assert backward_scores.ndim == 1
        backward_scores = backward_scores.reshape(-1, num_search_width)

    # ===== Get the best trajectory =====
    best_idx = torch.argmax(backward_scores, dim=1)
    # print("[MCTS Step: {}] Agent 0 Scores: ".format(start_steps), backward_scores[..., 0].cpu().numpy())
    # print("[MCTS Step: {}] Agent 1 Scores: ".format(start_steps), backward_scores[..., 1].cpu().numpy())
    # print("[MCTS Step: {}] Best idx: {}, scores: {}".format(start_steps, best_idx.cpu().numpy(), best_scores.values.cpu().numpy()))

    output_action_tmp = output_action_list[:, 1:]
    output_action = output_action_tmp[:, 0].reshape(original_B, num_search_width, N)

    if per_agent_argmax:
        selected_action = torch.gather(
            output_action, dim=1, index=best_idx[:, None, :].expand(original_B, 1, N)
        ).squeeze(1)

    else:
        selected_action = torch.gather(
            output_action, dim=1, index=best_idx[:, None, None].expand(original_B, 1, N)
        ).squeeze(1)

    return selected_action.reshape(original_B, 1, N), {"scores": backward_scores}


def autoregressive_rollout_with_mcts(
    model,
    data_dict,
    config,
    # num_decode_steps,
    num_decode_steps=None,
    use_cache=True,
    sampling_method="softmax",
    temperature=None,
    topp=None,
    num_modes_for_eval=None,
    **kwargs
):

    raw_data = data_dict
    # To avoid those overwriting operation.
    data_dict = copy.deepcopy(data_dict)

    tokenizer = get_tokenizer(config=config)

    if temperature is None:
        temperature = config.SAMPLING.TEMPERATURE
    if topp is None:
        topp = config.SAMPLING.TOPP

    B, T_input, N = data_dict["decoder/input_action"].shape

    if config.GPT_STYLE:
        start_action_step = 0
        assert T_input == 19
    else:
        start_action_step = 2
        assert T_input == 17
    autoregressive_start_step = 2

    if num_decode_steps is None:
        num_decode_steps = 19
        assert start_action_step + T_input == num_decode_steps
        assert num_decode_steps == 19
        assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
    else:
        print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

    # ===== Get initial data =====
    agent_pos = data_dict["decoder/agent_position"]  # .clone()
    agent_heading = data_dict["decoder/agent_heading"]  # .clone()
    agent_valid_mask = data_dict["decoder/agent_valid_mask"]  # .clone()
    agent_velocity = data_dict["decoder/agent_velocity"]  # .clone()
    agent_shape = data_dict["decoder/current_agent_shape"]  # .clone()
    B, T_full, N, _ = agent_pos.shape
    # TODO: hardcoded
    assert T_full == 91
    assert agent_pos.ndim == 4

    # ===== Skip some steps =====
    agent_pos = agent_pos[:, ::tokenizer.num_skipped_steps]
    agent_heading = agent_heading[:, ::tokenizer.num_skipped_steps]
    agent_valid_mask = agent_valid_mask[:, ::tokenizer.num_skipped_steps]
    agent_velocity = agent_velocity[:, ::tokenizer.num_skipped_steps]
    gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
    T_chunks = agent_pos.shape[1]
    assert T_chunks == 19

    # ===== Build up some variables =====
    # Should note that the modeled_agent_* is starting from t=0 (GPT) and t=10 (non-GPT). So using 0:1 to get the
    # first step for decoder is correct.
    current_pos = data_dict["decoder/modeled_agent_position"][:, :1].clone()
    current_heading = data_dict["decoder/modeled_agent_heading"][:, :1].clone()
    current_vel = data_dict["decoder/modeled_agent_velocity"][:, :1].clone()
    current_valid_mask = data_dict["decoder/input_action_valid_mask"][:, :1].clone()
    current_delta = data_dict["decoder/modeled_agent_delta"][:, :1].clone()
    current_model_step = torch.arange(1).to(current_pos.device)  # it's 0
    gt_input_action = data_dict["decoder/input_action"].clone()
    gt_target_action = data_dict["decoder/target_action"].clone()
    current_input_action = gt_input_action[:, :1].clone()

    output_logit_list = []
    output_action_list = []
    input_action_valid_mask_list = []
    assert use_cache

    pos = []
    head = []
    vel = []

    # Select correct bins:
    agent_type = data_dict["decoder/agent_type"]
    bin_centers = tokenizer.get_bin_centers(agent_type)

    data_dict = model.encode_scene(data_dict)
    data_dict["decoder/randomized_modeled_agent_id"] = model.motion_decoder.randomize_modeled_agent_id(
        data_dict, clip_agent_id=True
    )
    for decode_step in range(num_decode_steps):
        if decode_step == autoregressive_start_step:
            assert (current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step +
                                                           1]).all()
            assert (current_valid_mask == data_dict["decoder/current_agent_valid_mask"][:, None]).all()

        # ===== Fill a lot of stuff =====
        # Overwrite all necessary data:
        data_dict["decoder/modeled_agent_position"] = current_pos
        data_dict["decoder/modeled_agent_heading"] = current_heading
        data_dict["decoder/modeled_agent_velocity"] = current_vel
        data_dict["decoder/modeled_agent_valid_mask"] = current_valid_mask
        data_dict["decoder/modeled_agent_delta"] = current_delta
        data_dict["decoder/input_step"] = current_model_step
        data_dict["decoder/input_action"] = current_input_action
        data_dict["decoder/input_action_valid_mask"] = current_valid_mask
        input_action_valid_mask_list.append(current_valid_mask.clone())
        assert not (current_input_action == END_ACTION).any()

        selected_action, mcts_info = mcts_search(
            model,
            data_dict,
            config,
            start_steps=decode_step,
            num_search_steps=4,
            num_search_width=4,
            bin_centers=bin_centers,
        )

        # Note: Call model after MCTS search so the cache will not be used in MCTS.
        data_dict = model.decode_motion(data_dict, use_cache=use_cache)

        if "decoder/modeled_agent_position_history" in data_dict:
            assert data_dict["decoder/modeled_agent_position_history"].shape[1] == decode_step + 1 - start_action_step
        output_token = data_dict["decoder/output_logit"]
        assert output_token.shape[:3] == (B, 1, N)
        # selected_action = sample_action(
        #     logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
        # )

        if decode_step < autoregressive_start_step:
            # Overwrite the action by GT action
            selected_action = gt_target_action[:, decode_step:decode_step + 1]

        res = tokenizer.detokenize_step(
            current_pos=current_pos,
            current_heading=current_heading,
            current_valid_mask=current_valid_mask,
            current_vel=current_vel,
            action=selected_action,
            agent_shape=data_dict["decoder/current_agent_shape"],
            bin_centers=bin_centers,
            dt=tokenizer.dt,
        )
        recon_next_pos, recon_next_heading, recon_next_vel = res["pos"], res["heading"], res["vel"]

        # TODO: delta_pos computing is updated.
        raise ValueError
        relative_delta_pos = recon_next_pos.reshape(B, 1, N, 2) - current_pos
        relative_delta_pos = utils.rotate(
            relative_delta_pos[..., 0], relative_delta_pos[..., 1], angle=-recon_next_heading.reshape(B, 1, N)
        )

        current_pos = recon_next_pos.reshape(B, 1, N, 2)
        current_heading = recon_next_heading.reshape(B, 1, N)
        current_vel = recon_next_vel.reshape(B, 1, N, 2)
        current_delta = relative_delta_pos.reshape(B, 1, N, 2)
        current_model_step.fill_(decode_step + 1 - start_action_step)
        current_input_action = selected_action

        # Overwrite the data FOR NEXT STEP by the GT data:
        if decode_step < autoregressive_start_step:
            newly_added = agent_valid_mask[:, decode_step + 1:decode_step + 2] & (~current_valid_mask)
            if newly_added.any():
                current_pos[newly_added] = agent_pos[:, decode_step + 1:decode_step + 2, ..., :2][newly_added]
                current_heading[newly_added] = agent_heading[:, decode_step + 1:decode_step + 2][newly_added]
                current_vel[newly_added] = agent_velocity[:, decode_step + 1:decode_step + 2][newly_added]
                current_valid_mask[newly_added] = agent_valid_mask[:, decode_step + 1:decode_step + 2][newly_added]
                current_delta[newly_added] = gt_agent_delta[:, decode_step + 1:decode_step + 2][newly_added]

            # Overwrite the input action by GT action
            current_input_action = gt_input_action[:, decode_step + 1:decode_step + 2]
            output_token = torch.zeros_like(output_token)

        pos.append(current_pos.clone())
        head.append(current_heading.clone())
        vel.append(current_vel.clone())
        output_logit_list.append(output_token.clone())
        output_action_list.append(current_input_action.clone())

    output_action_list = torch.concatenate(output_action_list, dim=1)
    assert output_action_list.shape == (B, num_decode_steps - start_action_step, N)

    output_logit_list = torch.concatenate(output_logit_list, dim=1)
    data_dict["decoder/output_logit"] = output_logit_list
    data_dict["decoder/output_action"] = output_action_list

    # FIXME
    # FIXME
    # FIXME What is the score?
    data_dict["decoder/output_score"] = calculate_trajectory_probabilities(
        output_logit_list, output_action_list, mask=current_valid_mask
    )  # (B, N)

    input_action_valid_mask = torch.cat(input_action_valid_mask_list, dim=1)
    data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask

    data_dict["decoder/debug_ar_pos"] = torch.cat(pos, dim=1)
    data_dict["decoder/debug_ar_head"] = torch.cat(head, dim=1)
    data_dict["decoder/debug_ar_vel"] = torch.cat(vel, dim=1)

    valid_output_action = output_action_list[input_action_valid_mask]
    assert valid_output_action.max() < tokenizer.num_actions
    assert valid_output_action.min() >= 0

    # ===== Debug! rewrite output action by GT =====
    # tokenizer = get_tokenizer(config=self.config)
    # input_dict["decoder/output_action"] = input_dict["decoder/target_action"].clone()
    # fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
    # input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

    return data_dict


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1026_gpt.yaml")
def debug_run_model(config):
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.DATA.SD_PASSTHROUGH = False
    omegaconf.OmegaConf.set_struct(config, True)

    # Load model
    from scenestreamer.utils import utils
    # path = config.pretrain

    model = utils.get_model(config, device="cuda")
    device = model.device

    test_dataset = SceneStreamerDataset(config, "training")
    ddd = iter(test_dataset)

    backward_prediction = False

    search_width = 4

    while True:
        try:
            raw_data_dict = data_dict = next(ddd)

            # Create a new ADV in the data so backward prediction will help us generate it.
            # data_dict = create_new_adv(data_dict)

            from scenestreamer.tokenization import get_tokenizer
            tokenizer = get_tokenizer(config)

            # Force to run backward prediction first to make sure the data is tokenized correctly.
            tok_data_dict, _ = tokenizer.tokenize_numpy_array(
                data_dict,
                backward_prediction=backward_prediction,
            )
            data_dict.update(tok_data_dict)

            input_data_dict = utils.numpy_to_torch(data_dict, device=device)
            # Extend the batch dim:
            input_data_dict = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in input_data_dict.items()
            }
            input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)

            if backward_prediction:
                input_data_dict["in_backward_prediction"] = torch.tensor([1], dtype=bool).to(device)
            else:
                input_data_dict["in_backward_prediction"] = torch.tensor([0], dtype=bool).to(device)

            with torch.no_grad():
                output_dict = autoregressive_rollout_with_mcts(
                    model=model.model,
                    data_dict=input_data_dict,
                    config=config,
                    num_decode_steps=None,
                    sampling_method=config.SAMPLING.SAMPLING_METHOD,
                    temperature=config.SAMPLING.TEMPERATURE,
                )

            output_dict = tokenizer.detokenize(
                output_dict,

                # detokenizing_gt=True,
                detokenizing_gt=False,
                backward_prediction=backward_prediction,
            )

            # Get the first batch
            output_dict = {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()}

            output_dict = {
                k: (v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in output_dict.items()
            }

            raw_data_dict.update(output_dict)
            # plot_pred(raw_data)
            plot_pred(raw_data_dict, show=True)

        except StopIteration:
            break
    print("End")


if __name__ == '__main__':
    # debug()
    # debug_backward_prediction()
    debug_run_model()
