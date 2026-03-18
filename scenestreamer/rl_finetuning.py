import torch

from scenestreamer import utils


def _all_reduce(valid_returns, device, all_gather_func):
    with torch.no_grad():

        if valid_returns is not None:
            local_count = torch.tensor([valid_returns.numel()], device=device, dtype=torch.float32)
            local_sum = torch.tensor([valid_returns.sum()], device=device)
            local_sq_sum = torch.tensor([(valid_returns ** 2).sum()], device=device)
        else:
            local_count = torch.tensor([0], device=device, dtype=torch.float32)
            local_sum = torch.tensor([0], device=device)
            local_sq_sum = torch.tensor([0], device=device)

        # Reduce across all ranks
        global_count = local_count.clone()
        global_sum = local_sum.clone()
        global_sq_sum = local_sq_sum.clone()

        global_count = all_gather_func(global_count).sum()
        global_sum = all_gather_func(global_sum).sum()
        global_sq_sum = all_gather_func(global_sq_sum).sum()

        # Compute global mean and std
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean ** 2)
        global_std = torch.sqrt(global_var.clamp(min=1e-6))  # avoid nan
    return global_mean, global_std


class RLFinetuner:

    def __init__(self, model, all_gather):
        self.model = model
        self.replay_buffer = None
        self.replay_count = 0
        self.all_gather = all_gather

    def rollout(self, data_dict):

        original_B = data_dict["encoder/map_feature"].shape[0]

        # TODO
        num_modes_for_eval = 1

        # Autoregressive rollout
        from scenestreamer.eval.waymo_motion_prediction_evaluator import _repeat_for_modes
        expanded_data_dict = {
            k: _repeat_for_modes(data_dict[k], num_modes=num_modes_for_eval)
            for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("eval/") or k.startswith("decoder/") or k == "batch_idx" or k == "in_evaluation"
                    or k == "in_backward_prediction"
            )
        }

        from scenestreamer.infer.scenestreamer_motion import motion_prediction_task
        expanded_data_dict = motion_prediction_task(
            model=self.model,
            data_dict=expanded_data_dict,
            progress_bar=False,
            use_cache=True,
            keep_output_token=True,
            sampling_method="softmax",
            temperature=1.05,
            teacher_forcing_dest=False,  # TODO
        )

        output_action = expanded_data_dict["model/output_action"][:, :-1]

        # Compute reward
        pred_pos = expanded_data_dict["decoder/reconstructed_position"][:, ::5]
        B, T_pred, N, _ = pred_pos.shape
        pred_head = expanded_data_dict["decoder/reconstructed_heading"][:, ::5]
        pred_valid_mask = expanded_data_dict["decoder/reconstructed_valid_mask"][:, ::5]
        agent_shape = expanded_data_dict["decoder/current_agent_shape"][:, None]
        pred_contour = utils.cal_polygon_contour_torch(
            x=pred_pos[..., 0],
            y=pred_pos[..., 1],
            theta=pred_head,
            width=agent_shape[..., 1].expand(B, T_pred, N),
            length=agent_shape[..., 0].expand(B, T_pred, N)
        )

        gt_pos = _repeat_for_modes(data_dict["decoder/modeled_agent_position"], num_modes=num_modes_for_eval)
        T_gt = gt_pos.shape[1]
        gt_head = _repeat_for_modes(data_dict["decoder/modeled_agent_heading"], num_modes=num_modes_for_eval)
        gt_valid_mask = _repeat_for_modes(data_dict["decoder/input_action_valid_mask"], num_modes=num_modes_for_eval)
        gt_contour = utils.cal_polygon_contour_torch(
            x=gt_pos[..., 0],
            y=gt_pos[..., 1],
            theta=gt_head,
            width=agent_shape[..., 1].expand(B, T_gt, N),
            length=agent_shape[..., 0].expand(B, T_gt, N)
        )
        assert T_pred == T_gt + 1

        pred_contour = pred_contour[:, :-1]
        pred_valid_mask = pred_valid_mask[:, :-1]
        assert pred_contour.shape == gt_contour.shape, (pred_contour.shape, gt_contour.shape)

        error_pos = torch.norm(pred_contour - gt_contour, dim=-1).mean(-1)
        error_pos = error_pos[:, 1:]

        reward_valid_mask = pred_valid_mask[:, 1:] & gt_valid_mask[:, 1:]
        error_pos[~reward_valid_mask] = 0.0

        # Now we have B, T=18, N rewards.
        reward = -error_pos.detach()

        # Get it back in the original shape
        reward = reward.reshape(original_B, num_modes_for_eval, -1, N).detach()
        returns = torch.flip(torch.cumsum(torch.flip(reward, dims=[2]), dim=2), dims=[2]).detach()

        scenestreamer_tokens = expanded_data_dict["scenestreamer_tokens"]
        all_token = scenestreamer_tokens.output_token
        L = scenestreamer_tokens.L
        assert self.model.no_tg
        all_token = all_token.reshape(B, -1, N + L, self.model.d_model)
        motion_token = all_token[:, :, L:]
        if self.model.motion_prenorm is not None:
            motion_token = self.model.motion_prenorm(motion_token)
        motion_logit = self.model.motion_head(motion_token)

        # Get the log probs
        motion_logit = motion_logit[:, :-1]
        motion_logit = motion_logit.reshape(original_B, num_modes_for_eval, -1, N, self.model.num_actions)

        reward_valid_mask = reward_valid_mask.reshape(original_B, num_modes_for_eval, -1, N)
        output_action = output_action.reshape(original_B, num_modes_for_eval, -1, N)

        from scenestreamer.tokenization.motion_tokenizers import START_ACTION as MOTION_START_ACTION
        reward_valid_mask = reward_valid_mask & (output_action != MOTION_START_ACTION)

        if reward_valid_mask.any():
            log_probs = torch.distributions.Categorical(logits=motion_logit[reward_valid_mask]).log_prob(
                output_action[reward_valid_mask]
            )
        else:
            log_probs = torch.distributions.Categorical(logits=motion_logit.flatten()[:1]).log_prob(
                torch.zeros_like(output_action.flatten()[:1])
            ) * 0.0

        adv_mean, adv_std = _all_reduce(returns[reward_valid_mask], device=returns.device,
                                        all_gather_func=self.all_gather)

        advantages = (returns[reward_valid_mask] - adv_mean) / (adv_std + 1e-5)

        # Also do reward for traffic light:
        tl_token = all_token[:, :, :L]
        tl_token = self.model.traffic_light_prenorm(tl_token)
        traffic_light_logit = self.model.traffic_light_head(tl_token)
        traffic_light_gt = _repeat_for_modes(data_dict["encoder/traffic_light_state"], num_modes_for_eval)
        traffic_light_mask = _repeat_for_modes(data_dict["encoder/traffic_light_valid_mask"], num_modes_for_eval)
        pred_tl_state = expanded_data_dict["model/traffic_light_state"]

        pred_tl_state = pred_tl_state[:, :-1]
        traffic_light_logit = traffic_light_logit[:, :-1]
        gt_tl_state = traffic_light_gt[:, 1:]
        traffic_light_mask = traffic_light_mask[:, 1:]

        traffic_light_logit = traffic_light_logit.reshape(original_B, num_modes_for_eval, -1, L,
                                                          traffic_light_logit.shape[-1])
        pred_tl_state = pred_tl_state.reshape(original_B, num_modes_for_eval, -1, L)
        traffic_light_mask = traffic_light_mask.reshape(original_B, num_modes_for_eval, -1, L)
        gt_tl_state = gt_tl_state.reshape(original_B, num_modes_for_eval, -1, L)

        if traffic_light_mask.any():
            tl_log_probs = torch.distributions.Categorical(logits=traffic_light_logit[traffic_light_mask]).log_prob(
                pred_tl_state[traffic_light_mask])

            # For TL, we normalize advantage across the whole batch.
            # This is because it's easy to have model predict all 4 same actions, then adv=0.
            tl_reward = (pred_tl_state == gt_tl_state).float()
            tl_reward = tl_reward.reshape(original_B, num_modes_for_eval, -1, L).detach()
            tl_return = torch.flip(torch.cumsum(torch.flip(tl_reward, dims=[2]), dim=2), dims=[2]).detach()

            # print("RANK {}, before all reduce. tl return {}, tl return valid shape {}".format(
            #     self.global_rank,
            #     tl_return.shape,
            #     tl_return[traffic_light_mask].shape
            # ))

            tl_adv_mean, tl_adv_std = _all_reduce(tl_return[traffic_light_mask], device=tl_return.device,
                                                  all_gather_func=self.all_gather)
            # all_gather_tl_returns = self.all_gather(tl_return)
            # all_gather_tl_reward_valid_mask = self.all_gather(traffic_light_mask)
            # tl_returns_valid = all_gather_tl_returns[all_gather_tl_reward_valid_mask]
            # print(
            #     "RANK {}, ALL GATHER REWARD VALID MASK SHAPE: {}, {}. RETURN VALID {}. GLOBAL MEAN {}, STD {}. Local mean {}".format(
            #         self.global_rank,
            #         None,
            #         None,
            #         None, tl_adv_mean, tl_adv_std, tl_return[traffic_light_mask].mean()))

            tl_return = tl_return[traffic_light_mask].detach()
            tl_advantages = (tl_return - tl_adv_mean) / (tl_adv_std + 1e-5)
            tl_reward = tl_reward[traffic_light_mask].detach()
            tl_entropy = utils.safe_entropy(traffic_light_logit[traffic_light_mask])
            tl_accuracy = (
                    traffic_light_logit[traffic_light_mask].argmax(-1) == pred_tl_state[traffic_light_mask]
            ).float().mean()
        else:

            # print("NO TL PASS")
            # You have to keep this line for the case when there is no traffic light:
            tl_adv_mean, tl_adv_std = _all_reduce(None, device=traffic_light_logit.device, all_gather_func=self.all_gather)
            tl_log_probs = torch.distributions.Categorical(logits=traffic_light_logit.flatten(0, 2)[:1]).log_prob(
                pred_tl_state.flatten()[:1]) * 0.0
            tl_advantages = torch.zeros_like(tl_log_probs)
            tl_reward = torch.zeros_like(tl_log_probs)
            tl_return = torch.zeros_like(tl_log_probs)
            tl_entropy = torch.zeros_like(tl_log_probs)
            tl_accuracy = torch.zeros_like(tl_log_probs)

        return dict(
            # motion_logit=motion_logit,
            log_probs=log_probs,
            advantages=advantages,
            data_dict=expanded_data_dict,

            tl_entropy=tl_entropy,
            tl_log_probs=tl_log_probs,
            tl_advantages=tl_advantages,
            tl_reward=tl_reward,
            tl_return=tl_return,
            tl_accuracy=tl_accuracy,

            motion_logit=motion_logit,
            pred_valid_mask=pred_valid_mask,
            output_action=output_action,
            reward_valid_mask=reward_valid_mask,
            reward=reward,
            returns=returns,
            pred_pos=pred_pos,
            pred_head=pred_head,

        )

    def get_loss(self, data_dict):

        # if self.replay_buffer is None or self.replay_count >= 5:
        # with torch.no_grad():
        self.replay_buffer = self.rollout(data_dict)
        self.replay_count = 0

        # log_probs = self.get_log_probs(self.replay_buffer["data_dict"])

        log_probs = self.replay_buffer["log_probs"]
        advantages = self.replay_buffer["advantages"]

        # GRPO Loss:
        def grpo(new_log_prob, old_log_prob, advantages):
            clip_eps = 0.2
            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
            loss = -torch.min(surr1, surr2)  # + self.kl_weight * kl
            return loss.mean()

        # REINFORCE loss:
        # motion_loss = -log_probs * advantages.detach()
        # motion_loss = motion_loss.mean()
        motion_loss = grpo(new_log_prob=log_probs, old_log_prob=log_probs.detach(), advantages=advantages.detach())

        tl_log_probs = self.replay_buffer["tl_log_probs"]
        tl_advantages = self.replay_buffer["tl_advantages"]

        tl_loss = grpo(new_log_prob=tl_log_probs, old_log_prob=tl_log_probs.detach(), advantages=tl_advantages.detach())
        # tl_loss = -tl_log_probs * tl_advantages.detach()
        # tl_loss = tl_loss.mean()

        loss = motion_loss + tl_loss

        motion_logit = self.replay_buffer["motion_logit"]
        pred_valid_mask = self.replay_buffer["pred_valid_mask"]
        output_action = self.replay_buffer["output_action"]
        reward_valid_mask = self.replay_buffer["reward_valid_mask"]
        reward = self.replay_buffer["reward"]
        returns = self.replay_buffer["returns"]
        pred_pos = self.replay_buffer["pred_pos"]
        pred_head = self.replay_buffer["pred_head"]
        tl_reward = self.replay_buffer["tl_reward"]
        tl_return = self.replay_buffer["tl_return"]

        tl_entropy = self.replay_buffer["tl_entropy"]
        tl_accuracy = self.replay_buffer["tl_accuracy"]

        loss_stat = {
            "motion_loss": motion_loss,
            "motion_accuracy": (
                        motion_logit[reward_valid_mask].argmax(-1) == output_action[reward_valid_mask]).float().mean(),
            "motion_entropy": utils.safe_entropy(motion_logit[reward_valid_mask]).mean(),
            "motion_reward": reward[reward_valid_mask].mean(),
            "motion_return": returns.mean(),
            "motion_advantages": advantages.mean(),
            "total_loss": loss,

            "traffic_light_loss": tl_loss,
            "traffic_light_advantages": tl_advantages.mean(),
            "traffic_light_reward": tl_reward.mean(),
            "traffic_light_return": tl_return.mean(),
            "traffic_light_entropy": tl_entropy.mean(),
            "traffic_light_accuracy": tl_accuracy,
        }
        return loss, loss_stat
