import copy
import logging

import torch
import torch.nn as nn

from scenestreamer.models.gpt_scene_encoder import SceneEncoderGPT
from scenestreamer.models.layers import common_layers
from scenestreamer.models.motion_decoder import MotionDecoder
from scenestreamer.models.motion_decoder_gpt import MotionDecoderGPT
from scenestreamer.models.motion_decoder_gpt_diffusion import MotionDecoderGPTDiffusion
from scenestreamer.models.scene_encoder import SceneEncoder
from scenestreamer.models.trafficgen_decoder import TrafficGenDecoder
from scenestreamer.tokenization import get_tokenizer, SPECIAL_VALID, SPECIAL_START, END_ACTION, START_ACTION
from scenestreamer.utils import calculate_trajectory_probabilities, utils

logger = logging.getLogger(__file__)


def get_relative_velocity(vel, heading):
    return utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)


def _reconstruct_delta_pos_from_abs_vel(vel, heading, dt):
    vel = utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)
    pos = vel * dt
    return pos


def nucleus_sampling(logits, p=None, epsilon=1e-8):
    p = p or 0.9

    # logits = logits.clamp(-20, 20)

    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits).fill_(-1e9), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits).fill_(-1e9), logits)

    # Adding a small epsilon to logits to avoid log(0)
    # logits = logits + epsilon

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

    # original_probs += epsilon

    # Sample from the adjusted probability distribution
    # try:
    sampled_token_index = torch.distributions.Categorical(probs=original_probs).sample()
    # except ValueError:
    #     import ipdb; ipdb.set_trace()
    #     print(1111111)

    return sampled_token_index, {"cutoff_index": cutoff_index}


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


class MotionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = get_tokenizer(config=self.config)

        if self.config.MODEL.NAME == "motionlm":
            self.scene_encoder = SceneEncoder(config=self.config)
            self.motion_decoder = MotionDecoder(config=self.config)
        elif self.config.MODEL.NAME == "gpt":
            self.scene_encoder = SceneEncoderGPT(config=self.config)

            if self.config.USE_TRAFFICGEN:
                self.trafficgen_decoder = TrafficGenDecoder(config=self.config)

            if self.config.USE_MOTION:
                # TODO: For simplicity, remove motion for now if we want to train TG.
                if self.config.USE_DIFFUSION:
                    self.motion_decoder = MotionDecoderGPTDiffusion(config=self.config)
                else:

                    if self.config.TOKENIZATION.TOKENIZATION_METHOD == "fast":
                        from scenestreamer.models.motion_decoder_gpt_fast import MotionDecoderGPT as MotionDecoderGPTFast
                        self.motion_decoder = MotionDecoderGPTFast(config=self.config)
                    else:
                        self.motion_decoder = MotionDecoderGPT(config=self.config)

            assert (self.config.USE_TRAFFICGEN or self.config.USE_MOTION)

        else:
            raise ValueError(f"Unknown model name: {self.config.MODEL.NAME}")

        if self.config.RECONSTRUCT_MAP:
            d_model = self.scene_encoder.d_model
            map_feat_dim = self.config.PREPROCESSING.MAX_VECTORS
            self.map_recon_head = common_layers.build_mlps(
                c_in=d_model, mlp_channels=[d_model, map_feat_dim * 2], ret_before_act=True
            )
            self.map_recon_head_prenorm = nn.LayerNorm(d_model)

    def encode_scene(self, input_dict):
        return self.scene_encoder(input_dict)

    def decode_motion(self, *args, **kwargs):
        input_dict = self.motion_decoder(*args, **kwargs)
        return input_dict

    # def decode_trafficgen(self, *args, **kwargs):
    #     input_dict = self.trafficgen_decoder(*args, **kwargs)
    #     return input_dict
    #
    # def decode_trafficgen_offset(self, *args, **kwargs):
    #     input_dict = self.trafficgen_decoder.forward_offset(*args, **kwargs)
    #     return input_dict

    def forward(self, input_dict):
        input_dict = self.encode_scene(input_dict)

        if self.config.USE_MOTION:
            input_dict = self.decode_motion(input_dict)

        return input_dict

    def autoregressive_rollout(
        self,
        data_dict,
        # num_decode_steps,
        num_decode_steps=None,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None,
        autoregressive_start_step=2,
        **kwargs
    ):

        assert self.training is False, "This function is only for evaluation!"

        if "backward_prediction" in kwargs and kwargs["backward_prediction"]:
            return self.autoregressive_rollout_backward_prediction(
                data_dict=data_dict,
                num_decode_steps=num_decode_steps,
                use_cache=use_cache,
                sampling_method=sampling_method,
                temperature=temperature,
                topp=topp,
                num_modes_for_eval=num_modes_for_eval,
                flip_heading_accordingly=kwargs.get("flip_heading_accordingly", True),
            )

        if self.config.USE_DIFFUSION:
            return self.autoregressive_rollout_diffusion(
                data_dict,
                num_decode_steps=num_decode_steps,
                use_cache=use_cache,
                sampling_method=sampling_method,
                temperature=temperature,
                topp=topp,
                num_modes_for_eval=num_modes_for_eval,
                **kwargs
            )

        raw_data = data_dict
        # To avoid those overwriting operation.
        data_dict = copy.deepcopy(data_dict)

        tokenizer = self.tokenizer

        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = data_dict["decoder/input_action"].shape

        if self.config.GPT_STYLE:
            start_action_step = 0
            # assert T_input == 19  # Might not be True in waymo test set.
        else:
            start_action_step = 2
            assert T_input == 17

        if num_decode_steps is None:
            num_decode_steps = 19
            # assert start_action_step + T_input == num_decode_steps  # Might not be True in waymo test set.
            assert num_decode_steps == 19
            assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
        else:
            print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"]  #.clone()
        agent_heading = data_dict["decoder/agent_heading"]  #.clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  #.clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  #.clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  #.clone()
        B, T_full, N, _ = agent_pos.shape
        # TODO: hardcoded
        # assert T_full == 91  # Might not be True in waymo test set.
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::tokenizer.num_skipped_steps]
        agent_heading = agent_heading[:, ::tokenizer.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::tokenizer.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::tokenizer.num_skipped_steps]
        gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
        # T_chunks = agent_pos.shape[1]
        # assert T_chunks == 19  # Might not be True in waymo test set.

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
        if autoregressive_start_step > 0:
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

        if "encoder/scenario_token" not in data_dict:
            data_dict = self.encode_scene(data_dict)

        data_dict["decoder/randomized_modeled_agent_id"] = self.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=True
        )

        detokenization_state = None
        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            if decode_step < start_action_step:
                # For non-gpt model, skip first 2 steps.
                pos.append(agent_pos[:, decode_step:decode_step + 1, ..., :2])
                head.append(agent_heading[:, decode_step:decode_step + 1])
                vel.append(agent_velocity[:, decode_step:decode_step + 1])
                continue

            if decode_step == autoregressive_start_step:
                assert (
                    current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step + 1]
                ).all()
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

            use_mcts = self.config.MCTS.USE_MCTS
            if use_mcts:
                from scenestreamer.mcts import mcts_search
                selected_action, mcts_info = mcts_search(
                    self,
                    data_dict,
                    self.config,
                    start_steps=decode_step,
                    num_search_steps=self.config.MCTS.MCTS_DEPTH,  # D
                    num_search_width=self.config.MCTS.MCTS_WIDTH,  # W
                    bin_centers=bin_centers,
                )

            # Decode motion tokens
            data_dict = self.decode_motion(data_dict, use_cache=use_cache)

            if "decoder/modeled_agent_position_history" in data_dict:
                assert data_dict["decoder/modeled_agent_position_history"].shape[
                    1] == decode_step + 1 - start_action_step

            output_token = data_dict["decoder/output_logit"]
            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)

            if use_mcts:
                pass
            else:
                selected_action, sampling_info = sample_action(
                    logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
                )

            # avg_left_rate = utils.masked_average((~sampling_info["cutoff_index"]).float().mean(-1), current_valid_mask, dim=-1).mean()
            # avg_left_num = avg_left_rate*output_token.shape[-1]

            # print("With TOPP {:.2f}, TEMPERATURE {:.2f}, AVG_LEFT_RATE {:.2f}, AVG_LEFT_NUM {:.2f}".format(
            #     topp, temperature, avg_left_rate, avg_left_num
            # ))

            if decode_step < autoregressive_start_step:
                # Overwrite the action by GT action
                selected_action = gt_target_action[:, decode_step:decode_step + 1]

            # if self.config.MODEL.RELATIVE_PE_DECODER:
            res = tokenizer.detokenize_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=selected_action,
                agent_shape=agent_shape,
                bin_centers=bin_centers,
                dt=tokenizer.dt,
                flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
                detokenization_state=detokenization_state
            )
            detokenization_state = res
            recon_next_pos, recon_next_heading, recon_next_vel, relative_delta_pos = res["pos"], res["heading"], res[
                "vel"], res["delta_pos"]

            # break

            # Just for fun, detect collision:
            #
            #
            # from shapely.geometry import Polygon
            #
            # contours = utils.cal_polygon_contour_torch(
            #     x=recon_next_pos[..., 0],
            #     y=recon_next_pos[..., 1],
            #     theta=recon_next_heading,
            #     width=data_dict["decoder/current_agent_shape"][..., 1],
            #     length=data_dict["decoder/current_agent_shape"][..., 0],
            # )
            # def detect_collision(contour_list, mask):
            #     collision_detected = torch.zeros_like(mask, dtype=torch.bool)
            #     for i in range(len(contour_list)):
            #         for j in range(i + 1, len(contour_list)):
            #             if mask[i] and mask[j]:
            #                 poly1 = Polygon(contour_list[i].cpu().numpy())
            #                 poly2 = Polygon(contour_list[j].cpu().numpy())
            #                 if poly1.intersects(poly2):
            #                     collision_detected[i] = True
            #                     collision_detected[j] = True
            #     return collision_detected
            #
            # collision_detected = torch.stack(
            #     [detect_collision(contour_list=contours[b], mask=current_valid_mask.squeeze(1)[b]) for  b in range(B)],
            #     dim=0
            # )
            # print("Iter:", iteration)
            # if collision_detected.any():
            #     print("Collision detected!")
            #     B, T, N, D_actions = output_token.shape
            #
            #     # Create a one hot where selected action is 1
            #     should_mask = torch.nn.functional.one_hot(selected_action, num_classes=output_token.shape[-1]).bool()
            #     should_mask = should_mask & collision_detected.reshape(B, 1, N, 1).bool()
            #     output_token = torch.where(should_mask, float("-inf") * torch.ones_like(output_token), output_token)
            #     continue
            # else:
            #     break

            current_pos = recon_next_pos.reshape(B, 1, N, 2)
            current_heading = recon_next_heading.reshape(B, 1, N)
            current_vel = recon_next_vel.reshape(B, 1, N, 2)
            current_delta = relative_delta_pos.reshape(B, 1, N, 2)
            current_model_step = torch.full_like(current_model_step, decode_step + 1 - start_action_step)
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
        data_dict["decoder/output_score"] = utils.calculate_trajectory_probabilities(
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
        # input_dict["decoder/output_action"] = input_dict["decoder/target_action"].clone()
        # fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
        # input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

        return data_dict

    def autoregressive_rollout_with_replay(
        self,
        data_dict,
        # num_decode_steps,
        num_decode_steps=None,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None,
        teacher_forcing_ids=None,
        **kwargs
    ):

        if "backward_prediction" in kwargs and kwargs["backward_prediction"]:
            raise ValueError("Not implemented yet!")

        if self.config.USE_DIFFUSION:
            raise ValueError("Not implemented yet!")

        raw_data = data_dict
        # To avoid those overwriting operation.
        data_dict = copy.deepcopy(data_dict)

        tokenizer = self.tokenizer

        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = data_dict["decoder/input_action"].shape

        if self.config.GPT_STYLE:
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

        # ===== Build up some variables =====
        # Should note that the modeled_agent_* is starting from t=0 (GPT) and t=10 (non-GPT). So using 0:1 to get the
        # first step for decoder is correct.
        assert teacher_forcing_ids is not None
        sdc_id = int(data_dict['decoder/sdc_index'])
        sdc_adv_ids = torch.zeros_like(data_dict["decoder/agent_id"], dtype=torch.bool)
        for i in teacher_forcing_ids:
            sdc_adv_ids[:, i] = True
        sdc_adv_ids = sdc_adv_ids.reshape(B, 1, N)
        assert B == 1, "To avoid the confusion, we only support B=1"

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"]  #.clone()
        agent_heading = data_dict["decoder/agent_heading"]  #.clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"].clone()  #.clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  #.clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  #.clone()
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

        data_dict = self.encode_scene(data_dict)
        data_dict["decoder/randomized_modeled_agent_id"] = self.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=True
        )
        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            # if decode_step < start_action_step:
            #     # For non-gpt model, skip first 2 steps.
            #     pos.append(agent_pos[:, decode_step:decode_step + 1, ..., :2])
            #     head.append(agent_heading[:, decode_step:decode_step + 1])
            #     vel.append(agent_velocity[:, decode_step:decode_step + 1])
            #     continue

            if decode_step == autoregressive_start_step:
                assert (
                    current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step + 1]
                ).all()
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

            # Decode motion tokens
            data_dict = self.decode_motion(data_dict, use_cache=use_cache)

            if "decoder/modeled_agent_position_history" in data_dict:
                assert data_dict["decoder/modeled_agent_position_history"].shape[
                    1] == decode_step + 1 - start_action_step

            output_token = data_dict["decoder/output_logit"]
            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)

            # success = False
            # iteration = 0
            # while True:
            # iteration += 1
            selected_action = sample_action(
                logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
            )

            # We only overwrite SDC and ADV in Forward.
            # TODO: We might want to allow SDC to be free.
            selected_action[sdc_adv_ids] = gt_input_action[:, decode_step:decode_step + 1][sdc_adv_ids]

            # if decode_step < autoregressive_start_step:
            #     # Overwrite the action by GT action
            #     selected_action = gt_target_action[:, decode_step:decode_step + 1]

            # if self.config.MODEL.RELATIVE_PE_DECODER:
            res = tokenizer.detokenize_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=selected_action,
                agent_shape=agent_shape,
                bin_centers=bin_centers,
                dt=tokenizer.dt,
                flip_wrong_heading=True,  # TODO: Dirty workaround only used in AR with Replay!
            )
            recon_next_pos, recon_next_heading, recon_next_vel, relative_delta_pos = res["pos"], res["heading"], res[
                "vel"], res["delta_pos"]

            current_pos = recon_next_pos.reshape(B, 1, N, 2)
            current_heading = recon_next_heading.reshape(B, 1, N)
            current_vel = recon_next_vel.reshape(B, 1, N, 2)
            current_delta = relative_delta_pos.reshape(B, 1, N, 2)
            current_model_step = torch.full_like(current_model_step, decode_step + 1 - start_action_step)
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

            # ===== Teacher Forcing =====
            if decode_step < T_chunks - 1:
                current_pos[sdc_adv_ids] = agent_pos[:, decode_step + 1:decode_step + 2, ..., :2][sdc_adv_ids]
                current_heading[sdc_adv_ids] = agent_heading[:, decode_step + 1:decode_step + 2][sdc_adv_ids]
                current_vel[sdc_adv_ids] = agent_velocity[:, decode_step + 1:decode_step + 2][sdc_adv_ids]
                current_valid_mask[sdc_adv_ids] = agent_valid_mask[:, decode_step + 1:decode_step + 2][sdc_adv_ids]
                current_delta[sdc_adv_ids] = gt_agent_delta[:, decode_step + 1:decode_step + 2][sdc_adv_ids]

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
        # data_dict["decoder/output_score"] = calculate_trajectory_probabilities(
        #     output_logit_list, output_action_list, mask=current_valid_mask
        # )  # (B, N)

        input_action_valid_mask = torch.cat(input_action_valid_mask_list, dim=1)

        # invalid = output_action_list == -1
        # input_action_valid_mask[invalid] = False
        # invalid = output_action_list == START_ACTION
        # input_action_valid_mask[invalid] = False

        data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask

        data_dict["decoder/debug_ar_pos"] = torch.cat(pos, dim=1)
        data_dict["decoder/debug_ar_head"] = torch.cat(head, dim=1)
        data_dict["decoder/debug_ar_vel"] = torch.cat(vel, dim=1)

        valid_output_action = output_action_list[input_action_valid_mask]

        assert valid_output_action.max() <= START_ACTION
        assert valid_output_action.min() >= 0

        # ===== Debug! rewrite output action by GT =====
        # input_dict["decoder/output_action"] = input_dict["decoder/target_action"].clone()
        # fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
        # input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

        return data_dict

    def autoregressive_rollout_backward_prediction(
        self,
        data_dict,
        # num_decode_steps,
        num_decode_steps=None,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        flip_heading_accordingly=True,
        num_modes_for_eval=None,
        **kwargs
    ):

        if self.config.USE_DIFFUSION:
            raise ValueError()

        raw_data = data_dict
        # To avoid those overwriting operation.
        data_dict = copy.deepcopy(data_dict)

        tokenizer = self.tokenizer

        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = data_dict["decoder/input_action"].shape

        assert self.config.GPT_STYLE
        start_action_step = 0
        assert T_input == 19
        # else:
        #     start_action_step = 2
        #     assert T_input == 17
        # autoregressive_start_step = 2

        if num_decode_steps is None:
            num_decode_steps = 19
            assert start_action_step + T_input == num_decode_steps
            assert num_decode_steps == 19
            assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
        else:
            print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"]  #.clone()
        agent_heading = data_dict["decoder/agent_heading"]  #.clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  #.clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  #.clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  #.clone()
        B, T_full, N, _ = agent_pos.shape
        # TODO: hardcoded
        assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::tokenizer.num_skipped_steps]
        agent_heading = agent_heading[:, ::tokenizer.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::tokenizer.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::tokenizer.num_skipped_steps]
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

        import numpy as np

        pos = []
        head = []
        vel = []

        # Select correct bins:
        agent_type = data_dict["decoder/agent_type"]
        bin_centers = tokenizer.get_bin_centers(agent_type)

        data_dict = self.encode_scene(data_dict)
        data_dict["decoder/randomized_modeled_agent_id"] = self.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=True
        )
        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            # TODO: put back the following code
            # if decode_step == autoregressive_start_step:
            #     assert (
            #         current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step + 1]
            #     ).all()
            #     assert (current_valid_mask == data_dict["decoder/current_agent_valid_mask"][:, None]).all()

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

            assert not (current_input_action == START_ACTION).any()

            # Decode motion tokens
            data_dict = self.decode_motion(data_dict, use_cache=use_cache)

            if "decoder/modeled_agent_position_history" in data_dict:
                assert data_dict["decoder/modeled_agent_position_history"].shape[
                    1] == decode_step + 1 - start_action_step

            output_token = data_dict["decoder/output_logit"]
            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)
            selected_action, info = sample_action(
                logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
            )

            # if decode_step < autoregressive_start_step:
            #     # Overwrite the action by GT action
            #     selected_action = gt_target_action[:, decode_step:decode_step + 1]

            # if self.config.MODEL.RELATIVE_PE_DECODER:
            res = tokenizer.detokenize_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=selected_action,
                agent_shape=agent_shape,
                bin_centers=bin_centers,

                # dt=tokenizer.dt,
                dt=-tokenizer.dt,
                flip_heading_accordingly=flip_heading_accordingly,
                flip_wrong_heading=True
            )
            recon_next_pos, recon_next_heading, recon_next_vel, relative_delta_pos = res["pos"], res["heading"], res[
                "vel"], res["delta_pos"]

            current_pos = recon_next_pos.reshape(B, 1, N, 2)
            current_heading = recon_next_heading.reshape(B, 1, N)
            current_vel = recon_next_vel.reshape(B, 1, N, 2)
            current_delta = relative_delta_pos.reshape(B, 1, N, 2)
            # current_model_step.fill_(decode_step + 1 - start_action_step)
            current_model_step = torch.full_like(current_model_step, decode_step + 1 - start_action_step)
            current_input_action = selected_action

            # Overwrite the data FOR NEXT STEP by the GT data:
            # if decode_step < autoregressive_start_step:
            # Always adding new agents
            if True:
                # decode_step = 0, ..., 18
                forward_current_step = T_chunks - decode_step - 1
                # forward_current_step = 18, ..., 0
                forward_next_step = forward_current_step - 1
                # forward_next_step = 17, ..., 0

                newly_added = agent_valid_mask[:, forward_next_step:forward_next_step + 1] & (~current_valid_mask)
                if newly_added.any():
                    current_pos[newly_added] = agent_pos[:, forward_next_step:forward_next_step + 1,
                                                         ..., :2][newly_added]
                    current_heading[newly_added] = agent_heading[:,
                                                                 forward_next_step:forward_next_step + 1][newly_added]
                    current_vel[newly_added] = agent_velocity[:, forward_next_step:forward_next_step + 1][newly_added]
                    current_valid_mask[newly_added] = agent_valid_mask[:, forward_next_step:forward_next_step +
                                                                       1][newly_added]

                    if self.config.DELTA_POS_IS_VELOCITY:

                        current_delta[newly_added] = get_relative_velocity(
                            agent_velocity[:, forward_next_step:forward_next_step + 1][newly_added],
                            agent_heading[:, forward_next_step:forward_next_step + 1][newly_added]
                        )

                    else:
                        current_delta[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                            vel=current_vel[newly_added],

                            # heading=current_heading[newly_added],
                            heading=current_heading[newly_added] + np.pi,
                            dt=tokenizer.dt
                        )

                    # Overwrite the input action by GT action
                    current_input_action[newly_added] = gt_input_action[:, decode_step + 1:decode_step + 2][newly_added]
                    output_token[newly_added] = 0.0

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
        # input_dict["decoder/output_action"] = input_dict["decoder/target_action"].clone()
        # fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
        # input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

        return data_dict

    def autoregressive_rollout_backward_prediction_with_replay(
        self,
        data_dict,
        # num_decode_steps,
        num_decode_steps=None,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        flip_heading_accordingly=True,
        num_modes_for_eval=None,
        not_teacher_forcing_ids=None,
        **kwargs
    ):

        if self.config.USE_DIFFUSION:
            raise ValueError()

        raw_data = data_dict
        # To avoid those overwriting operation.
        data_dict = copy.deepcopy(data_dict)

        tokenizer = self.tokenizer

        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = data_dict["decoder/input_action"].shape

        assert self.config.GPT_STYLE
        start_action_step = 0
        assert T_input == 19
        # else:
        #     start_action_step = 2
        #     assert T_input == 17
        # autoregressive_start_step = 2

        if num_decode_steps is None:
            num_decode_steps = 19
            assert start_action_step + T_input == num_decode_steps
            assert num_decode_steps == 19
            assert data_dict["decoder/input_action_valid_mask"].shape == (B, T_input, N)
        else:
            print("WARNING: You are freely generating future trajectory! num_decode_steps (was 19) =", num_decode_steps)

        # ===== Get initial data =====

        agent_pos = data_dict["decoder/agent_position"][..., :2]  #.clone()
        agent_heading = data_dict["decoder/agent_heading"]  #.clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  #.clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  #.clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  #.clone()
        B, T_full, N, _ = agent_pos.shape
        # TODO: hardcoded
        assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====

        agent_pos = agent_pos[:, ::tokenizer.num_skipped_steps]  # (B, 19, N, 3)
        agent_heading = agent_heading[:, ::tokenizer.num_skipped_steps]  # (B, 19, N)
        agent_valid_mask = agent_valid_mask[:, ::tokenizer.num_skipped_steps]  # (B, 19, N)
        agent_velocity = agent_velocity[:, ::tokenizer.num_skipped_steps]  # (B, 19, N, 2)

        T_chunks = agent_pos.shape[1]
        assert T_chunks == 19

        # ===== Build up some variables =====
        # Should note that the modeled_agent_* is starting from t=0 (GPT) and t=10 (non-GPT). So using 0:1 to get the
        # first step for decoder is correct.
        assert not_teacher_forcing_ids is not None
        sdc_adv_ids = torch.zeros_like(data_dict["decoder/agent_id"], dtype=torch.bool)
        for aid in not_teacher_forcing_ids:
            sdc_adv_ids[:, aid] = True
        sdc_adv_ids = sdc_adv_ids.reshape(B, 1, N)
        assert B == 1, "To avoid the confusion, we only support B=1"

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
        gt_target_valid_mask = data_dict["decoder/target_action_valid_mask"].clone()
        current_input_action = gt_input_action[:, :1].clone()

        output_logit_list = []
        output_action_list = []
        output_action_valid_mask = []
        assert use_cache

        import numpy as np

        pos = []
        head = []
        vel = []

        # Select correct bins:
        agent_type = data_dict["decoder/agent_type"]
        bin_centers = tokenizer.get_bin_centers(agent_type)

        data_dict = self.encode_scene(data_dict)
        data_dict["decoder/randomized_modeled_agent_id"] = self.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=True
        )
        for decode_step in range(num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            # TODO: put back the following code
            # if decode_step == autoregressive_start_step:
            #     assert (
            #         current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step + 1]
            #     ).all()
            #     assert (current_valid_mask == data_dict["decoder/current_agent_valid_mask"][:, None]).all()

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

            assert not (current_input_action == START_ACTION).any()

            # Decode motion tokens
            data_dict = self.decode_motion(data_dict, use_cache=use_cache)

            if "decoder/modeled_agent_position_history" in data_dict:
                assert data_dict["decoder/modeled_agent_position_history"].shape[
                    1] == decode_step + 1 - start_action_step

            output_token = data_dict["decoder/output_logit"]
            if use_cache:
                assert output_token.shape[:3] == (B, 1, N)
            else:
                assert output_token.shape[:3] == (B, decode_step + 1, N)
                output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)
            selected_action = sample_action(
                logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
            )

            # ===== Teacher-forcing =====
            selected_action[~sdc_adv_ids] = gt_target_action[:, decode_step:decode_step + 1][~sdc_adv_ids]
            current_valid_mask[~sdc_adv_ids] = gt_target_valid_mask[:, decode_step:decode_step + 1][~sdc_adv_ids]

            # if self.config.MODEL.RELATIVE_PE_DECODER:
            res = tokenizer.detokenize_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=selected_action,
                agent_shape=data_dict["decoder/current_agent_shape"],
                bin_centers=bin_centers,
                # dt=tokenizer.dt,
                dt=-tokenizer.dt,
                flip_heading_accordingly=flip_heading_accordingly,
                flip_wrong_heading=True,  # TODO: This is a dirty workaround!
            )
            recon_next_pos, recon_next_heading, recon_next_vel, relative_delta_pos = res["pos"], res["heading"], res[
                "vel"], res["delta_pos"]

            current_pos = recon_next_pos.reshape(B, 1, N, 2)
            current_heading = recon_next_heading.reshape(B, 1, N)
            current_vel = recon_next_vel.reshape(B, 1, N, 2)
            current_delta = relative_delta_pos.reshape(B, 1, N, 2)
            current_model_step = torch.full_like(current_model_step, decode_step + 1 - start_action_step)
            current_input_action = selected_action

            # current_model_step.fill_(decode_step + 1 - start_action_step)
            # current_input_action = tf_action

            # Overwrite the data FOR NEXT STEP by the GT data:
            # if decode_step < autoregressive_start_step:
            # Always adding new agents

            # decode_step = 0, ..., 18
            forward_current_step = T_chunks - decode_step - 1
            # forward_current_step = 18, ..., 0
            forward_next_step = forward_current_step - 1
            # forward_next_step = 17, ..., 0

            if forward_next_step >= 0:
                # ===== Teacher-forcing =====
                overwrite_mask = ~sdc_adv_ids
                current_pos[overwrite_mask] = agent_pos[:,
                                                        forward_next_step:forward_next_step + 1, :, :2][overwrite_mask]
                current_heading[overwrite_mask] = agent_heading[:,
                                                                forward_next_step:forward_next_step + 1][overwrite_mask]
                current_vel[overwrite_mask] = agent_velocity[:, forward_next_step:forward_next_step + 1][overwrite_mask]

                if self.config.DELTA_POS_IS_VELOCITY:
                    current_delta[overwrite_mask] = tokenizer.get_relative_velocity(
                        vel=agent_velocity[:, forward_next_step:forward_next_step + 1][overwrite_mask],
                        heading=agent_heading[:, forward_next_step:forward_next_step + 1][overwrite_mask],
                    )

                else:
                    current_delta[overwrite_mask] = _reconstruct_delta_pos_from_abs_vel(
                        vel=current_vel[overwrite_mask],
                        heading=current_heading[overwrite_mask] + np.pi,
                        dt=tokenizer.dt
                    )
                current_input_action[overwrite_mask] = gt_target_action[:, decode_step:decode_step + 1][overwrite_mask]
                current_valid_mask[overwrite_mask] = gt_target_valid_mask[:,
                                                                          decode_step:decode_step + 1][overwrite_mask]
                output_token[overwrite_mask] = 0.0

            # The output action valid mask should before adding new agents.
            output_action_valid_mask.append(current_valid_mask.clone())

            newly_added = agent_valid_mask[:, forward_next_step:forward_next_step + 1] & (~current_valid_mask)
            if newly_added.any():
                current_pos[newly_added] = agent_pos[:, forward_next_step:forward_next_step + 1, ..., :2][newly_added]
                current_heading[newly_added] = agent_heading[:, forward_next_step:forward_next_step + 1][newly_added]
                current_vel[newly_added] = agent_velocity[:, forward_next_step:forward_next_step + 1][newly_added]
                current_valid_mask[newly_added] = agent_valid_mask[:,
                                                                   forward_next_step:forward_next_step + 1][newly_added]

                if self.config.DELTA_POS_IS_VELOCITY:

                    current_delta[newly_added] = get_relative_velocity(
                        vel=agent_velocity[:, forward_next_step:forward_next_step + 1][newly_added],
                        heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                    )

                else:
                    current_delta[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                        vel=current_vel[newly_added],

                        # heading=current_heading[newly_added],
                        heading=current_heading[newly_added] + np.pi,
                        dt=tokenizer.dt
                    )

                # Overwrite the input action by GT action
                current_input_action[newly_added] = gt_input_action[:, decode_step + 1:decode_step + 2][newly_added]
                output_token[newly_added] = 0.0

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

        # data_dict["decoder/output_score"] =

        # action_valid_mask = data_dict["decoder/target_action_valid_maks"].clone()
        # print(111)
        output_action_valid_mask = torch.cat(output_action_valid_mask, dim=1)
        data_dict["decoder/input_action_valid_mask"] = output_action_valid_mask

        data_dict["decoder/debug_ar_pos"] = torch.cat(pos, dim=1)
        data_dict["decoder/debug_ar_head"] = torch.cat(head, dim=1)
        data_dict["decoder/debug_ar_vel"] = torch.cat(vel, dim=1)

        # output_action_valid_mask = torch.cat(input_action_valid_mask_list, dim=1)

        valid_output_action = output_action_list[output_action_valid_mask]
        assert valid_output_action.max() < tokenizer.num_actions, valid_output_action.max()
        assert valid_output_action.min() >= 0, valid_output_action.min()

        assert valid_output_action.max() < END_ACTION, valid_output_action.max()
        assert valid_output_action.min() >= 0, valid_output_action.min()

        # ===== Debug! rewrite output action by GT =====
        # input_dict["decoder/output_action"] = input_dict["decoder/target_action"].clone()
        # fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
        # input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

        return data_dict

    def autoregressive_rollout_diffusion(
        self,
        data_dict,
        # num_decode_steps,
        num_decode_steps=None,
        use_cache=True,
        sampling_method="softmax",
        temperature=None,
        topp=None,
        num_modes_for_eval=None,
        **kwargs
    ):

        assert not ("backward_prediction" in kwargs and kwargs["backward_prediction"])
        assert self.config.USE_DIFFUSION

        # raw_data = data_dict
        # To avoid those overwriting operation.
        data_dict = copy.deepcopy(data_dict)

        tokenizer = self.tokenizer

        if temperature is None:
            temperature = self.config.SAMPLING.TEMPERATURE
        if topp is None:
            topp = self.config.SAMPLING.TOPP

        B, T_input, N = data_dict["decoder/input_action"].shape

        if self.config.GPT_STYLE:
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
        agent_pos = data_dict["decoder/agent_position"]  #.clone()
        agent_heading = data_dict["decoder/agent_heading"]  #.clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  #.clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  #.clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  #.clone()
        B, T_full, N, _ = agent_pos.shape
        # TODO: hardcoded
        assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::tokenizer.num_skipped_steps]
        agent_heading = agent_heading[:, ::tokenizer.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::tokenizer.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::tokenizer.num_skipped_steps]
        # gt_agent_delta = data_dict["decoder/modeled_agent_delta"].clone()
        T_chunks = agent_pos.shape[1]
        assert T_chunks == 19

        # ===== Build up some variables =====
        # Should note that the modeled_agent_* is starting from t=0 (GPT) and t=10 (non-GPT). So using 0:1 to get the
        # first step for decoder is correct.
        current_pos = data_dict["decoder/modeled_agent_position"][:, :1, ..., :2].clone()
        current_heading = data_dict["decoder/modeled_agent_heading"][:, :1].clone()
        current_vel = data_dict["decoder/modeled_agent_velocity"][:, :1].clone()
        current_valid_mask = data_dict["decoder/input_action_valid_mask"][:, :1].clone()
        current_delta = data_dict["decoder/modeled_agent_delta"][:, :1].clone()
        current_model_step = torch.arange(1).to(current_pos.device)  # it's 0

        current_input_agent_motion = data_dict["decoder/input_agent_motion"][:, :1].clone()

        gt_input_action = data_dict["decoder/input_action"].clone()
        gt_input_agent_motion = data_dict["decoder/input_agent_motion"].clone()
        gt_target_agent_motion = data_dict["decoder/target_agent_motion"].clone()

        current_input_action = gt_input_action[:, :1].clone()

        output_logit_list = []
        output_action_list = []
        output_motion_list = []
        input_action_valid_mask_list = []
        assert use_cache

        pos = [current_pos.clone()]
        head = [current_heading.clone()]
        vel = [current_vel.clone()]

        # Select correct bins:
        agent_type = data_dict["decoder/agent_type"]
        # bin_centers = tokenizer.get_bin_centers(agent_type)

        data_dict = self.encode_scene(data_dict)
        data_dict["decoder/randomized_modeled_agent_id"] = self.motion_decoder.randomize_modeled_agent_id(
            data_dict, clip_agent_id=True
        )
        for decode_step in range(0, num_decode_steps):
            logger.debug(f"======================= STEP {decode_step=} =======================")

            # if decode_step < start_action_step:
            #     # For non-gpt model, skip first 2 steps.
            #     pos.append(agent_pos[:, decode_step:decode_step + 1, ..., :2])
            #     head.append(agent_heading[:, decode_step:decode_step + 1])
            #     vel.append(agent_velocity[:, decode_step:decode_step + 1])
            #     continue

            if decode_step == autoregressive_start_step:
                assert (
                    current_valid_mask == agent_valid_mask[:, autoregressive_start_step:autoregressive_start_step + 1]
                ).all()
                assert (current_valid_mask == data_dict["decoder/current_agent_valid_mask"][:, None]).all()
            if decode_step == autoregressive_start_step + 1:
                current_input_action[current_input_action == SPECIAL_START] = SPECIAL_VALID

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
            data_dict["decoder/input_agent_motion"] = current_input_agent_motion

            input_action_valid_mask_list.append(current_valid_mask.clone())

            assert not (current_input_action == END_ACTION).any()

            assert not self.config.MCTS.USE_MCTS

            # Decode motion tokens
            # data_dict = self.decode_motion(data_dict, use_cache=use_cache)
            #
            # if "decoder/modeled_agent_position_history" in data_dict:
            #     assert data_dict["decoder/modeled_agent_position_history"].shape[
            #         1] == decode_step + 1 - start_action_step

            # output_token = data_dict["decoder/decoded_tokens"]
            # assert use_cache
            # assert output_token.shape[:3] == (B, 1, N)
            # else:
            #     assert output_token.shape[:3] == (B, decode_step + 1, N)
            #     output_token = output_token[:, -1:]  # -> output_token.shape == (B, 1, N, #actions)

            # selected_action = sample_action(
            #     logits=output_token, sampling_method=sampling_method, temperature=temperature, topp=topp
            # )

            data_dict = self.motion_decoder.sample_diffusion(data_dict, use_cache=use_cache)
            selected_action = data_dict["decoder/output_action"]
            if decode_step < autoregressive_start_step:
                # Overwrite the action by GT action
                selected_action = gt_target_agent_motion[:, decode_step:decode_step + 1]

            # FIXME: DEBUG
            # FIXME: DEBUG
            # FIXME: DEBUG
            # FIXME: DEBUG
            # FIXME: DEBUG
            # DEBUG
            # selected_action = gt_target_agent_motion[:, decode_step:decode_step + 1]

            res = tokenizer.detokenize_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=selected_action,
                agent_shape=data_dict["decoder/current_agent_shape"],
                # bin_centers=bin_centers,
                dt=tokenizer.dt,
                flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
                agent_type=agent_type,
            )
            recon_next_pos = res["pos"]
            recon_next_heading = res["heading"]
            recon_traj = res["reconstructed_pos"]
            recon_traj_heading = res["reconstructed_heading"]
            recon_next_vel = res["vel"]
            relative_delta_pos = res["delta_pos"]

            current_pos = recon_next_pos.reshape(B, 1, N, 2)
            current_heading = recon_next_heading.reshape(B, 1, N)
            current_vel = recon_next_vel.reshape(B, 1, N, 2)
            current_delta = relative_delta_pos.reshape(B, 1, N, 2)
            current_model_step = torch.full_like(current_model_step, decode_step + 1 - start_action_step)
            # current_input_action = selected_action

            agent_motion = data_dict["decoder/output_action"]
            assert current_input_agent_motion.shape == agent_motion.shape, (
                current_input_agent_motion.shape, agent_motion.shape
            )
            current_input_agent_motion = agent_motion

            # Overwrite the data FOR NEXT STEP by the GT data:
            if decode_step < autoregressive_start_step:
                # current_input_action[current_input_action == SPECIAL_START] = SPECIAL_VALID
                newly_added = agent_valid_mask[:, decode_step + 1:decode_step + 2] & (~current_valid_mask)
                if newly_added.any():
                    current_pos[newly_added] = agent_pos[:, decode_step + 1:decode_step + 2, ..., :2][newly_added]
                    current_heading[newly_added] = agent_heading[:, decode_step + 1:decode_step + 2][newly_added]
                    current_vel[newly_added] = agent_velocity[:, decode_step + 1:decode_step + 2][newly_added]
                    current_valid_mask[newly_added] = agent_valid_mask[:, decode_step + 1:decode_step + 2][newly_added]

                current_input_action = gt_input_action[:, decode_step + 1:decode_step + 2]
                current_input_agent_motion = gt_input_agent_motion[:, decode_step + 1:decode_step + 2].clone()

            pos.append(recon_traj.clone().permute(0, 1, 3, 2, 4).squeeze(1))
            head.append(recon_traj_heading.clone().permute(0, 1, 3, 2).squeeze(1))
            vel.append(current_vel.clone())
            # output_logit_list.append(output_token.clone())
            output_action_list.append(current_input_action.clone())
            output_motion_list.append(current_input_agent_motion.clone())

        output_action_list = torch.concatenate(output_action_list, dim=1)
        output_motion_list = torch.concatenate(output_motion_list, dim=1)
        assert output_action_list.shape == (B, num_decode_steps - start_action_step, N)

        # output_logit_list = torch.concatenate(output_logit_list, dim=1)
        # data_dict["decoder/output_logit"] = output_logit_list
        # data_dict["decoder/output_action"] = output_action_list

        # data_dict["decoder/output_score"] = calculate_trajectory_probabilities(
        #     output_logit_list, output_action_list, mask=current_valid_mask
        # )  # (B, N)

        input_action_valid_mask = torch.cat(input_action_valid_mask_list, dim=1)
        data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask

        data_dict["decoder/reconstructed_position"] = torch.cat(pos, dim=1)
        data_dict["decoder/reconstructed_heading"] = torch.cat(head, dim=1)
        data_dict["decoder/reconstructed_velocity"] = torch.cat(vel, dim=1)

        valid = input_action_valid_mask.reshape(B, -1, 1, N).expand(-1, -1, self.tokenizer.num_skipped_steps,
                                                                    -1).reshape(B, -1, N)
        valid = torch.cat([valid, input_action_valid_mask[:, -1:]], dim=1)
        data_dict["decoder/reconstructed_valid_mask"] = valid

        # valid_output_action = output_action_list[input_action_valid_mask]
        # assert valid_output_action.max() < tokenizer.num_actions
        # assert valid_output_action.min() >= 0

        return data_dict
