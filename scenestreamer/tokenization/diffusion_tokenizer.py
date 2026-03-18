import numpy as np
import torch

from scenestreamer.tokenization.motion_tokenizers import DeltaDeltaTokenizer, BaseTokenizer, get_relative_velocity, STEPS_PER_SECOND
from scenestreamer.utils import utils
import json

SPECIAL_INVALID = 0
SPECIAL_VALID = 1
SPECIAL_START = 2
SPECIAL_MASKED = 3

cyc_std = [
    0.19134897, 0.28934157, 0.52083063, 0.19307183, 0.4897204, 0.617495, 0.19566636, 0.6996683, 0.6919608, 0.21254513,
    0.93830705, 0.7470725, 0.23419037, 1.1697725, 0.80575556
]
cyc_mean = [
    0.0007314072, 0.38584468, -0.0005979259, 0.0015717002, 0.77193886, -0.0023218163, 0.002200475, 1.1587898,
    -0.0010374307, 0.0025438545, 1.5457957, -0.0017486577, 0.002413424, 1.9331585, -0.0021125625
]

# Just put it here in case need it. It's output 20 values including Z axis.
# output_std
#  [0.19134897, 0.28934157, 0.023779996, 0.52083063, 0.19307183, 0.4897204, 0.028927919, 0.617495, 0.19566636, 0.6996683, 0.034308836, 0.6919608, 0.21254513, 0.93830705, 0.03995999, 0.7470725, 0.23419037, 1.1697725, 0.046037905, 0.80575556]
# output_mean
#  [0.0007314072, 0.38584468, 6.2437284e-05, -0.0005979259, 0.0015717002, 0.77193886, 0.00020317949, -0.0023218163, 0.002200475, 1.1587898, 0.00037277717, -0.0010374307, 0.0025438545, 1.5457957, 0.00043893827, -0.0017486577, 0.002413424, 1.9331585, 0.0006425823, -0.0021125625]

# PED:
# output_std
ped_std = [
    0.04235751, 0.074799694, 0.68056464, 0.045664273, 0.13535422, 0.7549149, 0.049251433, 0.19858274, 0.81499416,
    0.05508655, 0.26276276, 0.86660916, 0.06212374, 0.32706332, 0.9120234
]
# output_mean
ped_mean = [
    1.4271492e-05, 0.09508697, -0.00029620097, 8.967689e-05, 0.18981859, -0.00076825253, 0.00014018304, 0.2842135,
    -0.000643687, 0.00017854733, 0.37860408, -0.0008555841, 0.00028553946, 0.4729763, -0.0006006232
]
# PED 20:
# output_std
#  [0.04235751, 0.074799694, 0.015911372, 0.68056464, 0.045664273, 0.13535422, 0.019296514, 0.7549149, 0.049251433, 0.19858274, 0.022285515, 0.81499416, 0.05508655, 0.26276276, 0.025021093, 0.86660916, 0.06212374, 0.32706332, 0.02777967, 0.9120234]
# output_mean
#  [1.4271492e-05, 0.09508697, -2.9287117e-05, -0.00029620097, 8.967689e-05, 0.18981859, 1.3563565e-05, -0.00076825253, 0.00014018304, 0.2842135, 2.268245e-05, -0.000643687, 0.00017854733, 0.37860408, 5.332513e-05, -0.0008555841, 0.00028553946, 0.4729763, 8.546651e-05, -0.0006006232]

# VEH:
# output_std
veh_std = [
    0.044777874, 0.5883173, 0.48556346, 0.06600924, 1.1707116, 0.5484921, 0.081244536, 1.7253877, 0.59927756, 0.1037226,
    2.3370876, 0.6422892, 0.13490802, 2.777383, 0.6803117
]
# output_mean
veh_mean = [
    -0.0006106305, 0.54284567, -0.00054906023, -0.0014649398, 1.0862343, -0.0009819986, -0.0025806348, 1.6136851,
    -0.0012528845, -0.003927998, 2.174847, -0.0012167478, -0.0055386806, 2.6897037, -0.0012875189
]

# VEH 20:
# output_std
#  [0.044777874, 0.5883173, 0.013346198, 0.48556346, 0.06600924, 1.1707116, 0.020044344, 0.5484921, 0.081244536, 1.7253877, 0.02681499, 0.59927756, 0.1037226, 2.3370876, 0.03373221, 0.6422892, 0.13490802, 2.777383, 0.040490452, 0.6803117]
# output_mean
#  [-0.0006106305, 0.54284567, 0.00023764589, -0.00054906023, -0.0014649398, 1.0862343, 0.00047408746, -0.0009819986, -0.0025806348, 1.6136851, 0.0007028891, -0.0012528845, -0.003927998, 2.174847, 0.00094223575, -0.0012167478, -0.0055386806, 2.6897037, 0.0011787575, -0.0012875189]


class DiffusionTokenizer(DeltaDeltaTokenizer):

    # ped_std = torch.from_numpy(np.asarray(ped_std)).float().reshape(1, 1, 1, 15)
    # ped_mean = torch.from_numpy(np.asarray(ped_mean)).float().reshape(1, 1, 1, 15)
    # cyc_std = torch.from_numpy(np.asarray(cyc_std)).float().reshape(1, 1, 1, 15)
    # cyc_mean = torch.from_numpy(np.asarray(cyc_mean)).float().reshape(1, 1, 1, 15)
    # veh_std = torch.from_numpy(np.asarray(veh_std)).float().reshape(1, 1, 1, 15)
    # veh_mean = torch.from_numpy(np.asarray(veh_mean)).float().reshape(1, 1, 1, 15)

    def __init__(self, config):
        BaseTokenizer.__init__(self, config)

        # self.dt = (1 / STEPS_PER_SECOND) * 1

        self.should_standardize = config.TOKENIZATION.SHOULD_STANDARDIZE

        with open(utils.REPO_ROOT / "scenestreamer" / "tokenization" / "motion_stats_FORMAL.json", "r") as f:
            motion_stats = json.load(f)
        self.veh_mean = torch.tensor(motion_stats["1"]["mean"]).reshape(1, 1, 1, 15)
        self.veh_std = torch.tensor(motion_stats["1"]["std"]).reshape(1, 1, 1, 15)
        self.ped_mean = torch.tensor(motion_stats["2"]["mean"]).reshape(1, 1, 1, 15)
        self.ped_std = torch.tensor(motion_stats["2"]["std"]).reshape(1, 1, 1, 15)
        self.cyc_mean = torch.tensor(motion_stats["3"]["mean"]).reshape(1, 1, 1, 15)
        self.cyc_std = torch.tensor(motion_stats["3"]["std"]).reshape(1, 1, 1, 15)

        self.use_delta = False
        self.use_delta_delta = True

        assert not (self.use_delta_delta is True and self.use_delta is True)

    def _get_stat(self, motion, agent_type):
        B, T, N, D = motion.shape
        if motion.device != self.ped_std.device:
            self.ped_std = self.ped_std.to(motion.device)
            self.ped_mean = self.ped_mean.to(motion.device)
            self.cyc_std = self.cyc_std.to(motion.device)
            self.cyc_mean = self.cyc_mean.to(motion.device)
            self.veh_std = self.veh_std.to(motion.device)
            self.veh_mean = self.veh_mean.to(motion.device)

        agent_type = agent_type.unsqueeze(-1)

        mean = torch.zeros_like(motion)
        mean = torch.where(agent_type == 1, self.veh_mean.expand(B, T, N, -1), mean)
        mean = torch.where(agent_type == 2, self.ped_mean.expand(B, T, N, -1), mean)
        mean = torch.where(agent_type == 3, self.cyc_mean.expand(B, T, N, -1), mean)

        std = torch.ones_like(motion)
        std = torch.where(agent_type == 1, self.veh_std.expand(B, T, N, -1), std)
        std = torch.where(agent_type == 2, self.ped_std.expand(B, T, N, -1), std)
        std = torch.where(agent_type == 3, self.cyc_std.expand(B, T, N, -1), std)

        # TODO: Just nullify the STD.
        std = std.fill_(10)

        return mean, std

    def standardize(self, motion, agent_type, valid_mask):
        if not self.should_standardize:
            motion[~valid_mask] = 0
            return motion

        if self.use_delta or self.use_delta_delta:
            # TODO: Reconsider this.
            motion[~valid_mask] = 0
            return motion

        assert motion.ndim == 4
        B, T, N, D = motion.shape
        if agent_type.ndim == 2:
            agent_type = agent_type.unsqueeze(1).expand(-1, T, -1)
        assert agent_type.ndim == 3
        assert D == 15
        assert valid_mask.shape == (B, T, N)

        # Do a hack here... Do not use mean-std normalization for heading to avoid wierd thing.
        heading = motion.reshape(B, T, N, 5, 3)[:, :, :, :, -1]
        normalized_heading = utils.wrap_to_pi(heading) / np.pi

        mean, std = self._get_stat(motion, agent_type)
        motion = (motion - mean) / std
        motion[~valid_mask] = 0

        motion = motion.reshape(B, T, N, 5, 3)
        motion[:, :, :, :, -1] = normalized_heading
        motion = motion.reshape(B, T, N, D)

        motion[~valid_mask] = 0
        return motion

    def unstandardize(self, motion, agent_type, valid_mask):
        if not self.should_standardize:
            motion[~valid_mask] = 0
            return motion

        if self.use_delta or self.use_delta_delta:
            # TODO: Reconsider this.
            motion[~valid_mask] = 0
            return motion

        assert motion.ndim == 4
        B, T, N, D = motion.shape
        if agent_type.ndim == 2:
            agent_type = agent_type.unsqueeze(1).expand(-1, T, -1)
        assert agent_type.ndim == 3
        assert D == 15
        assert valid_mask.shape == (B, T, N)
        mean, std = self._get_stat(motion, agent_type)

        # Do a hack here... Do not use mean-std normalization for heading to avoid wierd thing.
        heading = motion.reshape(B, T, N, 5, 3)[:, :, :, :, -1]
        unnormalized_heading = utils.wrap_to_pi(heading * np.pi)

        motion = motion * std + mean

        motion = motion.reshape(B, T, N, 5, 3)
        motion[:, :, :, :, -1] = unnormalized_heading
        motion = motion.reshape(B, T, N, D)

        motion[~valid_mask] = 0
        return motion

    def tokenize(self, data_dict, **kwargs):

        agent_pos = data_dict["decoder/agent_position"].clone()
        agent_heading = data_dict["decoder/agent_heading"].clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"].clone()
        agent_velocity = data_dict["decoder/agent_velocity"].clone()

        # Do the slicing
        B, T_full, N, _ = agent_pos.shape
        # TODO: hardcoded
        assert T_full == 91
        assert agent_pos.ndim == 4
        # Note: do [::5] slicing will keep 0, 5, 10 (the current step!), 15, ...

        # ===== Hole filling =====
        data_dict = self.hole_filling(data_dict)

        def unfold(t):
            """This function transforms the tensor from (B, T, N, 2) to (B, T', N, 6, ..) where T' = T // 5"""
            assert t.shape[1] == T_full
            t = t.unfold(dimension=1, size=6, step=5)
            if t.ndim == 5:
                t = t.permute(0, 1, 4, 2, 3)
            elif t.ndim == 4:
                t = t.permute(0, 1, 3, 2)
            else:
                raise ValueError
            return t

        agent_pos = unfold(agent_pos)
        agent_heading = unfold(agent_heading)
        agent_valid_mask = unfold(agent_valid_mask)
        agent_velocity = unfold(agent_velocity)

        T_action = T_full // self.num_skipped_steps
        assert T_action == agent_pos.shape[1] == 18

        # input_valid_mask = agent_valid_mask[:, :, 0, :]  # If the first step is valid, then is a valid input.

        start_valid_mask = agent_valid_mask[:, :1, 0].clone()

        target_valid_mask = agent_valid_mask.all(dim=2)  # If all steps are valid, then is a valid target.
        input_valid_mask = torch.cat([start_valid_mask, target_valid_mask], dim=1)
        target_valid_mask = torch.cat([target_valid_mask, target_valid_mask.new_zeros(B, 1, N)], dim=1)

        pos = torch.cat([agent_pos[:, :, 0], agent_pos[:, -1:, -1]], dim=1)
        heading = utils.wrap_to_pi(torch.cat([agent_heading[:, :, 0], agent_heading[:, -1:, -1]], dim=1))
        vel = torch.cat([agent_velocity[:, :, 0], agent_velocity[:, -1:, -1]], dim=1)

        relative_pos = agent_pos[:, :, :] - agent_pos[:, :, :1]
        relative_heading = utils.wrap_to_pi(agent_heading[:, :, :] - agent_heading[:, :, :1])

        def transform_to_relative_pos(x, h):
            # first rotate the absolute coordinate to the ego vehicle's coordinate
            assert x.ndim == 5
            assert h.ndim == 4

            # If we consider X to be the heading direction, then the following code is OK.
            # relative_pos = rotate(x=x[..., 0], y=x[..., 1], angle=-h, assert_shape=False)

            # However, because we need to do standardization, we need to follow strictly to the definition of
            # coordinate system to align with the stats from the dataset.
            local_y_wrt_global_x = h
            local_x_wrt_global_x = local_y_wrt_global_x - np.pi / 2
            relative_pos = utils.rotate(x=x[..., 0], y=x[..., 1], angle=-local_x_wrt_global_x, assert_shape=False)

            return relative_pos

        each_step_heading = agent_heading[:, :, 0]
        rotated_pos = transform_to_relative_pos(relative_pos, each_step_heading.unsqueeze(2).expand(-1, -1, 6, -1))
        rotated_pos = rotated_pos.permute((0, 1, 3, 2, 4))

        relative_heading = relative_heading.permute((0, 1, 3, 2)).unsqueeze(-1)

        if self.use_delta:
            # Compute the delta in the -2 dim.
            relative_heading = relative_heading[..., 1:, :] - relative_heading[..., :-1, :]
            rotated_pos = rotated_pos[..., 1:, :] - rotated_pos[..., :-1, :]

        elif self.use_delta_delta:
            # print(111)
            #
            # # Should not use relative_heading and rotated_pos here.
            # pred_vel_change_list = []
            # pred_heading_change_list = []
            #
            old_v = agent_velocity.permute((0, 1, 3, 2, 4))
            old_h = agent_heading.permute((0, 1, 3, 2))
            old_p = agent_pos.permute((0, 1, 3, 2, 4))

            dt = (1 / STEPS_PER_SECOND)

            # Compute velocity from position differences
            # velocity = (old_p[..., 1:, :2] - old_p[..., :-1, :2]) / dt  # Shape: (B, T, N, 5, 2)
            # velocity = torch.cat([velocity, old_v[..., -1:, :2],], dim=-2)  # Pad for shape consistency
            velocity = old_v
            old_v[~target_valid_mask[:, :-1]] = 0

            # Compute speed magnitude
            speed = torch.norm(velocity, dim=-1)  # Shape: (B, T, N, 6)

            # Compute acceleration as the change in speed
            acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
            acceleration[~target_valid_mask[:, :-1]] = 0

            # Compute yaw rate as the change in heading
            yaw_rate = utils.wrap_to_pi(old_h[..., 1:] - old_h[..., :-1]) / dt
            yaw_rate[~target_valid_mask[:, :-1]] = 0

            rotated_pos = acceleration
            relative_heading = yaw_rate  #.unsqueeze(-1)

            # for i in range(5):
            #     dpos0 = old_p[..., i+1, :2] - old_p[..., i, :2]
            #     v1 = dpos0 / self.dt
            #     v0 = old_v[..., i, :]
            #     dv0 = v1 - v0
            #     h0 = old_h[..., i]
            #     pred_vel_change = utils.rotate(dv0[..., 0], dv0[..., 1], -h0)
            #     pred_heading_change = old_h[..., i+1] - h0
            #     pred_vel_change_list.append(pred_vel_change)
            #     pred_heading_change_list.append(pred_heading_change)
            # pred_vel_change = torch.stack(pred_vel_change_list, dim=3)
            # pred_heading_change = torch.stack(pred_heading_change_list, dim=3)

            # Compute the delta in the -2 dim.
            # relative_heading = relative_heading[..., 1:, :] - relative_heading[..., :-1, :]
            # rotated_pos = rotated_pos[..., 1:, :] - rotated_pos[..., :-1, :]

        else:
            relative_heading = relative_heading[..., 1:, :]
            rotated_pos = rotated_pos[..., 1:, :]

        target_motion = torch.stack([rotated_pos, relative_heading], dim=-1)
        target_motion = target_motion.reshape(B, T_action, N, -1)

        target_agent_motion = torch.cat(
            [target_motion, target_motion.new_zeros(B, 1, N, target_motion.shape[-1])], dim=1
        )
        target_agent_motion = self.standardize(
            target_agent_motion, agent_type=data_dict["decoder/agent_type"], valid_mask=target_valid_mask
        )

        # Special token (total 5 options):
        # 0: invalid token
        # 1: valid token
        # 2: just started.
        # 3: masked.
        # 4: not used.
        input_special_token = torch.full(
            (B, T_action + 1, N), SPECIAL_INVALID, dtype=torch.int64, device=start_valid_mask.device
        )

        # Fill in MASKED for step between START and last VALID:
        cumsum = input_valid_mask.cumsum(dim=1)
        max_cumsum = cumsum.max(dim=1, keepdim=True).values
        input_special_token[cumsum < max_cumsum] = SPECIAL_MASKED

        already_started = input_valid_mask[:, :1].clone()
        input_special_token[input_valid_mask] = SPECIAL_VALID
        input_special_token[:, :1][input_valid_mask[:, :1]] = SPECIAL_START
        for step in range(1, T_action):
            newly_added = (~already_started) & input_valid_mask[:, step:step + 1]
            input_special_token[:, step:step + 1][newly_added] = SPECIAL_START
            already_started = torch.logical_or(already_started, newly_added)

        # Allow the model to know the existence of masked agents.
        # is_masked = input_special_token == SPECIAL_MASKED
        # pos[is_masked] = 0
        # heading[is_masked] = 0
        # vel[is_masked] = 0

        input_agent_motion = torch.cat(
            [target_motion.new_zeros(B, 1, N, target_motion.shape[-1]), target_motion], dim=1
        )
        # input_agent_motion[is_masked] = 0
        input_valid_mask[input_special_token == SPECIAL_MASKED] = 1

        data_dict["decoder/input_action"] = input_special_token
        data_dict["decoder/modeled_agent_position"] = pos
        data_dict["decoder/modeled_agent_heading"] = heading
        data_dict["decoder/modeled_agent_velocity"] = vel
        data_dict["decoder/modeled_agent_delta"] = get_relative_velocity(vel=vel, heading=heading)

        data_dict["decoder/target_agent_motion"] = target_agent_motion
        data_dict["decoder/target_action_valid_mask"] = target_valid_mask
        data_dict["decoder/input_agent_motion"] = input_agent_motion
        data_dict["decoder/input_action_valid_mask"] = input_valid_mask

        # All input actions should be >0
        # This assertion won't hold because we introduce MASKED token.
        # assert (input_special_token[~input_valid_mask] == 0).all()

        return data_dict, {}

    def _detokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, action, agent_type, **kwargs
    ):
        B, _, N, _ = action.shape

        # Do the unstandardization here!
        action = self.unstandardize(action, agent_type=agent_type, valid_mask=current_valid_mask)

        action = action.reshape(B, 1, N, 5, -1)

        # DEBUG
        # action = action.fill_(0)

        if self.use_delta_delta:

            acceleration = action[..., 0]
            yaw_rate = action[..., 1]
            dt = (1 / STEPS_PER_SECOND)

            # Compute speed from acceleration (cumulative sum)
            initial_speed = torch.norm(current_vel, dim=-1, keepdim=True)  # Shape: (B, T, N, 1)
            speed = torch.cat([initial_speed, acceleration * dt], dim=-1)  # Add initial speed
            speed = torch.cumsum(speed, dim=-1)  # Integrate over time
            speed = speed[..., 1:]

            # Compute heading from yaw rate (cumulative sum)
            reconstructed_h = torch.cumsum(torch.cat([current_heading.unsqueeze(-1), yaw_rate * dt], dim=-1), dim=-1)
            reconstructed_h = utils.wrap_to_pi(reconstructed_h)
            reconstructed_h = reconstructed_h[..., 1:]

            # Compute velocity in the global frame
            # local_y_wrt_global_x = reconstructed_h
            # local_x_wrt_global_x = local_y_wrt_global_x - np.pi / 2

            velocity_x = speed * torch.cos(reconstructed_h)
            velocity_y = speed * torch.sin(reconstructed_h)
            velocity = torch.stack([velocity_x, velocity_y], dim=-1)  # Shape: (B, T, N, 6, 2)

            # Compute position from velocity (cumulative sum)
            delta_p = velocity * dt  # Displacement
            reconstructed_p = torch.cumsum(torch.cat([current_pos.unsqueeze(-2), delta_p], dim=-2), dim=-2)
            reconstructed_p = reconstructed_p[..., 1:, :]
            reconstructed_velocity = velocity

            delta_pos = get_relative_velocity(vel=velocity[..., -1, :], heading=reconstructed_h[..., -1])

            reconstructed_p[~current_valid_mask] = 0
            reconstructed_h[~current_valid_mask] = 0
            delta_pos[~current_valid_mask] = 0

            AID = 1
            b = 0
            print(
                "CUR POS: {}, CUR HEA: {},  POS: {}, HEAD: {}, Speed: {} ".format(
                    current_pos[b, 0, AID].cpu().numpy(),
                    current_heading[b, 0, AID],
                    reconstructed_p[b, 0, AID].cpu().numpy(),
                    reconstructed_h[b, 0, AID],
                    current_vel[b, 0, AID].norm(dim=-1),
                    # reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy(),
                    # reconstructed_vel.norm(dim=-1)[0, 0, AID],
                    # unrotated_delta_vel[0, 0, AID].cpu().numpy(),
                    # current_vel[0, 0, AID].norm(dim=-1)
                )
            )

            return dict(
                pos=reconstructed_p[:, :, :, -1],
                heading=reconstructed_h[:, :, :, -1],
                reconstructed_pos=reconstructed_p,
                reconstructed_heading=reconstructed_h,
                delta_pos=delta_pos,
                vel=reconstructed_velocity[:, :, :, -1],
            )

        if self.use_delta:

            agent_pos_change = action[..., 0:2]  # .squeeze(1)
            agent_heading_change = action[..., 2:3]  # .squeeze(1)

            # Use cumsum on -2 dim to get the per-step delta.
            agent_pos_change = agent_pos_change.cumsum(dim=-2)
            agent_heading_change = agent_heading_change.cumsum(dim=-2)

        else:

            agent_pos_change = action[..., 0:2]  # .squeeze(1)
            agent_heading_change = action[..., 2:3]  # .squeeze(1)

        # Since we use strict coordinate system, we need to rotate the agent_pos_change back to the global coordinate.
        local_y_wrt_global_x = current_heading
        local_x_wrt_global_x = local_y_wrt_global_x - np.pi / 2

        rotated_agent_pos_change = utils.rotate(
            x=agent_pos_change[..., 0],
            y=agent_pos_change[..., 1],
            angle=local_x_wrt_global_x.reshape(B, 1, N, 1).expand(-1, -1, -1, 5)
        )
        reconstructed_pos = current_pos.reshape(B, 1, N, 1, 2) + rotated_agent_pos_change
        reconstructed_heading = utils.wrap_to_pi(current_heading.reshape(B, 1, N, 1) + agent_heading_change.squeeze(-1))

        delta_pos = get_relative_velocity(
            vel=rotated_agent_pos_change[..., -1, :], heading=agent_heading_change[..., -1, 0]
        )

        # AID = 3
        # print(
        #     "CUR POS: {}, CUR HEA: {},  POS: {}, HEAD: {}, ".format(
        #         current_pos[-1, 0, AID].cpu().numpy(),
        #         current_heading[-1, 0, AID],
        #         reconstructed_pos[-1, 0, AID].cpu().numpy(),
        #         reconstructed_heading[-1, 0, AID],
        #         # reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy(),
        #         # reconstructed_vel.norm(dim=-1)[0, 0, AID],
        #         # unrotated_delta_vel[0, 0, AID].cpu().numpy(),
        #         # current_vel[0, 0, AID].norm(dim=-1)
        #     )
        # )

        # TODO: Need to fix reconstructed_velocity

        return dict(
            pos=reconstructed_pos[:, :, :, -1],
            heading=reconstructed_heading[:, :, :, -1],
            reconstructed_pos=reconstructed_pos,
            reconstructed_heading=reconstructed_heading,
            vel=reconstructed_velocity[:, :, :, -1],
            delta_pos=delta_pos,
        )

    def detokenize(self, data_dict, **kwargs):
        assert "decoder/reconstructed_position" in data_dict
        assert "decoder/reconstructed_heading" in data_dict
        assert "decoder/reconstructed_valid_mask" in data_dict
        pos = data_dict["decoder/reconstructed_position"]
        B, T, N, _ = pos.shape
        data_dict["decoder/output_score"] = pos.new_zeros(size=(B, N))

        # pred = pos[:, :91]
        # gt =  data_dict["decoder/agent_position"][..., :2]
        # m = data_dict["decoder/reconstructed_valid_mask"][:, :91]
        # m2 = data_dict["decoder/target_action_valid_mask"]
        # # Expand m2:
        # m2 = m2.unsqueeze(2).expand(-1, -1, 5, -1).reshape(B, -1, N)[:, :91]
        # ade = torch.norm(pred - gt, dim=-1) * m
        # # m3 = m2.all(dim=1, keepdim=True).expand(-1, 91, -1)
        # ade[~m2] = 1000
        # adenp = ade.cpu().numpy()

        return data_dict
