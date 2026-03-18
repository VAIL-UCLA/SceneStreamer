import os
import pickle
import shutil
import time

import lightning.pytorch as pl
import torch
from scenestreamer.models.motion import MotionLM
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingWarmRestarts

from scenestreamer.dataset.preprocessor import sample_from_distributions_and_merge
from scenestreamer.utils.utils import rotate

# TODO: Add waymo eval
# from scenestreamer.eval.waymo_eval import waymo_evaluation


def get_derivative(array2d, dt=0.1):
    diff = (array2d[1:] - array2d[:-1]) / dt
    return np.concatenate([[diff[0]], diff], axis=0)


def find_last_valid(array, mask):
    assert mask.ndim + 1 == array.ndim
    assert mask.shape == array.shape[:-1]
    assert array.ndim == 4
    B, T, N, D = array.shape
    indices = mask * torch.arange(T, device=mask.device).reshape(1, T, 1).expand(*mask.shape)
    indices = indices.argmax(1, keepdims=True).unsqueeze(-1).expand(B, 1, N, D)
    ret = torch.gather(array, index=indices, dim=1)  # [B, 1, N, D]
    ret[~mask.any(1, keepdims=True)] = 0
    return ret


@torch.no_grad()
def sample_from_distributions_and_merge(
    step, copy_data_dict, model_output, batch_size, compress_step, sampling_method, max_known_step
):
    """
    This function merges the layers's output (the sampled position/velocity/...) into the input dict,
    for preparing next forward pass of the layers in inference.
    """

    # model_output["sampled_traffic_light_state"]
    effective_bs, _, L, _ = copy_data_dict["encoder/traffic_light_feature"].shape
    _, T, N, _ = copy_data_dict["encoder/agent_feature"].shape
    B = batch_size

    num_modes = effective_bs // batch_size

    pred_start = max(step - compress_step, 0)
    pred_end = min(step, T - compress_step)
    pred_length = min(compress_step, T - step)

    future_start = step
    future_end = min(step + compress_step, T)

    # [B, N, num_modes]
    score = model_output["score_logit"][:, -1].clone()  # This is already in logit! Not probability!

    # Only take the last token's first compress_step predictions -> [B, N, num_modes, pred_length (T), 3]
    sampled_position = model_output["sampled_position"][:, -1, :, :, :pred_length].clone()
    # [B, N, num_modes, pred_length, 2] -> [B, pred_length, N, num_modes, 3]
    sampled_position = sampled_position.permute(0, 3, 1, 2, 4)

    sampled_heading = model_output["sampled_heading"][:, -1, :, :, :pred_length].clone()
    sampled_heading = sampled_heading.permute(0, 3, 1, 2, 4)

    sampled_velocity = model_output["sampled_velocity"][:, -1, :, :, :pred_length].clone()
    sampled_velocity = sampled_velocity.permute(0, 3, 1, 2, 4)

    assert copy_data_dict["encoder/agent_feature"].shape[1] >= future_end

    # Build new agent feature
    new_agent_feature = copy_data_dict["encoder/agent_feature"][:, future_start:future_end].clone()

    # print("We are filling [{}, {}).".format(future_start, future_end))

    # pred_agent_pos = sampled_position[:, pred_start:pred_end]
    # pred_score = score[:, pred_start // compress_step:pred_start // compress_step + 1]

    if sampling_method == "native":
        # non-sampled based:
        # [B*num_modes, compress_T, N, num_modes, 2] -> [B, num_modes, compress_T, N, num_modes, 2]
        pred_agent_pos = sampled_position.reshape(B, num_modes, *sampled_position.shape[1:])

        # [B, num_modes, compress_T, N, num_modes, 2] -> [B, num_modes * num_modes, compress_T, N, 2]
        pred_agent_pos = pred_agent_pos.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)

        # [0+0, 6+1, 12+2, 18+3, 24+4, 30+5]
        ind = torch.arange(num_modes) * num_modes + torch.arange(num_modes)

        # [B, num_modes * num_modes, compress_T, N, 2] -> [B*num_modes, compress_T, N, 2]
        pred_agent_pos = pred_agent_pos[:, ind].flatten(0, 1)

        raise ValueError("Not finished yet")

    else:
        # pred_agent_pos is in shape [B*num_modes, compress_step (T), N, num_modes, 2]

        comp_mask = model_output["compress_agent_valid_mask"][:, -1, :]
        score = score[comp_mask]
        dist = torch.distributions.Categorical(logits=score)
        if sampling_method == "argmax":
            pred_score_ind = score.argmax(-1)
        elif sampling_method == "softmax":
            pred_score_ind = dist.sample()
        else:
            raise ValueError()

        log_probability = dist.log_prob(pred_score_ind)

        pred_score_ind = utils.unwrap(pred_score_ind.unsqueeze(-1), comp_mask)

        # This is extremely important!! Need to fill "-inf" to log_probability!
        log_probability = utils.unwrap(
            log_probability.unsqueeze(-1), comp_mask, fill=float("-inf")
        ).squeeze(-1)  # [B, N]

        # assert pred_score_ind.shape[1] == 1, pred_score_ind.shape
        pred_score_ind = pred_score_ind.reshape(effective_bs, 1, N, 1, 1)

        pred_score_ind_pos = pred_score_ind.expand(effective_bs, sampled_position.shape[1], N, 1, 3)
        pred_agent_pos = torch.gather(sampled_position, index=pred_score_ind_pos, dim=3)
        assert pred_agent_pos.shape == (effective_bs, pred_agent_pos.shape[1], N, 1, 3)
        pred_agent_pos = pred_agent_pos.squeeze(3)

    # pred_agent_pos_feat = pred_agent_pos.clone() / constants.POSITION_XY_RANGE
    pred_agent_pos_feat = pred_agent_pos.clone()
    assert new_agent_feature[..., :3].shape == pred_agent_pos_feat.shape
    new_agent_feature[..., :3] = pred_agent_pos_feat
    # Let the layers predict Z axis! Below is use old way to get Z.
    # new_agent_feature[..., 2:3] = find_last_valid(
    #     copy_data_dict["encoder/agent_feature"][:, :max_known_step, :, 2:3],
    #     copy_data_dict["encoder/agent_valid_mask"][:, :max_known_step],
    # )

    # Repeat above process for heading
    pred_score_ind_heading = pred_score_ind.expand(effective_bs, sampled_heading.shape[1], N, 1, 1)
    pred_agent_heading = torch.gather(sampled_heading, index=pred_score_ind_heading, dim=3)
    assert pred_agent_heading.ndim == 5
    pred_agent_heading = pred_agent_heading.squeeze(-1).squeeze(-1)

    # non-sampled based:
    # pred_agent_heading = pred_agent_heading.reshape(B, num_modes, *pred_agent_heading.shape[1:])
    # pred_agent_heading = pred_agent_heading.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
    # pred_agent_heading = pred_agent_heading[:, ind].flatten(0, 1).squeeze(-1)

    pred_agent_heading_feat = pred_agent_heading.clone()
    pred_agent_heading_feat = utils.wrap_to_pi(pred_agent_heading_feat)
    pred_agent_heading_feat /= constants.HEADING_RANGE

    # print("HEADING MIN: ", pred_agent_heading_feat.min())

    assert new_agent_feature[..., 3].shape == pred_agent_heading_feat.shape
    new_agent_feature[..., 3] = pred_agent_heading_feat
    new_agent_feature[..., 9] = torch.sin(pred_agent_heading)
    new_agent_feature[..., 10] = torch.cos(pred_agent_heading)

    # Repeat above process for velocity
    # pred_agent_vel = sampled_velocity[:, pred_start:pred_end]
    pred_score_ind_vel = pred_score_ind.expand(effective_bs, sampled_velocity.shape[1], N, 1, 2)
    pred_agent_vel = torch.gather(sampled_velocity, index=pred_score_ind_vel, dim=3)
    assert pred_agent_vel.ndim == 5
    pred_agent_vel = pred_agent_vel.squeeze(3)

    pred_agent_vel_feat = pred_agent_vel.clone() / constants.VELOCITY_XY_RANGE
    assert new_agent_feature[..., 4:6].shape == pred_agent_vel_feat.shape
    new_agent_feature[..., 4:6] = pred_agent_vel_feat

    # length width height
    new_agent_feature[..., [6, 7, 8]] = find_last_valid(
        copy_data_dict["encoder/agent_feature"][:, :max_known_step, :, [6, 7, 8]],
        copy_data_dict["encoder/agent_valid_mask"][:, :max_known_step],
    )

    # Input data already scaled.

    # speed
    speed = pred_agent_vel.norm(dim=-1) / constants.VELOCITY_XY_RANGE
    assert new_agent_feature[..., 11].shape == speed.shape
    new_agent_feature[..., 11] = speed

    # agent type
    new_agent_feature[..., 12:15] = find_last_valid(
        copy_data_dict["encoder/agent_feature"][:, :max_known_step, :, 12:15],
        copy_data_dict["encoder/agent_valid_mask"][:, :max_known_step],
    )

    # valid
    new_agent_feature[..., 15] = 1

    assert pred_agent_pos.shape == copy_data_dict["encoder/agent_position"][:, future_start:future_end].shape
    assert new_agent_feature.shape == copy_data_dict["encoder/agent_feature"][:, future_start:future_end].shape
    copy_data_dict["encoder/agent_position"][:, future_start:future_end] = pred_agent_pos.clone()
    copy_data_dict["encoder/agent_feature"][:, future_start:future_end] = new_agent_feature.clone()
    copy_data_dict["encoder/agent_valid_mask"][:, future_start:future_end] = 1

    # Build new traffic light feature
    # new_traffic_light_feature = data_dict["encoder/traffic_light_feature"].new_zeros((effective_bs, 1, L, TRAFFIC_LIGHT_STATE_DIM))
    new_traffic_light_feature = copy_data_dict["encoder/traffic_light_feature"][:, future_start:future_end].clone()
    new_T = new_traffic_light_feature.shape[1]
    assert new_T > 0
    if L > 0:
        # Fill "stop_point"
        new_traffic_light_feature[..., :3] = find_last_valid(
            copy_data_dict["encoder/traffic_light_feature"][:, :max_known_step, :, :3],
            copy_data_dict["encoder/traffic_light_valid_mask"][:, :max_known_step],
        )
        # [B, T, L]
        pred_light_state = model_output["sampled_traffic_light_state"][:, -1, :, :pred_length].permute(0, 2, 1)
        st = torch.nn.functional.one_hot(pred_light_state, num_classes=9)
        new_traffic_light_feature[:, :st.shape[1], ..., 3:] = st

    assert new_traffic_light_feature.shape == copy_data_dict["encoder/traffic_light_feature"][:, future_start:future_end
                                                                                              ].shape
    copy_data_dict["encoder/traffic_light_feature"][:, future_start:future_end] = new_traffic_light_feature.clone()
    copy_data_dict["encoder/traffic_light_valid_mask"][:, future_start:future_end] = 1

    return copy_data_dict, pred_agent_pos, future_start, future_end, log_probability


# TODO: This might be helpful
#  Handle unsupervised learning by using an IterableDataset where the dataset itself is constantly updated during training
#  https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/reinforce-learning-DQN.html?highlight=target


# TODO: Could move this to a util file?
def get_displacement(array, mask):
    assert mask.ndim + 1 == array.ndim
    assert mask.shape == array.shape[:-1]
    assert array.ndim == 4
    B, T, N, D = array.shape
    assert D == 2 or D == 3
    indices = mask * torch.arange(T, device=mask.device).reshape(1, T, 1).expand(*mask.shape)
    last_indices = indices.argmax(dim=1, keepdims=True).unsqueeze(-1).expand(B, 1, N, D)
    last_pos = torch.gather(array, index=last_indices, dim=1)

    first_indices = indices.argmin(dim=1, keepdims=True).unsqueeze(-1).expand(B, 1, N, D)
    first_pos = torch.gather(array, index=first_indices, dim=1)

    return (last_pos - first_pos).norm(dim=-1), last_pos


class MotionLMPL(pl.LightningModule):
    def __init__(self, cfg):
        if "SEED" in cfg:
            pl.seed_everything(cfg.SEED)
            print("Everything is seeded to: ", cfg.SEED)

        super().__init__()
        self.cfg = cfg
        self.model_cfg = self.cfg.MODEL
        self.sampling_method = self.cfg.EVALUATION.SAMPLING_METHOD

        self.motion_decoder = MotionLM(config=self.model_cfg)

        self.save_hyperparameters()

        self.validation_outputs = []
        self.validation_ground_truth = []

    def forward(self, batch_dict):
        forward_ret_dict = self.motion_decoder(batch_dict)
        return forward_ret_dict

    def get_loss(self, data_dict, gt_dict, forward_ret_dict):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss(data_dict, gt_dict, forward_ret_dict)
        return loss, tb_dict, disp_dict

    def training_step(self, batch, batch_idx):
        data_dict, gt_dict = batch
        forward_ret_dict = self(data_dict)
        loss, tb_dict, disp_dict = self.get_loss(data_dict, gt_dict, forward_ret_dict)
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in tb_dict.items()},
            batch_size=data_dict["encoder/agent_feature"].shape[0],
            # on_epoch=True,
            # prog_bar=True,
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()

    @torch.no_grad()
    def autoregressive_generate(
        self, batch, compress_step=None, total_step=None, return_step_model_out=False, sampling_method=None
    ):
        data_dict, gt_dict = batch

        start_time = gt_dict['current_time_index'].unique().item()

        B, T, N, D_actor = data_dict['actor_feature'].shape
        _, _, L, D_light = data_dict['traffic_light_feature'].shape

        if total_step is None:
            rollout_T = T
        else:
            assert total_step <= T
            rollout_T = total_step

        feat = data_dict["encoder/agent_feature"]

        num_modes = self.motion_decoder.num_modes

        num_repeat = num_modes

        # num_repeat = 8

        def _repeat_for_modes(v):
            d = v.ndim
            v = v.unsqueeze(1)
            v = v.repeat(1, num_repeat, *((1, ) * (d - 1)))
            v = v.flatten(0, 1)
            return v

        copy_data_dict = {
            "encoder/agent_feature": feat.new_zeros([B * num_repeat, T, N, D_actor]),
            "encoder/agent_position": feat.new_zeros([B * num_repeat, T, N, 3]),
            "encoder/agent_valid_mask": feat.new_zeros([B * num_repeat, T, N], dtype=bool),
            "decoder/actor_type": _repeat_for_modes(data_dict["decoder/actor_type"]),
            "encoder/map_feature": _repeat_for_modes(data_dict["encoder/map_feature"]),
            "encoder/map_feature_valid_mask": _repeat_for_modes(data_dict["encoder/map_feature_valid_mask"]),
            "encoder/map_position": _repeat_for_modes(data_dict["encoder/map_position"]),
            "encoder/traffic_light_feature": feat.new_zeros([B * num_repeat, T, L, D_light]),
            "encoder/traffic_light_position": _repeat_for_modes(data_dict["encoder/traffic_light_position"]),
            "encoder/traffic_light_valid_mask": feat.new_zeros([B * num_repeat, T, L], dtype=bool)
        }

        for k in [
                "encoder/agent_feature",
                "encoder/agent_valid_mask",
                "encoder/agent_position",
                "encoder/traffic_light_valid_mask",
                "encoder/traffic_light_feature",
        ]:
            v = data_dict[k]
            v = v.reshape(B, 1, *(v.shape[1:]))
            v = v.repeat(1, num_repeat, *(1, ) * len(v.shape[2:]))
            v = v.flatten(0, 1)
            assert v.shape[1] == T, (k, v.shape)
            copy_data_dict[k][:, :start_time + 1] = v[:, :start_time + 1]

        model_output_collection = {
            "sampled_position": copy_data_dict["encoder/map_position"].new_zeros([B, T, N, num_repeat, 3]),
            "log_probability": copy_data_dict["encoder/map_position"].new_zeros([B, T, N, num_repeat])
        }
        model_output_collection["log_probability"].fill_(float("-inf"))

        if compress_step is None:
            compress_step = self.motion_decoder.compress_step

        model_output = {}

        if return_step_model_out:
            step_model_out = []

        for step in range(start_time, rollout_T, compress_step):

            # input_dict is a snapshot of data_dict
            input_dict = {
                "encoder/map_feature": copy_data_dict["encoder/map_feature"].clone(),
                "encoder/map_feature_valid_mask": copy_data_dict["encoder/map_feature_valid_mask"].clone(),
                "encoder/map_position": copy_data_dict["encoder/map_position"].clone(),
                "encoder/traffic_light_position": copy_data_dict["encoder/traffic_light_position"].clone(),
                "encoder/traffic_light_valid_mask": copy_data_dict["encoder/traffic_light_valid_mask"][:, :step +
                                                                                                       1].clone(),
                "encoder/agent_feature": copy_data_dict["encoder/agent_feature"][:, :step + 1].clone(),
                "encoder/agent_valid_mask": copy_data_dict["encoder/agent_valid_mask"][:, :step + 1].clone(),
                "encoder/agent_position": copy_data_dict["encoder/agent_position"][:, :step + 1].clone(),
                "encoder/traffic_light_feature": copy_data_dict["encoder/traffic_light_feature"][:, :step + 1].clone(),
                "in_evaluation": True,
                "output_compress_step": compress_step,
                "decoder/actor_type": copy_data_dict["decoder/actor_type"]
            }

            for k in ["map_token", "query_cache"]:
                if k in model_output:
                    input_dict[k] = model_output[k]

            model_output = self(input_dict)

            if return_step_model_out:
                step_model_out.append(model_output)

            # Fuse the predicted state into data dict directly.
            copy_data_dict, pred_actor_pos, future_start, future_end, log_probability = sample_from_distributions_and_merge(
                step=step,
                copy_data_dict=copy_data_dict,
                model_output=model_output,
                batch_size=B,
                compress_step=compress_step,
                sampling_method=sampling_method or self.sampling_method,
                max_known_step=start_time + 1
            )

            if step == start_time:
                # overwrite the predicted position to initial position, if the actor is not moved in the initial
                # interval

                # [B, 1, N], [B, 1, N, 2]
                displacement, last_pos = get_displacement(
                    input_dict["encoder/agent_position"][:, :start_time],
                    input_dict["encoder/agent_valid_mask"][:, :start_time]
                )
                last_pos = last_pos.reshape(B, num_repeat, 1, N, 3)
                displacement = displacement.reshape(B, num_repeat, 1, N)

            last_pos_use = last_pos.repeat(1, 1, pred_actor_pos.shape[1], 1, 1)
            last_pos_use = last_pos_use.permute(0, 2, 3, 1, 4)

            static_actor_mask = displacement < 0.001
            static_actor_mask_pred = static_actor_mask.reshape(B, num_repeat, 1,
                                                               N).repeat(1, 1, pred_actor_pos.shape[1], 1)
            static_actor_mask_pred = static_actor_mask_pred.permute(0, 2, 3, 1)  # [B, step, N, num_modes]

            # [B*num_modes, T, N, 3] -> [B, num_modes, T, N, 3]
            pred_actor_pos_reform = pred_actor_pos.reshape(B, num_repeat, *pred_actor_pos.shape[1:])
            # -> [B, T, N, num_modes, 3]
            pred_actor_pos_reform = pred_actor_pos_reform.permute(0, 2, 3, 1, 4)

            pred_actor_pos_reform[static_actor_mask_pred] = last_pos_use[static_actor_mask_pred]

            model_output_collection["sampled_position"][:, future_start:future_end] = pred_actor_pos_reform

            # Since we will sum up all scores for each mode,
            # We can fill divide the score by 5 (compress_step) before filling them into matrix,
            # By doing so we avoid sum the same the log probability 5 times when computing the trajectory-level scores.
            log_probability_reform = log_probability.reshape(B, num_repeat, 1, N)
            log_probability_reform = log_probability_reform.permute(0, 2, 3, 1)  # -> [B, 1, N, num_modes]
            model_output_collection["log_probability"][:, future_start:future_end] = \
                log_probability_reform / (future_end - future_start)

            if step in [10, 15, 20, 40, 60, 90]:
                # Compute the first batch output's loss. It should be similar to training loss.
                self.log_dict(
                    self.motion_decoder.get_position_loss(
                        data_dict, model_output_collection, step, start_time, future_end
                    ),
                    # prog_bar=True
                )

        if return_step_model_out:
            return copy_data_dict, model_output_collection, step_model_out

        return copy_data_dict, model_output_collection

    def validation_step(
        self, batch, batch_idx, output_compress_step=None, total_step=None, return_dict=False, sampling_method=None
    ):

        # TODO: Add this back
        return

        data_dict, gt_dict = batch

        if output_compress_step is None:
            output_compress_step = self.config.EVALUATION.OUTPUT_COMPRESS_STEP

        copy_data_dict, model_output_collection = self.autoregressive_generate(
            batch, compress_step=output_compress_step, total_step=total_step, sampling_method=sampling_method
        )
        final_pred_dicts = generate_predicted_trajectory_for_eval(data_dict, gt_dict, model_output_collection)
        final_pred_dicts.update(gt_dict)
        self.validation_outputs.append(final_pred_dicts)
        if return_dict:
            final_pred_dicts["model_output_collection"] = model_output_collection
            final_pred_dicts["data_dict"] = data_dict
            return final_pred_dicts

    def on_validation_epoch_end(self):

        # TODO: Add this
        return

        st = time.time()

        # https://lightning.ai/docs/pytorch/latest/accelerators/accelerator_prepare.html?highlight=hardware
        torch.cuda.empty_cache()

        # PZH NOTE: Hack to implement our own all_gather across ranks.
        self.trainer.strategy.barrier()

        # if "Wandb" in str(self.trainer.logger):
        #     tmpdir = os.path.join(self.trainer.logger.version, "validation_tmpdir")
        #     os.makedirs(self.trainer.logger.version, exist_ok=True)
        #     os.makedirs(tmpdir, exist_ok=True)
        # else:
        tmpdir = os.path.join(self.trainer.log_dir, "validation_tmpdir")
        # os.makedirs(self.trainer.log_dir, exist_ok=True)
        os.makedirs(tmpdir, exist_ok=True)

        self.validation_outputs = [
            {
                k: v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in final_pred_dicts.items()
            } for final_pred_dicts in self.validation_outputs
        ]

        with open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(self.global_rank)), 'wb') as f:
            pickle.dump(self.validation_outputs, f)
        self.validation_outputs.clear()
        self.trainer.strategy.barrier()

        self.log("monitoring_step", float(self.global_step))

        if self.trainer.is_global_zero:
            validation_list = []
            for i in range(self.trainer.world_size):
                file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))

                for sleep in range(10):
                    if not os.path.isfile(file):
                        time.sleep(1)
                        print(f"Can't find file: {file}. Sleep {sleep} seconds.")
                with open(file, "rb") as f:
                    val_outputs = pickle.load(f)
                    validation_list.extend(val_outputs)

            if self.config.DELETE_EVAL_RESULT:
                shutil.rmtree(tmpdir)

            # print("==== log eval dir: ", eval_output_dir)

            # with open(os.path.join(eval_output_dir, 'validation_result.pkl'), 'wb') as f:
            #     pickle.dump(validation_list, f)

            # print(f"===== Start evaluation: {time.time() - st:.3f}")
            # scenario_id_map = self.trainer.val_dataloaders.dataset.scenario_id_map
            result_str, result_dict = waymo_evaluation(validation_list)
            result_dict = {f"eval/{k}": float(v) for k, v in result_dict.items()}
            self.log_dict(result_dict, rank_zero_only=True)

            for k in ['eval/minADE', 'eval/minFDE', 'eval/MissRate', 'eval/mAP']:
                self.log(name=k.split("/")[1], value=result_dict[k], rank_zero_only=True)

            self.print(result_str)
            print(f"===== Finish evaluation: {time.time() - st:.3f}")

        torch.cuda.empty_cache()

    def configure_optimizers(self):
        opt_cfg = self.cfg.OPTIMIZATION

        if opt_cfg.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                [each[1] for each in self.named_parameters()],
                lr=opt_cfg.LR,
                weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            )
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0), betas=(0.9, 0.95)
            )
        else:
            assert False

        if opt_cfg.get('SCHEDULER', None) == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2,
                T_mult=1,
                eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
                last_epoch=-1,
            )
        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':

            def lr_lbmd(cur_epoch):
                cur_decay = 1
                for decay_step in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20]):
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * opt_cfg.LR_DECAY
                return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

            scheduler = LambdaLR(optimizer, lr_lbmd)

        elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=opt_cfg.LR_CLIP / opt_cfg.LR,
                total_iters=opt_cfg.NUM_EPOCHS,
            )
        else:
            scheduler = None

        return {
            "optimizer": optimizer,

            # PZH NOTE: The scheduler step will be added 1 after each epoch.
            "lr_scheduler": scheduler
        }


def generate_predicted_trajectory_for_eval(data_dict, gt_dict, forward_ret_dict):
    """
    This function will extract the predicted state for all objects.
    We can then compare those state with the ground truths for formal evaluation in the motion forecasting task.
    Note that this function is only called in validation_step and for evaluation only.
    We don't use the result here to compute loss.
    """
    actor_valid_mask = data_dict["encoder/agent_valid_mask"]  # [B, T, N]
    predicted_position = forward_ret_dict["sampled_position"]  # [B*num_modes, T, N, num_modes, 2]

    # =======================================================================
    # For debug use only, overwrite the predicted result by GT to check evaluation pipeline.
    # pred_pos = data_dict["encoder/agent_position"][data_dict["encoder/agent_valid_mask"]]
    # pred_pos = pred_pos.reshape(-1, 1, 3).repeat(1, 1, 1)  # Extend the num_modes dim.
    # =======================================================================

    B, T, N = actor_valid_mask.shape
    _, _, _, num_modes, _ = predicted_position.shape

    track_index_to_predict = data_dict["track_index_to_predict"].long()  # [B, num interested actors]
    _, max_actor_to_predict = track_index_to_predict.shape

    map_center = gt_dict["map_center"][..., :2]  # [B, 2]
    map_heading = gt_dict["encoder/map_heading"]  # [B, ]

    map_center = map_center.reshape(B, 1, 1, 1, 2).repeat(1, T, N, num_modes, 1)

    map_heading = map_heading.reshape(B, 1, 1, 1).repeat(1, T, N, num_modes)

    predicted_position = rotate(x=predicted_position[..., 0], y=predicted_position[..., 1], angle=map_heading)

    predicted_position = predicted_position + map_center

    predicted_position[~data_dict["encoder/agent_valid_mask"][..., None, None].expand(B, T, N, num_modes, 2)] = 0

    # [B, T, N, num_modes, 2] -> [B, N, num_modes, T, 2]
    predicted_position = predicted_position.permute(0, 2, 3, 1, 4)

    # [B, N, num_modes, T, 2] -> [B * N, num_modes, T, 2]
    predicted_position = predicted_position.flatten(0, 1)

    # change the index range from [0, N] to [0, B*N]
    valid_track_index = torch.where(track_index_to_predict != -1)
    track_index_to_predict += torch.arange(B).to(track_index_to_predict).reshape(-1, 1) * N
    track_index_flatten = track_index_to_predict[valid_track_index]

    assert track_index_flatten.max().item() < predicted_position.shape[0], (
        track_index_flatten, predicted_position.shape
    )

    # [sum valid actor to predict, num_modes, T, 2]
    predicted_traj = predicted_position[track_index_flatten]

    # Assume we only have one current_time_index = 10.
    current_time_index = gt_dict["current_time_index"].unique().item()

    predicted_traj = predicted_traj[:, :, current_time_index + 1:, :2]

    if "score" in forward_ret_dict:
        score = forward_ret_dict["score"]
    else:
        score = forward_ret_dict["log_probability"]

    # [B, T, N, num_modes] -> [B, N, num_modes, T]
    score = score.permute(0, 2, 3, 1)
    # [B, N, num_modes, T] -> [B * N, num_modes, T]
    score = score.flatten(0, 1)
    # [sum valid actor to predict, num_modes, T]
    score = score[track_index_flatten]
    score = score[:, :, current_time_index + 1:]

    # ===== DEBUG CODE =====
    # score[:, :1] = 10000
    # score[:, 1:] = 0
    # score.fill_(1)
    # ===== DEBUG CODE =====

    # =======================================================================
    # debug_mask in [num valid, 80]
    # debug_mask = actor_valid_mask.permute(0, 2, 1, 3).flatten(0, 1)[track_index_flatten][:, 11:][..., 0]
    # pred_origin = predicted_traj[:, 0]
    # gt_origin = gt_dict["center_gt_trajs_src"][:, 11:, :2]
    # assert (pred_origin[debug_mask] - gt_origin[debug_mask]).abs().max().item() < 0.1
    # =======================================================================
    assert predicted_traj.shape[-1] == 2
    return {
        "pred_trajs": predicted_traj,
        "pred_scores": score.sum(-1)  # sum of log probability
    }
