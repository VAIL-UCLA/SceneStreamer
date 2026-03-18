import functools
import logging

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scenestreamer.infer.initial_state import generate_initial_state
from scenestreamer.models.gen_model import GenModel, SceneStreamerModel
from scenestreamer.dataset.preprocessor import NUM_TG_MULTI
from scenestreamer.models.language_motionlm import LanguageMotionLM
from scenestreamer.models.motionlm import MotionLM
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.tokenization.trafficgen_tokenizers import TrafficGenTokenizer
from scenestreamer.utils import lr_schedule
from scenestreamer.utils import utils
from scenestreamer.dataset.preprocessor import slice_trafficgen_data
from scenestreamer.models.scenestreamer_model import get_edge_info_for_scenestreamer

logger = logging.getLogger(__file__)


def update_ema(target_params, source_params, rate=0.99):
    """
    PZH: From https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/engine_mar.py#L19

    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def safe_entropy(logits, epsilon=1e-5):
    """
    Computes the entropy of the given logits safely by replacing NaN and Inf values.
    :param logits: Input logits tensor.
    :param epsilon: A small value to add to the logits to avoid log(0) which results in NaN.
    :return: Mean entropy of the logits.
    """
    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)

    # Adding a small epsilon to logits to avoid log(0)
    logits = logits + epsilon

    # Compute softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute entropy
    entropy = -(probs * torch.log(probs)).sum(-1)

    # Return the mean entropy
    return entropy.mean()


class MotionLMLightning(pl.LightningModule):
    def __init__(self, config):
        if "SEED" in config:
            pl.seed_everything(config.SEED)
            print("Everything is seeded to: ", config.SEED)
        super().__init__()
        self.config = config

        if config.MODEL.NAME in ["motionlm", "gpt"]:
            self.model = MotionLM(config=self.config)
        elif config.MODEL.NAME == "gen":
            self.model = GenModel(config=self.config)
        elif config.MODEL.NAME == "scenestreamer":
            # self.model = SceneStreamerModel(config=self.config)
            from scenestreamer.models.scenestreamer_model import SceneStreamer
            self.model = SceneStreamer(config=self.config)

        elif config.MODEL.NAME == "language_motionlm":
            self.model = LanguageMotionLM(config=self.config)
        else:
            raise ValueError(f"Unknown model name: {config.MODEL.NAME}")

        if config.EVALUATION.NAME in ["waymo_motion_prediction", "waymo_prediction", "womd"]:
            from scenestreamer.eval.waymo_motion_prediction_evaluator import WaymoMotionPredictionEvaluator
            self.evaluator = WaymoMotionPredictionEvaluator(config=config)
            # if self.config.EVALUATION.PREDICT_ALL_AGENTS is False:
            #     assert self.config.PREPROCESSING.ADD_SDC_TO_OBJECT_OF_INTEREST is False
            if self.config.SUBMISSION.GENERATE_SUBMISSION:
                assert self.config.EVALUATION.PREDICT_ALL_AGENTS is False
                assert self.config.PREPROCESSING.ADD_SDC_TO_OBJECT_OF_INTEREST is False

        elif config.EVALUATION.NAME in ["wosac2023", "wosac2024", "sgen"]:

            # Let's overwrite some configs here
            # Note that the WOSAC eval code will take care of tracks_to_predict
            assert config.EVALUATION.PREDICT_ALL_AGENTS is True
            # assert config.PREPROCESSING.ADD_SDC_TO_OBJECT_OF_INTEREST is True
            # assert config.EVALUATION.NUM_MODES == 32
            # config.EVALUATION.MAXIMUM_BATCH_SIZE = min(config.EVALUATION.MAXIMUM_BATCH_SIZE, 16)
            assert config.DATA.SD_PASSTHROUGH
            # config.DATA.SD_PASSTHROUGH = True

            from scenestreamer.eval.waymo_sim_agent_evaluator import WaymoSimAgentEvaluator
            self.evaluator = WaymoSimAgentEvaluator(config=config)
        elif config.EVALUATION.NAME in ["lmdb"]:
            from scenestreamer.eval.lmdb_evaluator import LMDBEvaluator
            self.evaluator = LMDBEvaluator(config=config)

        elif config.EVALUATION.NAME in ["peng"]:
            from scenestreamer.eval.peng_evaluator import PengEvaluator
            self.evaluator = PengEvaluator(config=config)

        else:
            raise ValueError(f"Unknown evaluation name: {config.EVALUATION.NAME}")

        self.save_hyperparameters(OmegaConf.to_container(self.config))

        self._tokenizer = get_tokenizer(self.config)
        # self.validation_outputs = []
        # self.validation_ground_truth = []

        self.exp_name = None

        if self.config.USE_TRAFFICGEN:
            self._trafficgen_tokenizer = TrafficGenTokenizer(config)
            from scenestreamer.eval.test_trafficgen_eval import TrafficGenEvaluator
            self._trafficgen_evaluator = TrafficGenEvaluator(config, device=self.device)

        self.rl_finetuner = None

    def forward(self, batch_dict):
        return self.model(batch_dict)

    def get_diffusion_loss(self, data_dict):
        loss_dict = self.model.motion_decoder.get_diffusion_loss(data_dict)
        loss_dict = {k: (v.mean() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        loss = loss_dict["loss"]
        try:
            loss_dict["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass

        return loss, loss_dict

    def get_loss(self, data_dict):
        if self.config.USE_DIFFUSION:
            return self.get_diffusion_loss(data_dict)

        loss_stat = {}
        loss = 0.0

        if self.config.USE_MOTION:

            # Get the decoder's output
            output_logit = data_dict["decoder/output_logit"]  # (B, T_skipped + 1, N, num_actions)

            # Get the GT actions
            target_action = data_dict["decoder/target_action"]  # (B, T_skipped, N)
            target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
            assert output_logit.shape[:3] == target_action.shape[:3], (output_logit.shape, target_action.shape)


            # Get loss
            if self.config.OPTIMIZATION.USE_FOCAL_LOSS:
                raise ValueError
                from torchvision.ops import sigmoid_focal_loss
                # Compute Focal Loss
                alpha = 0.25
                gamma = 2
                target_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                loss = sigmoid_focal_loss(
                    inputs=output_logit, targets=target_onehot, alpha=alpha, gamma=gamma, reduction="none"
                )
            else:

                if self.config.TOKENIZATION.TOKENIZATION_METHOD == "fast":
                    B, T_full, N, max_len, num_toks = output_logit.shape

                    input_mask = data_dict["decoder/input_action_valid_mask"]
                    output_mask = data_dict["decoder/target_action_valid_mask"]
                    assert input_mask.shape == output_mask.shape
                    assert input_mask.shape == (B, T_full, N)
                    valid_gt_mask = input_mask & output_mask

                    fast_input_token = (
                        utils.unwrap(data_dict["fast_input_token"], data_dict["decoder/input_action_valid_mask"], fill=44444)
                    )
                    fast_input_token[fast_input_token == data_dict["fast_pad_token"]] = -1
                    # fast_input_valid_mask = data_dict["decoder/input_action_valid_mask"]

                    # The input tokens should be in shape (B, T, N, max_len)
                    # At time t, it should already be the sequence of target actions! (this point is easily missed)
                    target_action = fast_input_token[valid_gt_mask]
                    masked_logit = output_logit[valid_gt_mask]

                    # In fast tokenization, the first token is always the SOS token so remove them.
                    # Note that the target_action and output_logit are already in 2D/3D.
                    target_action = target_action[:, 1:]
                    masked_logit = masked_logit[:, :-1]

                    assert not (target_action == 44444).any()
                    assert not (masked_logit == 0).all(-1).any()

                    target_action_neg1_mask = target_action != -1
                    target_action = target_action[target_action_neg1_mask]

                    assert (target_action == data_dict["fast_eos_token"]).any()

                    masked_logit = masked_logit[target_action_neg1_mask]

                    # rate_777 = (masked_logit.argmax(dim=-1) == 777).float().mean()
                    # print("RATE 777: ", rate_777)

                    loss = torch.nn.functional.cross_entropy(input=masked_logit, target=target_action, reduction="none")

                    output_logit = masked_logit

                else:

                    output_logit = output_logit[target_action_valid_mask]
                    target_action = target_action[target_action_valid_mask]

                    loss = torch.nn.functional.cross_entropy(input=output_logit, target=target_action, reduction="none")

            original_loss = loss
            loss = loss.mean()

            assert not np.isnan(loss.item())
            assert not np.isinf(loss.item())

            with torch.no_grad():
                encodings = F.one_hot(output_logit.argmax(-1),
                                      output_logit.shape[-1]).float().reshape(-1, output_logit.shape[-1])
                avg_probs = encodings.mean(0)
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
                cluster_use = torch.sum(avg_probs > 0)

                gt_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                gt_encodings = gt_onehot.reshape(-1, output_logit.shape[-1])
                gt_avg_probs = gt_encodings.mean(0)
                gt_perplexity = (-(gt_avg_probs * torch.log(gt_avg_probs + 1e-10)).sum()).exp()
                gt_cluster_use = torch.sum(gt_avg_probs > 0)
                debug_gt_c_use = (gt_encodings.sum(0) > 0).sum()  # .mean()

                pred_act = output_logit.argmax(-1)
                acc = torch.sum(pred_act == target_action) / target_action.shape[0]
                entropy = safe_entropy(output_logit)
                pred_act = pred_act.float()

                rate_default_pred = (pred_act == self._tokenizer.default_action).float().mean()
                rate_default_gt = (target_action == self._tokenizer.default_action).float().mean()

                num_trained_tokens = len(target_action)
                num_trained_tokens_sum = self.trainer.world_size * num_trained_tokens

                # print("ACCURACY: ", acc, "ENTROPY: ", entropy.mean())

                loss_stat.update(
                    {
                        "original_loss": original_loss.mean(),
                        "accuracy": acc,
                        "entropy": entropy.mean(),
                        "avg_action": pred_act.mean(),
                        "max_action": pred_act.max(),
                        "min_action": pred_act.min(),
                        "perplexity": perplexity,
                        "gt_perplexity": gt_perplexity,
                        "cluster_use": cluster_use,
                        "gt_cluster_use": gt_cluster_use,
                        "rate_84": rate_default_gt,
                        "rate_default_gt": rate_default_gt,
                        "rate_default_pred": rate_default_pred,
                        "num_trained_tokens": num_trained_tokens,
                        "num_trained_tokens_sum": num_trained_tokens_sum,
                        "toks": num_trained_tokens_sum,
                    }
                )

                if self.config.BACKWARD_PREDICTION:
                    in_back_mask = data_dict["in_backward_prediction"]
                    in_back_mask = in_back_mask.reshape(-1, 1, 1).expand(*target_action_valid_mask.shape)
                    in_back_mask = in_back_mask[target_action_valid_mask]
                    acc2 = (pred_act == target_action)
                    acc_in_back = (acc2 & in_back_mask).sum() / in_back_mask.sum()
                    acc_in_forward = (acc2 & ~in_back_mask).sum() / (~in_back_mask).sum()
                    loss_in_back = original_loss[in_back_mask].mean()
                    loss_in_forward = original_loss[~in_back_mask].mean()
                    entropy_in_back = safe_entropy(output_logit[in_back_mask]).mean()
                    entropy_in_forward = safe_entropy(output_logit[~in_back_mask]).mean()
                    loss_stat.update(
                        {
                            "accuracy_in_backward": acc_in_back,
                            "accuracy_in_forward": acc_in_forward,
                            "loss_in_backward": loss_in_back,
                            "loss_in_forward": loss_in_forward,
                            "entropy_in_backward": entropy_in_back,
                            "entropy_in_forward": entropy_in_forward,
                            "backward_ratio": in_back_mask.float().mean(),
                        }
                    )

        if self.config.RECONSTRUCT_MAP:
            gt_map_feat = data_dict["encoder/map_feature"]
            map_feat_valid_mask = data_dict["encoder/map_valid_mask"]
            polypoint_valid_mask = data_dict["encoder/map_feature_valid_mask"]
            polypoint_valid_mask = polypoint_valid_mask[map_feat_valid_mask]  # (valid points, 128)
            map_feat = gt_map_feat[map_feat_valid_mask]  # (num_valid_map_features, 128, 27)
            polypoint = map_feat[:, :, :2]  # (valid map feat, 128, 2)
            num_points = polypoint.shape[1]
            gt_valid_mask = polypoint_valid_mask.unsqueeze(-1).expand_as(polypoint)
            gt = torch.where(gt_valid_mask, polypoint, torch.zeros_like(polypoint))
            gt_valid_mask = gt_valid_mask.reshape(-1, num_points * 2)
            gt = gt.reshape(-1, num_points * 2)
            map_token = data_dict["encoder/map_token"]
            out = self.model.map_recon_head(self.model.map_recon_head_prenorm(map_token[map_feat_valid_mask]))

            # out.shape = (num_valid_map_features, 128 * 2)
            map_recon_loss = torch.nn.functional.mse_loss(out, gt, reduction="none")
            map_recon_loss = map_recon_loss[gt_valid_mask]
            map_recon_loss = map_recon_loss.mean()

            loss += map_recon_loss
            loss_stat["map_recon_loss"] = map_recon_loss
            loss_stat["map_recon_mask_rate"] = gt_valid_mask.float().mean()

        # DEBUG CODE to find unused parameters:
        # gs = torch.autograd.grad(loss.mean(), self.parameters(), allow_unused=True, retain_graph=True)
        # ns = [n for n, v in self.named_parameters()]
        # printed = False
        # for c, g in enumerate(gs):
        #     if g is None:
        #         print(ns[c])
        #         printed = True
        # if not printed:
        #     print("No unused parameters found.")

        if (self.config.USE_TRAFFICGEN and (self.config.TRAIN_TRAFFICGEN is True or self.config.TRAIN_TRAFFICGEN is None)):
            data_dict = self.model.trafficgen_decoder.forward(data_dict)

            tg_gt_action = data_dict["decoder/input_action_for_trafficgen"][:, 1:]

            tg_gt_mask = data_dict["decoder/input_action_valid_mask_for_trafficgen"][:, 1:]
            tg_gt = tg_gt_action[tg_gt_mask]
            tg_logit = data_dict["decoder/output_logit_for_trafficgen"][:, :-1][tg_gt_mask]
            tg_loss = torch.nn.functional.cross_entropy(input=tg_logit, target=tg_gt, reduction="none")

            tg_loss = tg_loss.mean()
            loss += tg_loss
            tg_accuracy = torch.sum(tg_logit.argmax(-1) == tg_gt) / tg_gt.shape[0]
            loss_stat.update(
                {
                    "trafficgen_loss": tg_loss,
                    "trafficgen_accuracy": tg_accuracy,
                    "trafficgen_entropy": safe_entropy(tg_logit).mean(),
                }
            )

            # current_input_action[:, 0] is the START_ACTION, so need to skip it.
            tg_gt_offset_mask = tg_gt_mask & (tg_gt_action != self._trafficgen_tokenizer.start_action_id
                                              ) & (tg_gt_action != self._trafficgen_tokenizer.end_action_id)

            gt_agent_type = data_dict["decoder/agent_type_for_trafficgen"][:, 1:]
            agent_type_output = self.model.trafficgen_decoder.forward_agent_type(data_dict, action=tg_gt_action)
            agent_type_loss = torch.nn.functional.cross_entropy(
                input=agent_type_output[tg_gt_offset_mask], target=gt_agent_type[tg_gt_offset_mask], reduction="mean"
            )
            loss += agent_type_loss

            offset_output = self.model.trafficgen_decoder.forward_offset(
                data_dict, action=tg_gt_action, agent_type=gt_agent_type
            )

            for kid, k in enumerate(["position_x", "position_y", "heading", "velocity_x", "velocity_y", "length",
                                     "width", "height"]):
                tg_gt_offset = data_dict["decoder/target_offset_for_trafficgen"][:, :, kid]
                tg_logit = offset_output[k]

                tg_logit = tg_logit[tg_gt_offset_mask]
                tg_gt_offset = tg_gt_offset[tg_gt_offset_mask]
                tg_offset_loss = torch.nn.functional.cross_entropy(
                    input=tg_logit, target=tg_gt_offset, reduction="mean"
                )
                loss += tg_offset_loss
                tg_accuracy = torch.sum(tg_logit.argmax(-1) == tg_gt_offset) / tg_gt_offset.shape[0]
                loss_stat.update({
                    f"trafficgen_loss_{k}": tg_loss,
                    f"trafficgen_accuracy_{k}": tg_accuracy,
                })

        loss_stat["total_loss"] = loss
        try:
            loss_stat["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass
        return loss, loss_stat

    def get_loss_for_scenestreamer(self, data_dict):

        def _safe_cross_entropy(input, target, reduction="mean"):
            assert input.shape[:-1] == target.shape, (input.shape, target.shape)
            assert target.min() >= 0, (target.min(), target.max())
            assert input.shape[-1] > target.max(), (input.shape, target.shape)
            return torch.nn.functional.cross_entropy(input=input, target=target, reduction=reduction)

        def _accuracy(input, target):
            assert input.ndim == 2
            assert target.ndim == 1
            pred_act = input.argmax(-1)
            acc = torch.sum(pred_act == target) / target.shape[0]
            return acc

        loss_stat = {}
        loss = 0.0


        # ===== motion loss =====
        output_logit = data_dict["model/motion_logit"]  # (B, T_skipped + 1, N, num_actions)
        # Get the GT actions
        target_action = data_dict["decoder/target_action"]  # (B, T_skipped, N)
        target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
        assert output_logit.shape[:3] == target_action.shape[:3], (output_logit.shape, target_action.shape)
        output_logit = output_logit[target_action_valid_mask]
        target_action = target_action[target_action_valid_mask]
        motion_loss = _safe_cross_entropy(input=output_logit, target=target_action)
        loss += motion_loss
        loss_stat["motion_loss"] = motion_loss
        motion_accuracy = _accuracy(input=output_logit, target=target_action)
        loss_stat["motion_accuracy"] = motion_accuracy

        # ===== trafficgen loss =====
        tg_gt_action = data_dict["decoder/input_action_for_trafficgen"]
        # tg_gt_mask = data_dict["decoder/input_action_valid_mask_for_trafficgen"]
        B, T, N = data_dict["decoder/target_action"].shape
        tg_gt_action = slice_trafficgen_data(tg_gt_action[:, :, 1:-1].reshape(B, T, N, NUM_TG_MULTI), dim=1)
        agent_valid_mask = data_dict["decoder/input_action_valid_mask"]

        if self.model.no_tg:
            pass

        else:
            tg_agent_valid_mask = slice_trafficgen_data(agent_valid_mask, dim=1)
            agent_type_logit = data_dict["model/trafficgen_agent_type_logit"]
            agent_type_gt = tg_gt_action[..., 1]
            agent_type_gt = agent_type_gt[tg_agent_valid_mask]
            assert agent_type_gt.min() != -1
            agent_type_gt[agent_type_gt == self.model.veh_id] = 0
            agent_type_gt[agent_type_gt == self.model.ped_id] = 1
            agent_type_gt[agent_type_gt == self.model.cyc_id] = 2
            agent_type_input = agent_type_logit[tg_agent_valid_mask]
            agent_type_loss = _safe_cross_entropy(
                input=agent_type_input,
                target=agent_type_gt,
            )
            loss += agent_type_loss
            loss_stat["trafficgen_agent_type_loss"] = agent_type_loss
            loss_stat["trafficgen_agent_type_accuracy"] = _accuracy(input=agent_type_input, target=agent_type_gt)

            map_id_logit = data_dict["model/trafficgen_map_id_logit"]
            map_id_gt = tg_gt_action[:, :, :, 2]
            assert map_id_gt.shape[:3] == map_id_logit.shape[:3]
            assert map_id_gt[tg_agent_valid_mask].min()>=0
            map_id_input = map_id_logit[tg_agent_valid_mask]
            map_id_target = map_id_gt[tg_agent_valid_mask]
            map_id_loss = _safe_cross_entropy(
                input=map_id_input,
                target=map_id_target
            )
            loss += map_id_loss
            loss_stat["trafficgen_map_id_loss"] = map_id_loss
            loss_stat["trafficgen_map_id_accuracy"] = _accuracy(input=map_id_input, target=map_id_target)

            agent_state_logit = data_dict["model/trafficgen_agent_state_logit"][..., :-1, :]
            agent_state_gt = slice_trafficgen_data(data_dict["decoder/target_offset_for_trafficgen"], dim=1)
            assert agent_state_logit.shape[:4] == agent_state_gt.shape
            agent_state_loss = _safe_cross_entropy(
                input=agent_state_logit[tg_agent_valid_mask].flatten(0, 1),
                target=agent_state_gt[tg_agent_valid_mask].flatten(),
            )
            loss += agent_state_loss
            loss_stat["trafficgen_agent_state_loss"] = agent_state_loss

            # dest_id_logit = data_dict["model/trafficgen_dest_id_logit"]
            # dest_valid_mask = slice_trafficgen_data(data_dict["decoder/dest_map_index_valid_mask"], dim=1)
            # # raise error if some agent_valid_mask is False but dest_valid_mask is True
            # assert (dest_valid_mask & ~tg_agent_valid_mask).sum() == 0
            # dest_valid_mask = dest_valid_mask & tg_agent_valid_mask
            # dest_id_gt = slice_trafficgen_data(data_dict["decoder/dest_map_index_gt"], dim=1)
            # # assert (dest_id_gt == tg_gt_action[..., 4]).all()  # It's normal that they are not aligned.
            # assert dest_id_gt.shape[:3] == dest_id_logit.shape[:3]
            # dest_id_gt = dest_id_gt[dest_valid_mask]
            # assert dest_id_gt.min() >= 0
            # dest_id_logit_input = dest_id_logit[dest_valid_mask]
            # dest_id_loss = _safe_cross_entropy(
            #     input=dest_id_logit_input,
            #     target=dest_id_gt,
            # )
            # if self.config.PREPROCESSING.DEST_DROPOUT >= 1.0:
            #     loss += dest_id_loss * 0.0
            # else:
            #     loss += dest_id_loss
            # loss_stat["trafficgen_dest_id_loss"] = dest_id_loss
            # loss_stat["trafficgen_dest_id_accuracy"] = _accuracy(input=dest_id_logit_input, target=dest_id_gt)
            # no_pad_mask = dest_id_gt != self.model.trafficgen_sequence_pad_id
            # loss_stat["trafficgen_dest_id_accuracy_no_pad"] = _accuracy(input=dest_id_logit_input[no_pad_mask], target=dest_id_gt[no_pad_mask])

        # ===== traffic light loss =====
        traffic_light_logit = data_dict["model/traffic_light_logit"]
        traffic_light_gt = data_dict["encoder/traffic_light_state"]
        traffic_light_mask = data_dict["encoder/traffic_light_valid_mask"]
        if traffic_light_mask.any():
            traffic_light_loss = _safe_cross_entropy(
                input=traffic_light_logit[traffic_light_mask],
                target=traffic_light_gt[traffic_light_mask],
            )
            loss += traffic_light_loss
            traffic_light_accuracy = _accuracy(
                input=traffic_light_logit[traffic_light_mask],
                target=traffic_light_gt[traffic_light_mask],
            )
            loss_stat["traffic_light_accuracy"] = traffic_light_accuracy
        else:
            traffic_light_loss = _safe_cross_entropy(
                input=traffic_light_logit.flatten(0, 2)[:1],
                target=traffic_light_gt.flatten()[:1],
                reduction="mean"
            ) * 0.0
            loss += traffic_light_loss
            loss_stat["traffic_light_accuracy"] = np.nan
        loss_stat["traffic_light_loss"] = traffic_light_loss

        # DEBUG CODE to find unused parameters:
        # gs = torch.autograd.grad(loss.mean(), self.parameters(), allow_unused=True, retain_graph=True)
        # ns = [n for n, v in self.named_parameters()]
        # printed = False
        # for c, g in enumerate(gs):
        #     if g is None:
        #         print(ns[c])
        #         printed = True
        # if not printed:
        #     print("No unused parameters found.")

        # List parameter name and the gradient:
        # gs = torch.autograd.grad(loss.mean(), self.parameters(), allow_unused=True, retain_graph=True)
        # gs = {name: g for (name, _), g in zip(self.named_parameters(), gs)}

        loss_stat["total_loss"] = loss
        try:
            loss_stat["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass
        return loss, loss_stat

    def training_step_rl_finetuning(self, data_dict, batch_idx):
        assert self.config.SCENESTREAMER_NO_TG is True
        assert self.config.USE_RL_FINETUNING is True

        if self.rl_finetuner is None:
            from scenestreamer.rl_finetuning import RLFinetuner
            self.rl_finetuner = RLFinetuner(model=self.model, all_gather=self.all_gather)

        loss, loss_stat = self.rl_finetuner.get_loss(data_dict)
        try:
            loss_stat["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass

        pbar_keys = ("total_loss", "toks", "lr")
        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}
        self.log_dict(
            {f"{k}": float(v)
             for k, v in loss_stat.items() if k in pbar_keys},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            prog_bar=True,
        )
        if motion_stat:
            self.log_dict(
                {f"{k}": float(v)
                 for k, v in motion_stat.items()},
                batch_size=data_dict["encoder/map_feature"].shape[0],
                prog_bar=False,
            )
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            # on_epoch=True,
            prog_bar=False,
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def training_step(self, data_dict, batch_idx):

        if self.config.USE_RL_FINETUNING:
            return self.training_step_rl_finetuning(data_dict, batch_idx)

        # For profiling GPU usage.
        # torch.cuda.empty_cache()

        # print("RANK {} SCENARIO ID {} START".format(self.global_rank, data_dict["scenario_id"]))

        data_dict = self(data_dict)

        if self.config.MODEL.NAME == "scenestreamer":
            loss, loss_stat = self.get_loss_for_scenestreamer(data_dict)

        else:
            loss, loss_stat = self.get_loss(data_dict)

        pbar_keys = ("total_loss", "toks", "lr")

        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}

        self.log_dict(
            {f"{k}": float(v)
             for k, v in loss_stat.items() if k in pbar_keys},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            prog_bar=True,
        )
        if motion_stat:
            self.log_dict(
                {f"{k}": float(v)
                 for k, v in motion_stat.items()},
                batch_size=data_dict["encoder/map_feature"].shape[0],
                prog_bar=False,
            )
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            # on_epoch=True,
            prog_bar=False,
        )
        self.log('monitoring_step', float(self.global_step))

        # print("RANK {} SCENARIO ID {} END".format(self.global_rank, data_dict["scenario_id"]))

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # do something on_after_optimizer_step

        # if self.config.USE_DIFFUSION:
        #     self.model.motion_decoder.update_diffusion_step()

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def validation_step(self, data_dict, batch_idx):

        if self.config.EVAL_MOTION:

            if data_dict["encoder/map_valid_mask"].shape[1] == 0:
                sid = data_dict["scenario_id"]
                print("Warning: Empty map_valid_mask found for scenario: ", sid)
                logger.error(f"Empty map_valid_mask found for scenario: {sid}")
                return

            try:
                self.evaluator.validation_step(
                    data_dict,
                    batch_idx,
                    model=self.model,
                    global_rank=self.global_rank,
                    trainer=self.trainer,
                    logger=self.logger,
                    log_func=self.log,
                    log_dict_func=self.log_dict,
                    print_func=self.print,
                    lightning_model=self,
                )
            except Exception as error:
                scenario_ids = data_dict["scenario_id"]
                rank = self.global_rank
                msg = f"Error in validation_step: {batch_idx=}, {scenario_ids=}, {rank=}, {error=}"
                print(msg)
                raise RuntimeError(msg) from error

        if self.config.EVAL_TRAFFICGEN:

            if self.config.MODEL.NAME == "scenestreamer":
                if not hasattr(self, "scenestreamer_generator"):
                    from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator
                    self.scenestreamer_generator = SceneStreamerGenerator(model=self.model, device=self.device)
                with torch.no_grad():
                    self.scenestreamer_generator.reset(new_data_dict=data_dict)
                    output_data_dict = self.scenestreamer_generator.generate_scenestreamer_initial_state(progress_bar=True)
                data_dict.update(output_data_dict)
                stat = {}

            else:
                assert self.config.USE_TRAFFICGEN

                # data_dict = self.model.encode_scene(data_dict)
                # data_dict, stat = self.model.trafficgen_decoder.autoregressive_rollout_trafficgen(data_dict)

                data_dict, stat = generate_initial_state(model=self.model, data_dict=data_dict, force_add=True)

                # import matplotlib.pyplot as plt
                # pos_pred = data_dict["decoder/modeled_agent_position_for_trafficgen"][0][1:]
                # pred_mask = data_dict["decoder/input_action_valid_mask_for_trafficgen"][0][1:]
                # pos_target = data_dict["decoder/agent_position"][0, 0]
                # gt_mask = data_dict["decoder/agent_valid_mask"][0, 0]
                # plt.figure()
                # plt.scatter(pos_pred[pred_mask][:, 0].cpu().numpy(), pos_pred[pred_mask][:, 1].cpu().numpy(), c='r')
                # plt.scatter(pos_target[gt_mask][:, 0].cpu().numpy(), pos_target[gt_mask][:, 1].cpu().numpy(), c='b')
                # from scenestreamer.gradio_ui.plot import _plot_map
                # _plot_map({k: v[0].cpu().numpy() for k, v in data_dict.items() if isinstance(v, torch.Tensor)}, plt.gca())
                # plt.gca().set_aspect('equal', adjustable='box')
                # # plt.title(f"mmd_pos={mmd_pos.item()}")
                # plt.show()

            self._trafficgen_evaluator.validation_step(
                data_dict,
                stat,
                model=self.model,
                global_rank=self.global_rank,
                trainer=self.trainer,
                logger=self.logger,
                log_func=functools.partial(self.log, sync_dist=False),
                log_dict_func=self.log_dict,
                print_func=self.print,
                lightning_model=self,
            )

    def on_validation_epoch_end(self):
        """
        This function gathers intermediate evaluation result and pass them to the Waymo
        evaluation pipeline together and log the final results.
        """
        if self.config.EVAL_MOTION:
            self.log("monitoring_step", float(self.global_step))
            self.evaluator.on_validation_epoch_end(
                global_rank=self.global_rank,
                trainer=self.trainer,
                logger=self.logger,
                log_func=self.log,
                log_dict_func=self.log_dict,
                print_func=self.print,
                exp_name=self.exp_name,
            )

        # if self.config.EVAL_TRAFFICGEN:
        #     import functools
        #     self._trafficgen_evaluator.on_validation_epoch_end(
        #         global_rank=self.global_rank,
        #         trainer=self.trainer,
        #         logger=self.logger,
        #         log_func=functools.partial(self.log, sync_dist=True),
        #         log_dict_func=self.log_dict,
        #         print_func=self.print,
        #         exp_name=self.exp_name,
        #     )

    def configure_optimizers(self):
        """Required by Lightning."""
        opt_cfg = self.config.OPTIMIZATION

        if opt_cfg.OPTIMIZER == 'Adam':
            # optimizer = torch.optim.Adam(
            #     [each[1] for each in self.named_parameters()],
            #     lr=opt_cfg.LR,
            #     weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            # )
            raise ValueError()
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.LR,
                weight_decay=opt_cfg.get('WEIGHT_DECAY', 0),
                betas=(0.9, 0.95),
                eps=1e-5
            )
        else:
            assert False

        if opt_cfg.get('SCHEDULER', None) == 'cosine':

            utils.rank_zero_print("=====================================")
            if self.trainer.train_dataloader is not None:
                num_steps_per_epoch = len(self.trainer.train_dataloader)
            elif self.trainer.datamodule is not None and self.trainer.datamodule.train_dataset is not None:
                utils.rank_zero_print(
                    "Finding num_steps_per_epoch from datamodule...", len(self.trainer.datamodule.train_dataset),
                    self.trainer.datamodule.train_batch_size, self.trainer.world_size
                )
                num_steps_per_epoch = len(self.trainer.datamodule.train_dataset
                                          ) // (self.trainer.datamodule.train_batch_size * self.trainer.world_size)
            else:
                raise ValueError("Can't find num_steps_per_epoch")

            num_epochs = self.config.epochs
            total_steps = num_steps_per_epoch * num_epochs
            utils.rank_zero_print("Configuring cosine scheduler")
            utils.rank_zero_print("Num Steps per epoch: ", num_steps_per_epoch)
            utils.rank_zero_print("Num Epochs: ", num_epochs)
            utils.rank_zero_print("Total Steps: ", total_steps)
            utils.rank_zero_print("=====================================")

            scheduler = lr_schedule.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            raise ValueError()
            # def lr_lbmd(cur_epoch):
            #     cur_decay = 1
            #     for decay_step in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20]):
            #         if cur_epoch >= decay_step:
            #             cur_decay = cur_decay * opt_cfg.LR_DECAY
            #     return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)
            #
            # scheduler = LambdaLR(optimizer, lr_lbmd)

        elif opt_cfg.get('SCHEDULER', None) == 'linear':
            raise ValueError()
            scheduler = lr_schedule.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=opt_cfg.TRAINING_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'inverse_sqrt':
            scheduler = lr_schedule.get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        else:
            raise ValueError()
