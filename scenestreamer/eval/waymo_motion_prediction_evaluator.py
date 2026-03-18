# Referenced from https://github.com/Tsinghua-MARS-Lab/InterSim/blob/main/simulator/proto.py
import copy
import os
import pathlib
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

try:
    from waymo_open_dataset.protos import motion_submission_pb2
except ModuleNotFoundError:
    motion_submission_pb2 = None
import uuid
from scenestreamer.dataset.preprocessor import centralize_to_map_center
from scenestreamer.eval.waymo_eval import waymo_evaluation_optimized
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import wrap_to_pi, rotate


def joint_trajectory_nms(
        pred_trajs,  # [NUM_MODES, 80, N, 2]
        mode_scores,  # [NUM_MODES, 80, N]
        threshold=2.5,
        num_ret_modes=6,
        global_rank=None,
):
    sorted_scores, sorted_indices = torch.sort(mode_scores, descending=True)
    sorted_trajs = pred_trajs[sorted_indices]

    num_modes = sorted_trajs.shape[0]
    suppressed = torch.zeros(num_modes, dtype=torch.bool).to(sorted_trajs.device)
    keep = []

    # Precompute pairwise similarities (e.g., goal distance or ADE)
    goal_points = sorted_trajs[:, -1, :, :2]  # [NUM_MODES, N, 2]
    assert goal_points.shape[1] == 2

    goal_distances = []
    for i in range(goal_points.shape[1]):
        goal_distances.append((goal_points[:, i][:, None] - goal_points[:, i][None, :]).norm(dim=-1))
    goal_distances = torch.stack(goal_distances, dim=0).mean(dim=0)  # [NUM_MODES, NUM_MODES]

    while (not suppressed.all()):
        # Find the next highest-score unsuppressed mode
        active_modes = (~suppressed).nonzero(as_tuple=True)[0]
        if len(active_modes) == 0:
            break  # Edge case: no modes left (unlikely with num_ret_modes=6)
        best_idx = active_modes[0]
        keep.append(sorted_indices[best_idx])
        suppressed[best_idx] = True

        # Suppress overlapping modes
        overlapping = (goal_distances[best_idx] < threshold) & (~suppressed)
        suppressed[overlapping] = True

    print("RANK {}, Keep modes: {} out of {}.".format(global_rank, len(keep), num_modes))
    if len(keep) > num_ret_modes:
        keep = torch.stack(keep)
        # Just randomized keep
        random_indices = torch.randperm(len(keep))
        keep = keep[random_indices[:num_ret_modes]]
    else:
        for v in sorted_indices:
            if v not in keep:
                keep.append(v)
            if len(keep) == num_ret_modes:
                break
        keep = torch.stack(keep)
    assert len(keep) == num_ret_modes
    kept_preds = pred_trajs[keep[:num_ret_modes]]
    kept_scores = mode_scores[keep[:num_ret_modes]]
    return kept_preds, kept_scores, None


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 1, 3)
    map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
    if "eval/agent_position" not in data_dict:
        print("Have you set EVALUATION.PREDICT_ALL_AGENTS to False?")
        data_dict["eval/agent_position"] = data_dict["decoder/agent_position"]
        data_dict["eval/agent_heading"] = data_dict["decoder/agent_heading"]
        data_dict["eval/agent_velocity"] = data_dict["decoder/agent_velocity"]
        data_dict["eval/agent_type"] = data_dict["decoder/agent_type"]
        data_dict["eval/agent_shape"] = data_dict["decoder/agent_shape"]
        data_dict["eval/agent_valid_mask"] = data_dict["decoder/agent_valid_mask"]

    B, T, N, _ = data_dict["eval/agent_position"].shape
    map_heading = map_heading.repeat(T, axis=1).repeat(N, axis=2)
    assert map_heading.shape == (B, T, N)
    data_dict["eval/agent_position"] = rotate(
        x=data_dict["eval/agent_position"][..., 0],
        y=data_dict["eval/agent_position"][..., 1],
        angle=map_heading,
        z=data_dict["eval/agent_position"][..., 2]
    )
    assert data_dict["eval/agent_position"].ndim == 4
    data_dict["eval/agent_position"] += map_center

    data_dict["eval/agent_heading"] = wrap_to_pi(data_dict["eval/agent_heading"] + map_heading)

    data_dict["eval/agent_velocity"] = rotate(
        x=data_dict["eval/agent_velocity"][..., 0],
        y=data_dict["eval/agent_velocity"][..., 1],
        angle=map_heading,
    )

    data_dict["pred_trajs"] = [
        centralize_to_map_center(
            traj, map_center=-data_dict["expanded_map_center"][b], map_heading=-data_dict["expanded_map_heading"][b]
        ) for b, traj in enumerate(data_dict["pred_trajs"])
    ]

    return data_dict


# ===== Preprocessing to expand the all data from bs=B to bs=B*num_modes =====
def _repeat_for_modes(v, num_modes):
    if isinstance(v, list):
        return v
    d = v.ndim
    if d > 1:
        v = v[:, None]
        if isinstance(v, np.ndarray):
            shape = v.shape
            v = v.repeat(num_modes, axis=1)
            v = v.reshape(-1, *(shape[2:]))
        else:
            v = v.repeat(1, num_modes, *((1, ) * (d - 1)))
            v = v.flatten(0, 1)
    else:
        v = v.reshape(-1, 1)
        if isinstance(v, np.ndarray):
            v = v.repeat(num_modes, axis=1)
        else:
            v = v.repeat(1, num_modes)
        v = v.reshape(-1)
    return v


def generate_submission(
    prediction_trajectory_list,
    prediction_score_list,
    scenario_id_list,
    object_id_list,
    num_model_parameters,
    prefix="submission",
    account_name="peng",
    unique_method_name="peng",
    output_dir=".",
):
    submission = motion_submission_pb2.MotionChallengeSubmission(
        submission_type=motion_submission_pb2.MotionChallengeSubmission.SubmissionType.INTERACTION_PREDICTION,
        account_name=account_name,
        unique_method_name=unique_method_name,
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        num_model_parameters=num_model_parameters,
    )
    MODE_NUM = 6


    done_scenarios = set()
    duplicated_scenarios = set()

    # for prediction_trajectory, prediction_score, \
    #     ground_truth_trajectory, ground_truth_is_valid, object_type, scenario_id, object_id in \
    #         tqdm(zip(prediction_trajectory_list, prediction_score_list, ground_truth_trajectory_list,
    #                  ground_truth_is_valid_list, object_type_list, scenario_id_list, object_id_list),
    #                 total=len(prediction_trajectory_list)):
    for prediction_trajectory, prediction_score, scenario_id, object_id in \
            tqdm(
                zip(prediction_trajectory_list, prediction_score_list, scenario_id_list, object_id_list),
                total=len(prediction_trajectory_list),
                desc="Generating submission"
            ):
        scenario_id = str(scenario_id)

        if scenario_id in done_scenarios:
            duplicated_scenarios.add(scenario_id)
            continue
        done_scenarios.add(scenario_id)

        # predict_num = len(prediction_trajectory)
        predict_num = (object_id != -1).sum()
        assert (object_id[:predict_num] != -1).all()
        assert (object_id[predict_num:] == -1).all()

        scenario_prediction = submission.scenario_predictions.add()

        # NOTE: This is for single_predictions
        # prediction_set = scenario_prediction.single_predictions
        # scenario_prediction.scenario_id = str(scenario_id)
        # for i in range(predict_num):
        #     # SingleObjectPrediction
        #     prediction = prediction_set.predictions.add()
        #     prediction.object_id = object_id[i]
        #     for k in range(MODE_NUM):
        #         # ScoredTrajectory
        #         scored_trajectory = prediction.trajectories.add()
        #         scored_trajectory.confidence = float(prediction_score[i, k])
        #         trajectory = scored_trajectory.trajectory
        #         traj = prediction_trajectory[i, k, :, :]
        #         assert traj.shape[0] == 1
        #         trajectory.center_x[:] = traj[0, :, 0].numpy().tolist()
        #         trajectory.center_y[:] = traj[0, :, 1].numpy().tolist()

        # NOTE: This is for joint_predictions
        joint_pred = scenario_prediction.joint_prediction
        scenario_prediction.scenario_id = str(scenario_id)
        for k in range(MODE_NUM):
            joint_traj = joint_pred.joint_trajectories.add()

            if prediction_score.ndim == 2:
                joint_conf = sum(prediction_score[:, k])  # (2, 6) -> (6,)
            else:
                joint_conf = prediction_score[k]
            joint_traj.confidence = joint_conf
            assert predict_num == 2
            for i in range(predict_num):
                obj_traj = joint_traj.trajectories.add()
                obj_traj.object_id = object_id[i]
                trajectory = obj_traj.trajectory
                traj = prediction_trajectory[i, k, :, :]
                assert traj.shape[0] == 1
                trajectory.center_x[:] = traj[0, :, 0].numpy().tolist()
                trajectory.center_y[:] = traj[0, :, 1].numpy().tolist()

    file_name = '{}_motion_val_submission_{:%Y_%m_%d_%H_%M_%S}'.format(prefix, datetime.now())
    path = pathlib.Path(output_dir) / file_name
    path = path.resolve()
    with open(path, "wb") as f:
        f.write(submission.SerializeToString())

    os.system(f'tar -zcvf {path}.tar.gz {path}')
    print("Submission is saved at: {}.tar.gz".format(path))
    return f"{path}.tar.gz", duplicated_scenarios, done_scenarios


class WaymoMotionPredictionEvaluator:
    def __init__(self, config):
        self.config = config
        self.validation_outputs = []

        print("[Prediction] SAMPLING CONFIG IS: ", self.config.SAMPLING)

    def _call_model(self, data_dict, model):
        """We might want to create mini batches to call model in case the of OOM..."""

        use_cache = self.config.EVALUATION.USE_CACHE

        temperature = self.config.SAMPLING.TEMPERATURE
        sampling_method = self.config.SAMPLING.SAMPLING_METHOD
        topp = self.config.SAMPLING.TOPP

        # ===== Autoregressive Decoding =====
        if self.config.MODEL.NAME == "scenestreamer":
            if not hasattr(self, "scenestreamer_generator"):
                from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator
                self.scenestreamer_generator = SceneStreamerGenerator(
                    model=model,
                    device=data_dict["encoder/agent_feature"].device,
                )
            with torch.no_grad():
                self.scenestreamer_generator.reset(new_data_dict=data_dict)
                expanded_data_dict = self.scenestreamer_generator.generate_scenestreamer_motion()

        elif self.config.MODEL.NAME == "gpt":
            from scenestreamer.infer.motion import generate_motion
            with torch.no_grad():
                expanded_data_dict = generate_motion(
                    model=model,
                    data_dict=data_dict,
                    autoregressive_start_step=2,
                    # num_decode_steps=num_decode_steps,
                )

        else:
            raise NotImplementedError(f"Model {self.config.MODEL.NAME} not implemented.")

        # ===== Postprocessing to extract predictions for the modeled agents =====
        scores = expanded_data_dict["decoder/output_score"]
        pred_trajs = expanded_data_dict["decoder/reconstructed_position"]

        if self.config.eval_backward_model:
            pred_trajs = pred_trajs[:, 16:]  # For Backward evaluation
            assert pred_trajs.shape[1] == 80
        else:
            if pred_trajs.shape[1] == 96:
                pred_trajs = pred_trajs[:, :-5]
            assert pred_trajs.shape[1] == 91, pred_trajs.shape
            pred_trajs = pred_trajs[:, 11:]

        # If training to predict all agents, but asking for eval on modeled agents,
        # need to pick the prediction for the modeled agents only.
        if self.config.TRAINING.PREDICT_ALL_AGENTS:
            scores_of_interested_agents = []
            pred_trajs_of_interested_agents = []

            if self.config.EVALUATION.PREDICT_ALL_AGENTS:
                pred_ids = expanded_data_dict["decoder/agent_id"]
            else:
                pred_ids = expanded_data_dict["decoder/object_of_interest_id"]
            for batch_index, track_indices in enumerate(pred_ids):
                scores_of_interested_agents.append(
                    torch.stack(
                        [
                            scores[batch_index][agent_index]  # .detach().cpu().numpy()
                            for agent_index in track_indices if agent_index != -1
                        ],
                        dim=0
                    )
                )
                pred_trajs_of_interested_agents.append(
                    torch.stack(
                        [
                            pred_trajs[batch_index][:, agent_index]  # .detach().cpu().numpy()
                            for agent_index in track_indices if agent_index != -1
                        ],
                        dim=1
                    )
                )
        else:
            assert self.config.EVALUATION.PREDICT_ALL_AGENTS is False
            scores_of_interested_agents = []
            pred_trajs_of_interested_agents = []
            for batch_index in range(expanded_data_dict["decoder/object_of_interest_id"].shape[0]):
                num_eval_objs = (expanded_data_dict["decoder/object_of_interest_id"][batch_index] != -1).sum()
                scores_of_interested_agents.append(scores[batch_index][:num_eval_objs].detach().cpu().numpy())
                pred_trajs_of_interested_agents.append(
                    pred_trajs[batch_index][:, :num_eval_objs].detach().cpu().numpy()
                )

        return pred_trajs_of_interested_agents, scores_of_interested_agents, expanded_data_dict

    def validation_step(self, data_dict, batch_idx, model, global_rank, **kwargs):
        # TODO: Pass this from config.
        num_decode_steps = 16

        num_modes_for_eval = self.config.EVALUATION.NUM_MODES
        maximum_batch_size = self.config.EVALUATION.MAXIMUM_BATCH_SIZE

        if num_modes_for_eval <= maximum_batch_size:
            num_repeat_calls = 1
        else:
            assert num_modes_for_eval % maximum_batch_size == 0
            num_repeat_calls = num_modes_for_eval // maximum_batch_size

        NUM_MODES_WAYMO_MOTION_PREDICTION = 6

        B = data_dict["encoder/agent_feature"].shape[0]
        data_dict["batch_idx"] = torch.arange(B)

        if num_repeat_calls == 1:
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_for_eval)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("eval/") or k.startswith("decoder/") or k == "batch_idx" or k == "in_evaluation"
                    or k == "in_backward_prediction"

                    # DEBUG:
                    # or k.startswith("decoder/")
                    or k.startswith("raw")
                )
            }
            pred_trajs_of_interested_agents, scores_of_interested_agents, output_data_dict = self._call_model(
                expanded_data_dict, model
            )

        else:
            assert B == 1, B
            num_modes_per_call = num_modes_for_eval // num_repeat_calls
            assert num_modes_per_call * num_repeat_calls == num_modes_for_eval
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_per_call)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("eval/") or k.startswith("decoder/") or k == "batch_idx" or k == "in_evaluation"
                    or k == "in_backward_prediction"
                )
            }

            pred_trajs_of_interested_agents = []
            scores_of_interested_agents = []
            for call in range(num_repeat_calls):
                traj, score, output_data_dict = self._call_model(copy.deepcopy(expanded_data_dict), model)
                pred_trajs_of_interested_agents.append(traj)
                scores_of_interested_agents.append(score)
            pred_trajs_of_interested_agents = [vv for v in pred_trajs_of_interested_agents for vv in v]
            scores_of_interested_agents = [vv for v in scores_of_interested_agents for vv in v]

        pred_to_scenario_id = _repeat_for_modes(
            np.asarray(data_dict["scenario_id"]), num_modes=NUM_MODES_WAYMO_MOTION_PREDICTION
        )
        expanded_map_center = _repeat_for_modes(
            data_dict["metadata/map_center"], num_modes=NUM_MODES_WAYMO_MOTION_PREDICTION
        )
        expanded_map_heading = _repeat_for_modes(
            data_dict["metadata/map_heading"], num_modes=NUM_MODES_WAYMO_MOTION_PREDICTION
        )

        # Conduct non-maximum suppression (NMS) to reduce the number of modes
        if num_modes_for_eval > NUM_MODES_WAYMO_MOTION_PREDICTION:
            # from scenestreamer.eval.nms import batch_nms
            # pred_trajs_of_interested_agents, scores_of_interested_agents = batch_nms(
            #     pred_trajs_of_interested_agents,
            #     scores_of_interested_agents,
            #     pred_to_scenario_id=np.repeat(data_dict["scenario_id"], num_modes_for_eval, axis=0),
            #     dist_thresh=2.5,  # Follow MTR
            #     num_ret_modes=NUM_MODES_WAYMO_MOTION_PREDICTION,
            #     num_original_modes=num_modes_for_eval,
            # )


            # Assume:
            # pred_trajs = [100 modes, 80 steps, 2 agents, 2D]
            # scores = [100, 80, 2]
            pred_trajs_of_interested_agents = torch.stack(pred_trajs_of_interested_agents, 0)
            mode_scores = torch.stack(scores_of_interested_agents, 0).sum(-1)  # [NUM_MODES, 80]

            pred_trajs_of_interested_agents, scores_of_interested_agents, _ = joint_trajectory_nms(
                pred_trajs_of_interested_agents,
                mode_scores,
                global_rank=global_rank
            )
            pred_trajs_of_interested_agents = list(pred_trajs_of_interested_agents)
            scores_of_interested_agents = list(scores_of_interested_agents)

            # print(f"Kept {len(kept_preds)} predictions after NMS.")

            # sort_scores = torch.stack(scores_of_interested_agents, 0).sum(-1).sort(descending=True)
            # selected_indices = sort_scores.indices[:NUM_MODES_WAYMO_MOTION_PREDICTION]
            # selected_scores = sort_scores.values[:NUM_MODES_WAYMO_MOTION_PREDICTION]
            #
            # pred_trajs_of_interested_agents = [
            #     pred_trajs_of_interested_agents[i] for i in selected_indices
            # ]
            # scores_of_interested_agents = [
            #     scores_of_interested_agents[i] for i in selected_indices
            # ]
            # print(f"Selected scores: {selected_scores}")

        # ===== Cache the prediction results =====
        prediction_dict = {
            "pred_trajs": pred_trajs_of_interested_agents,
            "pred_scores": scores_of_interested_agents,
            "pred_to_scenario_id": pred_to_scenario_id,
            "expanded_map_center": expanded_map_center,
            "expanded_map_heading": expanded_map_heading,
        }
        for k, v in data_dict.items():
            if k.startswith("decoder/") or k.startswith("eval/") or k.startswith("metadata/") or k in ["scenario_id"]:
                prediction_dict[k] = v

        new_prediction_dict = {}
        for k, v in prediction_dict.items():
            if isinstance(v, torch.Tensor):
                new_prediction_dict[k] = v.detach().cpu().numpy()
            elif isinstance(v, list):
                new_list = []
                for vv in v:
                    if isinstance(vv, torch.Tensor):
                        new_list.append(vv.detach().cpu().numpy())
                    else:
                        new_list.append(vv)
                new_prediction_dict[k] = new_list
            else:
                new_prediction_dict[k] = v
        # prediction_dict = copy.deepcopy(new_prediction_dict)  # Avoid memory issue

        # Transform back to global coordinate
        new_prediction_dict = transform_to_global_coordinate(new_prediction_dict)
        self.validation_outputs.append(new_prediction_dict)

        # print(debug_tools.using(f"val step start {batch_idx} DONE"))

        # DEBUG:
        # waymo_evaluation_optimized(
        #     [new_prediction_dict],
        #     generate_submission=False,
        # )

    def on_validation_epoch_end(
        self, trainer, logger, global_rank, log_dict_func, log_func, print_func, exp_name, **kwargs
    ):
        """
        This function gathers intermediate evaluation result and pass them to the Waymo
        evaluation pipeline together and log the final results.
        """
        st = time.time()

        # print(debug_tools.using(f"val epoch end start"))

        # https://lightning.ai/docs/pytorch/latest/accelerators/accelerator_prepare.html?highlight=hardware
        # torch.cuda.empty_cache()
        # PZH NOTE: Hack to implement our own all_gather across ranks.
        trainer.strategy.barrier()

        # Collect the intermediate evaluation results from each call to on_validation_step in this particular rank.
        self.validation_outputs = [
            {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
             for k, v in final_pred_dicts.items()} for final_pred_dicts in self.validation_outputs
        ]

        # Dump all results in this rank to a local file so that later the rank0 process can read them.
        tmpdir = self.config.ROOT_DIR / self.config.TMP_DIR / "validation_tmpdir_{}".format(exp_name)
        print(f"Rank {global_rank} saving validation results to {tmpdir}.")

        os.makedirs(tmpdir, exist_ok=True)
        with open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(global_rank)), 'wb') as f:
            pickle.dump(self.validation_outputs, f)
        self.validation_outputs.clear()

        # print(debug_tools.using(f"val epoch saved file."))

        # If this is the main process (rank0), read all results in local filesystem and call evaluation pipeline.
        torch.cuda.empty_cache()
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            print_func(f"===== Start evaluation: {time.time() - st:.3f} =====")

            # Gather results from different ranks
            validation_list = []
            for i in range(trainer.world_size):
                file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
                success = False
                for sleep in range(10):
                    if not os.path.isfile(file):
                        time.sleep(1)
                        print(f"Can't find file: {file}. Sleep {sleep}/{10} seconds.")
                    else:
                        success = True
                        break
                if not success:
                    print(f"[WARNING] Can't find file: {file}. Skip this rank.")
                    continue
                with open(file, "rb") as f:
                    val_outputs = pickle.load(f)
                    validation_list.extend(val_outputs)
            if self.config.EVALUATION.DELETE_EVAL_RESULT:
                shutil.rmtree(tmpdir)

            if not validation_list:
                print_func("No evaluation results found. Skip evaluation.")
                return

            # print(debug_tools.using(f"going to eval"))

            # Call evaluation pipeline
            torch.cuda.empty_cache()
            result_dict, result_str, submission_dict = waymo_evaluation_optimized(
                validation_list,

                # TODO: This flag
                generate_submission=self.config.SUBMISSION.GENERATE_SUBMISSION,
                predict_all_agents=self.config.EVALUATION.PREDICT_ALL_AGENTS,
            )
            torch.cuda.empty_cache()
            validation_list.clear()

            # Log result
            result_dict = {f"eval/{k}": float(v) for k, v in result_dict.items()}
            log_dict_func(result_dict, rank_zero_only=True)
            for k in ['eval/minADE', 'eval/minFDE', 'eval/MissRate', 'eval/mAP', "eval/mJADE", "eval/avgJADE",
                      "eval/mJFDE", "eval/avgJFDE"]:
                if k not in result_dict:
                    continue
                log_func(name=k.split("/")[1], value=result_dict[k], rank_zero_only=True)
            print_func(result_str)
            print_func(f"===== Finish evaluation: {time.time() - st:.3f} =====")

        print_func(f"Rank {global_rank} finished evaluation!")
        torch.cuda.empty_cache()
        trainer.strategy.barrier()

        # TODO This flag
        if trainer.is_global_zero and self.config.SUBMISSION.GENERATE_SUBMISSION:
            account_name = self.config.SUBMISSION.ACCOUNT
            unique_method_name = self.config.SUBMISSION.METHOD_NAME
            num_model_parameters = self.config.SUBMISSION.num_model_parameters

            output_dir = logger.log_dir
            submission_prefix = logger.name

            path, duplicated_scenarios, done_scenarios = generate_submission(
                prefix=submission_prefix,
                account_name=account_name,
                unique_method_name=unique_method_name,
                output_dir=output_dir,
                num_model_parameters=num_model_parameters,
                **submission_dict
            )
            print_func(
                "Submission created at: {}. Finished {} scenarios. Duplicated scenarios: {}.".format(
                    path, len(done_scenarios), duplicated_scenarios
                )
            )
