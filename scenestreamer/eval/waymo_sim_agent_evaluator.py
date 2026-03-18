"""
Script to generate submission files for Waymo SimAgent Challenge.
Please check out the end of this file where we provide a script to merge submission files.
"""
import copy
import os
import pathlib
import uuid

import numpy as np
import torch

from scenestreamer.dataset.preprocessor import centralize_to_map_center
from scenestreamer.eval.waymo_motion_prediction_evaluator import _repeat_for_modes
from scenestreamer.eval.wosac_eval import wosac_evaluation, load_metrics_config_from_file_name
from scenestreamer.tokenization import get_tokenizer
from scenestreamer.utils import wrap_to_pi, rotate
from scenestreamer.infer.motion import generate_motion


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 1, 3)
    map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
    B, T, N, _ = data_dict["decoder/agent_position"].shape
    map_heading = map_heading.repeat(T, axis=1).repeat(N, axis=2)
    assert map_heading.shape == (B, T, N)
    data_dict["decoder/agent_position"] = rotate(
        x=data_dict["decoder/agent_position"][..., 0],
        y=data_dict["decoder/agent_position"][..., 1],
        angle=map_heading,
        z=data_dict["decoder/agent_position"][..., 2]
    )
    assert data_dict["decoder/agent_position"].ndim == 4
    data_dict["decoder/agent_position"] += map_center

    data_dict["decoder/agent_heading"] = wrap_to_pi(data_dict["decoder/agent_heading"] + map_heading)

    data_dict["decoder/agent_velocity"] = rotate(
        x=data_dict["decoder/agent_velocity"][..., 0],
        y=data_dict["decoder/agent_velocity"][..., 1],
        angle=map_heading,
    )

    data_dict["decoder/agent_position"][~data_dict["decoder/agent_valid_mask"]] = 0
    data_dict["decoder/agent_heading"][~data_dict["decoder/agent_valid_mask"]] = 0
    data_dict["decoder/agent_velocity"][~data_dict["decoder/agent_valid_mask"]] = 0

    data_dict["pred_trajs"] = [
        centralize_to_map_center(
            traj, map_center=-data_dict["expanded_map_center"][b], map_heading=-data_dict["expanded_map_heading"][b]
        ) for b, traj in enumerate(data_dict["pred_trajs"])
    ]

    return data_dict


scenario_metrics_keys = [
    # 'scenario_id',
    'metametric',
    'average_displacement_error',
    'min_average_displacement_error',
    'linear_speed_likelihood',
    'linear_acceleration_likelihood',
    'angular_speed_likelihood',
    'angular_acceleration_likelihood',
    'distance_to_nearest_object_likelihood',
    'collision_indication_likelihood',
    'time_to_collision_likelihood',
    'distance_to_road_edge_likelihood',
    'offroad_indication_likelihood'
]

aggregate_metrics_keys = [
    'realism_meta_metric', 'kinematic_metrics', 'interactive_metrics', 'map_based_metrics', 'min_ade'
]


def scenario_metrics_to_dict(scenario_metrics):
    return {k: getattr(scenario_metrics, k) for k in scenario_metrics_keys}


def aggregate_metrics_to_dict(aggregate_metrics):
    return {k: getattr(aggregate_metrics, k) for k in aggregate_metrics_keys}

def joint_trajectory_nms(
        pred_trajs,  # [NUM_MODES, 80, N, 2]
        pred_headings,  # [NUM_MODES, 80, N]
        mode_scores,  # [NUM_MODES, 80, N]
        ooisdc,
        threshold=2.5,
        num_ret_modes=32,
        global_rank=None,
):
    sorted_scores, sorted_indices = torch.sort(mode_scores, descending=True)
    sorted_trajs = pred_trajs[sorted_indices]

    num_modes = sorted_trajs.shape[0]
    suppressed = torch.zeros(num_modes, dtype=torch.bool).to(sorted_trajs.device)
    keep = []

    # Precompute pairwise similarities (e.g., goal distance or ADE)
    goal_points = sorted_trajs[:, -1, :, :2]  # [NUM_MODES, N, 2]
    # assert goal_points.shape[1] == 2

    assert ooisdc.shape[0] == 1, "ooisdc should be 1"
    ooisdc = ooisdc[0].tolist()

    goal_distances = []
    for i in ooisdc:
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
        randomized = torch.randperm(len(sorted_indices)).tolist()
        for i in randomized:
            if len(keep) == num_ret_modes:
                break
            v = sorted_indices[i]
            if v not in keep:
                keep.append(v)
        keep = torch.stack(keep)
    assert len(keep) == num_ret_modes
    kept_preds = pred_trajs[keep[:num_ret_modes]]
    kept_headings = pred_headings[keep[:num_ret_modes]]
    kept_scores = mode_scores[keep[:num_ret_modes]]
    return kept_preds, kept_headings, kept_scores, None



class WaymoSimAgentEvaluator:
    def __init__(self, config):
        self.config = config

        self.metrics = []
        self.scenario_rollouts_list = []
        self.scenario_rollouts_list_91steps = []
        self.scenario_pb_list = []
        self.shard_count = 0
        self.scenario_count = 0

        self.shard_count_91steps = 0
        self.scenario_count_91steps = 0

        self.num_scenarios_per_shard = 10

        self.scenario_generation_challenge = (self.config.EVALUATION.NAME == "sgen")

        if self.config.EVALUATION.NAME in ["wosac2024", "sgen"]:
            self.use_2024 = True
        elif self.config.EVALUATION.NAME == "wosac2023":
            self.use_2024 = False
        else:
            raise ValueError()

        print("[Sim Agent] SAMPLING CONFIG IS: ", self.config.SAMPLING)

    def _call_model(self, data_dict, model):
        """We might want to create mini batches to call model in case the of OOM..."""

        # ===== Autoregressive Decoding =====
        if model.config.MODEL.NAME == "scenestreamer":
            if not hasattr(self, "scenestreamer_generator"):
                from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator
                self.scenestreamer_generator = SceneStreamerGenerator(
                    model=model,
                    device=data_dict["decoder/agent_position"].device,
                )
            with torch.no_grad():
                self.scenestreamer_generator.reset(new_data_dict=data_dict)

                if self.scenario_generation_challenge:
                    expanded_data_dict = self.scenestreamer_generator.generate_scenestreamer_initial_state_and_motion()

                else:
                    expanded_data_dict = self.scenestreamer_generator.generate_scenestreamer_motion()

        elif model.config.MODEL.NAME == "gpt":
            from scenestreamer.infer.motion import generate_motion
            with torch.no_grad():
                expanded_data_dict = generate_motion(
                    model=model,
                    data_dict=data_dict,
                    autoregressive_start_step=2,
                    # num_decode_steps=num_decode_steps,
                )

        # ===== Postprocessing to extract predictions for the modeled agents =====
        scores = expanded_data_dict["decoder/output_score"]
        pred_trajs = expanded_data_dict["decoder/reconstructed_position"]
        pred_heading = expanded_data_dict["decoder/reconstructed_heading"]

        if "decoder/reconstructed_shape" in expanded_data_dict:
            pred_shapes = expanded_data_dict["decoder/reconstructed_shape"]
        else:
            pred_shapes = None

        # If training to predict all agents, but asking for eval on modeled agents,
        # need to pick the prediction for the modeled agents only.
        assert self.config.TRAINING.PREDICT_ALL_AGENTS
        assert self.config.EVALUATION.PREDICT_ALL_AGENTS

        return pred_trajs, pred_heading, scores, expanded_data_dict, pred_shapes

    def validation_step(self, data_dict, batch_idx, model, log_dict_func, global_rank, logger, **kwargs):

        save_91steps_together = self.scenario_generation_challenge
        save_80steps_together = not self.scenario_generation_challenge

        disable_eval = True

        num_modes_for_eval = self.config.EVALUATION.NUM_MODES
        maximum_batch_size = self.config.EVALUATION.MAXIMUM_BATCH_SIZE

        if num_modes_for_eval <= maximum_batch_size:
            num_repeat_calls = 1
        else:
            assert num_modes_for_eval % maximum_batch_size == 0
            num_repeat_calls = num_modes_for_eval // maximum_batch_size

        NUM_MODES_WAYMO_SIM_AGENTS = 32

        B = data_dict["encoder/agent_feature"].shape[0]
        data_dict["batch_idx"] = torch.arange(B)

        # DEBUG:
        # print("RAW SCENARIO DESCPTION: (BEFORE) ", data_dict["raw_scenario_description"][0]['id'])

        if num_repeat_calls == 1:
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_for_eval)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("decoder/") or k in ["batch_idx", "in_evaluation", "scenario_id"]
                )
            }
            pred_trajs_of_interested_agents, pred_heading_of_interested_agents, scores_of_interested_agents, output_data_dict, pred_shapes = self._call_model(
                expanded_data_dict, model
            )

        else:
            assert B == 1
            num_modes_per_call = num_modes_for_eval // num_repeat_calls
            assert num_modes_per_call * num_repeat_calls == num_modes_for_eval
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_per_call)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("decoder/")
                    or k in ["batch_idx", "in_evaluation", "scenario_id", "in_backward_prediction"]
                )
            }

            pred_trajs_of_interested_agents = []
            scores_of_interested_agents = []
            pred_heading_of_interested_agents = []
            pred_shapes = []
            for call in range(num_repeat_calls):
                traj, head, score, output_data_dict, pred_s = self._call_model(copy.deepcopy(expanded_data_dict), model)
                pred_trajs_of_interested_agents.append(traj)
                scores_of_interested_agents.append(score)
                pred_heading_of_interested_agents.append(head)
                pred_shapes.append(pred_s)
            pred_trajs_of_interested_agents = [vv for v in pred_trajs_of_interested_agents for vv in v]
            scores_of_interested_agents = [vv for v in scores_of_interested_agents for vv in v]
            pred_heading_of_interested_agents = [vv for v in pred_heading_of_interested_agents for vv in v]
            if pred_shapes[0] is not None:
                pred_shapes = [vv for v in pred_shapes for vv in v]
            else:
                pred_shapes = None

        # ===== Postprocessing to extract predictions for the modeled agents =====
        if num_modes_for_eval > NUM_MODES_WAYMO_SIM_AGENTS:
            if self.scenario_generation_challenge:
                pred_trajs_of_interested_agents = pred_trajs_of_interested_agents[:NUM_MODES_WAYMO_SIM_AGENTS]
                pred_heading_of_interested_agents = pred_heading_of_interested_agents[:NUM_MODES_WAYMO_SIM_AGENTS]
                scores_of_interested_agents = scores_of_interested_agents[:NUM_MODES_WAYMO_SIM_AGENTS]
                pred_shapes = pred_shapes[:NUM_MODES_WAYMO_SIM_AGENTS]

            else:
                pred_trajs_of_interested_agents = torch.stack(pred_trajs_of_interested_agents, 0)
                pred_heading_of_interested_agents = torch.stack(pred_heading_of_interested_agents, 0)
                scores_of_interested_agents = torch.stack(scores_of_interested_agents, 0).sum(-1)
                ooisdc = data_dict["decoder/labeled_agent_id"].clone()
                pred_trajs_of_interested_agents, pred_heading_of_interested_agents, scores_of_interested_agents, _ = \
                    joint_trajectory_nms(
                        pred_trajs_of_interested_agents,
                        pred_heading_of_interested_agents,
                        scores_of_interested_agents,
                        num_ret_modes=NUM_MODES_WAYMO_SIM_AGENTS,
                        ooisdc=ooisdc,
                        global_rank=global_rank,
                    )
                pred_trajs_of_interested_agents = list(pred_trajs_of_interested_agents)
                pred_heading_of_interested_agents = list(pred_heading_of_interested_agents)
                scores_of_interested_agents = list(scores_of_interested_agents)

        pred_to_scenario_id = _repeat_for_modes(data_dict["scenario_id"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS)
        expanded_map_center = _repeat_for_modes(data_dict["metadata/map_center"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS)
        expanded_map_heading = _repeat_for_modes(
            data_dict["metadata/map_heading"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS
        )

        # ===== Cache the prediction results =====
        prediction_dict = {
            "pred_trajs": pred_trajs_of_interested_agents,
            "pred_headings": pred_heading_of_interested_agents,
            "pred_scores": scores_of_interested_agents,

            "pred_shape": pred_shapes,

            "pred_to_scenario_id": pred_to_scenario_id,
            "expanded_map_center": expanded_map_center,
            "expanded_map_heading": expanded_map_heading,
        }
        for k, v in data_dict.items():
            if k.startswith("decoder/") or k.startswith("decoder/") or k.startswith("metadata/") or k in [
                    "raw_scenario_description", "scenario_id"
            ]:
                prediction_dict[k] = v

        new_prediction_dict = {}
        for k, v in prediction_dict.items():
            if isinstance(v, torch.Tensor):
                new_prediction_dict[k] = v.detach().cpu().numpy()
            elif isinstance(v, list):
                new_prediction_dict[k] = [vv.detach().cpu().numpy() if isinstance(vv, torch.Tensor) else v for vv in v]
            else:
                new_prediction_dict[k] = v
        # prediction_dict = copy.deepcopy(new_prediction_dict)  # Avoid memory issue

        # Transform back to global coordinate
        new_prediction_dict = transform_to_global_coordinate(new_prediction_dict)
        # self.validation_outputs.append(new_prediction_dict)
        scenario_metrics, aggregate_metrics, scenario_rollouts_list_80steps, scenario_rollouts_list_91steps, scenario_pb_list = wosac_evaluation(
            [new_prediction_dict], disable_eval=disable_eval, use_2024=self.use_2024,
            save_91steps_together=save_91steps_together,
            save_80steps_together=save_80steps_together,
        )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        # from scenestreamer.gradio_ui.plot import _plot_map
        # from scenestreamer.utils import utils
        # np_d = utils.torch_to_numpy(data_dict)
        # np_d = {k: v[0] for k, v in np_d.items()}
        # _plot_map(np_d, ax=plt.gca())
        # AID = 2
        # for mode in pred_trajs_of_interested_agents:
        #     mode = utils.torch_to_numpy(mode)
        #     plt.plot(mode[:, AID, 0], mode[:, AID, 1])
        # plt.show()

        # TODO: Some assertions here to avoid WOSAC error ......
        # https://github.com/waymo-research/waymo-open-dataset/issues/807
        # Scenario 891805f154b4f0dd: Sim agents {1178} are missing from the simulation.
        # Scenario de8c427e65487b93: Sim agents {432, 569, 554} are missing from the simulation.
        # Scenario 386f0b2faebe74af: Sim agents {3361, 3332, 3399, 3371, 3375, 3311, 3345, 3346, 3378, 3319, 3352, 3388} are missing from the simulation.
        # Scenario cd861218ceb2dc1e: Sim agents {2145, 4805, 2150, 2123, 2162, 2131, 4730, 2139} are missing from the simulation.
        # Scenario 46a12cf2da1fdda8: Sim agents {1540, 1544, 4507, 4514, 4545, 1608, 1616, 1623, 1626, 1627, 1500, 1630, 1631, 1632, 1634, 4450, 1636, 4599, 1638, 1639, 4455, 4583, 4585, 4586, 1515, 4461, 1649, 4469, 4597, 4598, 4477} are missing from the simulation.
        watching = {
            "891805f154b4f0dd": [1178],
            "de8c427e65487b93": [432, 569, 554],
            "386f0b2faebe74af": [3361, 3332, 3399, 3371, 3375, 3311, 3345, 3346, 3378, 3319, 3352, 3388],
            "cd861218ceb2dc1e": [2145, 4805, 2150, 2123, 2162, 2131, 4730, 2139],
            "46a12cf2da1fdda8": [
                1540, 1544, 4507, 4514, 4545, 1608, 1616, 1623, 1626, 1627, 1500, 1630, 1631, 1632, 1634, 4450, 1636,
                4599, 1638, 1639, 4455, 4583, 4585, 4586, 1515, 4461, 1649, 4469, 4597, 4598, 4477
            ],
        }
        for r in scenario_rollouts_list_80steps:
            sid = r.scenario_id
            if sid in watching:
                obj_ids = {j.object_id for j in r.joint_scenes[10].simulated_trajectories}
                for oid in watching[sid]:
                    assert oid in obj_ids
        # # TODO: Some assertions here to avoid WOSAC error ......

        if len(scenario_rollouts_list_80steps) > 0:
            assert data_dict["raw_scenario_description"][0]['id'] == scenario_rollouts_list_80steps[0].scenario_id
        if len(scenario_rollouts_list_91steps) > 0:
            assert data_dict["raw_scenario_description"][0]['id'] == scenario_rollouts_list_91steps[0].scenario_id

        if not disable_eval:
            scenario_id = list(scenario_metrics.keys())

            scenario_metrics = {k: scenario_metrics_to_dict(scenario_metrics[k]) for k in scenario_metrics}
            aggregate_metrics = {k: aggregate_metrics_to_dict(aggregate_metrics[k]) for k in aggregate_metrics}

            stat = {}
            for k in scenario_metrics_keys:
                stat[f"scenario_metrics/{k}"] = np.mean([d[k] for d in scenario_metrics.values()])
            for k in aggregate_metrics_keys:
                stat[f"aggregate_metrics/{k}"] = np.mean([d[k] for d in aggregate_metrics.values()])

            log_dict_func(
                stat,
                batch_size=data_dict["encoder/agent_feature"].shape[0],
                on_epoch=True,
                prog_bar=True,
            )

            self.metrics.append(stat)

            print(
                "\n=============== RANK {} FINISHED {} SCENARIOS =============".format(global_rank, len(self.metrics))
            )
            print("Latest scenario ID: ", scenario_id)
            for k in self.metrics[0].keys():
                print(f"{k}: {np.mean([m[k] for m in self.metrics]):.4f}")
            print("===========================================================".format(len(self.metrics)))

        if save_80steps_together:
            self.scenario_rollouts_list.extend(scenario_rollouts_list_80steps)
            self.scenario_pb_list.extend(scenario_pb_list)
            if len(self.scenario_rollouts_list) >= self.num_scenarios_per_shard:
                output_dir = pathlib.Path(logger.log_dir) / "80steps"
                self.generate_submission_shard(output_dir, global_rank)

        if save_91steps_together:
            self.scenario_rollouts_list_91steps.extend(scenario_rollouts_list_91steps)
            if len(self.scenario_rollouts_list_91steps) >= self.num_scenarios_per_shard:
                output_dir = pathlib.Path(logger.log_dir) / "91steps"
                self.generate_submission_shard_91steps(output_dir, global_rank)

    def on_validation_epoch_end(self, *args, global_rank, logger, trainer, **kwargs):
        if self.metrics:
            print("======== FINAL RESULT RANK {} WITH {} SCENARIOS ==========".format(global_rank, len(self.metrics)))
            for k in self.metrics[0].keys():
                print(f"{k}: {np.mean([m[k] for m in self.metrics]):.4f}")
            print("===========================================================".format(len(self.metrics)))

        output_dir = pathlib.Path(logger.log_dir) / "80steps"
        print(
            f"RANK {global_rank} Storing the final submission files with {len(self.scenario_rollouts_list)} rollouts..."
        )
        import time
        sleep = np.random.randint(1, 5)
        print(f"RANK {global_rank} sleep {sleep} seconds.")
        time.sleep(sleep)

        if not self.scenario_generation_challenge:
            self.generate_submission_shard(output_dir, global_rank)

        if self.scenario_generation_challenge:
            output_dir = pathlib.Path(logger.log_dir) / "91steps"
            self.generate_submission_shard_91steps(output_dir, global_rank)

        print(f"RANK {global_rank} finished. Entering barrier...")
        # trainer.strategy.barrier()
        print(f"RANK {global_rank} left barrier...")
        # if global_rank == 0:
        print("RANK {} Generated {} shards total.".format(global_rank, self.shard_count))
        print("RANK {} Generated {} scenarios total.".format(global_rank, self.scenario_count))
        print("RANK {} ========== Please manually merge the submission files!!! ==========".format(global_rank))
        output_dir = pathlib.Path(output_dir).resolve()
        print("===============================================================================================\n")
        print("RANK {} Shard submission is saved at: {}".format(global_rank, output_dir))
        print("\n===============================================================================================")
        print("RANK {} Exit.".format(global_rank))

    def generate_submission_shard(self, output_dir, this_rank):
        from waymo_open_dataset.protos import sim_agents_submission_pb2
        account_name = self.config.SUBMISSION.ACCOUNT
        unique_method_name = self.config.SUBMISSION.METHOD_NAME
        num_model_parameters = self.config.SUBMISSION.num_model_parameters
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.scenario_rollouts_list,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=account_name,
            unique_method_name=unique_method_name,

            authors=['scenestreamer_authors'],

            # New fields, need changed.
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters=num_model_parameters,
            acknowledge_complies_with_closed_loop_requirement=True
        )

        # output_filename = f'submission.binproto-{global_rank:05d}-of-{total_ranks:05d}'
        output_filename = f'submission.binproto-tmp{uuid.uuid4()}'

        scenario_id_list = [s.scenario_id for s in self.scenario_rollouts_list]
        print("Scenario ID to be saved in shard: ", scenario_id_list, output_filename)

        output_dir = pathlib.Path(output_dir).absolute()

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = pathlib.Path(output_dir) / output_filename
        file_path = file_path.resolve()
        with open(file_path, 'wb') as f:
            f.write(shard_submission.SerializeToString())

        if self.config.SUBMISSION.SAVE_EVAL_DATA and (not self.config.SUBMISSION.GENERATE_SUBMISSION):
            for s in self.scenario_pb_list:
                print("Scenario ID to be saved together apart from shard: ", s.scenario_id)
                file_path = pathlib.Path(output_dir) / "scenario_pb"
                file_path.mkdir(parents=True, exist_ok=True)
                file_path = file_path / f"{s.scenario_id}.binproto"
                file_path = file_path.resolve()
                with open(file_path, 'wb') as f:
                    f.write(s.SerializeToString())

        print("=====================================================================================================\n")
        print("RANK {} Shard submission is saved at: {}".format(this_rank, file_path))
        print("To generate final submission, please manually run:")
        print("\npython -m scenestreamer.merge_shards --output_dir={}".format(output_dir))
        print("\n\nTo see evaluation results, please manually run: (please make sure SUBMISSION.SAVE_EVAL_DATA=True)")
        print("\npython -m scenestreamer.wosac_eval_async --output_dir={}".format(output_dir))
        print("\npython -m scenestreamer.wosac_eval --output_dir={}".format(output_dir))
        print("\n=====================================================================================================")
        # self.output_filenames.append(output_filename)
        self.scenario_rollouts_list = []
        self.scenario_pb_list = []
        self.shard_count += 1
        self.scenario_count += len(scenario_id_list)
    def generate_submission_shard_91steps(self, output_dir, this_rank):
        from waymo_open_dataset.protos import sim_agents_submission_pb2
        account_name = self.config.SUBMISSION.ACCOUNT
        unique_method_name = self.config.SUBMISSION.METHOD_NAME
        num_model_parameters = self.config.SUBMISSION.num_model_parameters
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.scenario_rollouts_list_91steps,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=account_name,
            unique_method_name=unique_method_name,

            # New fields, need changed.
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters=num_model_parameters,
            acknowledge_complies_with_closed_loop_requirement=True
        )

        # output_filename = f'submission.binproto-{global_rank:05d}-of-{total_ranks:05d}'
        output_filename = f'submission.binproto-tmp{uuid.uuid4()}'

        scenario_id_list = [s.scenario_id for s in self.scenario_rollouts_list_91steps]
        print("Scenario ID to be saved in shard: ", scenario_id_list, output_filename)

        output_dir = pathlib.Path(output_dir).absolute()

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = pathlib.Path(output_dir) / output_filename
        file_path = file_path.resolve()
        with open(file_path, 'wb') as f:
            f.write(shard_submission.SerializeToString())

        if self.config.SUBMISSION.SAVE_EVAL_DATA and (not self.config.SUBMISSION.GENERATE_SUBMISSION):
            raise ValueError

        print("=====================================================================================================\n")
        print("RANK {} Shard submission is saved at: {}".format(this_rank, file_path))
        print("To generate final submission, please manually run:")
        print("\npython -m scenestreamer.merge_shards --output_dir={}".format(output_dir))
        print("\n\nTo see evaluation results, please manually run: (please make sure SUBMISSION.SAVE_EVAL_DATA=True)")
        print("\npython -m scenestreamer.wosac_eval_async --output_dir={}".format(output_dir))
        print("\npython -m scenestreamer.wosac_eval --output_dir={}".format(output_dir))
        print("\n=====================================================================================================")
        # self.output_filenames.append(output_filename)
        self.scenario_rollouts_list_91steps = []
        assert not self.scenario_pb_list
        self.scenario_pb_list = []
        self.shard_count_91steps += 1
        self.scenario_count_91steps += len(scenario_id_list)
