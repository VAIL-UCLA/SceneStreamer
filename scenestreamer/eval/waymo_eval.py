"""
This file provides some utility functions for preparing waymo submission, and evaluate on waymo validation dataset.

To install:

conda install python=3.10
pip install waymo-open-dataset-tf-2-12-0==1.6.4

https://github.com/waymo-research/waymo-open-dataset.git
"""
from collections import defaultdict

import numpy as np

try:
    import tensorflow as tf
    from google.protobuf import text_format

    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    if all_gpus:
        try:
            for cur_gpu in all_gpus:
                tf.config.experimental.set_memory_growth(cur_gpu, True)
        except RuntimeError as e:
            print(e)

    # from trafficgen_v2.datasets.waymo.mtr_metadrive_dataset import get_type_string
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py as config_util
    from waymo_open_dataset.protos import motion_metrics_pb2
    from scenestreamer.dataset.constants import object_int_to_type
except ModuleNotFoundError:
    pass

import logging

logger = logging.getLogger(__file__)
object_type_to_id = {'TYPE_UNSET': 0, 'TYPE_VEHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_CYCLIST': 3, 'TYPE_OTHER': 4}


def _default_metrics_config(eval_second, num_modes_for_eval=6):
    assert eval_second in [3, 5, 8]
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
    }
    """
    config_text += f"""
    max_predictions: {num_modes_for_eval}
    """
    if eval_second == 3:
        config_text += """
        track_future_samples: 30
        """
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config


# def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8):
#     print(f'Total number for evaluation (intput): {len(pred_dicts)}')
#     temp_pred_dicts = []
#     for k in range(len(pred_dicts)):
#         if isinstance(pred_dicts[k], list):
#             temp_pred_dicts.extend(pred_dicts[k])
#         else:
#             temp_pred_dicts.append(pred_dicts[k])
#     pred_dicts = temp_pred_dicts
#     print(f'Total number for evaluation (after processed): {len(pred_dicts)}')
#
#     scene2preds = {}
#     num_max_objs_per_scene = 0
#     for k in range(len(pred_dicts)):
#         cur_scenario_id_list = pred_dicts[k]["scenario_id"]
#
#         for batch_index, cur_scenario_id in enumerate(cur_scenario_id_list):
#
#             if cur_scenario_id not in scene2preds:
#                 scene2preds[cur_scenario_id] = []
#
#             # PZH NOTE: A little workaround here to deal with the name mismatch
#             # pred_dicts[k]["object_type"] = get_type_string(pred_dicts[k][batch_index]["object_type"])
#
#             per_scenario_pred_dicts = {k: v[batch_index] for k, v in pred_dicts[k].items()}
#
#             scene2preds[cur_scenario_id].append(per_scenario_pred_dicts)
#             # num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
#
#     num_scenario = len(scene2preds)
#
#     # try:
#     topK, num_future_frames, _ = per_scenario_pred_dicts["pred_trajs"].shape
#     # except ValueError as e:
#     #     print(pred_dicts[0]['pred_trajs'].shape)
#     #     raise e
#
#     if top_k_for_eval != -1:
#         topK = min(top_k_for_eval, topK)
#
#     if num_future_frames in [30, 50, 80]:
#         sampled_interval = 5
#     assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
#     num_frame_to_eval = num_future_frames // sampled_interval
#
#     if eval_second == 3:
#         num_frames_in_total = 41
#         num_frame_to_eval = 6
#     elif eval_second == 5:
#         num_frames_in_total = 61
#         num_frame_to_eval = 10
#     else:
#         num_frames_in_total = 91
#         num_frame_to_eval = 16
#
#     batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
#     batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
#     gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
#     gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=int)
#     pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
#     pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=int)
#     object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=object)
#     object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=int)
#     scenario_id = np.zeros((num_scenario), dtype=object)
#
#     object_type_cnt_dict = {}
#     for key in object_type_to_id.keys():
#         object_type_cnt_dict[key] = 0
#
#     for scene_idx, val in enumerate(scene2preds.items()):
#         cur_scenario_id, preds_per_scene = val
#         scenario_id[scene_idx] = cur_scenario_id
#         for obj_idx, cur_pred in enumerate(preds_per_scene):
#             sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
#             cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
#             cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]
#
#             cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()
#
#             batch_pred_trajs[scene_idx,
#                              obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis,
#                                                                4::sampled_interval, :][:, :, :num_frame_to_eval, :]
#             batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
#             gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, [
#                 0, 1, 3, 4, 6, 7, 8
#             ]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
#             gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, -1]
#             pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
#             pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
#             object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
#             object_id[scene_idx, obj_idx] = cur_pred['object_id']
#
#             object_type_cnt_dict[cur_pred['object_type']] += 1
#
#     gt_infos = {
#         'scenario_id': scenario_id.tolist(),
#         'object_id': object_id.tolist(),
#         'object_type': object_type.tolist(),
#         'gt_is_valid': gt_is_valid,
#         'gt_trajectory': gt_trajs,
#         'pred_gt_indices': pred_gt_idxs,
#         'pred_gt_indices_mask': pred_gt_idx_valid_mask
#     }
#     return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict


def waymo_evaluation_optimized(
    pred_dicts, eval_second=8, num_modes_for_eval=6, verbose=True, generate_submission=False, predict_all_agents=False
):
    # Split all data based on scenario
    split_data = defaultdict(list)

    scenario_id_list = []
    # Split the prediction for each scenario, also flatten the data
    for d in pred_dicts:
        for sid in np.unique(d["pred_to_scenario_id"]):
            for k, v in d.items():
                if k in ["pred_trajs", "pred_scores"]:
                    assert len(d["pred_to_scenario_id"]) == len(v), (len(d["pred_to_scenario_id"]), len(v), k)
                    entry_in_same_scenario = [v[idx] for idx in range(len(v)) if d["pred_to_scenario_id"][idx] == sid]
                    assert entry_in_same_scenario
                    entry_in_same_scenario = np.stack(entry_in_same_scenario, axis=0)
                    split_data[k].append(entry_in_same_scenario)
                elif k in ["eval/agent_position", "eval/agent_velocity", "eval/agent_heading", "eval/agent_valid_mask",
                           "eval/agent_shape", "eval/agent_type", "decoder/object_of_interest_id",
                           "encoder/object_of_interest_name", "decoder/object_of_interest_name", "decoder/track_name"]:
                    entry_in_same_scenario = [v[idx] for idx in range(len(v)) if d["scenario_id"][idx] == sid]
                    assert entry_in_same_scenario
                    assert len(entry_in_same_scenario) == 1
                    entry_in_same_scenario = entry_in_same_scenario[0]
                    split_data[k].append(entry_in_same_scenario)
            scenario_id_list.append(sid)
    split_data = dict(split_data)

    num_scenario = len(split_data["pred_trajs"])

    trajectory_in_single_scenario = split_data["pred_trajs"][0]
    num_modes = len(trajectory_in_single_scenario)
    num_future_frames = trajectory_in_single_scenario[0].shape[0]
    num_max_objs_per_scene = max([v[0].shape[1] for v in split_data["pred_trajs"]])

    if num_future_frames in [30, 50, 80]:
        sampled_interval = 5
    else:
        raise ValueError("Unknown prediction with future steps: {}".format(num_future_frames))

    if eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6
    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10
    else:
        num_frames_in_total = 91
        num_frame_to_eval = 16

    # ===== Process each scenario's prediction =====
    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_modes, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, num_modes))
    pred_gt_indices = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=int)
    pred_gt_indices_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=int)
    for scenario_count in range(num_scenario):

        if predict_all_agents:
            num_objs = (split_data["eval/agent_type"][scenario_count] >= 0).sum()
        else:
            num_objs = (split_data["decoder/object_of_interest_id"][scenario_count] >= 0).sum()

        pred_trajs = split_data["pred_trajs"][scenario_count]
        pred_trajs = pred_trajs[:, :, :num_objs]
        assert pred_trajs.shape == (num_modes_for_eval, num_future_frames, num_objs, 2)

        # prev: scores in shape [num objects, num modes]
        scores = split_data["pred_scores"][scenario_count]  # (#modes, N)
        # assert scores.ndim == 2, scores.shape
        # scores = scores[:, :num_objs]

        # assert scores.min() >= 0.0

        # top_k_index = np.argsort(scores, axis=0)[::-1][:num_modes_for_eval]  # (#modes, N)
        # top_k_index = top_k_index[:, :num_objs]
        # assert top_k_index.shape == (num_modes_for_eval, num_objs)
        # top_k_scores = np.take_along_axis(scores, top_k_index, axis=0)  # (#modes, N)

        # pred_trajs = np.take_along_axis(pred_trajs, top_k_index.reshape(num_modes_for_eval, 1, num_objs, 1), axis=0)
        pred_trajs_processed = pred_trajs[:, (sampled_interval - 1)::sampled_interval]
        pred_trajs_processed = pred_trajs_processed[:, :num_frame_to_eval, :num_objs]

        # Till now, pred_trajs_processed.shape == (#modes, #steps, N, 2), need to change shape to
        # (N, #modes, 1, #steps, 2)
        assert pred_trajs_processed.ndim == 4
        pred_trajs_processed = pred_trajs_processed.swapaxes(0, 2)  # (#modes, #steps, N, 2) -> (N, #steps, #modes, 2)
        pred_trajs_processed = pred_trajs_processed.swapaxes(1, 2)  # -> (N, #modes, #steps, 2)
        pred_trajs_processed = pred_trajs_processed.reshape(num_objs, num_modes_for_eval, 1, num_frame_to_eval, 2)

        # batch_pred_trajs.shape == (#scenarios, #maxobjs, #modes, 1, #steps, 2)
        batch_pred_trajs[scenario_count, :num_objs] = pred_trajs_processed

        # scores in shape [num objects, num modes] = [7, 6]
        # normalize the scores for all modes of one object in this scene.
        # assert top_k_scores.ndim == 2
        # top_k_scores = top_k_scores.swapaxes(0, 1)  # (#modes, N) -> (N, #modes)
        # top_k_scores = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
            batch_pred_scores[scenario_count, :num_objs] = scores
        else:
            batch_pred_scores[scenario_count, :num_objs] = scores.swapaxes(0, 1)

        pred_gt_indices[scenario_count, :num_objs, 0] = np.arange(num_objs, dtype=int)
        pred_gt_indices_mask[scenario_count, :num_objs, 0] = 1

    # ===== Process GT data directly =====
    in_testing_set = split_data["eval/agent_position"][0].shape[0] == 11

    if in_testing_set:
        if generate_submission:
            object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=int)
            object_id.fill(-1)
            for scenario_count in range(num_scenario):
                if predict_all_agents:
                    num_objs = (split_data["eval/agent_type"][scenario_count] >= 0).sum()
                else:
                    num_objs = (split_data["decoder/object_of_interest_id"][scenario_count] >= 0).sum()

                assert (split_data["eval/agent_type"][scenario_count][num_objs:] == -1).all()
                if not predict_all_agents:
                    object_id[
                    scenario_count, :num_objs] = split_data["decoder/object_of_interest_name"][scenario_count][
                                                 :num_objs]
                else:
                    object_id[scenario_count, :num_objs] = split_data["decoder/track_name"][scenario_count][:num_objs]
            batch_pred_trajs = tf.convert_to_tensor(batch_pred_trajs, tf.float32)
            batch_pred_scores = tf.convert_to_tensor(batch_pred_scores, tf.float32)
            submission_data = dict(
                prediction_trajectory_list=batch_pred_trajs,
                prediction_score_list=batch_pred_scores,
                object_id_list=object_id,
                scenario_id_list=scenario_id_list,
            )
        else:
            submission_data = dict()
        return {}, "", submission_data


    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=int)
    object_id.fill(-1)

    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=int)
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=int)
    for scenario_count in range(num_scenario):

        if predict_all_agents:
            num_objs = (split_data["eval/agent_type"][scenario_count] >= 0).sum()
        else:
            num_objs = (split_data["decoder/object_of_interest_id"][scenario_count] >= 0).sum()

        assert (split_data["eval/agent_type"][scenario_count][num_objs:] == -1).all()

        object_type[scenario_count, :num_objs] = split_data["eval/agent_type"][scenario_count][:num_objs]

        if not predict_all_agents:
            object_id[
                scenario_count, :num_objs] = split_data["decoder/object_of_interest_name"][scenario_count][:num_objs]
        else:
            object_id[scenario_count, :num_objs] = split_data["decoder/track_name"][scenario_count][:num_objs]

        heading_in_0_2pi = (split_data["eval/agent_heading"][scenario_count][..., None]) % (2 * np.pi)

        gt_trajs_per_scenario = np.concatenate(
            [
                split_data["eval/agent_position"][scenario_count][..., :2],
                split_data["eval/agent_shape"][scenario_count][..., :2],
                heading_in_0_2pi,
                split_data["eval/agent_velocity"][scenario_count],
            ],
            axis=-1
        )
        gt_trajs_per_scenario = gt_trajs_per_scenario.swapaxes(0, 1)
        gt_trajs[scenario_count, :num_objs] = gt_trajs_per_scenario[:num_objs]

        gt_is_valid_per_scenario = split_data["eval/agent_valid_mask"][scenario_count]
        gt_is_valid_per_scenario = gt_is_valid_per_scenario.swapaxes(0, 1)
        gt_is_valid[scenario_count, :num_objs] = gt_is_valid_per_scenario[:num_objs]

    eval_config = _default_metrics_config(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval)

    # DEBUG:
    pred = batch_pred_trajs[:, :, :, 0]  # (B, N, M, 16, 2)
    gt = gt_trajs[:, :, :, :2]  # (B, N, 91, 2)
    vl = gt_is_valid[:, :, 15::5]
    gt = gt[:, :, 15::5]
    gt = gt.reshape(gt.shape[0], gt.shape[1], 1, -1, 2).repeat(num_modes_for_eval, axis=2)
    vl = vl.reshape(vl.shape[0], vl.shape[1], 1, -1).repeat(num_modes_for_eval, axis=2)
    # gt shape = (B, N, M, 16, 2)
    diff = (np.linalg.norm(pred - gt, axis=-1)) * vl

    # Avg over time dim
    valid = vl.sum(-1)
    error = diff.sum(-1) / np.maximum(valid, 1)

    last_valid_ind = (vl != 0).cumsum(axis=-1).argmax(axis=-1)
    fde = np.take_along_axis(diff, last_valid_ind[..., None], axis=-1).squeeze(-1)

    # Avg over agent dim
    valid = (valid > 0).sum(1)  # Num valid agents
    error = error.sum(1) / np.maximum(valid, 1)
    fde = fde.sum(1) / np.maximum(valid, 1)
    # Now error.shape = (B, M=6)
    avg_error = error.mean(-1).mean(0)
    min_error = error.min(-1).mean(0)

    avg_fde = fde.mean(-1).mean(0)
    min_fde = fde.min(-1).mean(0)

    batch_pred_scores = tf.convert_to_tensor(batch_pred_scores, tf.float32)
    batch_pred_trajs = tf.convert_to_tensor(batch_pred_trajs, tf.float32)
    gt_trajs = tf.convert_to_tensor(gt_trajs, tf.float32)
    gt_is_valid = tf.convert_to_tensor(gt_is_valid, bool)
    pred_gt_indices = tf.convert_to_tensor(pred_gt_indices, tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(pred_gt_indices_mask, bool)
    object_type = tf.convert_to_tensor(object_type, tf.int64)

    input_dict = dict(
        prediction_trajectory=batch_pred_trajs,
        # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, 2)
        prediction_score=batch_pred_scores,  # (batch_size, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,
        # (batch_size, num_pred_groups, num_agents_per_group)
        object_type=object_type  # (batch_size, num_total_agents)
    )

    metric_results = py_metrics_ops.motion_metrics(config=eval_config.SerializeToString(), **input_dict)

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'mAP', 'OverlapRate']):
        avg_results.update({f'{m}-VEHICLE': [0.0, 0], f'{m}-PEDESTRIAN': [0.0, 0], f'{m}-CYCLIST': [0.0, 0]})
        for j, n in enumerate(metric_names):
            cur_name = n.split('_')[1]
            avg_results[f'{m}-{cur_name}'][0] += float(metric_results[i][j])
            avg_results[f'{m}-{cur_name}'][1] += 1
            result_dict[f'{m}-{n}'] = float(metric_results[i][j])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / avg_results[key][1]

    if verbose:
        result_dict['-------------------------------------------------------------'] = 0

    result_dict.update(avg_results)

    object_type_cnt_dict = {k: 0 for k in object_int_to_type.values()}
    for type_int_list in split_data["eval/agent_type"]:
        for type_int in np.unique(type_int_list):
            object_type_cnt_dict[object_int_to_type[type_int]] += (type_int_list == type_int).sum()
    result_dict.update(object_type_cnt_dict)

    final_avg_results = {}
    result_format_list = [
        [
            'Waymo', 'Count', 'mAP', 'minADE', 'minFDE', 'MissRate', 'OverlapR', 'mJADE', 'avgJADE', 'mJFDE', 'avgJFDE',
            '\n'
        ],
        ['VEH', None, None, None, None, None, None, None, None, None, None, '\n'],
        ['PED', None, None, None, None, None, None, None, None, None, None, '\n'],
        ['CYC', None, None, None, None, None, None, None, None, None, None, '\n'],
        ['Avg', None, None, None, None, None, None, None, None, None, None, '\n'],
    ]
    name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
    name_to_col = {
        'Count': 1,
        'mAP': 2,
        'minADE': 3,
        'minFDE': 4,
        'MissRate': 5,
        'OverlapRate': 6,
        'mJADE': 7,
        'avgJADE': 8,
        'mJFDE': 9,
        'avgJFDE': 10,
    }

    for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP', 'OverlapRate']:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name}-{cur_name}']

            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = \
                '%.4f,' % avg_results[f'{cur_metric_name}-{cur_name}']

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

    for object_type in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        result_format_list[name_to_row[object_type]][name_to_col["Count"]] = str(object_type_cnt_dict[object_type])
    object_count_sum = (
        object_type_cnt_dict["VEHICLE"] + object_type_cnt_dict["PEDESTRIAN"] + object_type_cnt_dict["CYCLIST"]
    )
    result_format_list[name_to_row['Avg']][name_to_col["Count"]] = "{}".format(object_count_sum)
    final_avg_results["Count"] = object_count_sum

    result_format_list[name_to_row['Avg']][name_to_col["mJADE"]] = '%.4f,' % min_error.mean()
    result_format_list[name_to_row['Avg']][name_to_col["avgJADE"]] = '%.4f,' % avg_error.mean()
    result_format_list[name_to_row['Avg']][name_to_col["mJFDE"]] = '%.4f,' % min_fde.mean()
    result_format_list[name_to_row['Avg']][name_to_col["avgJFDE"]] = '%.4f,' % avg_fde.mean()
    final_avg_results["mJADE"] = min_error.mean()
    final_avg_results["avgJADE"] = avg_error.mean()
    final_avg_results["mJFDE"] = min_fde.mean()
    final_avg_results["avgJFDE"] = avg_fde.mean()

    result_format_str = ' '.join(
        [x.rjust(9) if x is not None else "     N/A" for items in result_format_list for x in items]
    )

    if verbose:
        result_dict['--------------------------------------------------------------'] = 0

    result_dict.update(final_avg_results)

    if verbose:
        result_dict['---------------------------------------------------------------'] = 0

    if verbose:
        result_dict[
            '-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'
        ] = 0

    if generate_submission:
        submission_data = dict(
            prediction_trajectory_list=batch_pred_trajs,
            prediction_score_list=batch_pred_scores,
            object_id_list=object_id,
            scenario_id_list=scenario_id_list,
        )
    else:
        submission_data = dict()

    return result_dict, result_format_str, submission_data


# def waymo_evaluation(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6, verbose=True, generate_proto=False):
#     # TODO FIXME: This part can be optimized????
#     #  Our output things is tensor. Why bother moving things to numpy???
#
#     pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = transform_preds_to_waymo_format(
#         pred_dicts,
#         top_k_for_eval=top_k,
#         eval_second=eval_second,
#     )
#     eval_config = _default_metrics_config(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval)
#
#     pred_score = tf.convert_to_tensor(pred_score, np.float32)
#     pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
#     gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
#     gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], bool)
#     pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
#     pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], bool)
#     object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)
#
#     metric_results = py_metrics_ops.motion_metrics(
#         config=eval_config.SerializeToString(),
#         prediction_trajectory=pred_trajs,
#         # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
#         prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
#         ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
#         ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
#         prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
#         prediction_ground_truth_indices_mask=pred_gt_indices_mask,
#         # (batch_size, num_pred_groups, num_agents_per_group)
#         object_type=object_type  # (batch_size, num_total_agents)
#     )
#
#     # Generate Proto for Waymo Motion submission
#     if generate_proto:
#         generate_submissition(
#             pred_trajs, pred_score, gt_trajs, gt_is_valid, object_type, gt_infos['scenario_id'], gt_infos['object_id']
#         )
#     metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
#
#     result_dict = {}
#     avg_results = {}
#     for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'mAP', 'OverlapRate']):
#         avg_results.update({f'{m}-VEHICLE': [0.0, 0], f'{m}-PEDESTRIAN': [0.0, 0], f'{m}-CYCLIST': [0.0, 0]})
#         for j, n in enumerate(metric_names):
#             cur_name = n.split('_')[1]
#             avg_results[f'{m}-{cur_name}'][0] += float(metric_results[i][j])
#             avg_results[f'{m}-{cur_name}'][1] += 1
#             result_dict[f'{m}-{n}'] = float(metric_results[i][j])
#
#     for key in avg_results:
#         avg_results[key] = avg_results[key][0] / avg_results[key][1]
#
#     if verbose:
#         result_dict['-------------------------------------------------------------'] = 0
#
#     result_dict.update(avg_results)
#
#     final_avg_results = {}
#     result_format_list = [
#         ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', 'OverlapRate', '\n'],
#         ['VEHICLE', None, None, None, None, None, '\n'],
#         ['PEDESTRIAN', None, None, None, None, None, '\n'],
#         ['CYCLIST', None, None, None, None, None, '\n'],
#         ['Avg', None, None, None, None, None, '\n'],
#     ]
#     name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
#     name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4, 'OverlapRate': 5}
#
#     for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP', 'OverlapRate']:
#         final_avg_results[cur_metric_name] = 0
#         for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
#             final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name}-{cur_name}']
#
#             result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]
#                                                       ] = '%.4f,' % avg_results[f'{cur_metric_name}-{cur_name}']
#
#         final_avg_results[cur_metric_name] /= 3
#         result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]
#
#     result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])
#
#     if verbose:
#         result_dict['--------------------------------------------------------------'] = 0
#
#     result_dict.update(final_avg_results)
#
#     if verbose:
#         result_dict['---------------------------------------------------------------'] = 0
#
#     result_dict.update(object_type_cnt_dict)
#
#     if verbose:
#         result_dict[
#             '-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'
#         ] = 0
#
#     return result_dict, result_format_str

# def main():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
#     parser.add_argument('--top_k', type=int, default=-1, help='')
#     parser.add_argument('--eval_second', type=int, default=8, help='')
#     parser.add_argument('--num_modes_for_eval', type=int, default=6, help='')
#     parser.add_argument('--generate_proto', type=bool, default=False, help='Generate proto file for Waymo challenge')
#
#     args = parser.parse_args()
#     print(args)
#
#     assert args.eval_second in [3, 5, 8]
#     with open(args.pred_infos, 'rb') as f:
#         pred_infos = pickle.load(f)
#
#     print('Start to evaluate the waymo format results...')
#
#     metric_results, result_format_str = waymo_evaluation(
#         pred_dicts=pred_infos,
#         top_k=args.top_k,
#         eval_second=args.eval_second,
#         num_modes_for_eval=args.num_modes_for_eval,
#         generate_proto=args.generate_proto,
#     )
#
#     print(metric_results)
#     metric_result_str = '\n'
#     for key in metric_results:
#         metric_results[key] = metric_results[key]
#         metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
#     print(metric_result_str)
#     print(result_format_str)
#
#
# if __name__ == '__main__':
#     main()
