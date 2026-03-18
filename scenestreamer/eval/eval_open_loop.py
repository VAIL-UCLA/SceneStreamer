import copy
import dataclasses
import itertools
from collections.abc import Iterable
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features

from scenestreamer.utils.utils import numpy_to_torch


def _overwrite_datadict_all_agents(source_data_dict, dest_data_dict, ooi=None):
    import copy
    new_data_dict = copy.deepcopy(dest_data_dict)
    B, T, N, _ = source_data_dict["decoder/reconstructed_position"].shape

    if ooi is None:
        ooi = np.arange(N)

    for id in ooi:  # overwrite all agents
        traj = source_data_dict["decoder/reconstructed_position"][:, :91, id, ]
        traj_mask = source_data_dict["decoder/reconstructed_valid_mask"][:, :91, id]
        theta = source_data_dict['decoder/reconstructed_heading'][:, :91, id]
        vel = source_data_dict['decoder/reconstructed_velocity'][:, :91, id]

        new_data_dict["decoder/agent_position"][:, :, id, :2] = traj
        new_data_dict["decoder/agent_position"][:, :, id, 2] = 0.0
        new_data_dict["decoder/agent_valid_mask"][:, :, id] = traj_mask
        new_data_dict["decoder/agent_heading"][:, :, id] = theta
        new_data_dict["decoder/agent_velocity"][:, :, id] = vel

    return new_data_dict


def detect_env_collision(contour_list1, mask1, lineString):
    collision_detected = []

    for i in range(len(contour_list1)):
        if mask1[i]:
            agent_poly = Polygon(contour_list1[i])

            if agent_poly.intersects(lineString):
                collision_detected.append(True)
            else:
                collision_detected.append(False)
        else:
            collision_detected.append(False)

    return collision_detected


from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour
from shapely.geometry import Polygon


def get_dists(args_list, device):
    return torch.stack(
        [
            tf_to_torch(interaction_features.compute_distance_to_nearest_object(**args_list[k]), device=device)
            for k in range(len(args_list))
        ]
    )


def build_collision_data(*, pred_data_dict, pred_shape, candidate_agents, evaluate_agents, z_values):
    candidate_agents = sorted(set(candidate_agents + evaluate_agents))
    candidate_agents_map = {int(v): k for k, v in enumerate(candidate_agents)}
    evaluate_agents_mask = np.zeros(len(candidate_agents), dtype=bool)

    assert evaluate_agents_mask.ndim == 1
    for k in evaluate_agents:
        evaluate_agents_mask[candidate_agents_map[int(k)]] = 1

    K = pred_shape.shape[0]

    return [
        dict(
            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 0].T),
            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 1].T),
            center_z=conv(z_values[k, candidate_agents]),
            length=conv(pred_shape[k, candidate_agents, :, 0]),
            width=conv(pred_shape[k, candidate_agents, :, 1]),
            height=conv(pred_shape[k, candidate_agents, :, 2]),
            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k, :, candidate_agents].T),
            valid=conv(pred_data_dict["decoder/reconstructed_valid_mask"][k, :, candidate_agents].T, dtype=tf.bool),
            evaluated_object_mask=conv(evaluate_agents_mask, dtype=tf.bool)
        ) for k in range(K)
    ]


def calc_collision(*, dists, valid_masks, T_context, T_gt):
    if type(valid_masks) == list:
        valid_masks = torch.stack(valid_masks)
    collisions = torch.le(dists, interaction_features.COLLISION_DISTANCE_THRESHOLD)
    collisions = collisions[..., T_context:T_gt]
    valid_masks = valid_masks[..., T_context:T_gt]

    collisions = collisions & valid_masks  # Shape: (B, N, T)
    # Number of agents that has coll.
    collisions_count = torch.any(collisions, dim=-1).double().sum(dim=-1)  # Shape: (B,)
    valid_agent_for_collision = torch.any(valid_masks, dim=-1)  # Shape: (B, N)

    # Ratio of agents that has coll.
    mode_cr = collisions_count / valid_agent_for_collision.sum(dim=-1)

    assert mode_cr.ndim == 1
    return collisions, mode_cr


def calc_collision_rate(
    *, pred_data_dict, pred_shape, candidate_agents, evaluate_agents, device, T_gt, T_context, z_values
):
    if isinstance(candidate_agents, torch.Tensor):
        candidate_agents = candidate_agents.cpu().numpy()
    if isinstance(evaluate_agents, torch.Tensor):
        evaluate_agents = evaluate_agents.cpu().numpy()
    if isinstance(candidate_agents, np.ndarray):
        candidate_agents = candidate_agents.tolist()
    if isinstance(evaluate_agents, np.ndarray):
        evaluate_agents = evaluate_agents.tolist()
    if not isinstance(candidate_agents, Iterable):
        candidate_agents = [candidate_agents]
    if not isinstance(evaluate_agents, Iterable):
        evaluate_agents = [evaluate_agents]

    args = build_collision_data(
        pred_data_dict=pred_data_dict,
        pred_shape=pred_shape,
        candidate_agents=candidate_agents,
        evaluate_agents=evaluate_agents,
        z_values=z_values
    )
    dist = get_dists(args, device=device)

    def _get_valid_masks(candidate_agents, evaluate_agents):

        candidate_agents = sorted(set(candidate_agents + evaluate_agents))
        candidate_agents_map = {int(v): k for k, v in enumerate(candidate_agents)}
        evaluate_agents_mask = np.zeros(len(candidate_agents), dtype=bool)
        assert evaluate_agents_mask.ndim == 1
        for k in evaluate_agents:
            evaluate_agents_mask[candidate_agents_map[int(k)]] = 1
        candidate_valid_mask = pred_data_dict["decoder/reconstructed_valid_mask"][:, T_context:T_gt, candidate_agents]
        evaluate_valid_mask = candidate_valid_mask[:, :, evaluate_agents_mask]
        return evaluate_valid_mask.swapaxes(1, 2)

    pred_veh_collisions, veh_cr_mode = calc_collision(
        dists=dist, valid_masks=_get_valid_masks(candidate_agents, evaluate_agents), T_context=T_context, T_gt=T_gt
    )
    return pred_veh_collisions, veh_cr_mode


def print_type_and_dtype(name, tensor):
    print(f"{name} - Type: {type(tensor)}, Dtype: {getattr(tensor, 'dtype', 'N/A')}")


def conv(tensor, dtype=tf.float32):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu()
    return tf.convert_to_tensor(tensor, dtype=dtype)


def tf_to_torch(tf_tensor, device=None):
    # Convert TensorFlow tensor to NumPy array on CPU if necessary
    if tf_tensor.device.endswith("GPU:0"):  # If on GPU, move to CPU first
        tf_tensor = tf_tensor.cpu()
    np_array = tf_tensor.numpy()

    return torch.from_numpy(np_array).to(
        device if device else torch.device("cuda" if tf_tensor.device.endswith("GPU:0") else "cpu")
    )


def jsd(gt_hist, pred_hist, epsilon=1e-10):
    gt_prob = gt_hist / gt_hist.sum()
    pred_prob = pred_hist / pred_hist.sum()
    gt_prob += epsilon
    pred_prob += epsilon
    m = 0.5 * (gt_prob + pred_prob)
    jsd = 0.0
    jsd += F.kl_div(gt_prob.log(), m, reduction="sum")
    jsd += F.kl_div(pred_prob.log(), m, reduction="sum")
    return (0.5 * jsd)


TIMER = False


@contextmanager
def timer(task_name: str):
    start = perf_counter()
    yield
    prof_t = perf_counter() - start
    if TIMER:
        print(f"{task_name}: {prof_t:.5f}")


@dataclasses.dataclass
class Metrics:
    scenario_count: int = 0
    sdc_coll_scenario_count: int = 0
    veh_coll_scenario_count: int = 0

    # Diversity
    sfde_avg: float = 0.0
    sade_avg: float = 0.0
    sfde_min: float = 0.0  # (supervised) avg over scenarios: minimum over all modes: average of L2 error of final positions of all agents
    sade_min: float = 0.0

    skipped_sfde_avg: float = 0.0
    skipped_sade_avg: float = 0.0
    skipped_sfde_min: float = 0.0  # (supervised) avg over scenarios: minimum over all modes: average of L2 error of final positions of all agents
    skipped_sade_min: float = 0.0

    fdd: float = 0.0  # (unsupervised) avg over scenarios: average over all agents: maximum L2 distance in final position of that agent between generated modes
    add: float = 0.0  # (unsupervised) avg over scenarios: average over all agents: maximum L2 distance in final position of that agent between generated modes
    # Xuanhao: In MixSim paper, they used squared norm of distance, but maybe they meant L2 norm not squared norm?
    # Unit given in AdvDiffuser for FDD is m not m^2, so I am using L2 norm here

    # Distribution Realism
    vel_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: velocity JS divergence
    acc_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: acceleration JS divergence
    ttc_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: time to collision JS divergence

    # Common Sense
    env_coll_max: float = 0.0  # offroad
    env_coll_min: float = 0.0  # offroad
    env_coll_avg: float = 0.0  # offroad

    veh_coll_max: float = 0.0  # collision rate
    veh_coll_min: float = 0.0  # collision rate
    veh_coll_avg: float = 0.0  # collision rate

    # SDC-ADV coll
    sdc_adv_coll_max: float = 0.0  # collision rate
    sdc_adv_coll_min: float = 0.0  # collision rate
    sdc_adv_coll_avg: float = 0.0  # collision rate

    sdc_bv_coll_max: float = 0.0  # collision rate
    sdc_bv_coll_min: float = 0.0  # collision rate
    sdc_bv_coll_avg: float = 0.0  # collision rate

    adv_bv_coll_max: float = 0.0  # collision rate
    adv_bv_coll_min: float = 0.0  # collision rate
    adv_bv_coll_avg: float = 0.0  # collision rate

    coll_vel_maxagent_avg: float = 0.0  # collision velocity max over agents, avg over modes
    coll_vel_maxagent_max: float = 0.0  # collision velocity max over agents, max over modes
    coll_vel_maxagent_min: float = 0.0  # collision velocity max over agents, min over modes
    coll_vel_sdc_avg: float = 0.0  # collision velocity only for SDC
    coll_vel_sdc_max: float = 0.0  # collision velocity only for SDC, max over modes
    coll_vel_sdc_min: float = 0.0  # collision velocity only for SDC, min over modes

    # no clue what collision JSD means so not calculating it for now

    # AV comfortable
    sdc_acc_maxtime_avg: float = 0.0
    sdc_acc_maxtime_min: float = 0.0
    sdc_acc_maxtime_max: float = 0.0
    sdc_acc_avgtime_avg: float = 0.0
    sdc_acc_avgtime_min: float = 0.0
    sdc_acc_avgtime_max: float = 0.0

    sdc_jerk_maxtime_avg: float = 0.0
    sdc_jerk_maxtime_min: float = 0.0
    sdc_jerk_maxtime_max: float = 0.0
    sdc_jerk_avgtime_avg: float = 0.0
    sdc_jerk_avgtime_min: float = 0.0
    sdc_jerk_avgtime_max: float = 0.0

    customized_max_sdc_adv_coll: float = 0.0
    customized_max_sdc_bv_coll: float = 0.0
    customized_max_adv_bv_coll: float = 0.0

    customized_min_sdc_adv_coll: float = 0.0
    customized_min_sdc_bv_coll: float = 0.0
    customized_min_adv_bv_coll: float = 0.0

    customized_avg_sdc_adv_coll: float = 0.0
    customized_avg_sdc_bv_coll: float = 0.0
    customized_avg_adv_bv_coll: float = 0.0

    customized_avg_overall_coll: float = 0.0

    customized_all_agent_coll: float = 0.0

    def clean(self):
        # If the entry is tensor, drop it to float.
        for k, v in dataclasses.asdict(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.item())

    def aggregate(self):
        self.clean()

        # Get all metrics
        all_metrics = dataclasses.asdict(self)
        for k, v in all_metrics.items():
            if k.startswith("coll_vel_sdc"):
                if self.sdc_coll_scenario_count > 0:
                    all_metrics[k] = v / self.sdc_coll_scenario_count
                else:
                    all_metrics[k] = torch.nan

            elif k.startswith("coll_vel_maxagent"):
                if self.veh_coll_scenario_count > 0:
                    all_metrics[k] = v / self.veh_coll_scenario_count
                else:
                    all_metrics[k] = torch.nan

            elif k != "scenario_count":
                all_metrics[k] = v / self.scenario_count
        return all_metrics


class Evaluator:
    SECONDS_PER_STEP = 0.1

    def __init__(self, CR_mode="mean", key_metrics_only=True, use_waymo=False):
        assert CR_mode in ["min", "max", "mean"]
        self.CR_mode = CR_mode
        self.jsd_config = {
            "vel": {
                "min_val": 0.0,
                "max_val": 50.0,
                "num_bins": 100
            },
            "acc": {
                "min_val": -10.0,
                "max_val": 10.0,
                "num_bins": 200
            },

            # From WOSAC: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto#L80C1-L89C2
            "ttc": {
                "min_val": 0.0,
                "max_val": 5.0,
                "num_bins": 10
            }
        }

        self.metrics = Metrics()
        self.key_metrics_only = key_metrics_only
        self.use_waymo = use_waymo

    def filter_static_agents(self, gt_data_dict):
        # return a mask for all static agetns (GT traj in both x and y less than 5m)

        mask = torch.zeros_like(gt_data_dict["decoder/agent_id"], dtype=torch.bool)

        for id in gt_data_dict["decoder/agent_id"]:
            traj = gt_data_dict["decoder/agent_position"][:, id][gt_data_dict["decoder/agent_valid_mask"][:, id], :2]
            diffs = traj[0] - traj[-1]  # calcualte the difference of start and end index

            dist = torch.norm(diffs, dim=-1)

            if dist < 5:
                mask[id] = 1

        return mask

    def add(self, gt_data_dict, pred_data_dict, adv_list, bv_list, device=None):

        self.metrics.scenario_count += 1

        T_gt = gt_data_dict["decoder/agent_position"].shape[0]
        T_context = 0

        T_pred = pred_data_dict["decoder/reconstructed_position"].shape[1]
        B = K = pred_data_dict["decoder/reconstructed_position"].shape[0]
        N = gt_data_dict["decoder/agent_position"].shape[1]

        vehicle_mask = numpy_to_torch(gt_data_dict["decoder/agent_type"] == 1, device=device)  # (num agents)
        static_agent_mask = self.filter_static_agents(gt_data_dict)

        ooi_mask = torch.zeros_like(vehicle_mask, dtype=torch.bool, device=device)
        # ooi_mask[(gt_data_dict["decoder/object_of_interest_id"])] = 1
        # ooi_mask[(gt_data_dict["decoder/sdc_id"])] = 1 # now only predict OOI
        ooi_mask[(gt_data_dict["decoder/agent_id"])] = 1  # (num agents)

        gt_valid_mask = numpy_to_torch(
            gt_data_dict["decoder/agent_valid_mask"], device=device
        ).T  # (num agents, num steps)
        pred_valid_mask = pred_data_dict["decoder/reconstructed_valid_mask"].transpose(
            1, 2
        )  # (K, num agents, num steps)
        # joint_mask = vehicle_mask.unsqueeze(-1) & valid_mask # (num agents, num steps)
        # gt_ooi_joint = ooi_mask.unsqueeze(-1) & gt_valid_mask[..., T_context:T_gt]  # (num agents, num steps)
        # pred_ooi_joint = ooi_mask[None, ..., None] & pred_valid_mask[..., T_context:T_gt]  # (K, num agents, num steps)

        gt_ooi_joint = gt_valid_mask[..., T_context:T_gt]  # (num agents, num steps)
        pred_ooi_joint = pred_valid_mask[..., T_context:T_gt]  # (K, num agents, num steps)

        gt_shape = numpy_to_torch(
            gt_data_dict["decoder/current_agent_shape"][None], device=device
        ).expand(T_gt, -1, -1).transpose(0, 1)
        pred_shape = pred_data_dict["decoder/current_agent_shape"][:, None].expand(-1, T_pred, -1, -1).transpose(1, 2)

        sdc_index = int(gt_data_dict["decoder/sdc_index"])
        sdc_index_in_ooi = list(gt_data_dict["decoder/agent_id"]).index(sdc_index)

        # minSFDE
        with timer("minSFDE"):
            gt_pos = numpy_to_torch(gt_data_dict["decoder/agent_position"], device=device)[None, ..., :2]
            pred_pos = pred_data_dict["decoder/reconstructed_position"][:, :T_gt]

            gt_valid = gt_ooi_joint[None]
            gt_valid_skipped = gt_valid[:, :, ::5]

            last_valid_ind = gt_valid.cumsum(dim=-1).argmax(dim=-1)
            last_valid_ind_skipped = gt_valid_skipped.cumsum(dim=-1).argmax(dim=-1)

            error = torch.linalg.norm(gt_pos - pred_pos, dim=-1)
            assert error.ndim == 3
            assert error.shape[0] == B

            last_valid_ind = last_valid_ind.unsqueeze(0).expand(B, 1, N)
            last_valid_ind_skipped = last_valid_ind_skipped.unsqueeze(0).expand(B, 1, N)

            assert last_valid_ind.shape == (B, 1, N)
            fde = torch.gather(error, 1, last_valid_ind).squeeze(1)  # shape: B, N
            fde_skipped = torch.gather(error, 1, last_valid_ind_skipped * 5).squeeze(1)  # shape: B, N

            assert fde.shape[0] == B
            agent_valid = gt_valid.any(-1).expand(B, N)  # shape: B, N
            agent_valid_skipped = gt_valid_skipped.any(-1).expand(B, N)

            sfde = (fde * agent_valid).sum(-1) / agent_valid.sum(-1)
            sfde_skipped = (fde_skipped * agent_valid_skipped).sum(-1) / agent_valid_skipped.sum(-1)

            gt_valid = gt_valid.permute(0, 2, 1)
            gt_valid_skipped = gt_valid_skipped.permute(0, 2, 1)

            gt_valid_expand = gt_valid.expand(B, T_gt, N)
            pred_ooi_joint_expand = pred_ooi_joint.permute(0, 2, 1)
            sade_per_agent = (error * gt_valid).sum(1) / gt_valid.sum(1).clamp(1)
            assert sade_per_agent.ndim == 2
            sade = (sade_per_agent * gt_valid.any(1)).sum(1) / gt_valid.any(1).sum(1)
            assert sade.ndim == 1
            sade_skipped_per_agent = (error[:, ::5] * gt_valid_skipped).sum(1) / gt_valid_skipped.sum(1).clamp(1)
            sade_skipped = sade_skipped_per_agent.sum(1) / gt_valid_skipped.any(1).sum(1)

            assert sfde.ndim == 1
            assert sfde.shape[0] == B
            self.metrics.sfde_min += sfde.min()
            self.metrics.sade_min += sade.min()
            self.metrics.sfde_avg += sfde.mean()
            self.metrics.sade_avg += sade.mean()

            self.metrics.skipped_sfde_min += sfde_skipped.min()
            self.metrics.skipped_sade_min += sade_skipped.min()
            self.metrics.skipped_sfde_avg += sfde_skipped.mean()
            self.metrics.skipped_sade_avg += sade_skipped.mean()

        # Following wosac_eval, fill in z with GT t = 10 data
        z_values = pred_data_dict["decoder/current_agent_position"][..., 2].unsqueeze(-1).expand(-1, -1, T_pred)

        # FDD
        with timer("FDD"):
            # there doesn't appear to be an easy way to do this with cartesian product
            cur_FDD = None
            pred_ooi_valid_mask = pred_valid_mask[:, ooi_mask]
            single_mode_ooi_valid_mask = pred_ooi_valid_mask[0]

            # assert torch.all(torch.any(pred_ooi_valid_mask, dim=-1))
            last_valid_ind = pred_ooi_valid_mask.cumsum(dim=-1).argmax(dim=-1)  # (K, N)
            ooi_reconstructed_pos = pred_data_dict["decoder/reconstructed_position"][:, :,
                                                                                     ooi_mask]  # (K, T_pred, N, 2)
            last_valid_ind_reshaped = last_valid_ind[:, None, :, None].expand(-1, -1, -1, 2)
            final_pos = torch.gather(ooi_reconstructed_pos, dim=1, index=last_valid_ind_reshaped).squeeze(1)
            for i, j in itertools.product(range(K), range(K)):
                final_dist = torch.linalg.norm(final_pos[i] - final_pos[j], dim=-1)
                assert final_dist.ndim == 1
                if cur_FDD == None:
                    cur_FDD = final_dist
                else:
                    cur_FDD = torch.maximum(cur_FDD, final_dist)
            self.metrics.fdd += (cur_FDD *
                                 single_mode_ooi_valid_mask.any(-1)).sum() / single_mode_ooi_valid_mask.any(-1).sum()

            def _add(pos, mask):
                assert pos.ndim == 4
                assert pos.shape[0] == B
                assert pos.shape[2] == N
                assert pos.shape[3] == 2
                T = pos.shape[1]
                pos = pos.reshape(B, -1, 2)
                pos_NB = pos.swapaxes(0, 1)
                dist_NBB = torch.cdist(pos_NB, pos_NB)
                max_dist_N = dist_NBB.amax((1, 2))
                assert max_dist_N.shape == (N * T, ), max_dist_N.shape
                max_dist_TN = max_dist_N.reshape(T, N)
                assert mask.shape == max_dist_TN.shape, (mask.shape, max_dist_TN.shape)
                avg_t = utils.masked_average(max_dist_TN, mask, dim=0)
                assert avg_t.shape == (N, )
                return avg_t

            # add_full_all = utils.masked_average(_add(pred_pos, gt_valid[0]), agent_valid[0], dim=0)
            add_skipped_all = utils.masked_average(
                _add(ooi_reconstructed_pos, single_mode_ooi_valid_mask.swapaxes(0, 1)),
                single_mode_ooi_valid_mask.any(-1),
                dim=0
            )
            self.metrics.add += add_skipped_all

        with timer("Kinematic Metrics"):
            gt_speed, gt_accel, gt_jerk = self._compute_kinematic_metrics(
                gt_data_dict["decoder/agent_velocity"].swapaxes(1, 0), device
            )  # (N, T)
            pred_speed, pred_accel, pred_jerk = self._compute_kinematic_metrics(
                pred_data_dict["decoder/reconstructed_velocity"].transpose(1, 2), device
            )  # (K, N, T)
            gt_speed = gt_speed[..., T_context:T_gt]
            gt_accel = gt_accel[..., T_context:T_gt]
            gt_jerk = gt_jerk[..., T_context:T_gt]
            pred_speed = pred_speed[..., T_context:T_gt]
            pred_accel = pred_accel[..., T_context:T_gt]
            pred_jerk = pred_jerk[..., T_context:T_gt]

        if self.use_waymo:
            candidate_agents = gt_data_dict["decoder/current_agent_valid_mask"]
            if isinstance(candidate_agents, torch.Tensor):
                candidate_agents = candidate_agents.cpu().numpy()
            candidate_agents = candidate_agents.nonzero()[0]

            pred_veh_collisions, veh_cr_mode = calc_collision_rate(
                candidate_agents=candidate_agents,
                evaluate_agents=gt_data_dict["decoder/agent_id"],
                pred_data_dict=pred_data_dict,
                pred_shape=pred_shape,
                device=device,
                T_gt=T_gt,
                T_context=T_context,
                z_values=z_values
            )

            assert veh_cr_mode.shape[0] == B
            self.metrics.veh_coll_avg += veh_cr_mode.mean()
            self.metrics.veh_coll_min += veh_cr_mode.min()
            self.metrics.veh_coll_max += veh_cr_mode.max()

            if adv_list is not None:
                assert len(adv_list) == 1
                assert adv_list[0] not in bv_list
                assert sdc_index not in adv_list

                # self.sdc_coll_adv_active = True
                for kk in adv_list:
                    assert int(kk.item()) in candidate_agents

                adv_sdc_coll, adv_sdc_coll_rate = calc_collision_rate(
                    candidate_agents=adv_list,
                    evaluate_agents=pred_data_dict["decoder/sdc_index"],
                    pred_data_dict=pred_data_dict,
                    pred_shape=pred_shape,
                    device=device,
                    T_gt=T_gt,
                    T_context=T_context,
                    z_values=z_values
                )

                assert adv_sdc_coll_rate.ndim == 1
                self.metrics.sdc_adv_coll_avg += adv_sdc_coll_rate.mean()
                self.metrics.sdc_adv_coll_min += adv_sdc_coll_rate.min()
                self.metrics.sdc_adv_coll_max += adv_sdc_coll_rate.max()

                adv_bv_coll, adv_bv_coll_rate = calc_collision_rate(
                    candidate_agents=bv_list,
                    evaluate_agents=adv_list,
                    pred_data_dict=pred_data_dict,
                    pred_shape=pred_shape,
                    device=device,
                    T_gt=T_gt,
                    T_context=T_context,
                    z_values=z_values
                )
                assert adv_bv_coll_rate.ndim == 1

                assert adv_bv_coll_rate.shape[0] == B
                self.metrics.adv_bv_coll_avg += adv_bv_coll_rate.mean()
                self.metrics.adv_bv_coll_min += adv_bv_coll_rate.min()
                self.metrics.adv_bv_coll_max += adv_bv_coll_rate.max()

            assert sdc_index not in bv_list
            assert bv_list is not None
            for kk in bv_list:
                assert int(kk.item()) in candidate_agents
            sdc_bv_coll, sdc_bv_coll_rate = calc_collision_rate(
                candidate_agents=bv_list,
                evaluate_agents=pred_data_dict["decoder/sdc_index"],
                pred_data_dict=pred_data_dict,
                pred_shape=pred_shape,
                device=device,
                T_gt=T_gt,
                T_context=T_context,
                z_values=z_values
            )
            assert sdc_bv_coll_rate.ndim == 1

            assert sdc_bv_coll_rate.shape[0] == B
            self.metrics.sdc_bv_coll_avg += sdc_bv_coll_rate.mean()
            self.metrics.sdc_bv_coll_min += sdc_bv_coll_rate.min()
            self.metrics.sdc_bv_coll_max += sdc_bv_coll_rate.max()

            # map_feature = gt_data_dict["encoder/map_feature"]
            map_feature = gt_data_dict["vis/map_feature"]
            assert map_feature.ndim == 3  # This is unbatched.

            road_edges = []
            for i in range(map_feature.shape[0]):
                # For each map feature
                if map_feature[i, 0, 15] == 1:
                    map_feat = []
                    for j in range(map_feature.shape[1]):
                        if gt_data_dict['encoder/map_feature_valid_mask'][i, j]:
                            map_feat.append(
                                map_pb2.MapPoint(
                                    x=map_feature[i, j, 0],
                                    y=map_feature[i, j, 1],
                                    z=0
                                    # map_feature[i, j, 2] # let's say there is no z axis any more
                                )
                            )
                    road_edges.append(map_feat)

            eval_mask = ooi_mask & (~static_agent_mask)
            env_nearest_distances = torch.stack(
                [
                    tf_to_torch(
                        map_metric_features.compute_distance_to_road_edge(
                            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 0].T),
                            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 1].T),
                            center_z=conv(z_values[k]),
                            length=conv(pred_shape[k, ..., 0]),
                            width=conv(pred_shape[k, ..., 1]),
                            height=conv(pred_shape[k, ..., 2]),
                            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k].T),
                            valid=conv(pred_valid_mask[k], dtype=tf.bool),
                            evaluated_object_mask=conv(eval_mask, dtype=tf.bool),
                            road_edge_polylines=road_edges,
                        ),
                        device=device
                    ) for k in range(K)
                ]
            )

            pred_valid_mask = pred_valid_mask[..., T_context:T_gt]

            pred_env_collisions = torch.greater(env_nearest_distances, map_metric_features.OFFROAD_DISTANCE_THRESHOLD)
            pred_env_collisions = pred_env_collisions[..., T_context:T_gt]
            pred_env_collisions_traj_level = pred_env_collisions.any(dim=-1)  # (B, num_ooi)
            # Avg over agent dim. Here we assume all evaluated agents are valid so don't do the masked_avg
            env_collision_rate = pred_env_collisions_traj_level.float().mean(-1)

            # ==================================== customized env collision rate ====================================
            # env_collision_rate = np.array(pred_env_collisions_traj_level).mean(-1)
            assert env_collision_rate.ndim == 1

            assert env_collision_rate.shape[0] == B
            self.metrics.env_coll_avg += env_collision_rate.mean()
            self.metrics.env_coll_min += env_collision_rate.min()
            self.metrics.env_coll_max += env_collision_rate.max()

            step_wise_collision = pred_veh_collisions
            scenario_has_collision = torch.any(step_wise_collision).item()

            if scenario_has_collision:
                speed_when_collision = torch.where(step_wise_collision, pred_speed[:, ooi_mask], 0)

                coll_vel_max_agent = speed_when_collision.amax(dim=(-1, -2))
                coll_valid_mask = speed_when_collision.any(-1).any(-1)
                coll_vel_max_agent = coll_vel_max_agent[coll_valid_mask]

                assert coll_vel_max_agent.ndim == 1

                if coll_vel_max_agent.numel() != 0:
                    self.metrics.coll_vel_maxagent_avg += (coll_vel_max_agent).sum() / coll_valid_mask.sum().clamp(1)
                    self.metrics.coll_vel_maxagent_min += coll_vel_max_agent.min()
                    self.metrics.coll_vel_maxagent_max += coll_vel_max_agent.max()

                    self.metrics.veh_coll_scenario_count += 1

            sdc_speed = pred_speed[:, sdc_index]
            sdc_coll = step_wise_collision[:, sdc_index_in_ooi]

            # sdc_speed_when_coll = torch.where(sdc_coll, sdc_speed, torch.nan).amax(-1)
            sdc_speed_when_coll = torch.where(sdc_coll, sdc_speed, torch.nan)
            valid_mask = ~torch.isnan(sdc_speed_when_coll)

            if torch.any(sdc_coll).item():  # if there is valid collision
                self.metrics.coll_vel_sdc_avg += (sdc_speed_when_coll[valid_mask]).sum() / valid_mask.sum()
                self.metrics.coll_vel_sdc_max += (sdc_speed_when_coll[valid_mask]).max()
                self.metrics.coll_vel_sdc_min += (sdc_speed_when_coll[valid_mask]).min()
                self.metrics.sdc_coll_scenario_count += 1
                assert sdc_speed_when_coll.shape[0] == B

            gt_ttc = tf_to_torch(
                interaction_features.compute_time_to_collision_with_object_in_front(
                    center_x=conv(gt_data_dict["decoder/agent_position"][..., 0].T),
                    center_y=conv(gt_data_dict["decoder/agent_position"][..., 1].T),
                    length=conv(gt_shape[..., 0]),
                    width=conv(gt_shape[..., 1]),
                    heading=conv(gt_data_dict["decoder/agent_heading"].T),
                    valid=conv(gt_valid_mask, dtype=tf.bool),
                    evaluated_object_mask=conv(ooi_mask, dtype=tf.bool),
                    seconds_per_step=self.SECONDS_PER_STEP
                ),
                device=device
            )
            pred_ttc = torch.stack(
                [
                    tf_to_torch(
                        interaction_features.compute_time_to_collision_with_object_in_front(
                            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 0].T),
                            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 1].T),
                            length=conv(pred_shape[k, ..., 0]),
                            width=conv(pred_shape[k, ..., 1]),
                            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k].T),
                            valid=conv(pred_data_dict["decoder/reconstructed_valid_mask"][k].T, dtype=tf.bool),
                            evaluated_object_mask=conv(ooi_mask, dtype=tf.bool),
                            seconds_per_step=self.SECONDS_PER_STEP
                        ),
                        device=device
                    ) for k in range(K)
                ]
            )
            gt_ttc = gt_ttc[..., T_context:T_gt]
            pred_ttc = pred_ttc[..., T_context:T_gt]

        sdc_acc = torch.abs(torch.nan_to_num(pred_accel[:, sdc_index]))  # Shape: (K, T)
        sdc_mask = pred_valid_mask[:, sdc_index][:, T_context:T_gt]
        sdc_acc_avgt = (sdc_acc * sdc_mask).sum(-1) / sdc_mask.sum(-1).clamp(1)
        assert sdc_acc_avgt.ndim == 1
        assert sdc_acc_avgt.shape[0] == B
        self.metrics.sdc_acc_avgtime_max += sdc_acc_avgt.max()
        self.metrics.sdc_acc_avgtime_avg += sdc_acc_avgt.mean()
        self.metrics.sdc_acc_avgtime_min += sdc_acc_avgt.min()

        sdc_acc_maxt = sdc_acc.amax(-1)
        assert sdc_acc_maxt.ndim == 1
        assert sdc_acc_maxt.shape[0] == B
        self.metrics.sdc_acc_maxtime_max += sdc_acc_maxt.max()
        self.metrics.sdc_acc_maxtime_avg += sdc_acc_maxt.mean()
        self.metrics.sdc_acc_maxtime_min += sdc_acc_maxt.min()

        sdc_jerk = torch.abs(torch.nan_to_num(pred_jerk[:, sdc_index]))  # Shape: (K, T)
        sdc_mask = pred_valid_mask[:, sdc_index][:, T_context:T_gt]
        sdc_jerk_avgt = (sdc_jerk * sdc_mask).sum(-1) / sdc_mask.sum(-1).clamp(1)
        assert sdc_jerk_avgt.ndim == 1
        assert sdc_jerk_avgt.shape[0] == B
        self.metrics.sdc_jerk_avgtime_max += sdc_jerk_avgt.max()
        self.metrics.sdc_jerk_avgtime_avg += sdc_jerk_avgt.mean()
        self.metrics.sdc_jerk_avgtime_min += sdc_jerk_avgt.min()

        sdc_jerk_maxt = sdc_jerk.amax(-1)
        assert sdc_jerk_maxt.ndim == 1
        assert sdc_jerk_maxt.shape[0] == B
        self.metrics.sdc_jerk_maxtime_max += sdc_jerk_maxt.max()
        self.metrics.sdc_jerk_maxtime_avg += sdc_jerk_maxt.mean()
        self.metrics.sdc_jerk_maxtime_min += sdc_jerk_maxt.min()

        with timer("Histograms"):
            gt_speed_hist, gt_speed_bins = torch.histogram(
                torch.clip(
                    gt_speed[gt_ooi_joint & ~gt_speed.isnan()], self.jsd_config["vel"]["min_val"],
                    self.jsd_config["vel"]["max_val"]
                ).cpu(),
                self.jsd_config["vel"]["num_bins"],
                density=False
            )
            # .cpu() since histogram doesn't support cuda backend
            pred_speed_hist, pred_speed_bins = torch.histogram(
                torch.clip(
                    pred_speed[pred_ooi_joint & ~pred_speed.isnan()], self.jsd_config["vel"]["min_val"],
                    self.jsd_config["vel"]["max_val"]
                ).cpu(),
                self.jsd_config["vel"]["num_bins"],
                density=False
            )
            gt_accel_hist, gt_accel_bins = torch.histogram(
                torch.clip(
                    gt_accel[gt_ooi_joint & ~gt_accel.isnan()], self.jsd_config["acc"]["min_val"],
                    self.jsd_config["acc"]["max_val"]
                ).cpu(),
                self.jsd_config["acc"]["num_bins"],
                density=False
            )
            pred_accel_hist, pred_accel_bins = torch.histogram(
                torch.clip(
                    pred_accel[pred_ooi_joint & ~pred_accel.isnan()], self.jsd_config["acc"]["min_val"],
                    self.jsd_config["acc"]["max_val"]
                ).cpu(),
                self.jsd_config["acc"]["num_bins"],
                density=False
            )

            if self.use_waymo:
                gt_ttc_hist, gt_ttc_bins = torch.histogram(
                    torch.clip(
                        gt_ttc[gt_valid_mask[ooi_mask, T_context:T_gt] & ~gt_ttc.isnan()],
                        self.jsd_config["ttc"]["min_val"], self.jsd_config["ttc"]["max_val"]
                    ).cpu(),
                    self.jsd_config["ttc"]["num_bins"],
                    density=False
                )
                pred_ttc_hist, pred_ttc_bins = torch.histogram(
                    torch.clip(
                        pred_ttc[pred_valid_mask[:, ooi_mask, T_context:T_gt] & ~pred_ttc.isnan()],
                        self.jsd_config["ttc"]["min_val"], self.jsd_config["ttc"]["max_val"]
                    ).cpu(),
                    self.jsd_config["ttc"]["num_bins"],
                    density=False
                )

        with timer("JSD"):
            speed_jsd = jsd(gt_speed_hist, pred_speed_hist)
            acc_jsd = jsd(gt_accel_hist, pred_accel_hist)
            self.metrics.vel_jsd += speed_jsd
            self.metrics.acc_jsd += acc_jsd
            if self.use_waymo:
                ttc_jsd = jsd(gt_ttc_hist, pred_ttc_hist)
                self.metrics.ttc_jsd += ttc_jsd

    def _compute_kinematic_metrics(self, vel, device):
        if type(vel) == np.ndarray:
            vel = numpy_to_torch(vel, device=device)
        speed = torch.linalg.norm(vel, axis=-1)
        accel = self._central_diff(speed, device, pad_value=torch.nan) / self.SECONDS_PER_STEP
        jerk = self._central_diff(accel, device, pad_value=torch.nan) / self.SECONDS_PER_STEP
        return speed, accel, jerk

    def _central_diff(self, tensor, device, pad_value=torch.nan):
        pad_shape = (*tensor.shape[:-1], 1)
        pad_tensor = torch.ones(pad_shape, device=device) * pad_value
        diff_t = (tensor[..., 2:] - tensor[..., :-2]) / 2
        return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)

    def add_customized_CR(
        self,
        max_sdc_adv_cr=None,
        max_sdc_bv_cr=None,
        max_adv_bv_cr=None,
        min_sdc_adv_cr=None,
        min_sdc_bv_cr=None,
        min_adv_bv_cr=None,
        avg_sdc_adv_cr=None,
        avg_sdc_bv_cr=None,
        avg_adv_bv_cr=None,
        all_agent_cr=None
    ):
        if max_sdc_adv_cr is not None:
            self.metrics.customized_max_sdc_adv_coll += max_sdc_adv_cr
        if max_sdc_bv_cr is not None:
            self.metrics.customized_max_sdc_bv_coll += max_sdc_bv_cr

        if max_adv_bv_cr is not None:
            self.metrics.customized_max_adv_bv_coll += max_adv_bv_cr

        if min_sdc_adv_cr is not None:
            self.metrics.customized_min_sdc_adv_coll += min_sdc_adv_cr
        if min_sdc_bv_cr is not None:
            self.metrics.customized_min_sdc_bv_coll += min_sdc_bv_cr
        if min_adv_bv_cr is not None:
            self.metrics.customized_min_adv_bv_coll += min_adv_bv_cr

        if avg_sdc_adv_cr is not None:
            self.metrics.customized_avg_sdc_adv_coll += avg_sdc_adv_cr
        if avg_sdc_bv_cr is not None:
            self.metrics.customized_avg_sdc_bv_coll += avg_sdc_bv_cr
        if avg_adv_bv_cr is not None:
            self.metrics.customized_avg_adv_bv_coll += avg_adv_bv_cr

        if all_agent_cr is not None:
            self.metrics.customized_all_agent_coll += all_agent_cr

    def aggregate(self):
        return self.metrics.aggregate()

    def print(self):
        metrics = self.metrics.aggregate()
        print("\n=====================================")
        print("Evaluation Metrics:")
        print(utils.pretty_print(metrics))
        print("=====================================")
        return metrics

    def save(self, save_path=None):
        if save_path is None:
            save_path = "evaluation_results"

        metrics = self.metrics.aggregate()
        metrics["save_path"] = save_path

        # Save a json:
        import json
        json_file = save_path + ".json"
        with open(json_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {json_file}")

        # Save a csv:
        import pandas as pd
        df = pd.DataFrame([metrics])
        csv_file = save_path + ".csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved metrics to {csv_file}")

        return metrics


class TurnAction:
    STOP = 0
    KEEP_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    U_TURN = 4

    num_actions = 5


class AccelerationAction:
    STOP = 0
    KEEP_SPEED = 1
    SPEED_UP = 2
    SLOW_DOWN = 3

    num_actions = 4


class SafetyAction:
    SAFE = 0
    COLLISION = 1
    num_actions = 2


def detect_collision(contour_list1, mask1, contour_list2, mask2):
    collision_detected = []

    contour_list1, len1 = contour_list1
    contour_list2, len2 = contour_list2

    assert len(contour_list1) == len(contour_list2)

    for i in range(len(contour_list1)):
        if mask1[i] and mask2[i]:
            pos1 = contour_list1[i].mean(0)
            pos2 = contour_list2[i].mean(0)
            dist = np.linalg.norm(pos1 - pos2)

            # PZH: Actually the largest possible distance is sqrt(2)/2*(len1 + len2)
            # We relax it to (len1+len2)
            if dist > (len1 + len2):
                collision_detected.append(False)
                continue

            poly1 = Polygon(contour_list1[i])
            poly2 = Polygon(contour_list2[i])

            if poly1.intersects(poly2):
                collision_detected.append(True)
            else:
                collision_detected.append(False)
        else:
            collision_detected.append(False)

    return collision_detected


def get_2D_collision_labels(data_dict, track_agent_indicies):
    # Now, instead of getting 1d-array of collision labels, let's do 2-d array to detect whether there is collision between given two agents.

    safety_actions = torch.zeros((track_agent_indicies.shape[0], track_agent_indicies.shape[0]), dtype=int)  # plus sdc

    contours = []
    for agent1_id in track_agent_indicies:
        traj = data_dict["decoder/agent_position"][:91, agent1_id, :]  # (91, 3)
        length = data_dict["decoder/agent_shape"][10, agent1_id, 0]
        width = data_dict["decoder/agent_shape"][10, agent1_id, 1]
        theta = data_dict['decoder/agent_heading'][:91, agent1_id]  # (91, ) # in pi
        mask = data_dict['decoder/agent_valid_mask'][:91, agent1_id]  # (91,)
        poly = cal_polygon_contour(traj[:, 0], traj[:, 1], theta, width, length)
        contours.append((poly, length))

    for i in range(track_agent_indicies.shape[0] - 1):
        for j in range(i + 1, track_agent_indicies.shape[0]):
            mask_1 = data_dict['decoder/agent_valid_mask'][:91, track_agent_indicies[i]]  # (91,)
            mask_2 = data_dict['decoder/agent_valid_mask'][:91, track_agent_indicies[j]]
            collision_detected = detect_collision(contours[i], mask_1, contours[j], mask_2)

            if any(collision_detected):
                # print(f"Collision between {i} and {j} happen at step: {np.array(collision_detected).nonzero()}")
                safety_actions[i][j] = 1  # Label collisions for OOIs now. Later we will build a larger dict.
                safety_actions[j][i] = 1  # Label collisions for OOIs now. Later we will build a larger dict.

    assert np.array_equal(safety_actions, safety_actions.T), "The 2D label is not symmetrical"
    return safety_actions


def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


class EvaluationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        evaluator: Evaluator,
        tokenizer,
        config,
        # dataset,
        autoregressive_start_step,
        num_modes=1,
        save_path=None,
        use_waymo=False
    ):
        super().__init__()
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        # self.dataset = dataset
        self.num_modes = num_modes
        self.cat_summary = None
        self.baseline_summary = None
        self.adv_index = None
        self.sid = None
        self.save_path = save_path
        self.autoregressive_start_step = autoregressive_start_step
        assert save_path is not None, "Please specify the save path for the evaluation results."

    def GPT_AR(self, input_data, backward_prediction=False, teacher_forcing=False):
        assert not teacher_forcing
        assert not backward_prediction
        autoregressive_start_step = self.autoregressive_start_step
        from scenestreamer.infer.motion import generate_motion
        return generate_motion(
            data_dict=input_data,
            model=self.model.model,
            autoregressive_start_step=autoregressive_start_step,
            allow_newly_added_agent_step=2,
            teacher_forcing_sdc=False
        )

    def preprocess_GPTmodel(self, raw_data, backward_prediction=False):
        input_data = utils.numpy_to_torch(raw_data, device=self.model.device)
        input_data["in_evaluation"] = torch.tensor([1], dtype=bool).to(self.model.device)

        input_data = {
            # k: utils.expand_for_modes(v, num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data.items()
        }

        # Force to run backward prediction first to make sure the data is tokenized correctly!!!
        tok_data_dict, _ = self.tokenizer.tokenize(input_data, backward_prediction=backward_prediction)
        input_data.update(tok_data_dict)

        if not backward_prediction:  # handle backward flag
            if self.config.BACKWARD_PREDICTION:
                input_data["in_backward_prediction"] = torch.tensor(
                    [False] * self.num_modes, dtype=bool
                ).to(self.model.device)
        else:
            input_data["in_backward_prediction"] = torch.tensor(
                [True] * self.num_modes, dtype=bool
            ).to(self.model.device)

        return input_data

    def validation_step(self, batch, batch_idx):

        data_dict = copy.deepcopy(batch)
        input_data = numpy_to_torch(data_dict, device=self.model.device)
        original_data_dict_tensor = copy.deepcopy(input_data)

        input_data = self.preprocess_GPTmodel(batch)

        with torch.no_grad():
            output_data = self.GPT_AR(input_data)

        gathered_output = output_data

        avg_sdc_adv_cr, avg_sdc_bv_cr, avg_adv_bv_cr, all_agent_cr = self.calculate_collision_statistics(
            output_data,
            is_CAT_data=False,
        )
        self.evaluator.add_customized_CR(
            avg_sdc_adv_cr=avg_sdc_adv_cr,
            avg_adv_bv_cr=avg_adv_bv_cr,
            avg_sdc_bv_cr=avg_sdc_bv_cr,
            all_agent_cr=all_agent_cr
        )

        all_agents = batch["decoder/agent_id"]  # prepare parameters for differet CR metrics
        sdc_id = batch["decoder/sdc_index"]
        all_agents_except_sdc = all_agents[all_agents != sdc_id]
        self.evaluator.add(
            original_data_dict_tensor,
            gathered_output,
            adv_list=None,
            bv_list=all_agents_except_sdc,
            device=self.device
        )

        return gathered_output

    def on_test_epoch_end(self):
        self.trainer.strategy.barrier()  # ensure all processes are done with evaluation
        if self.trainer.is_global_zero:
            self.evaluator.print()
            self.evaluator.save(self.save_path)

    def on_validation_epoch_end(self):
        self.trainer.strategy.barrier()  # ensure all processes are done with evaluation
        if self.trainer.is_global_zero:
            self.evaluator.print()
            self.evaluator.save(self.save_path)

    def configure_optimizers(self):
        # No optimizer required for evaluation
        return None

    def calculate_collision_statistics(self, output_data, cr_mode="avg", is_CAT_data=False):

        ooi_ind = output_data["decoder/agent_id"][0]  # ooi is all agent
        # from scenestreamer.dataset.preprocess_action_label import get_2D_collision_labels

        output_data_all_modes = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_data.items()
        }

        output_data_all_modes = _overwrite_datadict_all_agents(
            source_data_dict=output_data_all_modes, dest_data_dict=output_data_all_modes
        )  # overwrite pred to GT

        num_modes = self.num_modes

        sdc_adv_col = 0
        sdc_bv_col = 0
        adv_bv_col = 0

        avg_sdc_adv_col = 0
        avg_sdc_bv_col = 0
        avg_adv_bv_col = 0

        avg_all_agent_cr = 0
        num_bv_agent = 0

        for i in range(num_modes):
            output_dict_mode = _get_mode(output_data_all_modes, i, num_modes=num_modes)

            col_label = get_2D_collision_labels(data_dict=output_dict_mode, track_agent_indicies=ooi_ind)

            sdc_index = 0
            adv_index = self.adv_index  # value is None for eval_mode = GPTmodel

            if adv_index is not None and col_label[sdc_index][adv_index]:
                sdc_adv_col += 1

            for agent_id in ooi_ind:
                if agent_id == adv_index or agent_id == sdc_index:
                    continue

                if col_label[sdc_index][agent_id]:
                    sdc_bv_col += 1

                if adv_index is not None and col_label[adv_index][agent_id]:
                    adv_bv_col += 1

            avg_sdc_adv_col += sdc_adv_col
            avg_sdc_bv_col += sdc_bv_col
            avg_adv_bv_col += adv_bv_col
            avg_all_agent_cr += np.sum(np.triu(col_label, k=1)) / ooi_ind.shape[0]

            num_bv_agent += ooi_ind.shape[0] - 1  # only sdc no adv

        if num_bv_agent > 0:
            avg_sdc_bv_cr = avg_sdc_bv_col / (num_modes * num_bv_agent)
            avg_adv_bv_cr = avg_adv_bv_col / (num_modes * num_bv_agent)

        else:
            avg_sdc_bv_cr = None
            avg_adv_bv_cr = None

        avg_sdc_adv_cr = avg_sdc_adv_col / num_modes
        avg_all_agent_cr = avg_all_agent_cr / num_modes

        return avg_sdc_adv_cr, avg_sdc_bv_cr, avg_adv_bv_cr, avg_all_agent_cr


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from scenestreamer.utils import utils
    from scenestreamer.dataset.dataset import SceneStreamerDataset

    pl_model = utils.get_model(
        huggingface_repo="pengzhenghao97/scenestreamer_0301",
        huggingface_file="0228_MidGPT_V19_WTG_addstep_2025-02-28_epoch=14-step=426133.ckpt"
    )
    device = pl_model.device
    config = pl_model.config
    config.DATA.TRAINING_DATA_DIR = "data/20scenarios"
    config.PREPROCESSING.keep_all_data = True

    exp_name = "0307_arstep2_yuxin500"
    autoregressive_start_step = 2
    limit_test_batches = 5000000
    use_waymo = False
    # config.DATA.TEST_DATA_DIR = "data/20scenarios"
    # config.DATA.TEST_DATA_DIR = "/data/datasets/scenarionet/waymo/validation"
    config.DATA.TEST_DATA_DIR = "/bigdata/yuxin/scenarionet_waymo_training_500"

    num_modes = 6
    save_path = "{}_open_loop_results".format(exp_name)
    test_bs = 1

    tokenizer = pl_model.model.tokenizer
    evaluator = Evaluator(key_metrics_only=False, use_waymo=use_waymo)

    from scenestreamer.dataset.datamodule import SceneStreamerDataModule
    dataset = SceneStreamerDataset(config, "test")
    dataloader = DataLoader(dataset, batch_size=test_bs, collate_fn=lambda x: x[0])

    evaluation_module = EvaluationLightningModule(
        pl_model,
        evaluator,
        tokenizer,
        config,
        # dataset,
        num_modes=num_modes,
        save_path=save_path,
        autoregressive_start_step=autoregressive_start_step,
    )
    trainer = Trainer(limit_test_batches=limit_test_batches)

    # datamodule = SceneStreamerDataModule(
    #     config,
    #     train_batch_size=1,
    #     train_num_workers=0,
    #     train_prefetch_factor=0,
    #     val_batch_size=1,
    #     val_num_workers=0,
    #     val_prefetch_factor=0,
    # )
    # datamodule.setup("")
    # dataloader = datamodule.val_dataloader()

    trainer.validate(evaluation_module, dataloaders=dataloader)
