# Referenced from https://github.com/Tsinghua-MARS-Lab/InterSim/blob/main/simulator/proto.py

import copy
import dataclasses
import itertools

import hydra
import numpy as np
import omegaconf
import tensorflow as tf
import torch
import tqdm
from shapely.geometry import Polygon
from tqdm import tqdm
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features

from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour
from scenestreamer.infer.motion import generate_motion
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils
from scenestreamer.utils.utils import numpy_to_torch

try:
    from waymo_open_dataset.protos import motion_submission_pb2
except ModuleNotFoundError:
    motion_submission_pb2 = None


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


def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


def get_2D_collision_labels(data_dict, track_agent_indicies):
    # Now, instead of getting 1d-array of collision labels, let's do 2-d array to detect whether there is collision between given two agents.

    assert data_dict["decoder/agent_position"].ndim == 3

    N = data_dict["decoder/agent_position"].shape[1]

    safety_actions = np.zeros((N, N), dtype=bool)

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
                safety_actions[track_agent_indicies[i]][
                    track_agent_indicies[j]] = 1  # Label collisions for OOIs now. Later we will build a larger dict.
                safety_actions[track_agent_indicies[j]][
                    track_agent_indicies[i]] = 1  # Label collisions for OOIs now. Later we will build a larger dict.
    return safety_actions


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


class PengEvaluator:
    def __init__(self, config):
        self.config = config
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
        # self.key_metrics_only = key_metrics_only
        # self.use_waymo = use_waymo
        self.use_waymo = False

    def _call_model(self, model, expanded_data_dict):

        if self.config.MODEL.NAME == "scenestreamer":
            if not hasattr(self, "scenestreamer_generator"):
                from scenestreamer.infer.scenestreamer_generator import SceneStreamerGenerator
                self.scenestreamer_generator = SceneStreamerGenerator(
                    model=model,
                    device=expanded_data_dict["encoder/agent_feature"].device,
                )
            with torch.no_grad():
                self.scenestreamer_generator.reset(new_data_dict=expanded_data_dict)
                output_dict = self.scenestreamer_generator.generate_scenestreamer_motion(
                    teacher_forcing_sdc=False,
                )

        else:
            with torch.no_grad():
                output_dict = generate_motion(
                    data_dict=expanded_data_dict,
                    model=model,
                    autoregressive_start_step=2,
                    allow_newly_added_agent_step=2,
                )

        return output_dict

    def validation_step(self, data_dict, batch_idx, model, log_dict_func, **kwargs):
        # TODO: Pass this from config.
        num_decode_steps = 16

        num_modes_for_eval = self.config.EVALUATION.NUM_MODES
        maximum_batch_size = self.config.EVALUATION.MAXIMUM_BATCH_SIZE

        # assert num_modes_for_eval == 6


        if num_modes_for_eval <= maximum_batch_size:
            num_repeat_calls = 1
        else:
            assert num_modes_for_eval % maximum_batch_size == 0
            num_repeat_calls = num_modes_for_eval // maximum_batch_size



        B = data_dict["encoder/agent_feature"].shape[0]
        assert B == 1
        data_dict["batch_idx"] = torch.arange(B)


        if num_repeat_calls == 1:

            expanded_data_dict = {
                k: utils.repeat_for_modes(data_dict[k], num_modes=num_modes_for_eval)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("eval/") or k.startswith("decoder/") or k == "batch_idx" or k == "in_evaluation"
                    or k == "in_backward_prediction"
                )
            }

            output_dict = self._call_model(model, expanded_data_dict)

        else:

            assert B == 1, B
            num_modes_per_call = num_modes_for_eval // num_repeat_calls
            assert num_modes_per_call * num_repeat_calls == num_modes_for_eval
            expanded_data_dict = {
                k: utils.repeat_for_modes(data_dict[k], num_modes=num_modes_per_call)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("eval/") or k.startswith("decoder/") or k == "batch_idx" or k == "in_evaluation"
                    or k == "in_backward_prediction"
                )
            }
            output_dict = []
            for i in range(num_repeat_calls):
                expanded_data_dict["batch_idx"] = torch.arange(B) + i * maximum_batch_size
                output_dict.append(self._call_model(model, copy.deepcopy(expanded_data_dict)))
            output_dict = {
                k: (torch.cat([out[k] for out in output_dict], dim=0))
                if isinstance(output_dict[0][k], torch.Tensor) else None
                           for k in output_dict[0].keys()
            }
            output_dict.pop("batch_idx", None)

        MAX_MODES = 6
        if num_modes_for_eval > MAX_MODES:
            sort_scores = output_dict["decoder/output_score"].sum(-1).sort(descending=True)
            selected_indices = sort_scores.indices[:MAX_MODES]
            selected_scores = sort_scores.values[:MAX_MODES]
            selected_scores = selected_scores.to(output_dict["decoder/output_score"].device)
            output_dict = {
                k: v[selected_indices.to(v.device)] if isinstance(v, torch.Tensor) else v for k,v  in output_dict.items()
            }

        expanded_data_dict = utils.expand_for_modes(data_dict, num_modes=MAX_MODES)
        original_expanded = copy.deepcopy(expanded_data_dict)

        # log_dict_func(self.compute_collision_statistics(pred_data_dict=output_dict, num_modes=MAX_MODES))
        log_dict_func(self.compute_ade_fde_fdd(gt_data_dict=original_expanded, pred_data_dict=output_dict))


    def compute_collision_statistics(self, *, pred_data_dict, num_modes):
        # TODO: All agent?

        B, T, N, D = pred_data_dict["decoder/agent_position"].shape

        sdc_index = pred_data_dict["decoder/sdc_index"]

        agent_valid_mask = utils.torch_to_numpy(pred_data_dict["decoder/agent_valid_mask"].any(1))
        assert agent_valid_mask.shape == (B, N)

        # TODO: OOI index is also wrong. Can we use diff SDC / BV / OOI / ADV ???
        ooi_ind = pred_data_dict["decoder/object_of_interest_id"][0]

        all_ind = pred_data_dict["decoder/agent_id"][0]

        output_data_all_modes = utils.torch_to_numpy(pred_data_dict)
        output_data_all_modes = _overwrite_datadict_all_agents(
            source_data_dict=output_data_all_modes, dest_data_dict=output_data_all_modes
        )  # overwrite pred to GT

        sdc_ooi_cr = []
        sdc_bv_cr = []
        sdc_all_cr = []

        all_agent_cr = []

        for i in range(num_modes):
            output_dict_mode = _get_mode(output_data_all_modes, i, num_modes=num_modes)
            col_label = get_2D_collision_labels(data_dict=output_dict_mode, track_agent_indicies=all_ind)
            assert col_label.shape == (N, N)

            # TODO: I think this is wrong:
            sid = sdc_index[i]

            # TODO: THIS IS NOT FINISHED YET.

            sdc_bv_cr.append(sum([col_label[sid][agent_id] for agent_id in ooi_ind]))

            agent_has_coll = np.triu(col_label, k=1).astype(bool).any(-1)
            assert agent_has_coll.shape == (N, )
            all_agent_cr.append(utils.masked_average_numpy(agent_has_coll, agent_valid_mask[i], dim=0))

        return {
            "sdc_bv_cr": np.mean(sdc_bv_cr),
            "all_agent_cr": np.mean(all_agent_cr),
        }

    def compute_ade_fde_fdd(self, gt_data_dict, pred_data_dict):
        gt_valid = gt_data_dict["decoder/agent_valid_mask"]
        gt_valid_skipped = gt_data_dict["decoder/agent_valid_mask"][:, ::5]

        gt_pos = gt_data_dict["decoder/agent_position"][:, :91, :, :2]
        pred_pos = pred_data_dict["decoder/reconstructed_position"][:, :91, :, :2]
        pred_pos_skipped = pred_data_dict["decoder/reconstructed_position"][:, 0:91:5, :, :2]

        ooi_ind = gt_data_dict["decoder/object_of_interest_id"]
        sdc_ind = gt_data_dict["decoder/sdc_index"]

        ooi_and_sdc_ind = torch.cat([ooi_ind, sdc_ind[:, None]], dim=1)

        B, T, N, _ = gt_pos.shape

        assert gt_valid.ndim == 3

        # last_valid_ind = gt_valid.cumsum(dim=1).argmax(dim=1)
        last_valid_ind_skipped = gt_valid_skipped.cumsum(dim=1).argmax(dim=1)

        error = torch.linalg.norm(gt_pos - pred_pos, dim=-1)
        assert error.ndim == 3
        assert error.shape[0] == B

        # last_valid_ind = last_valid_ind.reshape(B, 1, N)
        last_valid_ind_skipped = last_valid_ind_skipped.reshape(B, 1, N)
        # assert last_valid_ind.shape == (B, 1, N)

        # fde = torch.gather(error, 1, last_valid_ind).squeeze(1)  # shape: B, N
        fde_skipped = torch.gather(error, 1, last_valid_ind_skipped * 5).squeeze(1)  # shape: B, N
        # assert fde.shape[0] == B

        # agent_valid = gt_valid.any(1).expand(B, N)
        agent_valid_skipped = gt_valid_skipped.any(1).expand(B, N)

        # assert fde.shape == agent_valid.shape == (B, N)
        assert fde_skipped.shape == agent_valid_skipped.shape == (B, N)

        # Set of OOI+SDC to True, and exclude other in agent_valid:
        # agent_valid_ooi = torch.zeros_like(agent_valid, dtype=torch.bool)
        batch_indices = torch.arange(B).unsqueeze(1)
        # agent_valid_ooi[batch_indices, ooi_and_sdc_ind] = True
        # agent_valid_ooi = agent_valid_ooi & agent_valid

        agent_valid_ooi_skipped = torch.zeros_like(agent_valid_skipped, dtype=torch.bool)
        agent_valid_ooi_skipped[batch_indices, ooi_and_sdc_ind] = True
        agent_valid_ooi_skipped = agent_valid_ooi_skipped & agent_valid_skipped

        # sfde_full_all = utils.masked_average(fde, agent_valid, dim=1)
        sfde_skipped_all = utils.masked_average(fde_skipped, agent_valid_skipped, dim=1)

        # sfde_full_ooisdc = utils.masked_average(fde, agent_valid_ooi, dim=1)
        sfde_skipped_ooisdc = utils.masked_average(fde_skipped, agent_valid_ooi_skipped, dim=1)

        # sade_per_agent = utils.masked_average(error, gt_valid, dim=1)
        sade_per_agent_skipped = utils.masked_average(error[:, ::5], gt_valid_skipped, dim=1)
        # assert sade_per_agent.shape == (B, N)

        # sade_full_all = utils.masked_average(sade_per_agent, agent_valid, dim=1)
        sade_skipped_all = utils.masked_average(sade_per_agent_skipped, agent_valid_skipped, dim=1)

        # sade_full_ooisdc = utils.masked_average(sade_per_agent, agent_valid_ooi, dim=1)
        sade_skipped_ooisdc = utils.masked_average(sade_per_agent_skipped, agent_valid_ooi_skipped, dim=1)

        # assert sfde_full_all.shape == sfde_skipped_all.shape == sfde_full_ooisdc.shape == sfde_skipped_ooisdc.shape == (B,)
        # assert sade_full_all.shape == sade_skipped_all.shape == sade_full_ooisdc.shape == sade_skipped_ooisdc.shape == (B,)

        # there doesn't appear to be an easy way to do this with cartesian product
        # final_pos = torch.gather(pred_pos, 1, last_valid_ind[..., None].expand(-1, -1, -1, 2)).squeeze(1)
        final_pos_skipped = torch.gather(pred_pos, 1, last_valid_ind_skipped[..., None].expand(-1, -1, -1,
                                                                                               2)).squeeze(1)

        # assert final_pos.shape == final_pos_skipped.shape == (B, N, 2)

        def _fdd(pos):
            assert pos.shape == (B, N, 2), pos.shape
            pos_NB = pos.swapaxes(0, 1)
            dist_NBB = torch.cdist(pos_NB, pos_NB)
            max_dist_N = dist_NBB.amax((1, 2))
            assert max_dist_N.shape == (N, )
            return max_dist_N

        # fdd_full_all = utils.masked_average(_fdd(final_pos), agent_valid[0], dim=0)
        fdd_skipped_all = utils.masked_average(_fdd(final_pos_skipped), agent_valid_skipped[0], dim=0)
        # fdd_full_ooisdc = utils.masked_average(_fdd(final_pos), agent_valid_ooi[0], dim=0)
        fdd_skipped_ooisdc = utils.masked_average(_fdd(final_pos_skipped), agent_valid_ooi_skipped[0], dim=0)

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
            _add(pred_pos_skipped, gt_valid_skipped[0]), agent_valid_skipped[0], dim=0
        )
        # add_full_ooisdc = utils.masked_average(_add(pred_pos, gt_valid[0]), agent_valid_ooi[0], dim=0)
        add_skipped_ooisdc = utils.masked_average(
            _add(pred_pos_skipped, gt_valid_skipped[0]), agent_valid_ooi_skipped[0], dim=0
        )

        return {
            # FDE
            # "sfde_full_all": sfde_full_all,
            # "sfde_skipped_all": sfde_skipped_all,
            "sfde_all_avg": sfde_skipped_all.mean(),
            "sfde_all_min": sfde_skipped_all.min(),
            # "sfde_full_ooisdc": sfde_full_ooisdc,
            # "sfde_skipped_ooisdc": sfde_skipped_ooisdc,
            "sfde_ooisdc_avg": sfde_skipped_ooisdc.mean(),
            "sfde_ooisdc_min": sfde_skipped_ooisdc.min(),
            # ADE
            # "sade_full_all": sade_full_all,
            # "sade_skipped_all": sade_skipped_all,
            "sade_all_avg": sade_skipped_all.mean(),
            "sade_all_min": sade_skipped_all.min(),
            # "sade_full_ooisdc": sade_full_ooisdc,
            # "sade_skipped_ooisdc": sade_skipped_ooisdc,
            "sade_ooisdc_avg": sade_skipped_ooisdc.mean(),
            "sade_ooisdc_min": sade_skipped_ooisdc.min(),
            # FDDD
            # "fdd_full_all": fdd_full_all,
            "fdd_all": fdd_skipped_all,
            # "fdd_full_ooisdc": fdd_full_ooisdc,
            "fdd_ooisdc": fdd_skipped_ooisdc,
            # ADD
            # "add_full_all": add_full_all,
            "add_all": add_skipped_all,
            # "add_full_ooisdc": add_full_ooisdc,
            "add_ooisdc": add_skipped_ooisdc,

            "num_all_agents": agent_valid_skipped.sum(-1).float().mean(),
            "num_ooisdc_agents": agent_valid_ooi_skipped.sum(-1).float().mean(),
        }

    def add(self, gt_expanded_data_dict):
        self.metrics.scenario_count += 1

        # T_gt = gt_data_dict["decoder/agent_position"].shape[0]
        # T_context = 0
        #
        # T_pred = pred_data_dict["decoder/reconstructed_position"].shape[1]
        # B = K = pred_data_dict["decoder/reconstructed_position"].shape[0]
        # N = gt_data_dict["decoder/agent_position"].shape[1]

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
        # there doesn't appear to be an easy way to do this with cartesian product
        cur_FDD = None
        pred_ooi_valid_mask = pred_valid_mask[:, ooi_mask]
        single_mode_ooi_valid_mask = pred_ooi_valid_mask[0]

        # assert torch.all(torch.any(pred_ooi_valid_mask, dim=-1))
        last_valid_ind = pred_ooi_valid_mask.cumsum(dim=-1).argmax(dim=-1)  # (K, N)
        ooi_reconstructed_pos = pred_data_dict["decoder/reconstructed_position"][:, :, ooi_mask]  # (K, T_pred, N, 2)
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

        speed_jsd = jsd(gt_speed_hist, pred_speed_hist)
        acc_jsd = jsd(gt_accel_hist, pred_accel_hist)
        self.metrics.vel_jsd += speed_jsd
        self.metrics.acc_jsd += acc_jsd
        if self.use_waymo:
            ttc_jsd = jsd(gt_ttc_hist, pred_ttc_hist)
            self.metrics.ttc_jsd += ttc_jsd

    def on_validation_epoch_end(self, *args, **kwargs):
        pass

    # def on_validation_epoch_end(
    #     self, trainer, logger, global_rank, log_dict_func, log_func, print_func, exp_name, **kwargs
    # ):
    #     """
    #     This function gathers intermediate evaluation result and pass them to the Waymo
    #     evaluation pipeline together and log the final results.
    #     """
    #     st = time.time()
    #
    #     # print(debug_tools.using(f"val epoch end start"))
    #
    #     # https://lightning.ai/docs/pytorch/latest/accelerators/accelerator_prepare.html?highlight=hardware
    #     # torch.cuda.empty_cache()
    #     # PZH NOTE: Hack to implement our own all_gather across ranks.
    #     trainer.strategy.barrier()
    #
    #     # Collect the intermediate evaluation results from each call to on_validation_step in this particular rank.
    #     self.validation_outputs = [
    #         {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
    #          for k, v in final_pred_dicts.items()} for final_pred_dicts in self.validation_outputs
    #     ]
    #
    #     # Dump all results in this rank to a local file so that later the rank0 process can read them.
    #     tmpdir = self.config.ROOT_DIR / self.config.TMP_DIR / "validation_tmpdir_{}".format(exp_name)
    #     print(f"Rank {global_rank} saving validation results to {tmpdir}.")
    #
    #     os.makedirs(tmpdir, exist_ok=True)
    #     with open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(global_rank)), 'wb') as f:
    #         pickle.dump(self.validation_outputs, f)
    #     self.validation_outputs.clear()
    #
    #     # print(debug_tools.using(f"val epoch saved file."))
    #
    #     # If this is the main process (rank0), read all results in local filesystem and call evaluation pipeline.
    #     torch.cuda.empty_cache()
    #     trainer.strategy.barrier()
    #     if trainer.is_global_zero:
    #         print_func(f"===== Start evaluation: {time.time() - st:.3f} =====")
    #
    #         # Gather results from different ranks
    #         validation_list = []
    #         for i in range(trainer.world_size):
    #             file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
    #             success = False
    #             for sleep in range(10):
    #                 if not os.path.isfile(file):
    #                     time.sleep(1)
    #                     print(f"Can't find file: {file}. Sleep {sleep}/{10} seconds.")
    #                 else:
    #                     success = True
    #                     break
    #             if not success:
    #                 print(f"[WARNING] Can't find file: {file}. Skip this rank.")
    #                 continue
    #             with open(file, "rb") as f:
    #                 val_outputs = pickle.load(f)
    #                 validation_list.extend(val_outputs)
    #         if self.config.EVALUATION.DELETE_EVAL_RESULT:
    #             shutil.rmtree(tmpdir)
    #
    #         if not validation_list:
    #             print_func("No evaluation results found. Skip evaluation.")
    #             return
    #
    #         # print(debug_tools.using(f"going to eval"))
    #
    #         # Call evaluation pipeline
    #         torch.cuda.empty_cache()
    #         result_dict, result_str, submission_dict = waymo_evaluation_optimized(
    #             validation_list,
    #
    #             # TODO: This flag
    #             generate_submission=self.config.SUBMISSION.GENERATE_SUBMISSION,
    #             predict_all_agents=self.config.EVALUATION.PREDICT_ALL_AGENTS,
    #         )
    #         torch.cuda.empty_cache()
    #         validation_list.clear()
    #
    #         # Log result
    #         result_dict = {f"eval/{k}": float(v) for k, v in result_dict.items()}
    #         log_dict_func(result_dict, rank_zero_only=True)
    #         for k in ['eval/minADE', 'eval/minFDE', 'eval/MissRate', 'eval/mAP', "eval/mJADE", "eval/avgJADE",
    #                   "eval/mJFDE", "eval/avgJFDE"]:
    #             if k not in result_dict:
    #                 continue
    #             log_func(name=k.split("/")[1], value=result_dict[k], rank_zero_only=True)
    #         print_func(result_str)
    #         print_func(f"===== Finish evaluation: {time.time() - st:.3f} =====")
    #
    #     print_func(f"Rank {global_rank} finished evaluation!")
    #     torch.cuda.empty_cache()
    #     trainer.strategy.barrier()
    #
    #     # TODO This flag
    #     if trainer.is_global_zero and self.config.SUBMISSION.GENERATE_SUBMISSION:
    #         account_name = self.config.SUBMISSION.ACCOUNT
    #         unique_method_name = self.config.SUBMISSION.METHOD_NAME
    #         output_dir = logger.log_dir
    #         submission_prefix = logger.name
    #         path, duplicated_scenarios, done_scenarios = generate_submission(
    #             prefix=submission_prefix,
    #             account_name=account_name,
    #             unique_method_name=unique_method_name,
    #             output_dir=output_dir,
    #             **submission_dict
    #         )
    #         print_func(
    #             "Submission created at: {}. Finished {} scenarios. Duplicated scenarios: {}.".format(
    #                 path, len(done_scenarios), duplicated_scenarios
    #             )
    #         )


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="0220_midgpt.yaml")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "/home/zhenghao/scenestreamer/lightning_logs/scenestreamer/0220_MidGPT_V19_2025-02-20/"
    pl_model = utils.get_model(checkpoint_path=path, device=device)

    # model = utils.get_model(config, device=device)
    # evaluator = TrafficGenEvaluator(config)

    config = pl_model.config
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING["keep_all_data"] = True
    config.DATA.TRAINING_DATA_DIR = "data/20scenarios"
    config.DATA.TEST_DATA_DIR = "data/20scenarios"

    test_dataset = SceneStreamerDataset(config, "training")
    # ddd = iter(test_dataset)

    START_ACTION = config.PREPROCESSING.MAX_MAP_FEATURES
    END_ACTION = config.PREPROCESSING.MAX_MAP_FEATURES + 1

    for count, raw_data_dict in enumerate(tqdm.tqdm(test_dataset)):
        data_dict = raw_data_dict
        data_dict = utils.numpy_to_torch(data_dict, device=device)
        batched_data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        data_dict = generate_motion(
            data_dict=batched_data_dict,
            model=pl_model.model,
            autoregressive_start_step=2,
            remove_out_of_map_agent=True,
            remove_static_agent=True,
            teacher_forcing_sdc=True,
        )

        data_dict = utils.unbatch_data(utils.torch_to_numpy(data_dict))
        from scenestreamer.gradio_ui.plot import plot_pred
        plot_pred(data_dict, show=True)

    print("End")


if __name__ == '__main__':
    main()
