import numpy as np
import torch


def batch_nms(
    predicted_trajectories,
    predicted_scores,
    pred_to_scenario_id,
    dist_thresh,
    num_ret_modes=6,
    num_original_modes=6,
):
    """
    Copy from MTR. Modified to support our data.
    """
    ret_predicted_trajectories = []
    ret_predicted_scores = []
    B = len(predicted_trajectories)
    num_scenarios = B // num_original_modes
    assert num_scenarios * num_original_modes == B

    for sid in range(num_scenarios):
        assert len(np.unique(pred_to_scenario_id[sid * num_original_modes:(sid + 1) * num_original_modes])) == 1
        batch_traj = predicted_trajectories[sid * num_original_modes:(sid + 1) * num_original_modes]
        batch_scores = predicted_scores[sid * num_original_modes:(sid + 1) * num_original_modes]
        batch_traj = torch.stack(batch_traj, dim=1)
        batch_scores = torch.stack(batch_scores, dim=1)
        pred_scores = batch_scores

        # batch_traj shape is (T, num_modes, N, 2)
        batch_traj = batch_traj.permute(2, 1, 0, 3)
        # Now becomes (N, num_modes, T, 2)
        pred_trajs = batch_traj

        batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

        sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
        bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
        sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
        sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
        sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

        dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
        point_cover_mask = (dist < dist_thresh)

        point_val = sorted_pred_scores.clone()  # (batch_size, N)
        point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

        ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
        ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
        ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
        bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

        for k in range(num_ret_modes):
            cur_idx = point_val.argmax(dim=-1)  # (batch_size)
            ret_idxs[:, k] = cur_idx

            new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
            point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
            point_val_selected[bs_idxs, cur_idx] = -1
            point_val += point_val_selected

            ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
            ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

        ret_trajs = ret_trajs.permute(1, 2, 0, 3)  # (N, num_modes, T, 2) -> (num_modes, T, N, 2)
        ret_scores = ret_scores.permute(1, 0)  # (N, num_modes) -> (num_modes, N)

        ret_predicted_trajectories.extend(list(ret_trajs))
        ret_predicted_scores.extend(list(ret_scores))

    return ret_predicted_trajectories, ret_predicted_scores
