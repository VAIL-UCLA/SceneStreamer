import copy
import datetime
import json
import logging
import numbers
import os
import pathlib
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import IO, Optional, Type, Union

import easydict
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()  # Root to the repo


def average_heading(heading1, heading2):
    if isinstance(heading1, np.ndarray):
        # Convert headings to unit vectors
        x1, y1 = np.cos(heading1), np.sin(heading1)
        x2, y2 = np.cos(heading2), np.sin(heading2)

        # Compute average vector
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2

        # Compute the angle of the average vector
        return np.arctan2(avg_y, avg_x)
    elif isinstance(heading1, torch.Tensor):
        # Convert headings to unit vectors
        x1, y1 = torch.cos(heading1), torch.sin(heading1)
        x2, y2 = torch.cos(heading2), torch.sin(heading2)

        # Compute average vector
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2

        # Compute the angle of the average vector
        return torch.atan2(avg_y, avg_x)
    else:
        raise ValueError("Input must be a NumPy array or PyTorch tensor")


def average_angles(angles):
    # Convert angles to Cartesian coordinates
    sum_sin = np.mean(np.sin(angles))
    sum_cos = np.mean(np.cos(angles))

    # Convert the average coordinates back to angles (in radians)
    avg_angle_rad = np.arctan2(sum_sin, sum_cos)
    return avg_angle_rad


def get_time_str(no_time=False):
    if no_time:
        return datetime.datetime.now().strftime("%Y-%m-%d")
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


def assert_shape(array, shape):
    assert array.ndim == len(shape)
    for i in range(array.ndim):
        if shape[i] is not None:
            assert array.shape[i] == shape[i]


def padding_2nd_dim(tensor_list):
    maxt_feat1 = max([x.shape[1] for x in tensor_list])
    ret_tensor_list = []
    for cur_tensor in tensor_list:
        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat1, *cur_tensor.shape[2:])
        new_tensor[:, :cur_tensor.shape[1]] = cur_tensor
        ret_tensor_list.append(new_tensor)
    return torch.stack(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)


def padding_1st_dim(tensor_list, fill=None, max_1st_dim=None):
    if max_1st_dim is None:
        max_feat0 = max([x.shape[0] for x in tensor_list])
    else:
        max_feat0 = max_1st_dim
    ret_tensor_list = []
    for cur_tensor in tensor_list:
        new_tensor = cur_tensor.new_zeros(max_feat0, *cur_tensor.shape[1:])
        if fill is not None:
            new_tensor.fill_(fill)
        new_tensor[:cur_tensor.shape[0]] = cur_tensor
        ret_tensor_list.append(new_tensor)
    return torch.stack(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)


def padding_1st_and_2nd_dim(tensor_list, max_1st_dim=None, max_2nd_dim=None, fill=None):
    maxt_feat1 = max([x.shape[1] for x in tensor_list]) if max_2nd_dim is None else max_2nd_dim
    maxt_feat0 = max([x.shape[0] for x in tensor_list]) if max_1st_dim is None else max_1st_dim
    ret_tensor_list = []
    for cur_tensor in tensor_list:
        new_tensor = cur_tensor.new_zeros(maxt_feat0, maxt_feat1, *cur_tensor.shape[2:])
        if fill is not None:
            new_tensor.fill_(fill)
        new_tensor[:cur_tensor.shape[0], :cur_tensor.shape[1]] = cur_tensor
        ret_tensor_list.append(new_tensor)
    return torch.stack(ret_tensor_list, dim=0)


def padding_all_dims(tensor_list, max_dims=None, fill=None):
    """
    Pad each tensor in tensor_list to have the same shape in all dimensions.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be padded.
        max_dims (sequence of int, optional): Desired maximum sizes for each dimension.
            If not provided, the function computes the maximum size along each dimension across tensor_list.
        fill (scalar, optional): Value to fill the padded areas with.
            If None, the padded areas are filled with zeros.

    Returns:
        torch.Tensor: A single tensor of shape (N, *max_dims) where N is the number of tensors,
                      with each tensor padded to the same shape.
    """
    # Determine the number of dimensions from the first tensor
    ndim = tensor_list[0].dim()

    if tensor_list[0].dtype == torch.bool:
        fill = False

    # Compute maximum dimensions if not provided
    if max_dims is None:
        max_dims = [max(t.shape[d] for t in tensor_list) for d in range(ndim)]
    else:
        max_dims = list(max_dims)

    padded_list = []
    for t in tensor_list:
        # Create a new tensor of the computed target shape with the same type and device as t
        new_tensor = t.new_zeros(*max_dims)
        if fill is not None:
            new_tensor.fill_(fill)
        # Create a slice object for each dimension to copy over the data from t
        slices = tuple(slice(0, t.shape[d]) for d in range(ndim))
        new_tensor[slices] = t
        padded_list.append(new_tensor)

    # Stack along a new first dimension
    return torch.stack(padded_list, dim=0)


def get_distribution(dist_parameter):
    weight = dist_parameter[..., 0].clamp(-100, 100)

    log_std_range = (-1.609, 5.0)
    para = dist_parameter[..., 1:].clamp(-100, 100)

    if para.shape[-1] == 5:
        loc, tril, diag = para[..., :2], para[..., 2], para[..., 3:]
        sigma_1 = torch.exp(diag[..., 0].clamp(log_std_range[0], log_std_range[1]))
        sigma_2 = torch.exp(diag[..., 1].clamp(log_std_range[0], log_std_range[1]))
        rho = torch.tanh(tril).clamp(-0.5, 0.5)
        cov = torch.stack([sigma_1**2, rho * sigma_1 * sigma_2, rho * sigma_1 * sigma_2, sigma_2**2],
                          dim=-1).view(*loc.shape[:-1], 2, 2)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, covariance_matrix=cov)
        pos_weight = torch.distributions.Categorical(logits=weight)
        dist = torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, dist)

    elif para.shape[-1] == 6:
        loc, log_scale = para[..., :3], para[..., 3:]
        scale = torch.exp(log_scale.clamp(log_std_range[0], log_std_range[1]))
        dist = torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1
        )
        pos_weight = torch.distributions.Categorical(logits=weight)
        dist = torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, dist)

    elif para.shape[-1] == 2:
        loc, scale = para[..., 0], para[..., 1]
        scale = torch.exp(scale.clamp(log_std_range[0], log_std_range[1]))
        dist = torch.distributions.Normal(loc, scale)
        pos_weight = torch.distributions.Categorical(logits=weight)
        dist = torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, dist)

    else:
        raise ValueError(para.shape)

    return dist


def unwrap(flatten_array, valid_mask, existing=None, fill=None):
    assert valid_mask.sum() == flatten_array.shape[0]
    if existing is None:
        ret = flatten_array.new_zeros(valid_mask.shape + (flatten_array.shape[-1], ))
    else:
        ret = existing
    if fill is not None:
        ret.fill_(fill)
    ret[valid_mask] = flatten_array
    return ret


def pack_sequences(seqs) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def wrap_to_pi(radians_array):
    """
    Wrap all input radians to range [-pi, pi]
    """
    if isinstance(radians_array, np.ndarray):
        wrapped_radians_array = np.mod(radians_array, 2 * np.pi)
        wrapped_radians_array[wrapped_radians_array > np.pi] -= 2 * np.pi
    elif isinstance(radians_array, torch.Tensor):
        wrapped_radians_array = radians_array % (2 * torch.tensor(np.pi))
        wrapped_radians_array[wrapped_radians_array > torch.tensor(np.pi)] -= 2 * np.pi
    elif isinstance(radians_array, (float, np.float32)):
        wrapped_radians_array = radians_array % (2 * np.pi)
        if wrapped_radians_array > np.pi:
            wrapped_radians_array -= 2 * np.pi
    else:
        raise ValueError("Input must be a NumPy array or PyTorch tensor")

    return wrapped_radians_array


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x), True  # .float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2)  # .float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
                                 dim=1).view(-1, 3, 3)  # .float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def absolute_to_relative(abs_pos, map_head):
    relative_y = map_head
    relative_x = relative_y - np.pi / 2
    object_heading_to_rotate = relative_x
    if abs_pos.shape[-1] == 3:
        z = abs_pos[..., 2]
    else:
        z = None
    rel_pos = rotate(abs_pos[..., 0], abs_pos[..., 1], -object_heading_to_rotate.squeeze(-1), z=z)
    return rel_pos


def relative_to_absolute(rel_pos, map_head):
    relative_y = map_head
    relative_x = relative_y - np.pi / 2
    object_heading_to_rotate = relative_x
    if rel_pos.shape[-1] == 3:
        z = rel_pos[..., 2]
    else:
        z = None
    abs_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], object_heading_to_rotate.squeeze(-1), z=z)
    return abs_pos


def rotate(x, y, angle, z=None, assert_shape=True):
    # TODO(pzh): Repeat function, remove one.
    if assert_shape:
        assert angle.shape == x.shape == y.shape, (angle.shape, x.shape, y.shape)
        if z is not None:
            assert x.shape == z.shape
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        if z is None:
            output_coords = torch.stack((other_x_trans, other_y_trans), dim=-1)
        else:
            output_coords = torch.stack((other_x_trans, other_y_trans, z), dim=-1)
    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        if z is None:
            output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        else:
            output_coords = np.stack((other_x_trans, other_y_trans, z), axis=-1)
    return output_coords


# def translate_pos_to_ego_centric(xyz, center, heading):
#     assert center.shape[-1] == 3
#     assert heading.shape[-1] == 1
#     # assert xyz.shape[0] == center.shape[0] == heading.shape[0]
#     assert xyz.ndim == 3
#     assert center.ndim == 2
#     assert heading.ndim == 2
#     xyz = xyz - center[:, None]
#     xyz = rotate_points_along_z(xyz, -heading)
#     return xyz


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def get_batch_offsets(batch_idxs, bs, device):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros([
        bs + 1,
    ], device=device).int()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def calculate_trajectory_probabilities(logits, sampled_actions, mask):
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Remove invalid actions:
    invalid_action_mask = torch.logical_or(sampled_actions < 0, sampled_actions >= probs.shape[-1])
    assert (logits[invalid_action_mask] == 0).all()
    sampled_actions = sampled_actions.masked_fill(invalid_action_mask, 0)

    # Gather the probabilities of the sampled actions
    gathered_probs = torch.gather(probs, -1, sampled_actions.unsqueeze(-1)).squeeze(-1)

    gathered_probs = gathered_probs.masked_fill(invalid_action_mask, 0)

    # Multiply probabilities across the time dimension for each trajectory
    # Use log probabilities for numerical stability
    log_probs = torch.log(gathered_probs)
    trajectory_log_probs = torch.sum(log_probs, dim=1)

    # Convert back from log probabilities if needed
    trajectory_probs = torch.exp(trajectory_log_probs)

    # Aggregate to get final shape (B, N)
    mask = mask.reshape(trajectory_probs.shape)
    trajectory_probs = trajectory_probs.masked_fill(~mask, float("-inf"))
    return trajectory_probs


def calculate_trajectory_probabilities_new(logits, sampled_actions, mask):
    """
    Compared to the old version, we allow the mask to have temporal dimension.
    """
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Remove invalid actions:
    invalid_action_mask = torch.logical_or(sampled_actions < 0, sampled_actions >= probs.shape[-1])
    assert (logits[invalid_action_mask] == 0).all()
    sampled_actions = sampled_actions.masked_fill(invalid_action_mask, 0)

    # Gather the probabilities of the sampled actions
    gathered_probs = torch.gather(probs, -1, sampled_actions.unsqueeze(-1)).squeeze(-1)

    gathered_probs = gathered_probs.masked_fill(invalid_action_mask, 0)

    # Multiply probabilities across the time dimension for each trajectory
    # Use log probabilities for numerical stability
    log_probs = torch.log(gathered_probs)

    log_probs = log_probs.masked_fill(~mask, 0)

    trajectory_log_probs = torch.sum(log_probs, dim=1) / mask.sum(dim=1)

    # Convert back from log probabilities if needed
    trajectory_probs = torch.exp(trajectory_log_probs)
    trajectory_probs = trajectory_probs.masked_fill(~mask.any(dim=1), 0)

    trajectory_log_probs = trajectory_log_probs.masked_fill(~mask.any(dim=1), float("-inf"))
    return trajectory_log_probs, trajectory_probs


def masked_average(tensor, mask, dim):
    """
    Compute the average of tensor along the specified dimension, ignoring masked elements.
    """
    assert tensor.shape == mask.shape
    count = mask.sum(dim=dim)
    count = torch.max(count, torch.ones_like(count))
    return (tensor * mask).sum(dim=dim) / count


def weight_init(m: nn.Module) -> None:
    raise ValueError
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out))**0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out))**0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def masked_average_numpy(tensor, mask, dim):
    """
    Compute the average of tensor along the specified dimension, ignoring masked elements.
    """
    assert tensor.shape == mask.shape
    count = mask.sum(axis=dim)
    count = np.maximum(count, np.ones_like(count))
    return (tensor * mask).sum(axis=dim) / count


def extract_data_by_agent_indices(data, agent_indices, agent_dim, fill=None):
    agent_indices = np.asarray(agent_indices, dtype=int)
    new_shape = [
        1,
    ] * data.ndim
    new_shape[agent_dim] = agent_indices.shape[0]
    agent_indices = agent_indices.reshape(*new_shape)
    data = np.take_along_axis(data, agent_indices, axis=agent_dim)
    data = np.where(agent_indices != -1, data, np.zeros_like(data))
    if fill is not None:
        data[agent_indices == -1] = fill
    return data


def average_angles(angles):
    # Convert angles to Cartesian coordinates
    sum_sin = np.mean(np.sin(angles))
    sum_cos = np.mean(np.cos(angles))

    # Convert the average coordinates back to angles (in radians)
    avg_angle_rad = np.arctan2(sum_sin, sum_cos)
    return avg_angle_rad


def masked_average_angles(angles, mask, axis):
    assert angles.shape == mask.shape

    # Convert masked angles to Cartesian coordinates
    sum_sin = np.sum(np.sin(angles) * mask, axis=axis)
    sum_cos = np.sum(np.cos(angles) * mask, axis=axis)

    # Calculate the number of valid entries along the specified axis
    count_valid = np.sum(mask, axis=axis)

    # Compute the mean of the sine and cosine, avoiding division by zero
    mean_sin = np.divide(sum_sin, count_valid, where=count_valid != 0)
    mean_cos = np.divide(sum_cos, count_valid, where=count_valid != 0)

    # Convert the average coordinates back to angles (in radians)
    avg_angle_rad = np.arctan2(mean_sin, mean_cos)
    return avg_angle_rad


def modulate(x, shift, scale):
    assert x.shape == shift.shape == scale.shape
    return x * (1 + scale) + shift


def _to_dict(d):
    if isinstance(d, easydict.EasyDict):
        return {k: _to_dict(v) for k, v in d.items()}
    return d


def load_from_checkpoint(
    cls: Union[Type["pl.LightningModule"], Type["pl.LightningDataModule"]],
    checkpoint_path: Union[_PATH, IO],
    config,
    default_config=None,
    map_location: _MAP_LOCATION_TYPE = None,
    hparams_file: Optional[_PATH] = None,
    strict: Optional[bool] = None,
    checkpoint_surgery_func=None,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
    if checkpoint_path is None:
        if default_config is not None:
            # Merge config and default config
            if isinstance(default_config, easydict.EasyDict):
                default_config = _to_dict(default_config)
            default_config = OmegaConf.create(default_config)
            config = OmegaConf.merge(default_config, config)
        return cls(config)

    with pl_legacy_patch():
        # PyTorch >= 2.6 defaults torch.load(weights_only=True), which can fail for Lightning checkpoints
        # that include non-tensor objects (e.g. pathlib.PosixPath). We explicitly request weights_only=False.
        try:
            checkpoint = pl_load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older lightning versions may not support weights_only in pl_load.
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        except RuntimeError as e:
            # Common when loading CUDA-saved checkpoints on CPU-only machines (e.g. macOS):
            # retry with CPU mapping if caller didn't request a specific map_location.
            msg = str(e)
            if map_location is None and (
                "Attempting to deserialize object on a CUDA device" in msg or "torch.cuda.is_available() is False" in msg
            ):
                try:
                    checkpoint = pl_load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
                except TypeError:
                    checkpoint = pl_load(checkpoint_path, map_location=torch.device("cpu"))
            else:
                raise

    # convert legacy checkpoints to the new format
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(checkpoint_path if isinstance(checkpoint_path, (str, Path)) else None)
    )

    from lightning.pytorch.core.saving import load_hparams_from_yaml, load_hparams_from_tags_csv

    if hparams_file is not None:
        extension = str(hparams_file).split(".")[-1]
        if extension.lower() == "csv":
            hparams = load_hparams_from_tags_csv(hparams_file)
        elif extension.lower() in ("yml", "yaml"):
            hparams = load_hparams_from_yaml(hparams_file)
        else:
            raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

        # overwrite hparams by the given file
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

    # TODO: make this a migration:
    # for past checkpoint need to add the new key
    checkpoint.setdefault(cls.CHECKPOINT_HYPER_PARAMS_KEY, {})
    # override the hparams with values that were passed in

    # PZH: Change
    # checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(config)
    if isinstance(default_config, omegaconf.DictConfig):
        default_config = OmegaConf.to_container(default_config)
    if default_config:
        default_config = copy.deepcopy(default_config)
        if "LOCAL_RANK" in default_config:
            default_config.pop("LOCAL_RANK")

        if "defaults" in default_config:
            default_config.pop("defaults")

        if "config" in checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]:

            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]["config"]
            if isinstance(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], easydict.EasyDict):

                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = _to_dict(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])

                if "ROOT_DIR" in checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]:
                    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]["ROOT_DIR"] = str(REPO_ROOT)
                if "SAMPLE_INTERVAL" in checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]["DATA"]:
                    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]["DATA"].pop("SAMPLE_INTERVAL")
                if "LOCAL_RANK" in checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]:
                    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].pop("LOCAL_RANK")

        if isinstance(default_config, easydict.EasyDict):
            default_config = _to_dict(default_config)
            if "ROOT_DIR" in default_config:
                default_config["ROOT_DIR"] = str(REPO_ROOT)
        default_config = OmegaConf.merge(default_config, checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = default_config

    if checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]:
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = OmegaConf.create(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY])

    if config:
        # if "SAMPLE_INTERVAL" in config.DATA:
        #     config.DATA.pop("SAMPLE_INTERVAL")
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY
                   ] = OmegaConf.merge(checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], config)

    config = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
    config = normalize_legacy_model_config(config, checkpoint_path=checkpoint_path)
    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = config

    from lightning.pytorch.core.saving import _load_state

    if checkpoint_surgery_func is not None:
        checkpoint = checkpoint_surgery_func(checkpoint, cls, config)

    if issubclass(cls, pl.LightningDataModule):
        return _load_state(cls, checkpoint, **config)
    if issubclass(cls, pl.LightningModule):
        storage = _load_state(cls, checkpoint, strict=strict, config=config)
        state_dict = checkpoint["state_dict"]
        if not state_dict:
            raise ValueError(f"The state dict in {checkpoint_path!r} contains no parameters.")
        map_location = list(state_dict.values())[0].device
        assert isinstance(storage, pl.LightningModule)
        return storage.to(map_location)

    raise NotImplementedError(f"Unsupported {cls}")


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1
    )

    return polygon_contour


def cal_polygon_contour_torch(x, y, theta, width, length):
    # Calculate corner points using torch operations
    left_front_x = x + 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
    left_front_y = y + 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
    left_front = torch.stack((left_front_x, left_front_y), dim=-1)

    right_front_x = x + 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
    right_front_y = y + 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
    right_front = torch.stack((right_front_x, right_front_y), dim=-1)

    right_back_x = x - 0.5 * length * torch.cos(theta) + 0.5 * width * torch.sin(theta)
    right_back_y = y - 0.5 * length * torch.sin(theta) - 0.5 * width * torch.cos(theta)
    right_back = torch.stack((right_back_x, right_back_y), dim=-1)

    left_back_x = x - 0.5 * length * torch.cos(theta) - 0.5 * width * torch.sin(theta)
    left_back_y = y - 0.5 * length * torch.sin(theta) + 0.5 * width * torch.cos(theta)
    left_back = torch.stack((left_back_x, left_back_y), dim=-1)

    # Stack all corner points into the desired shape (N, 4, 2)
    polygon_contour = torch.stack((left_front, right_front, right_back, left_back), dim=-2)

    return polygon_contour


def checkpoint_surgery_func(checkpoint, model_class, config):
    m = None

    # Update 2025-05-05:
    if config.MODEL.NAME == "scenestreamer" and config.get("SCENESTREAMER_NO_TG", False) is False:
        if "model.trafficgen_head.offset_token_embedding.tokens.weight" in checkpoint["state_dict"]:
            if checkpoint["state_dict"]["model.trafficgen_head.offset_token_embedding.tokens.weight"].shape[0] == 81:
                if m is None:
                    m = model_class(config)
                default_params = m.state_dict()
                new_weight = default_params["model.trafficgen_head.offset_token_embedding.tokens.weight"]
                # Remove the 81 (-1 id)
                new_weight[:81] = checkpoint["state_dict"]["model.trafficgen_head.offset_token_embedding.tokens.weight"]
                checkpoint["state_dict"]["model.trafficgen_head.offset_token_embedding.tokens.weight"] = new_weight
                print("====================================")
                print("[WARNING] The trafficgen_head.offset_token_embedding.tokens is not found in the checkpoint.")
                print("Using initial parameters for trafficgen_head.offset_token_embedding.tokens.")
                print("Writing: ", "model.trafficgen_head.offset_token_embedding.tokens.weight", new_weight.shape)
                print("====================================")

    # Update 2025-04-28:
    if config.MODEL.NAME == "scenestreamer" and config.MODEL.USE_MOTION_HEAD_PRENORM is True:
        if "model.motion_prenorm.weight" not in checkpoint["state_dict"]:
            print("====================================")
            print("[WARNING] The motion_prenorm is not found in the checkpoint.")
            print("Using initial parameters for motion_prenorm.")
            if m is None:
                m = model_class(config)
            default_params = m.state_dict()
            for k, v in default_params.items():
                if "motion_prenorm" in k:
                    assert k not in checkpoint["state_dict"], k
                    print("Writing: ", k, v.shape)
                    checkpoint["state_dict"][k] = v
            print("====================================")

        if config.get("SCENESTREAMER_NO_TG", True) is False:
            if "model.trafficgen_prenorm.weight" not in checkpoint["state_dict"]:
                print("====================================")
                print("[WARNING] The trafficgen_prenorm is not found in the checkpoint.")
                print("Using initial parameters for trafficgen_prenorm.")
                if m is None:
                    m = model_class(config)
                default_params = m.state_dict()
                for k, v in default_params.items():
                    if "trafficgen_prenorm" in k:
                        assert k not in checkpoint["state_dict"], k
                        print("Writing: ", k, v.shape)
                        checkpoint["state_dict"][k] = v
                print("====================================")

    # Update 2025-04-25:
    # If the pretrained checkpoint is SCENESTREAMER_NO_TG=True, we need to add the trafficgen decoder.
    if config.MODEL.NAME == "scenestreamer" and config.get("SCENESTREAMER_NO_TG", False) is False:
        if "model.trafficgen_intra_step.tokens.weight" not in checkpoint["state_dict"]:
            print("====================================")
            print("[WARNING] The trafficgen decoder is not found in the checkpoint.")
            print("Using initial parameters for trafficgen decoder.")
            if m is None:
                m = model_class(config)
            default_params = m.state_dict()
            for k, v in default_params.items():
                if "trafficgen_intra_step" in k or "model.trafficgen_head" in k or "model.trafficgen_feat_embed" in k:
                    assert k not in checkpoint["state_dict"], k
                    print("Writing: ", k, v.shape)
                    checkpoint["state_dict"][k] = v
            print("====================================")

    return checkpoint


def normalize_legacy_model_config(config, checkpoint_path=None):
    model_name = config.get("MODEL", {}).get("NAME", None)
    if model_name == "infgen":
        print("[model] Legacy MODEL.NAME `infgen` detected; treating it as `scenestreamer`.")
        config.MODEL.NAME = "scenestreamer"
    if config.get("MODEL", {}).get("NAME", None) == "scenestreamer" and config.get("SCENESTREAMER_NO_TG", None) is None:
        checkpoint_name = Path(checkpoint_path).name if checkpoint_path is not None else ""
        if "scenestreamer-base-" in checkpoint_name:
            config.SCENESTREAMER_NO_TG = True
        elif "scenestreamer-full-" in checkpoint_name:
            config.SCENESTREAMER_NO_TG = False
    return config


def get_model(
    *,
    huggingface_repo=None,
    huggingface_file=None,
    config=None,
    checkpoint_path=None,
    device=None,
    default_config="motion_default.yaml"
):
    from scenestreamer.model_hub import (
        DEFAULT_HF_REPO,
        get_default_config_name_for_checkpoint,
        get_default_hf_file,
    )

    def _normalize_optional(v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def resolve_runtime_device(requested_device):
        if requested_device is None or str(requested_device) == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        resolved = requested_device if isinstance(requested_device, torch.device) else torch.device(requested_device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            fallback = torch.device("mps") if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else torch.device("cpu")
            print(f"[device] Requested `{resolved}` but CUDA is unavailable; using `{fallback}` instead.")
            return fallback

        if resolved.type == "mps" and not (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ):
            fallback = torch.device("cpu")
            print(f"[device] Requested `{resolved}` but MPS is unavailable; using `{fallback}` instead.")
            return fallback

        return resolved

    device = resolve_runtime_device(device)

    huggingface_repo = _normalize_optional(huggingface_repo)
    huggingface_file = _normalize_optional(huggingface_file)
    checkpoint_path = _normalize_optional(checkpoint_path)

    pretrained_path_from_config = _normalize_optional(config.pretrain) if config is not None else None
    huggingface_repo_from_config = _normalize_optional(getattr(config, "huggingface_repo", None)) if config is not None else None
    huggingface_file_from_config = _normalize_optional(getattr(config, "huggingface_file", None)) if config is not None else None
    model_name_from_config = _normalize_optional(getattr(config, "exp_name", None)) if config is not None else None

    if config is not None and huggingface_repo_from_config is None and huggingface_file_from_config is None and pretrained_path_from_config is None:
        default_hf_file = get_default_hf_file(model_name_from_config)
        if default_hf_file is not None:
            huggingface_repo_from_config = DEFAULT_HF_REPO
            huggingface_file_from_config = default_hf_file

    huggingface_repo = huggingface_repo or huggingface_repo_from_config
    huggingface_file = huggingface_file or huggingface_file_from_config

    assert (huggingface_repo is None) == (huggingface_file is None), (
        "huggingface_repo and huggingface_file must be provided together."
    )

    pretrained_path_from_arg = checkpoint_path
    pretrained_path = pretrained_path_from_arg or pretrained_path_from_config

    assert huggingface_repo is None or pretrained_path is None, (
        "Provide either a HuggingFace checkpoint source or a local checkpoint path, not both."
    )

    if huggingface_repo is not None:
        try:
            import huggingface_hub
        except ImportError:
            raise ValueError("Please install huggingface_hub via: pip install --upgrade huggingface_hub")
        print("Downloading checkpoint from Huggingface Hub: ", huggingface_repo, huggingface_file)
        checkpoint_path = huggingface_hub.hf_hub_download(repo_id=huggingface_repo, filename=huggingface_file)
        print("Downloaded checkpoint from Huggingface Hub: ", checkpoint_path)
        pretrained_path = checkpoint_path

    assert config is not None or pretrained_path is not None, "Either config or checkpoint_path must be provided."
    from scenestreamer.models.motionlm_lightning import MotionLMLightning

    from scenestreamer.utils.config import global_config, cfg_from_yaml_file
    if config is None and default_config == "motion_default.yaml":
        inferred_default_config = get_default_config_name_for_checkpoint(huggingface_file)
        if inferred_default_config is None and pretrained_path is not None:
            inferred_default_config = get_default_config_name_for_checkpoint(str(pretrained_path))
        if inferred_default_config is not None:
            default_config = inferred_default_config
    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs" / default_config, global_config)
    if pretrained_path:
        pretrained_path = pathlib.Path(pretrained_path).expanduser()
        pretrained_path = REPO_ROOT / pretrained_path
        if pretrained_path.is_dir():
            if (pretrained_path / "last.ckpt").exists():
                pretrained_path = pretrained_path / "last.ckpt"
            # If only one file:
            elif len(list(pretrained_path.glob("*.ckpt"))) == 1:
                pretrained_path = list(pretrained_path.glob("*.ckpt"))[0]
            else:
                raise ValueError(
                    "Please provide a checkpoint file or a directory with only one checkpoint file or contains the "
                    "last.ckpt."
                )
        pretrained_path = str(pretrained_path.absolute())  # Don't call resolve() here as huggingface_hub will fail.
        assert os.path.isfile(pretrained_path), pretrained_path
        assert pretrained_path.endswith(".ckpt"), pretrained_path
        print("==============================")
        print("Loading pretrained model: ", pretrained_path)
        print("==============================")

        model = load_from_checkpoint(
            checkpoint_path=pretrained_path,
            cls=MotionLMLightning,
            config=config,
            default_config=default_config,
            strict=True,
            checkpoint_surgery_func=checkpoint_surgery_func,
            # For non-CUDA targets (CPU/MPS), first load weights onto CPU and then move.
            # This avoids torch.load failing when the checkpoint contains CUDA storages.
            map_location=device if (isinstance(device, torch.device) and device.type == "cuda") or str(device).startswith("cuda") else torch.device("cpu"),
        )

    else:
        model = MotionLMLightning(config=config)

    if device is not None:
        model.to(device)

    model.eval()

    return model


def repeat_for_modes(v, num_modes):
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


def expand_for_modes(v, num_modes):

    if isinstance(v, dict):
        ret = {}
        for k, vv in v.items():
            if isinstance(vv, torch.Tensor):
                ret[k] = expand_for_modes(vv, num_modes)
            else:
                ret[k] = vv
        return ret

    assert isinstance(v, torch.Tensor), "Only torch.Tensor is supported. Found: {}".format(type(v))
    d = v.ndim
    if d > 1:
        v = v[:, None]
        v = v.expand(-1, num_modes, *((-1, ) * (d - 1)))
        v = v.flatten(0, 1)
    else:
        v = v.reshape(-1, 1)
        v = v.expand(-1, num_modes)
        v = v.reshape(-1)
    return v


def numpy_to_torch(v, device=None):
    if isinstance(v, dict):
        return {k: numpy_to_torch(vv, device) for k, vv in v.items()}

    if isinstance(v, list) or isinstance(v, (float, int)):
        v = np.array(v)

    # Skip conversion for strings
    if isinstance(v, str):
        return v

    # Convert numpy arrays to torch tensors
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.number) or v.dtype == bool:
            v = torch.from_numpy(v)

    # Move tensor to the specified device if provided
    if isinstance(v, torch.Tensor) and device is not None:
        # MPS doesn't support float64; cast to float32 before moving.
        if (isinstance(device, torch.device) and device.type == "mps") or str(device).startswith("mps"):
            if v.dtype == torch.float64:
                v = v.float()
        v = v.to(device)

    return v


def batch_data(data_dict):
    """Add one additional dimension to all values in the dictionary."""
    ret = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            ret[k] = v[None]
        elif isinstance(v, torch.Tensor):
            ret[k] = v[None]
        else:
            ret[k] = v
    return ret


def unbatch_data(data_dict):
    """Remove the additional dimension from all values in the dictionary."""
    ret = {}
    first_key = list(data_dict.keys())[0]
    B = data_dict[first_key].shape[0]
    for k, v in data_dict.items():
        if "track_name" in k and v.ndim == 1:
            ret[k] = v
        elif isinstance(v, np.ndarray):
            assert v.shape[0] == B, f"Shape mismatch for {k}: {v.shape[0]} vs {B}"
            ret[k] = v[0]
        elif isinstance(v, torch.Tensor):
            assert v.shape[0] == B, f"Shape mismatch for {k}: {v.shape[0]} vs {B}"
            ret[k] = v[0]
        else:
            ret[k] = v
    return ret


def torch_to_numpy(v):
    if isinstance(v, dict):
        return {k: torch_to_numpy(vv) for k, vv in v.items()}

    if isinstance(v, list):
        v = np.array([torch_to_numpy(vv) for vv in v])

    if isinstance(v, (float, int)):
        v = np.array(v)

    # Convert torch tensors to numpy arrays
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()

    return v


class SafeFallbackEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__ and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeFallbackEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_print(result, prefix=""):
    """
    Should call print(pretty_print(result)) to print the result in a human-readable format.
    """
    result = result.copy()
    result = {prefix + k: v for k, v in result.items()}
    cleaned = json.dumps(result, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


rank_zero_print = rank_zero_only(print)

def get_relative_velocity(vel, heading):
    return rotate(vel[..., 0], vel[..., 1], angle=-heading)


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