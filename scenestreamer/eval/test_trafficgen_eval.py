import copy

import hydra
import omegaconf
import torch
import torchmetrics
import tqdm
import numpy as np
from numpy.core.defchararray import center

from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import utils

from shapely.geometry import Polygon

import copy
import datetime
import pickle
import time
from enum import Enum

import numpy as np
import torch
from shapely.geometry import Polygon
from torch import Tensor


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class AgentType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret = fn(*args, **kwargs)
        return ret, time.clock() - start

    return _wrapper


def MDdata_to_initdata(MDdata):
    ret = {}
    tracks = MDdata['tracks']

    ret['context_num'] = 1
    all_agent = np.zeros([128, 7])
    agent_mask = np.zeros(128)

    sdc = tracks[MDdata['sdc_index']]['state']
    all_agent[0, :2] = sdc[0, :2]
    all_agent[0, 2:4] = sdc[0, 7:9]
    all_agent[0, 4] = sdc[0, 6]
    all_agent[0, 5:7] = sdc[0, 3:5]

    cnt = 1
    for id, track in tracks.items():
        if id == MDdata['sdc_index']:
            continue
        if not track['type'] == AgentType.VEHICLE:
            continue
        if track['state'][0, -1] == 0:
            continue
        state = track['state']
        all_agent[cnt, :2] = state[0, :2]
        all_agent[cnt, 2:4] = state[0, 7:9]
        all_agent[cnt, 4] = state[0, 6]
        all_agent[cnt, 5:7] = state[0, 3:5]
        cnt += 1

    all_agent = all_agent[:32]
    agent_num = min(32, cnt)
    agent_mask[:agent_num] = 1
    agent_mask = agent_mask.astype(bool)

    lanes = []
    for k, lane in input['map'].items():
        a_lane = np.zeros([20, 4])
        tp = 0
        try:
            lane_type = lane['type']
        except:
            lane_type = lane['sign']
            poly_line = lane['polygon']
            if lane_type == 'cross_walk':
                tp = 18
            elif lane_type == 'speed_bump':
                tp = 19

        if lane_type == 'center_lane':
            poly_line = lane['polyline']
            tp = 1

        elif lane_type == RoadEdgeType.BOUNDARY or lane_type == RoadEdgeType.MEDIAN:
            tp = 15 if lane_type == RoadEdgeType.BOUNDARY else 16
            poly_line = lane['polyline']
        elif 'polyline' in lane:
            tp = 7
            poly_line = lane['polyline']
        if tp == 0:
            continue

        a_lane[:, 2] = tp
        a_lane[:, :2] = poly_line

        lanes.append(a_lane)
    lanes = np.stack(lanes)

    return


def get_polygon(center, yaw, L, W):
    l, w = L / 2, W / 2
    yaw += torch.pi / 2
    theta = torch.atan(w / l)
    s1 = torch.sqrt(l**2 + w**2)
    x1 = abs(torch.cos(theta + yaw) * s1)
    y1 = abs(torch.sin(theta + yaw) * s1)
    x2 = abs(torch.cos(theta - yaw) * s1)
    y2 = abs(torch.sin(theta - yaw) * s1)

    p1 = [center[0] + x1, center[1] + y1]
    p2 = [center[0] + x2, center[1] - y2]
    p3 = [center[0] - x1, center[1] - y1]
    p4 = [center[0] - x2, center[1] + y2]
    return Polygon([p1, p3, p2, p4])


def get_agent_coord_from_vec(vec, long_lat):
    vec = torch.tensor(vec)
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    vec_len = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    return coord


def get_agent_pos_from_vec(vec, long_lat, speed, vel_heading, heading, bbox, use_rel_heading=True):
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    vec_len = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    if use_rel_heading:
        agent_dir = vec_dir + heading
    else:
        agent_dir = heading

    v_dir = vel_heading + agent_dir

    vel = torch.stack([torch.cos(v_dir) * speed, torch.sin(v_dir) * speed], axis=-1)
    agent_num, _ = vel.shape

    type = Tensor([[1]]).repeat(agent_num, 1).to(coord.device)
    agent = torch.cat([coord, vel, agent_dir.unsqueeze(1), bbox, type], dim=-1).detach().cpu().numpy()

    vec_based_rep = torch.cat(
        [long_lat, speed.unsqueeze(-1),
         vel_heading.unsqueeze(-1),
         heading.unsqueeze(-1), vec], dim=-1
    ).detach().cpu().numpy()

    agent = WaymoAgent(agent, vec_based_rep)

    return agent


def process_lane(lane, max_vec, lane_range, offset=-40):
    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    vec_id_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vec_ids = np.ones([b_s, points.shape[1] - 1, 1]) * id
        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)
        vec_id_list.append(vec_ids)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)
    vec_id = np.concatenate(vec_id_list, axis=1) if vec_id_list else np.zeros([b_s, 0, 1])

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    all_id = np.zeros([b_s, max_vec, 1])

    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]
        vec_id_t = vec_id[t][mask_t]

        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])
        vec_id_t = vec_id_t[idx]

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]
        vec_id_t = vec_id_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        vec_id_t = np.pad(vec_id_t, ([0, max_vec - vec_id_t.shape[0]], [0, 0]))

        all_vec[t] = vector_t
        all_mask[t] = mask_t
        all_id[t] = vec_id_t

    return all_vec, all_mask.astype(bool), all_id.astype(int)


def process_map(lane, traf=None, center_num=384, edge_num=128, lane_range=60, offest=-40, rest_num=192):
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    # print(traf)
    if traf is not None:
        for i in range(b_s):
            traf_t = traf[i]
            lane_id_t = lane_id[i]
            # print(traf_t)
            for a_traf in traf_t:
                # print(a_traf)
                control_lane_id = a_traf[0]
                state = a_traf[-2]
                lane_idx = np.where(lane_id_t == control_lane_id)
                lane_with_traf[i, lane_idx, -1] = state
        lane = lane_with_traf

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask, cent_id = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask, _ = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask, _ = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask, _ = process_lane(lane[:, rest], rest_num, lane_range, offest)

    return cent, cent_mask, cent_id, bound, bound_mask, cross, cross_mask, rest, rest_mask


def get_time_str():
    return datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")


def normalize_angle(angle):
    if isinstance(angle, torch.Tensor):
        while not torch.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not torch.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2
        return angle

    else:
        while not np.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not np.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2

        return angle


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def from_list_to_batch(inp_list):
    keys = inp_list[0].keys()

    batch = {}
    for key in keys:
        one_item = [item[key] for item in inp_list]
        batch[key] = Tensor(np.stack(one_item))

    return batch


def get_type_class(line_type):
    if line_type in range(1, 4):
        return 'center_lane'
    elif line_type == 6:
        return RoadLineType.BROKEN_SINGLE_WHITE
    elif line_type == 7:
        return RoadLineType.SOLID_SINGLE_WHITE
    elif line_type == 8:
        return RoadLineType.SOLID_DOUBLE_WHITE
    elif line_type == 9:
        return RoadLineType.BROKEN_SINGLE_YELLOW
    elif line_type == 10:
        return RoadLineType.BROKEN_DOUBLE_YELLOW
    elif line_type == 11:
        return RoadLineType.SOLID_SINGLE_YELLOW
    elif line_type == 12:
        return RoadLineType.SOLID_DOUBLE_YELLOW
    elif line_type == 13:
        return RoadLineType.PASSING_DOUBLE_YELLOW
    elif line_type == 15:
        return RoadEdgeType.BOUNDARY
    elif line_type == 16:
        return RoadEdgeType.MEDIAN
    else:
        return 'other'


def transform_to_metadrive_data(pred_i, other):
    output_temp = {}
    output_temp['id'] = 'fake'
    output_temp['ts'] = [x / 10 for x in range(190)]
    output_temp['dynamic_map_states'] = [{}]
    output_temp['sdc_index'] = 0
    cnt = 0

    center_info = other['center_info']
    output = copy.deepcopy(output_temp)
    output['tracks'] = {}
    output['map'] = {}
    # extract agents
    agent = pred_i

    for i in range(agent.shape[1]):
        track = {}
        agent_i = agent[:, i]
        track['type'] = AgentType.VEHICLE
        state = np.zeros([agent_i.shape[0], 10])
        state[:, :2] = agent_i[:, :2]
        state[:, 3] = 5.286
        state[:, 4] = 2.332
        state[:, 7:9] = agent_i[:, 2:4]
        state[:, -1] = 1
        state[:, 6] = agent_i[:, 4]  # + np.pi / 2
        track['state'] = state
        output['tracks'][i] = track

    # extract maps
    lane = other['unsampled_lane']
    lane_id = np.unique(lane[..., -1]).astype(int)
    for id in lane_id:

        a_lane = {}
        id_set = lane[..., -1] == id
        points = lane[id_set]
        polyline = np.zeros([points.shape[0], 3])
        line_type = points[0, -2]
        polyline[:, :2] = points[:, :2]
        a_lane['type'] = get_type_class(line_type)
        a_lane['polyline'] = polyline
        if id in center_info.keys():
            a_lane.update(center_info[id])
        output['map'][id] = a_lane

    return output


def save_as_metadrive_data(pred_i, other, save_path):
    output = transform_to_metadrive_data(pred_i, other)

    with open(save_path, 'wb') as f:
        pickle.dump(output, f)


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def angle_to_vector(angles):
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([x, y], dim=0)
    n_samples = total.size(0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    l2_distance = ((total0 - total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul**(kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernels = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
    return sum(kernels)


class WaymoAgent:
    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:

            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])[..., np.newaxis]
            self.length_width = feature[..., 6:8]
            type = np.ones_like(self.heading)
            self.feature = np.concatenate(
                [self.position, self.velocity, self.heading, self.length_width, type], axis=-1
            )
            if vec_based_info is not None:
                vec_based_rep = copy.deepcopy(vec_based_info)
                vec_based_rep[..., 5:9] *= self.RANGE
                vec_based_rep[..., 2] *= self.MAX_SPEED
                self.vec_based_info = vec_based_rep

        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    @staticmethod
    def from_list_to_array(inp_list):
        MAX_AGENT = 32
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:MAX_AGENT]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, MAX_AGENT - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_num]))
        agent_mask[:agent_num] = 1
        agent_mask = agent_mask.astype(bool)
        return agent, agent_mask

    def get_agent(self, index):
        return WaymoAgent(self.feature[[index]], self.vec_based_info[[index]])

    def get_feature(self):
        self.feature[..., :2] = self.position
        self.feature[..., 2:4] = self.velocity
        self.feature[..., [4]] = self.heading

        return self.feature

    def get_list(self):
        # bs, agent_num, feature_dim = self.feature.shape
        agent_num, feature_dim = self.feature.shape
        vec_dim = self.vec_based_info.shape[-1]
        feature = self.feature.reshape([-1, feature_dim])
        vec_rep = self.vec_based_info.reshape([-1, vec_dim])
        agent_num = feature.shape[0]
        lis = []
        for i in range(agent_num):
            # lis.append(WaymoAgent(feature[[i]], vec_rep[[i]]))
            lis.append(WaymoAgent(feature[[i]], vec_rep[[i]]))
        return lis

    def get_inp(self, act=False, act_inp=False):

        if act:
            return np.concatenate([self.position, self.velocity, self.heading, self.length_width], axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate([pos, velo, cos_head, sin_head, self.length_width], axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        agent_feat = np.concatenate([pos, velo, cos_head, sin_head, self.length_width, vec_based_rep], axis=-1)
        return agent_feat

    def get_rect(self, pad=0):

        l, w = (self.length_width[..., 0] + pad) / 2, (self.length_width[..., 1] + pad) / 2
        x1, y1 = l, w
        x2, y2 = l, -w

        point1 = rotate(x1, y1, self.heading[..., 0])
        point2 = rotate(x2, y2, self.heading[..., 0])
        center = self.position

        x1, y1 = point1[..., [0]], point1[..., [1]]
        x2, y2 = point2[..., [0]], point2[..., [1]]

        p1 = np.concatenate([center[..., [0]] + x1, center[..., [1]] + y1], axis=-1)
        p2 = np.concatenate([center[..., [0]] + x2, center[..., [1]] + y2], axis=-1)
        p3 = np.concatenate([center[..., [0]] - x1, center[..., [1]] - y1], axis=-1)
        p4 = np.concatenate([center[..., [0]] - x2, center[..., [1]] - y2], axis=-1)

        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])

        agent_num, dim = p1.shape

        rect_list = []
        for i in range(agent_num):
            rect = np.stack([p1[i], p2[i], p3[i], p4[i]])
            rect_list.append(rect)
        return rect_list

    def get_polygon(self):
        rect_list = self.get_rect(pad=0.25)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a, b, c, d]))

        return poly_list


def compute_mmd_different_sizes(x, y, kernel='gaussian', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    assert x.ndim == 2
    assert y.ndim == 2
    if kernel == 'gaussian':
        kernels = gaussian_kernel(x, y, kernel_mul, kernel_num, fix_sigma)
    else:
        raise ValueError("Currently, only Gaussian kernel is supported for different sizes.")

    n_x = x.size(0)
    n_y = y.size(0)

    XX = kernels[:n_x, :n_x]  # Kernel matrix for x vs x
    YY = kernels[n_x:, n_x:]  # Kernel matrix for y vs y
    XY = kernels[:n_x, n_x:]  # Kernel matrix for x vs y
    YX = kernels[n_x:, :n_x]  # Kernel matrix for y vs x

    # Normalize expectations to account for different sizes
    # mmd = (XX.sum() / (n_x * n_x) + YY.sum() / (n_y * n_y) - XY.sum() / (n_x * n_y) - YX.sum() / (n_y * n_x))
    mmd = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    mmd = torch.clamp(mmd, min=0.0)
    return mmd


def normalize_angle(angle):
    """
    From: https://github.com/metadriverse/trafficgen/blob/28b109e8e640d820192d5485bf9a28128b38ca21/trafficgen/utils/utils.py#L20
    """
    if isinstance(angle, torch.Tensor):
        while not torch.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not torch.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2
        return angle

    else:
        while not np.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not np.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2

        return angle


class TrafficGenEvaluator:
    def __init__(self, config, device=None):
        self.use_tg_as_gt = config.EVALUATION.USE_TG_AS_GT
        assert self.use_tg_as_gt == 1111, "no need to set USE_TG_AS_GT"

    def _transform_coordinate_map(self, data):
        """
        Every frame is different
        """
        timestep = data['all_agent'].shape[0]

        ego = data['all_agent'][:, 0]
        pos = ego[:, [0, 1]][:, np.newaxis]

        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane, timestep, axis=0)
        lane[..., :2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        ego_heading = ego[:, [4]]
        lane[..., :2] = rotate(x, y, -ego_heading)

        unsampled_lane = data['unsampled_lane'][np.newaxis]
        unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
        unsampled_lane[..., :2] -= pos

        x = unsampled_lane[..., 0]
        y = unsampled_lane[..., 1]
        ego_heading = ego[:, [4]]
        unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
        return lane, unsampled_lane[0]

    def _get_trafficgen_data(self, data_dict, current_t):
        """
        PZH:
        I don't want to waste time to read through the LCTGen code,
        which essentially is from the TrafficGen code base.
        I've read the TrafficGen code base and I really really don't want
        to look into it for the second time.
        Just copy the code here and modify it to fit the current code base.
        """

        from scenestreamer.eval.scenarionet_to_trafficgen import metadrive_scenario_to_init_data

        data = metadrive_scenario_to_init_data(data_dict["raw_scenario_description"][0])
        PZH_TRACK_NAMES = data["PZH_TRACK_NAMES"]
        case_info = {}
        other = {}

        # agent = copy.deepcopy(data['all_agent'])
        other['traf'] = copy.deepcopy(data['traffic_light'])

        max_time_step = 190
        gap = 190
        index = -1
        RANGE = 50

        if index == -1:
            data['all_agent'] = data['all_agent'][current_t:max_time_step:gap]
            data['traffic_light'] = data['traffic_light'][current_t:max_time_step:gap]
        else:
            raise ValueError
        data['lane'], other['unsampled_lane'] = self._transform_coordinate_map(data)
        other['lane'] = data['lane']

        def _process_agent(agent, sort_agent):

            ego = agent[:, 0]

            # transform every frame into ego coordinate in the first frame
            ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
            ego_heading = ego[[0], [4]]

            agent[..., :2] -= ego_pos
            agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
            agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
            agent[..., 4] -= ego_heading

            agent_mask = agent[..., -1]
            agent_type_mask = agent[..., -2]
            agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)

            mask = agent_mask * agent_type_mask
            # use agent range mask only for the first frame
            # allow agent to be out of range in the future frames
            mask[0, :] *= agent_range_mask[0, :]

            return agent, mask.astype(bool)

        case_info["agent"], case_info["agent_mask"] = _process_agent(data['all_agent'], False)
        case_info['center'], case_info['center_mask'], case_info['center_id'], case_info['bound'], case_info[
            'bound_mask'], \
            case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'], data['traffic_light'], lane_range=RANGE, offest=0)

        # get vector-based representatiomn
        def _get_vec_based_rep(case_info, PZH_TRACK_NAMES):
            THRES = 5
            thres = THRES
            # max_agent_num = 32
            # _process future agent

            agent = case_info['agent']
            vectors = case_info["center"]

            agent_mask = case_info['agent_mask']

            vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
            vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

            agent_x = agent[..., 0]
            agent_y = agent[..., 1]

            b, vec_num = vec_y.shape
            _, agent_num = agent_x.shape

            vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
            vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

            agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
            agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

            dist = np.sqrt((vec_x - agent_x)**2 + (vec_y - agent_y)**2)

            cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
            dist[cent_mask == 0] = 10e5
            vec_index = np.argmin(dist, -1)
            min_dist_to_lane = np.min(dist, -1)
            min_dist_mask = min_dist_to_lane < thres

            selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

            vx, vy = agent[..., 2], agent[..., 3]
            v_value = np.sqrt(vx**2 + vy**2)
            low_vel = v_value < 0.1

            dir_v = np.arctan2(vy, vx)
            x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
            dir = np.arctan2(y2 - y1, x2 - x1)
            agent_dir = agent[..., 4]

            v_relative_dir = cal_rel_dir(dir_v, agent_dir)
            relative_dir = cal_rel_dir(agent_dir, dir)

            v_relative_dir[low_vel] = 0

            v_dir_mask = abs(v_relative_dir) < np.pi / 6
            dir_mask = abs(relative_dir) < np.pi / 4

            agent_x = agent[..., 0]
            agent_y = agent[..., 1]
            vec_x = (x1 + x2) / 2
            vec_y = (y1 + y2) / 2

            cent_to_agent_x = agent_x - vec_x
            cent_to_agent_y = agent_y - vec_y

            coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

            vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

            lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
            long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len

            # ignore other masks for future agents (to support out-of-range agent prediction)
            total_mask = agent_mask
            # for the first frame, use all masks to filter out off-road agents
            total_mask[0, :] = (min_dist_mask * agent_mask * v_dir_mask * dir_mask)[0, :]

            total_mask[:, 0] = 1
            total_mask = total_mask.astype(bool)

            b_s, agent_num, agent_dim = agent.shape
            agent_ = np.zeros([b_s, agent_num, agent_dim])
            agent_mask_ = np.zeros([b_s, agent_num]).astype(bool)

            the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
            # 0: vec_index
            # 1-2 long and lat percent
            # 3-5 velocity and direction
            # 6-9 lane vector
            # 10-11 lane type and traff state
            info = np.concatenate(
                [
                    vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis],
                    v_value[..., np.newaxis], v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec
                ], -1
            )

            info_ = np.zeros([b_s, agent_num, info.shape[-1]])

            start_mask = total_mask[0]
            for i in range(agent.shape[0]):
                agent_i = agent[i][start_mask]
                info_i = info[i][start_mask]

                step_mask = total_mask[i]
                valid_mask = step_mask[start_mask]

                agent_i = agent_i[:agent_num]
                info_i = info_i[:agent_num]

                valid_num = agent_i.shape[0]
                agent_i = np.pad(agent_i, [[0, agent_num - agent_i.shape[0]], [0, 0]])
                info_i = np.pad(info_i, [[0, agent_num - info_i.shape[0]], [0, 0]])

                agent_[i] = agent_i
                info_[i] = info_i
                agent_mask_[i, :valid_num] = valid_mask[:valid_num]

            PZH_TRACK_NAMES_new = np.array(list(PZH_TRACK_NAMES[start_mask]) + [None] * (agent_num - start_mask.sum()))

            case_info['vec_based_rep'] = info_[..., 1:]
            case_info['agent_vec_index'] = info_[..., 0].astype(int)
            case_info['agent_mask'] = agent_mask_
            case_info["agent"] = agent_

            return case_info, PZH_TRACK_NAMES_new

        case_info, PZH_TRACK_NAMES = _get_vec_based_rep(case_info, PZH_TRACK_NAMES)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k, v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)

        # PZH: Obviously, you only pick T=0 from the data.
        ret = case_list[0]
        ret["PZH_TRACK_NAMES"] = PZH_TRACK_NAMES
        return ret

    def validation_step(self, data_dict, stat, log_func, **kwargs):
        B = data_dict["decoder/modeled_agent_position_for_trafficgen"].shape[0]
        assert B == 1

        current_t = data_dict["metadata/current_time_index"].item()

        agent_pos = data_dict["decoder/modeled_agent_position_for_trafficgen"]  # (N, 2)
        agent_heading = data_dict["decoder/modeled_agent_heading_for_trafficgen"]  # (N, 1)
        agent_velocity = data_dict["decoder/modeled_agent_velocity_for_trafficgen"]  # (N, 2)
        agent_shape = data_dict["decoder/current_agent_shape_for_trafficgen"]  # (N, 3)
        agent_mask = data_dict["decoder/input_action_valid_mask_for_trafficgen"]  # (N,)
        agent_type = data_dict["decoder/agent_type_for_trafficgen"]  # (N,)

        trafficgen_data = self._get_trafficgen_data(data_dict, current_t)
        trafficgen_select_track_names = trafficgen_data["PZH_TRACK_NAMES"][trafficgen_data['agent_mask']]
        decoder_track_name = list(data_dict["decoder/track_name"][0])
        trafficgen_select_index = []
        for name in trafficgen_select_track_names:
            if name in decoder_track_name:
                trafficgen_select_index.append(decoder_track_name.index(name))
            else:
                # print(11)
                pass

        all_select_index = data_dict["decoder/agent_valid_mask"][0, current_t].nonzero()[:, 0]

        for i in range(B):
            pos_target = data_dict["decoder/agent_position"][i, current_t, trafficgen_select_index, :2]
            vel_target = data_dict["decoder/agent_velocity"][i, current_t, trafficgen_select_index, :2]
            head_target = data_dict["decoder/agent_heading"][i, current_t, trafficgen_select_index]
            size_target = data_dict["decoder/agent_shape"][i, current_t, trafficgen_select_index]
            actor_type = data_dict["decoder/agent_type"][i][trafficgen_select_index]
            num_target = len(trafficgen_select_index)

            pos_pred = agent_pos[i, agent_mask[i]]
            head_pred = agent_heading[i, agent_mask[i]]
            vel_pred = agent_velocity[i, agent_mask[i]]
            size_pred = agent_shape[i, agent_mask[i]]
            type_pred = agent_type[i, agent_mask[i]]
            num_pred = len(pos_pred)

            from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
            poly = cal_polygon_contour(
                x=pos_pred[:, 0].cpu().numpy(),
                y=pos_pred[:, 1].cpu().numpy(),
                theta=head_pred.cpu().numpy(),
                width=size_pred[:, 1].cpu().numpy(),
                length=size_pred[:, 0].cpu().numpy()
            )
            collision_detected = np.zeros(len(poly), dtype=bool)
            for i in range(len(poly) - 1):
                agent_collision_detected = []
                for j in range(i + 1, len(poly)):
                    poly1 = Polygon(poly[i])
                    poly2 = Polygon(poly[j])
                    if poly1.intersects(poly2):
                        collision_detected[i] = True
                        collision_detected[j] = True
            static_collision_rate = collision_detected.mean()
            log_func("static_collision_rate", static_collision_rate)

            # Compute the position matching here

            for suffix, (gt_mask, pred_mask) in {
                    "_all": (pos_pred.new_ones(num_target, dtype=bool), pos_pred.new_ones(num_pred, dtype=bool)),
                    # "_vehicle": (actor_type == 1, type_pred == 0),
                    # "_pedestrian": (actor_type == 2, type_pred == 1),
                    # "_cyclist": (actor_type == 3, type_pred == 2),
            }.items():
                if not gt_mask.any():
                    continue
                if not pred_mask.any():
                    continue
                log_func(f"num_gt_samples{suffix}", gt_mask.sum().item())
                log_func(f"num_pred_samples{suffix}", pred_mask.sum().item())

                # Follow: https://github.com/metadriverse/trafficgen/blob/28b109e8e640d820192d5485bf9a28128b38ca21/trafficgen/test_init.py#L44C5-L44C16
                kernel_mul = 1.0
                kernel_num = 1

                center_head = head_target[0]
                mmd_pos = compute_mmd_different_sizes(
                    x=pos_pred[pred_mask],
                    y=pos_target[gt_mask],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
                log_func(f"mmd_pos{suffix}", mmd_pos)

                log_func(
                    f"mmd_vel{suffix}",
                    compute_mmd_different_sizes(
                        x=vel_pred[pred_mask],
                        y=vel_target[gt_mask],
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )

                after_sdc_center = compute_mmd_different_sizes(
                    x=normalize_angle(head_pred[pred_mask] - center_head)[:, None],
                    y=normalize_angle(head_target[gt_mask] - center_head)[:, None],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
                no_sdc_center = compute_mmd_different_sizes(
                    x=normalize_angle(head_pred[pred_mask])[:, None],
                    y=normalize_angle(head_target[gt_mask])[:, None],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )

                log_func(f"mmd_head{suffix}", no_sdc_center)
                log_func(f"mmd_head_center{suffix}", after_sdc_center)

                transformed_pred = angle_to_vector(normalize_angle(head_pred[pred_mask]))
                transformed_target = angle_to_vector(normalize_angle(head_target[gt_mask]))
                log_func(
                    f"mmd_head_transformed{suffix}",
                    compute_mmd_different_sizes(
                        x=transformed_pred,
                        y=transformed_target,
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )
                log_func(
                    f"mmd_size{suffix}",
                    compute_mmd_different_sizes(
                        x=size_pred[pred_mask][..., :2],
                        y=size_target[gt_mask][..., :2],
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )

        if "num_collisions" in stat:
            log_func("num_collisions", stat["num_collisions"])
        if "num_violations" in stat:
            log_func("num_violations", stat["num_violations"])

        for i in range(B):
            pos_target = data_dict["decoder/agent_position"][i, current_t, all_select_index, :2]
            vel_target = data_dict["decoder/agent_velocity"][i, current_t, all_select_index, :2]
            head_target = data_dict["decoder/agent_heading"][i, current_t, all_select_index]
            size_target = data_dict["decoder/agent_shape"][i, current_t, all_select_index]
            actor_type = data_dict["decoder/agent_type"][i][all_select_index]
            num_target = len(all_select_index)

            pos_pred = agent_pos[i, agent_mask[i]]
            head_pred = agent_heading[i, agent_mask[i]]
            vel_pred = agent_velocity[i, agent_mask[i]]
            size_pred = agent_shape[i, agent_mask[i]]
            type_pred = agent_type[i, agent_mask[i]]
            num_pred = len(pos_pred)

            # Compute the position matching here
            for suffix, (gt_mask, pred_mask) in {
                "_all": (pos_pred.new_ones(num_target, dtype=bool), pos_pred.new_ones(num_pred, dtype=bool)),
                # "_vehicle": (actor_type == 1, type_pred == 0),
                # "_pedestrian": (actor_type == 2, type_pred == 1),
                # "_cyclist": (actor_type == 3, type_pred == 2),
            }.items():
                if not gt_mask.any():
                    continue
                if not pred_mask.any():
                    continue
                log_func(f"ALL_AGENT_num_gt_samples{suffix}", gt_mask.sum().item())
                log_func(f"ALL_AGENT_num_pred_samples{suffix}", pred_mask.sum().item())

                # Follow: https://github.com/metadriverse/trafficgen/blob/28b109e8e640d820192d5485bf9a28128b38ca21/trafficgen/test_init.py#L44C5-L44C16
                kernel_mul = 1.0
                kernel_num = 1

                center_head = head_target[0]
                mmd_pos = compute_mmd_different_sizes(
                    x=pos_pred[pred_mask],
                    y=pos_target[gt_mask],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
                log_func(f"ALL_AGENT_mmd_pos{suffix}", mmd_pos)

                log_func(
                    f"ALL_AGENT_mmd_vel{suffix}",
                    compute_mmd_different_sizes(
                        x=vel_pred[pred_mask],
                        y=vel_target[gt_mask],
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )

                after_sdc_center = compute_mmd_different_sizes(
                    x=normalize_angle(head_pred[pred_mask] - center_head)[:, None],
                    y=normalize_angle(head_target[gt_mask] - center_head)[:, None],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )
                no_sdc_center = compute_mmd_different_sizes(
                    x=normalize_angle(head_pred[pred_mask])[:, None],
                    y=normalize_angle(head_target[gt_mask])[:, None],
                    kernel_mul=kernel_mul,
                    kernel_num=kernel_num,
                )

                log_func(f"ALL_AGENT_mmd_head{suffix}", no_sdc_center)
                log_func(f"ALL_AGENT_mmd_head_center{suffix}", after_sdc_center)

                transformed_pred = angle_to_vector(normalize_angle(head_pred[pred_mask]))
                transformed_target = angle_to_vector(normalize_angle(head_target[gt_mask]))
                log_func(
                    f"ALL_AGENT_mmd_head_transformed{suffix}",
                    compute_mmd_different_sizes(
                        x=transformed_pred,
                        y=transformed_target,
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )
                log_func(
                    f"ALL_AGENT_mmd_size{suffix}",
                    compute_mmd_different_sizes(
                        x=size_pred[pred_mask][..., :2],
                        y=size_target[gt_mask][..., :2],
                        kernel_mul=kernel_mul,
                        kernel_num=kernel_num,
                    )
                )

    def on_validation_epoch_end(self, *args, trainer, global_rank, **kwargs):
        print("Rank", global_rank)
        trainer.strategy.barrier()

    #     for k in self.mmd_metrics:
    #         print_func(f'eval/{k}', self.mmd_metrics[k].compute())
    #         log_func(f'eval/{k}', self.mmd_metrics[k].compute())


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1214_midgpt_v14.yaml")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "/Users/pengzhenghao/PycharmProjects/scenestreamer/lightning_logs/1231_MidGPT_V17_Bicy_WTrafficGen_2024-12-31/"
    model = utils.get_model(checkpoint_path=path, device=device)

    # model = utils.get_model(config, device=device)

    evaluator = TrafficGenEvaluator(config)

    config = model.config
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING["keep_all_data"] = True
    config.DATA.TRAINING_DATA_DIR = "data/20scenarios"
    config.DATA.TEST_DATA_DIR = "data/20scenarios"

    assert config.USE_TRAFFICGEN is True

    test_dataset = SceneStreamerDataset(config, "training")
    # ddd = iter(test_dataset)

    START_ACTION = config.PREPROCESSING.MAX_MAP_FEATURES
    END_ACTION = config.PREPROCESSING.MAX_MAP_FEATURES + 1

    for count, raw_data_dict in enumerate(tqdm.tqdm(test_dataset)):
        data_dict = raw_data_dict
        data_dict = utils.numpy_to_torch(data_dict, device=device)
        data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        data_dict = copy.deepcopy(data_dict)

        data_dict = model.model.encode_scene(data_dict)
        output_dict, stat = model.model.trafficgen_decoder.autoregressive_rollout_trafficgen(data_dict)

        # output_dict = data_dict
        # output_dict = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in output_dict.items()}
        # output_dict = utils.torch_to_numpy(output_dict)

        # suffix = "gt" if use_gt else "pred"
        # save_path = pathlib.Path("0107_trafficgen") / f"trafficgen_{sid}_{suffix}.png"
        # save_path.parent.mkdir(exist_ok=True)
        # print(f"Saving to {save_path}")
        # plot_trafficgen(output_dict, show=False, save_path=save_path)

        # Call evaluator
        evaluator.validation_step(output_dict, stat, log_func=print)

        if count > 3:
            break

    evaluator.on_validation_epoch_end(print_func=print)
    print("End")


if __name__ == '__main__':
    main()
