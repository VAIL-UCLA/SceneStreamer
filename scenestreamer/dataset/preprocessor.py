"""
Translate a MetaDrive Scenario Description instance to a dict of tensors.
"""
import copy
import logging
import pickle

import numpy as np
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType

from scenestreamer import utils
from scenestreamer.dataset import constants
from scenestreamer.dataset.preprocess_action_label import prepare_action_label, prepare_safety_label

from scenestreamer.tokenization.trafficgen_tokenizers import TrafficGenTokenizerAutoregressive, TrafficGenTokenizer

logger = logging.getLogger(__file__)

extract_data_by_agent_indices = utils.extract_data_by_agent_indices

CANT_FIND_DESTINATION = 76543231


def centralize_to_map_center(position_array, map_center, map_heading):
    """
    Centralize the position array to the map center and rotate the position array to the map heading.
    Note that the map center and map heading do not change based on agent or timestep.
    """
    ndim = position_array.ndim
    # position_array = position_array.copy()
    if map_center is not None:
        assert map_center.shape == (3, )
        assert position_array.shape[-1] <= 3, position_array.shape
        map_center = map_center.reshape(*(1, ) * (ndim - 1), 3)
        position_array -= map_center[..., :position_array.shape[-1]]
    if map_heading == 0.0:
        return position_array

    if position_array.shape[-1] == 3:
        position_array = utils.rotate(
            position_array[..., 0], position_array[..., 1], -map_heading, z=position_array[..., 2]
        )
    elif position_array.shape[-1] == 2:
        position_array = utils.rotate(
            position_array[..., 0], position_array[..., 1], -map_heading, z=np.zeros_like(position_array[..., 0])
        )
    else:
        raise ValueError()
    return position_array


def extract_map_center_heading_locations(map_feature):
    assert isinstance(map_feature, dict)
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")
    min_x, min_y, min_z = float("+inf"), float("+inf"), float("+inf")
    for map_feat_id, map_feat in map_feature.items():
        if "polyline" in map_feat:
            locations = map_feat['polyline']
        elif "position" in map_feat:
            locations = map_feat['position']
        elif "polygon" in map_feat:
            locations = map_feat["polygon"]
        else:
            raise ValueError("Unknown map feature: {}, {}".format(map_feat_id, map_feat.keys()))
        locations = locations.reshape(-1, locations.shape[-1])
        map_feat["location"] = locations
        max_boundary = locations.max(axis=0)
        min_boundary = locations.min(axis=0)
        max_x = max_boundary[0]
        max_y = max_boundary[1]
        min_x = min_boundary[0]
        min_y = min_boundary[1]
        if locations.shape[-1] == 3:
            max_z = max_boundary[2]
            min_z = min_boundary[2]
    if max_z == float("-inf"):
        max_z = 0.0
    if min_z == float("+inf"):
        min_z = 0.0
    map_boundary_max = np.array([max_x, max_y, max_z])
    map_boundary_min = np.array([min_x, min_y, min_z])

    map_center = np.stack([map_boundary_max, map_boundary_min], axis=0).mean(axis=0)
    map_heading = 0.0

    return {
        "map_center": map_center,
        "map_heading": map_heading,
        "map_boundary_max": map_boundary_max,
        "map_boundary_min": map_boundary_min,
        "map_feature": map_feature
    }


def prepare_destination(data_dict, config, FUTURE_STEPS=None, skip_step=30, dropout=0.0):
    import torch
    assert FUTURE_STEPS is not None

    # def find_last_valid(array, mask):
    #
    #     array = torch.from_numpy(array[None])
    #     mask = torch.from_numpy(mask[None])
    #
    #     assert mask.ndim + 1 == array.ndim
    #     assert mask.shape == array.shape[:-1]
    #     assert array.ndim == 4
    #     B, T, N, D = array.shape
    #     indices = mask * torch.arange(T).reshape(1, T, 1).expand(*mask.shape)
    #     indices = indices.argmax(1, keepdims=True).unsqueeze(-1).expand(B, 1, N, D)
    #     ret = torch.gather(array, index=indices, dim=1)  # [B, 1, N, D]
    #     ret[~mask.any(1, keepdims=True)] = 0
    #
    #     ret = ret[0].numpy()
    #     return ret

    def find_closest_map_feature(*, agent_pos, agent_heading, map_pos, map_heading, valid_map_feat):
        heading_diff = utils.wrap_to_pi(agent_heading[:, None] - map_heading[None])
        valid_heading = np.abs(heading_diff) < np.deg2rad(90)
        valid_map_feat = valid_map_feat & valid_heading

        dist = np.linalg.norm(agent_pos[:, None] - map_pos[None], axis=-1)

        dist[~valid_map_feat] = np.inf
        closest_map_feat = np.argmin(dist, axis=1)
        closest_map_dist = np.min(dist, axis=1)
        closest_map_feat[closest_map_dist > 10] = CANT_FIND_DESTINATION
        closest_map_feat[np.isinf(dist).all(axis=1)] = CANT_FIND_DESTINATION
        return closest_map_feat

    agent_positions = data_dict["decoder/modeled_agent_position"]
    agent_valid_mask = data_dict["decoder/input_action_valid_mask"]
    agent_headings = data_dict["decoder/modeled_agent_heading"]

    # Use the center of map feature to be the destination.
    map_positions = data_dict["encoder/map_position"][..., :2]

    map_heading = data_dict["encoder/map_heading"]
    valid_map_feat = data_dict["encoder/map_valid_mask"]

    only_lane = True
    if only_lane:
        map_feature = data_dict["encoder/map_feature"]
        is_lane = map_feature[:, 0, 13] == 1
        # is_lane = is_lane[None].repeat(N, 0)
        valid_map_feat = is_lane & valid_map_feat

    num_agents = agent_positions.shape[1]
    num_steps = agent_positions.shape[0]

    closest_map_features = np.full((num_steps, num_agents), CANT_FIND_DESTINATION, dtype=int)
    dest_valid_mask = np.full((num_steps, num_agents), False, dtype=bool)
    # TODO: Here we don't start at t=10. This might be an issue.
    #  However, as we set all to CANT_FIND_DESTINATION, so it will be free generation for those agent that are
    #  invalid at t=30 but valid at t=10.
    for t in range(0, num_steps):
        future_t = t + FUTURE_STEPS // skip_step
        if future_t >= num_steps:
            break

        future_positions = agent_positions[future_t, :, :2]
        future_valid_mask = agent_valid_mask[future_t]
        future_headings = agent_headings[future_t]

        closest_map_features_now = find_closest_map_feature(
            agent_pos=future_positions,
            agent_heading=future_headings,
            map_pos=map_positions,
            map_heading=map_heading,
            valid_map_feat=valid_map_feat
        )

        # If the agent is static, don't set the destination.
        current_positions = agent_positions[t, :, :2]
        displacement = np.linalg.norm(future_positions - current_positions, axis=-1)
        closest_map_features_now[displacement < 5] = CANT_FIND_DESTINATION

        is_this_dest_valid = (agent_valid_mask[t] & future_valid_mask)

        dest_valid_mask[t:t + 1] = is_this_dest_valid[None]

        # If future pos is invalid, don't set the destination.
        closest_map_features_now[~is_this_dest_valid] = -1

        closest_map_features[t:t+1] = closest_map_features_now[None]

    closest_map_features[~agent_valid_mask] = -1

    closest_map_features[closest_map_features == CANT_FIND_DESTINATION] = -1

    dest = closest_map_features
    dest_valid_mask = dest_valid_mask  # This control what dest to be learned.

    data_dict["decoder/dest_map_index_gt"] = np.copy(dest)
    data_dict["decoder/dest_map_index_valid_mask"] = dest_valid_mask
    if dropout > 0:
        # Randomly drop some destination
        dropout_mask = np.random.rand(*dest.shape) < dropout
        dest[dropout_mask] = -1
    data_dict["decoder/dest_map_index"] = dest
    return data_dict


def process_map_and_traffic_light(
    *, data_dict, scenario, map_feature, dynamic_map_states, track_length, max_vectors, max_map_features,
    max_length_per_map_feature, max_traffic_lights, remove_traffic_light_state, limit_map_range, is_scenestreamer=False
):
    # ========== Find the boundary of the map first ==========
    map_center_info = extract_map_center_heading_locations(map_feature)
    map_center = map_center_info["map_center"]
    map_heading = map_center_info["map_heading"]
    map_feature_augmented = map_center_info["map_feature"]

    # ========== Process Map Features ==========
    # The output is a dict whose keys are the lane ID and key is a state array in shape [T, ???]

    # Get a compact representation of all points in the maps
    map_feature_list = []  # Key: map_feat_id, Value: A dict of processed values
    map_heading_list = []
    map_valid_mask_list = []
    map_position_list = []

    for map_index, (map_feat_id, map_feat) in enumerate(map_feature_augmented.items()):
        rotated_polyline = centralize_to_map_center(
            position_array=map_feat["location"],  # [num points, 2 or 3]
            map_center=map_center,  # [1, 1, 3]
            map_heading=map_heading
        )

        if "polygon" in map_feat:
            # For crosswalk, and other "polygon" based map features, we need to pad the last point to the first point.
            rotated_polyline = np.concatenate([rotated_polyline, rotated_polyline[:1]], axis=0)

        if rotated_polyline.shape[-1] == 2:
            rotated_polyline = np.concatenate([rotated_polyline, np.zeros((rotated_polyline.shape[0], 1))], axis=-1)

        start_points = rotated_polyline[:-1].copy()  # in shape [# map feats - 1, 2]
        end_points = rotated_polyline[1:].copy()  # in shape [# map feats - 1, 2]
        if start_points.shape[0] == 0:
            # A special case here is that the map feature contains only one points.
            # In this case, we suppose the vector has the same point as start point and end point (its len=0)
            start_points = rotated_polyline
            end_points = rotated_polyline
            num_vectors = 1

        else:
            num_vectors = start_points.shape[0]

        assert start_points.ndim == 2  # [num vectors, 3]
        assert start_points.shape[-1] == 3  # [num vectors, 3] # for CAT, start_points.shape[-1] = 2

        direction = end_points - start_points
        heading = np.arctan2(direction[..., 1], direction[..., 0])

        point_diff = np.linalg.norm(direction[..., :2], axis=-1)

        road_length = 0.0
        start_index = 0
        # Iterate over all "vectors" in a map feature.
        # We will produce a map features, containing a set of vectors, in these conditions:
        # (1) If the segment is a lane and has length > MAX_LENGTH_PER_MAP_FEATURE, or
        # (2) The segment has max_vectors vectors, or
        # (3) The segment contains the leftover vectors with less than max_vectors vectors.
        for i in range(num_vectors):
            road_length += point_diff[i]

            # 2025-04-21 Update: Only break the line if it's a lane.
            # map_feat_too_long = (
            #     (road_length >= max_length_per_map_feature) & MetaDriveType.is_lane(map_feat['type'])
            # )
            # # Exempt the crosswalk from the length limit
            map_feat_too_long = (
                (road_length >= max_length_per_map_feature) & ~MetaDriveType.is_crosswalk(map_feat['type'])
            )

            num_valid_vectors = i - start_index + 1

            too_many_vectors = num_valid_vectors >= max_vectors
            last_set_of_vectors = (i == num_vectors - 1) and ((i - start_index) > 0)
            if i - start_index == 0:
                continue
            if map_feat_too_long or too_many_vectors or last_set_of_vectors:
                # The map feature is a 2D array with shape [#vectors, 27].
                # map_feature = np.zeros([i - start_index, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)
                map_feature = np.zeros([max_vectors, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)

                end_index = i + 1
                map_feature[:num_valid_vectors, :3] = start_points[start_index:end_index]
                map_feature[:num_valid_vectors, 3:6] = end_points[start_index:end_index]
                map_feature[:num_valid_vectors, 6:9] = direction[start_index:end_index]
                map_feature[:num_valid_vectors, 9] = utils.wrap_to_pi(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 10] = np.sin(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 11] = np.cos(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 12] = point_diff[start_index:end_index]

                map_feature[:num_valid_vectors, 13] = MetaDriveType.is_lane(map_feat['type'])
                map_feature[:num_valid_vectors, 14] = MetaDriveType.is_sidewalk(map_feat['type'])
                map_feature[:num_valid_vectors, 15] = MetaDriveType.is_road_boundary_line(map_feat['type'])
                map_feature[:num_valid_vectors, 16] = MetaDriveType.is_road_line(map_feat['type'])
                map_feature[:num_valid_vectors, 17] = MetaDriveType.is_broken_line(map_feat['type'])
                map_feature[:num_valid_vectors, 18] = MetaDriveType.is_solid_line(map_feat['type'])
                map_feature[:num_valid_vectors, 19] = MetaDriveType.is_yellow_line(map_feat['type'])
                map_feature[:num_valid_vectors, 20] = MetaDriveType.is_white_line(map_feat['type'])
                map_feature[:num_valid_vectors, 21] = MetaDriveType.is_driveway(map_feat['type'])
                map_feature[:num_valid_vectors, 22] = MetaDriveType.is_crosswalk(map_feat['type'])
                map_feature[:num_valid_vectors, 23] = MetaDriveType.is_speed_bump(map_feat['type'])
                map_feature[:num_valid_vectors, 24] = MetaDriveType.is_stop_sign(map_feat['type'])
                map_feature[:num_valid_vectors, 25] = road_length
                # valid_mask = np.ones_like(start_points[start_index:i, 0])
                map_feature[:num_valid_vectors, 26] = 1

                assert map_feature.shape[0] > 0
                avg_position = ((map_feature[:num_valid_vectors, 0:3] + map_feature[:num_valid_vectors, 3:6]) /
                                2).mean(axis=0)
                avg_heading = utils.wrap_to_pi(utils.average_angles(map_feature[:num_valid_vectors, 9]))

                # if i - start_index < max_vectors:
                #     map_feature = np.pad(map_feature, pad_width=((0, max_vectors - (i - start_index)), (0, 0)))
                #     valid_mask = np.pad(valid_mask, pad_width=(0, max_vectors - (i - start_index)))

                valid_mask = map_feature[:, 26].copy()

                map_feature_list.append(map_feature)
                map_valid_mask_list.append(valid_mask)

                map_heading_list.append(avg_heading)
                map_position_list.append(avg_position)

                start_index = i
                road_length = 0.0

        # if MetaDriveType.is_lane(map_feat['type']):
        #     map_id_of_lanes.append(map_feat_id)

    if len(map_feature_list) == 0:
        map_feature_position = np.zeros([0, 0, 3], dtype=np.float32)
        map_feature_heading = np.zeros([0, 0], dtype=np.float32)
    else:
        map_feature_position = np.stack(map_position_list, axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_heading = np.stack(map_heading_list, axis=0).astype(np.float32)  # [num map feat, 2]
    # print(f"# MAP FEATURES: {len(map_position_list)}, # Avg Vectors: {np.mean(np.sum(map_valid_mask_list, axis=1), axis=0)}, # Max Vectors: {np.max(np.sum(map_valid_mask_list, axis=1), axis=0)}" )

    # Filter out too many map features
    if limit_map_range:
        # Should follow TrafficGen / LCTGen's preprocessing and crop map to
        # 50m range within SDC's position
        sdc_id = scenario['metadata']['sdc_id']
        sdc_tracks = scenario['tracks'][sdc_id]['state']['position']
        current_step = scenario['metadata']['current_time_index']
        sdc_position = sdc_tracks[current_step][..., :2] - map_center[..., :2]

        map_feature_position = np.stack(map_position_list)

        valid_map_feat = (
            (abs(map_feature_position[..., 0] - sdc_position[0]) < 50) &
            (abs(map_feature_position[..., 1] - sdc_position[1]) < 50)
        )
        indices = valid_map_feat.nonzero()[0]
        map_feature_position = map_feature_position[indices]
        map_feature_heading = np.stack([map_feature_heading[i] for i in indices],
                                       axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_list = [map_feature_list[i] for i in indices]
        map_valid_mask_list = [map_valid_mask_list[i] for i in indices]

    # print("Num map features: ", len(map_feature_list), "Max vectors: ", np.max(np.sum(map_valid_mask_list, axis=1)))

    if len(map_feature_position) > max_map_features:
        # Sorted based on the distance to the SDC
        sdc_id = scenario['metadata']['sdc_id']
        sdc_tracks = scenario['tracks'][sdc_id]['state']['position']
        current_step = scenario['metadata']['current_time_index']
        sdc_position = sdc_tracks[current_step][..., :2] - map_center[..., :2]

        dist = np.linalg.norm(map_feature_position[:, :2] - sdc_position[:2], axis=1)

        indices = np.argsort(dist)[:max_map_features]
        map_feature_position = map_feature_position[indices]
        map_feature_heading = np.stack([map_feature_heading[i] for i in indices],
                                       axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_list = [map_feature_list[i] for i in indices]
        map_valid_mask_list = [map_valid_mask_list[i] for i in indices]

    if len(map_valid_mask_list) > 0:
        map_feature = np.stack(map_feature_list, axis=0).astype(np.float32)  # [num map feat, max vectors, 27]
        assert map_feature.shape[-1] == constants.MAP_FEATURE_STATE_DIM
        map_feature_mask = np.stack(map_valid_mask_list, axis=0).astype(bool)  # [num map feat, max vectors]
        map_feature_heading = np.stack(map_feature_heading, axis=0).astype(np.float32)  # [num map feat, max vectors]

    else:
        map_feature = np.zeros([0, max_vectors, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)
        map_feature_mask = np.zeros([0, max_vectors], dtype=bool)
        map_feature_position = np.zeros([0, 3], dtype=np.float32)
        map_feature_heading = np.zeros([0], dtype=np.float32)

    num_map_feat = map_feature.shape[0]
    utils.assert_shape(map_feature, (num_map_feat, max_vectors, constants.MAP_FEATURE_STATE_DIM))
    utils.assert_shape(map_feature_mask, (
        num_map_feat,
        max_vectors,
    ))
    utils.assert_shape(map_feature_position, (num_map_feat, 3))
    utils.assert_shape(map_feature_heading, (num_map_feat, ))

    # num_lights = traffic_light_valid_mask.any(axis=0).sum()
    # print("num_lights: ", num_lights)

    data_dict.update(
        {
            "encoder/map_feature": map_feature,
            "encoder/map_position": map_feature_position,
            "encoder/map_heading": map_feature_heading,
            "encoder/map_valid_mask": map_feature_mask.any(-1),  # Token valid mask
            "encoder/map_feature_valid_mask": map_feature_mask,
            # "encoder/traffic_light_feature": traffic_light_feature,
            # "encoder/traffic_light_position": traffic_light_position,
            # "encoder/traffic_light_heading": traffic_light_heading,
            # "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
            "metadata/map_center": map_center,
            "metadata/map_heading": map_heading,
        }
    )

    if not is_scenestreamer:
        data_dict = process_traffic_light(
            data_dict,
            map_feature,
            dynamic_map_states,
            track_length,
            max_vectors,
            max_map_features,
            max_length_per_map_feature,
            max_traffic_lights,
            map_center,
            map_heading,
            remove_traffic_light_state=remove_traffic_light_state
        )
    else:
        data_dict = process_traffic_light_scenestreamer(
            data_dict,
            map_feature,
            dynamic_map_states,
            track_length,
            max_vectors,
            max_map_features,
            max_length_per_map_feature,
            max_traffic_lights,
            map_center,
            map_heading,
            remove_traffic_light_state=remove_traffic_light_state
        )
    return data_dict


def process_traffic_light(
    data_dict, map_feature, dynamic_map_states, track_length, max_vectors, max_map_features, max_length_per_map_feature,
    max_traffic_lights, map_center, map_heading, remove_traffic_light_state
):

    # ===== Extract traffic light features =====
    traffic_light_position = np.zeros([max_traffic_lights, 3], dtype=np.float32)

    if remove_traffic_light_state:
        traffic_light_heading = np.zeros([max_traffic_lights], dtype=np.float32)
        traffic_light_feature = np.zeros([max_traffic_lights, constants.TRAFFIC_LIGHT_STATE_DIM], dtype=np.float32)
        traffic_light_valid_mask = np.zeros([max_traffic_lights], dtype=bool)
        for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
            traffic_light_state = [v for v in traffic_light["state"]["object_state"] if v is not None]
            tl_states, tl_counts = np.unique(traffic_light_state, return_counts=True)
            tl_state = str(tl_states[np.argmax(tl_counts)])
            stop_point = centralize_to_map_center(
                position_array=traffic_light["stop_point"], map_center=map_center, map_heading=map_heading
            )
            traffic_light_position[tl_count] = stop_point[..., :3]
            traffic_light_feature[tl_count, :3] = stop_point
            traffic_light_feature[tl_count, 3] = MetaDriveType.is_traffic_light_in_green(tl_state)
            traffic_light_feature[tl_count, 4] = MetaDriveType.is_traffic_light_in_yellow(tl_state)
            traffic_light_feature[tl_count, 5] = MetaDriveType.is_traffic_light_in_red(tl_state)
            traffic_light_feature[tl_count, 6] = MetaDriveType.is_traffic_light_unknown(tl_state)
            traffic_light_valid_mask[tl_count] = True
    else:
        traffic_light_heading = np.zeros([
            max_traffic_lights,
        ], dtype=np.float32) + constants.HEADING_PLACEHOLDER
        traffic_light_feature = np.zeros(
            [track_length, max_traffic_lights, constants.TRAFFIC_LIGHT_STATE_DIM], dtype=np.float32
        )
        traffic_light_valid_mask = np.zeros([track_length, max_traffic_lights], dtype=bool)

        for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
            stop_point = centralize_to_map_center(
                position_array=traffic_light["stop_point"], map_center=map_center, map_heading=map_heading
            )

            traffic_light_position[tl_count] = stop_point[..., :3]
            for step in range(track_length):
                assert traffic_light['type'] == MetaDriveType.TRAFFIC_LIGHT
                traffic_light_state = {k: v[step] for k, v in traffic_light["state"].items()}
                traffic_light_feature[step, tl_count, :3] = stop_point
                traffic_light_feature[step, tl_count,
                                      3] = MetaDriveType.is_traffic_light_in_green(traffic_light_state["object_state"])
                traffic_light_feature[step, tl_count, 4] = MetaDriveType.is_traffic_light_in_yellow(
                    traffic_light_state["object_state"]
                )
                traffic_light_feature[step, tl_count,
                                      5] = MetaDriveType.is_traffic_light_in_red(traffic_light_state["object_state"])
                traffic_light_feature[step, tl_count,
                                      6] = MetaDriveType.is_traffic_light_unknown(traffic_light_state["object_state"])
                traffic_light_valid_mask[step, tl_count] = True
            if tl_count > max_traffic_lights:
                logger.debug(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")
                print(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")
                break

    data_dict.update(
        {
            "encoder/traffic_light_feature": traffic_light_feature,
            "encoder/traffic_light_position": traffic_light_position,
            "encoder/traffic_light_heading": traffic_light_heading,
            "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
        }
    )
    return data_dict



def process_traffic_light_scenestreamer(
    data_dict, map_feature, dynamic_map_states, track_length, max_vectors, max_map_features, max_length_per_map_feature,
    max_traffic_lights, map_center, map_heading, remove_traffic_light_state
):
    assert remove_traffic_light_state is False

    L = len(dynamic_map_states)

    if L == 0:
        L = 1

    traffic_light_position = np.zeros([L, 3], dtype=np.float32)
    # traffic_light_heading = np.zeros([L], dtype=np.float32)

    # TODO: hardcoded
    count = len(range(0, track_length, 5))
    traffic_light_state_np = np.zeros([count, L,], dtype=int)
    traffic_light_valid_mask = np.zeros([count, L], dtype=bool)

    # Find closest map feature
    for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
        stop_point = centralize_to_map_center(position_array=traffic_light["stop_point"], map_center=map_center, map_heading=map_heading)
        traffic_light_position[tl_count] = stop_point[..., :3]

    # Find the closest map feature
    map_pos = data_dict["encoder/map_position"]
    map_headings = data_dict["encoder/map_heading"]
    valid_map_feat = data_dict["encoder/map_valid_mask"]
    dist = np.linalg.norm((traffic_light_position[:, None, :2] - map_pos[None, :, :2]), axis=-1)
    assert valid_map_feat.all()
    closest_map_id = np.argmin(dist, axis=1)
    # closest_dist = np.min(dist, axis=1)
    traffic_light_heading = map_headings[closest_map_id]


    for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
        step_compressed = 0
        for step in range(0, track_length, 5):
            # TODO: Here we hardcode the step to 5. This might be an issue.
            assert traffic_light['type'] == MetaDriveType.TRAFFIC_LIGHT
            traffic_light_state = {k: v[step] for k, v in traffic_light["state"].items()}
            is_green = MetaDriveType.is_traffic_light_in_green(traffic_light_state["object_state"])
            is_yellow = MetaDriveType.is_traffic_light_in_yellow(traffic_light_state["object_state"])
            is_red = MetaDriveType.is_traffic_light_in_red(traffic_light_state["object_state"])
            is_unknown = MetaDriveType.is_traffic_light_unknown(traffic_light_state["object_state"])
            # Convert to int 0~3:
            if is_unknown:
                traffic_light_state_np[step_compressed, tl_count] = 0
            elif is_green:
                traffic_light_state_np[step_compressed, tl_count] = 1
            elif is_yellow:
                traffic_light_state_np[step_compressed, tl_count] = 2
            elif is_red:
                traffic_light_state_np[step_compressed, tl_count] = 3
            else:
                raise ValueError(f"Unknown traffic light state: {traffic_light_state}")
            traffic_light_valid_mask[step_compressed, tl_count] = True
            step_compressed += 1
        if tl_count > max_traffic_lights:
            logger.debug(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")
            print(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")

    valid_tl = traffic_light_valid_mask.any(axis=0)
    data_dict.update(
        {
            # "encoder/traffic_light_feature": traffic_light_feature,
            "encoder/traffic_light_position": traffic_light_position * valid_tl[:, None],
            "encoder/traffic_light_heading": traffic_light_heading * valid_tl,
            "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
            "encoder/traffic_light_state": traffic_light_state_np,
            "encoder/traffic_light_map_id": closest_map_id * valid_tl,
        }
    )
    return data_dict


def filter_and_reorder_agent(data_dict, max_agents=None):
    """
    Put modeled agents and SDC to the first place.
    """
    num_agents = data_dict["encoder/agent_feature"].shape[1]
    agent_valid_mask = data_dict["encoder/agent_valid_mask"]
    modeled_agent_indices = data_dict["encoder/object_of_interest_id"]

    sdc_index = data_dict["encoder/sdc_index"]
    new_sdc_index = sdc_index

    # Sort agent based on validity. Put useless agent to the back.
    index_to_validity = []
    for agent_index in range(num_agents):
        index_to_validity.append((agent_index, agent_valid_mask[:, agent_index].sum()))
    sorted_indices = sorted(index_to_validity, key=lambda v: v[1], reverse=True)
    selected_agents = [key for key, _ in sorted_indices]

    if modeled_agent_indices is not None:
        for agent_index in modeled_agent_indices:
            selected_agents.remove(agent_index)
            selected_agents.insert(0, int(agent_index))

    # Put SDC to first place.
    assert sdc_index in selected_agents
    selected_agents.remove(sdc_index)
    selected_agents.insert(0, sdc_index)
    new_sdc_index = 0
    # new_sdc_index = selected_agents.index(sdc_index)

    if max_agents is not None:
        selected_agents = selected_agents[:max_agents]

    selected_agents = np.asarray(selected_agents, dtype=int)

    # ===== Reorder all data =====
    # Those data whose first dim is the agent dim:
    for key in [
            "encoder/agent_type",
            "encoder/current_agent_shape",
            "encoder/current_agent_valid_mask",
            "encoder/current_agent_position",
            "encoder/current_agent_heading",
            "encoder/current_agent_velocity",
            "encoder/track_name",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=selected_agents, agent_dim=0)
    # Those data whose second dim is the agent dim:
    for key in [
            "encoder/agent_feature",
            "encoder/agent_valid_mask",
            "encoder/agent_position",
            "encoder/agent_velocity",
            "encoder/agent_heading",
            # "encoder/future_agent_position",
            # "encoder/future_agent_heading",
            # "encoder/future_agent_valid_mask",
            "encoder/agent_shape",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=selected_agents, agent_dim=1)

    # ===== Reorder modeled agents and SDC, change modeled_agent_indices if necessary =====
    if modeled_agent_indices is not None:
        # Need to translate track_index_to_predict
        new_modeled_agent_indices = []
        for old_agent_index in modeled_agent_indices:
            for new_ind, old_ind in enumerate(selected_agents):
                if old_agent_index == old_ind:
                    new_modeled_agent_indices.append(new_ind)
                    break
        assert len(new_modeled_agent_indices) == len(modeled_agent_indices)
        modeled_agent_indices = new_modeled_agent_indices
    new_sdc_index = 0
    if modeled_agent_indices is not None:
        # Also update SDC index
        if new_sdc_index in modeled_agent_indices:
            modeled_agent_indices.remove(new_sdc_index)
            modeled_agent_indices.insert(0, new_sdc_index)

    data_dict["encoder/sdc_index"] = new_sdc_index
    data_dict["encoder/object_of_interest_id"] = np.asarray(modeled_agent_indices)
    # Note that new ooi id doesn't change the order. So no need to change ooi name.

    assert bool(data_dict["encoder/current_agent_valid_mask"][new_sdc_index]) is True

    return data_dict


def filter_and_reorder_agent_for_scenestreamer(data_dict, max_agents=None):
    agent_valid_mask = data_dict["encoder/agent_valid_mask"]
    modeled_agent_indices = data_dict["encoder/object_of_interest_id"]

    sdc_index = data_dict["encoder/sdc_index"]

    default_max_agents = 128

    def _get_first_last_pos(pos, valid_mask):
        T, N = valid_mask.shape
        ind = np.arange(T).reshape(-1, 1).repeat(N, axis=1)  # T, N
        ind[~valid_mask] = 0
        ind = ind.max(axis=0)
        last = np.take_along_axis(pos, indices=ind.reshape(1, N, 1), axis=0)
        last = np.squeeze(last, axis=0)

        # Find the index of the first True (or 1) along axis 0 (time) for each agent
        # First, create a mask of where any True exists per column
        has_valid = valid_mask.any(axis=0)

        # Use argmax along time axis: this returns first occurrence of maximum (i.e. True)
        first_idx = valid_mask.argmax(axis=0)

        # Set result to -1 where there was no valid entry
        first_idx[~has_valid] = -1

        first = np.take_along_axis(pos, indices=first_idx.reshape(1, N, 1), axis=0)
        first = np.squeeze(first, axis=0)
        return first, last

    agent_types = data_dict["encoder/agent_type"]
    agent_position = data_dict["encoder/agent_position"]
    current_valid_mask = data_dict["encoder/current_agent_valid_mask"]
    current_valid_agent_id = current_valid_mask.nonzero()[0]

    first_pos, last_pos = _get_first_last_pos(agent_position, agent_valid_mask)
    moving_dist = np.linalg.norm((last_pos-first_pos)[:, :2], axis=-1)
    moving_dist[~current_valid_mask] = -1000

    # force to add all non-vehicle agent
    selected_agents = np.argsort(moving_dist)[::-1]

    # Remove agent id that are not in current_valid_agent_id
    selected_agents = np.array([i for i in selected_agents if i in current_valid_agent_id])

    if max_agents is not None:
        all_128_selected_agents = selected_agents[:default_max_agents]
    else:
        all_128_selected_agents = selected_agents

    all_128_selected_agents = all_128_selected_agents.tolist()
    if modeled_agent_indices is not None:
        for agent_index in modeled_agent_indices:
            if agent_index in all_128_selected_agents:
                all_128_selected_agents.remove(agent_index)
            all_128_selected_agents.insert(0, int(agent_index))
    # Put SDC to first place.
    if sdc_index in all_128_selected_agents:
        all_128_selected_agents.remove(sdc_index)
    all_128_selected_agents.insert(0, sdc_index)

    if max_agents is not None:
        all_128_selected_agents = all_128_selected_agents[:default_max_agents]

    # reorganize the order of the agents based on their types
    tmpagent_types = agent_types[all_128_selected_agents]
    all_128_selected_agents = np.asarray(all_128_selected_agents, dtype=int)
    new_selected_agents = []
    for atype in [1, 2, 3]:
        if atype in tmpagent_types:
            atype_ids = np.where(tmpagent_types == atype)[0].astype(int)
            atype_ids = all_128_selected_agents[atype_ids]
            new_selected_agents += list(atype_ids)
    all_128_selected_agents = np.asarray(new_selected_agents, dtype=int)

    # In those all 128 selected agents, we need to filter out
    if max_agents is not None and len(all_128_selected_agents) > max_agents:
        # Do second round of filtering, but this time we only set invalidity
        new_moving_dist = moving_dist[all_128_selected_agents]
        new_selected_agents = np.argsort(new_moving_dist)[::-1]
        new_selected_agents = new_selected_agents[:max_agents]
        new_selected_agents = all_128_selected_agents[new_selected_agents]
        new_selected_agents = new_selected_agents.tolist()
        if modeled_agent_indices is not None:
            for agent_index in modeled_agent_indices:
                assert agent_index in all_128_selected_agents
                if agent_index in new_selected_agents:
                    new_selected_agents.remove(agent_index)
                new_selected_agents.insert(0, int(agent_index))
        # Put SDC to first place.
        if sdc_index in new_selected_agents:
            new_selected_agents.remove(sdc_index)
        new_selected_agents.insert(0, sdc_index)
        new_selected_agents = new_selected_agents[:max_agents]
        new_selected_agents = np.asarray(new_selected_agents, dtype=int)

        valid_mask = np.zeros((agent_position.shape[1],), dtype=bool)
        valid_mask[new_selected_agents] = True
        data_dict["encoder/current_agent_valid_mask"] = np.logical_and(
            data_dict["encoder/current_agent_valid_mask"], valid_mask
        )
        data_dict["encoder/agent_valid_mask"] = np.logical_and(
            data_dict["encoder/agent_valid_mask"], valid_mask[None]
        )

    assert all_128_selected_agents[0] == sdc_index

    # ===== Reorder all data =====
    # Those data whose first dim is the agent dim:
    for key in [
            "encoder/agent_type",
            "encoder/current_agent_shape",
            "encoder/current_agent_valid_mask",
            "encoder/current_agent_position",
            "encoder/current_agent_heading",
            "encoder/current_agent_velocity",
            "encoder/track_name",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=all_128_selected_agents, agent_dim=0)
    # Those data whose second dim is the agent dim:
    for key in [
            "encoder/agent_feature",
            "encoder/agent_valid_mask",
            "encoder/agent_position",
            "encoder/agent_velocity",
            "encoder/agent_heading",
            # "encoder/future_agent_position",
            # "encoder/future_agent_heading",
            # "encoder/future_agent_valid_mask",
            "encoder/agent_shape",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=all_128_selected_agents, agent_dim=1)

    # ===== Reorder modeled agents and SDC, change modeled_agent_indices if necessary =====
    if modeled_agent_indices is not None:
        # Need to translate track_index_to_predict
        new_modeled_agent_indices = []
        for old_agent_index in modeled_agent_indices:
            for new_ind, old_ind in enumerate(all_128_selected_agents):
                if old_agent_index == old_ind:
                    new_modeled_agent_indices.append(new_ind)
                    break
        assert len(new_modeled_agent_indices) == len(modeled_agent_indices)
        modeled_agent_indices = new_modeled_agent_indices
    new_sdc_index = 0
    if modeled_agent_indices is not None:
        # Also update SDC index
        if new_sdc_index in modeled_agent_indices:
            modeled_agent_indices.remove(new_sdc_index)
            modeled_agent_indices.insert(0, new_sdc_index)

    data_dict["encoder/sdc_index"] = new_sdc_index
    data_dict["encoder/object_of_interest_id"] = np.asarray(modeled_agent_indices)
    # Note that new ooi id doesn't change the order. So no need to change ooi name.

    assert data_dict["encoder/sdc_index"] in list(data_dict["encoder/current_agent_valid_mask"].nonzero()[0])

    return data_dict


def process_track(
    *,
    data_dict,
    tracks,
    track_length,
    sdc_name,  # We need to translate sdc_name to sdc_index
    max_agents,
    exempt_max_agent_filtering=False,
    is_scenestreamer=False,
):
    map_center = data_dict["metadata/map_center"]
    map_heading = data_dict["metadata/map_heading"]
    current_t = data_dict["metadata/current_time_index"]

    agent_feature_dict = {}
    agent_valid_mask_dict = {}
    agent_velocity_dict = {}
    agent_position_dict = {}
    agent_heading_dict = {}
    agent_type_dict = {}
    agent_shape_dict = {}
    sdc_index = None
    sdc_name = str(sdc_name)

    valid_track_names = []
    track_count = 0

    for _, (track_name, cur_data) in enumerate(tracks.items()):  # number of objects

        # if not cur_data['type'] == 'VEHICLE': # CAT contains pedestrains which does not contain length, width, and height
        #     continue

        if not MetaDriveType.is_participant(cur_data["type"]):
            # TODO(pzh): TrafficCone is in tracks for some reason. Looks very weird. Might be some bug.
            continue
        track_name = str(track_name)

        if track_name == sdc_name:
            sdc_index = track_count

        cur_state = cur_data[SD.STATE]

        rotated_positions = centralize_to_map_center(
            position_array=cur_state["position"],  # [T, 3]
            map_center=map_center,
            map_heading=map_heading
        )  # [T, num agents, 3]

        rotated_heading = utils.wrap_to_pi(cur_state["heading"] - map_heading)  # [T, num agents]
        rotated_velocity = centralize_to_map_center(
            position_array=cur_state["velocity"], map_center=None, map_heading=map_heading
        )[..., :2]  # [T, num agents, 2]

        agent_shape_dict[track_name] = np.stack(
            [cur_state["length"].reshape(-1), cur_state["width"].reshape(-1), cur_state["height"].reshape(-1)], axis=1
        )  # (T, N, 3)

        speed = np.linalg.norm(cur_state["velocity"], axis=1)

        valid_mask = np.asarray(cur_state["valid"], dtype=bool)

        agent_state = np.zeros([track_length, constants.AGENT_STATE_DIM], dtype=np.float32)

        # print("shape of rotated_positions", rotated_positions.shape)
        # for CAT: we need to pad position dimension
        if rotated_positions.shape[1] != 3:
            rotated_positions = np.concatenate([rotated_positions, np.zeros((rotated_positions.shape[0], 1))], axis=-1)

        agent_state[:, :3] = rotated_positions
        agent_state[:, 3] = rotated_heading
        agent_state[:, 4] = np.sin(rotated_heading)
        agent_state[:, 5] = np.cos(rotated_heading)
        agent_state[:, 6:8] = rotated_velocity

        agent_state[:, 8] = speed

        agent_state[:, 9] = cur_state["length"].reshape(-1)
        agent_state[:, 10] = cur_state["width"].reshape(-1)
        agent_state[:, 11] = cur_state["height"].reshape(-1)

        agent_state[~valid_mask] = 0
        agent_state[:, 12] = MetaDriveType.is_vehicle(cur_data["type"])
        agent_state[:, 13] = MetaDriveType.is_pedestrian(cur_data["type"])
        agent_state[:, 14] = MetaDriveType.is_cyclist(cur_data["type"])
        agent_state[:, 15] = valid_mask

        # TODO(pzh): Remove mapping
        assert cur_data["type"] in constants.object_type_to_int

        agent_feature_dict[track_name] = agent_state
        agent_valid_mask_dict[track_name] = valid_mask

        agent_position_dict[track_name] = rotated_positions * valid_mask.reshape(-1, 1)
        agent_heading_dict[track_name] = rotated_heading * valid_mask
        agent_velocity_dict[track_name] = rotated_velocity * valid_mask.reshape(-1, 1)

        # TODO(pzh): Remove mapping
        agent_type_dict[track_name] = constants.object_type_to_int[cur_data["type"]]

        valid_track_names.append(str(track_name))

        track_count += 1

    assert sdc_index is not None

    # ===== Store all data into dict =====
    agent_feature = np.stack(list(agent_feature_dict.values()), axis=1)  # [T, ]
    num_agents = agent_feature.shape[1]
    utils.assert_shape(agent_feature, (track_length, num_agents, constants.AGENT_STATE_DIM))

    agent_valid_mask = np.stack(list(agent_valid_mask_dict.values()), axis=1).astype(bool)
    utils.assert_shape(agent_valid_mask, (
        track_length,
        num_agents,
    ))

    agent_position = np.stack(list(agent_position_dict.values()), axis=1)
    utils.assert_shape(agent_position, (track_length, num_agents, 3))

    agent_velocity = np.stack(list(agent_velocity_dict.values()), axis=1)
    utils.assert_shape(agent_velocity, (track_length, num_agents, 2))

    agent_heading = np.stack(list(agent_heading_dict.values()), axis=1)
    utils.assert_shape(agent_heading, (track_length, num_agents))

    agent_type = np.stack(list(agent_type_dict.values()), axis=0).astype(int)
    utils.assert_shape(agent_type, (num_agents, ))

    agent_shape = np.stack(list(agent_shape_dict.values()), axis=1)
    utils.assert_shape(agent_shape, (track_length, num_agents, 3))

    data_dict["encoder/agent_feature"] = agent_feature.astype(np.float32)  # [T, num agent, D_agent]
    data_dict["encoder/agent_valid_mask"] = agent_valid_mask.astype(bool)  # [T, num agent]
    data_dict["encoder/agent_position"] = agent_position.astype(np.float32)
    data_dict["encoder/agent_velocity"] = agent_velocity.astype(np.float32)
    data_dict["encoder/agent_heading"] = agent_heading.astype(np.float32)
    data_dict["encoder/agent_type"] = agent_type

    # data_dict["encoder/future_agent_position"] = agent_position.astype(np.float32)[current_t + 1:]
    # data_dict["encoder/future_agent_heading"] = agent_heading.astype(np.float32)[current_t + 1:]
    # data_dict["encoder/future_agent_valid_mask"] = agent_valid_mask.astype(bool)[current_t + 1:]
    # data_dict["encoder/future_agent_velocity"] = agent_velocity.astype(np.float32)[current_t + 1:]
    data_dict["encoder/current_agent_valid_mask"] = agent_valid_mask.astype(bool)[current_t]
    data_dict["encoder/current_agent_position"] = agent_position.astype(np.float32)[current_t]
    data_dict["encoder/current_agent_heading"] = agent_heading.astype(np.float32)[current_t]
    data_dict["encoder/current_agent_velocity"] = agent_velocity.astype(np.float32)[current_t]

    data_dict["encoder/track_name"] = np.array(valid_track_names, dtype=str)
    data_dict["encoder/agent_shape"] = agent_shape.astype(np.float32)
    data_dict["encoder/current_agent_shape"] = data_dict["encoder/agent_shape"][current_t]
    data_dict["encoder/sdc_index"] = sdc_index

    # ===== Process the case where the number of agents exceeds max_agents =====
    if is_scenestreamer:
        data_dict = filter_and_reorder_agent_for_scenestreamer(data_dict, max_agents=max_agents if not exempt_max_agent_filtering else None)
    else:
        data_dict = filter_and_reorder_agent(data_dict, max_agents=max_agents if not exempt_max_agent_filtering else None)

    # Add agent ID:
    num_agents = data_dict["encoder/agent_feature"].shape[1]
    data_dict["encoder/agent_id"] = np.arange(num_agents)

    # assert (data_dict["decoder/current_agent_valid_mask"] == data_dict["encoder/agent_valid_mask"][current_t]).all()
    assert data_dict["encoder/sdc_index"] in list(data_dict["encoder/current_agent_valid_mask"].nonzero()[0])

    return data_dict


def prepare_modeled_agent_and_eval_data(
    data_dict, predict_all_agents, eval_all_agents, current_t, add_sdc_to_object_of_interest
):
    # ===== Need to extract only the modeled agents for decoder and GT =====
    object_of_interest = data_dict["encoder/object_of_interest_id"]

    if predict_all_agents:
        modeled_agent_indices = list(data_dict["encoder/current_agent_valid_mask"].nonzero()[0])

        # In the following code, we will select only the valid agents at this step as modeled agents.
        # After the selection, the order of agents will change (again ..). So the object_of_interests
        # should also be changed.

        new_object_of_interests = []
        for old_agent_index in object_of_interest:
            for new_ind, old_ind in enumerate(modeled_agent_indices):
                if old_agent_index == old_ind:
                    new_object_of_interests.append(new_ind)
                    break
        assert len(new_object_of_interests) == len(object_of_interest)

        assert data_dict["encoder/sdc_index"] in modeled_agent_indices
        data_dict["decoder/sdc_index"] = modeled_agent_indices.index(data_dict["encoder/sdc_index"])
        data_dict["decoder/object_of_interest_id"] = np.asarray(new_object_of_interests)
        # Note that new ooi id doesn't change the order. So no need to change ooi name

    else:
        raise ValueError("Not sure what will happen...")
        object_of_interest = data_dict["encoder/object_of_interest_id"]
        modeled_agent_indices = object_of_interest
        # object_of_interest don't change
        assert eval_all_agents is False
        data_dict["decoder/object_of_interest_id"] = np.arange(len(object_of_interest))

    data_dict["decoder/agent_id"] = np.arange(len(modeled_agent_indices))

    assert modeled_agent_indices is not None

    data_dict["decoder/agent_type"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_type"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/track_name"] = extract_data_by_agent_indices(
        data_dict["encoder/track_name"], modeled_agent_indices, agent_dim=0, fill=-1
    )
    data_dict["encoder/modeled_agent_id"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_id"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["encoder/modeled_agent_type"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_type"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_valid_mask"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_valid_mask"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_position"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_position"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_heading"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_heading"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_shape"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_shape"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_velocity"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_velocity"], agent_indices=modeled_agent_indices, agent_dim=0
    )

    # agent_dim = 1
    # data_dict["decoder/future_agent_position"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_position"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_heading"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_heading"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_velocity"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_velocity"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_valid_mask"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_valid_mask"], agent_indices=modeled_agent_indices, agent_dim=1
    # )

    data_dict["decoder/agent_position"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_position"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_velocity"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_velocity"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_heading"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_heading"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_valid_mask"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_valid_mask"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_shape"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_shape"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/object_of_interest_name"] = data_dict["encoder/object_of_interest_name"]

    if add_sdc_to_object_of_interest:

        if data_dict["metadata/sdc_name"] not in data_dict["encoder/object_of_interest_name"]:
            data_dict["encoder/object_of_interest_name"] = np.concatenate(
                [[data_dict["metadata/sdc_name"]], data_dict["encoder/object_of_interest_name"]]
            )
        else:
            assert data_dict["metadata/sdc_name"] == data_dict["encoder/object_of_interest_name"][0]

        if data_dict["metadata/sdc_name"] not in data_dict["decoder/object_of_interest_name"]:
            data_dict["decoder/object_of_interest_name"] = np.concatenate(
                [[data_dict["metadata/sdc_name"]], data_dict["encoder/object_of_interest_name"]]
            )
        else:
            assert data_dict["metadata/sdc_name"] == data_dict["decoder/object_of_interest_name"][0]

    # Evaluation data: all with leading dimensions: (num of interested objects, T, ...)
    # If not eval all agents, a new index system `eval/` is introduced.
    if eval_all_agents:
        pass
        # data_dict["eval/track_name"] = data_dict["decoder/track_name"]
        # data_dict["eval/agent_type"] = data_dict["decoder/agent_type"]
        # data_dict["eval/agent_position"] = data_dict["decoder/agent_position"]
        # data_dict["eval/agent_velocity"] = data_dict["decoder/agent_velocity"]
        # data_dict["eval/agent_heading"] = data_dict["decoder/agent_heading"]
        # data_dict["eval/agent_valid_mask"] = data_dict["decoder/agent_valid_mask"]
        # data_dict["eval/agent_shape"] = data_dict["decoder/agent_shape"]
    else:
        assert new_object_of_interests is not None
        decoder_ooi_id = new_object_of_interests
        data_dict["eval/track_name"] = extract_data_by_agent_indices(
            data_dict["decoder/track_name"], decoder_ooi_id, agent_dim=0
        )
        data_dict["eval/agent_type"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_type"], decoder_ooi_id, agent_dim=0
        )
        data_dict["eval/agent_position"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_position"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_velocity"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_velocity"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_heading"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_heading"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_valid_mask"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_valid_mask"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_shape"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_shape"], decoder_ooi_id, agent_dim=1
        )
        assert data_dict["eval/agent_valid_mask"][current_t].all()  # not all object_of_interest is in CAT

    return data_dict


def preprocess_scenario_description(*args, config, **kwargs):
    # if scenario['length'] < 5: # TODO: filter out CAT data that is not valid
    #     return None
    # TODO: combine all cat info dictionary .pkl files into one and provide the paths in the config
    if config.MODEL.NAME in ["motionlm", "language_motionlm", "gpt", "scenestreamer"]:
        return preprocess_scenario_description_for_motionlm(*args, config=config, **kwargs)
    elif config.MODEL.NAME == "gen":
        return preprocess_scenario_description_for_gen(scenario, config, in_evaluation, keep_all_data)
    # elif config.MODEL.NAME == "scenestreamer":
    #     return preprocess_scenario_description_for_scenestreamer(scenario, config, in_evaluation, keep_all_data)
    else:
        raise ValueError(f"Unknown model name: {config.MODEL.NAME}")


def prepare_trafficgen_data(
    data_dict,
    config,
    scenario,
    force_t=True,  # Just disable this function...
    only_lane=False,
):

    sdc_index = data_dict["decoder/sdc_index"]
    T = data_dict["encoder/agent_feature"].shape[0]
    if force_t:
        # if 180 <= T <= 200:
        #     assert data_dict["metadata/current_time_index"] == 0
        #     current_t = 0
        # elif T == 91:
        current_t = data_dict["metadata/current_time_index"]
        # else:
        #     raise ValueError(f"Unknown T: {T}")

    else:
        current_t = np.random.randint(0, T)

    start_action_id = config.PREPROCESSING.MAX_MAP_FEATURES
    end_action_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1

    # Note that here we reuse the "current_agent".
    pos = data_dict["decoder/agent_position"][current_t, ..., :2]
    heading = data_dict["decoder/agent_heading"][current_t]
    valid = data_dict["decoder/agent_valid_mask"][current_t]
    vel = data_dict["decoder/agent_velocity"][current_t]

    agent_type = data_dict["decoder/agent_type"]  # in 123
    agent_type = np.clip(agent_type - 1, 0, 2)  # in 012

    current_agent_shape = data_dict["decoder/current_agent_shape"]
    # assert valid.all()
    N = len(pos)

    # Randomize the agent order (but still put SDC in the first place)
    agent_id = np.arange(len(pos))
    agent_id = agent_id[agent_id != sdc_index]
    randomized_agent_id = np.random.permutation(agent_id)
    randomized_agent_id = np.concatenate([np.array([sdc_index]), randomized_agent_id], axis=0)
    agent_type = agent_type[randomized_agent_id]
    pos = pos[randomized_agent_id]
    heading = heading[randomized_agent_id]
    vel = vel[randomized_agent_id]
    current_agent_shape = current_agent_shape[randomized_agent_id]

    # Filter map feature and only keep lanes:
    # map_feature = data_dict["encoder/map_feature"]
    # is_lane = map_feature[:, 0, 13] == 1
    map_pos = data_dict["encoder/map_position"][..., :2]
    map_heading = data_dict["encoder/map_heading"]

    # Get map feature valid mask
    valid_map_feat = data_dict["encoder/map_valid_mask"]
    heading_diff = utils.wrap_to_pi(heading[:, None] - map_heading[None])
    valid_heading = np.abs(heading_diff) < np.deg2rad(90)
    valid_map_feat = valid_map_feat & valid_heading

    if only_lane:
        map_feature = data_dict["encoder/map_feature"]
        is_lane = map_feature[:, 0, 13] == 1
        is_lane = is_lane[None].repeat(N, 0)
        valid_map_feat = is_lane & valid_map_feat

    # Find the closest map feature
    dist = np.linalg.norm((pos[:, None] - map_pos[None])[..., :2], axis=-1)
    dist[~valid_map_feat] = np.inf
    closest_map_feat = np.argmin(dist, axis=1)
    # closest_map_dist = dist[np.arange(N), closest_map_feat]

    # Set invalid if an agent is far away from the center of the map feat.
    # By saying far I mean exceeding the length of the map feat.
    # map_feat_length = map_feature[..., 25].max(-1)[closest_map_feat]
    # valid_mask = closest_map_dist < map_feat_length
    # valid_mask = np.ones(N, dtype=bool)
    valid_mask = valid

    # Get the selected map feature
    selected_map_pos = map_pos[closest_map_feat]
    selected_map_heading = map_heading[closest_map_feat]

    # Get relative information
    relative_pos = pos - selected_map_pos
    relative_pos = utils.rotate(x=relative_pos[:, 0], y=relative_pos[:, 1], angle=-selected_map_heading)
    relative_heading = utils.wrap_to_pi(heading - selected_map_heading)
    relative_vel = utils.rotate(x=vel[:, 0], y=vel[:, 1], angle=-selected_map_heading)

    # Filter out the agents that are out of the scope
    valid_mask = (
        valid_mask & (relative_pos[:, 0] >= TrafficGenTokenizer.limit["position_x"][0]) &
        (relative_pos[:, 0] <= TrafficGenTokenizer.limit["position_x"][1]) &
        (relative_pos[:, 1] >= TrafficGenTokenizer.limit["position_y"][0]) &
        (relative_pos[:, 1] <= TrafficGenTokenizer.limit["position_y"][1]) &
        (relative_heading >= TrafficGenTokenizer.limit["heading"][0]) &
        (relative_heading <= TrafficGenTokenizer.limit["heading"][1])
    )

    if config.FOLLOW_TRAFFICGEN:
        tg_select_index = _get_trafficgen_data(
            raw_scenario_description=scenario, data_dict=data_dict, current_t=current_t
        )
        new_valid_mask = np.zeros_like(valid_mask)
        new_valid_mask[tg_select_index] = True
        valid_mask = valid_mask & new_valid_mask

    if not valid_mask.any() and force_t is False:
        return prepare_trafficgen_data(data_dict, config, force_t=True)

    pos = pos[valid_mask]
    heading = heading[valid_mask]
    vel = vel[valid_mask]
    agent_type = agent_type[valid_mask]
    current_agent_shape = current_agent_shape[valid_mask]
    relative_pos = relative_pos[valid_mask]
    relative_heading = relative_heading[valid_mask]
    relative_vel = relative_vel[valid_mask]
    selected_map_pos = selected_map_pos[valid_mask]
    selected_map_heading = selected_map_heading[valid_mask]
    closest_map_feat = closest_map_feat[valid_mask]
    valid_mask = valid_mask[valid_mask]

    # Get the discretized relative position
    gt_position_x = TrafficGenTokenizer.bucketize(relative_pos[:, 0], "position_x")
    gt_position_y = TrafficGenTokenizer.bucketize(relative_pos[:, 1], "position_y")
    relative_pos_x = TrafficGenTokenizer.de_bucketize(gt_position_x, "position_x")
    relative_pos_y = TrafficGenTokenizer.de_bucketize(gt_position_y, "position_y")

    # Reconstruct the position with the bucketized value
    relative_pos = np.stack([relative_pos_x, relative_pos_y], axis=1)
    pos = utils.rotate(x=relative_pos_x, y=relative_pos_y, angle=selected_map_heading) + selected_map_pos

    # Reconstruct the heading and velocity
    gt_heading = TrafficGenTokenizer.bucketize(relative_heading, "heading")
    relative_heading = TrafficGenTokenizer.de_bucketize(gt_heading, "heading")
    heading = utils.wrap_to_pi(relative_heading + selected_map_heading)

    # Reconstruct the velocity
    gt_vel_x = TrafficGenTokenizer.bucketize(relative_vel[:, 0], "velocity_x")
    gt_vel_y = TrafficGenTokenizer.bucketize(relative_vel[:, 1], "velocity_y")
    relative_vel_x = TrafficGenTokenizer.de_bucketize(gt_vel_x, "velocity_x")
    relative_vel_y = TrafficGenTokenizer.de_bucketize(gt_vel_y, "velocity_y")
    relative_vel = np.stack([relative_vel_x, relative_vel_y], axis=1)
    vel = utils.rotate(x=relative_vel_x, y=relative_vel_y, angle=selected_map_heading)

    # Reconstruct shape
    gt_shape_l = TrafficGenTokenizer.bucketize(current_agent_shape[:, 0], "length")
    gt_shape_w = TrafficGenTokenizer.bucketize(current_agent_shape[:, 1], "width")
    gt_shape_h = TrafficGenTokenizer.bucketize(current_agent_shape[:, 2], "height")
    current_agent_shape = np.stack(
        [
            TrafficGenTokenizer.de_bucketize(gt_shape_l, "length"),
            TrafficGenTokenizer.de_bucketize(gt_shape_w, "width"),
            TrafficGenTokenizer.de_bucketize(gt_shape_h, "height")
        ],
        axis=1
    )

    # ===== Fill in the data for trafficgen =====
    data_dict["decoder/input_action_for_trafficgen"] = np.concatenate(
        [[start_action_id], closest_map_feat, [end_action_id]]
    ).astype(int)
    data_dict["decoder/input_action_valid_mask_for_trafficgen"] = np.concatenate([[1], valid_mask, [1]],
                                                                                 axis=0).astype(bool)
    data_dict["decoder/modeled_agent_position_for_trafficgen"] = np.concatenate([[[0, 0]], pos, [[0, 0]]],
                                                                                axis=0).astype(np.float32)

    data_dict["decoder/modeled_agent_velocity_for_trafficgen"] = np.concatenate([[[0, 0]], vel, [[0, 0]]],
                                                                                axis=0).astype(np.float32)

    data_dict["decoder/modeled_agent_heading_for_trafficgen"] = np.concatenate([[0], heading, [0]],
                                                                               axis=0).astype(np.float32)
    data_dict["decoder/current_agent_shape_for_trafficgen"] = np.concatenate(
        [[[0, 0, 0]], current_agent_shape, [[0, 0, 0]]], axis=0
    ).astype(np.float32)
    data_dict["decoder/agent_type_for_trafficgen"] = np.concatenate([[0], agent_type, [0]], axis=0).astype(int)

    feat = np.zeros((len(pos) + 2, 5), dtype=np.float32)
    # import matplotlib.pyplot as plt;plt.scatter(relative_pos[:, 0], relative_pos[:, 1]);plt.show()
    feat[1:-1, :2] = relative_pos
    feat[1:-1, 2] = relative_heading
    feat[1:-1, 3:5] = relative_vel
    # print("MAX: x={:.3f}, y={:.3f}, h={:.3f}, vx={:.3f}, vy={:.3f}".format(
    #     feat[:, 0].max(), feat[:, 1].max(), feat[:, 2].max(), feat[:, 3].max(), feat[:, 4].max()
    # ))
    # print("MIN: x={:.3f}, y={:.3f}, h={:.3f}, vx={:.3f}, vy={:.3f}".format(
    #     feat[:, 0].min(), feat[:, 1].min(), feat[:, 2].min(), feat[:, 3].min(), feat[:, 4].min()
    # ))
    # print("MAX: l={:.3f}, w={:.3f}, h={:.3f}".format(
    #     current_agent_shape[:, 0].max(), current_agent_shape[:, 1].max(), current_agent_shape[:, 2].max()
    # ))
    # print("MIN: l={:.3f}, w={:.3f}, h={:.3f}".format(
    #     current_agent_shape[:, 0].min(), current_agent_shape[:, 1].min(), current_agent_shape[:, 2].min()
    # ))
    data_dict["decoder/input_action_feature_for_trafficgen"] = feat

    data_dict["decoder/target_offset_for_trafficgen"] = np.stack(
        [gt_position_x, gt_position_y, gt_heading, gt_vel_x, gt_vel_y, gt_shape_l, gt_shape_w, gt_shape_h, agent_type],
        axis=1
    ).astype(int)

    data_dict["decoder/input_offset_for_trafficgen"] = np.concatenate(
        [
            np.full((1, 9), -1, dtype=int),
            data_dict["decoder/target_offset_for_trafficgen"]
        ], axis=0
    )

    # Pad one more step for "end action"
    data_dict["decoder/target_offset_for_trafficgen"] = np.concatenate(
        [data_dict["decoder/target_offset_for_trafficgen"],
         np.zeros((1, 9), dtype=int)], axis=0
    )

    return data_dict


NUM_TG_MULTI = 4

TG_SKIP_STEP = 2


def slice_trafficgen_data(tensor, dim):
    # We planned to slice TG data every 1s = 2 steps.
    num_skip = TG_SKIP_STEP
    if dim == 0:
        return tensor[::num_skip]
    elif dim == 1:
        return tensor[:, ::num_skip]
    elif dim == 2:
        return tensor[:, :, ::num_skip]
    else:
        raise ValueError(f"Unknown dimension: {dim}")


def prepare_trafficgen_data_for_scenestreamer(
    data_dict,
    config,
    scenario,
    force_t=True,  # Just disable this function...
    only_lane=False,
    dest_dropout=0.0
):

    data_dict = prepare_destination(data_dict, config, FUTURE_STEPS=30, skip_step=5, dropout=dest_dropout)

    sdc_index = data_dict["decoder/sdc_index"]
    T = data_dict["encoder/agent_feature"].shape[0]

    # start_action_id = config.PREPROCESSING.MAX_MAP_FEATURES
    # start_sequence_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1
    # end_sequence_id = config.PREPROCESSING.MAX_MAP_FEATURES + 2
    # dest_pad_id = config.PREPROCESSING.MAX_MAP_FEATURES + 3

    trafficgen_sequence_sos_id = config.PREPROCESSING.MAX_MAP_FEATURES
    trafficgen_sequence_eos_id = config.PREPROCESSING.MAX_MAP_FEATURES + 1
    trafficgen_sequence_pad_id = config.PREPROCESSING.MAX_MAP_FEATURES + 2
    veh_id = config.PREPROCESSING.MAX_MAP_FEATURES + 3
    ped_id = config.PREPROCESSING.MAX_MAP_FEATURES + 4
    cyc_id = config.PREPROCESSING.MAX_MAP_FEATURES + 5
    trafficgen_agent_sos_id = config.PREPROCESSING.MAX_MAP_FEATURES + 6

    data_dict["decoder/dest_map_index_gt"][data_dict["decoder/dest_map_index_gt"] == -1] = trafficgen_sequence_pad_id

    agent_type = data_dict["decoder/agent_type"]  # in 123 (that's why we using 5 possible actions)
    assert agent_type.max() < 4

    current_agent_shape = data_dict["decoder/current_agent_shape"]

    # Note that here we reuse the "current_agent".
    map_pos = data_dict["encoder/map_position"][..., :2]
    map_heading = data_dict["encoder/map_heading"]

    # Get map feature valid mask
    valid_map_feat = data_dict["encoder/map_valid_mask"]

    tg_action_list = []
    tg_valid_list = []
    tg_feat_list = []
    tg_target_offset_list = []
    tg_pos_list = []
    tg_head_list = []

    # TODO: Hardcoded
    for sparse_t, current_t in enumerate(range(0, T, 5)):

        pos = data_dict["decoder/modeled_agent_position"][sparse_t, ..., :2]
        heading = data_dict["decoder/modeled_agent_heading"][sparse_t]
        valid = data_dict["decoder/input_action_valid_mask"][sparse_t]
        vel = data_dict["decoder/modeled_agent_velocity"][sparse_t]
        dest = data_dict["decoder/dest_map_index"][sparse_t]

        tg_map_id, tg_valid, tg_feat, tg_target_offset, tg_pos, tg_head = prepare_trafficgen_data_for_scenestreamer_a_step(
            pos=pos,
            heading=heading,
            agent_valid_mask=valid,
            vel=vel,
            dest=dest,
            map_pos=map_pos,
            map_heading=map_heading,
            agent_type=agent_type,
            map_valid_mask=valid_map_feat,
            current_agent_shape=current_agent_shape,
            # start_action_id=start_action_id,
            # end_action_id=end_action_id,
            start_sequence_id=trafficgen_sequence_sos_id,
            end_sequence_id=trafficgen_sequence_eos_id,
            dest_pad_id=trafficgen_sequence_pad_id,
            veh_id=veh_id,
            cyc_id=cyc_id,
            ped_id=ped_id,
            start_agent_id=trafficgen_agent_sos_id,
        )

        tg_action_list.append(tg_map_id)
        tg_valid_list.append(tg_valid)
        tg_feat_list.append(tg_feat)
        tg_target_offset_list.append(tg_target_offset)
        tg_pos_list.append(tg_pos)
        tg_head_list.append(tg_head)

    tg_action_list = np.stack(tg_action_list, axis=0)
    tg_valid_list = np.stack(tg_valid_list, axis=0)
    tg_feat_list = np.stack(tg_feat_list, axis=0).astype(np.float32)
    tg_target_offset_list = np.stack(tg_target_offset_list, axis=0)
    tg_pos_list = np.stack(tg_pos_list, axis=0).astype(np.float32)
    tg_head_list = np.stack(tg_head_list, axis=0).astype(np.float32)

    data_dict["decoder/trafficgen_position"] = tg_pos_list
    data_dict["decoder/trafficgen_heading"] = tg_head_list

    data_dict["decoder/input_action_for_trafficgen"] = tg_action_list
    data_dict["decoder/input_action_valid_mask_for_trafficgen"] = tg_valid_list
    assert tg_action_list[tg_valid_list].min() >= 0

    data_dict["decoder/input_action_feature_for_trafficgen"] = tg_feat_list.astype(np.float32)

    data_dict["decoder/target_offset_for_trafficgen"] = (
            tg_target_offset_list * data_dict["decoder/agent_valid_mask"][::5][..., None]
    )
    tg_input_offset_list = np.concatenate([
        np.full((tg_target_offset_list.shape[0], tg_target_offset_list.shape[1], 1), -1, dtype=int),
        tg_target_offset_list], axis=-1
    )
    data_dict["decoder/input_offset_for_trafficgen"] = (
            tg_input_offset_list * data_dict["decoder/agent_valid_mask"][::5][..., None]
    )

    G = tg_action_list.shape[1]
    N = agent_type.shape[0]
    sparse_T = tg_action_list.shape[0]

    # agent_type = agent_type[None].repeat(tg_action_list.shape[0], axis=0)
    new_agent_type = np.full((sparse_T, N, NUM_TG_MULTI), -1)
    tmp_agent_type = np.full((N,), -1)
    tmp_agent_type[agent_type == 1] = veh_id
    tmp_agent_type[agent_type == 2] = ped_id
    tmp_agent_type[agent_type == 3] = cyc_id
    new_agent_type[:, :, 2:] = tmp_agent_type.reshape(1, N, 1)
    # new_new_agent_type = np.full((sparse_T, G), -1)
    # new_new_agent_type[:, 1:-1] = new_agent_type.reshape(sparse_T, -1)
    new_agent_type = np.concatenate([
        np.full((sparse_T, 1), -1),
        new_agent_type.reshape(sparse_T, -1),
        np.full((sparse_T, 1), -1),
    ], axis=1)
    assert new_agent_type.shape == (sparse_T, G)
    data_dict["decoder/agent_type_for_trafficgen"] = new_agent_type.astype(int)

    # data_dict["decoder/current_agent_shape_for_trafficgen"] = np.concatenate([[[0, 0, 0]], current_agent_shape, [[0, 0, 0]]], axis=0).astype(np.float32)
    agent_id = np.concatenate([
        np.full((sparse_T, 1), -1),
        data_dict["encoder/modeled_agent_id"].repeat(NUM_TG_MULTI)[None].repeat(sparse_T, axis=0),
        np.full((sparse_T, 1), -1),
    ], axis=1)
    assert agent_id.shape == (sparse_T, G)
    data_dict["decoder/agent_id_for_trafficgen"] = agent_id

    # Overwrite the original agent type with the new one in SceneStreamer
    data_dict["decoder/agent_type"] = tmp_agent_type

    return data_dict



def prepare_trafficgen_data_for_scenestreamer_a_step(
        *, pos, heading, agent_valid_mask, vel, map_pos, map_heading, map_valid_mask, agent_type, current_agent_shape,
        start_sequence_id, end_sequence_id, dest, dest_pad_id,
        veh_id, cyc_id, ped_id, start_agent_id,
):
    original_pos = pos
    original_heading = heading
    original_shape = current_agent_shape

    N = len(pos)

    from scenestreamer.models.scenestreamer_model import get_num_tg
    G = get_num_tg(N)

    heading_diff = utils.wrap_to_pi(heading[:, None] - map_heading[None])
    valid_heading = np.abs(heading_diff) < np.deg2rad(90)
    valid_map_feat = map_valid_mask & valid_heading

    # if only_lane:
    #     map_feature = data_dict["encoder/map_feature"]
    #     is_lane = map_feature[:, 0, 13] == 1
    #     is_lane = is_lane[None].repeat(N, 0)
    #     valid_map_feat = is_lane & valid_map_feat

    # Find the closest map feature
    dist = np.linalg.norm((pos[:, None] - map_pos[None])[..., :2], axis=-1)
    dist[~valid_map_feat] = np.inf
    closest_map_feat = np.argmin(dist, axis=1)

    # Get the selected map feature
    selected_map_pos = map_pos[closest_map_feat]
    selected_map_heading = map_heading[closest_map_feat]

    # Get relative information
    relative_pos = pos - selected_map_pos
    relative_pos = utils.rotate(x=relative_pos[:, 0], y=relative_pos[:, 1], angle=-selected_map_heading)
    original_relative_pos = relative_pos

    relative_heading = utils.wrap_to_pi(heading - selected_map_heading)
    relative_vel = utils.rotate(x=vel[:, 0], y=vel[:, 1], angle=-selected_map_heading)
    original_relative_heading = relative_heading
    original_relative_vel = relative_vel

    # Get the discretized relative position
    gt_position_x = TrafficGenTokenizerAutoregressive.bucketize(relative_pos[:, 0], "position_x")
    gt_position_y = TrafficGenTokenizerAutoregressive.bucketize(relative_pos[:, 1], "position_y")
    # relative_pos_x = TrafficGenTokenizerAutoregressive.de_bucketize(gt_position_x, "position_x")
    # relative_pos_y = TrafficGenTokenizerAutoregressive.de_bucketize(gt_position_y, "position_y")

    # Reconstruct the position with the bucketized value
    # relative_pos = np.stack([relative_pos_x, relative_pos_y], axis=1)
    # pos = utils.rotate(x=relative_pos_x, y=relative_pos_y, angle=selected_map_heading) + selected_map_pos

    # Reconstruct the heading and velocity
    gt_heading = TrafficGenTokenizerAutoregressive.bucketize(relative_heading, "heading")
    # relative_heading = TrafficGenTokenizerAutoregressive.de_bucketize(gt_heading, "heading")
    # heading = utils.wrap_to_pi(relative_heading + selected_map_heading)

    # Reconstruct the velocity
    gt_vel_x = TrafficGenTokenizerAutoregressive.bucketize(relative_vel[:, 0], "velocity_x")
    gt_vel_y = TrafficGenTokenizerAutoregressive.bucketize(relative_vel[:, 1], "velocity_y")
    # relative_vel_x = TrafficGenTokenizerAutoregressive.de_bucketize(gt_vel_x, "velocity_x")
    # relative_vel_y = TrafficGenTokenizerAutoregressive.de_bucketize(gt_vel_y, "velocity_y")
    # relative_vel = np.stack([relative_vel_x, relative_vel_y], axis=1)
    # vel = utils.rotate(x=relative_vel_x, y=relative_vel_y, angle=selected_map_heading)

    # Reconstruct shape
    gt_shape_l = TrafficGenTokenizerAutoregressive.bucketize(current_agent_shape[:, 0], "length")
    gt_shape_w = TrafficGenTokenizerAutoregressive.bucketize(current_agent_shape[:, 1], "width")
    gt_shape_h = TrafficGenTokenizerAutoregressive.bucketize(current_agent_shape[:, 2], "height")
    # current_agent_shape = np.stack(
    #     [
    #         TrafficGenTokenizerAutoregressive.de_bucketize(gt_shape_l, "length"),
    #         TrafficGenTokenizerAutoregressive.de_bucketize(gt_shape_w, "width"),
    #         TrafficGenTokenizerAutoregressive.de_bucketize(gt_shape_h, "height")
    #     ],
    #     axis=1
    # )

    # ===== Fill in the data for trafficgen =====
    # note: when generating one agent, we have 5 tokens:
    #   agent_start, map_id, agent_state, dest_map_id, agent_end

    # if dest is not None:
    #     dest_pos = map_pos[dest]
    #     dest_heading = map_heading[dest]
    #
    #     # Use agent's current position and heading if destination is not valid
    #     dest_pos[dest == -1] = original_pos[dest == -1]
    #     dest_heading[dest == -1] = original_heading[dest == -1]
    #
    # else:
    #     dest_pos = np.zeros((N, 2))
    #     dest_heading = np.zeros(N)
    #     dest = np.full((N,), -1)
    #
    # # destdist = np.linalg.norm(dest_pos - original_pos, axis=-1)
    # # print("Dist>1 Rate {}, Dist Avg {}".format((destdist > 1).mean(), destdist[destdist > 1].mean() if (destdist > 1).any() else 0))
    #
    # dest[dest==-1] = dest_pad_id
    # dest[~agent_valid_mask] = -1

    new_agent_type_id = np.full((N, ), -1)

    new_agent_type_id[agent_type == constants.object_type_to_int["VEHICLE"]] = veh_id
    new_agent_type_id[agent_type == veh_id] = veh_id
    new_agent_type_id[agent_type == constants.object_type_to_int["PEDESTRIAN"]] = ped_id
    new_agent_type_id[agent_type == ped_id] = ped_id
    new_agent_type_id[agent_type == constants.object_type_to_int["CYCLIST"]] = cyc_id
    new_agent_type_id[agent_type == cyc_id] = cyc_id
    # assert new_agent_type_id.min() != -1

    map_id = np.concatenate([
        np.full((N, 1), start_agent_id),
        new_agent_type_id.reshape(N, 1),
        closest_map_feat[:, None],
        closest_map_feat[:, None],
        # dest[:, None],
    ], axis=1).flatten()
    map_id = np.concatenate([[start_sequence_id], map_id, [end_sequence_id]]).astype(int)

    tg_valid = np.concatenate([[1], agent_valid_mask[:, None].repeat(NUM_TG_MULTI, axis=1).flatten(), [1]], axis=0).astype(bool)

    tg_target_offset = np.stack(
        [
            gt_shape_l,
            gt_shape_w,
            gt_shape_h,
            gt_position_x,
            gt_position_y,
            gt_heading,
            gt_vel_x,
            gt_vel_y
        ],
        axis=1
    ).astype(int)


    # The idea is that all the model's input should use GT data instead of TG's reconstructed data
    # == These are wrong:
    # tg_pos = np.concatenate([
    #     np.zeros((N, 1, 2)),
    #     selected_map_pos[:, None],
    #     pos[:, None],
    #     dest_pos[:, None],
    #     np.zeros((N, 1, 2)),
    # ], axis=1).reshape(-1, 2)
    # tg_head = np.concatenate([
    #     np.zeros((N, 1)),
    #     selected_map_heading[:, None],
    #     heading[:, None],
    #     dest_heading[:, None],
    #     np.zeros((N, 1)),
    # ], axis=1).reshape(-1)
    # tg_feat = np.zeros((len(pos), 5, 8), dtype=np.float32)
    # # import matplotlib.pyplot as plt;plt.scatter(relative_pos[:, 0], relative_pos[:, 1]);plt.show()
    # tg_feat[:, 2, :2] = relative_pos
    # tg_feat[:, 2, 2] = relative_heading
    # tg_feat[:, 2, 3:5] = relative_vel
    # tg_feat[:, 2, 5:] = current_agent_shape
    # tg_feat = tg_feat.reshape(-1, 8)
    # tg_feat = np.concatenate([
    #     np.full((1, 8), 0),
    #     tg_feat,
    #     np.full((1, 8), 0),
    # ], axis=0)
    # assert tg_feat.shape == (G, 8)
    # == These are correct:
    tg_pos = np.concatenate([
        np.zeros((N, 2, 2)),
        selected_map_pos[:, None],
        original_pos[:, None],
        # dest_pos[:, None],
    ], axis=1).reshape(-1, 2)
    tg_head = np.concatenate([
        np.zeros((N, 2)),
        selected_map_heading[:, None],
        original_heading[:, None],
        # dest_heading[:, None],
    ], axis=1).reshape(-1)
    tg_feat = np.zeros((len(pos), NUM_TG_MULTI, 8), dtype=np.float32)
    # import matplotlib.pyplot as plt;plt.scatter(relative_pos[:, 0], relative_pos[:, 1]);plt.show()
    tg_feat[:, 3, :2] = original_relative_pos
    tg_feat[:, 3, 2] = original_relative_heading
    tg_feat[:, 3, 3:5] = original_relative_vel
    tg_feat[:, 3, 5:] = original_shape
    tg_feat = tg_feat.reshape(-1, 8)
    tg_feat = np.concatenate([
        np.full((1, 8), 0),
        tg_feat,
        np.full((1, 8), 0),
    ], axis=0)
    assert tg_feat.shape == (G, 8)

    assert map_id.shape[0] == tg_valid.shape[0] == tg_feat.shape[0] == G
    assert tg_target_offset.shape[0] == N

    tg_pos = np.concatenate([
        np.zeros((1, 2)),
        tg_pos,
        np.zeros((1, 2)),
    ], axis=0)
    assert tg_pos.shape == (G, 2)

    tg_head = np.concatenate([
        np.zeros((1)),
        tg_head,
        np.zeros((1)),
    ], axis=0)
    assert tg_head.shape == (G,)

    return map_id, tg_valid, tg_feat, tg_target_offset, tg_pos, tg_head


def translate_abs_info_to_ego_centric(data_dict, current_t, retain_raw=False):

    if retain_raw:
        data_dict["vis/map_feature"] = data_dict["encoder/map_feature"].copy()
    data_dict["raw/map_feature"] = data_dict["encoder/map_feature"].copy()

    def _get_last_pos(pos, head, valid_mask):
        T, N = valid_mask.shape
        ind = np.arange(T).reshape(-1, 1).repeat(N, axis=1)  # T, N
        ind[~valid_mask] = 0
        ind = ind.max(axis=0)
        out = np.take_along_axis(pos, indices=ind.reshape(1, N, 1), axis=0)
        outh = np.take_along_axis(head, indices=ind.reshape(1, N), axis=0)
        out = np.squeeze(out, axis=0)
        outh = np.squeeze(outh, axis=0)
        return out, outh

    # === Agent features ===
    agent_p, agent_h = _get_last_pos(
        data_dict["encoder/agent_position"][:current_t + 1], data_dict["encoder/agent_heading"][:current_t + 1],
        data_dict["encoder/agent_valid_mask"][:current_t + 1]
    )

    pos = data_dict["encoder/agent_feature"][..., :3]
    pos = pos - agent_p[None]
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-agent_h.reshape(1, -1).repeat(pos.shape[0], axis=0), z=pos[..., 2]
    )
    data_dict["encoder/agent_feature"][..., :3] = pos

    head = data_dict["encoder/agent_feature"][..., 3]
    head = utils.wrap_to_pi(head - agent_h[None])
    data_dict["encoder/agent_feature"][..., 3] = head
    data_dict["encoder/agent_feature"][..., 4] = np.sin(head)
    data_dict["encoder/agent_feature"][..., 5] = np.cos(head)

    vel = data_dict["encoder/agent_feature"][..., 6:8]
    vel = utils.rotate(
        x=vel[..., 0],
        y=vel[..., 1],
        angle=-agent_h.reshape(1, -1).repeat(vel.shape[0], axis=0),
    )
    data_dict["encoder/agent_feature"][..., 6:8] = vel

    data_dict["encoder/agent_feature"][~data_dict["encoder/agent_valid_mask"]] = 0

    # === Map features ===
    map_pos = data_dict["encoder/map_position"][:, None]
    map_h = data_dict["encoder/map_heading"][:, None]

    pos = data_dict["encoder/map_feature"][..., :3] - map_pos
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., :3] = pos

    pos = data_dict["encoder/map_feature"][..., 3:6] - map_pos
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., 3:6] = pos

    pos = data_dict["encoder/map_feature"][..., 6:9]  # direction, no need to translate
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., 6:9] = pos

    head = data_dict["encoder/map_feature"][..., 9]
    head = utils.wrap_to_pi(head - map_h)
    data_dict["encoder/map_feature"][..., 9] = head
    data_dict["encoder/map_feature"][..., 10] = np.sin(head)
    data_dict["encoder/map_feature"][..., 11] = np.cos(head)

    # === Traffic light features ===
    # Note: We want to remove all absolute information so just remove traffic light position!
    if "encoder/traffic_light_feature" in data_dict:
        data_dict["encoder/traffic_light_feature"][..., :3] = 0

    return data_dict


def limit_map_range(data_dict, limit_range=50):
    sdc_index = data_dict["decoder/sdc_index"]
    current_t = data_dict["metadata/current_time_index"]
    sdc_center = data_dict["decoder/agent_position"][current_t, sdc_index]  # (3,)

    # Limit the map range
    margin = 0
    valid_map_feat = (
        (abs(data_dict["encoder/map_position"][..., 0] - sdc_center[0]) < limit_range + margin) &
        (abs(data_dict["encoder/map_position"][..., 1] - sdc_center[1]) < limit_range + margin)
    )
    valid_map_feat = valid_map_feat & data_dict["encoder/map_valid_mask"]
    data_dict["encoder/map_feature_valid_mask"][~valid_map_feat] = False
    data_dict["encoder/map_valid_mask"][~valid_map_feat] = False

    # Delete agents that are out of the map range
    agent_pos = data_dict["encoder/agent_position"][current_t]
    distance_mask = (
        (abs(agent_pos[..., 0] - sdc_center[0]) < limit_range) & (abs(agent_pos[..., 1] - sdc_center[1]) < limit_range)
    )
    data_dict["encoder/agent_valid_mask"][current_t] = (
        data_dict["encoder/agent_valid_mask"][current_t] & distance_mask
    )
    data_dict["encoder/current_agent_valid_mask"] = data_dict["encoder/agent_valid_mask"][current_t].copy()
    agent_pos = data_dict["decoder/agent_position"][current_t]
    distance_mask = (
        (abs(agent_pos[..., 0] - sdc_center[0]) < limit_range) & (abs(agent_pos[..., 1] - sdc_center[1]) < limit_range)
    )
    data_dict["decoder/agent_valid_mask"][current_t] = (
        data_dict["decoder/agent_valid_mask"][current_t] & distance_mask
    )
    data_dict["decoder/current_agent_valid_mask"] = data_dict["decoder/agent_valid_mask"][current_t].copy()

    # TODO: eval/agent_valid_mask is not touched yet. But it's fine now...
    return data_dict


def preprocess_scenario_description_for_motionlm(
    scenario, config, in_evaluation, keep_all_data=False, backward_prediction=None, tokenizer=None
):
    metadata = scenario[SD.METADATA]

    if in_evaluation:
        max_agents = 128  # TODO: hardcoded
    else:
        max_agents = config.PREPROCESSING.MAX_AGENTS

    tracks_to_predict_dict = metadata.get('tracks_to_predict', {})
    track_index_to_predict = np.array([int(v['track_index']) for v in tracks_to_predict_dict.values()])
    track_name_to_predict = [int(k) for k in tracks_to_predict_dict.keys()]

    # Put SDC name to the first place.
    sdc_name = metadata["sdc_id"]
    try:
        sdc_name = int(sdc_name)
    except:
        pass
    if sdc_name in track_name_to_predict:
        track_name_to_predict.remove(sdc_name)
        track_name_to_predict.insert(0, sdc_name)
    track_name_to_predict = np.array(track_name_to_predict)

    data_dict = {
        "in_evaluation": in_evaluation,
        "metadata/sdc_name": sdc_name,
        "encoder/object_of_interest_name": track_name_to_predict,
        "encoder/object_of_interest_id": track_index_to_predict,
        "scenario_id": scenario[SD.ID],
    }
    if "current_time_index" in metadata:
        data_dict["metadata/current_time_index"] = metadata['current_time_index']
    else:
        # TODO: Not sure in nuscenes if there is no current_time_index. Might need to check.
        data_dict["metadata/current_time_index"] = 0
        metadata['current_time_index'] = 0

    # ===== Extract map and traffic light features =====
    data_dict = process_map_and_traffic_light(
        data_dict=data_dict,
        scenario=scenario,
        map_feature=scenario[SD.MAP_FEATURES],
        dynamic_map_states=scenario[SD.DYNAMIC_MAP_STATES],
        track_length=scenario[SD.LENGTH],
        max_vectors=config.PREPROCESSING.MAX_VECTORS,
        max_map_features=config.PREPROCESSING.MAX_MAP_FEATURES,
        limit_map_range=config.LIMIT_MAP_RANGE,
        max_length_per_map_feature=config.PREPROCESSING.MAX_LENGTH_PER_MAP_FEATURE,
        max_traffic_lights=config.PREPROCESSING.MAX_TRAFFIC_LIGHTS,
        remove_traffic_light_state=config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE,
        is_scenestreamer=config.MODEL.NAME == "scenestreamer",
    )

    # ===== Extract agent features =====
    data_dict = process_track(
        data_dict=data_dict,
        tracks=scenario[SD.TRACKS],
        track_length=scenario[SD.LENGTH],
        sdc_name=metadata["sdc_id"],
        max_agents=max_agents,
        exempt_max_agent_filtering=in_evaluation,
        is_scenestreamer=config.MODEL.NAME == "scenestreamer",
    )
    data_dict = prepare_modeled_agent_and_eval_data(
        data_dict=data_dict,
        predict_all_agents=config.TRAINING.PREDICT_ALL_AGENTS,
        eval_all_agents=config.EVALUATION.PREDICT_ALL_AGENTS,
        current_t=metadata['current_time_index'],
        add_sdc_to_object_of_interest=config.PREPROCESSING.ADD_SDC_TO_OBJECT_OF_INTEREST
    )

    if config.LIMIT_MAP_RANGE:
        data_dict = limit_map_range(data_dict)

    if config.MODEL.RELATIVE_PE:
        data_dict = translate_abs_info_to_ego_centric(
            data_dict, current_t=data_dict["metadata/current_time_index"], retain_raw=keep_all_data
        )

    # if use_action_label:
    sdc_ind = data_dict["decoder/sdc_index"]
    object_of_interest = data_dict["decoder/object_of_interest_id"]
    if sdc_ind not in object_of_interest:
        object_of_interest = np.concatenate([[sdc_ind], object_of_interest])
    data_dict["decoder/labeled_agent_id"] = np.asarray(object_of_interest).astype(int)

    # ===== Call the tokenizer and generate target discretized actions =====
    # Error stats is removed from here. It's used in independent test script.
    use_backward_prediction = config.BACKWARD_PREDICTION
    if in_evaluation:
        use_backward_prediction = False
    if use_backward_prediction:
        # Use 50% probability to set backward_prediction to True
        use_backward_prediction = np.random.rand() < 0.5
    if backward_prediction is not None:  # Overwrite the value
        use_backward_prediction = backward_prediction
    if config.USE_DIFFUSION:
        detok, error_stat = tokenizer.tokenize_numpy_array(data_dict, backward_prediction=use_backward_prediction)
        for k in ["decoder/target_agent_motion", "decoder/input_agent_motion", "decoder/target_action_valid_mask",
                  "decoder/input_action", "decoder/input_action_valid_mask", "decoder/modeled_agent_position",
                  "decoder/modeled_agent_heading", "decoder/modeled_agent_velocity", "decoder/modeled_agent_delta",
                  "in_backward_prediction"]:
            if k in detok:
                data_dict[k] = detok[k]

    else:
        if config.TOKENIZATION.TOKENIZATION_METHOD is not None:
            detok, error_stat = tokenizer.tokenize_numpy_array(data_dict, backward_prediction=use_backward_prediction)
            for k in ["decoder/target_action", "decoder/target_action_valid_mask", "decoder/input_action",
                      "decoder/input_action_valid_mask", "decoder/modeled_agent_position",
                      "decoder/modeled_agent_heading", "decoder/modeled_agent_velocity", "decoder/modeled_agent_delta",
                      "in_backward_prediction"]:
                if k in detok:
                    data_dict[k] = detok[k]

    if config.ACTION_LABEL.USE_ACTION_LABEL:
        data_dict = prepare_action_label(
            data_dict=data_dict,
            dt=0.1,  # TODO(PZH): Hardcoded here.
            config=config,
            mask_probability=config.ACTION_LABEL.MASK_PROBABILITY_ACTION_LABEL if in_evaluation else 0.0
        )

    if config.get("ACTION_LABEL") and config.ACTION_LABEL.USE_SAFETY_LABEL:
        data_dict = prepare_safety_label(
            data_dict=data_dict,
            dt=0.1,  # TODO(PZH): Hardcoded here.
            config=config,
            mask_probability=config.ACTION_LABEL.MASK_PROBABILITY_SAFETY_LABEL if in_evaluation else 0.0
        )

    if config["USE_TRAFFICGEN"]:
        is_scenestreamer = config.MODEL.NAME == "scenestreamer"

        if is_scenestreamer:
            assert not config.USE_DESTINATION
            data_dict = prepare_trafficgen_data_for_scenestreamer(
                data_dict=data_dict, config=config, scenario=scenario, force_t=True, dest_dropout=config.PREPROCESSING.DEST_DROPOUT,
            )

        else:
            data_dict = prepare_trafficgen_data(
                data_dict=data_dict, config=config, scenario=scenario, only_lane=config.ONLY_LANE_FOR_TRAFFICGEN,
            )

    if config.PREPROCESSING.TRUNCATE_TIME >= 0:
        for k in [
            "encoder/traffic_light_state",
            "encoder/traffic_light_valid_mask",
            "decoder/target_action",
            "decoder/input_action",
            "decoder/target_action_valid_mask",
            "decoder/input_action_valid_mask",
            "decoder/modeled_agent_position",
            "decoder/modeled_agent_heading",
            "decoder/modeled_agent_velocity",
            "decoder/modeled_agent_delta",
            "decoder/trafficgen_position",
            "decoder/trafficgen_heading",
            "decoder/input_action_for_trafficgen",
            "decoder/input_action_valid_mask_for_trafficgen",
            "decoder/input_action_feature_for_trafficgen",
            "decoder/target_offset_for_trafficgen",
            "decoder/input_offset_for_trafficgen",
            "decoder/agent_type_for_trafficgen",
            "decoder/agent_id_for_trafficgen",
        ]:
            data_dict[k] = data_dict[k][:config.PREPROCESSING.TRUNCATE_TIME]

    if config.USE_DESTINATION:
        data_dict = prepare_destination(data_dict, config)

    # TODO: A little hack here...
    if config.EVALUATION.NAME == "lmdb":
        keep_all_data = False
        in_evaluation = False

    if not keep_all_data:
        if in_evaluation:
            pass
            # data_dict = {k: v for k, v in data_dict.items() if not k.startswith("decoder/")}  # Remove decoder/ data
        else:

            # Discard these data
            for pattern in [
                    "eval/",
                    "encoder/current_",
                    "encoder/future_",
            ]:
                data_dict = {k: v for k, v in data_dict.items() if not k.startswith(pattern)}
            if config.GPT_STYLE and config.REMOVE_AGENT_FROM_SCENE_ENCODER:
                data_dict = {k: v for k, v in data_dict.items() if not k.startswith("encoder/agent_")}

            # Keep these data
            new_data_dict = {}
            for pattern in ["scenario_id", "decoder/label_", "decoder/agent_id", "decoder/agent_type",
                            "decoder/current_", "decoder/modeled_agent_", "decoder/input_", "decoder/target_",
                            "encoder/", "in_evaluation", "in_backward_prediction", "decoder/dest_map_index",
                            "decoder/trafficgen_", "decoder/sdc_", "metadata/", ]:
                new_data_dict.update({k: v for k, v in data_dict.items() if k.startswith(pattern)})
            data_dict = new_data_dict

    sorted_keys = sorted(data_dict.keys())
    data_dict = {k: data_dict[k] for k in sorted_keys}
    return data_dict


def _get_trafficgen_data(raw_scenario_description, data_dict, current_t):
    """
    PZH:
    I don't want to waste time to read through the LCTGen code,
    which essentially is from the TrafficGen code base.
    I've read the TrafficGen code base and I really really don't want
    to look into it for the second time.
    Just copy the code here and modify it to fit the current code base.
    """
    def rotate(x, y, angle):

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

    def normalize_angle(angle):
        """
        From: https://github.com/metadriverse/trafficgen/blob/28b109e8e640d820192d5485bf9a28128b38ca21/trafficgen/utils/utils.py#L20
        """
        # if isinstance(angle, torch.Tensor):
        #     while not torch.all(angle >= 0):
        #         angle[angle < 0] += np.pi * 2
        #     while not torch.all(angle < np.pi * 2):
        #         angle[angle >= np.pi * 2] -= np.pi * 2
        #     return angle
        #
        # else:
        while not np.all(angle >= 0):
            angle[angle < 0] += np.pi * 2
        while not np.all(angle < np.pi * 2):
            angle[angle >= np.pi * 2] -= np.pi * 2

        return angle

    from scenestreamer.eval.scenarionet_to_trafficgen import metadrive_scenario_to_init_data

    data = metadrive_scenario_to_init_data(raw_scenario_description)
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
        # index = min(index, len(data['all_agent']) - 1)
        # data['all_agent'] = data['all_agent'][index:index + self.data_cfg.MAX_TIME_STEP:gap]
        # data['traffic_light'] = data['traffic_light'][index:index + self.data_cfg.MAX_TIME_STEP:gap]

    def _transform_coordinate_map(data):
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

    data['lane'], other['unsampled_lane'] = _transform_coordinate_map(data)
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

    trafficgen_select_track_names = ret["PZH_TRACK_NAMES"][ret['agent_mask']]
    decoder_track_name = list(data_dict["decoder/track_name"])
    trafficgen_select_index = []
    for name in trafficgen_select_track_names:
        if name in decoder_track_name:
            trafficgen_select_index.append(decoder_track_name.index(name))
        else:
            # print(11)
            pass

    return trafficgen_select_index
