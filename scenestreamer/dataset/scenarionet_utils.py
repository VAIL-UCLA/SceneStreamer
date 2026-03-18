import numpy as np
import torch
from scenestreamer.utils import wrap_to_pi, rotate


def overwrite_gt_to_pred_field(data_dict):
    import copy
    new_data_dict = copy.deepcopy(data_dict)
    T, N, _ = data_dict["decoder/agent_position"].shape

    new_data_dict["decoder/reconstructed_position"] = np.zeros((96, N, 2)).astype(np.float32)
    new_data_dict["decoder/reconstructed_valid_mask"] = np.zeros((
        96,
        N,
    )).astype(bool)
    new_data_dict["decoder/reconstructed_heading"] = np.zeros((
        96,
        N,
    )).astype(np.float32)
    new_data_dict["decoder/reconstructed_velocity"] = np.zeros((96, N, 2)).astype(np.float32)

    for id in range(N):  # overwrite all agents
        traj = new_data_dict["decoder/agent_position"][:91, id, :2].astype(np.float32)
        traj_mask = new_data_dict["decoder/agent_valid_mask"][:91, id].astype(bool)
        theta = new_data_dict['decoder/agent_heading'][:91, id].astype(np.float32)
        vel = new_data_dict['decoder/agent_velocity'][:91, id].astype(np.float32)

        new_data_dict["decoder/reconstructed_position"][:91, id, :2] = traj
        # new_data_dict["decoder/reconstructed_position"][:91, id, 2] = 0.0
        new_data_dict["decoder/reconstructed_valid_mask"][:91, id] = traj_mask
        # print(traj_mask)
        new_data_dict["decoder/reconstructed_heading"][:91, id] = theta
        new_data_dict["decoder/reconstructed_velocity"][:91, id] = vel

    return new_data_dict


def create_new_adv(data_dict):
    ego_id = data_dict["decoder/sdc_index"]

    ego_traj = data_dict["decoder/agent_position"][:, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:, ego_id]

    last_valid_step = np.where(ego_mask)[0][-1]

    # Create a new ADV at the final step.

    adv_mask = np.zeros_like(ego_mask)
    adv_mask[:last_valid_step + 1] = True

    adv_traj = np.zeros_like(ego_traj)
    adv_heading = np.zeros_like(ego_heading)
    adv_velocity = np.zeros_like(ego_velocity)
    adv_shape = np.zeros_like(ego_shape)

    # Copy the final pos/head/vel/shape of ego
    adv_traj[last_valid_step] = ego_traj[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=3)
    adv_heading[last_valid_step] = ego_heading[last_valid_step] + np.random.normal(loc=0.0, scale=0.1, size=1)
    adv_velocity[last_valid_step] = ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=2)

    for i in range(data_dict["decoder/agent_shape"].shape[0]):
        adv_shape[i] = ego_shape[last_valid_step]

    # Insert data back:
    data_dict["decoder/agent_position"] = np.concatenate(
        [data_dict["decoder/agent_position"], adv_traj[:, None]], axis=1
    )
    data_dict["decoder/agent_heading"] = np.concatenate(
        [data_dict["decoder/agent_heading"], adv_heading[:, None]], axis=1
    )
    data_dict["decoder/agent_velocity"] = np.concatenate(
        [data_dict["decoder/agent_velocity"], adv_velocity[:, None]], axis=1
    )
    # data_dict["decoder/agent_shape"] = np.concatenate([data_dict["decoder/agent_shape"], adv_shape[:, None]], axis=1)

    data_dict["decoder/agent_shape"] = np.concatenate([data_dict["decoder/agent_shape"], adv_shape[:, None]], axis=1)

    data_dict["decoder/agent_valid_mask"] = np.concatenate(
        [data_dict["decoder/agent_valid_mask"], adv_mask[:, None]], axis=1
    )

    data_dict["decoder/current_agent_shape"] = np.concatenate(
        [data_dict["decoder/current_agent_shape"], data_dict["decoder/current_agent_shape"][ego_id:ego_id + 1]], axis=0
    )
    data_dict["decoder/agent_type"] = np.concatenate(
        [data_dict["decoder/agent_type"], data_dict["decoder/agent_type"][ego_id:ego_id + 1]], axis=0
    )
    data_dict["decoder/agent_id"] = np.concatenate(
        [data_dict["decoder/agent_id"], [len(data_dict["decoder/agent_id"])]], axis=0
    )

    # Add ADV into OOI:
    data_dict["decoder/object_of_interest_id"] = np.concatenate(
        [data_dict["decoder/object_of_interest_id"], [len(data_dict["decoder/agent_id"]) - 1]], axis=0
    )

    # Deal with some thing for forward prediction:
    data_dict["decoder/current_agent_valid_mask"] = np.concatenate(
        [data_dict["decoder/current_agent_valid_mask"], [1]], axis=0
    )

    print("====================================")
    print(
        "The new ADV is created at the final step {}, it's ID is: {}".format(
            last_valid_step,
            len(data_dict["decoder/agent_id"]) - 1
        )
    )
    print("====================================")

    return data_dict


def overwrite_to_scenario_description(output_dict_mode, original_SD, ooi=None, adv_id=None):
    # overwrite original SD with all predicted ooi trajectories included
    # import pdb; pdb.set_trace()
    if not ooi:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents
    sdc_track_name = original_SD['metadata']['sdc_id']
    adv_track_name = str(output_dict_mode['decoder/track_name'][int(adv_id)].item())

    for id in ooi:
        agent_track_name = str(output_dict_mode['decoder/track_name'][id].item())

        # begin to overwrite original scenario_data
        agent_traj = output_dict_mode["decoder/agent_position"][:91, id, ]
        agent_heading = output_dict_mode["decoder/agent_heading"][:91, id]
        agent_vel = output_dict_mode["decoder/agent_velocity"][:91, id]
        agent_traj_mask = output_dict_mode["decoder/agent_valid_mask"][:91, id]

        # modify adv info
        # agent_z = original_SD['tracks'][agent_track_name]['state']['position'][10, 2]  # fill the z-axis
        # agent_traj_z = np.full((91, 1), agent_z)
        # agent_new_traj = np.concatenate([agent_traj, agent_traj_z], axis=1)
        # print("new_traj:", agent_new_traj.shape)
        original_SD['tracks'][agent_track_name]['state']['position'] = agent_traj
        original_SD['tracks'][agent_track_name]['state']['velocity'] = agent_vel
        original_SD['tracks'][agent_track_name]['state']['heading'] = agent_heading
        original_SD['tracks'][agent_track_name]['state']['valid'] = agent_traj_mask

        length = original_SD['tracks'][agent_track_name]['state']['length'][10]
        width = original_SD['tracks'][agent_track_name]['state']['width'][10]
        height = original_SD['tracks'][agent_track_name]['state']['height'][10]
        original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
        original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
        original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)

    original_SD['metadata']['selected_adv_id'] = adv_track_name

    return original_SD


def overwrite_to_scenario_description_new_agent(output_dict_mode, original_SD, ooi=None):
    # overwrite original SD with all predicted ooi trajectories included
    ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    adv_track_name = 'new_adv_agent'
    original_SD['tracks'][adv_track_name] = {'state': {}, 'type': 'VEHICLE', 'metadata': {}}
    sdc_track_name = original_SD['metadata']['sdc_id']

    for id in ooi:
        if id == ooi[-1]:
            agent_track_name = 'new_adv_agent'
        else:
            agent_track_name = str(output_dict_mode['decoder/track_name'][id].item())

        # begin to overwrite original scenario_data
        agent_traj = output_dict_mode["decoder/agent_position"][:, id, ]
        agent_heading = output_dict_mode["decoder/agent_heading"][:, id]
        agent_vel = output_dict_mode["decoder/agent_velocity"][:, id]
        agent_traj_mask = output_dict_mode["decoder/agent_valid_mask"][:, id]

        # modify adv info
        # agent_z = original_SD['tracks'][agent_track_name]['state']['position'][10, 2]  # fill the z-axis
        # agent_traj_z = np.full((91, 1), agent_z)
        # agent_new_traj = np.concatenate([agent_traj, agent_traj_z], axis=1)
        # print("new_traj:", agent_new_traj.shape)
        original_SD['tracks'][agent_track_name]['state']['position'] = agent_traj

        original_SD['tracks'][agent_track_name]['state']['velocity'] = agent_vel
        original_SD['tracks'][agent_track_name]['state']['heading'] = agent_heading
        original_SD['tracks'][agent_track_name]['state']['valid'] = agent_traj_mask

        length = original_SD['tracks'][sdc_track_name]['state']['length'][10]
        width = original_SD['tracks'][sdc_track_name]['state']['width'][10]
        height = original_SD['tracks'][sdc_track_name]['state']['height'][10]
        original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
        original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
        original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)

    original_SD['tracks'][adv_track_name]['metadata']['dataset'] = 'waymo'
    original_SD['tracks'][adv_track_name]['metadata']['object_id'] = 'new_adv_agent'
    original_SD['tracks'][adv_track_name]['metadata']['track_length'] = 91
    original_SD['tracks'][adv_track_name]['metadata']['type'] = 'VEHICLE'
    original_SD['metadata']['new_adv_id'] = 'new_adv_agent'
    original_SD['metadata']['objects_of_interest'].append('new_adv_agent')
    tracks_length = len(list(original_SD['tracks'].keys()))
    original_SD['metadata']['tracks_to_predict']['new_adv_agent'] = {
        'difficulty': 0,
        'object_type': 'VEHICLE',
        'track_id': 'new_adv_agent',
        'track_index': tracks_length - 1
    }

    return original_SD


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    assert "decoder/agent_position" in data_dict, "Have you set EVALUATION.PREDICT_ALL_AGENTS to False?"
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert data_dict["decoder/agent_position"].ndim == 3
    data_dict["decoder/agent_position"] += map_center

    return data_dict


def _overwrite_datadict_all_agents(data_dict):
    import copy
    new_data_dict = copy.deepcopy(data_dict)

    T, N, _ = data_dict["decoder/reconstructed_position"].shape

    for id in range(N):  # overwrite all agents
        traj = data_dict["decoder/reconstructed_position"][:91, id, ]
        traj_mask = data_dict["decoder/reconstructed_valid_mask"][:91, id]
        theta = data_dict['decoder/reconstructed_heading'][:91, id]
        vel = data_dict['decoder/reconstructed_velocity'][:91, id]

        new_data_dict["decoder/agent_position"][:, id, :2] = traj
        new_data_dict["decoder/agent_position"][:, id, 2] = 0.0
        new_data_dict["decoder/agent_valid_mask"][:, id] = traj_mask
        new_data_dict["decoder/agent_heading"][:, id] = theta
        new_data_dict["decoder/agent_velocity"][:, id] = vel

    return new_data_dict
