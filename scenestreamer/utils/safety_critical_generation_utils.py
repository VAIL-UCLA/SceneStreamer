import PIL
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle, Rectangle

from scenestreamer.dataset.dataset import SceneStreamerDataset
from scenestreamer.utils import REPO_ROOT
import torch
import copy
import pdb
import pathlib


def _overwrite_data_given_agents_not_ooi(original_data_dict, data_dict, ooi):
    new_data_dict = copy.deepcopy(original_data_dict)

    B, T, N, _ = data_dict["decoder/reconstructed_position"].shape

    assert B == 1
    for b in range(B):
        for aid in range(N):
            if aid in ooi:
                continue
            traj = data_dict["decoder/reconstructed_position"][b, :91, aid, ]
            traj_mask = data_dict["decoder/reconstructed_valid_mask"][b, :91, aid]
            vel = data_dict['decoder/reconstructed_velocity'][b, :91, aid]
            theta = data_dict['decoder/reconstructed_heading'][b, :91, aid]

            new_data_dict["decoder/agent_position"][b, :, aid, :2] = traj
            new_data_dict["decoder/agent_position"][b, :, aid, 2] = 0.0
            new_data_dict["decoder/agent_valid_mask"][b, :, aid] = traj_mask
            new_data_dict["decoder/agent_heading"][b, :, aid] = theta
            new_data_dict["decoder/agent_velocity"][b, :, aid] = vel
            new_data_dict["decoder/agent_shape"][b, :, aid] = new_data_dict["decoder/current_agent_shape"][b, aid]

    return new_data_dict


def get_ego_edge_points(x, y, theta, width, length):
    # Calculate each corner of the rectangle
    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.array([left_front_x, left_front_y])

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.array([right_front_x, right_front_y])

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.array([right_back_x, right_back_y])

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.array([left_back_x, left_back_y])

    # Function to calculate intermediate points on an edge
    def sample_edge_points(start, end, num_points=2):
        return [start + (end - start) * (i / (num_points + 1)) for i in range(1, num_points + 1)]

    # Sample points on each edge
    front_edge_points = sample_edge_points(left_front, right_front)
    right_edge_points = sample_edge_points(right_front, right_back)
    back_edge_points = sample_edge_points(right_back, left_back)
    left_edge_points = sample_edge_points(left_back, left_front)

    # Combine all points: corners and sampled edge points
    polygon_contour = np.array(
        [
            left_front, *front_edge_points, right_front, *right_edge_points, right_back, *back_edge_points, left_back,
            *left_edge_points
        ]
    )

    return polygon_contour


def get_ego_edge_points_old(agent_position, length, width):
    """
    Returns 8 evenly spaced points on the edge of an agent's rectangular contour.
    """
    # Calculate half dimensions
    half_length = length / 2
    half_width = width / 2

    # Define the contour as a rectangle centered at the agent's position
    x, y = agent_position
    contour_points = [
        (x - half_length, y - half_width),  # Bottom-left
        (x + half_length, y - half_width),  # Bottom-right
        (x + half_length, y + half_width),  # Top-right
        (x - half_length, y + half_width)  # Top-left
    ]
    contour = Polygon(contour_points)

    # Calculate 2 points per side (8 points total on the edge)
    edge_points = []
    for i in range(len(contour_points)):
        start = contour_points[i]
        end = contour_points[(i + 1) % len(contour_points)]

        points = [(start[0] + (end[0] - start[0]) * t / 3, start[1] + (end[1] - start[1]) * t / 3) for t in range(1, 3)]
        edge_points.extend(points)

    return edge_points


def post_process_adv_traj(data_dict, adv_id, sdc_id=0):
    """

    """
    from scenestreamer.dataset.preprocess_action_label import get_safety_action_from_sdc_adv, cal_polygon_contour
    # import pdb; pdb.set_trace()
    collision_label = np.array(get_safety_action_from_sdc_adv(data_dict, adv_id, sdc_id))

    if not np.any(collision_label):
        return data_dict, False
    else:
        first_collision_step = np.argmax(collision_label)
        if first_collision_step < 5:
            return data_dict, False

        print("first_collision_step", first_collision_step)

        adv_mask = data_dict["decoder/agent_valid_mask"][:, adv_id]
        adv_mask[first_collision_step:] = False
        data_dict["decoder/agent_valid_mask"][:, adv_id] = adv_mask

    return data_dict, True


def _overwrite_data_given_agents_ooi(original_data_dict, data_dict, ooi):
    new_data_dict = copy.deepcopy(original_data_dict)

    B, T, N, _ = data_dict["decoder/reconstructed_position"].shape

    assert B == 1
    for b in range(B):
        for aid in ooi:
            traj = data_dict["decoder/reconstructed_position"][b, :91, aid, ]
            traj_mask = data_dict["decoder/reconstructed_valid_mask"][b, :91, aid]
            vel = data_dict['decoder/reconstructed_velocity'][b, :91, aid]
            theta = data_dict['decoder/reconstructed_heading'][b, :91, aid]

            new_data_dict["decoder/agent_position"][b, :, aid, :2] = traj
            new_data_dict["decoder/agent_position"][b, :, aid, 2] = 0.0
            new_data_dict["decoder/agent_valid_mask"][b, :, aid] = traj_mask
            new_data_dict["decoder/agent_heading"][b, :, aid] = theta
            new_data_dict["decoder/agent_velocity"][b, :, aid] = vel
            new_data_dict["decoder/agent_shape"][b, :, aid] = new_data_dict["decoder/current_agent_shape"][b, aid]

    return new_data_dict


def _overwrite_data_given_agents(original_data_dict, data_dict, sdc_id, adv_id):
    new_data_dict = copy.deepcopy(original_data_dict)

    T, N, _ = data_dict["decoder/reconstructed_position"].shape

    # for id in ooi_arr:  # overwrite all agents
    traj = data_dict["decoder/reconstructed_position"][:91, sdc_id, ]
    traj_mask = data_dict["decoder/reconstructed_valid_mask"][:91, sdc_id]
    theta = data_dict['decoder/reconstructed_heading'][:91, sdc_id]

    new_data_dict["decoder/agent_position"][:, sdc_id, :2] = traj
    new_data_dict["decoder/agent_position"][:, sdc_id, 2] = 0.0
    new_data_dict["decoder/agent_valid_mask"][:, sdc_id] = traj_mask
    new_data_dict["decoder/agent_heading"][:, sdc_id] = theta

    adv_traj = data_dict["decoder/reconstructed_position"][:91, adv_id][:, None]
    new_dim = np.zeros((adv_traj.shape[0], adv_traj.shape[1], 1))
    adv_traj = np.concatenate([adv_traj, new_dim], axis=-1)
    adv_traj_mask = data_dict["decoder/reconstructed_valid_mask"][:91, adv_id][:, None]
    adv_theta = data_dict['decoder/reconstructed_heading'][:91, adv_id][:, None]

    new_data_dict["decoder/agent_position"] = np.concatenate(
        [new_data_dict["decoder/agent_position"], adv_traj], axis=1
    )
    new_data_dict["decoder/agent_valid_mask"] = np.concatenate(
        [new_data_dict["decoder/agent_valid_mask"], adv_traj_mask], axis=1
    )
    new_data_dict["decoder/agent_heading"] = np.concatenate([new_data_dict["decoder/agent_heading"], adv_theta], axis=1)

    return new_data_dict


def check_last_step_sdc_adv_collision(data_dict, sdc_id, sdc_pos, sdc_heading, adv_id, adv_pos, adv_heading):

    from scenestreamer.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
    contours = []

    sdc_contour = cal_polygon_contour(
        sdc_pos[0], sdc_pos[1], sdc_heading, data_dict["decoder/agent_shape"][10, sdc_id, 1],
        data_dict["decoder/agent_shape"][10, sdc_id, 0]
    )
    adv_contour = cal_polygon_contour(
        adv_pos[0], adv_pos[1], adv_heading, data_dict["decoder/agent_shape"][10, adv_id, 1],
        data_dict["decoder/agent_shape"][10, adv_id, 0]
    )

    collision_tags = detect_collision(adv_contour, [True], sdc_contour, [True])
    collision_detected = np.array(collision_tags)

    if np.any(collision_detected):
        print("collision")
        return True

    return False


def set_adv(data_dict):
    """
    here is the current design: from existing agents, choose the one with its lastest step having nearest distance among all
    """
    ego_id = data_dict["decoder/sdc_index"]
    ego_traj = data_dict["decoder/agent_position"][:, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:, ego_id]

    adv_id, adv_pos, adv_heading, adv_vel, last_valid_step = choose_nearest_adv(data_dict)
    last_valid_step = np.where(ego_mask)[0][-1]  # force setting the last valid step

    ego_last_pos = data_dict["decoder/agent_position"][last_valid_step, ego_id, :2]
    ego_last_heading = data_dict["decoder/agent_heading"][last_valid_step, ego_id]

    # begin to search
    alphas = np.arange(0, 1.02, 0.05)
    collision_point = ego_last_pos  #- np.random.normal(loc=0.0, scale=1, size=ego_last_pos.shape[0])

    for alpha in alphas:
        cand_adv_pos = (1 - alpha) * adv_pos + alpha * ego_last_pos

        if check_last_step_sdc_adv_collision(data_dict, ego_id, ego_last_pos, ego_last_heading, adv_id, cand_adv_pos,
                                             adv_heading):
            collision_point = cand_adv_pos
            break

    # collision_points = np.array(get_ego_edge_points(ego_last_pos[0], ego_last_pos[1], ego_heading[last_valid_step].item(), ego_shape[10,1], ego_shape[10,0]))
    # distances = np.linalg.norm(points_array - adv_pos)
    # closest_index = np.argmin(distances)
    # collision_point = collision_points[int(closest_index)]

    adv_mask = np.zeros_like(ego_mask)
    adv_mask[:last_valid_step + 1] = 1
    data_dict["decoder/agent_valid_mask"][:, adv_id] = adv_mask

    # ===== Position =====
    # import random
    # collision_point = random.choice(collision_points) # choose the nearest edge point
    data_dict["decoder/agent_position"][
        last_valid_step,
        adv_id, :2] = collision_point  # ego_traj[last_valid_step] - np.random.normal(loc=0.0, scale=2, size=3)
    # ====================

    # ===== Heading =====
    data_dict["decoder/agent_heading"][last_valid_step, adv_id] = adv_heading + np.random.normal(
        loc=0.0, scale=0.1, size=1
    )
    print("Ego heading: ", ego_heading[last_valid_step])
    print("Adv heading: ", adv_heading)

    # ===================

    # ===== Velocity =====
    # adv_velocity[last_valid_step] = ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=2)
    # adv_vel = 0.5 * (adv_vel + np.random.normal(loc=0.0, scale=0.1, size=2))
    ego_vel = 0.5 * (ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.1, size=2))
    adv_vel = ego_vel
    print("Ego velocity: ", ego_vel, ego_velocity[last_valid_step])
    print("Adv velocity: ", adv_vel)
    data_dict["decoder/agent_velocity"][last_valid_step, ego_id] = ego_vel
    data_dict["decoder/agent_velocity"][last_valid_step, adv_id] = adv_vel
    # ====================

    return data_dict, adv_id


def choose_nearest_adv(data_dict):
    # find nearest adv for ego's ending position
    sdc_id = data_dict["decoder/sdc_index"]
    all_ooi = data_dict["decoder/agent_id"]
    sdc_mask = data_dict["decoder/agent_valid_mask"][:91, sdc_id]
    last_valid_step = np.where(sdc_mask)[0][-1]

    min_dist = float('inf')
    adv_id = None
    adv_closes_step = None

    for id in all_ooi:
        if id == sdc_id:
            continue
        agent_mask = data_dict["decoder/agent_valid_mask"][:91, id]

        mask = sdc_mask & agent_mask
        valid_steps = np.where(mask)[0]  # get the original indices where valid_step is True
        sdc_pos = data_dict["decoder/agent_position"][:91, id, :2]
        agent_pos = data_dict["decoder/agent_position"][:91, id, :2]
        distances = np.linalg.norm(sdc_pos[mask] - agent_pos[mask])
        dist = np.min(distances)
        closest_index = np.argmin(distances)
        closest_step = valid_steps[closest_index]

        if dist < min_dist:
            adv_id = id
            min_dist = dist
            adv_closes_step = closest_step

    # now get adv last valid step's information
    adv_pos = data_dict["decoder/agent_position"][adv_closes_step, adv_id, :2]
    adv_heading = data_dict["decoder/agent_heading"][adv_closes_step, adv_id]
    adv_vel = data_dict["decoder/agent_velocity"][adv_closes_step, adv_id]

    return adv_id, adv_pos, adv_heading, adv_vel, adv_closes_step


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
    # ===== Position =====
    adv_traj[last_valid_step] = ego_traj[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=3)
    print("Ego position: ", ego_traj[last_valid_step])
    print("Adv position: ", adv_traj[last_valid_step])
    # ====================

    # ===== Heading =====
    adv_heading[last_valid_step] = ego_heading[last_valid_step] + np.random.normal(loc=0.0, scale=0.1, size=1)
    print("Ego heading: ", ego_heading[last_valid_step])
    print("Adv heading: ", adv_heading[last_valid_step])
    # ===================

    # ===== Velocity =====
    # adv_velocity[last_valid_step] = ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=2)
    adv_vel = 0.5 * (ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.1, size=2))
    adv_velocity[last_valid_step] = adv_vel
    ego_vel = 0.5 * (ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.1, size=2))
    print("Ego velocity: ", ego_vel, ego_velocity[last_valid_step])
    print("Adv velocity: ", adv_velocity[last_valid_step])
    data_dict["decoder/agent_velocity"][last_valid_step, ego_id] = ego_vel
    # ====================

    # ===== Shape =====
    for i in range(data_dict["decoder/agent_shape"].shape[0]):
        adv_shape[i] = ego_shape[last_valid_step]
    # =================

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


def run_backward_prediction_with_teacher_forcing(
    model, config, backward_input_dict, tokenizer, not_teacher_forcing_ids
):
    # pdb.set_trace()
    device = backward_input_dict["decoder/agent_position"].device

    # Force to run backward prediction first to make sure the data is tokenized correctly.
    tok_data_dict, _ = tokenizer.tokenize(backward_input_dict, backward_prediction=True)
    backward_input_dict.update(tok_data_dict)

    backward_input_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
    backward_input_dict["in_backward_prediction"] = torch.tensor([1], dtype=bool).to(device)
    with torch.no_grad():
        ar_func = model.model.autoregressive_rollout_backward_prediction_with_replay
        # ar_func = model.model.autoregressive_rollout_backward_prediction
        backward_output_dict = ar_func(
            backward_input_dict,
            num_decode_steps=None,
            sampling_method=config.SAMPLING.SAMPLING_METHOD,
            temperature=config.SAMPLING.TEMPERATURE,
            not_teacher_forcing_ids=not_teacher_forcing_ids,
        )
    backward_output_dict = tokenizer.detokenize(
        backward_output_dict,
        detokenizing_gt=False,
        backward_prediction=True,
        flip_wrong_heading=True,
    )
    return backward_output_dict
