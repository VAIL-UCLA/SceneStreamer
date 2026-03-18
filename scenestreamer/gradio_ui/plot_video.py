import copy
import tempfile
import pickle
import os
from scenestreamer.utils import REPO_ROOT
from scenestreamer.gradio_ui.metadrive_render import render
import subprocess


def plot_gt_video(scenario):
    video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=os.environ['GRADIO_TEMP_DIR'])

    with tempfile.TemporaryDirectory(dir=os.environ['GRADIO_TEMP_DIR']) as in_dir:
        in_pickle = tempfile.NamedTemporaryFile(suffix='.pkl', prefix='sd_', delete=True, dir=in_dir)
        pickle.dump(scenario, in_pickle)
        subprocess.run(
            [
                'python', REPO_ROOT / 'scenestreamer/gradio_ui/metadrive_render.py', '--input_dir', in_dir, '--output_path',
                video_file.name
            ]
        )
        # render(in_dir, video_file.name)
    return video_file


def plot_pred_video(
    scenario, output_dict, agents_safe_0, agents_safe_1, agents_turn_stop, agents_turn_straight, agents_turn_left,
    agents_turn_right, agents_turn_uturn
):
    scenario_description_copy = copy.deepcopy(scenario)
    agent_id_union = agents_safe_0 + agents_safe_1 + agents_turn_stop + agents_turn_straight + agents_turn_left + agents_turn_right + agents_turn_uturn
    agent_id_union = set(agent_id_union)
    id_track_map = {
        agent_id: track_id
        for agent_id, track_id in
        zip(output_dict["encoder/agent_id"].tolist(), output_dict["encoder/track_name"].tolist())
    }
    agent_track_map = {k: v for k, v in id_track_map.items() if k in agent_id_union}

    for agent_id, track in agent_track_map.items():
        # reconstructed position only has dim 2
        track_state = scenario_description_copy["tracks"][str(track)]['state']
        track_state['position'][11:, :2] = output_dict["decoder/reconstructed_position"][:, agent_id, :] + output_dict[
            "metadata/map_center"][:2]
        track_state['velocity'][11:] = output_dict["decoder/reconstructed_velocity"][:, agent_id, :]
        track_state['heading'][11:] = output_dict["decoder/reconstructed_heading"][:, agent_id]
        track_state['valid'][11:] = output_dict["decoder/interpolated_target_action_valid_mask"][:, agent_id]

    video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=os.environ['GRADIO_TEMP_DIR'])

    with tempfile.TemporaryDirectory(dir=os.environ['GRADIO_TEMP_DIR']) as in_dir:
        in_pickle = tempfile.NamedTemporaryFile(suffix='.pkl', prefix='sd_', delete=True, dir=in_dir)
        pickle.dump(scenario_description_copy, in_pickle)
        subprocess.run(
            [
                'python', REPO_ROOT / 'scenestreamer/gradio_ui/metadrive_render.py', '--input_dir', in_dir, '--output_path',
                video_file.name
            ]
        )
        # render(in_dir, video_file.name)

    return video_file
