import numpy as np
import torch
from tqdm import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.eval.waymo_motion_prediction_evaluator import transform_to_global_coordinate
from scenestreamer.eval.wosac_eval import wosac_evaluation
from scenestreamer.utils import debug_tools

# from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
# from waymo_open_dataset.protos import sim_agents_submission_pb2

# from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
# from waymo_open_dataset.utils.sim_agents import visualizations
# from waymo_open_dataset.utils import trajectory_utils


def data_dict_to_sim_agents_prediction(data_dict, pred_trajs, pred_headings):
    """
    Transforms data_dict (the global dictionary that contains all information) to the format used in the sim agents prediction challenge evaluation pipeline.
    pipeline.

    Args:
        data_dict: the global dictionary that contains all information. Important keys:
            encoder/
            decoder/
            decoder/
            eval/
            in_evaluation
            metadata/
            decoder/
        pred_trajs: the predicted trajectories. Shape: (B, T, N, 2).
        pred_headings: the predicted headings. Shape: (B, T, N).

    Returns:
        prediction_dict: a dictionary that contains all information needed for the sim agents prediction pipeline.
            pred_trajs: Trajectories of agents to evaluate. Shape: (B * num_modes, T, N, 2).
            pred_scores: Scores of the predicted trajectories. Shape: (B * num_modes, N).
            pred_to_scenario_id: Scenario ID of each prediction. Shape: (B * num_modes,).
            Anything with the prefix eval/ from data_dict.
    """
    # scores = data_dict["decoder/output_score"]
    # pred_trajs = data_dict["decoder/reconstructed_position"]

    num_modes_for_eval = 32

    # Let's test the GT trajectory first.

    B, T, N, _ = pred_trajs.shape
    scores = pred_trajs.new_ones(size=(B, N))

    scores_of_interested_agents = []
    pred_trajs_of_interested_agents = []
    pred_headings_of_interested_agents = []
    for batch_index, track_indices in enumerate(data_dict["eval/modeled_agent_id"]):
        for mode_index in range(num_modes_for_eval):
            sc = np.stack(
                [
                    scores[batch_index][agent_index].detach().cpu().numpy()
                    for agent_index in track_indices if agent_index != -1
                ],
                axis=0
            )
            traj = np.stack(
                [
                    pred_trajs[batch_index][:, agent_index].detach().cpu().numpy()
                    for agent_index in track_indices if agent_index != -1
                ],
                axis=1
            )
            heading = np.stack(
                [
                    pred_headings[batch_index][:, agent_index].detach().cpu().numpy()
                    for agent_index in track_indices if agent_index != -1
                ],
                axis=1
            )
            pred_trajs_of_interested_agents.append(traj)
            scores_of_interested_agents.append(sc)
            pred_headings_of_interested_agents.append(heading)

    prediction_dict = {
        "pred_trajs": pred_trajs_of_interested_agents,
        "pred_scores": scores_of_interested_agents,
        "pred_headings": pred_headings_of_interested_agents,
        "pred_to_scenario_id": np.repeat(data_dict["scenario_id"], num_modes_for_eval, axis=0),
        "expanded_map_center": data_dict["metadata/map_center"][:, None].repeat(1, num_modes_for_eval, 1).flatten(0, 1),
        "expanded_map_heading": data_dict["metadata/map_heading"][:, None].repeat(1, num_modes_for_eval,
                                                                                  1).flatten(0, 1),
    }

    # Copy over all eval/ keys from data_dict.
    for k, v in data_dict.items():
        if k.startswith("eval/") or k.startswith("metadata/"):
            prediction_dict[k] = v

    # Convert all torch.Tensor to numpy.
    prediction_dict = {
        k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in prediction_dict.items()
    }
    return prediction_dict


def test_tokenizer_with_wosac_eval():
    """
    Run the tokenizer and evaluate the result with the Waymo evaluation pipeline.
    """
    cfg_file = "cfgs/motion_debug_2_local_train.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    config.PREPROCESSING["keep_all_data"] = True
    config.EVALUATION.PREDICT_ALL_AGENTS = True
    config.DATA.TRAINING_DATA_DIR = "data/waymo_8s_debug"
    config.DATA.TEST_DATA_DIR = "data/waymo_8s_debug"

    config.DATA.SAMPLE_INTERVAL = {'training': 1, 'test': 1}

    config.TOKENIZATION.TOKENIZATION_METHOD = "delta_delta"
    config.TOKENIZATION.X_MAX = 4
    config.TOKENIZATION.X_MIN = -4
    config.TOKENIZATION.Y_MAX = 3
    config.TOKENIZATION.Y_MIN = -3

    # config.TOKENIZATION.TOKENIZATION_METHOD = "delta"

    config.DATA["SD_PASSTHROUGH"] = True
    # Note: The datamodules, when iterated over, return dictionaries (data_dicts).
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=10,
        train_num_workers=0,
        val_batch_size=8,
        val_num_workers=0,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")  # "fit" here doesn't mean anything (yet!)
    dataloader = datamodule.val_dataloader()

    from scenestreamer.tokenization import motion_tokenizers
    tokenizer = motion_tokenizers.get_tokenizer(config)

    validation_list = []
    for data_dict in tqdm(dataloader):
        # We can check our discretization error by detokenizing the tokenized ground truth.
        data_dict["decoder/output_action"] = data_dict["decoder/target_action"]
        with torch.no_grad():
            data_dict = tokenizer.detokenize(data_dict)
        # pred_trajs = data_dict["decoder/reconstructed_position"]
        pred_headings = data_dict["decoder/reconstructed_heading"]
        pred_trajs = data_dict["decoder/future_agent_position"][..., :2]

        new_prediction_dict = data_dict_to_sim_agents_prediction(
            data_dict, pred_trajs=pred_trajs, pred_headings=pred_headings
        )

        new_prediction_dict = transform_to_global_coordinate(new_prediction_dict)

        validation_list.append(new_prediction_dict)
    # Validation list: A list of pred_dicts.
    print("Evaluating...")
    scenario_metrics, aggregate_metrics = wosac_evaluation(validation_list)
    print(scenario_metrics)
    print("\n\n\n")
    print(aggregate_metrics)


# def run_wosac_submission(
#     test_file='/scratch/metadrive/data/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150'):
#     # Read the dataset from the .tfrecord file.
#     filename = tf.io.matching_files(test_file)
#
#     dataset = tf.data.TFRecordDataset(filename)
#     dataset_iterator = dataset.as_numpy_iterator()
#
#     bytes_example = next(dataset_iterator)
#     scenario = scenario_pb2.Scenario.FromString(bytes_example)
#     print(f'Checking type: {type(scenario)}')
#     print(f'Loaded scenario with ID: {scenario.scenario_id}')
#
#     print(f'Simulation length, in steps: {submission_specs.N_SIMULATION_STEPS}')
#     print(
#         f'Duration of a step, in seconds: {submission_specs.STEP_DURATION_SECONDS}s (frequency: {1 / submission_specs.STEP_DURATION_SECONDS}Hz)')
#     print(f'Number of parallel simulations per Scenario: {submission_specs.N_ROLLOUTS}')
#
#     logged_trajectories, simulated_states = simulate_with_extrapolation(
#         scenario, print_verbose_comments=True)
#     generate_submission([scenario], [simulated_states], [logged_trajectories])

if __name__ == "__main__":
    test_tokenizer_with_wosac_eval()
    # run_wosac_submission()
