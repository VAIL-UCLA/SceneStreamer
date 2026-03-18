import numpy as np
import torch
from tqdm import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.eval.waymo_eval import waymo_evaluation_optimized
from scenestreamer.eval.waymo_motion_prediction_evaluator import generate_submission, transform_to_global_coordinate
from scenestreamer.utils import debug_tools


def _unbatch_to_numpy(tensor_dict, index=0):
    ret = {}
    for k, v in tensor_dict.items():
        ret[k] = v[index].numpy()
    return ret


def _batch_to_tensor(array_list):
    return torch.from_numpy(np.array(array_list))


def run_waymo_eval(data_dict, pred_trajs):
    # scores = data_dict["decoder/output_score"]
    # pred_trajs = data_dict["decoder/reconstructed_position"]

    num_modes_nms = 32
    num_modes_for_eval = 6

    # Let's test the GT trajectory first.

    B, T, N, _ = pred_trajs.shape
    scores = pred_trajs.new_ones(size=(B, N))

    scores_of_interested_agents = []
    pred_trajs_of_interested_agents = []
    for batch_index, track_indices in enumerate(data_dict["eval/modeled_agent_id"]):
        for mode_index in range(num_modes_nms):
            sc = torch.stack(
                [
                    scores[batch_index][agent_index]  # .detach().cpu().numpy()
                    for agent_index in track_indices if agent_index != -1
                ],
                dim=0
            )
            traj = torch.stack(
                [
                    pred_trajs[batch_index][:, agent_index]  # .detach().cpu().numpy()
                    for agent_index in track_indices if agent_index != -1
                ],
                dim=1
            )
            pred_trajs_of_interested_agents.append(traj)
            scores_of_interested_agents.append(sc)

    from scenestreamer.eval.nms import batch_nms
    pred_trajs_of_interested_agents, scores_of_interested_agents = batch_nms(
        pred_trajs_of_interested_agents,
        scores_of_interested_agents,
        pred_to_scenario_id=np.repeat(data_dict["scenario_id"], num_modes_nms, axis=0),
        dist_thresh=2.5,  # Follow MTR
        num_ret_modes=num_modes_for_eval,  # TODO
        num_original_modes=num_modes_nms,  # TODO
    )

    prediction_dict = {
        "pred_trajs": pred_trajs_of_interested_agents,
        "pred_scores": scores_of_interested_agents,
        "pred_to_scenario_id": np.repeat(data_dict["scenario_id"], num_modes_for_eval, axis=0),
        "expanded_map_center": np.repeat(data_dict["metadata/map_center"], num_modes_for_eval, axis=0),
        "expanded_map_heading": np.repeat(data_dict["metadata/map_heading"], num_modes_for_eval, axis=0)
    }
    for k, v in data_dict.items():
        if k.startswith("eval/") or k.startswith("metadata/"):
            prediction_dict[k] = v
    prediction_dict = {
        k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in prediction_dict.items()
    }
    prediction_dict = transform_to_global_coordinate(prediction_dict)

    return prediction_dict


def toy_test():
    cfg_file = "cfgs/motion_debug_2_local_train.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)
    config.PREPROCESSING["keep_all_data"] = True
    config.EVALUATION.PREDICT_ALL_AGENTS = True
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=10,
        train_num_workers=0,
        val_batch_size=8,
        val_num_workers=2,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    validation_list = []
    for data_dict in tqdm(dataloader):
        pred_trajs = data_dict["decoder/future_agent_position"][..., :2]
        validation_list.append(run_waymo_eval(data_dict, pred_trajs=pred_trajs))
    result_dict, result_str = waymo_evaluation_optimized(validation_list)
    print(result_str)


def test_tokenizer_with_waymo_eval():
    cfg_file = "cfgs/motion_debug_2_local_train.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    config.PREPROCESSING["keep_all_data"] = True
    config.EVALUATION.PREDICT_ALL_AGENTS = False

    # config.DATA.TRAINING_DATA_DIR = "data/waymo_debug_f9d25ee7375ca381"
    # config.DATA.TEST_DATA_DIR = "data/waymo_debug_f9d25ee7375ca381"

    config.DATA.TRAINING_DATA_DIR = 'data/metadrive_processed_waymo/validation'
    config.DATA.TEST_DATA_DIR = 'data/metadrive_processed_waymo/validation'

    config.TOKENIZATION.TOKENIZATION_METHOD = "delta_delta"
    config.TOKENIZATION.X_MAX = 4
    config.TOKENIZATION.X_MIN = -4
    config.TOKENIZATION.Y_MAX = 3
    config.TOKENIZATION.Y_MIN = -3

    # config.TOKENIZATION.TOKENIZATION_METHOD = "delta"

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=10,
        train_num_workers=0,
        val_batch_size=8,
        val_num_workers=2,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    from scenestreamer.tokenization import motion_tokenizers
    tokenizer = motion_tokenizers.get_tokenizer(config)

    validation_list = []
    for data_dict in tqdm(dataloader):
        data_dict["decoder/output_action"] = data_dict["decoder/target_action"]
        with torch.no_grad():
            data_dict = tokenizer.detokenize(data_dict)
        # pred_trajs = data_dict["decoder/reconstructed_position"]

        pred_trajs = data_dict["decoder/future_agent_position"][..., :2]

        validation_list.append(run_waymo_eval(data_dict, pred_trajs=pred_trajs))
    result_dict, result_str, submission_dict = waymo_evaluation_optimized(validation_list, generate_submission=True)

    print("\n\n", result_str)

    submission_prefix = "test"
    account_name = "test"
    unique_method_name = "test"
    output_dir = "."
    path, duplicated_scenarios, done_scenarios = generate_submission(
        prefix=submission_prefix,
        account_name=account_name,
        unique_method_name=unique_method_name,
        output_dir=output_dir,
        **submission_dict
    )
    print(
        "Submission created at: {}. Finished {} scenarios. Duplicated scenarios: {}.".format(
            path, len(done_scenarios), duplicated_scenarios
        )
    )


if __name__ == '__main__':
    # toy_test()
    test_tokenizer_with_waymo_eval()
