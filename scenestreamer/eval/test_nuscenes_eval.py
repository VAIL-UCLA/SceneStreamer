from tqdm import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.utils import debug_tools


def toy_test():
    cfg_file = "cfgs/motion_debug.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    config.PREPROCESSING["keep_all_data"] = True
    config.EVALUATION.PREDICT_ALL_AGENTS = False

    config.DATA.TRAINING_DATA_DIR = 'data/nuscenes_debug'
    config.DATA.TEST_DATA_DIR = 'data/nuscenes_debug'

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

    for data_dict in tqdm(dataloader):
        data_dict["decoder/output_action"] = data_dict["decoder/target_action"]
        ground_truth_trajectory = data_dict["decoder/future_agent_position"][..., :2]

        # TODO: Call the eval function from nuscenes?


if __name__ == '__main__':
    toy_test()
