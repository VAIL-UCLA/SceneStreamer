import torch

from scenestreamer.utils import debug_tools


def toy_test():
    print("Start")

    from time import time as t

    s = t()

    cfg_file = "cfgs/motion_debug.yaml"
    config = debug_tools.get_debug_config(cfg_file)
    config.DATA.TRAINING_DATA_DIR = "/data/datasets/scenarionet/waymo/training"
    config.DATA.TEST_DATA_DIR = "/data/datasets/scenarionet/waymo/validation"

    config.PREPROCESSING.keep_all_data = True

    dataloader = debug_tools.get_debug_dataloader(config=config, in_evaluation=False)

    print("After dataloader", t() - s)
    for input_dict in dataloader:
        B, M = input_dict["encoder/map_feature"].shape[:2]
        B, T, N = input_dict["encoder/agent_feature"].shape[:3]

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if torch.isinf(v).any() or torch.isnan(v).any():
                print(f"Found {k} has nan or inf. Data: ", input_dict["scenario_id"])

        print(t() - s, "map: ", M, " agent: ", N)
        s = t()
        continue


if __name__ == '__main__':
    toy_test()
