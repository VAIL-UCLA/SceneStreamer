import hydra
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.tokenization.motion_tokenizers import DeltaDeltaTokenizer
from scenestreamer.utils import REPO_ROOT
from scenestreamer.utils import debug_tools, get_time_str
from scenestreamer.utils import utils


def _unbatch_to_numpy(tensor_dict, index=0):
    ret = {}
    for k, v in tensor_dict.items():
        if isinstance(v[index], (np.ndarray, str, np.str_)):
            ret[k] = v[index]
        else:
            ret[k] = v[index].numpy()
    return ret


def _batch_to_tensor(array_list):
    return torch.from_numpy(np.array(array_list))


def evaluate_tokenizer(tokenizer, dataloader, num_scenarios_limit=100, test_tokenizer=False):
    # error_list = []
    ade_list = []
    fde_list = []
    num_scenarios = 0
    num_objects = 0

    pbar = tqdm(total=num_scenarios_limit)

    data_dict_list = []

    for input_dict in dataloader:

        B = input_dict["in_evaluation"].shape[0]
        num_scenarios += B
        if test_tokenizer:
            # Do the tokenization (already done by dataloader and filled decoder/target_action.
            output_dict, stats = tokenizer.tokenize(input_dict)
            # error_list.append(stats['reconstruction_error'])

            output_dict["decoder/output_action"] = output_dict["decoder/target_action"]
            fill_zero = ~output_dict["decoder/target_action_valid_mask"]
            output_dict["decoder/input_action_valid_mask"][fill_zero] = False

        else:
            raise ValueError
            input_dict["decoder/output_action"] = input_dict["decoder/target_action"]

        with torch.no_grad():
            input_dict = tokenizer.detokenize(output_dict, detokenizing_gt=True)

        # recon_mask = input_dict['decoder/interpolated_target_action_valid_mask'][:, 4::5]  # 80 -> 16
        # input_act_mask = input_dict['decoder/input_action_valid_mask'][:, 2:]  # 18 -> 16
        # assert (recon_mask == input_act_mask).all()
        # assert (recon_mask >= input_dict['decoder/agent_valid_mask'][:, 14::5]).all()

        pred = input_dict["decoder/reconstructed_position"]
        pred_head = input_dict["decoder/reconstructed_heading"]
        if pred.shape[1] == 96:
            pred = pred[:, 11:]
            pred_head = pred_head[:, 11:]
        elif pred.shape[1] == 91:
            pred = pred[:, 11:]
            pred_head = pred_head[:, 11:]
        else:
            raise ValueError
        gt = input_dict["decoder/agent_position"][:, 11:, :, :2]
        gt_head = input_dict["decoder/agent_heading"][:, 11:]

        target_action_valid_mask = input_dict["decoder/target_action_valid_mask"]
        gt_action = input_dict["decoder/target_action"].clone()
        gt_action[~target_action_valid_mask] = 0
        # encodings = torch.nn.functional.one_hot(gt_action, num_classes=tokenizer.num_actions)

        future_valid_mask = input_dict["decoder/agent_valid_mask"][:, 11:]
        current_valid_mask = input_dict["decoder/current_agent_valid_mask"]
        raw_valid_mask = torch.logical_and(future_valid_mask, current_valid_mask[:, None])

        T_pred = pred.shape[1]
        T_gt = gt.shape[1]
        T_compare = min(T_pred, T_gt)
        pred = pred[:, :T_compare]
        gt = gt[:, :T_compare]
        pred_head = pred_head[:, :T_compare]
        error = (pred - gt).norm(dim=-1)
        error_head = utils.wrap_to_pi(pred_head[:, :T_compare] - gt_head[:, :T_compare]).abs()

        contours = utils.cal_polygon_contour_torch(
            x=pred[..., 0],
            y=pred[..., 1],
            theta=pred_head,
            length=input_dict["decoder/current_agent_shape"][..., 0][:, None],
            width=input_dict["decoder/current_agent_shape"][..., 1][:, None]
        )
        gt_contours = utils.cal_polygon_contour_torch(
            x=gt[..., 0],
            y=gt[..., 1],
            theta=gt_head,
            length=input_dict["decoder/current_agent_shape"][..., 0][:, None],
            width=input_dict["decoder/current_agent_shape"][..., 1][:, None]
        )

        contour_error = (contours - gt_contours).norm(dim=-1).mean(dim=-1)

        error = error[:, 4::5]
        error_head = error_head[:, 4::5]
        contour_error = contour_error[:, 4::5]

        stat = {}

        at = input_dict["decoder/agent_type"].unsqueeze(1).expand(-1, 80, -1)

        # === all ===
        if input_dict["decoder/reconstructed_valid_mask"].shape[1] == 96:
            input_mask = input_dict["decoder/reconstructed_valid_mask"][:, :-5]
        elif input_dict["decoder/reconstructed_valid_mask"].shape[1] == 91:
            input_mask = input_dict["decoder/reconstructed_valid_mask"][:, :]
        else:
            raise ValueError
        formal_valid_mask = input_dict["decoder/agent_valid_mask"]
        assert input_mask.shape == formal_valid_mask.shape
        formal_valid_mask = torch.logical_and(formal_valid_mask, input_mask)
        formal_valid_mask = formal_valid_mask[:, ::5]

        num_all_objects = formal_valid_mask.sum().item()
        if error.shape[1] == 17:
            error = error[:, :-1]
        if error_head.shape[1] == 17:
            error_head = error_head[:, :-1]
        if contour_error.shape[1] == 17:
            contour_error = contour_error[:, :-1]
        if formal_valid_mask.shape[1] == 17:
            formal_valid_mask = formal_valid_mask[:, :-1]
        if formal_valid_mask.shape[1] == 19:
            formal_valid_mask = formal_valid_mask[:, 3:]

        # tmp = (error * formal_valid_mask)[0].sum(0) / formal_valid_mask[0].sum(0)
        error_masked = error * formal_valid_mask
        ade = (error_masked).sum()  # / valid_mask.sum()
        ade_head = (error_head * formal_valid_mask).sum()  # / valid_mask.sum()
        ade_contour = (contour_error * formal_valid_mask).sum()  # / valid_mask.sum()
        # ade_list.append(ade)

        valid_mask_any_step = formal_valid_mask.any(dim=1)
        fde = torch.masked_fill(error, ~formal_valid_mask, float("-inf")).max(dim=1)[0]
        fde = torch.masked_fill(fde, ~valid_mask_any_step, 0).sum()  # / valid_mask_any_step.sum()

        # fde_list.append(fde)

        stat["all/ade_contour_sum"] = ade_contour.item()
        stat["all/ade_head_sum"] = ade_head.item()
        stat["all/ade_sum"] = ade.item()
        stat["all/ade_count"] = formal_valid_mask.sum().item()
        stat["all/fde_sum"] = fde.item()
        stat["all/fde_count"] = valid_mask_any_step.sum().item()
        stat["all/num_objects"] = num_all_objects
        stat["all/num_scenarios"] = B

        # stat["all/cluster_use"] = torch.sum(encodings[target_action_valid_mask].float().mean(0) > 0).item()

        for ot in at.unique():
            if ot == -1:
                continue

            is_type = at == ot
            valid_mask = torch.logical_and(formal_valid_mask, is_type[:, ::5])
            N = int(valid_mask.sum())
            if N == 0:
                stat["obj{}/ade".format(ot)] = -1
                stat["obj{}/fde".format(ot)] = -1
                stat["obj{}/num_objects".format(ot)] = 0
                stat["obj{}/num_scenarios".format(ot)] = 0
                continue

            ade_contour = (contour_error * valid_mask).sum()  # / valid_mask.sum()
            ade_head = (error_head * valid_mask).sum()  # / valid_mask.sum()
            ade = (error * valid_mask).sum()  # / valid_mask.sum()

            real_ade = ade / valid_mask.sum()

            valid_mask_any_step = valid_mask.any(dim=1)
            fde = torch.masked_fill(error, ~valid_mask, float("-inf")).max(dim=1)[0]
            fde = torch.masked_fill(fde, ~valid_mask_any_step, 0).sum()  # / valid_mask_any_step.sum()
            stat["obj{}/ade_sum".format(ot)] = ade.item()
            stat["obj{}/ade_contour_sum".format(ot)] = ade_contour.item()
            stat["obj{}/ade_head_sum".format(ot)] = ade_head.item()
            stat["obj{}/ade_count".format(ot)] = valid_mask.sum().item()
            stat["obj{}/fde_sum".format(ot)] = fde.item()
            stat["obj{}/fde_count".format(ot)] = valid_mask_any_step.sum().item()
            stat["obj{}/num_objects".format(ot)] = N
            stat["obj{}/num_scenarios".format(ot)] = B
            # stat["obj{}/cluster_use".format(ot)] = torch.sum(encodings[target_action_valid_mask].float().mean(0) > 0
            #                                                  ).item()

        data_dict_list.append(stat)
        num_objects += num_all_objects
        pbar.update(B)
        if num_scenarios_limit is not None and num_scenarios > num_scenarios_limit:
            break

    pbar.close()

    return data_dict_list, num_objects, num_scenarios


def evaluate_tokenizer_gpt(tokenizer, dataloader, num_scenarios_limit=100, test_tokenizer=False):
    # error_list = []
    ade_list = []
    fde_list = []
    num_scenarios = 0
    num_objects = 0

    pbar = tqdm(total=num_scenarios_limit)

    data_dict_list = []

    for input_dict in dataloader:

        B = input_dict["in_evaluation"].shape[0]
        num_scenarios += B
        if test_tokenizer:
            # Do the tokenization (already done by dataloader and filled decoder/target_action.
            output_dict, stats = tokenizer.tokenize_gpt_style(input_dict)
            # error_list.append(stats['reconstruction_error'])
            input_dict["decoder/output_action"] = output_dict["decoder/target_action"]

            fill_zero = ((input_dict["decoder/output_action"] == -1) & input_dict["decoder/input_action_valid_mask"])
            input_dict["decoder/output_action"][fill_zero] = tokenizer.default_action

        # else:
        #     input_dict["decoder/output_action"] = input_dict["decoder/target_action"]

        with torch.no_grad():
            input_dict = tokenizer.detokenize_gpt_style(input_dict)

        recon_mask = input_dict['decoder/interpolated_target_action_valid_mask'][:, ::5][:, :-1]  # 91 -> 18
        input_act_mask = input_dict['decoder/input_action_valid_mask'][:, :]
        assert (recon_mask == input_act_mask).all()

        pred = input_dict["decoder/reconstructed_position"]
        gt = input_dict["decoder/agent_position"][..., :2]

        target_action_valid_mask = input_dict["decoder/target_action_valid_mask"]
        gt_action = input_dict["decoder/target_action"].clone()
        gt_action[~target_action_valid_mask] = 0
        encodings = torch.nn.functional.one_hot(gt_action, num_classes=tokenizer.num_actions)

        # future_valid_mask = input_dict["decoder/future_agent_valid_mask"]
        # current_valid_mask = input_dict["decoder/current_agent_valid_mask"]
        # raw_valid_mask = torch.logical_and(future_valid_mask, current_valid_mask[:, None])

        # T_pred = pred.shape[1]
        # T_gt = gt.shape[1]
        # T_compare = min(T_pred, T_gt)
        # pred = pred[:, :T_compare]
        # gt = gt[:, :T_compare]
        error = (pred - gt).norm(dim=-1)

        error = error[:, ::5][:, 1:]

        stat = {}

        at = input_dict["decoder/agent_type"].unsqueeze(1).expand(-1, 91, -1)

        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        next_action_valid_mask = input_dict["decoder/target_action_valid_mask"]

        # === all ===
        formal_valid_mask = action_valid_mask & next_action_valid_mask
        num_all_objects = formal_valid_mask.sum().item()

        tmp = (error * formal_valid_mask)[0].sum(0) / formal_valid_mask[0].sum(0)

        ade = (error * formal_valid_mask).sum()  # / valid_mask.sum()
        # ade_list.append(ade)

        valid_mask_any_step = formal_valid_mask.any(dim=1)
        fde = torch.masked_fill(error, ~formal_valid_mask, float("-inf")).max(dim=1)[0]
        fde = torch.masked_fill(fde, ~valid_mask_any_step, 0).sum()  # / valid_mask_any_step.sum()
        # fde_list.append(fde)

        stat["all/ade_sum"] = ade.item()
        stat["all/ade_count"] = formal_valid_mask.sum().item()
        stat["all/fde_sum"] = fde.item()
        stat["all/fde_count"] = valid_mask_any_step.sum().item()
        stat["all/num_objects"] = num_all_objects
        stat["all/num_scenarios"] = B

        stat["all/cluster_use"] = torch.sum(encodings[target_action_valid_mask].float().mean(0) > 0).item()

        for ot in at.unique():
            if ot == -1:
                continue

            is_type = at == ot
            valid_mask = torch.logical_and(formal_valid_mask, is_type[:, ::5][:, 1:])
            N = int(valid_mask.sum())
            if N == 0:
                stat["obj{}/ade".format(ot)] = -1
                stat["obj{}/fde".format(ot)] = -1
                stat["obj{}/num_objects".format(ot)] = 0
                stat["obj{}/num_scenarios".format(ot)] = 0
                continue

            ade = (error * valid_mask).sum()  # / valid_mask.sum()
            valid_mask_any_step = valid_mask.any(dim=1)
            fde = torch.masked_fill(error, ~valid_mask, float("-inf")).max(dim=1)[0]
            fde = torch.masked_fill(fde, ~valid_mask_any_step, 0).sum()  # / valid_mask_any_step.sum()
            stat["obj{}/ade_sum".format(ot)] = ade.item()
            stat["obj{}/ade_count".format(ot)] = valid_mask.sum().item()
            stat["obj{}/fde_sum".format(ot)] = fde.item()
            stat["obj{}/fde_count".format(ot)] = valid_mask_any_step.sum().item()
            stat["obj{}/num_objects".format(ot)] = N
            stat["obj{}/num_scenarios".format(ot)] = B
            stat["obj{}/cluster_use".format(ot)] = torch.sum(encodings[target_action_valid_mask].float().mean(0) > 0
                                                             ).item()

        data_dict_list.append(stat)
        num_objects += num_all_objects
        pbar.update(B)
        if num_scenarios_limit is not None and num_scenarios > num_scenarios_limit:
            break

    pbar.close()

    return data_dict_list, num_objects, num_scenarios


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="0220_midgpt.yaml")
def test_delta_delta(config):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(config, False)
    OmegaConf.set_struct(config, True)

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=1,
        train_num_workers=config.num_workers,
        train_prefetch_factor=config.prefetch_factor,
        val_batch_size=1,
        val_num_workers=config.val_num_workers,
        val_prefetch_factor=config.prefetch_factor,
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    # config.TOKENIZATION.TOKENIZATION_METHOD = "delta_delta"

    file_name = str(config.TOKENIZATION.TOKENIZATION_METHOD)

    from scenestreamer.tokenization import get_tokenizer
    tokenizer = get_tokenizer(config=config)

    num_scenarios_limit = min(500, len(datamodule.val_dataset))

    stat, num_objects, num_scenarios = evaluate_tokenizer(
        tokenizer, dataloader, test_tokenizer=True, num_scenarios_limit=num_scenarios_limit
    )
    exp_state = {
        "total_num_scenarios": num_scenarios,
        "total_num_objects": num_objects,
        "file_name": file_name,
    }
    stat = pd.DataFrame(stat)
    for name in ["all", "obj1", "obj2", "obj3"]:

        # if name in ['obj2', 'obj3']:
        #     continue

        if f"{name}/ade_sum" not in stat:
            continue

        ade = stat[f"{name}/ade_sum"].sum() / stat[f"{name}/ade_count"].sum()
        ade_head = stat[f"{name}/ade_head_sum"].sum() / stat[f"{name}/ade_count"].sum()
        ade_contour = stat[f"{name}/ade_contour_sum"].sum() / stat[f"{name}/ade_count"].sum()
        fde = stat[f"{name}/fde_sum"].sum() / stat[f"{name}/fde_count"].sum()
        obj_count = stat[f"{name}/num_objects"].sum()
        scenario_count = stat[f"{name}/num_scenarios"].sum()
        exp_state.update(
            {
                f"{name}/ade": ade,
                f"{name}/ade_head": ade_head,
                f"{name}/ade_contour": ade_contour,
                f"{name}/fde": fde,
                f"{name}/num_objects": obj_count,
                f"{name}/num_scenarios": scenario_count,
                # f"{name}/cluster_use": stat[f"{name}/cluster_use"].mean()
            }
        )
    print(f"{num_scenarios=}, {num_objects=}.\n" f"{exp_state}" f"\n==========\n")
    print({k: round(v, 4) for k, v in exp_state.items() if "obj1" in k})
    print({k: round(v, 4) for k, v in exp_state.items() if "obj2" in k})
    print({k: round(v, 4) for k, v in exp_state.items() if "obj3" in k})
    print(f"\n==========\n")
    print({k: round(v, 4) for k, v in exp_state.items() if "all" in k})
    print(f"\n==========\n")
    # Print average for obj1/2/3
    keys = [k.split("obj3/")[-1] for k, v in exp_state.items() if "obj3" in k]
    res = {}
    for k in keys:
        res[k] = np.mean([v for kee, v in exp_state.items() if kee.endswith(k) and (not kee.startswith("all"))])
    print({"avg/{}".format(k): round(v, 4) for k, v in res.items()})
    print(f"\n==========\n")
    df = pd.DataFrame([exp_state])

    def applytab(row):
        s = ""
        for v in row.values:
            if isinstance(v, float):
                s += f"{v:.3f}" + '\t'
            else:
                s += str(v) + '\t'
        print(s)

    # print('\t'.join(map(str,df.columns))) # to print the column names if required
    df.apply(applytab, axis=1)

    df.to_csv(f"{file_name}_EVAL.csv")


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def test_delta(config):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(config, False)
    config.DATA.TRAINING_DATA_DIR = 'data/20scenarios'
    config.DATA.TEST_DATA_DIR = 'data/20scenarios'
    OmegaConf.set_struct(config, True)

    num_scenarios_limit = 100

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=config.batch_size,
        train_num_workers=config.num_workers,
        train_prefetch_factor=config.prefetch_factor,
        val_batch_size=config.val_batch_size,
        val_num_workers=config.val_num_workers,
        val_prefetch_factor=config.prefetch_factor,
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    config.TOKENIZATION.TOKENIZATION_METHOD = "delta"

    file_name = "delta"

    from scenestreamer.tokenization import get_tokenizer
    tokenizer = get_tokenizer(config=config)

    stat, num_objects, num_scenarios = evaluate_tokenizer(
        tokenizer, dataloader, test_tokenizer=True, num_scenarios_limit=num_scenarios_limit
    )
    exp_state = {
        "total_num_scenarios": num_scenarios,
        "total_num_objects": num_objects,
        "file_name": file_name,
    }
    stat = pd.DataFrame(stat)
    for name in ["all", "obj1", "obj2", "obj3"]:

        # if name in ['obj2', 'obj3']:
        #     continue

        ade = stat[f"{name}/ade_sum"].sum() / stat[f"{name}/ade_count"].sum()
        fde = stat[f"{name}/fde_sum"].sum() / stat[f"{name}/fde_count"].sum()
        obj_count = stat[f"{name}/num_objects"].sum()
        scenario_count = stat[f"{name}/num_scenarios"].sum()
        exp_state.update(
            {
                f"{name}/ade": ade,
                f"{name}/fde": fde,
                f"{name}/num_objects": obj_count,
                f"{name}/num_scenarios": scenario_count,
                f"{name}/cluster_use": stat[f"{name}/cluster_use"].mean()
            }
        )
    print(f"{num_scenarios=}, {num_objects=}.\n" f"{exp_state}" f"\n==========\n")
    print({k: v for k, v in exp_state.items() if "obj1" in k})
    print({k: v for k, v in exp_state.items() if "obj2" in k})
    print({k: v for k, v in exp_state.items() if "obj3" in k})
    print(f"\n==========\n")
    df = pd.DataFrame([exp_state])

    def applytab(row):
        s = ""
        for v in row.values:
            if isinstance(v, float):
                s += f"{v:.3f}" + '\t'
            else:
                s += str(v) + '\t'
        print(s)

    # print('\t'.join(map(str,df.columns))) # to print the column names if required
    df.apply(applytab, axis=1)

    df.to_csv(f"{file_name}_EVAL.csv")


def test_precomputed(num_scenarios_limit=100):
    cfg_file = "cfgs/motion_debug.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)

    large_dataset_file = 'data/metadrive_processed_waymo/validation'
    config.DATA["TRAINING_DATA_DIR"] = large_dataset_file
    config.DATA["TEST_DATA_DIR"] = large_dataset_file

    dataloader = debug_tools.get_debug_dataloader(
        cfg_file=cfg_file, in_evaluation=False, train_num_workers=8, train_batch_size=16
    )

    # config.TOKENIZATION.FILE_NAME = "test_0308-2125.json"
    # config.TOKENIZATION.FILE_NAME = "test_0308-2210.json"
    # config.TOKENIZATION.FILE_NAME = "test_0308-2221.json"
    file_name = "precomputed_delta_delta_0309sol1.json"

    config.TOKENIZATION.FILE_NAME = file_name

    config.TOKENIZATION.TOKENIZATION_METHOD = None

    stat, num_objects, num_scenarios = evaluate_tokenizer(
        PrecomputedDeltaDeltaTokenizer(config),
        dataloader,
        test_tokenizer=True,
        num_scenarios_limit=num_scenarios_limit
    )
    exp_state = {
        "total_num_scenarios": num_scenarios,
        "total_num_objects": num_objects,
        "file_name": file_name,
    }
    stat = pd.DataFrame(stat)
    for name in ["all", "obj1", "obj2", "obj3"]:
        ade = stat[f"{name}/ade_sum"].sum() / stat[f"{name}/ade_count"].sum()
        fde = stat[f"{name}/fde_sum"].sum() / stat[f"{name}/fde_count"].sum()
        obj_count = stat[f"{name}/num_objects"].sum()
        scenario_count = stat[f"{name}/num_scenarios"].sum()
        exp_state.update(
            {
                f"{name}/ade": ade,
                f"{name}/fde": fde,
                f"{name}/num_objects": obj_count,
                f"{name}/num_scenarios": scenario_count,
                f"{name}/cluster_use": stat[f"{name}/cluster_use"].mean()
            }
        )
    print(f"{num_scenarios=}, {num_objects=}.\n" f"{exp_state}" f"\n==========\n")
    print({k: v for k, v in exp_state.items() if "obj1" in k})
    print({k: v for k, v in exp_state.items() if "obj2" in k})
    print({k: v for k, v in exp_state.items() if "obj3" in k})
    print(f"\n==========\n")
    df = pd.DataFrame([exp_state])

    def applytab(row):
        s = ""
        for v in row.values:
            if isinstance(v, float):
                s += f"{v:.3f}" + '\t'
            else:
                s += str(v) + '\t'
        print(s)

    # print('\t'.join(map(str,df.columns))) # to print the column names if required
    df.apply(applytab, axis=1)

    df.to_csv(f"{file_name}_EVAL.csv")


def grid_search_delta_tokenizer(num_scenarios_limit=5000):
    # large_dataset_file = 'data/waymo_8s_debug'
    # large_dataset_file = '/data1/datasets/metadrive_processed_waymo/validation'
    # large_dataset_file = '/home/zhenghao/Datasets/metadrive_processed_waymo/validation'
    large_dataset_file = '/data1/datasets/metadrive_processed_waymo/validation'
    tokenizer_class = DeltaTokenizer
    config = debug_tools.get_debug_config()
    config.DATA["TRAINING_DATA_DIR"] = large_dataset_file
    config.DATA["TEST_DATA_DIR"] = large_dataset_file
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=1,
        train_num_workers=0,
        val_batch_size=4,
        val_num_workers=4,
        train_prefetch_factor=2,
        val_prefetch_factor=2
    )
    datamodule.setup("fit")
    # dataloader = datamodule.val_dataloader()
    dataloader = datamodule.train_dataloader()
    file_name = "tokenizer_test_{}.csv".format(get_time_str())
    result = []

    for nbins in [13]:  # 17, 21, 25]:
        config.TOKENIZATION["NUM_BINS"] = nbins
        ymax = 35
        ymin = -3
        xmax = 2
        xmin = -2
        config.TOKENIZATION["X_MAX"] = xmax
        config.TOKENIZATION["X_MIN"] = xmin
        config.TOKENIZATION["Y_MAX"] = ymax
        config.TOKENIZATION["Y_MIN"] = ymin
        config.TOKENIZATION["NUM_BINS"] = nbins
        # config.TOKENIZATION["NUM_SKIPPED_STEPS"] = 1

        _, ade_list, fde_list, num_scenarios, num_objects = evaluate_tokenizer(
            tokenizer_class(config), dataloader, num_scenarios_limit=num_scenarios_limit
        )

        print(
            f"{xmax=}, {xmin=}, {ymax=}, {ymin=}, {nbins=}. "
            f"Reconstruction ADE: {np.mean(ade_list)}, FDE: {np.mean(fde_list)}. "
            f"Num scenarios: {num_scenarios}, Num objects: {num_objects}."
        )

        result.append(
            dict(
                X_MAX=xmax,
                X_MIN=xmin,
                Y_MAX=ymax,
                Y_MIN=ymin,
                NUM_BINS=nbins,
                # error=np.mean(error_list),
                ade=np.mean(ade_list),
                fde=np.mean(fde_list),
                num_scenarios=num_scenarios
            )
        )
        # pd.DataFrame(result).to_csv("tmp_" + file_name)
    pd.DataFrame(result).to_csv(file_name)


def grid_search_delta_delta_tokenizer(num_scenarios_limit=5000, batch_size=32):
    # large_dataset_file = 'data/waymo_8s_debug'
    # large_dataset_file = '/data/datasets/scenarionet/waymo/training'
    large_dataset_file = 'data/metadrive_processed_waymo/validation'
    # large_dataset_file = '/data1/datasets/metadrive_processed_waymo/validation'
    # large_dataset_file = '/home/zhenghao/Datasets/metadrive_processed_waymo/validation'

    tokenizer_class = DeltaDeltaTokenizer

    config = debug_tools.get_debug_config()
    config.DATA["TRAINING_DATA_DIR"] = large_dataset_file
    config.DATA["TEST_DATA_DIR"] = large_dataset_file
    config.TOKENIZATION.TOKENIZATION_METHOD = "delta_delta"
    # config.TRAINING["PREDICT_ALL_AGENTS"] = True

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=batch_size,
        train_num_workers=0,
        val_batch_size=4,
        val_num_workers=4,
        train_prefetch_factor=2,
        val_prefetch_factor=2
    )
    datamodule.setup("fit")
    # dataloader = datamodule.val_dataloader()
    dataloader = datamodule.train_dataloader()
    file_name = "tokenizer_test_{}.csv".format(get_time_str())
    result = []

    for nbins in [21]:  # 17, 21, 25]:
        for xlimit in [3.5]:
            for ymax in [3.5]:
                for ymin in [-3.5]:

                    config.TOKENIZATION["NUM_BINS"] = nbins

                    xmax = xlimit
                    xmin = -xlimit

                    config.TOKENIZATION["X_MAX"] = xmax
                    config.TOKENIZATION["X_MIN"] = xmin
                    config.TOKENIZATION["Y_MAX"] = ymax
                    config.TOKENIZATION["Y_MIN"] = ymin
                    config.TOKENIZATION["NUM_BINS"] = nbins

                    stat, num_objects, num_scenarios = evaluate_tokenizer(
                        tokenizer_class(config),
                        dataloader,
                        num_scenarios_limit=num_scenarios_limit,
                        # object_type=object_type,
                    )
                    exp_state = {
                        "X_MAX": xmax,
                        "X_MIN": xmin,
                        "Y_MAX": ymax,
                        "Y_MIN": ymin,
                        "NUM_BINS": nbins,
                        "total_num_scenarios": num_scenarios,
                        "total_num_objects": num_objects
                    }
                    stat = pd.DataFrame(stat)
                    for name in ["all", "obj1", "obj2", "obj3"]:
                        ade = stat[f"{name}/ade_sum"].sum() / stat[f"{name}/ade_count"].sum()
                        fde = stat[f"{name}/fde_sum"].sum() / stat[f"{name}/fde_count"].sum()
                        obj_count = stat[f"{name}/num_objects"].sum()
                        scenario_count = stat[f"{name}/num_scenarios"].sum()
                        exp_state.update(
                            {
                                f"{name}/ade": ade,
                                f"{name}/fde": fde,
                                f"{name}/num_objects": obj_count,
                                f"{name}/num_scenarios": scenario_count,
                                f"{name}/cluster_use": stat[f"{name}/cluster_use"].mean()
                            }
                        )
                    cu = {k: v for k, v in exp_state.items() if "obj3" in k}
                    print(
                        f"\n\n==========\n{xmax=}, {xmin=}, {ymax=}, {ymin=}, {nbins=}.\n"
                        f"{num_scenarios=}, {num_objects=}.\n"
                        f"{exp_state}"
                        f"\n==========\n"
                        f"{cu}"
                        f"\n==========\n"
                    )
                    result.append(exp_state)
                    pd.DataFrame(result).to_csv("tmp_" + file_name)
    df = pd.DataFrame(result)
    df.to_csv(file_name)
    print("Data saved to", file_name)
    print(df)
    return df


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def test_bicycle_model(config):

    from omegaconf import OmegaConf

    OmegaConf.set_struct(config, False)
    # config.DATA.TRAINING_DATA_DIR = 'data/20scenarios'
    # config.DATA.TEST_DATA_DIR = 'data/20scenarios'
    OmegaConf.set_struct(config, True)

    num_scenarios_limit = 100

    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=config.batch_size,
        train_num_workers=config.num_workers,
        train_prefetch_factor=config.prefetch_factor,
        val_batch_size=config.val_batch_size,
        val_num_workers=config.val_num_workers,
        val_prefetch_factor=config.prefetch_factor,
    )
    datamodule.setup("fit")
    dataloader = datamodule.val_dataloader()

    config.TOKENIZATION.TOKENIZATION_METHOD = "bicycle"

    file_name = "bicycle"

    from scenestreamer.tokenization import get_tokenizer
    tokenizer = get_tokenizer(config=config)

    stat, num_objects, num_scenarios = evaluate_tokenizer(
        tokenizer, dataloader, test_tokenizer=True, num_scenarios_limit=num_scenarios_limit
    )
    exp_state = {
        "total_num_scenarios": num_scenarios,
        "total_num_objects": num_objects,
        "file_name": file_name,
    }
    stat = pd.DataFrame(stat)
    for name in ["all", "obj1", "obj2", "obj3"]:
        # if name in ['obj2', 'obj3']:
        #     continue

        ade = stat[f"{name}/ade_sum"].sum() / stat[f"{name}/ade_count"].sum()
        ade_head = stat[f"{name}/ade_head_sum"].sum() / stat[f"{name}/ade_count"].sum()
        ade_contour = stat[f"{name}/ade_contour_sum"].sum() / stat[f"{name}/ade_count"].sum()
        fde = stat[f"{name}/fde_sum"].sum() / stat[f"{name}/fde_count"].sum()
        obj_count = stat[f"{name}/num_objects"].sum()
        scenario_count = stat[f"{name}/num_scenarios"].sum()
        exp_state.update(
            {
                f"{name}/ade": ade,
                f"{name}/ade_head": ade_head,
                f"{name}/ade_contour": ade_contour,
                f"{name}/fde": fde,
                f"{name}/num_objects": obj_count,
                f"{name}/num_scenarios": scenario_count,
                f"{name}/cluster_use": stat[f"{name}/cluster_use"].mean()
            }
        )
    print(f"{num_scenarios=}, {num_objects=}.\n" f"{exp_state}" f"\n==========\n")
    print({k: v for k, v in exp_state.items() if "obj1" in k})
    print({k: v for k, v in exp_state.items() if "obj2" in k})
    print({k: v for k, v in exp_state.items() if "obj3" in k})
    print(f"\n==========\n")
    df = pd.DataFrame([exp_state])

    def applytab(row):
        s = ""
        for v in row.values:
            if isinstance(v, float):
                s += f"{v:.3f}" + '\t'
            else:
                s += str(v) + '\t'
        print(s)

    # print('\t'.join(map(str,df.columns))) # to print the column names if required
    df.apply(applytab, axis=1)

    df.to_csv(f"{file_name}_EVAL.csv")


if __name__ == '__main__':
    # grid_search_delta_tokenizer(num_scenarios_limit=1000)
    # grid_search_delta_delta_tokenizer(num_scenarios_limit=200, batch_size=32)
    # test_precomputed(num_scenarios_limit=2000)
    test_delta_delta()
    # test_delta()
    # test_bicycle_model()
