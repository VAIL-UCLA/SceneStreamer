import lightning.pytorch as pl
import torch
import tqdm

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.models.motionlm_lightning import MotionLMLightning
from scenestreamer.utils import debug_tools


def toy_test(bs, in_eval):
    cfg_file = "cfgs/motion_default.yaml"
    config = debug_tools.get_debug_config(cfg_file=cfg_file)
    config.DATA.TRAINING_DATA_DIR = 'data/metadrive_processed_waymo/validation'
    config.DATA.TEST_DATA_DIR = 'data/metadrive_processed_waymo/validation'

    # config.MODEL.update(dict(
    #     D_MODEL=512,
    #     NUM_ATTN_LAYERS=6,
    #     NUM_ATTN_HEAD=8,
    #     NUM_DECODER_LAYERS=6,
    # ))
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=bs,
        train_num_workers=0,
        val_batch_size=bs * 6,
        val_num_workers=8,
        train_prefetch_factor=2,
        val_prefetch_factor=1
    )
    datamodule.setup("fit")
    if in_eval:
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.train_dataloader()

    model: pl.LightningModule = MotionLMLightning(config)
    model.train()
    model.cuda()

    for input_dict in dataloader:
        break

    # ===== Fill in some fake data =====
    N = config.PREPROCESSING.MAX_AGENTS
    M = config.PREPROCESSING.MAX_MAP_FEATURES
    V = config.PREPROCESSING.MAX_VECTORS

    def _extend_3rd_dim(key):
        tensor = input_dict[key]
        new_shape = list(tensor.shape)
        if len(new_shape) < 3:
            new_shape[1] = N
            input_dict[key] = tensor.new_ones(*new_shape)  #+ torch.randint(0, 1, size=new_shape)
        else:
            new_shape[2] = N
            input_dict[key] = tensor.new_ones(*new_shape)  #+ torch.randint(0, 1, size=new_shape)

    def _extend_map(key):
        tensor = input_dict[key]
        new_shape = list(tensor.shape)
        new_shape[1] = M
        if len(new_shape) > 2 and new_shape[2] > 3:
            new_shape[2] = V
        input_dict[key] = tensor.new_ones(*new_shape)  #+ torch.randint(0, 1, size=new_shape)

    for k in input_dict.keys():
        if k.startswith("encoder/agent") or k.startswith("decoder"):
            _extend_3rd_dim(k)
        if k.startswith("encoder/map"):
            _extend_map(k)
    # ===== Fill in some fake data END =====
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda()

    optimizer = model.configure_optimizers()['optimizer']
    for _ in tqdm.trange(10000):
        if in_eval:
            with torch.no_grad():
                out = model.forward(input_dict)
                loss = out["decoder/output_logit"].mean()
        else:
            out = model.forward(input_dict)

            optimizer.zero_grad()
            loss = out["decoder/output_logit"].mean()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    toy_test(bs=8, in_eval=False)
