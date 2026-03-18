import argparse
import datetime
import os
import pathlib

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
# from scenestreamer.models.motionlm_lightning import MotionLMLightning
from scenestreamer.tokenization.reltok import ReltokLightning
from scenestreamer.utils import global_config, cfg_from_yaml_file, REPO_ROOT, get_time_str


def main():
    parser = argparse.ArgumentParser(description='arg parser')

    # Experiment
    parser.add_argument(
        '--cfg_file',
        type=str,
        default="cfgs/motion_debug.yaml",
        help='The config file path, relative to the repo root.'
    )
    parser.add_argument('--exp_name', type=str, default='train_reltok', help='Experiment name.')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to pretrained checkpoint.')
    parser.add_argument('--log_dir', type=str, default=None, help='Path to store all logs/ckpts/files.')
    parser.add_argument('--debug', action='store_true', default=False, help='Whether to quickly set debug config.')
    parser.add_argument('--eval', action='store_true', default=False, help='Whether to evaluate the model.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether to use wandb logging.')

    # Training
    parser.add_argument('--batch_size', type=int, default=25, required=False, help='Batch size for training.')
    parser.add_argument(
        '--prefetch_factor', type=int, default=2, required=False, help='Datamodule prefetch factor for training.'
    )
    parser.add_argument(
        '--limit_train_batches',
        type=int,
        default=-1,
        required=False,
        help='Number of validation steps in each iteration.'
    )
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader.')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='Number of epochs for training.')

    # Validation
    parser.add_argument('--val_batch_size', type=int, default=6, required=False, help='Batch size for validation.')
    parser.add_argument(
        '--val_num_workers', type=int, default=4, help='Number of workers for dataloader in validation.'
    )
    parser.add_argument(
        '--num_sanity_val_steps',
        type=int,
        default=20,
        required=False,
        help='Number of validation steps before first training epoch.'
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=-1,
        required=False,
        help='Number of validation steps in each iteration. Default to whole validation dataset.'
    )

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    print("Everything is seeded to: ", args.seed)

    # Set up config
    cfg_file = REPO_ROOT / args.cfg_file
    config = cfg_from_yaml_file(cfg_file, global_config)
    exp_name = args.exp_name
    max_epochs = args.epochs  #or config.OPTIMIZATION.NUM_EPOCHS
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    val_num_workers = args.val_num_workers
    log_dir = args.log_dir or None
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)

    # Setup wandb logger
    trial_id = get_time_str()
    name = "{}_{}".format(exp_name, trial_id)
    if log_dir:
        save_dir = log_dir / "lightning_logs"
    else:
        save_dir = os.path.join(REPO_ROOT, "lightning_logs")
    if args.wandb and not args.eval:
        with open(os.path.abspath(os.path.expanduser("~/wandb_api_key_file.txt")), "rt") as fp:
            api_key = fp.readline().strip()
        wandb.login(key=api_key)
        logger = WandbLogger(
            name=name,
            save_dir=save_dir,
            id=name,
            project="scenestreamer",
            log_model=True,
            group=exp_name,
        )
    else:
        logger = TensorBoardLogger(save_dir=save_dir, name=exp_name)

    # Set up trainer arguments
    callbacks = [
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            monitor="monitoring_step",
            every_n_epochs=1,
            save_last=True,
            auto_insert_metric_name=True,
            mode="max",
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            train_time_interval=datetime.timedelta(minutes=30),
            auto_insert_metric_name=True,
            save_on_train_epoch_end=True,
            every_n_train_steps=None,
            every_n_epochs=None,
        )
    ]
    trainer_kwargs = dict(
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else None,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else None,
        gradient_clip_val=config.OPTIMIZATION.GRAD_NORM_CLIP,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=2,
        # strategy='ddp_find_unused_parameters_true'
    )
    if args.debug:
        # from lightning.pytorch.profilers import PyTorchProfiler
        # profiler = PyTorchProfiler(filename="profile")
        trainer_kwargs.update(
            num_sanity_val_steps=0,
            # profiler=profiler,
            detect_anomaly=True,
            limit_val_batches=2,
            limit_train_batches=2,
            log_every_n_steps=1,
        )
        num_workers = 0
        val_num_workers = 0
    datamodule = SceneStreamerDataModule(
        config,
        train_batch_size=batch_size,
        train_num_workers=num_workers,
        train_prefetch_factor=args.prefetch_factor,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        val_prefetch_factor=args.prefetch_factor,
    )
    if torch.cuda.device_count() > 1:
        trainer_kwargs["strategy"] = 'ddp'
        # trainer_kwargs["strategy"] = 'ddp_find_unused_parameters_true'
    if log_dir:
        trainer_kwargs["default_root_dir"] = log_dir

    # Set up trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Set up model
    ckpt_path = args.ckpt
    if ckpt_path is not None:
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
        assert os.path.isfile(ckpt_path), ckpt_path
        assert ckpt_path.endswith(".ckpt"), ckpt_path
        print("==============================")
        print("Loading checkpoint: ", ckpt_path)
        print("==============================")

    model = ReltokLightning(config=config)

    if args.eval:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
