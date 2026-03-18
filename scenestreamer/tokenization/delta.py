# from scenestreamer.tokenization.motion_tokenizers import BaseTokenizer

import argparse
import datetime
import os
import pathlib
from typing import *

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from scipy.cluster.vq import kmeans2

from scenestreamer.dataset.datamodule import SceneStreamerDataModule
from scenestreamer.models.layers import common_layers
# from scenestreamer.models.motionlm_lightning import MotionLMLightning
# from scenestreamer.tokenization.tokenizers import rotate
from scenestreamer.utils import global_config, cfg_from_yaml_file, REPO_ROOT, get_time_str
from scenestreamer.utils import lr_schedule
# from scenestreamer.tokenization.tokenizers import DeltaTokenizer, DeltaDeltaTokenizer
from scenestreamer.utils import rotate, unwrap, wrap_to_pi


def compute_3d_translation(data_dict, num_skipped_steps, offset=0):
    future_pos = data_dict["encoder/agent_position"]
    future_heading = data_dict["encoder/agent_heading"]
    future_valid_mask = data_dict["encoder/agent_valid_mask"]

    assert offset < num_skipped_steps

    future_pos = future_pos[:, offset:]
    future_heading = future_heading[:, offset:]
    future_valid_mask = future_valid_mask[:, offset:]

    current_pos = future_pos[:, 0]
    current_heading = future_heading[:, 0]
    current_valid_mask = future_valid_mask[:, 0]

    # T_action = future_pos.shape[1]

    B, T, N, _ = future_pos.shape

    # T_action = T

    # future_pos_sliced = future_pos[:, num_skipped_steps - 1::self.num_skipped_steps]
    # assert future_pos_sliced.shape[1] == T_action
    #
    # future_heading_sliced = future_heading[:, self.num_skipped_steps - 1::self.num_skipped_steps]
    # assert future_heading_sliced.shape[1] == T_action
    #
    # future_valid_mask = future_valid_mask[:, self.num_skipped_steps - 1::self.num_skipped_steps]
    # assert future_valid_mask.shape[1] == T_action

    reconstructed_pos = current_pos[..., :2].clone().reshape(B, 1, N, 2)
    reconstructed_heading = current_heading.clone().reshape(B, 1, N)
    reconstructed_valid_mask = current_valid_mask.clone().reshape(B, 1, N)

    target_action = []
    target_action_valid_mask = []

    delta_heading = []

    # reconstruction_error = []  # For stats

    current_t = 0

    while True:
        current_t += num_skipped_steps

        if current_t + num_skipped_steps > T:
            break

        # Real position at this step:
        real_pos = future_pos[:, current_t:current_t + num_skipped_steps, ..., :2]  # (1, N, 2)
        real_heading = future_heading[:, current_t:current_t + num_skipped_steps]  # (1, N, 2)

        # Update valid mask
        real_valid_mask = future_valid_mask[:, current_t:current_t + num_skipped_steps]
        real_valid_mask = real_valid_mask.all(dim=1, keepdims=True)
        reconstructed_valid_mask = torch.logical_and(reconstructed_valid_mask, real_valid_mask)
        assert reconstructed_valid_mask.shape == (B, 1, N)
        target_action_valid_mask.append(reconstructed_valid_mask)

        abs_delta = real_pos - reconstructed_pos
        y_axis_in_relative_coord = reconstructed_heading.repeat(1, num_skipped_steps, 1)
        x_axis_in_relative_coord = y_axis_in_relative_coord - np.pi / 2
        candidate_pos = rotate(abs_delta[..., 0], abs_delta[..., 1], -x_axis_in_relative_coord)

        target_action.append(candidate_pos)
        delta_heading.append(wrap_to_pi(real_heading - reconstructed_heading))

        new_reconstructed_pos = real_pos[:, -1:]
        reconstructed_pos = new_reconstructed_pos
        reconstructed_heading = real_heading[:, -1:]

    target_actions = torch.stack(target_action, dim=1)  # (B, T_skipped, N)
    delta_heading = torch.stack(delta_heading, dim=1)  # (B, T_skipped, N)

    deltas = torch.concat([target_actions, delta_heading[..., None]], dim=-1)
    deltas = deltas.swapaxes(-3, -2)

    assert deltas.ndim == 5

    target_action_valid_mask = torch.concatenate(target_action_valid_mask, dim=1)  # (B, T_skipped, N)

    return deltas, target_action_valid_mask


class VectorQuantizer(nn.Module):
    """
    PZH: From huggingface


    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float = 0.25,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        # legacy: bool = True,
        legacy: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.used: torch.Tensor
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.register_buffer('data_initialized', torch.zeros(1))

    def remap_to_used(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.LongTensor) -> torch.LongTensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z: torch.FloatTensor, disable=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, Tuple]:
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # PZH: https://github.com/karpathy/deep-vector-quantization/blob/c3c026a1ccea369bc892ad6dde5e6d6cd5a508a4/dvq/model/quantize.py
        # DeepMind def does not do this but I find I have to... ;\
        if self.training and self.data_initialized.item() == 0:
            print('running kmeans!!')  # data driven initialization for the embeddings
            rp = torch.randperm(z_flattened.size(0))
            kd = kmeans2(z_flattened[rp[:20000]].data.cpu().numpy(), self.n_e, minit='points')
            self.embedding.weight.data.copy_(torch.from_numpy(kd[0]))
            self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)
        else:
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q: torch.FloatTensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        if disable:
            return z, loss, (perplexity, min_encodings, min_encoding_indices)
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
        # return z, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.LongTensor, shape: Tuple[int, ...]) -> torch.FloatTensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q: torch.FloatTensor = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class RelationEncoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2):  # , num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.proj = common_layers.build_mlps(
            c_in=15,
            mlp_channels=[d_model] * num_layers,
            ret_before_act=True,
        )

    def forward(self, diff, mask, batch_dict):
        B, T, N, _ = diff.shape
        x = diff[mask]
        x = self.proj(x)
        x = unwrap(x, mask)
        return x, mask


class RelationDecoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2):
        super(RelationDecoder, self).__init__()
        self.d_model = d_model
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model] * num_layers + [15], ret_before_act=True
        )

    def forward(self, latent, mask, batch_dict):
        B, T, N, _ = latent.shape
        x = latent[mask]
        x = unwrap(self.prediction_head(x), mask)
        return x


class DeltaVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = 512
        self.enc = RelationEncoder(num_layers=4, d_model=d_model)
        self.dec = RelationDecoder(num_layers=4, d_model=d_model)
        self.quantizer = VectorQuantizer(1024, d_model)
        self.save_hyperparameters()

    def forward(self, batch_dict):
        with torch.no_grad():
            data, mask = compute_3d_translation(
                batch_dict, num_skipped_steps=5, offset=np.random.randint(0, 5)
            )  # , num_samples=None)
            data = data.flatten(-2, -1)
        latent, mask = self.enc(data, mask, batch_dict)
        z, quant_loss, (perplexity, min_encodings, min_encoding_indices) = self.quantizer(latent, disable=False)
        # emask = get_mask(mask)
        # count = emask.sum(-1, keepdims=True)
        # count = torch.masked_fill(count, count == 0, 1)
        # target = (data * emask[..., None]).sum(-2) / count
        return {
            "output": self.dec(z, mask=mask, batch_dict=batch_dict),
            "target": data,
            # "rel_matrix": data,
            "quant_loss": quant_loss,
            # "dist": posterior,
            "data": batch_dict,
            "valid_mask": mask,
            "quant_idxs": min_encoding_indices,
        }

    def get_loss(self, data_dict):
        output_logit = data_dict["output"]

        # target_action = data_dict["target"]
        target_action = data_dict["target"]
        mask = data_dict["valid_mask"]  # (B, N)

        # Masking
        output_logit = output_logit[mask]
        target_action = target_action[mask]

        mse = nn.functional.mse_loss(input=output_logit, target=target_action)
        loss = (mse * 1 + data_dict["quant_loss"] * 10)

        # output_logit_scaled = output_logit.clone()
        # target_action_scaled = target_action.clone()
        #
        # recon_rel_matrix = pairwise_relative_diff(data_dict["output"])
        # rel_matrix = data_dict["rel_matrix"]
        # emask = get_mask(mask)
        # # recon_loss1 = nn.functional.l1_loss(input=recon_rel_matrix[emask], target=rel_matrix[emask])
        # recon_loss2 = nn.functional.l1_loss(input=-recon_rel_matrix[emask], target=rel_matrix[emask])

        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        # encodings = F.one_hot(data_dict["quant_idxs"][data_dict["valid_mask"].flatten()], self.reltok.quantizer.n_e).float().reshape(-1, self.reltok.quantizer.n_e)
        # flat_mask = get_mask(data_dict["valid_mask"]).flatten()
        flat_mask = mask.flatten()
        encodings = F.one_hot(data_dict["quant_idxs"][flat_mask],
                              self.quantizer.n_e).float().reshape(-1, self.quantizer.n_e)

        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        # self.log('val_perplexity', perplexity, prog_bar=True)
        # self.log('val_cluster_use', cluster_use, prog_bar=True)

        # scaled_mse = nn.functional.mse_loss(input=output_logit_scaled, target=target_action_scaled)
        # scaled_norm = (output_logit_scaled[..., :1] - target_action_scaled[..., :1]).norm(dim=-1).mean()

        loss_stat = {
            # "recon/loss1": recon_loss1,
            # "recon/loss2": recon_loss2,
            "loss/total_loss": loss,
            "loss/mse": mse,
            "mse": mse,
            "perplexity": perplexity,
            "cluster_use": cluster_use,
            # "scaled_mse": scaled_mse,
            # "scaled_norm": scaled_norm,
            "loss/quant_loss": data_dict["quant_loss"],  # ["codebook_loss"],
            # "loss/commitment_loss": data_dict["quant_loss"]["commitment_loss"],
            "output/output_mean": output_logit.mean(),
            "output/output_max": output_logit.max(),
            "output/output_min": output_logit.min(),
            "output/target_mean": target_action.mean(),
            "output/target_max": target_action.max(),
            "output/target_min": target_action.min(),
            "quant/quant_idxs_mean": data_dict["quant_idxs"][flat_mask].float().mean(),
            "quant/quant_idxs_max": data_dict["quant_idxs"][flat_mask].float().max(),
            "quant/quant_idxs_min": data_dict["quant_idxs"][flat_mask].float().min(),
        }
        try:
            loss_stat["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass
        return loss, loss_stat

    def training_step(self, data_dict, batch_idx):
        data_dict = self(data_dict)
        loss, loss_stat = self.get_loss(data_dict)
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in loss_stat.items()},
            batch_size=data_dict["data"]["encoder/agent_feature"].shape[0],
            # on_epoch=True,
            prog_bar=True,
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def configure_optimizers(self):
        """Required by Lightning."""
        opt_cfg = self.config.OPTIMIZATION
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.get("LR"),
            weight_decay=opt_cfg.get('WEIGHT_DECAY', 0),
            betas=(0.9, 0.95),
            eps=1e-5
        )
        scheduler = lr_schedule.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            # num_warmup_steps=opt_cfg.WARMUP_STEPS,
            # num_training_steps=opt_cfg.TRAINING_STEPS,
            num_warmup_steps=200,  # TODO
            num_training_steps=opt_cfg.TRAINING_STEPS,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }


if __name__ == '__main__':

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
    parser.add_argument('--batch_size', type=int, default=20, required=False, help='Batch size for training.')
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

    model = DeltaVAE(config=config)

    if args.eval:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
