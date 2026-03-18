# from scenestreamer.tokenization.motion_tokenizers import BaseTokenizer

from typing import *

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
from torch.nn.modules.transformer import TransformerEncoderLayer as NativeTransformerEncoderLayer

from scenestreamer.models.layers import common_layers
from scenestreamer.utils import lr_schedule, wrap_to_pi

# from scenestreamer.tokenization.tokenizers import DeltaTokenizer, DeltaDeltaTokenizer

RELATION_DIM = 2


def masked_average(tensor, mask, dim):
    """
    Compute the average of tensor along the specified dimension, ignoring masked elements.
    """
    assert tensor.shape == mask.shape
    count = mask.sum(dim=dim)
    count = torch.max(count, torch.ones_like(count))
    return (tensor * mask).sum(dim=dim) / count


def get_mask(mask):
    """
    input mask is in shape (B, N), we need to prepare a pairwise mask in shape (B, N, N).
    It's not correct to naively expand the mask. We need to maintain the symmetry of the mask.
    """
    B, N = mask.shape
    mask = mask.unsqueeze(1).expand(B, N, N)
    mask = mask & mask.transpose(1, 2)
    return mask


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1, ) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def pairwise_relative_diff(positions):
    """
    Compute pairwise relative diffs for a batch of objects.
    For the ouput [b, i, j, :], it means the relative differences of [b, j] - [b, i],
    which is the pos of j in i's coordinate system.

    Parameters:
    - positions: A PyTorch tensor of shape (B, N, 2)

    Returns:
    - A PyTorch tensor of shape (B, N, N, 2) containing pairwise relative positions.
    """

    # Expand dimensions to get tensors of shapes (B, N, 1, ...) and (B, 1, N, ...)
    positions_expanded_a = positions.unsqueeze(2)  # Shape: (B, N, 1, ...)
    positions_expanded_b = positions.unsqueeze(1)  # Shape: (B, 1, N, ...)

    # Compute the pairwise relative positions by subtraction
    relative_positions = positions_expanded_b - positions_expanded_a  # Shape: (B, N, N, ...)

    return relative_positions


#
# def prepro(data, num_samples=1024):
#     B, T, N, _ = data["encoder/agent_position"].shape
#
#     # pos = data["encoder/agent_position"][..., :2].reshape(B, T * N, -1)
#     # vel = data["encoder/agent_velocity"].reshape(B, T * N, -1)
#     # head = data["encoder/agent_heading"].reshape(B, T * N, -1)
#     # time = torch.arange(T).reshape(1, T, 1).expand(B, T, N).to(pos.device).reshape(B, T * N, -1)
#     #
#     # mask = data["encoder/agent_valid_mask"].reshape(B, T * N)
#
#     T = 1
#     pos = data["encoder/agent_position"][..., :2][:, 10].reshape(B, T * N, -1)
#     vel = data["encoder/agent_velocity"][:, 10].reshape(B, T * N, -1)
#     head = data["encoder/agent_heading"][:, 10].reshape(B, T * N, -1)
#     time = torch.arange(1).reshape(1, 1, 1).expand(B, T, N).to(pos.device).reshape(B, T * N, -1)
#     mask = data["encoder/agent_valid_mask"][:, 10].reshape(B, T * N)
#
#     if num_samples is not None:
#         indices = torch.randint(high=T * N, size=(B, num_samples, 1)).to(pos.device)  # (B, 1024, 1)
#         pos = torch.gather(pos, index=indices.expand(B, num_samples, 2), dim=1)
#         vel = torch.gather(vel, index=indices.expand(B, num_samples, 2), dim=1)
#         head = torch.gather(head, index=indices.expand(B, num_samples, 1), dim=1)
#         time = torch.gather(time, index=indices.expand(B, num_samples, 1), dim=1)
#         mask = torch.gather(mask, index=indices.reshape(B, num_samples), dim=1)
#
#     # compute pairwise relative position: (B, N, N, D)
#     rel_pos = pairwise_relative_diff(pos)
#     rel_vel = pairwise_relative_diff(vel)
#     rel_head = wrap_to_pi(pairwise_relative_diff(head))
#     rel_time = pairwise_relative_diff(time)
#
#     # rotated to local coordinate
#
#     num_selected = head.shape[1]
#
#     # i's local coordinate's y-axis (the heading) in the global coordinate
#     i_local_y_wrt_global = head.reshape(B, -1, 1).expand(B, num_selected, num_selected)
#
#     i_local_x_wrt_global = i_local_y_wrt_global - np.pi / 2
#
#     # rotated_pos = rel_pos
#     rotated_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x_wrt_global)
#
#     rotated_vel = rotate(rel_vel[..., 0], rel_vel[..., 1], angle=-i_local_x_wrt_global)
#
#     relation_matrix = torch.concatenate([
#         rotated_pos,
#         rotated_vel,
#         rel_head,
#         rel_time
#     ], dim=-1)
#
#     relation_matrix[..., 0] /= 400
#     relation_matrix[..., 1] /= 400
#     relation_matrix[..., 2] /= 25
#     relation_matrix[..., 3] /= 25
#     relation_matrix[..., 4] /= 3.1415
#     relation_matrix[..., 5] /= 90
#
#     return relation_matrix, mask


def prepro(data, num_samples=1024):
    B, T, N, _ = data["encoder/agent_position"].shape

    # pos = data["encoder/agent_position"][..., :2].reshape(B, T * N, -1)
    # vel = data["encoder/agent_velocity"].reshape(B, T * N, -1)
    # head = data["encoder/agent_heading"].reshape(B, T * N, -1)
    # time = torch.arange(T).reshape(1, T, 1).expand(B, T, N).to(pos.device).reshape(B, T * N, -1)
    # mask = data["encoder/agent_valid_mask"].reshape(B, T * N)

    T = 1
    pos = data["encoder/agent_position"][..., :2][:, 10].reshape(B, T * N, -1)
    vel = data["encoder/agent_velocity"][:, 10].reshape(B, T * N, -1)
    head = data["encoder/agent_heading"][:, 10].reshape(B, T * N, -1)
    time = torch.arange(1).reshape(1, 1, 1).expand(B, T, N).to(pos.device).reshape(B, T * N, -1)
    mask = data["encoder/agent_valid_mask"][:, 10].reshape(B, T * N)

    if num_samples is not None:
        indices = torch.randint(high=T * N, size=(B, num_samples, 1)).to(pos.device)  # (B, 1024, 1)
        pos = torch.gather(pos, index=indices.expand(B, num_samples, 2), dim=1)
        vel = torch.gather(vel, index=indices.expand(B, num_samples, 2), dim=1)
        head = torch.gather(head, index=indices.expand(B, num_samples, 1), dim=1)
        time = torch.gather(time, index=indices.expand(B, num_samples, 1), dim=1)
        mask = torch.gather(mask, index=indices.reshape(B, num_samples), dim=1)

    # compute pairwise relative position: (B, N, N, D)
    rel_pos = pairwise_relative_diff(pos)
    rel_vel = pairwise_relative_diff(vel)
    rel_head = wrap_to_pi(pairwise_relative_diff(head))
    rel_time = pairwise_relative_diff(time)

    # rotated to local coordinate

    num_selected = head.shape[1]

    # i's local coordinate's y-axis (the heading) in the global coordinate
    i_local_y_wrt_global = head.reshape(B, -1, 1).expand(B, num_selected, num_selected)

    i_local_x_wrt_global = i_local_y_wrt_global - np.pi / 2

    rotated_pos = rel_pos
    # rotated_pos = rel_pos.norm(dim=-1, keepdim=True)
    # rotated_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x_wrt_global)

    rotated_vel = rel_vel.norm(dim=-1, keepdim=True)
    # rotated_vel = rotate(rel_vel[..., 0], rel_vel[..., 1], angle=-i_local_x_wrt_global)

    rel_dir = wrap_to_pi(torch.arctan2(rel_pos[..., 1], rel_pos[..., 0]) - i_local_x_wrt_global)

    relation_matrix = torch.concatenate(
        [
            rotated_pos,
            # rotated_vel,
            # rel_head,
            # rel_dir[..., None],
            # rel_time
        ],
        dim=-1
    )

    relation_matrix[..., 0] /= 400
    relation_matrix[..., 1] /= 400
    # relation_matrix[..., 1] /= 25
    # # relation_matrix[..., 3] /= 25
    # relation_matrix[..., 2] /= 3.1415
    # relation_matrix[..., 3] /= 3.1415
    # relation_matrix[..., 4] /= 90

    return relation_matrix, mask


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
    def __init__(self, d_model=128, num_layers=2):  #, num_heads=4):
        super().__init__()
        self.num_layers = 3
        nhead = 1
        self.d_model = d_model
        self_attn_layers = []
        for _ in range(self.num_layers):
            self_attn_layers.append(
                NativeTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=self.d_model * 4,
                    # dropout=dropout,
                    batch_first=True
                )
            )
        self.self_attn_layers = nn.ModuleList(self_attn_layers)

        # # TODO: Add config
        self.agent_pe = nn.Embedding(128, self.d_model)
        self.pre_proj = common_layers.build_mlps(
            c_in=RELATION_DIM,
            mlp_channels=[d_model],  # * (num_layers - 1) + [d_model],
            ret_before_act=True,
        )
        self.proj = common_layers.build_mlps(
            c_in=d_model * 2,
            mlp_channels=[d_model],  # * (num_layers - 1) + [d_model],
            ret_before_act=True,
        )
        self.out = common_layers.build_mlps(
            c_in=d_model,
            mlp_channels=[d_model, d_model],  # * (num_layers - 1) + [d_model],
            ret_before_act=True,
        )

    def forward(self, rel_matrix, mask, batch_dict):
        B, N, _, D = rel_matrix.shape
        x = self.pre_proj(rel_matrix.reshape(-1, D)).reshape(B, N, N, -1)
        # pooled = x.max(dim=-2)[0]
        pooled = masked_average(x, mask=get_mask(mask).reshape(B, N, N, 1).expand(B, N, N, self.d_model), dim=-2)
        x = torch.cat([x, pooled[:, :, None].repeat(1, 1, N, 1)], dim=-1)
        x = self.proj(x)
        # x = x.max(dim=-2)[0]
        x = masked_average(x, mask=get_mask(mask).reshape(B, N, N, 1).expand(B, N, N, self.d_model), dim=-2)
        x = self.out(x)

        # x = batch_dict["encoder/agent_position"][:, 10][..., :2]
        # B, N, D = x.shape
        # x = self.pre_proj(x.reshape(-1, RELATION_DIM)).reshape(B, -1, self.d_model)
        # x = self.out(x.reshape(-1, self.d_model)).reshape(B, N, -1)

        agent_pe = self.agent_pe(batch_dict["encoder/agent_id"])
        x += agent_pe
        for k in range(len(self.self_attn_layers)):
            x = self.self_attn_layers[k](src=x, src_key_padding_mask=~mask)
        return x, mask


class RelationDecoder(nn.Module):
    def __init__(self, d_model=128, num_layers=2):
        super(RelationDecoder, self).__init__()
        self.num_layers = 3
        nhead = 1
        self.d_model = d_model
        self_attn_layers = []
        for _ in range(self.num_layers):
            self_attn_layers.append(
                NativeTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=self.d_model * 4,
                    # dropout=dropout,
                    batch_first=True
                )
            )
        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model, RELATION_DIM], ret_before_act=True
        )
        self.agent_pe = nn.Embedding(128, self.d_model)

    def forward(self, latent, mask, batch_dict):

        B, N, D = latent.shape

        # FIXME: TODO:
        x = self.prediction_head(latent.reshape(-1, self.d_model)).reshape(B, N, -1)
        return x

        x = latent
        agent_pe = self.agent_pe(batch_dict["encoder/agent_id"])
        x += agent_pe
        for k in range(len(self.self_attn_layers)):
            x = self.self_attn_layers[k](src=x, src_key_padding_mask=~mask)
        x = self.prediction_head(x.reshape(-1, self.d_model)).reshape(B, N, -1)
        return x


class RelationDecoderDEPRECATED(nn.Module):
    def __init__(self, d_model=128, num_layers=2, num_heads=4):
        super().__init__()
        self.d_model = d_model
        nhead = 1
        self_attn_layers = []
        self.num_layers = 3
        for _ in range(self.num_layers):
            self_attn_layers.append(
                NativeTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=self.d_model * 4,
                    # dropout=dropout,
                    batch_first=True
                )
            )
        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.proj1 = common_layers.build_mlps(
            c_in=d_model,  # TODO: or 6?
            # mlp_channels=[d_out] * (num_layers - 1) + [6],
            mlp_channels=[d_model],
            ret_before_act=True,
        )
        self.proj2 = common_layers.build_mlps(
            c_in=d_model,  # TODO: or 6?
            # mlp_channels=[d_out] * (num_layers - 1) + [6],
            mlp_channels=[d_model],
            ret_before_act=True,
        )
        self.proj3 = common_layers.build_mlps(
            c_in=d_model,  # TODO: or 6?
            mlp_channels=[d_model] * (num_layers - 1) + [RELATION_DIM],
            # mlp_channels=[RELATION_DIM],
            ret_before_act=True,
        )
        self.norm1 = torch.nn.LayerNorm(d_model, eps=1e-5, bias=True)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=1e-5, bias=True)

    def forward(self, latent, mask=None):
        B, N1, D = latent.shape
        x = latent
        for k in range(len(self.self_attn_layers)):
            x = self.self_attn_layers[k](src=x, src_key_padding_mask=~mask)

        q = self.norm1(self.proj1(x.reshape(-1, D)).reshape(B, N1, -1) + x)
        k = self.norm2(self.proj2(x.reshape(-1, D)).reshape(B, N1, -1) + x)
        x = torch.einsum("bnd,bmd->bnmd", q, k)
        x = self.proj3(x.reshape(-1, D)).reshape(B, N1, N1, -1)

        # B, N1, _, D = latent.shape
        # x = latent
        # for k in range(len(self.self_attn_layers)):
        #     x = self.self_attn_layers[k](src=x, src_key_padding_mask=~mask)

        # x = latent
        # x = self.norm1(self.proj1(x.reshape(-1, D)).reshape(B, N1, N1, -1) + x)
        # k = self.norm2(self.proj2(x.reshape(-1, D)).reshape(B, N1, N1, -1) + x)
        # x = torch.einsum("bnmd,bnmd->bnd", q, k)
        # x = self.proj3(x.reshape(-1, D)).reshape(B, N1, N1, -1)

        return x


class Reltok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        d_model = 128
        self.enc = RelationEncoder(num_layers=3, d_model=d_model)

        # from scenestreamer.models.scene_encoder import SceneEncoder
        # self.config = config
        # self.scene_encoder = SceneEncoder(config=self.config)

        self.dec = RelationDecoder(num_layers=3, d_model=d_model)

        self.quantizer = VectorQuantizer(1024, d_model)

    def forward(self, batch_dict):
        data, mask = prepro(batch_dict, num_samples=None)
        # latent, agent_pe, new_mask = self.enc(data, mask, batch_dict)
        latent, mask = self.enc(data, mask, batch_dict)
        z, quant_loss, (perplexity, min_encodings, min_encoding_indices) = self.quantizer(latent, disable=False)

        emask = get_mask(mask)
        count = emask.sum(-1, keepdims=True)
        count = torch.masked_fill(count, count == 0, 1)
        target = (data * emask[..., None]).sum(-2) / count

        return {
            "output": self.dec(z, mask=mask, batch_dict=batch_dict),
            "target": target,
            "rel_matrix": data,
            "quant_loss": quant_loss,
            # "dist": posterior,
            "data": batch_dict,
            "valid_mask": mask,
            "quant_idxs": min_encoding_indices,
        }


class ReltokLightning(pl.LightningModule):
    def __init__(self, config):
        if "SEED" in config:
            pl.seed_everything(config.SEED)
            print("Everything is seeded to: ", config.SEED)
        super().__init__()
        self.config = config

        # self.enc = RelationEncoder()
        # self.dec = RelationDecoder()

        self.reltok = Reltok(config)

        self.save_hyperparameters()
        self.validation_outputs = []
        self.validation_ground_truth = []

    def forward(self, batch_dict):
        # data = prepro(batch_dict,num_samples=256)
        return self.reltok(batch_dict)

    def get_loss(self, data_dict):
        output_logit = data_dict["output"]

        target_action = data_dict["target"]
        mask = data_dict["valid_mask"]  # (B, N)

        # Masking
        output_logit = output_logit[mask]
        target_action = target_action[mask]

        mse = nn.functional.mse_loss(input=output_logit, target=target_action)
        loss = (mse * 1 + data_dict["quant_loss"] * 0.1)

        output_logit_scaled = output_logit.clone()
        output_logit_scaled[..., 0] *= 400
        output_logit_scaled[..., 1] *= 400
        # output_logit_scaled[..., 1] *= 25
        # # output_logit_scaled[..., 3] *= 25
        # output_logit_scaled[..., 2] *= 3.1415
        # output_logit_scaled[..., 3] *= 3.1415
        # output_logit_scaled[..., 4] *= 90

        target_action_scaled = target_action.clone()
        target_action_scaled[..., 0] *= 400
        target_action_scaled[..., 1] *= 400
        # target_action_scaled[..., 1] *= 25
        # # target_action_scaled[..., 3] *= 25
        # target_action_scaled[..., 2] *= 3.1415
        # target_action_scaled[..., 3] *= 3.1415
        # target_action_scaled[..., 4] *= 90

        recon_rel_matrix = pairwise_relative_diff(data_dict["output"])
        rel_matrix = data_dict["rel_matrix"]
        emask = get_mask(mask)
        # recon_loss1 = nn.functional.l1_loss(input=recon_rel_matrix[emask], target=rel_matrix[emask])
        recon_loss2 = nn.functional.l1_loss(input=-recon_rel_matrix[emask], target=rel_matrix[emask])

        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        # encodings = F.one_hot(data_dict["quant_idxs"][data_dict["valid_mask"].flatten()], self.reltok.quantizer.n_e).float().reshape(-1, self.reltok.quantizer.n_e)
        # flat_mask = get_mask(data_dict["valid_mask"]).flatten()
        flat_mask = data_dict["valid_mask"].flatten()
        encodings = F.one_hot(data_dict["quant_idxs"][flat_mask],
                              self.reltok.quantizer.n_e).float().reshape(-1, self.reltok.quantizer.n_e)

        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        # self.log('val_perplexity', perplexity, prog_bar=True)
        # self.log('val_cluster_use', cluster_use, prog_bar=True)

        scaled_mse = nn.functional.mse_loss(input=output_logit_scaled, target=target_action_scaled)
        scaled_norm = (output_logit_scaled[..., :1] - target_action_scaled[..., :1]).norm(dim=-1).mean()

        loss_stat = {
            # "recon/loss1": recon_loss1,
            "recon/loss2": recon_loss2,
            "loss/total_loss": loss,
            "loss/mse": mse,
            "mse": mse,
            "perplexity": perplexity,
            "cluster_use": cluster_use,
            "scaled_mse": scaled_mse,
            "scaled_norm": scaled_norm,
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
            self.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0), betas=(0.9, 0.95), eps=1e-5
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
    pass
