"""
Modified from https://github.com/IDEA-opensource/DAB-DETR/blob/main/models/DAB_DETR/transformer.py

TODO: Why this decoder layer does not has self-attention? It's pretty weird!!!!
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .multi_head_attention_local import MultiheadAttentionLocal


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class RelativePE(nn.Module):
    """
    Credit: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L122
    """
    def __init__(self, nhead):
        super().__init__()
        self.relative_position_bias_table = nn.Parameter(torch.zeros(401 * 401, nhead))  # 2*Wh-1 * 2*Ww-1, nH
        self.invalid = nn.Parameter(torch.zeros(nhead))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        torch.nn.init.trunc_normal_(self.invalid, std=0.02)

    def forward(self, pos, index_pair):
        assert pos.ndim == 3
        shape = index_pair.shape
        index_pair = index_pair.clone()
        mask = index_pair == -1
        index_pair[mask] = 0
        index_pair = index_pair.unsqueeze(-1).repeat(1, 1, 2)
        pos = torch.gather(pos, index=index_pair.long(), dim=1)
        ind = torch.floor(pos).clamp(-200, 200).int() + 200
        ind = ind[..., 0] + ind[..., 1] * 401
        assert ind.max() < 401 * 401
        ret = self.relative_position_bias_table[ind]
        ret = ret.reshape(*shape, ret.shape[-1])
        ret[mask] = self.invalid
        return ret


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        relative_pe,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        keep_query_pos=False,
        rm_self_attn_decoder=False,
        use_local_attn=False,
        is_first=False,
    ):
        super().__init__()
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)

        if is_first:
            self.sa_qpos_proj = nn.Linear(2, d_model)
            self.ca_qpos_proj = nn.Linear(d_model, d_model)

        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.use_local_attn = use_local_attn

        self.cross_attn = MultiheadAttentionLocal(d_model, nhead, dropout=dropout, vdim=d_model, without_weight=True)

        self.nhead = nhead
        # self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos
        self.d_model = d_model
        self.relative_pe = relative_pe

        if relative_pe:
            self.pe = RelativePE(self.nhead)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        *,
        tgt,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        memory,
        memory_pos_emb,
        memory_pos,
        is_first=False,
        key_batch_cnt=None,
        index_pair=None,
        index_pair_batch=None
    ):
        assert index_pair_batch.max() + 1 == key_batch_cnt.shape[0]

        cross_query = self.ca_qcontent_proj(tgt)

        k_content_valid = self.ca_kcontent_proj(memory)
        cross_key = k_content_valid

        valid_pos = memory_pos_emb
        k_pos_valid = self.ca_kpos_proj(valid_pos)
        cross_key_position = k_pos_valid

        v_valid = self.ca_v_proj(memory)
        cross_value = v_valid

        # TODO: I remove the query pos here. Double check.
        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        # if is_first or self.keep_query_pos:
        #     cross_query_position = self.ca_qpos_proj(self.sa_qpos_proj(query_pos))
        #     cross_query = cross_query + cross_query_position
        #     cross_key = cross_key + cross_key_position

        assert self.use_local_attn
        num_q_all, n_model = cross_query.shape

        cross_query = cross_query.view(num_q_all, self.nhead, n_model // self.nhead)

        # TODO: Query PE is removed now. Double check.
        # query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # query_sine_embed = query_sine_embed.view(num_q_all, self.nhead, n_model // self.nhead)
        # cross_query += query_sine_embed

        cross_query = cross_query.view(num_q_all, n_model)

        num_valid_key = cross_key.shape[0]
        cross_key = cross_key.view(num_valid_key, self.nhead, n_model // self.nhead)
        cross_key_position = cross_key_position.view(num_valid_key, self.nhead, n_model // self.nhead)
        cross_key += cross_key_position
        cross_key = cross_key.view(num_valid_key, n_model)

        # TODO: Relative PE is not working.
        # [num Q, num K, 2]
        # if self.relative_pe:
        #     raise ValueError()
        #     relative_pos = (memory_pos.unsqueeze(0) - query_pos.unsqueeze(1))
        #     relative_pe = self.pe(relative_pos, index_pair)
        # else:
        relative_pe = None

        tgt2 = self.cross_attn(
            query=cross_query,  # [num valid objects, 2 * d_model]
            key=cross_key,
            value=cross_value,
            index_pair=index_pair,
            query_batch_cnt=key_batch_cnt,
            key_batch_cnt=key_batch_cnt,
            index_pair_batch=index_pair_batch,
            attn_mask=None,
            relative_atten_weights=relative_pe,
            vdim=n_model
        )[0]

        # ========== End of Cross-Attention =============
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
