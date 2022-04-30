import math
import copy
from typing import Optional, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head_attention import *

class SFM(nn.Module):
    def __init__(self, sfm_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(sfm_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        key: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                key=key,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class SFMLayer(nn.Module):
    def __init__(
        self,
        args,
        d_model,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        ## self.self_attn = MultiHeadAttention(args, d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        key: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        query = key = self.with_pos_embed(src, pos)
        src2 = self.self_attn(query, key, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        ## src2 = self.self_attn(query, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        key: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # import pdb; pdb.set_trace();
        src2 = self.norm1(src)
        query = key = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(query, key, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        ## src2 = self.self_attn(query, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        key: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, key, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, key, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
