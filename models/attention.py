import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

# from linformer_pytorch import MHAttention, LinearAttentionHead, get_EF
# from linformer_pytorch.linformer_pytorch import gen_causal_mask

# from linformer import LinformerSelfAttention

# from performer_pytorch import CrossAttention

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

# class LinformerSelfAttention(nn.Module):
#     def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
#         super().__init__()
#         assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

#         self.seq_len = seq_len
#         self.k = k

#         self.heads = heads

#         dim_head = default(dim_head, dim // heads)
#         self.dim_head = dim_head

#         self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

#         kv_dim = dim_head if one_kv_head else (dim_head * heads)
#         self.to_k = nn.Linear(dim, kv_dim, bias = False)
#         self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

#         self.share_kv = share_kv
#         if not share_kv:
#             self.to_v = nn.Linear(dim, kv_dim, bias = False)
#             self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

#         self.dropout = nn.Dropout(dropout)
#         self.to_out = nn.Linear(dim_head * heads, dim)

#     def forward(self, queries, keys, values, context = None):
#         b, n, _, d_h, h, k = *values.shape, self.dim_head, self.heads, self.k

#         kv_len = n if context is None else context.shape[1]
#         assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

#         # queries = self.to_q(x)
#         queries = self.to_q(queries)

#         proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

#         # kv_input = x if context is None else context

#         keys = self.to_k(keys)
#         # values = self.to_v(kv_input) if not self.share_kv else keys
#         values = self.to_v(values) if not self.share_kv else values

#         kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

#         # project keys and values along the sequence length dimension to k

#         keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

#         # merge head into batch for queries and key / values

#         queries = queries.reshape(b, n, h, -1).transpose(1, 2)

#         merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
#         keys, values = map(merge_key_values, (keys, values))

#         # attention

#         dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
#         attn = dots.softmax(dim=-1)
#         attn = self.dropout(attn)
#         out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

#         # split heads
#         out = out.transpose(1, 2).reshape(b, n, -1)
#         return self.to_out(out)

# class LinearAttention(nn.Module):
#     def __init__(self, seq_len, channels, dim_k=128, nhead=8, dropout=0.2):
#         super(LinearAttention, self).__init__()

#         head_dim = channels // nhead
#         E = get_EF(seq_len, dim_k, "learnable", head_dim)

#         self.multihead_linear_attn = MHAttention(
#             input_size=seq_len,
#             channels=channels,
#             dim=head_dim,
#             dim_k=dim_k,
#             nhead=nhead,
#             dropout=dropout,
#             checkpoint_level="C0",
#             parameter_sharing="layerwise",
#             E_proj=E,
#             F_proj=E,
#             full_attention=False,
#             causal_mask=None,
#             w_o_intermediate_dim=None,
#         )

#     def forward(self, x):
#         x = rearrange(x, 'l b c -> b l c')
#         out = self.multihead_linear_attn(x)
#         out = rearrange(out, 'b l c -> l b c')
#         return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        out, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out

class ContextAwareAttention(nn.Module):
    def __init__(self, d_model, nhead=8, num_regions=196, dropout=0.1):
        super(ContextAwareAttention, self).__init__()

        self.num_regions=num_regions
        
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        
        # self.two_neighbor_mlp = nn.Linear(d_model, 1)
        # self.three_neighbor_mlp = nn.Linear(d_model, 1)

        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(1.))
        self.gamma = nn.Parameter(torch.tensor(1.))
        
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        
        rearrange_query = rearrange(query, 'l b c -> b l c')
        visual_query, text_query = rearrange_query[:, :self.num_regions, :], rearrange_query[:, self.num_regions:, :]
        
        padded_two_neighbor_query = F.pad(text_query, pad=(0, 0, 1, 1), mode='constant', value=0)
        padded_three_neighbor_query = F.pad(text_query, pad=(0, 0, 2, 2), mode='constant', value=0)
        
        # padded_two_neighbor_query = F.pad(rearrange_query, pad=(0, 0, 1, 1), mode='constant', value=0)
        # padded_three_neighbor_query = F.pad(rearrange_query, pad=(0, 0, 2, 2), mode='constant', value=0)
        
        # import pdb; pdb.set_trace()

        two_neighbor_query = torch.zeros_like(text_query)
        for i in range(1, padded_two_neighbor_query.shape[1] - 1):
            window_feat = padded_two_neighbor_query[:, i-1:i+2, :]
            anchor = padded_two_neighbor_query[:, i]
            anchor = torch.stack([anchor]*window_feat.shape[1], dim=1)
            weight = F.cosine_similarity(anchor, window_feat, dim=2)
            two_neighbor_query[:, i-1, :] = (weight[:, :, None]*window_feat).sum(dim=1)
            
        # two_neighbor_query = torch.zeros_like(rearrange_query)
        # for i in range(1, padded_two_neighbor_query.shape[1] - 1):
        #     window_feat = padded_two_neighbor_query[:, i-1:i+2, :]
        #     window_out = self.two_neighbor_mlp(window_feat)
        #     two_neighbor_query[:, i-1, :] = (window_feat * window_out).sum(dim=1)
        
        three_neighbor_query = torch.zeros_like(text_query)
        for i in range(2, padded_three_neighbor_query.shape[1] - 2):
            window_feat = padded_three_neighbor_query[:, i-2:i+3, :]
            anchor = padded_two_neighbor_query[:, i]
            anchor = torch.stack([anchor]*window_feat.shape[1], dim=1)
            weight = F.cosine_similarity(anchor, window_feat, dim=2)      
            three_neighbor_query[:, i-2, :] = (weight[:, :, None]*window_feat).sum(dim=1)
            
        # three_neighbor_query = torch.zeros_like(rearrange_query)
        # for i in range(2, padded_three_neighbor_query.shape[1] - 2):
        #     window_feat = padded_three_neighbor_query[:, i-2:i+3, :]
        #     window_out = self.three_neighbor_mlp(window_feat)
        #     three_neighbor_query[:, i-2, :] = (window_feat * window_out).sum(dim=1)
            
        two_neighbor_query = torch.cat([visual_query, two_neighbor_query], dim=1)
        two_neighbor_query = rearrange(two_neighbor_query, 'b l c -> l b c')
        
        three_neighbor_query = torch.cat([visual_query, three_neighbor_query], dim=1)
        three_neighbor_query = rearrange(three_neighbor_query, 'b l c -> l b c')
        
        out1, _ = self.multihead_attn1(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        out2, _ = self.multihead_attn2(two_neighbor_query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        out3, _ = self.multihead_attn3(three_neighbor_query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        out = self.alpha*out1 + self.beta*out2 + self.gamma*out3
        return out
    
# class LinearContextAttention(nn.Module):
#     def __init__(self, d_model, nhead=8, num_regions=196, phrase_len=20, dropout=0.1):
#         super(LinearContextAttention, self).__init__()

#         self.num_regions=num_regions
#         seq_len = num_regions+phrase_len
        
#         self.linear_attn1 = LinformerSelfAttention(dim=d_model, seq_len=seq_len, heads=nhead, k=32, one_kv_head=False, share_kv=False)
#         self.linear_attn2 = LinformerSelfAttention(dim=d_model, seq_len=seq_len, heads=nhead, k=32, one_kv_head=False, share_kv=False)
#         self.linear_attn3 = LinformerSelfAttention(dim=d_model, seq_len=seq_len, heads=nhead, k=32, one_kv_head=False, share_kv=False)
        
#         self.two_neighbor_mlp = nn.Linear(d_model, 1)
#         self.three_neighbor_mlp = nn.Linear(d_model, 1)

#         self.alpha = nn.Parameter(torch.tensor(.7))
#         self.beta = nn.Parameter(torch.tensor(.2))
#         self.gamma = nn.Parameter(torch.tensor(.1))
        
        
#     def forward(self, x):
        
#         rearrange_x = rearrange(x, 'l b c -> b l c')
#         # visual_query, text_query = rearrange_query[:, :self.num_regions, :], rearrange_query[:, self.num_regions:, :]
        
#         # padded_two_neighbor_query = F.pad(text_query, pad=(0, 0, 1, 1), mode='constant', value=0)
#         # padded_three_neighbor_query = F.pad(text_query, pad=(0, 0, 2, 2), mode='constant', value=0)
        
#         padded_two_neighbor_x = F.pad(rearrange_x, pad=(0, 0, 1, 1), mode='constant', value=0)
#         padded_three_neighbor_x = F.pad(rearrange_x, pad=(0, 0, 2, 2), mode='constant', value=0)
        
#         # two_neighbor_query = torch.zeros_like(text_query)
#         two_neighborx = torch.zeros_like(rearrange_x)
#         for i in range(1, padded_two_neighbor_x.shape[1] - 1):
#             window_feat = padded_two_neighbor_x[:, i-1:i+2, :]
#             window_out = self.two_neighbor_mlp(window_feat)
#             two_neighborx[:, i-1, :] = (window_feat * window_out).sum(dim=1)
        
#         # three_neighbor_query = torch.zeros_like(text_query)
#         three_neighbor_x = torch.zeros_like(rearrange_x)
#         for i in range(2, padded_three_neighbor_x.shape[1] - 2):
#             window_feat = padded_three_neighbor_x[:, i-2:i+3, :]
#             window_out = self.three_neighbor_mlp(window_feat)
#             three_neighbor_x[:, i-2, :] = (window_feat * window_out).sum(dim=1)
            
#         # two_neighbor_query = torch.cat([visual_query, two_neighbor_query], dim=1)
#         # two_neighbor_query = rearrange(two_neighbor_query, 'b l c -> l b c')
        
#         # three_neighbor_query = torch.cat([visual_query, three_neighbor_query], dim=1)
#         # three_neighbor_query = rearrange(three_neighbor_query, 'b l c -> l b c')
        
#         out1 = self.linear_attn1(rearrange_x)
#         out2 = self.linear_attn2(two_neighborx)
#         out3 = self.linear_attn3(three_neighbor_x)
        
#         out = self.alpha*out1 + self.beta*out2 + self.gamma*out3
#         out = rearrange(out, 'b l c -> l b c')
#         return out
    
# class MultiModalSelfAttention(nn.Module):
#     def __init__(self, d_model, nhead=8, num_regions=196, dropout=0.1):
#         super(MultiModalSelfAttention, self).__init__()
        
#         self.num_regions = num_regions
        
#         self.visual_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.textual_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
#         self.joint_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
#         self.visual_linear = nn.Linear(d_model, d_model)
#         self.textual_linear = nn.Linear(d_model, d_model)
#         self.joint_linear = nn.Linear(d_model, d_model)
        
#         self.visual_norm = nn.LayerNorm(d_model)
#         self.textual_norm = nn.LayerNorm(d_model)
#         self.joint_norm = nn.LayerNorm(d_model)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, query, key, value, key_padding_mask=None):
#         v_query, t_query = query[:self.num_regions], query[self.num_regions:]
#         v_key, t_key = key[:self.num_regions], key[self.num_regions:]
#         v_value, t_value = value[:self.num_regions], value[self.num_regions:]
        
#         v_mask, t_mask = key_padding_mask[:, :self.num_regions], key_padding_mask[:, self.num_regions:]
        
#         v_self = self.visual_self_attn(v_query, v_key, v_value, key_padding_mask=v_mask)[0]
#         v_value = v_value + self.dropout(v_self)
#         v_self = self.visual_norm(v_self)
#         v_self = self.dropout(F.relu(self.visual_linear(v_self)))
#         v_value = v_value + self.dropout(v_self)
        
#         t_self = self.textual_self_attn(t_query, t_key, t_value, key_padding_mask=t_mask)[0]
#         t_value = t_value + self.dropout(t_self)
#         t_self = self.textual_norm(t_self)
#         t_self = self.dropout(F.relu(self.textual_linear(t_self)))
#         t_value = t_value + self.dropout(t_self)
        
#         joint_value = torch.cat([v_value, t_value], dim=0)
#         joint_self = self.joint_self_attn(joint_value, joint_value, joint_value, key_padding_mask=key_padding_mask)[0]
#         # joint_value = joint_value + self.dropout(joint_self)
#         # joint_self = self.joint_norm(joint_self)
#         # joint_self = self.dropout(F.relu(self.joint_linear(joint_self)))
#         # joint_value = joint_value + self.dropout(joint_self)
        
#         return joint_self
        
    
# class CrossModalAttention(nn.Module):
#     def __init__(self, d_model, nhead=8, num_regions=196, dropout=0.1):
#         super(CrossModalAttention, self).__init__()
        
#         self.num_regions = num_regions
        
#         self.cross_attn = CrossAttention(dim = d_model, heads = nhead)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         x = rearrange(x, 'l b c -> b l c')
#         text_context = x[:, self.num_regions:]
        
#         out = self.cross_attn(x, context=text_context)
#         out = rearrange(out, 'b l c -> l b c')
#         return out
