import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter

__all__ = ['MultiHeadAttention']

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 args,
                 embed_dim,
                 num_heads,
                 dropout=0,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param embed_dim: Size of each input sample.
        :param num_heads: Number of heads.
        :param dropout: Probability of omiting a value in tensor
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        
        self.feature_dim = args.feature_dim
        self.phrase_len = args.phrase_len
        
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.dummy_param = nn.Parameter(torch.empty(0))

        ## H = W = self.feature_dim
        ## T = self.phrase_len
        ## kernel_matrix = torch.zeros(H*W, H*W, 3, 3)
        ## kernel_matrix[:, :, 1, 1] = 1
        ## kernel_matrix = torch.rand(H*W + T, H*W + T, 5, 5)
        ## self.kernel = nn.Parameter(kernel_matrix, requires_grad=True)
        ## self.kernel = nn.Parameter(torch.ones(H*W + T, H*W + T, 5, 5), requires_grad=False)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)        

    def forward(self, query, key_padding_mask=None):

        tgt_len, bsz, embed_dim = query.size()

        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5

        query, key, value = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        query = query * scaling

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        query = query.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        key = key.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        value = value.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        src_len = key.size(1)
        
        attn_output_weights = torch.bmm(query, key.transpose(1, 2))
        
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
            
        ## import pdb; pdb.set_trace();
        ## attn_output_weights = torch.cat([F.softmax(attn_output_weights[:, :, :196], dim=-1), F.softmax(attn_output_weights[:, :, 196:], dim=-1)], dim=-1)
        
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
 
        ## H = W = self.feature_dim
        ## T = self.phrase_len

        ## vv_wts, vt_wts = attn_output_weights[:, :H*W, :H*W], attn_output_weights[:, :H*W, H*W:]
        ## updated_vt_wts = torch.bmm(vv_wts, vt_wts)

        ## tt_wts, tv_wts = attn_output_weights[:, H*W:, H*W:], attn_output_weights[:, H*W:, :H*W]
        ## updated_tv_wts = torch.bmm(tt_wts, tv_wts)

        ## tv_mask = torch.zeros(attn_output_weights.shape, device=self.dummy_param.device, dtype=torch.bool)
        ## tv_mask[:, H*W:, :H*W] = True

        ## vt_mask = torch.zeros(attn_output_weights.shape, device=self.dummy_param.device, dtype=torch.bool)
        ## vt_mask[:, :H*W, H*W:] = True
       

        ## attn_output_weights = attn_output_weights.masked_scatter(tv_mask, updated_tv_wts)
        ## attn_output_weights = attn_output_weights.masked_scatter(vt_mask, updated_vt_wts)

        ## attn_output_weights = structured_softmax(attn_output_weights, self.kernel, H, W, T)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, value)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


def structured_softmax(attn_output_weights, kernel, H, W, T):
    
    ## H = W = self.feature_dim
    ## T = self.phrase_len
    
    exp_weights = torch.exp(attn_output_weights)
    denominator = exp_weights.sum(dim=-1, keepdim=True)
    
    visual_weights, textual_weights = exp_weights[:,:, :H*W], exp_weights[:,:,H*W:]
   
    ## vv_weights, tv_weights = visual_weights[:, :H*W, :], visual_weights[:, H*W:, :]
    ## vv_weights = vv_weights.view(-1, H*W, H, W)
    visual_weights = visual_weights.view(-1, H*W + T, H, W)
    
    ## vv_weights = F.conv2d(vv_weights, kernel, padding=2)
    visual_weights = F.conv2d(visual_weights, kernel, padding=2)
    
    ## vv_weights = vv_weights.view(-1, H*W, H*W)
    ## visual_weights = torch.cat([vv_weights, tv_weights], dim=1)
    visual_weights = visual_weights.view(-1, H*W + T, H*W)
    
    joint_weights = torch.cat([visual_weights, textual_weights], dim=-1)
    
    return joint_weights/denominator
