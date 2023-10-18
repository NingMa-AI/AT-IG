import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from visualizer import get_local

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.distances = torch.zeros((window_size, window_size)).to(self.device)
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
                
    @get_local('series')
    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape  # B=256, L=100, H=8, E=64, D=64, S=100
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)      # 0.125
        
        # vis_SCORES=torch.einsum("blhe,bshe->bhls", queries, keys)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)     # [256,100,8,64],[235,100,8,64]->[256,8,100,100]
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores       # [256,8,100,100]
        
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L    [256,8,100]
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L  [256,8,100,100]
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))  # [256,8,100,100]

        series = self.dropout(torch.softmax(attn, dim=-1))      # [256,8,100,100]

        # print("series", series)
        
        V = torch.einsum("bhls,bshd->blhd", series, values)     # [256,8,100,100],[256,100,8,64]->[256,100,8,64]

        if self.output_attention:
            # print("dddddddddddd",V.contiguous().shape, series.shape, prior.shape, sigma.shape)
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):    # q、k、v都是同一个x [256,100,512]
        B, L, _ = queries.shape     # B=256, L=100, S=100, H=8
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)      # [256,20,8,64]
        keys = self.key_projection(keys).view(B, S, H, -1)              # [256,20,8,64]
        values = self.value_projection(values).view(B, S, H, -1)        # [256,20,8,64]
        sigma = self.sigma_projection(x).view(B, L, H)                  # [256,20,8]

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )   # [256,20,8,64], 后三个都是[256,8,20,20]
        out = out.view(B, L, -1)    # [256,20,512]

        # return out, series, prior, sigma
        return self.out_projection(out), series, prior, sigma
