# matsushibadenki/snn4/snn_research/models/attention.py
# スパイク駆動自己注意メカニズム
# 概要: スパイクベースの計算に最適化された自己注意モジュールを定義する。
# BugFix: ファイル末尾の不正な閉じ括弧を削除し、mypyの構文エラーを修正。

import torch
import torch.nn as nn
from snn_research.bio_models.lif_neuron import BioLIFNeuron as LIFNeuron

class SpikeDrivenSelfAttention(nn.Module):
    """
    スパイク駆動の自己注意メカニズム。
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_lif = LIFNeuron()
        self.k_lif = LIFNeuron()
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_spikes, _ = self.q_lif(q)
        k_spikes, _ = self.k_lif(k)

        q_spikes = q_spikes.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_spikes = k_spikes.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q_spikes, k_spikes) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        output = self.out_proj(output)
        return output