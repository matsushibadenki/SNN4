# matsushibadenki/snn4/snn_research/models/spiking_transformer.py

# [!!DEPRECATION WARNING!!]
# このファイルは旧バージョンの実装です。
# 現在、アクティブに開発・利用されているモデルは
# `snn_research.core.snn_core.py` 内の `SpikingTransformer` です。
# 新規のコードではそちらを利用してください。
#
# スパイキングトランスフォーマーモデル
# 概要: SpikeDrivenSelfAttentionを組み合わせて、完全なトランスフォーマーブロックとモデルを構築する。


import torch
import torch.nn as nn
from snn_research.bio_models.lif_neuron import BioLIFNeuron as LIFNeuron
from snn_research.models.attention import SpikeDrivenSelfAttention

class SpikingTransformerBlock(nn.Module):
    """単一のスパイキングトランスフォーマーブロック。"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = SpikeDrivenSelfAttention(embed_dim, num_heads)
        self.lif1 = LIFNeuron(n_neurons=embed_dim, neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0})
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            LIFNeuron(n_neurons=4 * embed_dim, neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0}),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        spikes = self.lif1(x.view(-1, x.size(-1))).view(x.size())
        return spikes

class SpikingTransformer(nn.Module):
    """完全なスパイキングトランスフォーマーモデル。"""
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size, max_len=512, **kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList(
            [SpikingTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding[:, :T, :]
        x = token_embed + pos_embed
        for layer in self.layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
