# ファイルパス: snn_research/core/mamba_core.py
# (新規作成)
# Title: Spiking-MAMBAモデル コア実装
# Description:
# - MAMBAアーキテクチャをSNNのパラダイムに適応させた実装。
# - 状態空間モデル(SSM)と選択的スキャンメカニズムをスパイクニューロンを用いて構築する。
# - 長文脈処理における線形計算量を実現し、Transformerの計算量問題を解決することを目指す。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
from omegaconf import DictConfig
import math

from .neurons import AdaptiveLIFNeuron
from .snn_core import BaseModel, SNNLayerNorm

class SpikingMambaBlock(nn.Module):
    """
    Spiking-MAMBAの基本ブロック。
    選択的SSMをスパイクベースで実装。
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, neuron_config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # 入力から内層への線形射影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 1D畳み込み層と活性化ニューロン
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.lif_conv = AdaptiveLIFNeuron(features=self.d_inner, **neuron_config)

        # 選択的SSMのパラメータを学習するための線形層
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # 状態遷移行列Aと射影行列B, Cを定義
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 出力層
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = SNNLayerNorm(d_model)
        self.lif_out = AdaptiveLIFNeuron(features=d_model, **neuron_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力テンソル (B, L, D)
        Returns:
            torch.Tensor: ブロックの出力テンソル (B, L, D)
        """
        B, L, D = x.shape
        
        # 1. 入力射影と分割
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_in, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. 畳み込みと非線形変換
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv_spikes, _ = self.lif_conv(x_conv.reshape(B * L, -1))
        x_conv_spikes = x_conv_spikes.reshape(B, L, -1)
        
        # 3. 選択的SSM
        x_ssm_params = self.x_proj(x_conv_spikes)
        delta, B_param, C_param = x_ssm_params.split(split_size=[self.d_inner, self.d_state, self.d_state], dim=-1)
        
        # Δ (delta) を計算
        delta = F.softplus(self.dt_proj(delta))

        # 離散化された状態遷移行列 A_bar と 入力行列 B_bar を計算
        A = -torch.exp(self.A_log.float())
        A_bar = torch.exp(A * delta.unsqueeze(-1))
        B_bar = B_param.unsqueeze(-1) * delta.unsqueeze(-1)

        # 選択的スキャン (逐次処理)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_scan = []
        for i in range(L):
            h = A_bar[:, i] * h + B_bar[:, i] * x_conv_spikes[:, i].unsqueeze(-1)
            y = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            y_scan.append(y)
        
        y = torch.stack(y_scan, dim=1) + x_conv_spikes * self.D

        # 4. 残差結合と出力
        y = y * F.silu(res) # ゲート機構
        
        out = self.norm(x + self.out_proj(y))
        out_spikes, _ = self.lif_out(out.reshape(B * L, -1))
        
        return out_spikes.reshape(B, L, -1)

class SpikingMamba(BaseModel):
    """
    SpikingMambaBlockを複数層重ねた、完全なSpiking-MAMBAモデル。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, d_conv: int, expand: int, num_layers: int, time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SpikingMambaBlock(d_model, d_state, d_conv, expand, neuron_config)
            for _ in range(num_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        
        # SNNの時間ステップループ
        for _ in range(self.time_steps):
            for layer in self.layers:
                x = layer(x)

        x = self.norm(x)
        logits = self.output_projection(x)
        
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (L * self.time_steps * B) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        
        # 互換性のためにmemを返す
        mem = torch.tensor(0.0, device=input_ids.device) 
        
        return logits, avg_spikes, mem