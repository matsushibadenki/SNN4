# ファイルパス: snn_research/core/hrm_core.py
# (新規作成)
# Title: Spiking-HRMモデル コア実装
# Description:
# - 脳の階層的な情報処理を模倣する階層的循環記憶（HRM）モデルの実装。
# - 各層が異なる時間スケール（クロック）で動作し、下位層からの情報を抽象化し、
#   上位層からの予測（トップダウン信号）を受け取る。
# - プロジェクトの核心である「未来予測」能力をニューラルレベルで具現化する。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

from .neurons import AdaptiveLIFNeuron
from .snn_core import BaseModel, SNNLayerNorm

class HRMLayer(nn.Module):
    """
    HRMの単一階層を表現するモジュール。
    それぞれが異なる時間スケールで動作する。
    """
    def __init__(self, input_dim: int, hidden_dim: int, top_down_dim: int, neuron_config: Dict[str, Any]):
        super().__init__()
        # Bottom-up path (下位層からの入力)
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # Top-down path (上位層からの予測)
        self.top_down_fc = nn.Linear(top_down_dim, hidden_dim)
        # Recurrent connection (自身の過去の状態)
        self.recurrent_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.neuron = AdaptiveLIFNeuron(features=hidden_dim, **neuron_config)
        self.norm = SNNLayerNorm(hidden_dim)

    def forward(self, x_bottom_up: torch.Tensor, h_recurrent: torch.Tensor, z_top_down: torch.Tensor) -> torch.Tensor:
        """
        1ステップの更新を実行する。
        Args:
            x_bottom_up (torch.Tensor): 下位層からの入力状態
            h_recurrent (torch.Tensor): この層の1ステップ前の状態
            z_top_down (torch.Tensor): 上位層からのトップダウン予測
        """
        current_input = self.input_fc(x_bottom_up) + self.recurrent_fc(h_recurrent) + self.top_down_fc(z_top_down)
        h_new_spikes, _ = self.neuron(current_input)
        return self.norm(h_new_spikes)

class SpikingHRM(BaseModel):
    """
    複数のHRMLayerを階層的に組み合わせた、完全なSpiking-HRMモデル。
    """
    def __init__(self, vocab_size: int, d_model: int, hrm_layers: int, layer_dims: List[int], layer_clocks: List[int], time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        assert hrm_layers == len(layer_dims) == len(layer_clocks), "HRM parameters must have the same length."

        self.time_steps = time_steps
        self.hrm_layers = hrm_layers
        self.layer_clocks = layer_clocks

        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # HRM階層を構築
        self.layers = nn.ModuleList()
        input_dim = d_model
        for i in range(hrm_layers):
            top_down_dim = layer_dims[i+1] if i < hrm_layers - 1 else layer_dims[i]
            layer = HRMLayer(input_dim, layer_dims[i], top_down_dim, neuron_config)
            self.layers.append(layer)
            input_dim = layer_dims[i]

        self.output_projection = nn.Linear(sum(layer_dims), vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        x_emb = self.embedding(input_ids)
        
        # 各階層の状態を初期化
        h_states = [torch.zeros(B, L, dim, device=input_ids.device) for dim in self.config.layer_dims]
        
        # 全時間ステップの最終出力を記録
        all_logits = []

        for t in range(self.time_steps):
            # 最下位層への入力
            bottom_up_input = x_emb
            
            for i in range(self.hrm_layers):
                # このタイムステップで層を更新するかどうかをクロックに基づいて決定
                if t % self.layer_clocks[i] == 0:
                    # 上位層からのトップダウン信号 (最上位層は自身の状態を使用)
                    z_top_down = h_states[i+1] if i < self.hrm_layers - 1 else h_states[i]
                    
                    # 1シーケンス分のバッチ処理
                    h_new = self.layers[i](bottom_up_input, h_states[i], z_top_down)
                    h_states[i] = h_new
                
                # 次の層への入力は、現在の層の出力
                bottom_up_input = h_states[i]
            
            # 全階層の状態を結合して出力を予測
            combined_state = torch.cat(h_states, dim=-1)
            logits = self.output_projection(combined_state)
            all_logits.append(logits)

        # 時間ステップで平均化して最終的なロジットとする
        final_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        
        avg_spikes_val = self.get_total_spikes() / (L * self.time_steps * B) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        mem = torch.tensor(0.0, device=input_ids.device)
        
        return final_logits, avg_spikes, mem
