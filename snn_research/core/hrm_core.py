# ファイルパス: snn_research/core/hrm_core.py
# (修正)
# Title: Spiking-HRMモデル コア実装
# Description:
# - 循環インポートエラーを解消するため、BaseModelとSNNLayerNormの
#   インポート元を `snn_core` から新しい `base` モジュールに変更。
# - mypyエラーを解消するため、__init__でインスタンス変数 `self.layer_dims` を
#   正しく保存し、forwardメソッドで `self.config` ではなく `self.layer_dims` を参照するように修正。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

from .neurons import AdaptiveLIFNeuron
from .base import BaseModel, SNNLayerNorm

class HRMLayer(nn.Module):
    """
    HRMの単一階層を表現するモジュール。
    それぞれが異なる時間スケールで動作する。
    """
    def __init__(self, input_dim: int, hidden_dim: int, top_down_dim: int, neuron_config: Dict[str, Any]):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.top_down_fc = nn.Linear(top_down_dim, hidden_dim)
        self.recurrent_fc = nn.Linear(hidden_dim, hidden_dim)
        
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        self.neuron = AdaptiveLIFNeuron(features=hidden_dim, **neuron_params)
        self.norm = SNNLayerNorm(hidden_dim)

    def forward(self, x_bottom_up: torch.Tensor, h_recurrent: torch.Tensor, z_top_down: torch.Tensor) -> torch.Tensor:
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
        self.layer_dims = layer_dims
        self.layer_clocks = layer_clocks
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        input_dim = d_model
        for i in range(hrm_layers):
            top_down_dim = self.layer_dims[i+1] if i < self.hrm_layers - 1 else self.layer_dims[i]
            layer = HRMLayer(input_dim, self.layer_dims[i], top_down_dim, neuron_config)
            self.layers.append(layer)
            input_dim = self.layer_dims[i]
        self.output_projection = nn.Linear(sum(self.layer_dims), vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        x_emb = self.embedding(input_ids)
        h_states = [torch.zeros(B, L, dim, device=input_ids.device) for dim in self.layer_dims]
        all_logits = []

        for t in range(self.time_steps):
            bottom_up_input = x_emb
            for i in range(self.hrm_layers):
                if t % self.layer_clocks[i] == 0:
                    z_top_down = h_states[i+1] if i < self.hrm_layers - 1 else h_states[i]
                    h_new = self.layers[i](bottom_up_input, h_states[i], z_top_down)
                    h_states[i] = h_new
                bottom_up_input = h_states[i]
            
            combined_state = torch.cat(h_states, dim=-1)
            logits = self.output_projection(combined_state)
            all_logits.append(logits)

        final_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        avg_spikes_val = self.get_total_spikes() / (L * self.time_steps * B) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        mem = torch.tensor(0.0, device=input_ids.device)
        return final_logits, avg_spikes, mem
