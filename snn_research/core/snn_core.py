# ファイルパス: snn_research/core/snn_core.py
# (修正)
# Title: SNNモデルの定義、次世代ニューロンなど、中核となるロジック
# Description:
# - 循環インポートを解消するため、共有クラスを `base.py` に移動し、そこからインポートする。
# - mypyエラーとNameErrorを解消するため、`SNNCore.__init__`メソッドの実装を修正。
# - 新しいアーキテクチャ(SpikingMamba, SpikingHRM)を `model_map` に追加。
# - 修正(v2): AttributeErrorを解消するため、BreakthroughSNNにd_model属性を保存する処理を追加。
# - 修正(v3): ベンチマーク実行時のRuntimeErrorを解消するため、output_hidden_statesフラグに応じて
#            モデルが出力を切り替えるよう修正。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type, cast
import math
from omegaconf import DictConfig, OmegaConf

from .base import BaseModel, SNNLayerNorm
from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from .mamba_core import SpikingMamba
from .hrm_core import SpikingHRM

# --- レイヤーとモジュール ---

class PredictiveCodingLayer(nn.Module):
    error_mean: torch.Tensor
    error_std: torch.Tensor

    def __init__(self, d_model: int, d_state: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_neuron = neuron_class(features=d_model, **neuron_params)
        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_neuron = neuron_class(features=d_state, **neuron_params)
        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)
        self.error_scale = nn.Parameter(torch.ones(1))
        
        self.register_buffer('error_mean', torch.zeros(1))
        self.register_buffer('error_std', torch.ones(1))
        self.error_momentum = 0.9

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, _ = self.generative_neuron(self.generative_fc(self.norm_state(top_down_state)))
        raw_error = bottom_up_input - prediction
        
        if self.training:
            with torch.no_grad():
                batch_mean = raw_error.mean()
                batch_std = raw_error.std() + 1e-5
                self.error_mean = self.error_momentum * self.error_mean + (1 - self.error_momentum) * batch_mean
                self.error_std = self.error_momentum * self.error_std + (1 - self.error_momentum) * batch_std
        
        normalized_error = (raw_error - self.error_mean) / self.error_std
        prediction_error = normalized_error * self.error_scale
        
        state_update, _ = self.inference_neuron(self.inference_fc(self.norm_error(prediction_error)))
        updated_state = top_down_state * 0.9 + state_update * 0.1
        
        return updated_state, prediction_error, prediction

class SpikeDrivenSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.neuron_q = AdaptiveLIFNeuron(features=d_model)
        self.neuron_k = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, _ = self.neuron_q(self.q_proj(x))
        k, _ = self.neuron_k(self.k_proj(x))
        v = self.v_proj(x)
        q = q.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, N, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, N, self.n_head, self.d_head).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k) / math.sqrt(self.d_head)
        attn_weights = torch.sigmoid(attn_scores)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out_proj(attn_output)

class STAttenBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1 = SNNLayerNorm(d_model)
        self.attn = SpikeDrivenSelfAttention(d_model, n_head)
        self.lif1 = AdaptiveLIFNeuron(features=d_model)
        self.norm2 = SNNLayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.lif2 = AdaptiveLIFNeuron(features=d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.lif3 = AdaptiveLIFNeuron(features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        attn_out = self.attn(self.norm1(x))
        x_attn = x + attn_out
        x_flat = x_attn.reshape(B * T, D)
        spike_flat, _ = self.lif1(x_flat)
        x_res = spike_flat.reshape(B, T, D)
        ffn_in = self.norm2(x_res)
        ffn_flat = ffn_in.reshape(B * T, D)
        ffn_hidden, _ = self.lif2(self.fc1(ffn_flat))
        ffn_out_flat = self.fc2(ffn_hidden)
        ffn_out = ffn_out_flat.reshape(B, T, D)
        x_ffn = x_res + ffn_out
        x_ffn_flat = x_ffn.reshape(B * T, D)
        out_flat, _ = self.lif3(x_ffn_flat)
        out = out_flat.reshape(B, T, D)
        return out

class BreakthroughSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Linear(d_model, d_model)

        neuron_params = neuron_config.copy() if neuron_config is not None else {}
        neuron_params.pop('type', None)
        neuron_params.pop('num_branches', None)
        neuron_params.pop('branch_features', None)

        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, AdaptiveLIFNeuron, neuron_params) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_state * num_layers, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        token_emb = self.token_embedding(input_ids)
        embedded_sequence = self.input_encoder(token_emb)
        inference_neuron = cast(AdaptiveLIFNeuron, self.pc_layers[0].inference_neuron)
        states = [torch.zeros(batch_size, self.d_state, device=device) for _ in range(self.num_layers)]
        all_timestep_outputs = []
        for _ in range(self.time_steps):
            sequence_outputs = []
            for i in range(seq_len):
                bottom_up_input = embedded_sequence[:, i, :]
                for j in range(self.num_layers):
                    states[j], error, _ = self.pc_layers[j](bottom_up_input, states[j])
                    bottom_up_input = error
                sequence_outputs.append(torch.cat(states, dim=1))
            all_timestep_outputs.append(torch.stack(sequence_outputs, dim=1))
        
        final_hidden_states = all_timestep_outputs[-1]
        
        if output_hidden_states:
             # 分類タスクなど、最終層の前の特徴量が必要な場合に備える
             output = final_hidden_states
        else:
             output = self.output_projection(final_hidden_states)
        
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        return output, avg_spikes, torch.tensor(0.0, device=device)

class SpikingTransformer(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, num_layers: int, time_steps: int, **kwargs: Any):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        self.layers = nn.ModuleList([STAttenBlock(d_model, n_head) for _ in range(num_layers)])
        self.final_norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, output_hidden_states: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        for layer in self.layers:
            block = cast(STAttenBlock, layer)
            cast(AdaptiveLIFNeuron, block.lif1).set_stateful(True)
            cast(AdaptiveLIFNeuron, block.lif2).set_stateful(True)
            cast(AdaptiveLIFNeuron, block.lif3).set_stateful(True)

        for _ in range(self.time_steps):
            for layer in self.layers:
                x = layer(x)
        
        for layer in self.layers:
            block = cast(STAttenBlock, layer)
            cast(AdaptiveLIFNeuron, block.lif1).set_stateful(False)
            cast(AdaptiveLIFNeuron, block.lif2).set_stateful(False)
            cast(AdaptiveLIFNeuron, block.lif3).set_stateful(False)

        x_normalized = self.final_norm(x)
        
        if output_hidden_states:
            output = x_normalized
        else:
            output = self.output_projection(x_normalized)
        
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (seq_len * self.time_steps * batch_size) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=device)
        return output, avg_spikes, torch.tensor(0.0, device=device)

class SimpleSNN(BaseModel):
    def __init__(self, vocab_size: int, d_model: int, hidden_size: int, **kwargs: Any):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.lif1 = AdaptiveLIFNeuron(features=hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        outputs = []
        functional.reset_net(self)
        for t in range(T):
            x_t = x[:, t, :]
            out, _ = self.lif1(self.fc1(x_t))
            out = self.fc2(out)
            outputs.append(out)
        logits = torch.stack(outputs, dim=1)
        avg_spikes_val = self.get_total_spikes() / (B * T) if return_spikes else 0.0
        avg_spikes = torch.tensor(avg_spikes_val, device=input_ids.device)
        return logits, avg_spikes, torch.tensor(0.0, device=input_ids.device)

class SNNCore(nn.Module):
    def __init__(self, config: DictConfig, vocab_size: int):
        super(SNNCore, self).__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        model_type = self.config.get("architecture_type", "simple")
        self.model: nn.Module
        
        params: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True))
        params.pop('path', None)
        neuron_config = params.pop('neuron', {})

        model_map = {
            "predictive_coding": BreakthroughSNN,
            "spiking_transformer": SpikingTransformer,
            "spiking_mamba": SpikingMamba,
            "spiking_hrm": SpikingHRM,
            "simple": SimpleSNN
        }
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = model_map[model_type](vocab_size=vocab_size, neuron_config=neuron_config, **params)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
