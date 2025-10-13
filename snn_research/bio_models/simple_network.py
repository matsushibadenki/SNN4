# snn_research/bio_models/simple_network.py
# (修正)
# 修正: learning_rule.update がタプルを返すようになったため、
#       戻り値を正しくアンパックして使用する。

# ... (import文などは変更なし) ...
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment


class BioSNN(nn.Module):
    # ... (init, forward は変更なし) ...
    def __init__(self, layer_sizes: List[int], neuron_params: dict, learning_rule: BioLearningRule, 
                 sparsification_config: Optional[Dict[str, Any]] = None): # ◾️ 追加
        super().__init__()
        self.layer_sizes = layer_sizes
        self.learning_rule = learning_rule
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        if self.sparsification_enabled:
            print(f"🧬 適応的因果スパース化が有効です (貢献度閾値: {self.contribution_threshold})")
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        # 層と重みのリストを作成
        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BioLIFNeuron(layer_sizes[i+1], neuron_params))
            # 重みをParameterとして登録
            weight = nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i]) * 0.5)
            self.weights.append(weight)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """推論のみを実行し、最終出力スパイクと各層のスパイク履歴を返す。"""
        hidden_spikes_history = []
        current_spikes = input_spikes
        
        for i, layer in enumerate(self.layers):
            current = torch.matmul(self.weights[i], current_spikes)
            current_spikes = layer(current)
            hidden_spikes_history.append(current_spikes)
            
        return current_spikes, hidden_spikes_history
        
    def update_weights(
        self,
        all_layer_spikes: List[torch.Tensor],
        optional_params: Optional[Dict[str, Any]] = None
    ):
        if not self.training:
            return

        backward_credit: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        for i in reversed(range(len(self.weights))):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            
            if backward_credit is not None:
                reward_signal = current_params.get("reward", 0.0)
                modulated_reward = reward_signal + backward_credit.mean().item()
                current_params["reward"] = modulated_reward

            # dw, backward_credit のタプルをアンパックして受け取る
            dw, backward_credit_new = self.learning_rule.update(
                pre_spikes=pre_spikes, 
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=current_params
            )
            # 次のループのためにクレジット信号を更新
            backward_credit = backward_credit_new

            if self.sparsification_enabled and isinstance(self.learning_rule, CausalTraceCreditAssignment):
                if self.learning_rule.causal_contribution is not None:
                    contribution_mask = self.learning_rule.causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask

            self.weights[i].data += dw
            self.weights[i].data.clamp_(min=0)