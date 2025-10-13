# ファイルパス: snn_research/bio_models/simple_network.py
# (修正)
# 修正: learning_rule.update がタプルを返すようになったため、
#       戻り値を正しくアンパックして使用する。
# 改善:
# - 階層的因果学習のため、クレジット信号を層間で逆方向に伝播させるロジックを実装。
# - 適応的因果スパース化のメカニズムを追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from .lif_neuron import BioLIFNeuron
from snn_research.learning_rules.base_rule import BioLearningRule
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment


class BioSNN(nn.Module):
    def __init__(self, layer_sizes: List[int], neuron_params: dict, learning_rule: BioLearningRule, 
                 sparsification_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.learning_rule = learning_rule
        
        # 適応的因果スパース化の設定
        self.sparsification_enabled = sparsification_config.get("enabled", False) if sparsification_config else False
        self.contribution_threshold = sparsification_config.get("contribution_threshold", 0.0) if sparsification_config else 0.0
        if self.sparsification_enabled:
            print(f"🧬 適応的因果スパース化が有効です (貢献度閾値: {self.contribution_threshold})")

        self.layers = nn.ModuleList()
        self.weights = nn.ParameterList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BioLIFNeuron(layer_sizes[i+1], neuron_params))
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
        """階層的因果クレジット伝播を用いて重みを更新する。"""
        if not self.training:
            return

        # backward_credit: 後段の層から伝播してきたクレジット信号
        backward_credit: Optional[torch.Tensor] = None
        current_params = optional_params.copy() if optional_params else {}

        # 出力層から入力層に向かって逆向きにループ
        for i in reversed(range(len(self.weights))):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            
            # 後段からのクレジット信号が存在すれば、それを局所的な報酬として使用
            if backward_credit is not None:
                # 大域的報酬と局所的クレジット信号を組み合わせる
                # (例: 大域的報酬に局所的クレジットの平均値を加算)
                global_reward = current_params.get("reward", 0.0)
                modulated_reward = global_reward + backward_credit.mean().item()
                current_params["reward"] = modulated_reward

            # 学習則を適用し、重み変化量(dw)と、さらに前段に伝えるクレジット信号(backward_credit_new)を取得
            dw, backward_credit_new = self.learning_rule.update(
                pre_spikes=pre_spikes, 
                post_spikes=post_spikes,
                weights=self.weights[i],
                optional_params=current_params
            )
            
            # 次のループのためにクレジット信号を更新
            backward_credit = backward_credit_new

            # --- 適応的因果スパース化 ---
            if self.sparsification_enabled and isinstance(self.learning_rule, CausalTraceCreditAssignment):
                if self.learning_rule.causal_contribution is not None:
                    # 因果的貢献度が閾値より大きいシナプスのみ重みを更新
                    contribution_mask = self.learning_rule.causal_contribution > self.contribution_threshold
                    dw = dw * contribution_mask

            # 重みを更新
            self.weights[i].data += dw
            # 重みが負にならないようにクランプ
            self.weights[i].data.clamp_(min=0)
