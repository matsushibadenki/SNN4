# snn_research/learning_rules/causal_trace.py
# (修正)
# 修正: 親クラスのupdateメソッドがタプルを返すようになったため、アンパックして使用する。

import torch
from typing import Dict, Any, Optional, Tuple
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignment(RewardModulatedSTDP):
    # ... (init, _initialize_contribution_trace は変更なし) ...
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        self.causal_contribution: Optional[torch.Tensor] = None
        print("🧠 Causal Trace Credit Assignment learning rule initialized.")

    def _initialize_contribution_trace(self, weight_shape: tuple, device: torch.device):
        """因果的貢献度を記録するトレースを初期化する。"""
        self.causal_contribution = torch.zeros(weight_shape, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        報酬信号と後段からのクレジットに基づき重み変化量を計算し、
        前段へのクレジット信号を生成して返す。
        """
        # 親クラスのupdateは (dw, None) を返すので、必要なdwのみ受け取る
        dw, _ = super().update(pre_spikes, post_spikes, weights, optional_params)

        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)
        
        assert self.causal_contribution is not None, "Causal contribution trace not initialized."

        if optional_params and optional_params.get("reward", 0.0) != 0.0:
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

        if self.eligibility_trace is not None:
            backward_credit = torch.einsum('ij,ij->j', self.eligibility_trace, weights)
        else:
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit