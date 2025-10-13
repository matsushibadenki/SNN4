# ファイルパス: snn_research/learning_rules/causal_trace.py
# (修正)
# 修正: 親クラスのupdateメソッドがタプルを返すようになったため、アンパックして使用する。
# 改善:
# - 因果的貢献度を追跡する`causal_contribution`トレースを追加。
# - 戻り値として、前段の層に伝えるクレジット信号`backward_credit`を明確に計算して返す。

import torch
from typing import Dict, Any, Optional, Tuple
from .reward_modulated_stdp import RewardModulatedSTDP

class CausalTraceCreditAssignment(RewardModulatedSTDP):
    """
    報酬信号と階層的クレジット信号に基づき重みを更新し、
    前段へのクレジット信号を生成する、進化した因果学習則。
    """
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, tau_eligibility, dt)
        # 長期的な因果的貢献度を記録するトレース
        self.causal_contribution: Optional[torch.Tensor] = None
        print("🧠 Advanced Causal Trace Credit Assignment rule initialized.")

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
        # 親クラス(RewardModulatedSTDP)のupdateを呼び出し、基本的な重み変化量dwを計算
        dw, _ = super().update(pre_spikes, post_spikes, weights, optional_params)

        # 因果的貢献度トレースの初期化
        if self.causal_contribution is None or self.causal_contribution.shape != weights.shape:
            self._initialize_contribution_trace(weights.shape, weights.device)
        
        assert self.causal_contribution is not None, "Causal contribution trace not initialized."

        # 報酬が発生した場合、その重み変化の大きさを長期的な貢献度として記録（指数移動平均）
        if optional_params and optional_params.get("reward", 0.0) != 0.0:
            self.causal_contribution = self.causal_contribution * 0.99 + torch.abs(dw) * 0.01

        # クレジット信号の逆方向伝播
        # 適格度トレース(eligibility_trace)と現在の重みに基づいて、
        # この層の発火(post_spikes)に貢献した前段のニューロン(pre_spikes)への
        # クレジットを計算する。
        if self.eligibility_trace is not None:
            # backward_creditの形状は pre_spikes と同じになる
            backward_credit = torch.einsum('ij,j->i', weights, self.eligibility_trace.sum(dim=0))
        else:
            backward_credit = torch.zeros_like(pre_spikes)

        return dw, backward_credit