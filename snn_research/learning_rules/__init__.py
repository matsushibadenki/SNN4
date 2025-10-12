# ファイルパス: snn_research/learning_rules/__init__.py
# (修正)
# 修正: CausalTraceCreditAssignmentの生成時に、STDPのパラメータも正しく結合して渡すように修正。
#       これにより、`TypeError: CausalTraceCreditAssignment.__init__() missing ... arguments` エラーを解消する。

from typing import Dict, Any
from .base_rule import BioLearningRule
from .stdp import STDP
from .reward_modulated_stdp import RewardModulatedSTDP
from .causal_trace import CausalTraceCreditAssignment

def get_bio_learning_rule(name: str, params: Dict[str, Any]) -> BioLearningRule:
    """指定された名前に基づいて生物学的学習ルールオブジェクトを生成して返す。"""
    if name == "STDP":
        return STDP(**params['stdp'])
    elif name == "REWARD_MODULATED_STDP":
        return RewardModulatedSTDP(**params['reward_modulated_stdp'])
    elif name == "CAUSAL_TRACE":
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # STDPのパラメータとCausalTrace固有のパラメータを結合して渡す
        stdp_params = params.get('stdp', {})
        causal_params = params.get('causal_trace', {})
        combined_params = {**stdp_params, **causal_params}
        return CausalTraceCreditAssignment(**combined_params)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    else:
        raise ValueError(f"未知の学習ルール名です: {name}")

__all__ = ["BioLearningRule", "STDP", "RewardModulatedSTDP", "CausalTraceCreditAssignment", "get_bio_learning_rule"]
