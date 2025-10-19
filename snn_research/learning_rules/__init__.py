# ファイルパス: snn_research/learning_rules/__init__.py
# (更新)
# - ProbabilisticHebbian を追加

from typing import Dict, Any
from .base_rule import BioLearningRule
from .stdp import STDP
from .reward_modulated_stdp import RewardModulatedSTDP
from .causal_trace import CausalTraceCreditAssignment
from .probabilistic_hebbian import ProbabilisticHebbian # 新しいルールをインポート

def get_bio_learning_rule(name: str, params: Dict[str, Any]) -> BioLearningRule:
    """指定された名前に基づいて生物学的学習ルールオブジェクトを生成して返す。"""
    if name == "STDP":
        return STDP(**params.get('stdp', {}))
    elif name == "REWARD_MODULATED_STDP":
        return RewardModulatedSTDP(**params.get('reward_modulated_stdp', {}))
    elif name == "CAUSAL_TRACE":
        stdp_params = params.get('stdp', {})
        causal_params = params.get('causal_trace', {})
        combined_params = {**stdp_params, **causal_params}
        return CausalTraceCreditAssignment(**combined_params)
    elif name == "PROBABILISTIC_HEBBIAN": # 新しいルールを追加
        return ProbabilisticHebbian(**params.get('probabilistic_hebbian', {}))
    else:
        raise ValueError(f"未知の学習ルール名です: {name}")

__all__ = [
    "BioLearningRule", "STDP", "RewardModulatedSTDP",
    "CausalTraceCreditAssignment",
    "ProbabilisticHebbian", # 公開リストに追加
    "get_bio_learning_rule"
]
