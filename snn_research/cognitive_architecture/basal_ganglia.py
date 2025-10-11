# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# (新規作成)
#
# Title: Basal Ganglia (大脳基底核) モジュール
#
# Description:
# - 人工脳アーキテクチャの「価値評価層」に属し、行動選択を担うコンポーネント。
# - 脳の直接路（Go）と間接路（NoGo）の機能を模倣し、複数の選択肢から
#   最適な行動を決定する。
# - Amygdalaなどから受け取った価値信号に基づき、最も価値の高い行動を選択し、
#   競合する行動を抑制する。

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

class BasalGanglia:
    """
    価値信号に基づいて行動選択を行う大脳基底核モジュール。
    """
    def __init__(self, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        """
        Args:
            selection_threshold (float): 行動を実行に移すための最低活性化レベル。
            inhibition_strength (float): 選択されなかった行動に対する抑制の強さ。
        """
        self.selection_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        print("🧠 大脳基底核モジュールが初期化されました。")

    def select_action(self, action_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        提示された行動候補の中から、実行すべき最適な行動を一つ選択する。

        Args:
            action_candidates (List[Dict[str, Any]]):
                各要素が行動とその価値を持つ辞書のリスト。
                例: [{'action': 'A', 'value': 0.9}, {'action': 'B', 'value': 0.6}]

        Returns:
            Optional[Dict[str, Any]]: 選択された行動。どの行動も閾値に達しない場合はNone。
        """
        if not action_candidates:
            return None

        # 各候補の価値をテンソルに変換
        values = torch.tensor([candidate.get('value', 0.0) for candidate in action_candidates])

        # 1. 直接路 (Go Pathway) のシミュレーション:
        #    最も価値の高い行動を特定する。
        winner_takes_all = F.softmax(values * 5.0, dim=0) # softmaxで勝者を強調

        # 2. 間接路 (NoGo Pathway) のシミュレーション:
        #    競合する行動に対する抑制を計算する。
        #    ここでは、勝者以外の活性を弱める形で簡易的に表現。
        inhibition_mask = torch.ones_like(values)
        winner_index = torch.argmax(values)
        inhibition_mask[winner_index] = 1.0 - self.inhibition_strength

        # 最終的な行動活性を計算
        final_activation = winner_takes_all * inhibition_mask

        # 3. 最終的な意思決定
        # 最も活性の高い行動が、実行閾値を超えているか確認
        best_action_index = torch.argmax(final_activation)
        if final_activation[best_action_index] >= self.selection_threshold:
            selected_action = action_candidates[best_action_index]
            print(f"🏆 行動選択: '{selected_action.get('action')}' (活性値: {final_activation[best_action_index]:.2f})")
            return selected_action
        else:
            print(f"🤔 行動棄却: どの行動も実行閾値 ({self.selection_threshold}) に達しませんでした。")
            return None