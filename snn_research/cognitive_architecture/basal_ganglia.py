# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# タイトル: 大脳基底核：情動変調を伴う行動選択モジュール
# 機能説明:
# - 脳の直接路（Go）と間接路（NoGo）の機能を模倣し、複数の選択肢から最適な行動を決定する。
# - Amygdalaから受け取った情動コンテキスト（快・不快、覚醒・沈静）に基づき、
#   意思決定の閾値を動的に調整する。例えば、危険を察知した場合（不快・高覚醒）、
#   より迅速に行動を起こせるように閾値を下げる。
# - 実行ログを強化し、情動が意思決定に与えた影響を明確に表示するようにした。

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

class BasalGanglia:
    """
    価値信号と情動文脈に基づいて行動選択を行う大脳基底核モジュール。
    """
    def __init__(self, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        """
        Args:
            selection_threshold (float): 行動を実行に移すための基本的な活性化レベル。
            inhibition_strength (float): 選択されなかった行動に対する抑制の強さ。
        """
        self.base_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        print("🧠 大脳基底核モジュールが初期化されました。")

    def _modulate_threshold(self, emotion_context: Optional[Dict[str, float]]) -> float:
        """情動状態に基づいて行動選択の閾値を動的に調整する。"""
        if emotion_context is None:
            return self.base_threshold

        valence = emotion_context.get("valence", 0.0)
        arousal = emotion_context.get("arousal", 0.0)
        
        # 覚醒度が高いほど、閾値は下がり、より反応的になる
        # valenceが負（不快）の場合、その効果はさらに増幅される (危険回避など)
        arousal_effect = -arousal * 0.2
        valence_effect = -valence * arousal * 0.1 # 不快で覚醒度が高いほど、さらに閾値を下げる
        
        modulated_threshold = self.base_threshold + arousal_effect + valence_effect
        
        # 閾値が0.1〜0.9の範囲に収まるようにクリップ
        final_threshold = max(0.1, min(0.9, modulated_threshold))
        
        if final_threshold != self.base_threshold:
            print(f"  - 大脳基底核: 情動により閾値を調整 ({self.base_threshold:.2f} -> {final_threshold:.2f})")
        
        return final_threshold

    def select_action(
        self, 
        action_candidates: List[Dict[str, Any]],
        emotion_context: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        提示された行動候補の中から、実行すべき最適な行動を一つ選択する。

        Args:
            action_candidates (List[Dict[str, Any]]):
                各要素が行動とその価値を持つ辞書のリスト。
                例: [{'action': 'A', 'value': 0.9}, {'action': 'B', 'value': 0.6}]
            emotion_context (Optional[Dict[str, float]]):
                現在の情動状態。例: {'valence': -0.8, 'arousal': 0.9}

        Returns:
            Optional[Dict[str, Any]]: 選択された行動。どの行動も閾値に達しない場合はNone。
        """
        if not action_candidates:
            print("🤔 大脳基底核: 行動候補が提示されませんでした。")
            return None
            
        current_threshold = self._modulate_threshold(emotion_context)

        # 各候補の価値をテンソルに変換
        values = torch.tensor([candidate.get('value', 0.0) for candidate in action_candidates])
        print(f"  - 大脳基底核: 検討中の行動候補: {[c.get('action') for c in action_candidates]}, 価値: {[round(v.item(), 2) for v in values]}")


        # 1. 直接路 (Go Pathway) のシミュレーション:
        #    最も価値の高い行動を特定する。
        winner_takes_all = F.softmax(values * 5.0, dim=0) # softmaxで勝者を強調

        # 2. 間接路 (NoGo Pathway) のシミュレーション:
        #    競合する行動に対する抑制を計算する。
        #    ここでは、勝者以外の活性を弱める形で簡易的に表現。
        inhibition_mask = torch.ones_like(values)
        winner_index = torch.argmax(values)
        # 勝者以外のインデックスに抑制を適用
        for i in range(len(inhibition_mask)):
            if i != winner_index:
                inhibition_mask[i] = 1.0 - self.inhibition_strength


        # 最終的な行動活性を計算
        final_activation = winner_takes_all * inhibition_mask

        # 3. 最終的な意思決定
        # 最も活性の高い行動が、実行閾値を超えているか確認
        best_action_index = torch.argmax(final_activation)
        if final_activation[best_action_index] >= current_threshold:
            selected_action = action_candidates[best_action_index]
            print(f"🏆 行動選択: '{selected_action.get('action')}' (活性値: {final_activation[best_action_index]:.2f}, 閾値: {current_threshold:.2f})")
            return selected_action
        else:
            print(f"🤔 行動棄却: どの行動も実行閾値 ({current_threshold:.2f}) に達しませんでした。(最大活性値: {final_activation.max():.2f})")
            return None
