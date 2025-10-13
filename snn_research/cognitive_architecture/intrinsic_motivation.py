# ファイルパス: snn_research/cognitive_architecture/intrinsic_motivation.py
# 改善点:
# - 好奇心の源泉となった文脈（予測誤差が最大だった時の情報）を記録する機能を追加。

import numpy as np
from collections import deque
from typing import Dict, Any, Optional

class IntrinsicMotivationSystem:
    """
    エージェントの内部状態（好奇心、自信、退屈）と、その源泉を管理するシステム。
    """
    def __init__(self, history_length: int = 100):
        self.prediction_errors = deque(maxlen=history_length)
        self.task_success_rates = deque(maxlen=history_length)
        self.task_similarities = deque(maxlen=history_length)
        self.loss_history = deque(maxlen=history_length)
        # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
        # 好奇心の対象を記録する
        self.curiosity_context: Optional[Any] = None
        self.max_prediction_error: float = 0.0
        # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

    def update_metrics(self, prediction_error: float, success_rate: float, task_similarity: float, loss: float, context: Optional[Any] = None):
        """
        最新のタスク実行結果から各メトリクスを更新する。
        """
        self.prediction_errors.append(prediction_error)
        self.task_success_rates.append(success_rate)
        self.task_similarities.append(task_similarity)
        self.loss_history.append(loss)

        # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
        # 過去最大の予測誤差であれば、その時の文脈を「最も興味深いこと」として記憶
        if prediction_error > self.max_prediction_error:
            self.max_prediction_error = prediction_error
            self.curiosity_context = context
            print(f"🌟 新しい好奇心の対象を発見: {str(context)[:100]}")
        # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

    def get_internal_state(self) -> Dict[str, Any]:
        """
        現在の内部状態を定量的な指標として計算する。
        """
        state = {
            "curiosity": self._calculate_curiosity(),
            "confidence": self._calculate_confidence(),
            "boredom": self._calculate_boredom(),
            # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
            "curiosity_context": self.curiosity_context
            # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---
        }
        return state

    def _calculate_curiosity(self) -> float:
        # (変更なし)
        if not self.prediction_errors:
            return 0.5
        return np.mean(self.prediction_errors)

    def _calculate_confidence(self) -> float:
        # (変更なし)
        if not self.task_success_rates:
            return 0.5
        return np.mean(self.task_success_rates)

    def _calculate_boredom(self) -> float:
        # (変更なし)
        if len(self.loss_history) < 2: return 0.0
        loss_change_rate = np.mean(np.abs(np.diff(list(self.loss_history))))
        stagnation = 1.0 - np.tanh(loss_change_rate * 10)
        avg_similarity = np.mean(self.task_similarities)
        return stagnation * avg_similarity
