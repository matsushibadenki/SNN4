# snn_research/cognitive_architecture/intrinsic_motivation.py
# 内発的動機付けシステム
# 概要：エージェントの内部状態を定量化し、行動選択の原動力を提供する。
import numpy as np
from collections import deque

class IntrinsicMotivationSystem:
    """
    エージェントの内部状態（好奇心、自信、退屈）を定量化し、
    自律的な意思決定のための動機付けを提供するシステム。
    """
    def __init__(self, history_length=100):
        """
        Args:
            history_length (int): 内部状態の計算に使用する過去のデータポイント数。
        """
        self.prediction_errors = deque(maxlen=history_length)
        self.task_success_rates = deque(maxlen=history_length)
        self.task_similarities = deque(maxlen=history_length)
        self.loss_history = deque(maxlen=history_length)

    def update_metrics(self, prediction_error, success_rate, task_similarity, loss):
        """
        最新のタスク実行結果から各メトリクスを更新する。

        Args:
            prediction_error (float): 最新のタスクの予測誤差。
            success_rate (float): 最新のタスクの成功率（0.0-1.0）。
            task_similarity (float): 現在のタスクと過去のタスクの類似度。
            loss (float): 最新の学習ステップでの損失関数の値。
        """
        self.prediction_errors.append(prediction_error)
        self.task_success_rates.append(success_rate)
        self.task_similarities.append(task_similarity)
        self.loss_history.append(loss)

    def get_internal_state(self):
        """
        現在の内部状態を定量的な指標として計算する。

        Returns:
            dict: 好奇心、自信、退屈のスコアを含む辞書。
        """
        state = {
            "curiosity": self._calculate_curiosity(),
            "confidence": self._calculate_confidence(),
            "boredom": self._calculate_boredom(),
        }
        return state

    def _calculate_curiosity(self):
        """
        好奇心を計算する。予測誤差の大きさや情報の不確実性（エントロピー）として定義。
        予測誤差が大きいほど、新しい情報や未知の環境に対する好奇心が高いと判断。
        
        Returns:
            float: 好奇心スコア。
        """
        if not self.prediction_errors:
            return 0.5  # デフォルト値
        # 予測誤差の平均値を正規化して好奇心スコアとする
        return np.mean(self.prediction_errors)

    def _calculate_confidence(self):
        """
        自信を計算する。特定タスクにおける成功率や予測の安定性として定義。
        成功率が高いほど、そのタスクに対する自信が高いと判断。

        Returns:
            float: 自信スコア。
        """
        if not self.task_success_rates:
            return 0.5  # デフォルト値
        return np.mean(self.task_success_rates)

    def _calculate_boredom(self):
        """
        退屈を計算する。同一または類似タスクの継続時間や学習進捗の停滞度として定義。
        タスクの類似度が高く、かつ損失の変化率が低い場合に退屈と判断。

        Returns:
            float: 退屈スコア。
        """
        if len(self.loss_history) < 2 or not self.task_similarities:
            return 0.0 # データ不足の場合は退屈していない

        # 学習の停滞度（損失の変化率の低さ）
        loss_change_rate = np.mean(np.abs(np.diff(list(self.loss_history))))
        stagnation = 1.0 - np.tanh(loss_change_rate * 10) # 変化が小さいほど1に近づく

        # タスクの類似度
        avg_similarity = np.mean(self.task_similarities)

        # 退屈スコア = 学習の停滞度 * タスクの類似度
        boredom_score = stagnation * avg_similarity
        return boredom_score