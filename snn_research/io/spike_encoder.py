# ファイルパス: snn_research/io/spike_encoder.py
# (新規作成)
#
# Title: Spike Encoder (スパイクエンコーダー)
#
# Description:
# - 人工脳アーキテクチャの「符号化層」を担うコンポーネント。
# - SensoryReceptorから受け取った内部表現を、SNNが処理可能な
#   スパイクパターンに変換（符号化）する。
# - 設計書に基づき、レート符号化（Rate Coding）を実装する。

import torch
from typing import Dict, Any

class SpikeEncoder:
    """
    感覚情報をスパイクパターンに符号化するモジュール。
    """
    def __init__(self, num_neurons: int, max_rate: int = 100):
        """
        Args:
            num_neurons (int): 符号化に使用するニューロンの数。
            max_rate (int): 最大発火率 (Hz)。
        """
        self.num_neurons = num_neurons
        self.max_rate = max_rate
        print("⚡️ スパイクエンコーダーモジュールが初期化されました。")

    def encode(self, sensory_info: Dict[str, Any], duration: int = 100) -> torch.Tensor:
        """
        感覚情報をレート符号化を用いてスパイクパターンに変換する。

        Args:
            sensory_info (Dict[str, Any]): SensoryReceptorからの出力。
            duration (int): スパイクを生成する期間 (ミリ秒)。

        Returns:
            torch.Tensor: 生成されたスパイクパターン (time_steps, num_neurons)。
        """
        if sensory_info['type'] == 'text':
            return self._rate_encode_text(sensory_info['content'], duration)
        # 他のデータタイプ用のエンコーダーもここに追加可能
        else:
            # 不明なタイプの場合は空のスパイク列を返す
            return torch.zeros((duration, self.num_neurons))

    def _rate_encode_text(self, text: str, duration: int) -> torch.Tensor:
        """
        テキストをレート符号化する。各文字を特定のニューロンにマッピングする。
        """
        time_steps = duration
        spikes = torch.zeros((time_steps, self.num_neurons))

        for char_index, char in enumerate(text):
            # 文字のASCII値をニューロンIDとして使用
            neuron_id = ord(char) % self.num_neurons
            
            # 文字の出現頻度や重要度に応じて発火率を変化させる（ここでは簡易的に固定）
            fire_prob = (self.max_rate * (duration / 1000.0)) / time_steps
            
            # ポアソン分布に従うスパイクを生成
            spikes[:, neuron_id] = torch.poisson(torch.full((time_steps,), fire_prob))

        # スパイクは0か1なので、1より大きい値は1にクリップ
        spikes = torch.clamp(spikes, 0, 1)
        
        print(f"📈 テキストを {time_steps}x{self.num_neurons} のスパイクパターンにレート符号化しました。")
        return spikes