# ファイルパス: snn_research/communication/spike_encoder_decoder.py
# (新規作成)
# Title: スパイク エンコーダー/デコーダー
# Description: ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、
#              抽象データ（テキスト、辞書）とスパイクパターンを相互に変換する。
# 修正点:
# - mypyエラー `Name "random" is not defined` を解消するため、randomモジュールをインポート。

import torch
import json
import random
from typing import Dict, Any

class SpikeEncoderDecoder:
    """
    テキストや辞書などの抽象データをスパイクパターンに変換し、
    またその逆の変換も行うクラス。
    """
    def __init__(self, num_neurons: int = 256, time_steps: int = 16):
        """
        Args:
            num_neurons (int): スパイク表現に使用するニューロン数。ASCII文字セットをカバーできる必要がある。
            time_steps (int): スパイクパターンの時間長。
        """
        self.num_neurons = num_neurons
        self.time_steps = time_steps

    def encode_text_to_spikes(self, text: str) -> torch.Tensor:
        """
        テキスト文字列をレート符号化を用いてスパイクパターンに変換する。
        """
        spike_pattern = torch.zeros((self.num_neurons, self.time_steps))
        for char in text:
            # 文字のASCII値をニューロンIDとして使用
            neuron_id = ord(char) % self.num_neurons
            # 簡単なレート符号化: 1文字あたり数回のスパイクをランダムなタイムステップで発火
            num_spikes = random.randint(1, 3)
            for _ in range(num_spikes):
                t = random.randint(0, self.time_steps - 1)
                spike_pattern[neuron_id, t] = 1
        return spike_pattern

    def decode_spikes_to_text(self, spikes: torch.Tensor) -> str:
        """
        スパイクパターンをテキスト文字列にデコードする。
        """
        if spikes is None or not isinstance(spikes, torch.Tensor):
            return ""
        # 各ニューロンの発火数を合計
        spike_counts = spikes.sum(dim=1)
        # 発火数が最も多いニューロンから文字を復元（複数可）
        char_indices = torch.where(spike_counts > 0)[0]
        
        text = "".join([chr(int(idx)) for idx in char_indices if int(idx) < 256])
        return text

    def encode_dict_to_spikes(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        辞書をJSON文字列に変換してからスパイクにエンコードする。
        """
        json_str = json.dumps(data, sort_keys=True)
        return self.encode_text_to_spikes(json_str)

    def decode_spikes_to_dict(self, spikes: torch.Tensor) -> Dict[str, Any]:
        """
        スパイクをデコードしてJSON文字列に戻し、辞書にパースする。
        """
        json_str = self.decode_spikes_to_text(spikes)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to decode spikes to dict", "raw_text": json_str}