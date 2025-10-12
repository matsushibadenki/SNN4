# ファイルパス: snn_research/communication/spike_encoder_decoder.py
# (更新)
# Title: スパイク エンコーダー/デコーダー
# Description: ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、
#              抽象データ（テキスト、辞書）とスパイクパターンを相互に変換する。
# 修正点:
# - mypyエラー `Name "random" is not defined` を解消するため、randomモジュールをインポート。
# 改善点(v2): エージェント間通信プロトコルの基礎として、メッセージに
#              「意図」と「内容」を含めるようにエンコード・デコード機能を拡張。

import torch
import json
import random
from typing import Dict, Any, Optional

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

    def _encode_text_to_spikes(self, text: str) -> torch.Tensor:
        """
        テキスト文字列をレート符号化を用いてスパイクパターンに変換する内部メソッド。
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

    def _decode_spikes_to_text(self, spikes: torch.Tensor) -> str:
        """
        スパイクパターンをテキスト文字列にデコードする内部メソッド。
        """
        if spikes is None or not isinstance(spikes, torch.Tensor):
            return ""
        # 各ニューロンの発火数を合計
        spike_counts = spikes.sum(dim=1)
        # 発火数が最も多いニューロンから文字を復元（複数可）
        char_indices = torch.where(spike_counts > 0)[0]
        
        # 発火数でソートし、より確からしい文字から再構築する
        sorted_indices = sorted(char_indices, key=lambda idx: spike_counts[idx], reverse=True)

        text = "".join([chr(int(idx)) for idx in sorted_indices if int(idx) < 256])
        return text

    def encode_message(self, intent: str, payload: Dict[str, Any]) -> torch.Tensor:
        """
        意図とペイロードを持つメッセージをスパイクパターンにエンコードする。
        """
        message_dict = {
            "intent": intent,
            "payload": payload
        }
        json_str = json.dumps(message_dict, sort_keys=True)
        return self._encode_text_to_spikes(json_str)

    def decode_message(self, spikes: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        スパイクをデコードしてメッセージ（辞書）にパースする。
        """
        json_str = self._decode_spikes_to_text(spikes)
        try:
            message = json.loads(json_str)
            if "intent" in message and "payload" in message:
                return message
            return {"error": "Invalid message format", "raw_text": json_str}
        except json.JSONDecodeError:
            return {"error": "Failed to decode spikes to dict", "raw_text": json_str}
