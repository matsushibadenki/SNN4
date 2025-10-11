# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# (更新)
#
# Title: Global Workspace with Attention Mechanism
#
# Description:
# - mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
# - 改善点: ROADMAPフェーズ4「スパイクベース通信プロトコル」に基づき、SpikeEncoderDecoderを導入。
# - 改善点 (v2): 設計図に基づき、注意機構(AttentionHub)を統合。
#              各モジュールからの誤差信号を競合させ、勝者となった情報を
#              システム全体にブロードキャストする「意識」の仕組みを実装。

from typing import Dict, Any, List, Callable, Optional, Tuple
import random
import operator

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.communication.spike_encoder_decoder import SpikeEncoderDecoder

class AttentionHub:
    """
    Winner-Take-All競合により、最も重要な情報を選択する注意メカニズム。
    """
    def __init__(self, inhibition_strength: float = 0.5):
        """
        Args:
            inhibition_strength (float): 最近選択された情報源に対する抑制の強さ。
        """
        self.history: List[str] = []
        self.inhibition_strength = inhibition_strength

    def select_winner(self, error_signals: Dict[str, float]) -> Optional[str]:
        """
        誤差信号の大きさと過去の履歴に基づき、最も注意を向けるべき情報源（勝者）を選択する。

        Args:
            error_signals (Dict[str, float]): 各モジュール名とその予測誤差の大きさ。

        Returns:
            Optional[str]: 勝者となったモジュールの名前。
        """
        if not error_signals:
            return None

        # 過去に選択された情報源に抑制をかける (Inhibition of Return)
        adjusted_signals: Dict[str, float] = {}
        for name, signal_strength in error_signals.items():
            inhibition = self._get_inhibition_factor(name)
            adjusted_signals[name] = signal_strength * (1 - inhibition)
            if inhibition > 0:
                print(f"  - AttentionHub: '{name}' に抑制を適用 (抑制率: {inhibition:.2f})")

        # 最も誤差が大きいモジュールを選択
        winner = max(adjusted_signals.items(), key=operator.itemgetter(1))[0]
        print(f"🏆 AttentionHub: '{winner}' が注意を獲得しました (調整後誤差: {adjusted_signals[winner]:.4f})。")

        # 履歴を更新
        self.history.append(winner)
        if len(self.history) > 10:  # 履歴の長さを制限
            self.history.pop(0)

        return winner

    def _get_inhibition_factor(self, module_name: str) -> float:
        """最近選択された頻度に基づいて抑制係数を計算する。"""
        recent_wins = self.history[-5:]  # 直近5回の履歴を参照
        win_count = recent_wins.count(module_name)
        return self.inhibition_strength * (win_count / 5)


class GlobalWorkspace:
    """
    注意機構を備え、認知アーキテクチャ全体で情報をスパイクパターンとして共有する中央情報ハブ。
    """
    def __init__(self, model_registry: ModelRegistry):
        self.blackboard: Dict[str, Any] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.model_registry = model_registry
        self.encoder_decoder = SpikeEncoderDecoder()
        self.attention_hub = AttentionHub()
        self.conscious_broadcast_content: Optional[Any] = None

    def broadcast(self, source: str, data: Any, is_error_signal: bool = False, error_magnitude: float = 0.0):
        """
        情報をスパイクパターンにエンコードしてブラックボードに書き込む。
        誤差信号の場合は、注意機構に通知する。
        """
        print(f"[GlobalWorkspace] '{source}' から情報を受信...")
        # データをスパイクパターンにエンコード
        if isinstance(data, dict):
            spiked_data = self.encoder_decoder.encode_dict_to_spikes(data)
        elif isinstance(data, str):
            spiked_data = self.encoder_decoder.encode_text_to_spikes(data)
        else:
            spiked_data = self.encoder_decoder.encode_text_to_spikes(str(data))
            
        self.blackboard[source] = {"data": spiked_data, "is_error": is_error_signal, "magnitude": error_magnitude}

    def conscious_broadcast_cycle(self):
        """
        意識的な情報処理サイクルを実行する。
        1. 全モジュールから誤差信号を収集する。
        2. 注意機構が最も重要な情報（勝者）を選択する。
        3. 勝者の情報をシステム全体にブロードキャストする。
        """
        print("\n--- 意識的ブロードキャストサイクル開始 ---")
        # 1. 誤差信号を収集
        error_signals = {
            source: info["magnitude"]
            for source, info in self.blackboard.items()
            if info["is_error"]
        }
        print(f"  - 収集された誤差信号: {error_signals}")

        # 2. 注意を向ける勝者を選択
        winner = self.attention_hub.select_winner(error_signals)

        if winner and winner in self.blackboard:
            # 3. 勝者の情報をデコードしてブロードキャスト
            winner_info = self.get_information(winner)
            self.conscious_broadcast_content = winner_info
            print(f"📡 意識的ブロードキャスト: '{winner}' からの情報を全システムに伝達します。")
            self._notify_subscribers(winner, winner_info)
        else:
            print("  - ブロードキャストするべき支配的な情報はありませんでした。")
        
        print("--- 意識的ブロードキャストサイクル終了 ---\n")

    def subscribe(self, source: str, callback: Callable):
        """特定のソースからの情報更新を購読する。"""
        if source not in self.subscribers:
            self.subscribers[source] = []
        self.subscribers[source].append(callback)

    def _notify_subscribers(self, source: str, decoded_info: Any):
        """更新があったソースの購読者に通知する。"""
        if source in self.subscribers:
            for callback in self.subscribers[source]:
                try:
                    callback(source, decoded_info)
                except Exception as e:
                    print(f"Error notifying subscriber for '{source}': {e}")

    def get_information(self, source: str) -> Any:
        """
        ブラックボードからスパイクパターンを取得し、デコードして返す。
        """
        source_info = self.blackboard.get(source)
        if source_info is None:
            return None
        
        spiked_data = source_info["data"]
        
        # まず辞書としてデコードを試みる
        decoded_data = self.encoder_decoder.decode_spikes_to_dict(spiked_data)
        if isinstance(decoded_data, dict) and "error" in decoded_data:
            # 辞書へのデコードが失敗した場合、単純なテキストとしてデコードする
            return self.encoder_decoder.decode_spikes_to_text(spiked_data)
        return decoded_data

    def get_full_context(self) -> Dict[str, Any]:
        """
        現在のワークスペースの全コンテキストをデコードして取得する。
        """
        decoded_context: Dict[str, Any] = {}
        for source in self.blackboard:
            decoded_context[source] = self.get_information(source)
        return decoded_context