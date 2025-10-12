# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# (更新)
#
# Title: Hippocampus (海馬) モジュール
#
# Description:
# - 人工脳アーキテクチャの「記憶層」に属し、短期記憶（ワーキングメモリ）を担う。
# - 新しい情報や経験を「エピソード」として時系列で短期的に保持する。
# - 保持できる情報量には限りがあり、古い記憶は忘却される（FIFO）。
# - 将来的には、長期記憶への転送（記憶の固定）や、
#   注意機構と連携した情報の重み付けなどの機能拡張を想定。
#
# 改善点(v2):
# - ROADMAPフェーズ3に基づき、長期記憶への固定化プロセスを明確にするためのメソッドを追加。

from typing import List, Dict, Any
from collections import deque

class Hippocampus:
    """
    短期的なエピソード記憶を管理する海馬モジュール（ワーキングメモリ）。
    """
    def __init__(self, capacity: int = 100):
        """
        Args:
            capacity (int): ワーキングメモリが保持できるエピソードの最大数。
        """
        self.capacity = capacity
        # 時系列順にエピソードを保持するための両端キュー
        self.working_memory: deque = deque(maxlen=capacity)
        print(f"🧠 海馬（ワーキングメモリ）モジュールが初期化されました (容量: {capacity} エピソード)。")

    def store_episode(self, episode: Dict[str, Any]):
        """
        新しいエピソード（経験や観測）をワーキングメモリに保存する。
        容量を超えた場合、最も古いエピソードが自動的に忘却される。

        Args:
            episode (Dict[str, Any]): 保存するエピソード情報。
                                     例: {'observation': ..., 'action': ..., 'result': ...}
        """
        print(f" hippocampus.py STORE_EPISODE {episode}")
        self.working_memory.append(episode)
        print(f"📝 海馬: 新しいエピソードを記憶しました。 (現在の記憶数: {len(self.working_memory)})")

    def retrieve_recent_episodes(self, num_episodes: int = 5) -> List[Dict[str, Any]]:
        """
        直近のいくつかのエピソードをワーキングメモリから検索して返す。

        Args:
            num_episodes (int): 検索するエピソードの数。

        Returns:
            List[Dict[str, Any]]: 直近のエピソードのリスト。
        """
        if num_episodes <= 0:
            return []

        # キューの右側（最後に追加された要素）から取得
        num_to_retrieve = min(num_episodes, len(self.working_memory))
        recent_episodes = [self.working_memory[-i] for i in range(1, num_to_retrieve + 1)]

        return recent_episodes
    
    def get_and_clear_episodes_for_consolidation(self) -> List[Dict[str, Any]]:
        """
        長期記憶への固定化のために、現在の全エピソードを返し、メモリをクリアする。
        """
        episodes_to_consolidate = list(self.working_memory)
        self.clear_memory()
        print(f"📤 海馬: 長期記憶への固定化のため、{len(episodes_to_consolidate)}件のエピソードを転送しました。")
        return episodes_to_consolidate

    def clear_memory(self):
        """
        ワーキングメモリの内容をすべて消去する。
        """
        self.working_memory.clear()
        print("🗑️ 海馬: ワーキングメモリをクリアしました。")
