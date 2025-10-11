# ファイルパス: snn_research/cognitive_architecture/cortex.py
# (修正)
#
# Title: Cortex (大脳皮質) モジュール
#
# Description:
# - mypyエラーを解消するため、辞書のキーとして使用する変数がNoneでないこと、
#   かつ文字列であることをisinstanceで明示的にチェックする処理を追加。
# - 人工脳アーキテクチャの「記憶層」に属し、長期記憶を担うコンポーネント。
# - Hippocampus (海馬) から送られてきた短期記憶（エピソード）を、
#   永続的な知識として構造化し、固定する役割を持つ。
# - 知識をナレッジグラフとして表現し、概念間の関連性を基にした検索を可能にする。

from typing import Dict, Any, Optional, List

class Cortex:
    """
    長期的な知識をナレッジグラフとして管理する大脳皮質モジュール。
    """
    def __init__(self):
        # 知識を格納するためのグラフ構造 (辞書で簡易的に表現)
        # 例: {'concept_A': [{'relation': 'is_a', 'target': 'category_X'}]}
        self.knowledge_graph: Dict[str, List[Dict[str, Any]]] = {}
        print("🧠 大脳皮質（長期記憶）モジュールが初期化されました。")

    def consolidate_memory(self, episode: Dict[str, Any]):
        """
        短期記憶のエピソードを解釈し、長期記憶として知識グラフに統合（固定）する。

        Args:
            episode (Dict[str, Any]):
                Hippocampusから送られてきた単一の記憶エピソード。
                {'source': 'concept_A', 'relation': 'is_a', 'target': 'category_X'}
                のような構造を期待する。
        """
        source = episode.get("source")
        relation = episode.get("relation")
        target = episode.get("target")

        # sourceが文字列であることを明示的にチェック
        if isinstance(source, str) and source and relation and target:
            # 'source'がNoneでないことが保証されたため、安全にキーとして使用できる
            if source not in self.knowledge_graph:
                self.knowledge_graph[source] = []

            # 新しい知識（関係性）を追加
            self.knowledge_graph[source].append({"relation": relation, "target": target})
            print(f"📚 大脳皮質: 新しい知識を固定しました: '{source}' --({relation})--> '{target}'")
        else:
            print("⚠️ 大脳皮質: 知識として統合するには情報が不十分なエピソードです。")
            return


    def retrieve_knowledge(self, concept: str) -> Optional[List[Dict[str, Any]]]:
        """
        指定された概念に関連する知識を長期記憶から検索する。

        Args:
            concept (str): 検索のキーとなる概念。

        Returns:
            Optional[List[Dict[str, Any]]]:
                見つかった関連知識のリスト。見つからない場合はNone。
        """
        print(f"🔍 大脳皮質: 概念 '{concept}' に関連する知識を検索中...")
        return self.knowledge_graph.get(concept)

    def get_all_knowledge(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        現在保持している全ての知識グラフを返す。
        """
        return self.knowledge_graph