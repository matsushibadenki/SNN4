# ファイルパス: snn_research/agent/memory.py
# (更新)
# Title: 長期記憶システム
# 改善点 (v4): ロードマップ「因果的記憶アクセス」を実装。
#              経験を記録する際に、その成功に寄与したと考えられる
#              「因果スナップショット」を保存する機能を追加。
# 改善点 (v5): retrieve_similar_experiences のダミー実装を
#              TF-IDFに基づくベクトル類似度検索に置き換え。
# 修正点: mypyエラー [import-untyped] を解消するため、type: ignoreを追加。
# 改善点(v6): RAGSystemと連携し、記憶の記録と検索をセマンティックに行うように強化。

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from snn_research.cognitive_architecture.rag_snn import RAGSystem # ◾️ 追加

class Memory:
    """
    エージェントの経験を構造化されたタプルとして長期記憶に記録し、
    RAGSystemと連携してセマンティック検索を行うクラス。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, rag_system: RAGSystem, memory_path: Optional[str] = "runs/agent_memory.jsonl"):
        self.rag_system = rag_system
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        if memory_path is None:
            print("⚠️ MemoryにNoneのパスが渡されたため、デフォルト値 'runs/agent_memory.jsonl' を使用します。")
            self.memory_path: str = "runs/agent_memory.jsonl"
        else:
            self.memory_path = memory_path
        
        if os.path.dirname(self.memory_path):
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def _experience_to_text(self, experience: Dict[str, Any]) -> str:
        """経験の辞書を検索可能なテキスト形式に変換する。"""
        action = experience.get("action", "NoAction")
        result = experience.get("result", {})
        reward = experience.get("reward", {}).get("external", 0.0)
        reason = experience.get("decision_context", {}).get("reason", "NoReason")
        return f"Action '{action}' was taken because '{reason}', resulting in '{str(result)}' with a reward of {reward:.2f}."

    def record_experience(
        self,
        state: Dict[str, Any],
        action: str,
        result: Any,
        reward: Dict[str, Any],
        expert_used: List[str],
        decision_context: Dict[str, Any],
        causal_snapshot: Optional[str] = None
    ):
        """
        単一の経験を記録し、その内容をRAGシステムのベクトルストアにも追加する。
        """
        experience_tuple = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": state,
            "action": action,
            "result": result,
            "reward": reward,
            "expert_used": expert_used,
            "decision_context": decision_context,
            "causal_snapshot": causal_snapshot,
        }
        # ログファイルへの追記
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experience_tuple, ensure_ascii=False) + "\n")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 経験をテキスト化してRAGシステムにリアルタイムで追加
        experience_text = self._experience_to_text(experience_tuple)
        self.rag_system.add_relationship(
            source_concept=f"experience_{experience_tuple['timestamp']}",
            relation="is_described_as",
            target_concept=experience_text
        )
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def retrieve_similar_experiences(self, query_state: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        現在の状態に類似した過去の経験をRAGSystemのセマンティック検索で検索する。
        """
        if not self.rag_system.vector_store:
            print("⚠️ 記憶検索のためのベクトルストアが初期化されていません。")
            return []

        # クエリ状態をテキストに変換
        query_text = f"Find similar past experiences for a situation where the last action was '{query_state.get('last_action')}' and the result was '{str(query_state.get('last_result'))}'."
        
        print(f"🧠

 過去の経験を検索中: {query_text}")
        
        # RAGSystemを使って類似のドキュメント（経験）を検索
        search_results = self.rag_system.search(query_text, k=top_k)

        # 検索結果のテキストから元の経験データを再構築（この例ではテキストをそのまま返す）
        reconstructed_experiences = []
        for res_text in search_results:
            reconstructed_experiences.append({
                "retrieved_text": res_text
            })

        return reconstructed_experiences
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def retrieve_successful_experiences(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        過去の経験の中から、総合的な報酬が高かったものを検索する。
        """
        experiences = []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line in f:
                    experiences.append(json.loads(line))
        except FileNotFoundError:
            return []

        # 報酬（外部報酬と物理的報酬の合計）に基づいて経験をソート
        def get_total_reward(exp: Dict[str, Any]) -> float:
            reward_info = exp.get("reward", {})
            if isinstance(reward_info, dict):
                # 多目的報酬ベクトルの加重合計を計算
                w_external = 1.0
                w_physical = 0.2
                w_curiosity = 0.5
                
                external_reward = float(reward_info.get("external", 0.0))
                
                physical_rewards = reward_info.get("physical", {})
                sparsity_reward = physical_rewards.get("sparsity_reward", 0.0)
                smoothness_reward = physical_rewards.get("smoothness_reward", 0.0)
                
                curiosity_reward = float(reward_info.get("curiosity", 0.0))

                total = (w_external * external_reward +
                         w_physical * (sparsity_reward + smoothness_reward) +
                         w_curiosity * curiosity_reward)
                return total
                
            elif isinstance(reward_info, (int, float)):
                # 古い形式の報酬データとの後方互換性
                return float(reward_info)
            return 0.0

        experiences.sort(key=get_total_reward, reverse=True)
        
        return experiences[:top_k]
