# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# (更新)
#
# Title: 因果推論エンジン (Causal Inference Engine)
#
# Description:
# - GlobalWorkspaceを購読し、連続する認知サイクルで意識に上った情報の連鎖を観察する。
# - 観察された情報のペア（原因候補と結果候補）の出現頻度を記録する。
# - 特定のペアが繰り返し観測された場合、それらを因果関係とみなし、
#   RAGSystemの知識グラフに「Causal Relation」として記録する。
# - AIが自身の経験から世界の法則を自律的に学習するためのコアコンポーネント。
#
# 改善点(v2):
# - ArtificialBrainが予測誤差を評価するために、因果推論が成功したかを
#   知るための `just_inferred` フラグとリセットメソッドを追加。

from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

class CausalInferenceEngine:
    """
    意識の連鎖を観察し、因果関係を推論して知識グラフを構築するエンジン。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: int = 3
    ):
        """
        Args:
            rag_system (RAGSystem): 推論した因果関係を記録するための知識グラフ。
            workspace (GlobalWorkspace): 意識のブロードキャストを購読するためのハブ。
            inference_threshold (int): 因果関係があると判断するための観測回数の閾値。
        """
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.co_occurrence_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        self.just_inferred: bool = False # 新しい因果関係を推論したかどうかのフラグ
        
        # GlobalWorkspaceからのブロードキャストを購読
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🔍 因果推論エンジンが初期化され、Workspaceを購読しました。")

    def reset_inference_flag(self):
        """ArtificialBrainが確認した後にフラグをリセットするためのメソッド。"""
        self.just_inferred = False

    def _get_event_description(self, conscious_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """意識に上った情報を簡潔なイベント記述に変換する。"""
        if not conscious_data:
            return None
            
        event_type = conscious_data.get("type")
        if event_type == "emotion":
            valence = conscious_data.get("valence", 0.0)
            return "strong_negative_emotion" if valence < -0.5 else "strong_positive_emotion" if valence > 0.5 else None
        elif event_type == "perception":
            return "novel_perception"
        elif isinstance(conscious_data, str) and conscious_data.startswith("Fulfill external request"):
             return "external_request_received"
        elif isinstance(conscious_data, dict) and 'action' in conscious_data:
            return f"action_{conscious_data['action']}"
        return None


    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        意識に上った情報の連鎖を観察し、因果関係を推論する。
        """
        current_event = self._get_event_description(conscious_data)
        previous_event = self._get_event_description(self.previous_conscious_info)

        if previous_event and current_event:
            event_pair = (previous_event, current_event)
            self.co_occurrence_counts[event_pair] += 1
            
            print(f"  - 因果推論: イベントペア観測 -> ({previous_event}, {current_event}), 回数: {self.co_occurrence_counts[event_pair]}")

            if self.co_occurrence_counts[event_pair] == self.inference_threshold:
                print(f"  - 🔥 因果関係を推論・記録！")
                self.rag_system.add_causal_relationship(
                    cause=previous_event,
                    effect=current_event
                )
                self.just_inferred = True # フラグを立てる
        
        self.previous_conscious_info = conscious_data
