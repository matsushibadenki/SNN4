# ファイルパス: snn_research/cognitive_architecture/causal_inference_engine.py
# (更新)
#
# 改善点:
# - 文脈依存の因果関係を推論するロジックを追加。
# - `_get_context_description`メソッドを実装し、PFCの現在の目標などを文脈として利用する。

from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

class CausalInferenceEngine:
    """
    意識の連鎖を観察し、文脈依存の因果関係を推論して知識グラフを構築するエンジン。
    """
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: int = 3
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        
        self.previous_conscious_info: Optional[Dict[str, Any]] = None
        self.previous_context: Optional[str] = None
        # キーを (文脈, 原因, 結果) のタプルに変更
        self.co_occurrence_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        self.just_inferred: bool = False
        
        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🔍 因果推論エンジンが初期化され、Workspaceを購読しました。")

    def reset_inference_flag(self):
        self.just_inferred = False

    def _get_event_description(self, conscious_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """意識に上った情報を簡潔なイベント記述に変換する。"""
        if not conscious_data:
            return None
        # (このメソッドの中身は前回と同じ)
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
        return "general_observation"

    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    def _get_context_description(self) -> str:
        """現在の認知的な文脈を記述する文字列を生成する。"""
        # GlobalWorkspaceからPFC(前頭前野)の現在の目標を取得
        pfc_goal = self.workspace.get_information("prefrontal_cortex_goal") # 仮のAPI
        if pfc_goal and isinstance(pfc_goal, str):
            if "boredom" in pfc_goal:
                return "reducing_boredom"
            if "curiosity" in pfc_goal:
                return "satisfying_curiosity"
        return "general_context"

    def handle_conscious_broadcast(self, source: str, conscious_data: Dict[str, Any]):
        """
        意識に上った情報の連鎖と、その時の文脈を観察し、因果関係を推論する。
        """
        current_event = self._get_event_description(conscious_data)
        previous_event = self._get_event_description(self.previous_conscious_info)
        current_context = self._get_context_description()

        if previous_event and current_event and self.previous_context:
            # (文脈, 原因, 結果) の三つ組で共起をカウント
            event_tuple = (self.previous_context, previous_event, current_event)
            self.co_occurrence_counts[event_tuple] += 1
            
            count = self.co_occurrence_counts[event_tuple]
            print(f"  - 因果推論: イベント組観測 -> ({self.previous_context}, {previous_event}, {current_event}), 回数: {count}")

            # 閾値に達したら、文脈付きの因果関係として記録
            if count == self.inference_threshold:
                print(f"  - 🔥 因果関係を推論・記録！")
                self.rag_system.add_causal_relationship(
                    cause=previous_event,
                    effect=current_event,
                    condition=self.previous_context
                )
                self.just_inferred = True
        
        self.previous_conscious_info = conscious_data
        self.previous_context = current_context
    # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---
