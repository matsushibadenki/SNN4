# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# (修正)
# 修正: mypyエラー [annotation-unchecked] を解消するため、__init__に戻り値の型ヒントを追加。

from typing import Dict, Any

class PrefrontalCortex:
    """
    高レベルの目標設定と戦略選択を行う前頭前野モジュール。
    """
    def __init__(self) -> None:
        self.current_goal: str = "Explore and learn"  # デフォルトの目標
        print("🧠 前頭前野（実行制御）モジュールが初期化されました。")

    def decide_goal(self, system_context: Dict[str, Any]) -> str:
        """
        システム全体の文脈を評価し、次の高レベルな目標を決定する。

        Args:
            system_context (Dict[str, Any]):
                Global Workspaceから提供される、システム全体の現在の状態。
                例: {'internal_state': {'boredom': 0.8}, 'external_request': 'summarize'}

        Returns:
            str: 決定された新しい高レベル目標。
        """
        print("🤔 前頭前野: 次の目標を思考中...")

        # 外部からの明確な要求があれば、それを最優先する
        external_request = system_context.get("external_request")
        if external_request:
            self.current_goal = f"Fulfill external request: {external_request}"
            print(f"🎯 新目標（外部要求）: {self.current_goal}")
            return self.current_goal

        # 内発的動機に基づいて目標を決定する
        internal_state = system_context.get("internal_state", {})
        if internal_state.get("boredom", 0.0) > 0.7:
            self.current_goal = "Explore a new topic to reduce boredom"
            print(f"🎯 新目標（内発的動機）: {self.current_goal}")
            return self.current_goal
        
        if internal_state.get("curiosity", 0.0) > 0.6:
            self.current_goal = "Acquire new knowledge about an uncertain topic"
            print(f"🎯 新目標（内発的動機）: {self.current_goal}")
            return self.current_goal
            
        # 特に強い動機がなければ、既存の知識を整理・最適化する
        self.current_goal = "Organize and optimize existing knowledge"
        print(f"🎯 新目標（デフォルト）: {self.current_goal}")
        return self.current_goal