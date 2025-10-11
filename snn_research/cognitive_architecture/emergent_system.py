# ファイルパス: snn_research/cognitive_architecture/emergent_system.py
#
# Title: 創発システム
#
# Description: 異なる認知コンポーネント間の相互作用を管理し、創発的な振る舞いを引き出すシステム。
#              mypyエラー修正: ModelRegistryの具象クラスをDIで受け取るように変更。
#              mypyエラー修正: 非同期メソッド呼び出しにawaitを追加。
#              循環インポートエラー修正: TYPE_CHECKINGを使用して型ヒントのみインポートする。
#
# 改善点:
# - ROADMAPフェーズ8に基づき、エージェント間の協調行動を実装。
# - タスク失敗時にプランナーに代替案を問い合わせ、別のエージェントに
#   タスクを再割り当てするロジックを追加。
#
# 修正点:
# - mypyエラー解消のため、`random`モジュールをインポート。
# - mypyエラー解消のため、`expert_id`がNoneの場合の`in`演算子の使用を修正。
#
# 改善点 (v2):
# - ROADMAPフェーズ8「協調的タスク解決」をさらに強化。
# - タスク失敗時に、他のエージェントがより高性能なモデルを所有しているか検索し、
#   最適な協力者にタスクを再割り当てする`_find_collaborator_for_task`を実装。
#
# 改善点 (v3):
# - ダミーのタスク実行を、実際の`agent.handle_task`呼び出しに置き換え。
# - これにより、エージェントの実際の能力に基づいてタスクの成否が決定され、
#   協調行動がより現実的なシナリオでトリガーされるようになった。
#
# 修正点 (v4):
# - mypyエラー `Item "None" of "dict[str, Any] | None" has no attribute "get"` を修正。
#   `execution_result` がNoneでないことを明示的にチェックする処理を追加。

import asyncio
from typing import List, Dict, Any, TYPE_CHECKING, Optional, Tuple
import random

from .global_workspace import GlobalWorkspace
from .hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry

# --- 循環インポート解消のための修正 ---
if TYPE_CHECKING:
    from snn_research.agent.autonomous_agent import AutonomousAgent


class EmergentCognitiveSystem:
    """
    複数の認知コンポーネントを統合し、協調させることで
    創発的な高次機能を実現するシステム。
    """

    def __init__(self, planner: HierarchicalPlanner, agents: List['AutonomousAgent'], global_workspace: GlobalWorkspace, model_registry: ModelRegistry):
        self.planner = planner
        self.agents = {agent.name: agent for agent in agents}
        self.global_workspace = global_workspace
        self.model_registry = model_registry

    def execute_task(self, high_level_goal: str) -> str:
        """
        高レベルの目標を受け取り、計画、実行、情報統合のサイクルを実行する。
        """
        return asyncio.run(self.execute_task_async(high_level_goal))

    async def _find_collaborator_for_task(self, failed_task: Dict[str, Any], failed_agent: 'AutonomousAgent') -> Optional[Tuple[str, Dict[str, Any]]]:
        """失敗したタスクに対して、より優れた能力を持つ協力者エージェントを探す。"""
        task_desc = failed_task.get("description", "")
        alternative_experts = await self.model_registry.find_models_for_task(str(task_desc), top_k=5)

        # 現在の専門家モデルの性能を取得
        original_expert_id = failed_task.get("expert_id")
        original_expert_info = await self.model_registry.get_model_info(original_expert_id) if original_expert_id else None
        original_performance = original_expert_info.get("metrics", {}).get("accuracy", 0.0) if original_expert_info else 0.0

        best_collaborator: Optional[str] = None
        best_new_task: Optional[Dict[str, Any]] = None
        best_performance = original_performance

        # 他のエージェントが持つ、より優れた専門家を探す
        for agent_name, agent in self.agents.items():
            if agent.name == failed_agent.name:
                continue # 自分自身は除外

            # このエージェントが利用可能な専門家モデルの中から探す（ここでは簡略化のため全レジストリを検索）
            for expert in alternative_experts:
                expert_performance = expert.get("metrics", {}).get("accuracy", 0.0)
                if expert.get("model_id") != original_expert_id and expert_performance > best_performance:
                    best_performance = expert_performance
                    best_collaborator = agent_name
                    new_task: Dict[str, Any] = failed_task.copy()
                    new_task["expert_id"] = expert["model_id"]
                    new_task["description"] = expert["task_description"]
                    best_new_task = new_task

        if best_collaborator and best_new_task:
            print(f"✅ Collaborator found: Agent '{best_collaborator}' has a better model ('{best_new_task['expert_id']}') with performance {best_performance:.4f}.")
            return best_collaborator, best_new_task

        print("❌ No better collaborator found.")
        return None


    async def execute_task_async(self, high_level_goal: str) -> str:
        """非同期でタスク実行サイクルを処理する。協調的再計画ロジックを含む。"""
        print(f"--- Emergent System: Executing Goal: {high_level_goal} ---")

        # 1. 初期計画の作成
        plan = await self.planner.create_plan(high_level_goal)
        self.global_workspace.broadcast("plan", f"New plan created: {plan.task_list}")

        # 2. 計画の実行
        results = []
        task_queue = plan.task_list.copy()
        
        # どのエージェントにタスクを割り当てるかのキュー（名前）
        agent_assignment_queue: List[Optional[str]] = [None] * len(task_queue)


        while task_queue:
            task = task_queue.pop(0)
            assigned_agent_name = agent_assignment_queue.pop(0)
            
            # 優先的に割り当てられたエージェントがいるか確認
            if assigned_agent_name and assigned_agent_name in self.agents:
                agent = self.agents[assigned_agent_name]
            else:
                # デフォルトのエージェント選択ロジック
                agent = random.choice(list(self.agents.values()))

            if not agent:
                error_msg = f"No agent available for task '{task.get('description')}'."
                results.append(error_msg)
                continue

            task_description = task.get("description", "")
            print(f"-> Assigning task '{task_description}' to agent '{agent.name}'")
            
            # 実際のタスク実行を試みる
            # 成功すればモデル情報が、失敗すればNoneが返る
            execution_result = await agent.handle_task(
                task_description=task_description,
                # 協力時に学習データがない場合もあるため、ここではNoneを渡す
                unlabeled_data_path=None, 
                force_retrain=False
            )
            
            is_success = execution_result is not None
            
            if is_success:
                # Mypyエラー修正: is_successチェック後でもexecution_resultがNoneの可能性があると判断されるため、
                # 明示的なチェックを追加してエラーを回避する。
                expert_id = execution_result.get('model_id', 'unknown') if execution_result else 'unknown'
                result = f"SUCCESS: Task '{task_description}' completed by '{agent.name}' using expert '{expert_id}'."
                results.append(result)
                self.global_workspace.broadcast(agent.name, result)
            else:
                # --- 協調行動: タスク失敗と協力者の探索 ---
                result = f"FAILURE: Task '{task_description}' failed by '{agent.name}' (no suitable expert found)."
                results.append(result)
                self.global_workspace.broadcast(agent.name, result)
                
                print(f"!! Task failed. Attempting to find a collaborator...")
                collaboration_proposal = await self._find_collaborator_for_task(task, agent)
                
                if collaboration_proposal:
                    collaborator_name, new_task = collaboration_proposal
                    print(f"++ Collaboration proposed! Re-assigning task to agent '{collaborator_name}'.")
                    task_queue.insert(0, new_task) # 新しいタスクをキューの先頭に追加
                    agent_assignment_queue.insert(0, collaborator_name) # 次の実行者を指定
                else:
                    print("-- No collaborator found. Aborting this task branch.")

        # 3. 統合と要約
        final_report = self._synthesize_results(results)
        self.global_workspace.broadcast("system", f"Goal '{high_level_goal}' completed. Final report generated.")
        print(f"--- Emergent System: Goal Execution Finished ---")
        return final_report

    def _synthesize_results(self, results: List[str]) -> str:
        """
        各エージェントからの結果を統合し、最終的なレポートを生成する。
        """
        report = "Execution Summary:\n"
        for i, res in enumerate(results):
            report += f"- Step {i+1}: {res}\n"
        return report