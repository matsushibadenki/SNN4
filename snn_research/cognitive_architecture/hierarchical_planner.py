# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# (修正)
# 改善点:
# - タスク失敗時に因果的記憶を検索し、失敗原因を特定して計画を修正する
#   `refine_plan_after_failure`メソッドを実装。

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
import asyncio

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem
# Memoryクラスをインポート
from snn_research.agent.memory import Memory

class Plan:
    # (変更なし)
    def __init__(self, goal: str, task_list: List[Dict[str, Any]]):
        self.goal = goal
        self.task_list = task_list
    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', tasks={len(self.task_list)})"

class HierarchicalPlanner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        rag_system: RAGSystem,
        # Memoryを依存性として追加
        memory: Memory,
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.rag_system = rag_system
        # Memoryインスタンスを保持
        self.memory = memory
        self.planner_model = planner_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = getattr(self.tokenizer, 'model_max_length', 1024)
        self.device = device
        if self.planner_model: self.planner_model.to(self.device)
        self.SKILL_MAP: Dict[int, Dict[str, Any]] = asyncio.run(self._build_skill_map())
        print(f"🧠 Planner initialized with {len(self.SKILL_MAP)} skills and Causal Memory access.")

    # _build_skill_map, _create_rule_based_plan, execute_task は変更なし
    async def _build_skill_map(self) -> Dict[int, Dict[str, Any]]:
        all_models = await self.model_registry.list_models()
        skill_map: Dict[int, Dict[str, Any]] = {}
        for i, model_info in enumerate(all_models):
            skill_map[i] = {"task": model_info.get("model_id"), "description": model_info.get("task_description"), "expert_id": model_info.get("model_id")}
        if not any(skill['task'] == 'general_qa' for skill in skill_map.values()):
            skill_map[len(skill_map)] = {"task": "general_qa", "description": "Answer a general question.", "expert_id": "general_snn_v3"}
        return skill_map

    def _create_rule_based_plan(self, prompt: str, skills_to_avoid: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if skills_to_avoid is None: skills_to_avoid = []
        task_list = []
        prompt_lower = prompt.lower()
        available_skills = [s for s in self.SKILL_MAP.values() if s.get('task') not in skills_to_avoid]
        for skill in available_skills:
            task_keywords = (skill.get('task') or '').lower().split('_')
            desc_keywords = (skill.get('description') or '').lower().split()
            if any(kw in prompt_lower for kw in task_keywords if kw) or any(kw in prompt_lower for kw in desc_keywords if kw):
                if skill not in task_list: task_list.append(skill)
        if not task_list and not skills_to_avoid:
            fallback = next((s for s in available_skills if "general" in (s.get("task") or "")), None)
            if fallback: task_list.append(fallback)
        return task_list

    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    async def create_plan(
        self,
        high_level_goal: str,
        context: Optional[str] = None,
        skills_to_avoid: Optional[List[str]] = None
    ) -> Plan:
        """
        目標に基づいて計画を作成する。避けるべきスキルのリストを考慮に入れる。
        """
        if skills_to_avoid is None: skills_to_avoid = []
        print(f"🌍 Creating plan for goal: {high_level_goal}, avoiding skills: {skills_to_avoid}")
        self.SKILL_MAP = await self._build_skill_map()
        
        # (既存の計画ロジックはルールベースに簡略化し、避けるべきスキルをフィルタリング)
        task_list = self._create_rule_based_plan(high_level_goal, skills_to_avoid)

        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)

    async def refine_plan_after_failure(
        self,
        failed_plan: Plan,
        failed_task: Dict[str, Any]
    ) -> Optional[Plan]:
        """
        タスク失敗後、因果的記憶を検索して計画を練り直す。
        """
        print(f"🤔 Task '{failed_task.get('task')}' failed. Refining plan using causal memory...")
        
        # 1. 失敗の因果関係をクエリとして作成
        causal_query = f"The action '{failed_task.get('task')}' resulted in a failure while pursuing the goal '{failed_plan.goal}'."

        # 2. 因果的記憶を検索
        similar_failures = self.memory.retrieve_similar_experiences(causal_query=causal_query, top_k=3)

        skills_to_avoid = {failed_task.get('task')}

        # 3. 過去の失敗から学ぶ
        if similar_failures:
            print("  - Found similar past failures. Analyzing causes...")
            for failure in similar_failures:
                # 記録された因果関係テキストから、原因となった行動を抽出
                retrieved_text = failure.get("retrieved_causal_text", "")
                if "leads to the effect 'failure'" in retrieved_text:
                    parts = retrieved_text.split("'")
                    if len(parts) >= 4:
                        cause_event = parts[3]
                        if cause_event.startswith("action_"):
                            failed_action = cause_event.replace("action_", "")
                            print(f"    - Past data suggests that action '{failed_action}' often leads to failure in this context.")
                            skills_to_avoid.add(failed_action)

        # 4. 失敗原因を避けて新しい計画を立案
        print(f"  - Attempting to create a new plan avoiding: {list(skills_to_avoid)}")
        new_plan = await self.create_plan(
            high_level_goal=failed_plan.goal,
            skills_to_avoid=list(skills_to_avoid)
        )

        # 新しい計画が作成できたか、または元の計画と異なるか確認
        if new_plan.task_list and new_plan.task_list != failed_plan.task_list:
            print("✅ Successfully created a revised plan.")
            return new_plan
        else:
            print("❌ Could not find a viable alternative plan.")
            return None
    # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

    # (execute_taskはデモ用に簡略化)
    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        plan = asyncio.run(self.create_plan(task_request, context))
        if plan.task_list:
            # (ここではダミーで最初のタスクが失敗したと仮定)
            failed_task = plan.task_list[0]
            print(f"\n--- [SIMULATION] Task '{failed_task.get('task')}' is assumed to have failed. ---")
            
            # 失敗を受けて計画を練り直す
            new_plan = asyncio.run(self.refine_plan_after_failure(plan, failed_task))
            
            if new_plan:
                return f"Original plan failed. Revised plan: {new_plan.task_list}"
            else:
                return "Original plan failed and no alternative was found."
        return "Could not create an initial plan."
