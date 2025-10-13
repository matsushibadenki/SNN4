# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# (修正)
# 改善点(v2):
# - 因果推論エンジンの導入に伴い、計画立案時に因果知識グラフを検索するロジックを追加。

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
import asyncio

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem

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
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        # (変更なし)
        self.model_registry = model_registry
        self.rag_system = rag_system
        self.planner_model = planner_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = getattr(self.tokenizer, 'model_max_length', 1024)
        self.device = device
        if self.planner_model: self.planner_model.to(self.device)
        self.SKILL_MAP: Dict[int, Dict[str, Any]] = asyncio.run(self._build_skill_map())
        print(f"🧠 Planner initialized with {len(self.SKILL_MAP)} skills.")

    async def _build_skill_map(self) -> Dict[int, Dict[str, Any]]:
        # (変更なし)
        all_models = await self.model_registry.list_models()
        skill_map: Dict[int, Dict[str, Any]] = {}
        for i, model_info in enumerate(all_models):
            skill_map[i] = {"task": model_info.get("model_id"), "description": model_info.get("task_description"), "expert_id": model_info.get("model_id")}
        if not any(skill['task'] == 'general_qa' for skill in skill_map.values()):
            skill_map[len(skill_map)] = {"task": "general_qa", "description": "Answer a general question.", "expert_id": "general_snn_v3"}
        return skill_map

    async def create_plan(self, high_level_goal: str, context: Optional[str] = None) -> Plan:
        print(f"🌍 Creating plan for goal: {high_level_goal}")
        self.SKILL_MAP = await self._build_skill_map()
        task_list: List[Dict[str, Any]] = []

        # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
        # 1. 因果知識グラフに基づく推論的計画 (最優先)
        task_list = self._create_causal_plan(high_level_goal)
        if task_list:
            print(f"✅ Plan created with {len(task_list)} step(s) based on causal inference.")
            return Plan(goal=high_level_goal, task_list=task_list)
        # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

        # 2. PlannerSNNによる学習ベースの計画 (次善)
        if self.planner_model and len(self.SKILL_MAP) > 0:
            knowledge_query = f"Find concepts and relations for: {high_level_goal}"
            retrieved_knowledge = self.rag_system.search(knowledge_query, k=3)
            knowledge_summary = " ".join(doc[:250] + "..." for doc in retrieved_knowledge)
            if len(knowledge_summary) > 800: knowledge_summary = knowledge_summary[:800] + "..."
            full_prompt = f"Goal: {high_level_goal}\n\nKnowledge:\n{knowledge_summary}"
            if context: full_prompt += f"\n\nContext:\n{context}"
            
            print(f"🧠 PlannerSNN is reasoning...")
            self.planner_model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                input_ids = inputs['input_ids'].to(self.device)
                skill_logits, _, _ = self.planner_model(input_ids)
                predicted_skill_id = int(torch.argmax(skill_logits, dim=-1).item())
                
                if predicted_skill_id in self.SKILL_MAP:
                    task = self.SKILL_MAP[predicted_skill_id]
                    task_list = [task]
                    print(f"🧠 PlannerSNN predicted skill: {task.get('task')}")
        
        # 3. ルールベースのフォールバック (最終手段)
        if not task_list:
            print("⚠️ Falling back to rule-based planning.")
            task_list = self._create_rule_based_plan(high_level_goal)

        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)

    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    def _create_causal_plan(self, high_level_goal: str) -> List[Dict[str, Any]]:
        """
        ナレッジグラフ（RAGSystem）を検索し、因果関係を辿って計画を推論する。
        """
        print(f"🔍 Inferring plan from causal knowledge graph for: {high_level_goal}")
        query = f"Find causal relation where the effect is related to the goal: {high_level_goal}"
        retrieved_docs = self.rag_system.search(query, k=3)
        
        available_skills = list(self.SKILL_MAP.values())

        for doc in retrieved_docs:
            if "Causal Relation:" in doc:
                # "event 'cause' leads to the effect 'effect'" の形式をパース
                parts = doc.split("'")
                if len(parts) >= 6:
                    cause_event = parts[3] 
                    effect_event = parts[5]
                    
                    # effectがゴールに関連し、causeが実行可能なスキルであれば計画に採用
                    if high_level_goal.lower() in effect_event.lower():
                        for skill in available_skills:
                            # スキル名は "action_..." の形式で記録されていると仮定
                            skill_action_name = f"action_{skill.get('task', '')}"
                            if skill_action_name == cause_event:
                                print(f"  - Found causal link: To achieve '{effect_event}', perform '{cause_event}'. Using skill '{skill.get('task')}'.")
                                return [skill]
        
        print("  - No direct causal path found in the knowledge graph.")
        return []
    # --- ◾️◾️◾️◾️◾️↑ここまでが重要↑◾️◾️◾️◾️◾️ ---

    def _create_rule_based_plan(self, prompt: str) -> List[Dict[str, Any]]:
        # (変更なし)
        task_list = []
        prompt_lower = prompt.lower()
        available_skills = list(self.SKILL_MAP.values())
        for skill in available_skills:
            task_keywords = (skill.get('task') or '').lower().split('_')
            desc_keywords = (skill.get('description') or '').lower().split()
            if any(kw in prompt_lower for kw in task_keywords if kw) or any(kw in prompt_lower for kw in desc_keywords if kw):
                 if skill not in task_list: task_list.append(skill)
        if not task_list:
            fallback = next((s for s in available_skills if "general" in (s.get("task") or "")), None)
            if fallback: task_list.append(fallback)
        return task_list

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        # (変更なし)
        plan = asyncio.run(self.create_plan(task_request, context))
        if plan.task_list:
            result = f"Plan for '{task_request}':\n"
            for i, task in enumerate(plan.task_list):
                result += f"  Step {i+1}: Execute '{task.get('task')}' using expert '{task.get('expert_id')}'.\n"
            return result + "Task completed (dummy execution)."
        return "Could not create a plan."
