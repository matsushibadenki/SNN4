# ファイルパス: snn_research/cognitive_architecture/hierarchical_planner.py
# (修正)
# 修正点: PlannerSNNに渡すプロンプトが長くなりすぎる問題を修正。
#         - RAGで取得した知識を要約・切り詰める処理を追加。
#         - トークナイザ呼び出し時にtruncation=Trueを指定し、入力をモデルの最大長に制限。

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer
import asyncio

from .planner_snn import PlannerSNN
from snn_research.distillation.model_registry import ModelRegistry
from .rag_snn import RAGSystem

class Plan:
    """
    タスクのシーケンスを表現するクラス。
    """
    def __init__(self, goal: str, task_list: List[Dict[str, Any]]):
        self.goal = goal
        self.task_list = task_list

    def __repr__(self) -> str:
        return f"Plan(goal='{self.goal}', tasks={len(self.task_list)})"


class HierarchicalPlanner:
    """
    高レベルの目標をサブタスクに分解する階層型プランナー。
    PlannerSNNとRAGSystemを内部で利用して、動的に計画を生成する。
    """
    def __init__(
        self,
        model_registry: ModelRegistry,
        rag_system: RAGSystem,
        planner_model: Optional[PlannerSNN] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu"
    ):
        self.model_registry = model_registry
        self.rag_system = rag_system
        self.planner_model = planner_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if hasattr(self.tokenizer, 'model_max_length'):
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = 1024
        
        self.device = device
        if self.planner_model:
            self.planner_model.to(self.device)

        self.SKILL_MAP: Dict[int, Dict[str, Any]] = asyncio.run(self._build_skill_map())
        print(f"🧠 Planner initialized with {len(self.SKILL_MAP)} skills from the registry.")

    async def _build_skill_map(self) -> Dict[int, Dict[str, Any]]:
        """モデルレジストリから動的にスキルマップを構築する"""
        all_models = await self.model_registry.list_models()
        skill_map: Dict[int, Dict[str, Any]] = {}
        fallback_skill: Dict[str, Any] = {
            "task": "general_qa", 
            "description": "Answer a general question.", 
            "expert_id": "general_snn_v3"
        }
        
        for i, model_info in enumerate(all_models):
            skill_map[i] = {
                "task": model_info.get("model_id"),
                "description": model_info.get("task_description"),
                "expert_id": model_info.get("model_id")
            }
        
        if not any(skill['task'] == 'general_qa' for skill in skill_map.values()):
            skill_map[len(skill_map)] = fallback_skill
            
        return skill_map

    async def create_plan(self, high_level_goal: str, context: Optional[str] = None) -> Plan:
        """
        目標に基づいて計画を作成する。PlannerSNNが利用可能であればそれを使用する。
        RAGシステムのナレッジグラフを活用して、記号推論に基づいた計画を試みる。
        """
        print(f"🌍 Creating plan for goal: {high_level_goal}")

        self.SKILL_MAP = await self._build_skill_map()

        if self.planner_model and len(self.SKILL_MAP) > 0:
            knowledge_query = f"Find concepts and relations for: {high_level_goal}"
            retrieved_knowledge = self.rag_system.search(knowledge_query, k=3)
            
            # 取得した知識が長くなりすぎないように要約・切り詰め
            knowledge_summary = " ".join(doc[:250] + "..." for doc in retrieved_knowledge)
            if len(knowledge_summary) > 800:
                knowledge_summary = knowledge_summary[:800] + "..."

            full_prompt = f"Goal: {high_level_goal}\n\nRetrieved Knowledge:\n{knowledge_summary}"
            if context:
                full_prompt += f"\n\nUser Provided Context:\n{context}"
            
            print(f"🧠 PlannerSNN is reasoning with prompt: {full_prompt[:250]}...")
            
            self.planner_model.eval()
            with torch.no_grad():
                # truncation=True を指定して、入力をモデルの最大長に制限する
                inputs = self.tokenizer(
                    full_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = inputs['input_ids'].to(self.device)
                skill_logits, _, _ = self.planner_model(input_ids)
                predicted_skill_id = int(torch.argmax(skill_logits, dim=-1).item())
                
                if predicted_skill_id in self.SKILL_MAP:
                    task = self.SKILL_MAP[predicted_skill_id]
                    task_list = [task]
                    print(f"🧠 PlannerSNN predicted skill ID: {predicted_skill_id} -> Task: {task.get('task')}")
                else:
                    print(f"⚠️ PlannerSNN predicted an invalid skill ID: {predicted_skill_id}. Falling back to causal planning.")
                    task_list = self._create_causal_plan(high_level_goal)
        else:
            print("⚠️ PlannerSNN not available. Attempting causal planning...")
            task_list = self._create_causal_plan(high_level_goal)

        if not task_list:
            print("⚠️ Causal planning failed. Falling back to rule-based planning.")
            task_list = self._create_rule_based_plan(high_level_goal)

        print(f"✅ Plan created with {len(task_list)} step(s).")
        return Plan(goal=high_level_goal, task_list=task_list)

    def _create_causal_plan(self, high_level_goal: str) -> List[Dict[str, Any]]:
        """
        ナレッジグラフ（RAGSystem）を検索し、因果関係を辿って計画を推論する。
        """
        print(f"🔍 Inferring plan from knowledge graph for: {high_level_goal}")
        task_list = []
        
        query = f"Goal: {high_level_goal}. Find skills or actions that achieve this."
        retrieved_docs = self.rag_system.search(query, k=3)
        
        available_skills = list(self.SKILL_MAP.values())

        for doc in retrieved_docs:
            for skill in available_skills:
                skill_name = (skill.get('task') or '').lower()
                if skill_name and skill_name in doc.lower():
                    if skill not in task_list:
                        print(f"  - Found relevant skill from KG: {skill_name}")
                        task_list.append(skill)
                        return task_list
        
        print("  - No direct causal path found in the knowledge graph.")
        return []

    def _create_rule_based_plan(self, prompt: str) -> List[Dict[str, Any]]:
        """ルールベースで簡易的な計画を作成するフォールバックメソッド。"""
        task_list = []
        prompt_lower = prompt.lower()
        
        available_skills = list(self.SKILL_MAP.values())
        
        for skill in available_skills:
            task_keywords = (skill.get('task') or '').lower().split('_')
            desc_keywords = (skill.get('description') or '').lower().split()
            
            if any(kw in prompt_lower for kw in task_keywords if kw) or any(kw in prompt_lower for kw in desc_keywords if kw):
                 if skill not in task_list:
                    task_list.append(skill)

        if not task_list:
            fallback_skill = next((s for s in available_skills if "general" in (s.get("task") or "")), None)
            if fallback_skill:
                task_list.append(fallback_skill)
        
        return task_list

    async def refine_plan(self, failed_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        失敗したタスクの代替案（協力者）を提案する。
        """
        task_desc = failed_task.get("description", "")
        print(f"🤔 Refining plan for failed task: {task_desc}")

        alternative_experts = await self.model_registry.find_models_for_task(str(task_desc), top_k=5)

        original_expert_id = failed_task.get("expert_id")
        for expert in alternative_experts:
            if expert.get("model_id") != original_expert_id:
                print(f"✅ Found alternative expert: {expert['model_id']}")
                new_task: Dict[str, Any] = failed_task.copy()
                new_task["expert_id"] = expert["model_id"]
                new_task["description"] = expert["task_description"]
                return new_task
        
        print("❌ No alternative expert found.")
        return None

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        """
        タスク要求を受け取り、計画立案から実行までを行う。
        """
        print(f"Executing task: {task_request} with context: {context}")
        
        plan = asyncio.run(self.create_plan(task_request, context))
        
        if plan.task_list:
            final_result = f"Plan for '{task_request}':\n"
            for i, task in enumerate(plan.task_list):
                final_result += f"  Step {i+1}: Execute '{task.get('task')}' using expert '{task.get('expert_id')}'.\n"
            final_result += "Task completed successfully (dummy execution)."
            return final_result
        else:
            return "Could not create a plan for the given task."