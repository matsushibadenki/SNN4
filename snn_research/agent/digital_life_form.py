# ファイルパス: snn_research/agent/digital_life_form.py
# (更新)
# 改善点:
# - DigitalLifeFormがHierarchicalPlannerを直接利用して、
#   高レベルの目標から具体的な行動計画を立案・実行するように修正。
# - `_decide_next_action`を`_formulate_goal`に改名し、自然言語の目標を生成するようにした。
# - `life_cycle_step`で、生成された計画をループ実行するロジックを実装。

import time
import logging
import torch
import random
import json
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import operator
import os

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
# HierarchicalPlannerをインポート
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

if TYPE_CHECKING:
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter
    from snn_research.training.bio_trainer import BioRLTrainer
    from snn_research.rl_env.grid_world import GridWorldEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    プランナーと連携し、目標に基づいた計画を実行する、進化したオーケストレーター。
    """
    def __init__(
        self,
        planner: HierarchicalPlanner,
        autonomous_agent: AutonomousAgent,
        rl_agent: ReinforcementLearnerAgent,
        self_evolving_agent: SelfEvolvingAgent,
        motivation_system: IntrinsicMotivationSystem,
        meta_cognitive_snn: MetaCognitiveSNN,
        memory: Memory,
        physics_evaluator: PhysicsEvaluator,
        symbol_grounding: SymbolGrounding,
        langchain_adapter: "SNNLangChainAdapter",
        global_workspace: GlobalWorkspace
    ):
        # --- ▼ 修正 ▼ ---
        self.planner = planner
        # --- ▲ 修正 ▲ ---
        self.autonomous_agent = autonomous_agent
        self.rl_agent = rl_agent
        self.self_evolving_agent = self_evolving_agent
        self.motivation_system = motivation_system
        self.meta_cognitive_snn = meta_cognitive_snn
        self.memory = memory
        self.physics_evaluator = physics_evaluator
        self.symbol_grounding = symbol_grounding
        self.langchain_adapter = langchain_adapter
        self.workspace = global_workspace
        
        self.running = False
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None, "last_task": "unknown"}

    # ... (start, stop, life_cycleメソッドは変更なし) ...
    def start(self): self.running = True; logging.info("DigitalLifeForm activated."); self.life_cycle()
    def stop(self): self.running = False; logging.info("DigitalLifeForm deactivating.")
    def life_cycle(self):
        while self.running: self.life_cycle_step(); time.sleep(10)

    # --- ▼ 修正 ▼ ---
    def life_cycle_step(self):
        """計画主導の認知サイクルを実行する。"""
        logging.info("\n--- 🧠 New Cognitive Cycle ---")
        self._handle_causal_credit()

        # 1. 内部状態を評価し、高レベルの目標を策定する
        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()
        goal = self._formulate_goal(internal_state, performance_eval)
        logging.info(f"🎯 New Goal: {goal}")
        
        # 2. プランナーに目標を渡し、行動計画を立案させる
        plan = asyncio.run(self.planner.create_plan(goal))

        # 3. 計画されたタスクを順番に実行する
        if not plan.task_list:
            logging.warning("Planner could not create a plan. Idling for this cycle.")
            return

        logging.info(f"📋 Plan Created: {[task.get('task') for task in plan.task_list]}")
        for task in plan.task_list:
            action = task.get('task')
            if not action: continue

            logging.info(f"▶️ Executing task from plan: {action}")
            result, reward, expert_used = self._execute_action(action, internal_state)

            # 4. 各ステップの結果を記録・評価する
            if isinstance(result, dict): self.symbol_grounding.process_observation(result, context=f"action '{action}'")
            reward_vector = {"external": reward, "curiosity": internal_state.get("curiosity", 0.0)}
            decision_context = {"goal": goal, "plan": [t.get('task') for t in plan.task_list]}
            self.memory.record_experience(self.state, action, result, reward, expert_used, decision_context)
            
            # (簡易的な動機更新)
            self.motivation_system.update_metrics(random.random(), 1.0 if reward > 0 else 0.0, random.random(), random.random())

            self.state["last_action"] = action; self.state["last_result"] = result
            logging.info(f"  - Task Result: {str(result)[:100]}, Reward: {reward:.2f}")

            if reward < 0:
                logging.warning(f"  - Task '{action}' failed. Aborting current plan.")
                break # 計画の途中で失敗したら中止

    def _formulate_goal(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        """内部状態とパフォーマンス評価から、自然言語の目標を生成する。"""
        if internal_state.get("curiosity", 0.0) > 0.8 and internal_state.get("curiosity_context"):
            topic = internal_state.get("curiosity_context")
            return f"Explore the unknown concept related to '{str(topic)}' to satisfy curiosity."
        
        if performance_eval.get("status") == "capability_gap":
            return "Evolve my architecture to overcome a capability gap."

        if internal_state.get("boredom", 0.0) > 0.7:
            return "Explore a completely new and random task to alleviate boredom."

        return "Practice an existing skill to improve confidence and performance."

    # --- ▼ 修正 ▼ ---
    def _handle_causal_credit(self):
        """GlobalWorkspaceを監視し、因果的クレジット信号があれば処理する。"""
        # Workspaceから意識に上った最新の情報を取得
        conscious_content = self.workspace.conscious_broadcast_content
        
        if conscious_content and isinstance(conscious_content, dict) and conscious_content.get("type") == "causal_credit":
            target_action = conscious_content.get("target_action")
            credit = conscious_content.get("credit", 0.0)
            
            print(f"✨ 因果的クレジット信号を検知！ Target: {target_action}, Credit: {credit}")

            # 直近の行動がクレジットの対象と一致するか確認
            if self.state.get("last_action") and target_action == f"action_{self.state['last_action']}":
                print(f"  - 直近の行動 '{self.state['last_action']}' にクレジットを割り当てます。")
                
                # ここで、クレジットを適切なエージェントの学習則に渡す
                # この例では、RLエージェントが対象だと仮定
                if self.state['last_action'] in ["explore_new_task_with_rl", "practice_skill_with_rl"]:
                    # learnメソッドに特別なパラメータを渡す
                    self.rl_agent.learn(reward=0.0, causal_credit=credit)
                    print("  - RLエージェントのシナプス可塑性をトップダウンで変調しました。")

    def _decide_next_action(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        # (変更なし)
        action_scores: Dict[str, float] = { "explore_curiosity": internal_state.get("curiosity", 0.0) * 20.0, "evolve_architecture": 0.0, "practice_skill_with_rl": internal_state.get("confidence", 0.5) * 2.0 }
        if performance_eval.get("status") == "capability_gap": action_scores["evolve_architecture"] += 10.0
        if internal_state.get("boredom", 0.0) > 0.8: action_scores["explore_curiosity"] += internal_state.get("boredom", 0.0) * 15.0
        chosen_action = max(action_scores.items(), key=operator.itemgetter(1))[0]
        logging.info(f"Action scores: {action_scores} -> Chosen: {chosen_action}")
        return chosen_action
    # --- ▲ 修正 ▲ ---

    def _execute_action(self, action: str, internal_state: Dict[str, Any]) -> tuple[Dict[str, Any], float, List[str]]:
        # (変更なし)
        from snn_research.rl_env.grid_world import GridWorldEnv
        from snn_research.training.bio_trainer import BioRLTrainer
        try:
            if action == "explore_curiosity":
                curiosity_topic = internal_state.get("curiosity_context")
                if not curiosity_topic: return {"status": "skipped", "info": "No specific curiosity context found."}, 0.0, []
                topic_str = str(curiosity_topic.get("action", "AI concept"))
                logging.info(f"🔬 Curiosity triggered! Researching topic: '{topic_str}'")
                new_model_info = asyncio.run(self.autonomous_agent.handle_task(task_description=topic_str, unlabeled_data_path="data/sample_data.jsonl", force_retrain=True))
                if new_model_info: return {"status": "success", "info": f"Learned about '{topic_str}' and created new expert.", "model_info": new_model_info}, 1.0, ["autonomous_agent"]
                else: return {"status": "failure", "info": f"Failed to learn about '{topic_str}'."}, -0.5, ["autonomous_agent"]
            elif action == "evolve_architecture":
                return {"status": "success", "info": self.self_evolving_agent.evolve()}, 0.9, ["self_evolver"]
            elif action == "practice_skill_with_rl":
                env = GridWorldEnv(size=5, max_steps=20, device=self.rl_agent.device)
                trainer = BioRLTrainer(agent=self.rl_agent, env=env)
                res = trainer.train(num_episodes=10)
                return {"status": "success", "results": res}, res.get("final_average_reward", 0.0), ["rl_agent"]
            else: return {"status": "idle", "info": "No compelling action to take."}, 0.0, []
        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}")
            return {"status": "error", "info": str(e)}, -1.0, []

    def awareness_loop(self, cycles: int):
        # (変更なし)
        print(f"🧬 Digital Life Form awareness loop starting for {cycles} cycles.")
        self.running = True
        for i in range(cycles):
            if not self.running: break
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            self.life_cycle_step()
            time.sleep(2)
        print("🧬 Awareness loop finished.")

    def explain_last_action(self) -> Optional[str]:
        try:
            with open(self.memory.memory_path, "rb") as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n': f.seek(-2, os.SEEK_CUR)
                except OSError: f.seek(0)
                last_line = f.readline().decode()
            last_experience = json.loads(last_line)
        except (IOError, json.JSONDecodeError, IndexError):
            return "行動履歴が見つかりません。"

        prompt = f"""
        あなたは、自身の行動を分析し、その理由を分かりやすく説明するAIです。
        以下の内部ログは、あなた自身の直近の行動記録です。この記録を基に、なぜその行動を取ったのかを一人称（「私」）で説明してください。

        ### 行動ログ
        - **実行した行動:** {last_experience.get('action')}
        - **意思決定の根拠:**
          - **内発的動機（内部状態）:** {last_experience.get('decision_context', {}).get('internal_state')}
          - **自己パフォーマンス評価:** {last_experience.get('decision_context', {}).get('performance_eval')}
          - **物理効率評価:** {last_experience.get('decision_context', {}).get('physical_rewards')}

        ### 指示
        上記の根拠を統合し、あなたの思考プロセスを平易な言葉で説明してください。
        """
        try:
            snn_llm = self.langchain_adapter
            explanation = snn_llm._call(prompt)
            return explanation
        except Exception as e:
            logging.error(f"LLMによる自己言及の生成に失敗しました: {e}")
            return "エラー: 自己言及の生成に失敗しました。"
