# ファイルパス: snn_research/agent/digital_life_form.py
# (更新)
#
# Title: DigitalLifeForm オーケストレーター
#
# 修正点 (v9):
# - 循環インポートエラーを解消するため、SNNLangChainAdapterのトップレベルインポートを削除し、
#   TYPE_CHECKINGとForward Reference（文字列による型指定）を使用するように修正。
#
# 実装更新 (v10):
# - _execute_actionメソッドの強化学習関連のダミー実装を、
#   実際にBioRLTrainerを呼び出して短期間の学習サイクルを実行する具体的なロジックに置き換え。

import time
import logging
import torch
import random
import json
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import DistributedModelRegistry
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.training.bio_trainer import BioRLTrainer


# 型チェック時のみインポートを実行する
if TYPE_CHECKING:
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
    """
    内発的動機付けシステムとメタ認知SNNを統合し、
    永続的で自己駆動する学習ループを実現するオーケストレーター。
    """
    def __init__(
        self,
        autonomous_agent: AutonomousAgent,
        rl_agent: ReinforcementLearnerAgent,
        self_evolving_agent: SelfEvolvingAgent,
        motivation_system: IntrinsicMotivationSystem,
        meta_cognitive_snn: MetaCognitiveSNN,
        memory: Memory,
        physics_evaluator: PhysicsEvaluator,
        symbol_grounding: SymbolGrounding,
        # Forward Reference (文字列) を使って型ヒントを記述
        langchain_adapter: "SNNLangChainAdapter"
    ):
        self.autonomous_agent = autonomous_agent
        self.rl_agent = rl_agent
        self.self_evolving_agent = self_evolving_agent
        self.motivation_system = motivation_system
        self.meta_cognitive_snn = meta_cognitive_snn
        self.memory = memory
        self.physics_evaluator = physics_evaluator
        self.symbol_grounding = symbol_grounding
        self.langchain_adapter = langchain_adapter
        
        self.running = False
        self.state: Dict[str, Any] = {"last_action": None, "last_result": None, "last_task": "unknown"}


    def start(self):
        self.running = True
        logging.info("DigitalLifeForm activated. Starting autonomous loop.")
        self.life_cycle()

    def stop(self):
        self.running = False
        logging.info("DigitalLifeForm deactivating.")

    def life_cycle(self):
        while self.running:
            self.life_cycle_step()
            time.sleep(10)
    
    def life_cycle_step(self):
        """life_cycleの1回分の処理"""
        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()
        dummy_mem_sequence = torch.randn(100)
        dummy_spikes = (torch.rand(100) > 0.8).float()
        physical_rewards = self.physics_evaluator.evaluate_physical_consistency(dummy_mem_sequence, dummy_spikes)
        
        action = self._decide_next_action(internal_state, performance_eval, physical_rewards)
        
        result, external_reward, expert_used = self._execute_action(action)

        if isinstance(result, dict):
            self.symbol_grounding.process_observation(result, context=f"action '{action}'")
        
        # 報酬ベクトルに好奇心を追加
        reward_vector = {
            "external": external_reward,
            "physical": physical_rewards,
            "curiosity": internal_state.get("curiosity", 0.0)
        }
        decision_context = {"internal_state": internal_state, "performance_eval": performance_eval, "physical_rewards": physical_rewards}
        self.memory.record_experience(self.state, action, result, reward_vector, expert_used, decision_context)
        
        dummy_prediction_error = result.get("prediction_error", 0.1) if isinstance(result, dict) else 0.1
        dummy_success_rate = result.get("success_rate", 0.9) if isinstance(result, dict) else 0.9
        dummy_task_similarity = 0.8
        dummy_loss = result.get("loss", 0.05) if isinstance(result, dict) else 0.05
        dummy_time = result.get("computation_time", 1.0) if isinstance(result, dict) else 1.0
        dummy_accuracy = result.get("accuracy", 0.95) if isinstance(result, dict) else 0.95

        self.motivation_system.update_metrics(dummy_prediction_error, dummy_success_rate, dummy_task_similarity, dummy_loss)
        self.meta_cognitive_snn.update_metadata(dummy_loss, dummy_time, dummy_accuracy)
        self.state["last_action"] = action
        self.state["last_result"] = result
        
        logging.info(f"Action: {action}, Result: {str(result)[:200]}, Reward Vector: {reward_vector}")
        logging.info(f"New Internal State: {self.motivation_system.get_internal_state()}")

    def _decide_next_action(self, internal_state: Dict[str, float], performance_eval: Dict[str, Any], physical_rewards: Dict[str, float]) -> str:
        action_scores: Dict[str, float] = {
            "acquire_new_knowledge": 0.0,
            "evolve_architecture": 0.0,
            "explore_new_task_with_rl": 0.0,
            "plan_and_execute": 0.0,
            "practice_skill_with_rl": 0.0,
            "publish_successful_skill": 0.0,
            "download_skill_from_community": 0.0,
        }

        if performance_eval.get("status") == "knowledge_gap":
            action_scores["download_skill_from_community"] += 20.0 
            logging.info("Decision reason: Knowledge gap detected. Prioritizing skill download.")
        else:
            action_scores["acquire_new_knowledge"] += 5.0

        if internal_state.get("confidence", 0.5) > 0.9:
            action_scores["publish_successful_skill"] += 10.0
            logging.info("Decision reason: High confidence. Considering publishing skill.")

        if performance_eval.get("status") == "capability_gap":
            action_scores["evolve_architecture"] += 5.0
            logging.info("Decision reason: Capability gap detected.")
        if physical_rewards.get("sparsity_reward", 1.0) < 0.5:
            action_scores["evolve_architecture"] += 8.0
            logging.info("Decision reason: Low energy efficiency (sparsity).")

        action_scores["explore_new_task_with_rl"] += internal_state.get("curiosity", 0.5) * 5.0
        action_scores["plan_and_execute"] += internal_state.get("curiosity", 0.5) * 3.0
        
        if internal_state.get("boredom", 0.0) > 0.7:
            action_scores["explore_new_task_with_rl"] += internal_state.get("boredom", 0.0) * 10.0
            logging.info("Decision reason: High boredom.")

        action_scores["practice_skill_with_rl"] += internal_state.get("confidence", 0.5) * 2.0
        action_scores["explore_new_task_with_rl"] += internal_state.get("confidence", 0.5) * 1.0

        action_scores["practice_skill_with_rl"] += 1.0

        actions = list(action_scores.keys())
        scores = [max(0, s) for s in action_scores.values()]
        total_score = sum(scores)

        if total_score == 0:
            chosen_action = "practice_skill_with_rl"
        else:
            probabilities = [s / total_score for s in scores]
            chosen_action = random.choices(actions, weights=probabilities, k=1)[0]
        
        logging.info(f"Action scores: {action_scores}")
        if total_score > 0:
            logging.info(f"Probabilities: { {a: f'{p:.2%}' for a, p in zip(actions, probabilities)} }")
        logging.info(f"Chosen action: {chosen_action}")

        return chosen_action

    def _execute_action(self, action: str) -> tuple[Dict[str, Any], float, List[str]]:
        """
        選択された行動に対応するエージェントの機能を実際に呼び出す。
        """
        try:
            if action == "publish_successful_skill":
                if isinstance(self.autonomous_agent.model_registry, DistributedModelRegistry):
                    successful_experiences = self.memory.retrieve_successful_experiences(top_k=1)
                    if successful_experiences and successful_experiences[0].get("expert_used"):
                        skill_to_publish = successful_experiences[0]["expert_used"][0]
                        success = asyncio.run(self.autonomous_agent.model_registry.publish_skill(skill_to_publish))
                        return {"status": "success" if success else "failure", "info": f"Published skill {skill_to_publish}"}, 1.0, ["model_registry"]
                return {"status": "skipped", "info": "Not using DistributedModelRegistry"}, 0.0, []

            elif action == "download_skill_from_community":
                if isinstance(self.autonomous_agent.model_registry, DistributedModelRegistry):
                    task_needed = self.state.get("last_task", "text_summarization")
                    downloaded_skill = asyncio.run(self.autonomous_agent.model_registry.download_skill(task_needed, "runs/downloaded_skills"))
                    return {"status": "success" if downloaded_skill else "failure", "info": f"Downloaded skill for {task_needed}"}, 1.0, ["model_registry"]
                return {"status": "skipped", "info": "Not using DistributedModelRegistry"}, 0.0, []

            elif action == "acquire_new_knowledge":
                self.state["last_task"] = "web_research"
                result_str = self.autonomous_agent.learn_from_web("latest trends in neuromorphic computing")
                return {"status": "success", "info": result_str}, 0.8, ["web_crawler", "summarizer"]
                
            elif action == "evolve_architecture":
                self.state["last_task"] = "self_evolution"
                result_str = self.self_evolving_agent.evolve()
                return {"status": "success", "info": result_str}, 0.9, ["self_evolver"]
                
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            elif action == "explore_new_task_with_rl" or action == "practice_skill_with_rl":
                self.state["last_task"] = "rl_training"
                logging.info(f"Initiating RL session: {action}")
                
                # 環境とトレーナーを初期化
                env = GridWorldEnv(size=5, max_steps=20, device=self.rl_agent.device)
                trainer = BioRLTrainer(agent=self.rl_agent, env=env)
                
                # 短い学習セッションを実行
                num_episodes = 20 if action == "explore_new_task_with_rl" else 10
                training_results = trainer.train(num_episodes=num_episodes)
                
                # 報酬は学習結果の平均報酬とする
                reward = training_results.get("final_average_reward", 0.0)
                
                return {"status": "success", "info": f"RL session '{action}' finished.", "results": training_results}, reward, ["rl_agent"]
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                
            elif action == "plan_and_execute":
                task = "Research the concept of 'Predictive Coding' and summarize its main ideas."
                self.state["last_task"] = "planning"
                result_str = self.autonomous_agent.execute(task)
                return {"status": "success", "info": result_str}, 0.8, ["planner", "web_crawler", "summarizer_snn"]
                
            else:
                return {"status": "failed", "info": "Unknown action"}, 0.0, []
        except Exception as e:
            logging.error(f"Error executing action '{action}': {e}")
            return {"status": "error", "info": str(e)}, -1.0, []

    def awareness_loop(self, cycles: int):
        print(f"🧬 Digital Life Form awareness loop starting for {cycles} cycles.")
        self.running = True
        for i in range(cycles):
            if not self.running:
                break
            print(f"\n----- Cycle {i+1}/{cycles} -----")
            self.life_cycle_step()
            time.sleep(2)
        self.stop()
        print("🧬 Awareness loop finished.")

    def explain_last_action(self) -> Optional[str]:
        try:
            with open(self.memory.memory_path, "rb") as f:
                try:
                    f.seek(-2, 2)
                    while f.read(1) != b'\n':
                        f.seek(-2, 1)
                except OSError:
                    f.seek(0)
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
        print("\n--- 自己言及プロンプト ---")
        print(prompt)
        print("--------------------------\n")

        try:
            snn_llm = self.langchain_adapter
            explanation = snn_llm._call(prompt)
            return explanation
        except Exception as e:
            logging.error(f"LLMによる自己言及の生成に失敗しました: {e}")
            return "エラー: 自己言及の生成に失敗しました。"
