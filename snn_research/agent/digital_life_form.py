# ファイルパス: snn_research/agent/digital_life_form.py
# (更新)
# 改善点:
# - `_decide_next_action`で好奇心に基づく行動選択の重みを増加。
# - `_execute_action`に、好奇心の対象を自律的に調査・学習する
#   `explore_curiosity`アクションを実装。

import time
import logging
import torch
import random
import json
import asyncio
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import operator
import os

# (import文は変更なし)
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent

if TYPE_CHECKING:
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DigitalLifeForm:
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
        langchain_adapter: "SNNLangChainAdapter"
    ):
        # (変更なし)
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

    def life_cycle_step(self):
        # (変更なし)
        internal_state = self.motivation_system.get_internal_state()
        performance_eval = self.meta_cognitive_snn.evaluate_performance()
        dummy_mem = torch.randn(100); dummy_spikes = (torch.rand(100) > 0.8).float()
        physical_rewards = self.physics_evaluator.evaluate_physical_consistency(dummy_mem, dummy_spikes)
        
        action = self._decide_next_action(internal_state, performance_eval)
        result, external_reward, expert_used = self._execute_action(action, internal_state)

        if isinstance(result, dict): self.symbol_grounding.process_observation(result, context=f"action '{action}'")
        
        reward_vector = {"external": external_reward, "physical": physical_rewards, "curiosity": internal_state.get("curiosity", 0.0)}
        decision_context = {"internal_state": internal_state, "performance_eval": performance_eval}
        # 記憶に現在の文脈も記録
        self.memory.record_experience(self.state, action, result, reward_vector, expert_used, decision_context, causal_snapshot=str(internal_state.get('curiosity_context')))
        
        # motivation_systemの更新
        context_for_motivation = {"action": action, "result": result}
        self.motivation_system.update_metrics(random.random(), random.random(), random.random(), random.random(), context=context_for_motivation)
        
        self.state["last_action"] = action; self.state["last_result"] = result
        logging.info(f"Action: {action}, Result: {str(result)[:100]}, Reward: {external_reward:.2f}")

    def _decide_next_action(self, internal_state: Dict[str, Any], performance_eval: Dict[str, Any]) -> str:
        # (好奇心に関するスコアの重みを増加)
        action_scores: Dict[str, float] = {
            "explore_curiosity": internal_state.get("curiosity", 0.0) * 20.0, # 好奇心探求の優先度を大幅に上げる
            "evolve_architecture": 0.0,
            "practice_skill_with_rl": internal_state.get("confidence", 0.5) * 2.0,
        }
        if performance_eval.get("status") == "capability_gap":
            action_scores["evolve_architecture"] += 10.0
        if internal_state.get("boredom", 0.0) > 0.8:
            action_scores["explore_curiosity"] += internal_state.get("boredom", 0.0) * 15.0 # 退屈な時も探求を優先

        chosen_action = max(action_scores.items(), key=operator.itemgetter(1))[0]
        logging.info(f"Action scores: {action_scores} -> Chosen: {chosen_action}")
        return chosen_action

    # --- ◾️◾️◾️◾️◾️↓ここからが重要↓◾️◾️◾️◾️◾️ ---
    def _execute_action(self, action: str, internal_state: Dict[str, Any]) -> tuple[Dict[str, Any], float, List[str]]:
        """
        選択された行動に対応するエージェントの機能を呼び出す。
        好奇心探求アクションを実装。
        """
        try:
            if action == "explore_curiosity":
                # 1. 好奇心の対象を取得
                curiosity_topic = internal_state.get("curiosity_context")
                if not curiosity_topic:
                    return {"status": "skipped", "info": "No specific curiosity context found."}, 0.0, []

                # 2. 好奇心の対象を自然言語の検索クエリに変換（簡易的）
                topic_str = str(curiosity_topic.get("action", "AI concept"))
                logging.info(f"🔬 Curiosity triggered! Researching topic: '{topic_str}'")
                
                # 3. 自律エージェントにWeb学習と専門家育成を依頼
                # handle_taskは専門家モデルを検索し、なければ学習を試みる
                new_model_info = asyncio.run(self.autonomous_agent.handle_task(
                    task_description=topic_str,
                    # Web学習を実行させるため、ダミーのデータパスを指定（将来的にはWebCrawlerの結果を直接渡す）
                    unlabeled_data_path="data/sample_data.jsonl",
                    force_retrain=True # 常に新しい専門家を育成
                ))

                if new_model_info:
                    return {"status": "success", "info": f"Learned about '{topic_str}' and created new expert.", "model_info": new_model_info}, 1.0, ["autonomous_agent", "web_crawler", "distillation_manager"]
                else:
                    return {"status": "failure", "info": f"Failed to learn about '{topic_str}'."}, -0.5, ["autonomous_agent"]
            
            # (他のアクションは変更なし)
            elif action == "evolve_architecture":
                result_str = self.self_evolving_agent.evolve()
                return {"status": "success", "info": result_str}, 0.9, ["self_evolver"]
            
            elif action == "practice_skill_with_rl":
                from snn_research.rl_env.grid_world import GridWorldEnv
                from snn_research.training.bio_trainer import BioRLTrainer
                env = GridWorldEnv(size=5, max_steps=20, device=self.rl_agent.device)
                trainer = BioRLTrainer(agent=self.rl_agent, env=env)
                training_results = trainer.train(num_episodes=10)
                return {"status": "success", "results": training_results}, training_results.get("final_average_reward", 0.0), ["rl_agent"]

            else:
                return {"status": "idle", "info": "No compelling action to take."}, 0.0, []

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
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
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

