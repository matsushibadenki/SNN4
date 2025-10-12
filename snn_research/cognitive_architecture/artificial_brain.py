# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (更新)
# 改善点: 新しいHybridPerceptionCortexのperceive_and_learnメソッドを
#          呼び出すように修正。
# 改善点(v2): ROADMAPフェーズ2に基づき、Amygdalaからの情動出力を
#            BasalGangliaの行動選択に伝達するよう修正。
# 改善点(v3): ROADMAPフェーズ3に基づき、記憶の固定と能動的想起のプロセスを実装。

from typing import Dict, Any, List
import asyncio
import re

# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from .hybrid_perception_cortex import HybridPerceptionCortex # 型ヒントを更新
from .prefrontal_cortex import PrefrontalCortex
from .hierarchical_planner import HierarchicalPlanner
# Memory systems
from .hippocampus import Hippocampus
from .cortex import Cortex
# Value and action selection
from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
# Motor control
from .cerebellum import Cerebellum
from .motor_cortex import MotorCortex

class ArtificialBrain:
    """
    認知アーキテクチャ全体を統合・制御する人工脳システム。
    """
    def __init__(
        self,
        # Input/Output
        sensory_receptor: SensoryReceptor,
        spike_encoder: SpikeEncoder,
        actuator: Actuator,
        # Core Cognitive Flow
        perception_cortex: HybridPerceptionCortex, # 型ヒントを更新
        prefrontal_cortex: PrefrontalCortex,
        hierarchical_planner: HierarchicalPlanner,
        # Memory
        hippocampus: Hippocampus,
        cortex: Cortex,
        # Value and Action
        amygdala: Amygdala,
        basal_ganglia: BasalGanglia,
        # Motor
        cerebellum: Cerebellum,
        motor_cortex: MotorCortex
    ):
        print("🚀 人工脳システムの起動を開始...")
        self.receptor = sensory_receptor
        self.encoder = spike_encoder
        self.actuator = actuator
        self.perception = perception_cortex
        self.pfc = prefrontal_cortex
        self.planner = hierarchical_planner
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.amygdala = amygdala
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.motor = motor_cortex
        
        self.global_context: Dict[str, Any] = {
            "internal_state": {}, "external_request": None
        }
        self.cycle_count = 0
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def run_cognitive_cycle(self, raw_input: Any):
        """
        外部からの感覚入力（テキストなど）を受け取り、
        知覚から行動までの一連の認知プロセスを実行する。
        """
        self.cycle_count += 1
        print(f"\n--- 🧠 新しい認知サイクルを開始 ({self.cycle_count}) --- \n入力: '{raw_input}'")
        
        # ... (感覚入力から知覚、短期記憶への保存までのプロセスは変更なし) ...
        sensory_info = self.receptor.receive(raw_input)
        spike_pattern = self.encoder.encode(sensory_info, duration=50)
        perception_result = self.perception.perceive_and_learn(spike_pattern)
        episode = {'type': 'perception', 'content': perception_result, 'source_input': raw_input}
        self.hippocampus.store_episode(episode)

        emotion = self.amygdala.evaluate_emotion(raw_input if isinstance(raw_input, str) else "")
        self.global_context['internal_state']['emotion'] = emotion
        print(f"💖 扁桃体による評価: {emotion}")

        self.global_context['recent_memory'] = self.hippocampus.retrieve_recent_episodes(1)
        goal = self.pfc.decide_goal(self.global_context)
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 能動的想起: 計画立案のために長期記憶から関連知識を検索
        knowledge_context = self._active_recall(goal)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        plan = asyncio.run(self.planner.create_plan(goal, context=knowledge_context))
        action_candidates = self._convert_plan_to_candidates(plan)
        
        selected_action = self.basal_ganglia.select_action(action_candidates, emotion_context=emotion)

        if selected_action:
            motor_commands = self.cerebellum.refine_action_plan(selected_action)
            command_logs = self.motor.execute_commands(motor_commands)
            self.actuator.run_command_sequence(command_logs)

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 記憶の固定: 一定サイクルごとに短期記憶を長期記憶に転送
        if self.cycle_count % 5 == 0: # 5サイクルごとに実行
            self.consolidate_memories()
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️

        print("--- ✅ 認知サイクル完了 ---")
        
    def _active_recall(self, goal: str) -> str:
        """長期記憶から目標に関連する知識を検索し、文脈として整形する。"""
        keywords = set(re.findall(r'\b[a-zA-Z]{5,}\b', goal.lower()))
        retrieved_knowledge = ""
        for keyword in keywords:
            knowledge = self.cortex.retrieve_knowledge(keyword)
            if knowledge:
                retrieved_knowledge += f"過去の知識'{keyword}': {knowledge}\n"
        
        if retrieved_knowledge:
            print(f"📖 長期記憶から関連知識を想起しました。")
        return retrieved_knowledge

    def consolidate_memories(self):
        """短期記憶（海馬）を長期記憶（大脳皮質）に固定化する。"""
        print("\n--- 🧠 記憶の固定プロセスを開始 ---")
        episodes_to_consolidate = self.hippocampus.get_and_clear_episodes_for_consolidation()
        for episode in episodes_to_consolidate:
            self.cortex.consolidate_memory(episode)
        print("--- ✅ 記憶の固定プロセス完了 ---\n")

    def _convert_plan_to_candidates(self, plan) -> List[Dict[str, Any]]:
        """プランナーからの計画を、大脳基底核が解釈できる行動候補リストに変換する。"""
        candidates = []
        for task in plan.task_list:
            candidates.append({
                'action': task.get('task', 'unknown_action'),
                'value': 0.8, 
                'duration': 1.0 
            })
        return candidates