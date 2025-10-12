# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# (更新)
# 改善点: 新しいHybridPerceptionCortexのperceive_and_learnメソッドを
#          呼び出すように修正。
# 改善点(v2): ROADMAPフェーズ2に基づき、Amygdalaからの情動出力を
#            BasalGangliaの行動選択に伝達するよう修正。
# 修正点(v3): mypyが検出したファイル末尾の不要な括弧を削除。

from typing import Dict, Any, List
import asyncio

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
        print("✅ 人工脳システムの全モジュールが正常に起動しました。")

    def run_cognitive_cycle(self, raw_input: Any):
        """
        外部からの感覚入力（テキストなど）を受け取り、
        知覚から行動までの一連の認知プロセスを実行する。
        """
        print(f"\n--- 🧠 新しい認知サイクルを開始 --- \n入力: '{raw_input}'")
        
        sensory_info = self.receptor.receive(raw_input)
        spike_pattern = self.encoder.encode(sensory_info, duration=50)

        # 知覚と同時に学習も行うメソッドを呼び出す
        perception_result = self.perception.perceive_and_learn(spike_pattern)
        
        episode = {'type': 'perception', 'content': perception_result, 'source_input': raw_input}
        self.hippocampus.store_episode(episode)

        emotion = self.amygdala.evaluate_emotion(raw_input if isinstance(raw_input, str) else "")
        self.global_context['internal_state']['emotion'] = emotion
        print(f"💖 扁桃体による評価: {emotion}")

        self.global_context['recent_memory'] = self.hippocampus.retrieve_recent_episodes(1)
        goal = self.pfc.decide_goal(self.global_context)
        
        plan = asyncio.run(self.planner.create_plan(goal))
        action_candidates = self._convert_plan_to_candidates(plan)
        
        # 行動選択の際に、現在の情動状態を伝達する
        selected_action = self.basal_ganglia.select_action(action_candidates, emotion_context=emotion)

        if selected_action:
            motor_commands = self.cerebellum.refine_action_plan(selected_action)
            command_logs = self.motor.execute_commands(motor_commands)
            self.actuator.run_command_sequence(command_logs)

        print("--- ✅ 認知サイクル完了 ---")

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
