# ファイルパス: snn_research/cognitive_architecture/artificial_brain.py
# タイトル: 人工脳 統合認知サイクル
# 機能説明:
# - 人工脳の全コンポーネントを統合し、知覚から行動までの一連の認知プロセスを実行する。
# - ROADMAP v8.0のフェーズ2「動的かつ情動的な意思決定」とフェーズ3「記憶の固定と能動的想起」を実装。
# - Amygdalaからの情動出力をBasalGangliaに伝達し、行動選択の閾値を動的に変化させる。
# - 一定サイクルごとにHippocampusの短期記憶をCortexの長期記憶へと「固定化」する。
# - 計画立案時にCortexから関連知識を「能動的に想起」し、プランナーの文脈情報として活用する。
# 修正点: mypyエラー `Need type annotation for "candidates"` を解消するため、型ヒントを追加。

from typing import Dict, Any, List
import asyncio
import re

# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from .hybrid_perception_cortex import HybridPerceptionCortex
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
        perception_cortex: HybridPerceptionCortex,
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
        
        # 1. 感覚入力 -> スパイク変換 -> 知覚 -> 短期記憶への保存
        sensory_info = self.receptor.receive(raw_input)
        spike_pattern = self.encoder.encode(sensory_info, duration=50)
        perception_result = self.perception.perceive_and_learn(spike_pattern)
        episode = {'type': 'perception', 'content': perception_result, 'source_input': raw_input}
        self.hippocampus.store_episode(episode)

        # 2. 情動評価 (扁桃体)
        emotion = self.amygdala.evaluate_emotion(raw_input if isinstance(raw_input, str) else "")
        self.global_context['internal_state']['emotion'] = emotion
        print(f"💖 扁桃体による評価: {emotion}")

        # 3. 目標設定 (前頭前野)
        self.global_context['recent_memory'] = self.hippocampus.retrieve_recent_episodes(1)
        goal = self.pfc.decide_goal(self.global_context)
        
        # 4. 能動的想起 (長期記憶からの知識検索)
        knowledge_context = self._active_recall(goal)
        
        # 5. 計画立案 (階層プランナー)
        plan = asyncio.run(self.planner.create_plan(goal, context=knowledge_context))
        action_candidates = self._convert_plan_to_candidates(plan)
        
        # 6. 行動選択 (大脳基底核) - 情動コンテキストを伝達
        selected_action = self.basal_ganglia.select_action(action_candidates, emotion_context=emotion)

        # 7. 運動実行 (小脳、運動野)
        if selected_action:
            motor_commands = self.cerebellum.refine_action_plan(selected_action)
            command_logs = self.motor.execute_commands(motor_commands)
            self.actuator.run_command_sequence(command_logs)

        # 8. 記憶の固定 (一定サイクルごと)
        if self.cycle_count % 5 == 0:
            self.consolidate_memories()

        print("--- ✅ 認知サイクル完了 ---")
        
    def _active_recall(self, goal: str) -> str:
        """長期記憶から目標に関連する知識を検索し、文脈として整形する。"""
        # 5文字以上の単語をキーワードとして抽出
        keywords = set(re.findall(r'\b[a-zA-Z]{5,}\b', goal.lower()))
        retrieved_knowledge = ""
        for keyword in keywords:
            knowledge = self.cortex.retrieve_knowledge(keyword)
            if knowledge:
                # 関連知識をテキストに変換
                knowledge_text = f"過去の知識'{keyword}': " + ", ".join([f"{rel['relation']} '{rel['target']}'" for rel in knowledge])
                retrieved_knowledge += knowledge_text + "\n"
        
        if retrieved_knowledge:
            print(f"📖 長期記憶から関連知識を想起しました。")
            print(f"  - {retrieved_knowledge.strip()}")
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
        candidates: List[Dict[str, Any]] = []
        if not plan or not plan.task_list:
            return candidates
            
        for task in plan.task_list:
            # 各タスクに基本的な価値(value)と持続時間(duration)を割り当てる
            candidates.append({
                'action': task.get('task', 'unknown_action'),
                'value': 0.8, # デフォルトの価値
                'duration': 1.0 # デフォルトの持続時間
            })
        return candidates

