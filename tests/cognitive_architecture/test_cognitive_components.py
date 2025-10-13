# ファイルパス: tests/cognitive_architecture/test_cognitive_components.py
# タイトル: 認知コンポーネント単体テスト
# 機能説明:
# - 人工脳を構成する各モジュールが、個別に正しく機能することを確認する単体テスト。
# - Amygdala, BasalGanglia, Cerebellum, MotorCortex, Hippocampus,
#   Cortex, PrefrontalCortex の基本動作とエッジケースを検証する。
# 改善点 (v4):
# - ロードマップ フェーズ5に基づき、エッジケース（空の入力、混合感情など）を
#   検証するテストを追加し、各コンポーネントの堅牢性を向上。

import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex

# --- Amygdala Tests ---
def test_amygdala_evaluates_positive_emotion():
    amygdala = Amygdala()
    emotion = amygdala.evaluate_emotion("素晴らしい成功体験でした。")
    assert "valence" in emotion and "arousal" in emotion
    assert emotion["valence"] > 0
    assert emotion["arousal"] > 0.5
    print("\n✅ Amygdala: ポジティブな感情の評価テストに成功。")

def test_amygdala_evaluates_negative_emotion():
    amygdala = Amygdala()
    emotion = amygdala.evaluate_emotion("危険なエラーが発生し、失敗した。")
    assert emotion["valence"] < 0
    assert emotion["arousal"] > 0.5
    print("✅ Amygdala: ネガティブな感情の評価テストに成功。")

def test_amygdala_handles_mixed_emotion():
    """ポジティブとネガティブが混在するテキストを評価できるかテストする。"""
    amygdala = Amygdala()
    # 喜び(0.9, 0.6)と失敗(-0.8, 0.8)の平均に近い値になるはず
    emotion = amygdala.evaluate_emotion("失敗の中に喜びを見出す。")
    assert -0.1 < emotion["valence"] < 0.2
    assert 0.6 < emotion["arousal"] < 0.8
    print("✅ Amygdala: 混合感情の評価テストに成功。")

def test_amygdala_handles_neutral_text():
    amygdala = Amygdala()
    emotion = amygdala.evaluate_emotion("これはただの事実です。")
    assert emotion["valence"] == 0.0
    assert emotion["arousal"] == 0.1
    print("✅ Amygdala: 中立的なテキストの評価テストに成功。")

def test_amygdala_handles_empty_string():
    """空の文字列が入力された場合にエラーなくデフォルト値を返すかテストする。"""
    amygdala = Amygdala()
    emotion = amygdala.evaluate_emotion("")
    assert emotion["valence"] == 0.0
    assert emotion["arousal"] == 0.1
    print("✅ Amygdala: 空文字列入力のテストに成功。")

# --- BasalGanglia Tests ---
def test_basal_ganglia_selects_best_action():
    basal_ganglia = BasalGanglia(selection_threshold=0.4)
    candidates = [
        {'action': 'A', 'value': 0.9},
        {'action': 'B', 'value': 0.6},
        {'action': 'C', 'value': 0.2},
    ]
    selected = basal_ganglia.select_action(candidates)
    assert selected is not None and selected['action'] == 'A'
    print("✅ BasalGanglia: 最適行動選択のテストに成功。")

def test_basal_ganglia_rejects_low_value_actions():
    basal_ganglia = BasalGanglia(selection_threshold=0.8)
    candidates = [{'action': 'A', 'value': 0.7}]
    selected = basal_ganglia.select_action(candidates)
    assert selected is None
    print("✅ BasalGanglia: 低価値行動の棄却テストに成功。")

def test_basal_ganglia_emotion_modulates_selection():
    basal_ganglia = BasalGanglia(selection_threshold=0.5)
    candidates = [{'action': 'run_away', 'value': 0.8}]
    fear_context = {'valence': -0.8, 'arousal': 0.9} # 恐怖
    selected_fear = basal_ganglia.select_action(candidates, emotion_context=fear_context)
    assert selected_fear is not None and selected_fear['action'] == 'run_away'
    print("✅ BasalGanglia: 情動による意思決定変調のテストに成功。")

def test_basal_ganglia_handles_no_candidates():
    basal_ganglia = BasalGanglia()
    selected = basal_ganglia.select_action([])
    assert selected is None
    print("✅ BasalGanglia: 行動候補が空の場合のテストに成功。")

# --- Cerebellum & MotorCortex Tests ---
def test_cerebellum_and_motor_cortex_pipeline():
    cerebellum = Cerebellum()
    motor_cortex = MotorCortex(actuators=['test_actuator'])
    action = {'action': 'do_something', 'duration': 0.5}
    
    commands = cerebellum.refine_action_plan(action)
    assert len(commands) > 1 and commands[0]['command'] == 'do_something_start'
    
    log = motor_cortex.execute_commands(commands)
    assert len(log) > 1 and "test_start" in log[0]
    print("✅ Cerebellum -> MotorCortex パイプラインのテストに成功。")

def test_cerebellum_and_motor_handles_empty():
    """空の入力がパイプライン全体でエラーなく処理されるかテストする。"""
    cerebellum = Cerebellum()
    motor_cortex = MotorCortex()
    commands = cerebellum.refine_action_plan({})
    assert commands == []
    log = motor_cortex.execute_commands(commands)
    assert log == []
    print("✅ Cerebellum & MotorCortex: 空入力のテストに成功。")

# --- Hippocampus & Cortex (Memory System) Tests ---
def test_memory_system_pipeline():
    hippocampus = Hippocampus(capacity=3)
    cortex = Cortex()
    
    # 1. 短期記憶へ保存
    hippocampus.store_episode({'source_input': 'A cat is a small animal.'})
    hippocampus.store_episode({'source_input': 'A dog is a friendly pet.'})
    assert len(hippocampus.working_memory) == 2
    
    # 2. 長期記憶へ固定化
    episodes_for_consolidation = hippocampus.get_and_clear_episodes_for_consolidation()
    assert len(episodes_for_consolidation) == 2
    assert len(hippocampus.working_memory) == 0 # クリアされたか確認
    
    for episode in episodes_for_consolidation:
        cortex.consolidate_memory(episode)
    
    # 3. 長期記憶から検索
    knowledge = cortex.retrieve_knowledge('animal')
    assert knowledge is not None and len(knowledge) > 0
    assert any(rel['target'] == 'small' for rel in knowledge)
    print("✅ Hippocampus -> Cortex (記憶固定化) パイプラインのテストに成功。")

# --- PrefrontalCortex Tests ---
@pytest.mark.parametrize("context, expected_keyword", [
    ({"external_request": "summarize the document"}, "summarize"),
    ({"internal_state": {"boredom": 0.8}}, "boredom"),
    ({"internal_state": {"curiosity": 0.9}}, "Acquire new knowledge"),
    ({"internal_state": {"boredom": 0.1, "curiosity": 0.2}}, "Organize"),
])
def test_prefrontal_cortex_decides_goals(context, expected_keyword):
    pfc = PrefrontalCortex()
    goal = pfc.decide_goal(context)
    assert expected_keyword in goal
    print(f"✅ PrefrontalCortex: '{expected_keyword}'に基づく目標設定のテストに成功。")
