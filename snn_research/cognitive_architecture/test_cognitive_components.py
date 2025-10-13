# ファイルパス: snn_research/cognitive_architecture/test_cognitive_components.py
# (修正)
#
# Title: 認知コンポーネント単体テスト
#
# Description:
# - 人工脳を構成する各モジュールが、個別に正しく機能することを確認する単体テスト。
# - Amygdala, BasalGanglia, Cerebellum, MotorCortex, Hippocampus,
#   Cortex, PrefrontalCortex の基本動作とエッジケースを検証する。
#
# 修正点(v4):
# - 失敗していたテストを修正。Cortexモジュールの現在の仕様に合わせて、
#   テストデータ（episode）の形式を{'source_input': '...'}に更新しました。
# - このファイルは tests/cognitive_architecture/test_cognitive_components.py と
#   重複していましたが、内容を統一することでエラーを解消します。

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem

# --- Mocks for dependencies ---
@pytest.fixture
def mock_workspace():
    """GlobalWorkspaceのモックを作成するフィクスチャ。"""
    return MagicMock(spec=GlobalWorkspace)

@pytest.fixture
def mock_motivation_system():
    """IntrinsicMotivationSystemのモックを作成するフィクスチャ。"""
    return MagicMock(spec=IntrinsicMotivationSystem)

# --- Amygdala Tests ---
def test_amygdala_evaluates_emotion(mock_workspace):
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("素晴らしい成功体験でした。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['salience'] > 0.5
    assert kwargs['data']['valence'] > 0
    print("\n✅ Amygdala: ポジティブな感情の評価テストに成功。")


def test_amygdala_handles_neutral_text(mock_workspace):
    amygdala = Amygdala(workspace=mock_workspace)
    amygdala.evaluate_and_upload("これはただの事実です。")
    mock_workspace.upload_to_workspace.assert_called_once()
    _, kwargs = mock_workspace.upload_to_workspace.call_args
    assert kwargs['data']['valence'] == 0.0
    print("✅ Amygdala: 中立的なテキストの評価テストに成功。")

# --- BasalGanglia Tests ---
def test_basal_ganglia_selects_best_action(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.4)
    candidates = [
        {'action': 'A', 'value': 0.9},
        {'action': 'B', 'value': 0.6},
        {'action': 'C', 'value': 0.2},
    ]
    selected = basal_ganglia.select_action(candidates)
    assert selected is not None
    assert selected['action'] == 'A'

def test_basal_ganglia_rejects_low_value_actions(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.8)
    candidates = [{'action': 'A', 'value': 0.7}]
    selected = basal_ganglia.select_action(candidates)
    assert selected is None

def test_basal_ganglia_emotion_modulates_selection(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace, selection_threshold=0.5)
    candidates = [{'action': 'run_away', 'value': 0.8}]
    fear_context = {'valence': -0.8, 'arousal': 0.9}
    selected_fear = basal_ganglia.select_action(candidates, emotion_context=fear_context)
    assert selected_fear is not None
    assert selected_fear['action'] == 'run_away'

def test_basal_ganglia_handles_no_candidates(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace)
    selected = basal_ganglia.select_action([])
    assert selected is None

# --- Cerebellum Tests ---
def test_cerebellum_refines_plan():
    cerebellum = Cerebellum()
    action = {'action': 'do_something', 'duration': 0.5}
    commands = cerebellum.refine_action_plan(action)
    assert len(commands) > 1
    assert commands[0]['command'] == 'do_something_start'
    assert commands[-1]['command'] == 'do_something_end'

def test_cerebellum_handles_empty_plan():
    cerebellum = Cerebellum()
    commands = cerebellum.refine_action_plan({})
    assert commands == []

# --- MotorCortex Tests ---
def test_motor_cortex_executes_commands():
    motor_cortex = MotorCortex(actuators=['test_actuator'])
    commands = [{'timestamp': 0.0, 'command': 'test_start'}]
    log = motor_cortex.execute_commands(commands)
    assert len(log) == 1
    assert "test_start" in log[0]
    assert "test_actuator" in log[0]

def test_motor_cortex_handles_empty_commands():
    motor_cortex = MotorCortex()
    log = motor_cortex.execute_commands([])
    assert log == []

# --- Hippocampus Tests ---
def test_hippocampus_stores_and_retrieves_episodes(mock_workspace):
    hippocampus = Hippocampus(workspace=mock_workspace, capacity=3)
    e1, e2, e3 = {'id': 1}, {'id': 2}, {'id': 3}
    hippocampus.store_episode(e1)
    hippocampus.store_episode(e2)
    hippocampus.store_episode(e3)
    recent = hippocampus.retrieve_recent_episodes(2)
    assert len(recent) == 2
    assert recent[0] == e3
    assert recent[1] == e2

def test_hippocampus_capacity_limit(mock_workspace):
    hippocampus = Hippocampus(workspace=mock_workspace, capacity=2)
    hippocampus.store_episode({'id': 1})
    hippocampus.store_episode({'id': 2})
    hippocampus.store_episode({'id': 3})
    memory_content = list(hippocampus.working_memory)
    assert len(memory_content) == 2
    assert {'id': 1} not in memory_content

def test_hippocampus_retrieves_from_empty_memory(mock_workspace):
    hippocampus = Hippocampus(workspace=mock_workspace)
    recent = hippocampus.retrieve_recent_episodes(5)
    assert recent == []

# --- Cortex Tests ---
def test_cortex_consolidates_and_retrieves_knowledge():
    cortex = Cortex()
    # 修正: 現在のCortexの仕様に合わせたエピソード形式を使用
    episode = {'source_input': 'A cat is a small animal.'}
    cortex.consolidate_memory(episode)
    # 修正: 新しいナレッジグラフの構造に合わせてアサーションを更新
    knowledge = cortex.retrieve_knowledge('animal')
    assert knowledge is not None
    assert len(knowledge) == 1
    assert knowledge[0]['relation'] == 'co-occurred_with'
    assert knowledge[0]['target'] == 'small'

def test_cortex_retrieves_non_existent_concept():
    cortex = Cortex()
    knowledge = cortex.retrieve_knowledge('unknown_concept')
    assert knowledge is None

# --- PrefrontalCortex Tests ---
def test_prefrontal_cortex_decides_goal_from_request(mock_workspace, mock_motivation_system):
    pfc = PrefrontalCortex(workspace=mock_workspace, motivation_system=mock_motivation_system)
    context = {"external_request": "summarize the document"}
    goal = pfc.decide_goal(context)
    assert "summarize" in goal

def test_prefrontal_cortex_decides_goal_from_boredom(mock_workspace, mock_motivation_system):
    mock_motivation_system.get_internal_state.return_value = {"boredom": 0.8}
    pfc = PrefrontalCortex(workspace=mock_workspace, motivation_system=mock_motivation_system)
    context = {"internal_state": {"boredom": 0.8}}
    goal = pfc.decide_goal(context)
    assert "boredom" in goal

def test_prefrontal_cortex_decides_default_goal(mock_workspace, mock_motivation_system):
    mock_motivation_system.get_internal_state.return_value = {"boredom": 0.1, "curiosity": 0.2}
    pfc = PrefrontalCortex(workspace=mock_workspace, motivation_system=mock_motivation_system)
    context = {"internal_state": {"boredom": 0.1, "curiosity": 0.2}, "external_request": None}
    goal = pfc.decide_goal(context)
    assert "Organize" in goal
