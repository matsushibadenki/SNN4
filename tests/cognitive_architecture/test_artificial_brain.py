# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# (修正)
# 修正: DIコンテナの階層構造に合わせて、rag_systemに正しくアクセスするよう修正。
# 改善(v2): ロードマップ フェーズ5 に基づき、認知サイクル実行後の状態変化を
#           具体的に検証するアサーションを追加。

import sys
from pathlib import Path
import pytest

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.containers import BrainContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

@pytest.fixture(scope="module")
def brain_container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。"""
    container = BrainContainer()
    # テスト用の設定をロード
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/models/small.yaml")
    
    # RAGSystemの初回セットアップをシミュレート
    # rag_systemはagent_containerの配下にあるため、正しくアクセスする
    rag_system = container.agent_container.rag_system()
    if not rag_system.vector_store:
        rag_system.setup_vector_store()
    return container

def test_artificial_brain_instantiation(brain_container: BrainContainer):
    """
    BrainContainerがArtificialBrainインスタンスを正常に構築できるかテストする。
    """
    brain = brain_container.artificial_brain()
    assert brain is not None
    assert brain.pfc is not None
    assert brain.hippocampus is not None
    assert brain.motor is not None
    print("✅ ArtificialBrainインスタンスの構築に成功しました。")

def test_cognitive_cycle_runs_and_updates_state(brain_container: BrainContainer):
    """
    run_cognitive_cycleがサンプル入力に対してエラーなく実行され、
    各モジュールの内部状態が意図通りに更新されるかテストする。
    """
    brain: ArtificialBrain = brain_container.artificial_brain()
    
    test_input = "これはシステム全体の統合テストです。"
    
    # 実行前の状態を確認
    initial_hippocampus_size = len(brain.hippocampus.working_memory)
    
    try:
        brain.run_cognitive_cycle(test_input)
        print(f"✅ 認知サイクルが入力 '{test_input}' に対して正常に完了しました。")
    except Exception as e:
        pytest.fail(f"run_cognitive_cycleで予期せぬエラーが発生しました: {e}")

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓追加開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 実行後の状態変化を検証
    # 1. 海馬（短期記憶）に新しいエピソードが記録されたか
    assert len(brain.hippocampus.working_memory) == initial_hippocampus_size + 1, \
        "海馬のワーキングメモリに新しいエピソードが追加されていません。"
    
    # 2. 記録されたエピソードの内容が正しいか
    latest_episode = brain.hippocampus.retrieve_recent_episodes(1)[0]
    assert latest_episode is not None, "海馬から最新のエピソードを取得できませんでした。"
    assert latest_episode['source_input'] == test_input, \
        f"記録されたエピソードの入力が一致しません: expected '{test_input}', got '{latest_episode['source_input']}'"
    assert 'features' in latest_episode['content'], \
        "記録されたエピソードに知覚結果（features）が含まれていません。"
        
    print("✅ 認知サイクル後の状態変化（海馬への記憶）を正常に確認しました。")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑追加終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
