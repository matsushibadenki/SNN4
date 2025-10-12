# ファイルパス: tests/cognitive_architecture/test_artificial_brain.py
# (更新)
# 修正: DIコンテナの階層構造に合わせて、rag_systemに正しくアクセスするよう修正。
# 改善(v2): ロードマップ フェーズ5 に基づき、認知サイクル実行後の状態変化を
#           具体的に検証するアサーションを追加。
# 改善(v3): 記憶の固定化プロセスを検証するテストを追加。

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

def test_cognitive_cycle_runs_and_consolidates_memory(brain_container: BrainContainer):
    """
    run_cognitive_cycleが実行され、記憶の固定化が正しく行われるかテストする。
    """
    brain: ArtificialBrain = brain_container.artificial_brain()
    
    # サイクル実行前の状態
    initial_cortex_size = len(brain.cortex.get_all_knowledge())
    
    # 記憶の固定がトリガーされるまで5回サイクルを実行
    test_inputs = [
        "This is a test about system integration.",
        "Another test focused on memory and learning.",
        "A third input to populate the hippocampus.",
        "Fourth cycle continues the process.",
        "Fifth cycle should trigger consolidation."
    ]
    
    try:
        for i, text in enumerate(test_inputs):
            brain.run_cognitive_cycle(text)
            # 5サイクル目に統合が起こることを確認
            if i < 4:
                assert len(brain.hippocampus.working_memory) == i + 1
            else:
                # 5サイクル目にクリアされるはず
                assert len(brain.hippocampus.working_memory) == 0

        print(f"✅ 5回の認知サイクルが正常に完了しました。")
    except Exception as e:
        pytest.fail(f"run_cognitive_cycleで予期せぬエラーが発生しました: {e}")

    # 実行後の状態変化を検証
    # 1. 海馬（短期記憶）がクリアされたか
    assert len(brain.hippocampus.working_memory) == 0, \
        "5サイクル後に海馬のワーキングメモリがクリアされていません。"
        
    # 2. 大脳皮質（長期記憶）に新しい知識が追加されたか
    final_cortex_size = len(brain.cortex.get_all_knowledge())
    assert final_cortex_size > initial_cortex_size, \
        "大脳皮質のナレッジグラフに新しい知識が追加されていません。"
        
    # 3. 記録された知識の内容を簡易的に確認
    knowledge = brain.cortex.retrieve_knowledge("system")
    assert knowledge is not None
    assert any(rel['target'] == 'integration' for rel in knowledge)
        
    print("✅ 記憶の固定化プロセスが正しく実行されたことを確認しました。")