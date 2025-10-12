# ファイルパス: tests/test_smoke_all_paradigms.py
# Title: 全学習パラダイム・推論 スモークテスト
# Description: プロジェクトに実装されている主要な学習・推論機能が、
#              極小データとモデルでエラーなく実行されることを高速に確認する。
# 修正点: pytestのフィクスチャのスコープを'module'から'function'に変更し、
#         各テストが完全に独立して実行されるように修正。
# 修正点(v2): ParticleFilterテストのデバイス不整合エラーを修正。

import sys
import os
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import TrainingContainer
from snn_research.deployment import SNNInferenceEngine
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent

# --- テスト用の設定 ---
MODEL_CONFIG = "configs/models/micro.yaml"
SMOKE_CONFIG = "configs/smoke_test_config.yaml"
DATA_PATH = "data/smoke_test_data.jsonl"


@pytest.fixture
def container():
    """DIコンテナを初期化し、テストフィクスチャとして提供する。（スコープを関数ごとに変更）"""
    c = TrainingContainer()
    c.config.from_yaml(SMOKE_CONFIG)
    c.config.from_yaml(MODEL_CONFIG)
    return c


def test_gradient_based_training(container: TrainingContainer):
    """勾配ベース学習の動作確認テスト。"""
    print("\n--- Testing: Gradient-based Training ---")
    container.config.training.paradigm.from_value("gradient_based")
    
    device = container.device()
    model = container.snn_model().to(device)
    optimizer = container.optimizer(params=model.parameters())
    scheduler = container.scheduler(optimizer=optimizer)
    trainer = container.standard_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    dummy_input = torch.randint(0, 100, (2, 4))
    dummy_target = torch.randint(0, 100, (2, 4))
    dummy_loader = DataLoader(TensorDataset(dummy_input, dummy_target), batch_size=2)
    
    trainer.train_epoch(dummy_loader)
    assert True


def test_distillation_training(container: TrainingContainer):
    """知識蒸留学習の動作確認テスト。"""
    print("\n--- Testing: Distillation Training ---")
    container.config.training.paradigm.from_value("gradient_based")
    container.config.training.gradient_based.type.from_value("distillation")

    device = container.device()
    model = container.snn_model().to(device)
    optimizer = container.optimizer(params=model.parameters())
    scheduler = container.scheduler(optimizer=optimizer)
    trainer = container.distillation_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    vocab_size = container.tokenizer().vocab_size
    dummy_input = torch.randint(0, vocab_size, (2, 4))
    dummy_target = torch.randint(0, vocab_size, (2, 4))
    dummy_teacher_logits = torch.randn(2, 4, vocab_size)
    
    dummy_batch = (dummy_input, torch.ones_like(dummy_input), dummy_target, dummy_teacher_logits)
    
    metrics = trainer._run_step(dummy_batch, is_train=True)
    assert "total" in metrics
    assert not torch.isnan(torch.tensor(metrics["total"])).any()


def test_physics_informed_training(container: TrainingContainer):
    """物理情報学習の動作確認テスト。"""
    print("\n--- Testing: Physics-informed Training ---")
    container.config.training.paradigm.from_value("physics_informed")

    device = container.device()
    model = container.snn_model().to(device)
    optimizer = container.pi_optimizer(params=model.parameters())
    scheduler = container.pi_scheduler(optimizer=optimizer)
    trainer = container.physics_informed_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    dummy_input = torch.randint(0, 100, (2, 4))
    dummy_target = torch.randint(0, 100, (2, 4))
    dummy_loader = DataLoader(TensorDataset(dummy_input, dummy_target), batch_size=2)

    trainer.train_epoch(dummy_loader)
    assert True


def test_bio_rl_training(container: TrainingContainer):
    """生物学的強化学習の動作確認テスト。"""
    print("\n--- Testing: Biologically Plausible RL Training ---")
    device = container.device()
    env = GridWorldEnv(size=3, max_steps=5, device=device)
    agent = ReinforcementLearnerAgent(input_size=4, output_size=4, device=device)

    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(reward)
        state = next_state
    assert True


def test_particle_filter_training(container: TrainingContainer):
    """パーティクルフィルタ学習の動作確認テスト。"""
    print("\n--- Testing: Particle Filter Training ---")
    container.config.training.paradigm.from_value("bio-particle-filter")
    trainer = container.particle_filter_trainer()
    device = container.device()

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # BioSNNの入出力サイズに合わせ、かつ正しいデバイス上に作成する
    dummy_data = torch.randn(10, device=device) 
    dummy_targets = torch.randn(2, device=device)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    
    loss = trainer.train_step(dummy_data, dummy_targets)
    assert not torch.isnan(torch.tensor(loss))


def test_inference_engine(container: TrainingContainer):
    """推論エンジンの動作確認テスト。"""
    print("\n--- Testing: Inference Engine ---")
    engine = SNNInferenceEngine(config=container.config())
    
    prompt = "test"
    response_generator = engine.generate(prompt, max_len=4)
    
    full_response = ""
    for chunk, _ in response_generator:
        full_response += chunk
        
    assert isinstance(full_response, str)
