# ファイルパス: tests/test_smoke_all_paradigdms.py
# (修正)
# Description:
# - train.pyがサポートする主要な学習パラダイムの煙テスト。
# - 各パラダイムで、ごく少数のステップ（数バッチ）の学習がエラーなく実行できることを確認する。
# - mypyエラー [call-arg] を解消するため、train_epoch呼び出しにepoch引数を追加。
# - mypyエラー [no-redef] を解消するため、重複していたコンテナの定義を削除。

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.containers import TrainingContainer
import os

# DIコンテナをフィクスチャとして初期化
@pytest.fixture(scope="module")
def container():
    c = TrainingContainer()
    c.config.from_yaml("configs/base_config.yaml")
    c.config.from_yaml("configs/models/small.yaml")
    # テスト用に設定を上書き
    c.config.training.epochs.from_value(1)
    c.config.training.log_dir.from_value("runs/test_logs")
    return c

# ダミーデータローダーをフィクスチャとして作成
@pytest.fixture(scope="module")
def dummy_dataloader(container: TrainingContainer):
    tokenizer = container.tokenizer()
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (8, 20))
    dummy_target_ids = torch.randint(0, tokenizer.vocab_size, (8, 20))
    dataset = TensorDataset(dummy_input_ids, dummy_target_ids)
    return DataLoader(dataset, batch_size=4)

# --- 煙テストの定義 ---

def test_smoke_gradient_based(container: TrainingContainer, dummy_dataloader: DataLoader):
    """勾配ベース学習の煙テスト"""
    print("\n--- Testing: gradient_based ---")
    device = container.device()
    model = container.snn_model().to(device)
    optimizer = container.optimizer(params=model.parameters())
    scheduler = container.scheduler(optimizer=optimizer)
    
    trainer = container.standard_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1
    )
    trainer.train_epoch(dummy_dataloader, epoch=1)
    assert True # エラーなく実行されればOK

def test_smoke_physics_informed(container: TrainingContainer, dummy_dataloader: DataLoader):
    """物理情報学習の煙テスト"""
    print("\n--- Testing: physics_informed ---")
    device = container.device()
    model = container.snn_model().to(device)
    optimizer = container.pi_optimizer(params=model.parameters())
    scheduler = container.pi_scheduler(optimizer=optimizer)
    
    trainer = container.physics_informed_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=-1
    )
    trainer.train_epoch(dummy_dataloader, epoch=1)
    assert True # エラーなく実行されればOK

def test_smoke_bio_causal_sparse(container: TrainingContainer):
    """生物学的因果学習の煙テスト"""
    print("\n--- Testing: bio-causal-sparse ---")
    container.config.training.biologically_plausible.adaptive_causal_sparsification.enabled.from_value(True)
    trainer = container.bio_rl_trainer()
    trainer.train(num_episodes=2) # 2エピソードだけ実行
    assert True

def test_smoke_bio_particle_filter(container: TrainingContainer):
    """パーティクルフィルタ学習の煙テスト"""
    print("\n--- Testing: bio-particle-filter ---")
    device = container.device()
    trainer = container.particle_filter_trainer()
    dummy_data = torch.rand(1, 10, device=device)
    dummy_targets = torch.rand(1, 2, device=device)
    trainer.train_step(dummy_data, dummy_targets)
    assert True