# matsushibadenki/snn4/train_planner.py
# Title: 学習可能プランナー訓練スクリプト
# Description: PlannerSNNモデルを訓練するためのスクリプト。
#              DIコンテナから必要なコンポーネントを取得し、訓練を実行する。
#              mypyエラー修正: PlannerTrainerを正しくインポートする。

import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch

from app.containers import TrainingContainer
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from snn_research.training.trainers import PlannerTrainer
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

def main():
    parser = argparse.ArgumentParser(description="SNN Planner Trainer")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="Model architecture config file path")
    args = parser.parse_args()

    # DIコンテナのインスタンス化
    container = TrainingContainer()
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)

    # DIコンテナから必要なコンポーネントを取得
    planner_model = container.planner_snn()
    planner_optimizer = container.planner_optimizer(params=planner_model.parameters())
    planner_loss = container.planner_loss()
    device = container.device()

    # PlannerTrainerのインスタンス化
    trainer = PlannerTrainer(
        model=planner_model,
        optimizer=planner_optimizer,
        criterion=planner_loss,
        device=device
    )

    # ダミーデータセットの作成
    # 実際のデータセット準備は別途スクリプトで行う
    dummy_input_ids = torch.randint(0, container.tokenizer().vocab_size, (16, 20))
    # ターゲットはスキル（専門家モデル）のIDシーケンス
    dummy_target_plan = torch.randint(0, 10, (16, 5)) # 10種類のスキル、最大5ステップ
    
    dummy_dataset = TensorDataset(dummy_input_ids, dummy_target_plan)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=container.config.training.batch_size())

    # 訓練の実行
    epochs = container.config.training.epochs()
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(dummy_dataloader, epoch)

    print("Planner training finished.")

if __name__ == "__main__":
    main()