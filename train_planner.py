# matsushibadenki/snn3/train_planner.py
# Title: 学習可能プランナー訓練スクリプト
# Description: PlannerSNNモデルを訓練するためのスクリプト。
#              DIコンテナから必要なコンポーネントを取得し、訓練を実行する。
#              mypyエラー修正: PlannerTrainerを正しくインポートする。
# 改善点: ダミーデータではなく、実際のデータファイルから学習データを生成するように修正。

import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch
import json

from app.containers import TrainingContainer
from snn_research.training.trainers import PlannerTrainer

def main():
    parser = argparse.ArgumentParser(description="SNN Planner Trainer")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="Model architecture config file path")
    parser.add_argument("--data_path", type=str, default="data/sample_data.jsonl", help="Path to the training data.")
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
    tokenizer = container.tokenizer()

    # PlannerTrainerのインスタンス化
    trainer = PlannerTrainer(
        model=planner_model,
        optimizer=planner_optimizer,
        criterion=planner_loss,
        device=device
    )

    # --- データセットの作成 ---
    # sample_data.jsonl からテキストを読み込み、プランナーの学習データに変換する
    texts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(json.loads(line)['text'])

    # 簡単なルールでテキストからターゲットとなるスキルIDを生成
    # (例: 'summarize'が含まれていればスキルID 0, 'sentiment'ならID 1など)
    def get_skill_id(text: str) -> int:
        if any(kw in text.lower() for kw in ["summarize", "what is", "explain"]):
            return 0  # General QA or Summarization skill
        if any(kw in text.lower() for kw in ["sentiment", "feel", "enjoy"]):
            return 1  # Sentiment analysis skill
        return 2 # Default skill

    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=20)
    input_ids = tokenized_inputs['input_ids']
    target_plan = torch.tensor([get_skill_id(text) for text in texts]).unsqueeze(1)

    dataset = TensorDataset(input_ids, target_plan)
    dataloader = DataLoader(dataset, batch_size=container.config.training.batch_size())

    # 訓練の実行
    epochs = container.config.training.epochs()
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(dataloader, epoch)

    print("Planner training finished.")

if __name__ == "__main__":
    main()