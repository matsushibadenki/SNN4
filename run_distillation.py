# matsushibadenki/snn4/run_distillation.py
# Title: 知識蒸留実行スクリプト
# Description: KnowledgeDistillationManagerを使用して、知識蒸留プロセスを開始します。
#              設定ファイルとコマンドライン引数からパラメータを読み込みます。
#              mypyエラー修正: ContainerをTrainingContainerに修正。
# 改善点: argparseを追加し、asyncio.runで実行するように修正。
# 改善点(snn_4_ann_parity_plan):
# - ANN教師モデルとして、AutoModelForCausalLMの代わりに具体的なANNBaselineModelを
#   インスタンス化するように修正し、より管理された蒸留プロセスを実現。

import argparse
import asyncio
from app.containers import TrainingContainer
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.benchmark.ann_baseline import ANNBaselineModel

async def main():
    parser = argparse.ArgumentParser(description="SNN Knowledge Distillation Runner")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Base config file path")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="Model architecture config file path")
    args = parser.parse_args()

    # DIコンテナのインスタンス化
    container = TrainingContainer()
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)

    # --- ▼ 修正 ▼ ---
    # DIコンテナから必要なコンポーネントを正しい順序で取得・構築
    device = container.device()
    student_model = container.snn_model().to(device)
    optimizer = container.optimizer(params=student_model.parameters())
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

    # --- ▼ snn_4_ann_parity_planに基づく修正 ▼ ---
    # 教師モデルとしてANNBaselineModelを明示的に構築
    print("🧠 Initializing ANN teacher model (ANNBaselineModel)...")
    snn_config = container.config.model.to_dict()
    teacher_model = ANNBaselineModel(
        vocab_size=container.tokenizer.provided.vocab_size(),
        d_model=snn_config.get('d_model', 128),
        nhead=snn_config.get('n_head', 2),
        d_hid=snn_config.get('d_model', 128) * 4, # 一般的なFFNの拡張率
        nlayers=snn_config.get('num_layers', 4),
        num_classes=container.tokenizer.provided.vocab_size()
    ).to(device)
    # 注: 実際の使用例では、ここで教師モデルの学習済み重みをロードします
    # teacher_model.load_state_dict(torch.load("path/to/teacher.pth"))
    # --- ▲ snn_4_ann_parity_planに基づく修正 ▲ ---

    distillation_trainer = container.distillation_trainer(
        model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    model_registry = container.model_registry()

    manager = KnowledgeDistillationManager(
        student_model=student_model,
        # teacher_model_nameの代わりに、インスタンス化された教師モデルを渡すように変更
        teacher_model=teacher_model,
        trainer=distillation_trainer,
        tokenizer_name=container.config.data.tokenizer_name(),
        model_registry=model_registry,
        device=device
    )
    # --- ▲ 修正 ▲ ---

    # (仮) データセットの準備
    # 実際には、ファイルからテキストデータをロードする
    sample_texts = [
        "Spiking Neural Networks are a promising alternative to traditional ANNs.",
        "They operate based on discrete events, which can lead to greater energy efficiency.",
        "Knowledge distillation is a technique to transfer knowledge from a large model to a smaller one."
    ]
    train_loader = manager.prepare_dataset(
        sample_texts,
        max_length=container.config.model.time_steps(),
        batch_size=container.config.training.batch_size()
    )
    val_loader = train_loader # 簡単のため同じデータを使用

    # 蒸留の実行
    await manager.run_distillation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3, # テスト用のエポック数
        model_id="distilled_snn_expert_v1",
        task_description="An expert SNN for explaining AI concepts, created via distillation.",
        student_config=container.config.model.to_dict()
    )

if __name__ == "__main__":
    asyncio.run(main())
