# /run_web_learning.py
# ファイルパス: matsushibadenki/snn4/snn4-176e5ceb739db651438b22d74c0021f222858011/run_web_learning.py
# タイトル: Autonomous Web Learning Script
# 機能説明: 知識蒸留マネージャーを呼び出す際に、モデルのアーキテクチャ設定を正しく渡すように修正し、AttributeErrorを解消する。
# BugFix: 設定ファイル(use_scheduler)を尊重して学習率スケジューラを条件付きで有効にするように修正。

import argparse
import os
import asyncio
from snn_research.tools.web_crawler import WebCrawler
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from app.containers import TrainingContainer # DIコンテナを利用

def main():
    """
    Webクローラーとオンデマンド学習パイプラインを連携させ、
    指定されたトピックに関する専門家モデルを自律的に生成する。
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Web Learning Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="学習させたいトピック（タスク名として使用）。\n例: '最新のAI技術'"
    )
    parser.add_argument(
        "--start_url",
        type=str,
        required=True,
        help="情報収集を開始する起点となるURL。\n例: 'https://www.itmedia.co.jp/news/subtop/aiplus/'"
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=5, # デモ用に少なく設定
        help="収集するWebページの最大数。"
    )

    args = parser.parse_args()

    # --- ステップ1: Webクローリングによるデータ収集 ---
    print("\n" + "="*20 + " 🌐 Step 1: Web Crawling " + "="*20)
    crawler = WebCrawler()
    crawled_data_path = crawler.crawl(start_url=args.start_url, max_pages=args.max_pages)

    if not os.path.exists(crawled_data_path) or os.path.getsize(crawled_data_path) == 0:
        print("❌ データが収集できなかったため、学習を中止します。")
        return

    # --- ステップ2: オンデマンド知識蒸留による学習 ---
    print("\n" + "="*20 + " 🧠 Step 2: On-demand Learning " + "="*20)
    
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml("configs/models/medium.yaml")

    # 依存関係を正しい順序で構築する
    device = container.device()
    student_model = container.snn_model()
    optimizer = container.optimizer(params=student_model.parameters())
    
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 設定ファイルに基づき、スケジューラを条件付きで作成
    scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None
    
    distillation_trainer = container.distillation_trainer(
        model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    distillation_manager = KnowledgeDistillationManager(
        student_model=student_model,
        trainer=distillation_trainer,
        teacher_model_name=container.config.training.gradient_based.distillation.teacher_model(),
        tokenizer_name=container.config.data.tokenizer_name(),
        model_registry=container.model_registry(),
        device=device
    )

    student_config_dict = container.config.model.to_dict()

    asyncio.run(distillation_manager.run_on_demand_pipeline(
        task_description=args.topic,
        unlabeled_data_path=crawled_data_path,
        force_retrain=True,
        student_config=student_config_dict
    ))

    print("\n🎉 自律的なWeb学習サイクルが完了しました。")
    print(f"  トピック「{args.topic}」に関する新しい専門家モデルが育成されました。")

if __name__ == "__main__":
    main()