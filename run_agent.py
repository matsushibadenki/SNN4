# ファイルパス: matsushibadenki/snn4/snn4-190ede29139f560c909685675a68ccf65069201c/run_agent.py
#
# 自律エージェントを起動し、タスクを実行させるためのインターフェース
#
# 変更点:
# - 推論実行ロジックのコメントアウトを解除。
# - ヘルプメッセージを改善。
# - 改善点: DIコンテナと同様に、エージェントに必要な依存関係を初期化して注入するように修正。

import argparse
import asyncio
from app.containers import AgentContainer # DIコンテナをインポート

def main():
    """
    自律エージェントにタスクを依頼し、最適な専門家SNNモデルの選択または生成を行わせる。
    """
    parser = argparse.ArgumentParser(
        description="自律的SNNエージェント実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="解決したいタスクの自然言語による説明。\n例: '感情分析', '文章要約'"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="(オプション) 選択/学習させたモデルで推論を実行する場合の入力プロンプト。\n例: 'この映画は最高だった！'"
    )
    parser.add_argument(
        "--unlabeled_data_path",
        type=str,
        help="エージェントが新しい専門家モデルを学習する必要がある場合に使用する、ラベルなしデータへのパス。\n例: 'data/sample_data.jsonl'"
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="このフラグを立てると、モデル登録簿のチェックをスキップして強制的に再学習します。"
    )

    args = parser.parse_args()

    # --- 改善: DIコンテナを使用して依存関係を構築 ---
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    
    # AgentContainerから直接AutonomousAgentをインスタンス化
    # (AgentContainerにAutonomousAgentのプロバイダを追加する必要がある)
    # ここでは、必要なコンポーネントを個別に取得して注入する
    planner = container.hierarchical_planner()
    model_registry = container.model_registry()
    memory = container.memory()
    web_crawler = container.web_crawler()

    # --- 自律エージェントの初期化 ---
    # AutonomousAgentクラスをインポート
    from snn_research.agent.autonomous_agent import AutonomousAgent
    agent = AutonomousAgent(
        name="run_agent_instance",
        planner=planner,
        model_registry=model_registry,
        memory=memory,
        web_crawler=web_crawler
    )


    # --- エージェントにタスク処理を依頼 ---
    selected_model_info = asyncio.run(agent.handle_task(
        task_description=args.task_description,
        unlabeled_data_path=args.unlabeled_data_path,
        force_retrain=args.force_retrain
    ))

    if selected_model_info:
        print("\n" + "="*20 + " ✅ TASK COMPLETED " + "="*20)
        print(f"最適な専門家モデルが準備されました: '{args.task_description}'")
        if 'path' in selected_model_info:
             print(f"  - モデルパス: {selected_model_info['path']}")
        if 'metrics' in selected_model_info:
             print(f"  - 性能: {selected_model_info['metrics']}")

        if args.prompt:
            print("\n" + "="*20 + " 🧠 INFERENCE " + "="*20)
            print(f"入力プロンプト: {args.prompt}")
            asyncio.run(agent.run_inference(selected_model_info, args.prompt))
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。")

if __name__ == "__main__":
    main()