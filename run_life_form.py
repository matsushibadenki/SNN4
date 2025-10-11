# ファイルパス: matsushibadenki/snn4/snn4-190ede29139f560c90968675a68ccf65069201c/run_life_form.py
#
# デジタル生命体 起動スクリプト
# (省略)
# 修正点 (v3): 循環インポートを避けるため、トップレベルのインポートを削除し、
#              main関数内で局所的にインポートするように修正。

import time
import argparse
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# トップレベルのインポートを削除
# from app.containers import AgentContainer, AppContainer
# from snn_research.agent.digital_life_form import DigitalLifeForm
# from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
# from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
# from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
# from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
# from snn_research.agent.autonomous_agent import AutonomousAgent
# from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
# from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

def main():
    """
    デジタル生命体を起動し、指定時間（または無限に）活動させる。
    """
    parser = argparse.ArgumentParser(description="Digital Life Form Orchestrator")
    parser.add_argument("--duration", type=int, default=60, help="実行時間（秒）。0を指定すると無限に実行します。")
    args = parser.parse_args()
    
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    # 必要なモジュールを関数内で局所的にインポート
    from app.containers import AgentContainer, AppContainer
    from snn_research.agent.digital_life_form import DigitalLifeForm
    from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
    from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
    from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
    from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding
    from snn_research.agent.autonomous_agent import AutonomousAgent
    from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
    from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


    # --- 改善: DIコンテナを使用して依存関係を構築 ---
    print("Initializing Digital Life Form with dependencies...")
    agent_container = AgentContainer()
    agent_container.config.from_yaml("configs/base_config.yaml")
    
    app_container = AppContainer()
    app_container.config.from_yaml("configs/base_config.yaml")
    
    # DigitalLifeFormに必要なすべての依存関係をコンテナから取得または手動で作成
    planner = agent_container.hierarchical_planner()
    model_registry = agent_container.model_registry()
    memory = agent_container.memory()
    web_crawler = agent_container.web_crawler()
    rag_system = agent_container.rag_system()
    langchain_adapter = app_container.langchain_adapter()

    autonomous_agent = AutonomousAgent(
        name="AutonomousAgent", planner=planner, model_registry=model_registry, 
        memory=memory, web_crawler=web_crawler
    )
    # rl_agent と self_evolving_agent も同様にコンテナから取得または生成
    rl_agent = ReinforcementLearnerAgent(input_size=4, output_size=4, device="cpu")
    self_evolving_agent = SelfEvolvingAgent(
        name="SelfEvolvingAgent", planner=planner, model_registry=model_registry, 
        memory=memory, web_crawler=web_crawler
    )

    # life_formに必要なその他のコンポーネントをインスタンス化
    motivation_system = IntrinsicMotivationSystem()
    meta_cognitive_snn = MetaCognitiveSNN()
    physics_evaluator = PhysicsEvaluator()
    symbol_grounding = SymbolGrounding(rag_system)


    life_form = DigitalLifeForm(
        autonomous_agent=autonomous_agent,
        rl_agent=rl_agent,
        self_evolving_agent=self_evolving_agent,
        motivation_system=motivation_system,
        meta_cognitive_snn=meta_cognitive_snn,
        memory=memory,
        physics_evaluator=physics_evaluator,
        symbol_grounding=symbol_grounding,
        langchain_adapter=langchain_adapter
    )
    
    try:
        life_form.start()
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down.")
    finally:
        life_form.stop()
        print("DigitalLifeForm has been deactivated.")

if __name__ == "__main__":
    main()