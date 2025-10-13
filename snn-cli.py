# matsushibadenki/snn4/snn4-79496245059a9838ecdcdf953e28024581f28ba2/snn-cli.py
# Title: 統合CLIツール
# Description:
# - プロジェクトの全機能を一元的に管理・実行するためのコマンドラインインターフェース。
#
# 修正点 (v13):
# - life-formコマンド実行時のTypeErrorを解消。get_life_form_instanceがモデル設定ファイルを
#   読み込むように修正し、PlannerSNNの初期化に必要なパラメータが渡されるようにした。
#
# 改善点 (v14):
# - 性能証明のワークフローを統一するため、`benchmark`コマンドグループを正式に実装。
#   - `benchmark train`: 分類タスク用のモデルを訓練する。
#   - `benchmark run`: 訓練済みモデルでベンチマークを測定する。
#
# 修正点 (v15):
# - SyntaxErrorを解消するため、ファイル末尾に誤って混入していたPythonコード以外の説明文を削除。
#
# 改善点 (v16):
# - スパイクベースの通信タスクを実行するための `emergent-system communicate` コマンドを追加。

import sys
from pathlib import Path
import asyncio
import torch
import typer
from typing import List, Optional

# --- プロジェクトルートをPythonパスに追加 ---
sys.path.append(str(Path(__file__).resolve().parent))


# --- CLIアプリケーションの定義 ---
app = typer.Typer(
    help="Project SNN: 統合CLIツール",
    rich_markup_mode="markdown",
    add_completion=False
)

# --- サブコマンドグループの作成 ---
agent_app = typer.Typer(help="自律エージェントを操作して単一タスクを実行")
app.add_typer(agent_app, name="agent")

planner_app = typer.Typer(help="高次認知プランナーを操作して複雑なタスクを実行")
app.add_typer(planner_app, name="planner")

life_form_app = typer.Typer(help="デジタル生命体の自律ループを開始")
app.add_typer(life_form_app, name="life-form")

evolve_app = typer.Typer(help="自己進化サイクルを実行")
app.add_typer(evolve_app, name="evolve")

rl_app = typer.Typer(help="生物学的強化学習を実行")
app.add_typer(rl_app, name="rl")

ui_app = typer.Typer(help="Gradioベースの対話UIを起動")
app.add_typer(ui_app, name="ui")

emergent_app = typer.Typer(help="創発的なマルチエージェントシステムを操作")
app.add_typer(emergent_app, name="emergent-system")

brain_app = typer.Typer(help="人工脳シミュレーションを直接制御")
app.add_typer(brain_app, name="brain")

benchmark_app = typer.Typer(help="ベンチマークの実行と関連タスク")
app.add_typer(benchmark_app, name="benchmark")


@agent_app.command("solve", help="指定されたタスクを解決します。専門家モデルの検索、オンデマンド学習、推論を実行します。")
def agent_solve(
    task: str = typer.Option(..., help="タスクの自然言語説明 (例: '感情分析')"),
    prompt: Optional[str] = typer.Option(None, help="推論を実行する場合の入力プロンプト"),
    unlabeled_data: Optional[Path] = typer.Option(None, help="新規学習時に使用するデータパス", exists=True, file_okay=True, dir_okay=False),
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
    force_retrain: bool = typer.Option(False, "--force-retrain", help="モデル登録簿を無視して強制的に再学習"),
    min_accuracy: float = typer.Option(0.6, help="専門家モデルを選択するための最低精度要件"),
    max_spikes: float = typer.Option(10000.0, help="専門家モデルを選択するための平均スパイク数上限")
):
    from app.containers import AgentContainer
    from snn_research.agent.autonomous_agent import AutonomousAgent
    
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(str(model_config))
    
    agent = AutonomousAgent(
        name="cli-agent",
        planner=container.hierarchical_planner(),
        model_registry=container.model_registry(),
        memory=container.memory(),
        web_crawler=container.web_crawler(),
        accuracy_threshold=min_accuracy,
        energy_budget=max_spikes
    )
    
    selected_model_info = asyncio.run(agent.handle_task(
        task_description=task,
        unlabeled_data_path=str(unlabeled_data) if unlabeled_data else None,
        force_retrain=force_retrain
    ))
    
    if selected_model_info and prompt:
        print("\n" + "="*20 + " 🧠 INFERENCE " + "="*20)
        print(f"入力プロンプト: {prompt}")
        asyncio.run(agent.run_inference(selected_model_info, prompt))
    elif not selected_model_info:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。")

@planner_app.command("execute", help="複雑なタスク要求を実行します。内部で計画を立案し、複数の専門家を連携させます。")
def planner_execute(
    request: str = typer.Option(..., help="タスク要求 (例: '記事を要約して感情を分析')"),
    context: str = typer.Option(..., help="処理対象のデータ")
):
    from app.containers import AgentContainer
    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    planner = container.hierarchical_planner()
    
    final_result = planner.execute_task(task_request=request, context=context)
    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)


@life_form_app.command("start", help="意識ループを開始します。AIが自律的に思考・学習します。")
def life_form_start(
    cycles: int = typer.Option(5, help="実行する意識サイクルの回数"),
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
):
    from app.containers import BrainContainer
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(str(model_config))
    life_form = container.digital_life_form()
    life_form.awareness_loop(cycles=cycles)

@life_form_app.command("explain-last-action", help="AI自身に、直近の行動理由を自然言語で説明させます。")
def life_form_explain(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
):
    from app.containers import BrainContainer
    print("🤔 AIに自身の行動理由を説明させます...")
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(str(model_config))
    life_form = container.digital_life_form()
    explanation = life_form.explain_last_action()
    print("\n" + "="*20 + " 🤖 AIによる自己解説 " + "="*20)
    if explanation:
        print(explanation)
    else:
        print("説明の生成に失敗しました。")
    print("="*64)

@evolve_app.command("run", help="自己進化サイクルを1回実行します。AIが自身の性能を評価し、アーキテクチャを改善します。")
def evolve_run(
    task_description: str = typer.Option(..., help="自己評価の起点となるタスク説明"),
    training_config: Path = typer.Option("configs/base_config.yaml", help="進化対象の基本設定ファイル", exists=True),
    model_config: Path = typer.Option("configs/models/small.yaml", help="進化対象のモデル設定ファイル", exists=True),
    initial_accuracy: float = typer.Option(0.75, help="自己評価のための初期精度"),
    initial_spikes: float = typer.Option(1500.0, help="自己評価のための初期スパイク数")
):
    from app.containers import AgentContainer
    from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
    container = AgentContainer()
    container.config.from_yaml(str(training_config))
    container.config.from_yaml(str(model_config))

    agent = SelfEvolvingAgent(
        name="evolving-agent",
        planner=container.hierarchical_planner(),
        model_registry=container.model_registry(),
        memory=container.memory(),
        web_crawler=container.web_crawler(),
        project_root=".",
        model_config_path=str(model_config),
        training_config_path=str(training_config)
    )
    initial_metrics = {
        "accuracy": initial_accuracy,
        "avg_spikes_per_sample": initial_spikes
    }
    agent.run_evolution_cycle(
        task_description=task_description,
        initial_metrics=initial_metrics
    )


@rl_app.command("run", help="強化学習ループを開始します。エージェントがGridWorld環境を探索します。")
def rl_run(
    episodes: int = typer.Option(500, help="学習エピソード数"),
    grid_size: int = typer.Option(5, help="グリッドワールドのサイズ"),
    max_steps: int = typer.Option(50, help="1エピソードあたりの最大ステップ数"),
    output_dir: str = typer.Option("runs/rl_results_cli", help="結果を保存するディレクトリ"),
):
    import subprocess
    print(f"🚀 強化学習スクリプト 'run_rl_agent.py' を呼び出します...")
    
    command = [
        sys.executable, # 現在のPythonインタプリタを使用
        "run_rl_agent.py",
        "--episodes", str(episodes),
        "--grid_size", str(grid_size),
        "--max_steps", str(max_steps),
        "--output_dir", output_dir
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"\n✅ 強化学習が完了しました。結果は '{output_dir}' を確認してください。")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n❌ 強化学習スクリプトの実行に失敗しました: {e}")
        print("   プロジェクトルートからコマンドを実行していることを確認してください。")

@ui_app.command("start", help="標準のGradio UIを起動します。")
def ui_start(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
    model_path: Optional[str] = typer.Option(None, help="モデルのパス（設定ファイルを上書き）"),
):
    import app.main as gradio_app
    original_argv = sys.argv
    sys.argv = [
        "app/main.py",
        "--model_config", str(model_config),
    ]
    if model_path:
        sys.argv.extend(["--model_path", model_path])
    
    try:
        print("🚀 標準のGradio UIを起動します...")
        gradio_app.main()
    finally:
        sys.argv = original_argv

@ui_app.command("start-langchain", help="LangChain連携版のGradio UIを起動します。")
def ui_start_langchain(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
    model_path: Optional[str] = typer.Option(None, help="モデルのパス（設定ファイルを上書き）"),
):
    import app.langchain_main as langchain_gradio_app
    original_argv = sys.argv
    sys.argv = [
        "app/langchain_main.py",
        "--model_config", str(model_config),
    ]
    if model_path:
        sys.argv.extend(["--model_path", model_path])

    try:
        print("🚀 LangChain連携版のGradio UIを起動します...")
        langchain_gradio_app.main()
    finally:
        sys.argv = original_argv

@emergent_app.command("execute", help="高レベルの目標を与え、マルチエージェントシステムに協調的に解決させます。")
def emergent_execute(
    goal: str = typer.Option(..., help="システムに達成させたい高レベルの目標")
):
    from app.containers import AgentContainer
    from snn_research.agent.autonomous_agent import AutonomousAgent
    from snn_research.cognitive_architecture.emergent_system import EmergentCognitiveSystem
    from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
    print(f"🚀 Emergent System Activated. Goal: {goal}")

    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")

    planner = container.hierarchical_planner()
    model_registry = container.model_registry()
    memory = container.memory()
    web_crawler = container.web_crawler()
    
    global_workspace = GlobalWorkspace(model_registry=model_registry)

    agent1 = AutonomousAgent(name="AutonomousAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    agent2 = AutonomousAgent(name="SpecialistAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    
    emergent_system = EmergentCognitiveSystem(
        planner=planner,
        agents=[agent1, agent2],
        global_workspace=global_workspace,
        model_registry=model_registry
    )

    final_report = emergent_system.execute_task(goal)

    print("\n" + "="*20 + " ✅ FINAL REPORT " + "="*20)
    print(final_report)
    print("="*60)

# --- ◾️◾️◾️◾️◾️↓コマンド追加↓◾️◾️◾️◾️◾️ ---
@emergent_app.command("communicate", help="エージェント間のスパイクベース通信タスクを実行します。")
def emergent_communicate():
    """スパイク通信の協調タスクを実行する。"""
    from app.containers import AgentContainer
    from snn_research.agent.autonomous_agent import AutonomousAgent
    from snn_research.cognitive_architecture.emergent_system import EmergentCognitiveSystem
    from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace

    container = AgentContainer()
    container.config.from_yaml("configs/base_config.yaml")
    
    planner = container.hierarchical_planner()
    model_registry = container.model_registry()
    memory = container.memory()
    web_crawler = container.web_crawler()
    global_workspace = GlobalWorkspace(model_registry=model_registry)

    agent1 = AutonomousAgent(name="AutonomousAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    agent2 = AutonomousAgent(name="SpecialistAgent", planner=planner, model_registry=model_registry, memory=memory, web_crawler=web_crawler)
    
    emergent_system = EmergentCognitiveSystem(
        planner=planner,
        agents=[agent1, agent2],
        global_workspace=global_workspace,
        model_registry=model_registry
    )
    
    asyncio.run(emergent_system.run_cooperative_observation_task())
# --- ◾️◾️◾️◾️◾️↑コマンド追加↑◾️◾️◾️◾️◾️ ---


@brain_app.command("run", help="単一の入力で人工脳の認知サイクルを1回実行します。")
def brain_run(
    input_text: str = typer.Option(..., help="人工脳への感覚入力（テキスト）"),
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
):
    """人工脳シミュレーションを1サイクル実行する。"""
    from app.containers import BrainContainer
    
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(str(model_config))
    
    brain = container.artificial_brain()
    brain.run_cognitive_cycle(input_text)
    print("\n✅ 人工脳の認知サイクルが1回完了しました。")

@brain_app.command("loop", help="対話形式で人工脳の認知サイクルを繰り返し実行します。")
def brain_loop(
    model_config: Path = typer.Option("configs/models/small.yaml", help="モデルアーキテクチャ設定ファイル", exists=True),
):
    """人工脳シミュレーションの対話ループを開始する。"""
    from app.containers import BrainContainer
    
    container = BrainContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(str(model_config))
    
    brain = container.artificial_brain()
    
    print("🧠 人工脳との対話ループを開始します。終了するには 'exit' または Ctrl+C を入力してください。")
    while True:
        try:
            input_text = input("> ")
            if input_text.lower() == 'exit':
                break
            brain.run_cognitive_cycle(input_text)
        except KeyboardInterrupt:
            break
    print("\n👋 対話ループを終了しました。")

@benchmark_app.command("train", help="ベンチマーク用の分類モデルを訓練します。")
def benchmark_train(
    task: str = typer.Option("sst2", help="訓練対象のタスク名 (例: sst2)"),
    epochs: int = typer.Option(5, help="訓練エポック数"),
):
    """scripts/train_classifier.py を実行するラッパー。"""
    import scripts.train_classifier as classifier_trainer
    original_argv = sys.argv
    sys.argv = [
        "scripts/train_classifier.py",
        "--task", task,
        "--epochs", str(epochs),
    ]
    try:
        classifier_trainer.main()
    finally:
        sys.argv = original_argv

@benchmark_app.command("run", help="訓練済みモデルでベンチマークを測定します。")
def benchmark_run(
    task: str = typer.Option("sst2", help="評価対象のタスク名 (例: sst2)"),
    model_path: Optional[Path] = typer.Option(None, help="評価する訓練済みモデルのパス"),
):
    """scripts/run_benchmark.py を実行するラッパー。"""
    import scripts.run_benchmark as benchmark_runner
    original_argv = sys.argv
    
    run_args = [
        "scripts/run_benchmark.py",
        "--task", task,
    ]
    if model_path:
        run_args.extend(["--model_path", str(model_path)])

    sys.argv = run_args
    try:
        benchmark_runner.main()
    finally:
        sys.argv = original_argv

@app.command(
    "gradient-train",
    help="""
    勾配ベースでSNNモデルを手動学習します (train.pyを呼び出します)。
    このコマンドの後に、train.pyに渡したい引数をそのまま続けてください。
    
    例: `python snn-cli.py gradient-train --model_config configs/models/large.yaml --data_path data/sample_data.jsonl`
    """,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def gradient_train(ctx: typer.Context):
    import train as gradient_based_trainer
    print("🔧 勾配ベースの学習プロセスを開始します...")
    train_args = ctx.args
    
    original_argv = sys.argv
    sys.argv = ["train.py"] + train_args
    
    try:
        gradient_based_trainer.main()
    finally:
        sys.argv = original_argv


@app.command(
    "train-ultra",
    help="""
    🚀 **最強のエンジン（Ultraモデル）**を学習します。
    
    プロジェクトで利用可能な最大規模のSpiking Transformer（configs/models/ultra.yaml）を、
    大規模データセット（wikitext-103）を用いて本格的に学習させます。
    """,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_ultra_model(ctx: typer.Context):
    """
    最強設定（ultra.yaml）で学習プロセスを開始するラッパーコマンド。
    """
    import train as gradient_based_trainer
    from scripts.data_preparation import prepare_wikitext_data
    
    print("--------------------------------------------------")
    print("🚀 「最強のエンジン」の学習プロセスを開始します...")
    print("--------------------------------------------------")

    # ステップ1: 大規模データセットの準備
    print("\n[ステップ1/2] 大規模データセット（wikitext-103）を準備しています...")
    wikitext_path = prepare_wikitext_data()
    print(f"✅ データセット準備完了: {wikitext_path}")

    # ステップ2: 学習の開始
    print("\n[ステップ2/2] train.pyを呼び出し、Ultraモデルの学習を開始します...")
    
    # train.pyに渡す引数を構築
    train_args = [
        "--model_config", "configs/models/ultra.yaml",
        "--data_path", wikitext_path,
        "--paradigm", "gradient_based"
    ] + ctx.args # ユーザーが追加で渡した引数（--override_configなど）も反映

    original_argv = sys.argv
    sys.argv = ["train.py"] + train_args
    
    try:
        gradient_based_trainer.main()
        print("\n🎉 「最強のエンジン」の学習が完了しました！")
        print("次に、'snn-cli.py ui start --model_config configs/models/ultra.yaml' を実行して対話ができます。")
    except Exception as e:
        print(f"\n❌ 学習中にエラーが発生しました: {e}")
    finally:
        sys.argv = original_argv
        
        
if __name__ == "__main__":
    app()

