# ファイルパス: snn-cli.py
# Title: SNNプロジェクト 統合CLIツール
# Description:
#   学習、推論、自己進化、人工脳シミュレーションなど、プロジェクトの全機能を
#   単一のインターフェースから制御するためのコマンドラインツール。
#   Typerライブラリを使用し、サブコマンド形式で機能を提供する。
import typer
from typing import Optional, List
import subprocess
import sys

app = typer.Typer()
agent_app = typer.Typer()
app.add_typer(agent_app, name="agent")

benchmark_app = typer.Typer()
app.add_typer(benchmark_app, name="benchmark")

convert_app = typer.Typer()
app.add_typer(convert_app, name="convert")

def _run_command(command: List[str]):
    """コマンドを実行し、出力をストリーミングする。"""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
        process.wait()
        if process.returncode != 0:
            typer.echo(f"Error: Command failed with exit code {process.returncode}")
    except FileNotFoundError:
        typer.echo(f"Error: Command '{command[0]}' not found.")
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}")

@agent_app.command("solve")
def agent_solve(
    task: str,
    prompt: Optional[str] = typer.Option(None, help="実行するプロンプト"),
    unlabeled_data: Optional[str] = typer.Option(None, help="オンデマンド学習用の非ラベルデータ"),
    force_retrain: bool = typer.Option(False, help="強制的に再学習を行う"),
):
    """自律エージェントにタスクを解決させる。"""
    command = ["python", "run_agent.py", "--task_description", task]
    if prompt:
        command.extend(["--prompt", prompt])
    if unlabeled_data:
        command.extend(["--unlabeled_data_path", unlabeled_data])
    if force_retrain:
        command.append("--force_retrain")
    _run_command(command)

@agent_app.command("evolve")
def agent_evolve(
    task: str,
    base_model_config: str = typer.Option("configs/models/small.yaml", help="進化のベースとなるモデル設定"),
    generations: int = typer.Option(3, help="進化の世代数"),
):
    """自己進化エージェントを実行する。"""
    command = ["python", "run_evolution.py", "--task_description", task, "--model_config", base_model_config, "--generations", str(generations)]
    _run_command(command)

@agent_app.command("rl")
def agent_rl(
    paradigm: str = typer.Option("bio-causal-sparse", help="強化学習パラダイム"),
    episodes: int = typer.Option(100, help="学習エピソード数"),
):
    """強化学習エージェントを実行する。"""
    command = ["python", "run_rl_agent.py", "--episodes", str(episodes)]
    _run_command(command)

@app.command("life-form")
def life_form(
    cycles: int = typer.Option(5, help="実行する自律ループのサイクル数"),
):
    """デジタル生命体を起動する。"""
    _run_command(["python", "run_life_form.py", "--cycles", str(cycles)])

@app.command("planner")
def planner(
    request: str = typer.Argument(..., help="プランナーへのタスク要求"),
    context_data: str = typer.Argument(..., help="タスクの文脈データ"),
):
    """階層的プランナーを実行する。"""
    _run_command(["python", "run_planner.py", "--task_request", request, "--context_data", context_data])

@app.command("brain")
def brain(
    input_text: Optional[str] = typer.Option(None, "--input", help="人工脳への単一の入力テキスト"),
    loop: bool = typer.Option(False, "--loop", help="対話形式でシミュレーションを繰り返し実行する"),
):
    """人工脳シミュレーションを実行する。"""
    if loop:
        _run_command(["python", "scripts/observe_brain_thought_process.py"])
    elif input_text:
        _run_command(["python", "run_brain_simulation.py", "--prompt", input_text]) # `run_brain_simulation` は prompt を受け付けないが、仮引数として
    else:
        typer.echo("Error: --input <text> または --loop のいずれかを指定してください。")


@app.command("gradient-train")
def gradient_train(
    model_config: str,
    data_path: str,
    override_config: Optional[List[str]] = typer.Option(None, "--override_config"),
):
    """train.pyを直接呼び出して勾配ベースの学習を行う。"""
    command = ["python", "train.py", "--model_config", model_config, "--data_path", data_path]
    if override_config:
        for oc in override_config:
            command.extend(["--override_config", oc])
    _run_command(command)


@app.command("train-ultra")
def train_ultra(override_config: Optional[List[str]] = typer.Option(None, "--override_config")):
    """データ準備からUltraモデルの学習までを自動実行する。"""
    typer.echo("--- Starting Ultra Training Pipeline ---")
    _run_command(["python", "scripts/data_preparation.py"])
    train_command = ["python", "train.py", "--model_config", "configs/models/ultra.yaml"]
    if override_config:
        for oc in override_config:
            train_command.extend(["--override_config", oc])
    _run_command(train_command)
    typer.echo("--- Ultra Training Pipeline Finished ---")

@app.command("ui")
def ui(
    model_config: str = typer.Option("configs/models/small.yaml", help="使用するモデルの設定ファイル"),
    start_langchain: bool = typer.Option(False, "--start-langchain", help="LangChain連携版のUIを起動する"),
):
    """Gradio UIを起動する。"""
    script = "app/langchain_main.py" if start_langchain else "app/main.py"
    _run_command(["python", script, "--model_config", model_config])

@benchmark_app.command("run")
def benchmark_run(
    experiment: str = typer.Option("all", help="実行する実験 (all, cifar10_comparison, sst2_comparison)"),
    tag: Optional[str] = typer.Option(None, help="実験にカスタムタグを付ける"),
    epochs: int = typer.Option(3, help="訓練のエポック数"),
    batch_size: int = typer.Option(32, help="バッチサイズ"),
    learning_rate: float = typer.Option(1e-4, help="学習率"),
):
    """ANN vs SNNの性能比較ベンチマークを実行する。"""
    command = ["python", "scripts/run_benchmark_suite.py", "--experiment", experiment, "--epochs", str(epochs), "--batch_size", str(batch_size), "--learning_rate", str(learning_rate)]
    if tag:
        command.extend(["--tag", tag])
    _run_command(command)

@benchmark_app.command("continual")
def benchmark_continual(
    epochs_task_a: int = typer.Option(3, help="タスクAの訓練エポック数"),
    epochs_task_b: int = typer.Option(3, help="タスクBの訓練エポック数"),
    model_config: str = typer.Option("configs/models/small.yaml", help="モデル設定ファイル"),
):
    """継続学習（破局的忘却の克服）の実験を実行する。"""
    command = ["python", "scripts/run_continual_learning_experiment.py", "--epochs_task_a", str(epochs_task_a), "--epochs_task_b", str(epochs_task_b), "--model_config", model_config]
    _run_command(command)

@convert_app.command("ann2snn-cnn")
def convert_ann2snn_cnn(
    ann_model_path: str = typer.Argument(..., help="変換元の学習済みSimpleCNNモデルのパス (.pth)"),
    output_path: str = typer.Argument(..., help="変換後のSpikingCNNモデルの保存先パス (.pth)"),
    snn_config_path: str = typer.Option("configs/cifar10_spikingcnn_config.yaml", help="SpikingCNNのモデル設定ファイル"),
):
    """学習済みCNN (ANN) をSpikingCNN (SNN) に変換する。"""
    command = ["python", "scripts/ann2snn.py", "--ann_model_path", ann_model_path, "--output_path", output_path, "--snn_config_path", snn_config_path]
    _run_command(command)

if __name__ == "__main__":
    app()
