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
    command = ["python", "run_agent.py", task]
    if prompt:
        command.extend(["--prompt", prompt])
    if unlabeled_data:
        command.extend(["--unlabeled-data", unlabeled_data])
    if force_retrain:
        command.append("--force-retrain")
    _run_command(command)

@agent_app.command("evolve")
def agent_evolve(
    task: str,
    base_model_config: str = typer.Option("configs/models/small.yaml", help="進化のベースとなるモデル設定"),
    generations: int = typer.Option(3, help="進化の世代数"),
):
    """自己進化エージェントを実行する。"""
    command = ["python", "run_evolution.py", task, "--base_model_config", base_model_config, "--generations", str(generations)]
    _run_command(command)

@agent_app.command("rl")
def agent_rl(
    paradigm: str = typer.Option("bio-causal-sparse", help="強化学習パラダイム"),
    episodes: int = typer.Option(100, help="学習エピソード数"),
):
    """強化学習エージェントを実行する。"""
    command = ["python", "run_rl_agent.py", "--paradigm", paradigm, "--episodes", str(episodes)]
    _run_command(command)

@app.command("life-form")
def life_form():
    """デジタル生命体を起動する。"""
    _run_command(["python", "run_life_form.py"])

@app.command("planner")
def planner(prompt: str):
    """階層的プランナーを実行する。"""
    _run_command(["python", "run_planner.py", "--prompt", prompt])

@app.command("brain")
def brain(prompt: str):
    """人工脳シミュレーションを実行する。"""
    _run_command(["python", "run_brain_simulation.py", "--prompt", prompt])

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
    model_config: str = typer.Option("configs/models/small.yaml", help="使用するモデルの設定ファイル")
):
    """Gradio UIを起動する。"""
    _run_command(["python", "app/main.py", "--model_config", model_config])

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

if __name__ == "__main__":
    app()
