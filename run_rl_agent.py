# ファイルパス: matsushibadenki/snn4/run_rl_agent.py
# Title: 強化学習エージェント実行スクリプト
# Description: 生物学的学習則に基づくSNNエージェントを起動し、
#              強化学習のループを実行します。
# 改善点:
# - ROADMAPフェーズ2検証のため、GridWorldEnvに対応。
# - エピソードベースの学習ループを実装し、複数ステップのタスクを実行できるようにした。
# 改善点 (v2):
# - ROADMAPフェーズ2完了のため、学習結果を可視化・保存する機能を追加。
# - 学習終了後に報酬の推移をグラフとしてプロットし、画像ファイルとして保存。
# - 訓練済みのエージェントモデルをファイルに保存。

import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.rl_env.grid_world import GridWorldEnv

def plot_rewards(rewards: list, save_path: Path):
    """報酬の推移をプロットして保存する。"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode')
    # 移動平均を計算してプロット
    moving_avg = [sum(rewards[i-50:i]) / 50 for i in range(50, len(rewards))]
    plt.plot(range(50, len(rewards)), moving_avg, label='Moving Average (50 episodes)', color='red')
    plt.title('Reinforcement Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📊 学習曲線グラフを '{save_path}' に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="Biologically Plausible Reinforcement Learning Framework")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of learning episodes.")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid world.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode.")
    parser.add_argument("--output_dir", type=str, default="runs/rl_results", help="Directory to save results.")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = GridWorldEnv(size=args.grid_size, max_steps=args.max_steps, device=device)
    agent = ReinforcementLearnerAgent(input_size=4, output_size=4, device=device)

    print("\n" + "="*20 + "🤖 生物学的強化学習開始 (Grid World) 🤖" + "="*20)
    
    progress_bar = tqdm(range(args.episodes))
    total_rewards = []

    for episode in progress_bar:
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(reward)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        avg_reward = sum(total_rewards[-20:]) / len(total_rewards[-20:])
        
        progress_bar.set_description(f"Episode {episode+1}/{args.episodes}")
        progress_bar.set_postfix({"Last Reward": f"{episode_reward:.2f}", "Avg Reward (last 20)": f"{avg_reward:.3f}"})

    final_avg_reward = sum(total_rewards) / args.episodes if args.episodes > 0 else 0.0
    print("\n" + "="*20 + "✅ 学習完了" + "="*20)
    print(f"最終的な平均報酬: {final_avg_reward:.4f}")

    # 学習結果のプロットとモデルの保存
    plot_rewards(total_rewards, output_path / "rl_learning_curve.png")
    
    model_save_path = output_path / "trained_rl_agent.pth"
    torch.save(agent.model.state_dict(), model_save_path)
    print(f"💾 訓練済みモデルを '{model_save_path}' に保存しました。")


if __name__ == "__main__":
    main()