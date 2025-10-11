# ファイルパス: snn_research/agent/reinforcement_learner_agent.py
# Title: 強化学習エージェント
# (省略)
# 修正点:
# - mypyエラー `Incompatible return value type` を解消するため、
#   get_actionの戻り値を明示的にintにキャストした。
#
# 修正点 (v2):
# - 循環インポートエラーを解消するため、app.containers のインポートを削除し、
#   必要な SpikeEncoderDecoder を直接インポートするように変更。

import torch
from typing import Dict, Any, List

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# 循環インポートの原因となるため、DIコンテナのインポートを削除
# from app.containers import TrainingContainer
# 必要なモジュールを直接インポートする
from snn_research.communication import SpikeEncoderDecoder
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

class ReinforcementLearnerAgent:
    """
    BioSNNと報酬変調型STDPを用いて強化学習を行うエージェント。
    """
    def __init__(self, input_size: int, output_size: int, device: str):
        self.device = device
        
        # 生物学的学習則を定義
        learning_rule = RewardModulatedSTDP(
            learning_rate=0.005,
            a_plus=1.0, a_minus=1.0,
            tau_trace=20.0,
            tau_eligibility=50.0 # 短期的な因果関係を重視
        )
        
        # ネットワークの層構造を定義
        hidden_size = (input_size + output_size) * 2
        layer_sizes = [input_size, hidden_size, output_size]
        
        # BioSNNモデルを初期化
        self.model = BioSNN(
            layer_sizes=layer_sizes,
            neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
            learning_rule=learning_rule
        ).to(device)

        # 状態をスパイクに変換するエンコーダー
        self.encoder = SpikeEncoderDecoder(num_neurons=input_size, time_steps=1)

        # 複数ステップの経験を蓄積するためのバッファ
        self.experience_buffer: List[List[torch.Tensor]] = []


    def get_action(self, state: torch.Tensor) -> int:
        """
        現在の状態から、モデルの推論によって単一の行動インデックスを決定する。
        """
        self.model.eval() # 推論モード
        with torch.no_grad():
            # 状態ベクトルをスパイクにエンコード
            # ここでは簡略化のため、状態の各要素をニューロンの発火確率と見なす
            input_spikes = (torch.rand_like(state) < (state * 0.5 + 0.5)).float()
            
            # モデルのフォワードパスを実行
            output_spikes, hidden_spikes_history = self.model(input_spikes)

            # 次の学習ステップのために全層のスパイク活動をバッファに保存
            self.experience_buffer.append([input_spikes] + hidden_spikes_history)

            # 最も発火したニューロンのインデックスを行動として選択
            action = torch.argmax(output_spikes).item()
            return int(action)

    def learn(self, reward: float):
        """
        受け取った報酬信号を用いて、蓄積された経験に基づいてモデルの重みを更新する。
        """
        if not self.experience_buffer:
            return

        self.model.train() # 学習モード
        
        # 報酬を各ステップに分配（ここでは簡略的に最後の報酬を全ステップに適用）
        optional_params = {"reward": reward}
        
        # バッファに蓄積された各タイムステップのスパイク活動を使って重みを更新
        for step_spikes in self.experience_buffer:
            self.model.update_weights(
                all_layer_spikes=step_spikes,
                optional_params=optional_params
            )
        
        # エピソードが終了したらバッファをクリア
        if reward != -0.05: # ゴール時または最大ステップ到達時
            self.experience_buffer = []