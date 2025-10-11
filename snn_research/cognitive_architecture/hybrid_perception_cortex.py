# ファイルパス: snn_research/cognitive_architecture/hybrid_perception_cortex.py
# (修正)
# 修正: mypyエラーを解消するため、Optional型を明示的にインポート・使用。

import torch
from typing import Dict, Any, Optional

from .som_feature_map import SomFeatureMap

class HybridPerceptionCortex:
    """
    自己組織化マップ(SOM)を統合した、高忠実度な知覚野モジュール。
    """
    def __init__(self, num_neurons: int, feature_dim: int = 64, som_map_size=(8, 8), stdp_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            num_neurons (int): 入力スパイクパターンのニューロン数。
            feature_dim (int): SOMへの入力特徴ベクトルの次元数。
            som_map_size (tuple): SOMのマップサイズ。
            stdp_params (Optional[dict]): SOMが使用するSTDP学習則のパラメータ。
        """
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim
        
        # 特徴抽出のための簡易的な線形層（重み）
        self.input_projection = torch.randn((num_neurons, feature_dim))
        
        # 自己組織化マップを初期化
        if stdp_params is None:
            stdp_params = {'learning_rate': 0.005, 'a_plus': 1.0, 'a_minus': 1.0, 'tau_trace': 20.0}
        
        self.som = SomFeatureMap(
            input_dim=feature_dim,
            map_size=som_map_size,
            stdp_params=stdp_params
        )
        print("🧠 ハイブリッド知覚野モジュールが初期化されました (SOM統合)。")

    def perceive_and_learn(self, spike_pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        入力スパイクを知覚し、SOMの重みを更新（学習）しながら特徴を抽出する。

        Args:
            spike_pattern (torch.Tensor):
                入力スパイクパターン (time_steps, num_neurons)。

        Returns:
            Dict[str, torch.Tensor]:
                SOMの活性化パターンを特徴として含む辞書。
        """
        if spike_pattern.shape[1] != self.num_neurons:
            raise ValueError(f"入力スパイクのニューロン数 ({spike_pattern.shape[1]}) が"
                             f"知覚野のニューロン数 ({self.num_neurons}) と一致しません。")

        # 1. 時間的プーリング: 時間全体のスパイク活動を集約
        temporal_features = torch.sum(spike_pattern, dim=0)

        # 2. 特徴射影: より低次元の特徴空間に射影
        feature_vector = torch.matmul(temporal_features, self.input_projection)
        feature_vector = torch.relu(feature_vector)

        # 3. SOMによる特徴分類と学習
        #    ここではシミュレーションのため、1つの特徴ベクトルで複数回学習ステップを実行
        for _ in range(5): # 簡易的な学習ループ
            som_spikes = self.som(feature_vector)
            self.som.update_weights(feature_vector, som_spikes)
        
        # 最終的なSOMの活性化パターンをこの知覚の結果とする
        final_som_activation = self.som(feature_vector)
        
        print(f"  - SOMが特徴を分類し、勝者ニューロンが発火しました。")
        return {"features": final_som_activation}

