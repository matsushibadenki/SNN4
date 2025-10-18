# ファイルパス: snn_research/core/trm_core.py

# Title: Tiny Recursive Model (TRM) コア実装
#
# Description:
# - HRMの階層構造を廃止し単一の小型SNNブロックによる再帰的推論モデル(TRM)を実装。
# - SNNのタイムステップを推論の再帰ステップとして活用し超低パラメータで推論能力を向上させる。

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Type, cast
from typing import Union # Any, Unionを明示的にインポート

from .neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from .base import BaseModel, SNNLayerNorm
from spikingjelly.activation_based import functional # type: ignore

class TRMBlock(nn.Module):
    """
    TRMにおける単一の小型SNNブロック。
    潜在状態zの更新（再帰的推論）を担う。
    """
    # self.neuron に明示的な型注釈を追加
    neuron: nn.Module

    def __init__(self, input_dim: int, hidden_dim: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        # 入力（質問 x）と現在の回答（y_i）と現在の潜在状態（z_i）を結合して入力とする
        self.input_mix = nn.Linear(input_dim, hidden_dim)
        
        # 潜在状態の非線形更新
        self.recurrent_fc = nn.Linear(hidden_dim, hidden_dim)
        
        neuron_params = neuron_params.copy()
        # LIFやIzhikevichのパラメータを渡すためにtypeキーを除去
        neuron_params.pop('type', None) 
        
        # 修正1: self.neuronに明示的な型アノテーションを付与
        self.neuron = neuron_class(features=hidden_dim, **neuron_params)
        self.norm = SNNLayerNorm(hidden_dim)

    def forward(self, combined_input: torch.Tensor, h_recurrent: torch.Tensor) -> torch.Tensor:
        """
        潜在状態（h_recurrent）を1ステップ更新する。
        """
        # (Combined Input) + (Recurrent State)
        current_input = self.input_mix(combined_input) + self.recurrent_fc(h_recurrent)
        
        # SNNのステップ実行
        h_new_spikes, _ = self.neuron(current_input)
        
        return self.norm(h_new_spikes)

class TinyRecursiveModel(BaseModel):
    """
    Tiny Recursive Model (TRM): 単一ブロックを時間軸で再帰させるモデル。
    """
    # self.lif_out に明示的な型注釈を追加
    lif_out: nn.Module
    
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, layer_dims: List[int], time_steps: int, neuron_config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        # TRMではnum_layers, layer_dimsは使用しないが互換性のためコンストラクタに残す
        self.time_steps = time_steps # これが再帰の深さ N_sup (思考ステップ数) に相当
        self.d_model = d_model       # 質問埋め込み後の次元 (x)
        self.d_state = d_state       # 潜在状態 z と回答 y の次元
        
        neuron_type = neuron_config.get("type", "lif")
        neuron_class = AdaptiveLIFNeuron if neuron_type == 'lif' else IzhikevichNeuron
        
        # 1. 入力層
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. メインの再帰ブロック（単一）
        # 入力次元 = d_model (x) + d_state (y_i) + d_state (z_i)
        input_dim_for_block = d_model + d_state + d_state 
        self.recurrent_block = TRMBlock(input_dim_for_block, d_state, neuron_class, neuron_config)
        
        # 3. 初期回答と潜在状態の初期化層
        self.init_state = nn.Linear(d_model, d_state * 2) 
        
        # 4. 出力層（最終的なロジットを生成する）
        # 入力次元: d_state (更新された潜在状態 z)
        self.output_projection = nn.Linear(d_state, vocab_size)
        
        # 修正2: self.lif_out の引数渡しを明確にし、型の曖昧さを解消
        lif_out_config = neuron_config.copy()
        lif_out_config.pop('type', None)
        self.lif_out = neuron_class(features=vocab_size, **lif_out_config)

        self._init_weights()


    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device
        functional.reset_net(self)
        
        if L != 1:
            # TRMは単一の質問ベクトルを入力として受け取ることを想定
            input_ids = input_ids[:, 0].unsqueeze(1) 
            L = 1

        # 1. 初期埋め込み (x)
        x_emb = self.embedding(input_ids).squeeze(1) # B x D_model

        # 2. 初期状態の生成 (y_0, z_0)
        initial_state_and_answer = self.init_state(x_emb)
        # z_0 (潜在状態)とy_0 (初期回答/次の入力)を分離
        latent_z, answer_y_spikes = initial_state_and_answer.split(self.d_state, dim=-1)
        answer_y = answer_y_spikes # 初期はアナログ値から開始

        # 再帰ブロック内のニューロンをstatefulモードに設定し状態をリセット
        # 修正3: cast(nn.Module, ...) -> cast(Any, ...) に変更し型チェックを回避
        trm_block_neuron = cast(Any, self.recurrent_block.neuron)
        if hasattr(trm_block_neuron, 'set_stateful'):
            trm_block_neuron.set_stateful(True)
        
        # 出力ニューロンもstatefulに設定し状態をリセット
        lif_out_neuron = cast(Any, self.lif_out)
        if hasattr(lif_out_neuron, 'set_stateful'):
            lif_out_neuron.set_stateful(True)
        
        # 念のため状態をリセット
        if hasattr(trm_block_neuron, 'reset'):
            trm_block_neuron.reset()
        if hasattr(lif_out_neuron, 'reset'):
            lif_out_neuron.reset()

        # 3. 再帰的推論ループ
        for t in range(self.time_steps):
            # a) 再帰ブロックへの入力準備: x + y_t + z_t
            combined_input = torch.cat([x_emb, answer_y, latent_z], dim=-1) # B x (D_model + 2*D_state)
            
            # b) 潜在状態 z の更新 (リカレント入力として z_t を受け取る)
            latent_z = self.recurrent_block(combined_input, latent_z) # B x D_state
            
            # c) 回答 y の更新 (更新された潜在状態 z_new から新しい回答 y_new を生成)
            logits_t_raw = self.output_projection(latent_z)
            
            # SNN化: ロジットをスパイクパターン（次のステップの回答 y）に変換
            answer_y_spikes, _ = self.lif_out(logits_t_raw)
            answer_y = answer_y_spikes # 次の再帰ステップへの入力となる

        # 再帰ループ終了後ニューロンの状態をリセット
        if hasattr(trm_block_neuron, 'set_stateful'):
            trm_block_neuron.set_stateful(False)
            trm_block_neuron.reset()
        if hasattr(lif_out_neuron, 'set_stateful'):
            lif_out_neuron.set_stateful(False)
            lif_out_neuron.reset()

        # 4. 最終出力 (最終ステップの潜在状態からロジットを生成)
        final_logits = self.output_projection(latent_z)        
        # スパイク統計の計算
        total_spikes = self.get_total_spikes()
        avg_spikes_val = total_spikes / (L * self.time_steps * B) if return_spikes else 0.0
        
        # 1. avg_spikes を作成 (1要素のスカラーテンソル)
        avg_spikes = torch.zeros(1, dtype=torch.float32, device=device).squeeze()
        avg_spikes.fill_(float(avg_spikes_val))
        
        # 2. mem を作成
        mem = torch.zeros(1, dtype=torch.float32, device=device).squeeze()
        # ==============================================================================
        
        # SNN Coreのインターフェースに合わせて (B, L, V) で返す
        return final_logits.unsqueeze(1), avg_spikes, mem
