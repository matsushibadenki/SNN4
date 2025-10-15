# ファイルパス: snn_research/training/losses.py
# (更新)
# 改善点: 学習初期段階での妨げとなる可能性があるmem_reg_lossを無効化。
# 改善点(v2): 継続学習のためのElastic Weight Consolidation (EWC) 損失を追加。
# 改善点(v3): スパース性を促す正則化項(sparsity_reg_weight)を追加し、汎化性能を向上。
# 改善点(snn_4_ann_parity_plan):
# - temporal/latency codingの導入として、時間的圧縮を促す正則化項
#   (temporal_compression_weight)を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import PreTrainedTokenizerBase

class CombinedLoss(nn.Module):
    """クロスエントロピー損失、各種正則化、EWC損失を組み合わせた損失関数。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 1.0, spike_reg_weight: float = 0.0, mem_reg_weight: float = 0.0, sparsity_reg_weight: float = 0.0, temporal_compression_weight: float = 0.0, target_spike_rate: float = 0.02, ewc_weight: float = 0.0, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight, 
            'spike_reg': spike_reg_weight, 
            'mem_reg': mem_reg_weight, 
            'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight,
            'ewc': ewc_weight
        }
        self.target_spike_rate = target_spike_rate
        # EWCのためのFisher情報行列と最適パラメータを保持
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        # L1正則化によるスパース損失
        sparsity_loss = torch.mean(torch.abs(spikes))

        mem_reg_loss = torch.mean(mem**2)
        
        # 時間的圧縮損失: 時間の経過とともにスパイクにペナルティ
        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            # スパイクテンソルの次元に合わせて拡張
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        # EWC損失の計算
        ewc_loss = torch.tensor(0.0, device=logits.device)
        if self.weights['ewc'] > 0 and self.fisher_matrix:
            for name, param in model.named_parameters():
                if name in self.fisher_matrix and param.requires_grad:
                    fisher = self.fisher_matrix[name].to(param.device)
                    opt_param = self.optimal_params[name].to(param.device)
                    ewc_loss += (fisher * (param - opt_param)**2).sum()
        
        total_loss = (self.weights['ce'] * ce_loss + 
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss +
                      self.weights['ewc'] * ewc_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate,
            'temporal_compression_loss': temporal_compression_loss,
            'ewc_loss': ewc_loss
        }

# (以降のDistillationLossなどのクラスは変更なし)
class DistillationLoss(nn.Module):
    """知識蒸留のための損失関数（各種正則化付き）。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 0.3, distill_weight: float = 0.7,
                 spike_reg_weight: float = 0.01, mem_reg_weight: float = 0.0, sparsity_reg_weight: float = 0.00001, 
                 temporal_compression_weight: float = 0.0, temperature: float = 2.0, target_spike_rate: float = 0.02, **kwargs):
        super().__init__()
        student_pad_id = tokenizer.pad_token_id
        self.temperature = temperature
        self.weights = {
            'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight, 
            'mem_reg': mem_reg_weight, 'sparsity_reg': sparsity_reg_weight,
            'temporal_compression': temporal_compression_weight
        }
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id if student_pad_id is not None else -100)
        self.distill_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)
        self.target_spike_rate = target_spike_rate

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:

        assert student_logits.shape == teacher_logits.shape, \
            f"Shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}"

        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss_unreduced = self.distill_loss_fn(soft_student_log_probs, soft_teacher_log_probs).sum(dim=-1)
        
        if attention_mask is None:
            mask = (targets != self.ce_loss_fn.ignore_index)
        else:
            mask = attention_mask.bool()
            
        masked_distill_loss = distill_loss_unreduced.where(mask, torch.tensor(0.0, device=distill_loss_unreduced.device))
        
        num_valid_tokens = mask.sum()
        if num_valid_tokens > 0:
            distill_loss = masked_distill_loss.sum() / num_valid_tokens
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)
            
        distill_loss = distill_loss * (self.temperature ** 2)

        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(self.target_spike_rate, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        sparsity_loss = torch.mean(torch.abs(spikes))
        
        mem_reg_loss = torch.mean(mem**2)
        
        temporal_compression_loss = torch.tensor(0.0, device=spikes.device)
        if self.weights['temporal_compression'] > 0 and spikes.ndim > 1 and spikes.shape[1] > 1:
            time_steps = spikes.shape[1]
            time_weights = torch.linspace(0, 1, time_steps, device=spikes.device).view(1, -1, 1)
            if spikes.ndim > 3:
                time_weights = time_weights.view(1, time_steps, 1, 1)
            temporal_compression_loss = (spikes * time_weights).mean()

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss +
                      self.weights['mem_reg'] * mem_reg_loss +
                      self.weights['temporal_compression'] * temporal_compression_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss,
            'sparsity_loss': sparsity_loss, 'mem_reg_loss': mem_reg_loss,
            'temporal_compression_loss': temporal_compression_loss
        }
        
class SelfSupervisedLoss(nn.Module):
    """
    時間的自己教師あり学習のための損失関数。
    """
    def __init__(self, prediction_weight: float, spike_reg_weight: float, mem_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.prediction_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'prediction': prediction_weight,
            'spike_reg': spike_reg_weight,
            'mem_reg': mem_reg_weight
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        prediction_loss = self.prediction_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        mem_reg_loss = torch.mean(mem**2)
        
        total_loss = (self.weights['prediction'] * prediction_loss + 
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_reg'] * mem_reg_loss)
        
        return {
            'total': total_loss, 'prediction_loss': prediction_loss,
            'spike_reg_loss': spike_reg_loss, 
            'mem_reg_loss': mem_reg_loss, 'spike_rate': spike_rate
        }


class PhysicsInformedLoss(nn.Module):
    """
    物理法則（膜電位の滑らかさ）を制約として組み込んだ損失関数。
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float = 1.0, spike_reg_weight: float = 0.0, mem_smoothness_weight: float = 0.0, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {
            'ce': ce_weight,
            'spike_reg': spike_reg_weight,
            'mem_smoothness': mem_smoothness_weight,
        }
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem_sequence: torch.Tensor, model: nn.Module) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        if mem_sequence.numel() > 1:
            mem_diff = torch.diff(mem_sequence)
            mem_smoothness_loss = torch.mean(mem_diff**2)
        else:
            mem_smoothness_loss = torch.tensor(0.0, device=logits.device)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['mem_smoothness'] * mem_smoothness_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss,
            'mem_smoothness_loss': mem_smoothness_loss,
            'spike_rate': spike_rate
        }

class PlannerLoss(nn.Module):
    """
    プランナーSNNの学習用損失関数。
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, predicted_logits: torch.Tensor, target_plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target_plan[:, 0]
        
        loss = self.loss_fn(predicted_logits, target)
        
        return {'total': loss, 'planner_loss': loss}

class ProbabilisticEnsembleLoss(nn.Module):
    """
    確率的アンサンブル学習のための損失関数。
    出力のばらつきを抑制する正則化項を持つ。
    """
    def __init__(self, ce_weight: float, variance_reg_weight: float, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {'ce': ce_weight, 'variance_reg': variance_reg_weight}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, mem: torch.Tensor, model: nn.Module, **kwargs) -> dict:
        # logitsは (ensemble_size, batch_size, seq_len, vocab_size) の形状を想定
        
        # 1. クロスエントロピー損失
        # アンサンブル全体で平均したロジットでCE損失を計算
        mean_logits = logits.mean(dim=0)
        ce_loss = self.ce_loss_fn(mean_logits.view(-1, mean_logits.size(-1)), targets.view(-1))
        
        # 2. 分散正則化損失
        # アンサンブル間の出力のばらつき（分散）をペナルティとする
        # softmax後の確率分布の分散を計算
        probs = F.softmax(logits, dim=-1)
        variance = probs.var(dim=0).mean()
        variance_reg_loss = variance
        
        total_loss = (self.weights['ce'] * ce_loss + 
                      self.weights['variance_reg'] * variance_reg_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'variance_reg_loss': variance_reg_loss
        }
