# ファイルパス: snn_research/training/trainers.py
# (修正 v9)
# 修正: `NameError: name 'BioSNN' is not defined` を解消するため、BioSNNをインポート。
# 修正: ParticleFilterTrainerのコンストラクタを修正し、KeyErrorを解消する。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import os
import collections
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, cast
import shutil
import time
from torch.optim import Adam
from spikingjelly.activation_based import functional # type: ignore

from snn_research.training.losses import CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss, PlannerLoss, ProbabilisticEnsembleLoss
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from torch.utils.tensorboard import SummaryWriter

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from snn_research.bio_models.simple_network import BioSNN
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
import copy


class BreakthroughTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool, log_dir: str,
                 astrocyte_network: Optional[AstrocyteNetwork] = None,
                 meta_cognitive_snn: Optional[MetaCognitiveSNN] = None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        self.use_amp = use_amp and self.device != 'mps'
        self.astrocyte_network = astrocyte_network
        self.meta_cognitive_snn = meta_cognitive_snn
        
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging enabled. Log directory: {log_dir}")


    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        functional.reset_net(self.model)
        start_time = time.time()
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        with torch.amp.autocast(device_type=self.device if self.device != 'mps' else 'cpu', enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                logits, spikes, mem = self.model(input_ids, return_spikes=True, return_full_mems=True)
                loss_dict = self.criterion(logits, target_ids, spikes, mem, self.model)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            if self.meta_cognitive_snn:
                end_time = time.time()
                computation_time = end_time - start_time
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                        ignore_idx = self.criterion.ce_loss_fn.ignore_index
                        mask = target_ids != ignore_idx
                        num_masked_elements = cast(torch.Tensor, mask).sum()
                        accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                        loss_dict['accuracy'] = accuracy.item()
                
                self.meta_cognitive_snn.update_metadata(
                    loss=loss_dict['total'].item(),
                    computation_time=computation_time,
                    accuracy=loss_dict.get('accuracy', 0.0)
                )

        with torch.no_grad():
            if 'accuracy' not in loss_dict:
                preds = torch.argmax(logits, dim=-1)
                if hasattr(self.criterion, 'ce_loss_fn') and hasattr(self.criterion.ce_loss_fn, 'ignore_index'):
                    ignore_idx = self.criterion.ce_loss_fn.ignore_index
                    mask = target_ids != ignore_idx
                    num_masked_elements = cast(torch.Tensor, mask).sum()
                    accuracy = (preds[mask] == target_ids[mask]).float().sum() / num_masked_elements if num_masked_elements > 0 else torch.tensor(0.0)
                    loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}


    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        self.model.train()
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value
            progress_bar.set_postfix({k: v / (progress_bar.n + 1) for k, v in total_metrics.items()})

        if self.scheduler: self.scheduler.step()
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            if self.scheduler:
                self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0], epoch)
            else:
                self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_metrics

    # ... (evaluate, save_checkpoint, load_checkpointメソッドは変更なし) ...

class DistillationTrainer(BreakthroughTrainer):
    # ... (変更なし) ...
    pass

class SelfSupervisedTrainer(BreakthroughTrainer):
    pass

class PhysicsInformedTrainer(BreakthroughTrainer):
    # ... (変更なし) ...
    pass

class ProbabilisticEnsembleTrainer(BreakthroughTrainer):
    # ... (変更なし) ...
    pass

class PlannerTrainer:
    # ... (変更なし) ...
    pass

class BPTTTrainer:
    # ... (変更なし) ...
    pass

class ParticleFilterTrainer:
    """
    逐次モンテカルロ法（パーティクルフィルタ）を用いて、微分不可能なSNNを学習するトレーナー。
    """
    def __init__(self, base_model: BioSNN, config: Dict[str, Any], device: str):
        self.base_model = base_model.to(device)
        self.device = device
        self.config = config
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # configの階層構造を変更
        self.num_particles = config['num_particles']
        self.noise_std = config['noise_std']
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        self.particles = [copy.deepcopy(self.base_model) for _ in range(self.num_particles)]
        self.particle_weights = torch.ones(self.num_particles, device=self.device) / self.num_particles
        print(f"🌪️ ParticleFilterTrainer initialized with {self.num_particles} particles.")

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        """1ステップの学習（予測、尤度計算、再サンプリング）を実行する。"""
        data = data.to(self.device)
        targets = targets.to(self.device)

        for particle in self.particles:
            with torch.no_grad():
                for param in particle.parameters():
                    param.add_(torch.randn_like(param) * self.noise_std)
        
        log_likelihoods = []
        for particle in self.particles:
            particle.eval()
            with torch.no_grad():
                squeezed_data = data.squeeze(0) if data.dim() > 1 else data
                input_spikes = (torch.rand_like(squeezed_data) > 0.5).float().to(self.device)
                outputs, _ = particle(input_spikes)
                
                squeezed_targets = targets.squeeze(0) if targets.dim() > 1 else targets
                loss = F.mse_loss(outputs, squeezed_targets)
                log_likelihoods.append(-loss)
        
        log_likelihoods_tensor = torch.tensor(log_likelihoods, device=self.device)
        self.particle_weights *= torch.exp(log_likelihoods_tensor - log_likelihoods_tensor.max())
        
        if self.particle_weights.sum() > 0:
            self.particle_weights /= self.particle_weights.sum()
        else:
            self.particle_weights.fill_(1.0 / self.num_particles)

        if 1. / (self.particle_weights**2).sum() < self.num_particles / 2:
            indices = torch.multinomial(self.particle_weights, self.num_particles, replacement=True)
            new_particles = [copy.deepcopy(self.particles[i]) for i in indices]
            self.particles = new_particles
            self.particle_weights.fill_(1.0 / self.num_particles)
        
        best_particle_loss = -log_likelihoods_tensor.max().item()
        return best_particle_loss
