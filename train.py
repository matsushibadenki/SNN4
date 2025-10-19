# matsushibadenki/snn3/train.py
# (更新)
# 新しい統合学習実行スクリプト (完全版)
#
# 修正(mypy): [annotation-unchecked] noteを解消するため、型ヒントを追加。
# 修正(mypy): [name-defined]エラーを解消するため、Unionをインポート。
# 修正(mypy): [union-attr]エラーを解消するため、trainerの種類に応じて処理を分岐。

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from dependency_injector.wiring import inject, Provide
from typing import Optional, Tuple, List, Dict, Any, Callable, cast, Union
from transformers import PreTrainedTokenizerBase

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset
from snn_research.training.trainers import BreakthroughTrainer, ParticleFilterTrainer
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.training.quantization import apply_qat, convert_to_quantized_model
from snn_research.training.pruning import apply_magnitude_pruning
from scripts.data_preparation import prepare_wikitext_data
from snn_research.core.snn_core import SNNCore
from app.utils import get_auto_device


# DIコンテナのセットアップ
container = TrainingContainer()

@inject
def train(
    args: argparse.Namespace,
    config: Dict[str, Any] = Provide[TrainingContainer.config],
    tokenizer: PreTrainedTokenizerBase = Provide[TrainingContainer.tokenizer],
) -> None:
    """学習プロセスを実行するメイン関数"""
    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available() else get_auto_device()
    
    paradigm = config['training']['paradigm']

    print(f"🚀 学習パラダイム '{paradigm}' で学習を開始します...")
    
    trainer: Union[BreakthroughTrainer, BioRLTrainer, ParticleFilterTrainer]

    if paradigm.startswith("bio-"):
        # --- 生物学的学習パラダイムの実行 ---
        if paradigm == "bio-causal-sparse":
            print("🧬 適応的因果スパース化を有効にした強化学習を開始します。")
            container.config.training.biologically_plausible.adaptive_causal_sparsification.enabled.from_value(True)
            trainer = container.bio_rl_trainer()
            cast(BioRLTrainer, trainer).train(num_episodes=config['training']['epochs'])
        elif paradigm == "bio-particle-filter":
            print("🌪️ パーティクルフィルタによる確率的学習を開始します (CPUベース)。")
            container.config.training.biologically_plausible.particle_filter.enabled.from_value(True)
            trainer = container.particle_filter_trainer()
            dummy_data = torch.rand(1, 10, device=device)
            dummy_targets = torch.rand(1, 2, device=device)
            for epoch in range(config['training']['epochs']):
                loss = cast(ParticleFilterTrainer, trainer).train_step(dummy_data, dummy_targets)
                print(f"Epoch {epoch+1}/{config['training']['epochs']}: Particle Filter Loss = {loss:.4f}")
        elif paradigm == "bio-probabilistic-hebbian":
            print("🧬 確率的ヘブ学習を開始します...")
            # DIコンテナから確率的トレーナーを取得
            prob_trainer: BioRLTrainer = container.probabilistic_trainer()
            # 論文のアルゴリズムは教師なし学習の可能性が高いため、
            # RLループではなく、データローダーからデータを取得して
            # モデルの update_weights を直接呼び出す形になるかもしれない
            # ここでは BioRLTrainer を流用する前提で train を呼び出す
            prob_trainer.train(num_episodes=config['training']['epochs'])
        else:
            raise ValueError(f"不明な生物学的学習パラダイム: {paradigm}")

    elif paradigm in ["gradient_based", "self_supervised", "physics_informed", "probabilistic_ensemble"]:
        # --- 勾配ベース学習パラダイムの実行 ---
        if is_distributed and paradigm != "gradient_based":
            raise NotImplementedError(f"{paradigm} learning does not support DDP yet.")

        is_distillation = paradigm == "gradient_based" and config['training']['gradient_based']['type'] == "distillation"
        
        # データセットの準備
        wikitext_path = "data/wikitext-103_train.jsonl"
        if os.path.exists(wikitext_path):
            data_path = wikitext_path
        else:
            data_path = args.data_path or config['data']['path']
        
        DatasetClass = get_dataset_class(DataFormat(config['data']['format']))
        dataset: SNNBaseDataset
        if is_distillation:
            dataset = DistillationDataset(file_path=os.path.join(data_path, "distillation_data.jsonl"), data_dir=data_path, tokenizer=tokenizer, max_seq_len=config['model']['time_steps'])
        else:
            dataset = DatasetClass(file_path=data_path, tokenizer=tokenizer, max_seq_len=config['model']['time_steps'])
            
        train_size = int((1.0 - config['data']['split_ratio']) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_sampler: Optional[DistributedSampler] = DistributedSampler(train_dataset) if is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn(tokenizer, is_distillation))
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn(tokenizer, is_distillation))

        snn_model: nn.Module = container.snn_model(backend=args.backend)

        if config.get('training', {}).get('quantization', {}).get('enabled', False):
            snn_model = apply_qat(snn_model.to('cpu'))
        snn_model.to(device)

        if is_distributed:
            snn_model = DDP(snn_model, device_ids=[rank], find_unused_parameters=True)
        
        astrocyte = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None
        
        trainer_provider: Callable[..., BreakthroughTrainer]
        optimizer: torch.optim.Optimizer
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]

        if paradigm == "gradient_based":
            optimizer = container.optimizer(params=snn_model.parameters())
            scheduler = container.scheduler(optimizer=optimizer) if config['training']['gradient_based']['use_scheduler'] else None
            trainer_provider = container.distillation_trainer if is_distillation else container.standard_trainer
        elif paradigm == "self_supervised":
            optimizer = container.optimizer(params=snn_model.parameters())
            scheduler = container.scheduler(optimizer=optimizer) if config['training']['self_supervised']['use_scheduler'] else None
            trainer_provider = container.self_supervised_trainer
        elif paradigm == "physics_informed":
            optimizer = container.pi_optimizer(params=snn_model.parameters())
            scheduler = container.pi_scheduler(optimizer=optimizer) if config['training']['physics_informed']['use_scheduler'] else None
            trainer_provider = container.physics_informed_trainer
        else: # probabilistic_ensemble
            optimizer = container.optimizer(params=snn_model.parameters())
            scheduler = container.scheduler(optimizer=optimizer) if config['training']['probabilistic_ensemble']['use_scheduler'] else None
            trainer_provider = container.probabilistic_ensemble_trainer

        trainer = trainer_provider(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte)

        if args.load_ewc_data:
            trainer.load_ewc_data(args.load_ewc_data)

        start_epoch = trainer.load_checkpoint(args.resume_path) if args.resume_path else 0
        for epoch in range(start_epoch, config['training']['epochs']):
            if train_sampler: train_sampler.set_epoch(epoch)
            trainer.train_epoch(train_loader, epoch)
            if rank in [-1, 0] and (epoch % config['training']['eval_interval'] == 0 or epoch == config['training']['epochs'] - 1):
                val_metrics = trainer.evaluate(val_loader, epoch)
                if epoch % config['training']['log_interval'] == 0:
                    checkpoint_path = os.path.join(config['training']['log_dir'], f"checkpoint_epoch_{epoch}.pth")
                    trainer.save_checkpoint(path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')), tokenizer_name=config['data']['tokenizer_name'], config=config['model'])
        
        if rank in [-1, 0] and args.task_name and config['training']['gradient_based']['loss']['ewc_weight'] > 0:
            trainer._compute_ewc_fisher_matrix(train_loader, args.task_name)
            
        # 最終モデルの処理 (量子化、プルーニング)
        if rank in [-1, 0]:
            final_model = trainer.model.module if is_distributed else trainer.model
            if isinstance(final_model, nn.Module): # 型ガード
                if config.get('training', {}).get('quantization', {}).get('enabled', False):
                    quantized_model = convert_to_quantized_model(final_model.to('cpu'))
                    quantized_path = os.path.join(config['training']['log_dir'], 'quantized_best_model.pth')
                    torch.save(quantized_model.state_dict(), quantized_path)
                if config.get('training', {}).get('pruning', {}).get('enabled', False):
                    pruning_amount = config['training']['pruning'].get('amount', 0.2)
                    pruned_model = apply_magnitude_pruning(final_model, amount=pruning_amount)
                    pruned_path = os.path.join(config['training']['log_dir'], 'pruned_best_model.pth')
                    torch.save(pruned_model.state_dict(), pruned_path)

    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'.")

    print("✅ 学習が完了しました。")


def collate_fn(tokenizer: PreTrainedTokenizerBase, is_distillation: bool) -> Callable[[List[Tuple[torch.Tensor, ...]]], Tuple[torch.Tensor, ...]]:
    def collate(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        padding_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=float(padding_val))
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=float(padding_val))

        if is_distillation:
            logits = [item[2] for item in batch]
            padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
            return padded_inputs.long(), padded_targets.long(), padded_logits
        return padded_inputs.long(), padded_targets.long()
    return collate
    
def main() -> None:
    parser = argparse.ArgumentParser(description="SNN 統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, help="モデルアーキテクチャ設定ファイル")
    parser.add_argument("--data_path", type=str, help="データセットのパス（configを上書き）")
    parser.add_argument("--task_name", type=str, help="EWCのためにタスク名を指定 (例: 'sst2')")
    parser.add_argument("--override_config", type=str, action='append', help="設定を上書き (例: 'training.epochs=5')")
    parser.add_argument("--distributed", action="store_true", help="分散学習を有効にする")
    parser.add_argument("--resume_path", type=str, help="チェックポイントから学習を再開する")
    parser.add_argument("--load_ewc_data", type=str, help="事前計算されたEWCのFisher行列とパラメータのパス")
    parser.add_argument("--use_astrocyte", action="store_true", help="アストロサイトネットワークを有効にする (gradient_based系のみ)")
    parser.add_argument("--paradigm", type=str, help="学習パラダイムを上書き (例: gradient_based, bio-causal-sparse, bio-particle-filter)")
    parser.add_argument("--backend", type=str, default="spikingjelly", choices=["spikingjelly", "snntorch"], help="SNNシミュレーションバックエンドライブラリ")
    args = parser.parse_args()

    container.config.from_yaml(args.config)
    if args.model_config: container.config.from_yaml(args.model_config)
    if args.data_path: container.config.data.path.from_value(args.data_path)
    if args.paradigm: container.config.training.paradigm.from_value(args.paradigm)
    
    if args.override_config:
        for override in args.override_config:
            keys, value_str = override.split('=', 1)
            try:
                value: Any = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str
            
            config_dict: Dict[str, Any] = {}
            temp_dict = config_dict
            key_parts = keys.split('.')
            for i, part in enumerate(key_parts):
                if i == len(key_parts) - 1:
                    temp_dict[part] = value
                else:
                    temp_dict[part] = {}
                    temp_dict = temp_dict[part]
            container.config.from_dict(config_dict)

    if args.distributed: dist.init_process_group(backend="nccl")
    
    container.wire(modules=[__name__])
    
    injected_config = container.config()
    injected_tokenizer = container.tokenizer()
    train(args, config=injected_config, tokenizer=injected_tokenizer)
    
    if args.distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
