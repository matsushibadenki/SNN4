# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# タイトル: 知識蒸留マネージャー
# 機能説明: 知識蒸留プロセスを統括するマネージャークラス。
# BugFix: データセット側で入力とターゲットのペアを正しく作成するように修正し、
#         collate_fnを簡素化することで、学習データの不整合問題を解消。
# BugFix: ファイル内にあった不正な閉じ括弧を削除し、mypyの構文エラーを修正。
# 修正: mypyエラー `Name "Tuple" is not defined` を解消するため、Tupleをインポート。
# 修正(mypy): [annotation-unchecked] noteを解消するため、内部クラス・関数の
#             型ヒントを修正・追加。
# 改善点(v2): データセットの準備ロジックを汎用化し、画像データセットにも対応。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast, Tuple, Callable
import asyncio
import os
import json
from tqdm import tqdm
from omegaconf import OmegaConf

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.benchmark.metrics import calculate_perplexity, calculate_energy_consumption
from snn_research.core.snn_core import SNNCore

# --- 循環インポート解消のための修正 ---
# 型チェック時のみインポートを実行し、実行時の循環参照を回避する
if TYPE_CHECKING:
    from snn_research.training.trainers import DistillationTrainer

class KnowledgeDistillationManager:
    """
    知識蒸留プロセスを統括するマネージャークラス。
    """
    def __init__(
        self,
        student_model: torch.nn.Module,
        trainer: "DistillationTrainer",
        tokenizer_name: str,
        model_registry: ModelRegistry,
        device: str,
        teacher_model: Optional[torch.nn.Module] = None,
        teacher_model_name: Optional[str] = None
    ):
        self.student_model = student_model.to(device)
        self.distillation_trainer = trainer
        
        if teacher_model is not None:
            self.teacher_model = teacher_model.to(device)
        elif teacher_model_name is not None:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
        else:
            raise ValueError("Either teacher_model or teacher_model_name must be provided.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_registry = model_registry
        self.device = device

    def prepare_dataset(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        collate_fn: Callable,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        既存のデータセットをラップし、教師モデルのロジットを動的に付与するデータローダーを準備する。
        """
        class _DistillationWrapperDataset(Dataset):
            def __init__(self, original_dataset: Dataset, teacher_model: nn.Module, device: str):
                self.original_dataset = original_dataset
                self.teacher_model = teacher_model
                self.device = device

            def __len__(self) -> int:
                return len(self.original_dataset)

            @torch.no_grad()
            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = self.original_dataset[idx]
                
                # collate_fnでバッチ化されることを想定し、辞書形式で返す
                # 'input_images' or 'input_ids'
                inputs = item[0].unsqueeze(0).to(self.device)
                teacher_output = self.teacher_model(inputs)
                teacher_logits = (teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output).squeeze(0).cpu()

                return {"inputs": item[0], "labels": item[1], "teacher_logits": teacher_logits}

        train_wrapper = _DistillationWrapperDataset(train_dataset, self.teacher_model, self.device)
        val_wrapper = _DistillationWrapperDataset(val_dataset, self.teacher_model, self.device)
        
        def distillation_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            inputs = torch.stack([item['inputs'] for item in batch])
            labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
            teacher_logits = torch.stack([item['teacher_logits'] for item in batch])
            return inputs, labels, teacher_logits

        train_loader = DataLoader(train_wrapper, batch_size=batch_size, collate_fn=distillation_collate_fn, shuffle=True)
        val_loader = DataLoader(val_wrapper, batch_size=batch_size, collate_fn=distillation_collate_fn)

        return train_loader, val_loader


    async def run_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_id: str,
        task_description: str,
        student_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        safe_model_id = model_id.lower().replace(" ", "_")
        print(f"--- Starting Knowledge Distillation for model: {safe_model_id} ---")

        final_metrics: Dict[str, float] = {}

        # 1. 知識蒸留の実行
        print("Step 1: Running distillation training...")
        for epoch in range(epochs):
            self.distillation_trainer.train_epoch(train_loader, epoch)
            final_metrics = self.distillation_trainer.evaluate(val_loader, epoch)
        print("Distillation training finished.")

        # 2. モデルの評価 (最終)
        print("Step 2: Evaluating the distilled model...")
        evaluation_results = await self.evaluate_model(val_loader)
        final_metrics.update(evaluation_results)
        print(f"Evaluation finished. Metrics: {final_metrics}")

        # 3. モデルの保存
        save_dir = os.path.join("runs", "specialists", safe_model_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")
        
        # 【根本修正】推論時の不整合を防ぐため、保存すべきでない一時的なバッファを全て除外する
        model_to_save = self.distillation_trainer.model.module if isinstance(self.distillation_trainer.model, nn.parallel.DistributedDataParallel) else self.distillation_trainer.model
        buffers_to_exclude = {
            name for name, _ in model_to_save.named_buffers() 
            if any(keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold'])
        }
        model_state_to_save = {k: v for k, v in model_to_save.state_dict().items() if k not in buffers_to_exclude}
        torch.save(model_state_to_save, save_path)
        print("Model saved.")

        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
        await self.model_registry.register_model(
            model_id=safe_model_id,
            task_description=task_description,
            metrics=final_metrics,
            model_path=save_path,
            config=student_config
        )
        print(f"Model '{safe_model_id}' successfully registered.")

        print("--- Knowledge Distillation Finished ---")
        return {"model_id": safe_model_id, "metrics": final_metrics, "path": save_path, "config": student_config}

    async def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: str, force_retrain: bool, student_config: Optional[Dict[str, Any]] = None):
        """Webクローラー等からのデータでオンデマンド学習を実行するパイプライン。"""
        print(f"🚀 Starting on-demand pipeline for task: {task_description}")

        if student_config is None:
            print("student_config not provided, attempting to retrieve from student model...")
            if hasattr(self.student_model, 'config'):
                student_config_resolved = OmegaConf.to_container(self.student_model.config, resolve=True)
                student_config = cast(Dict[str, Any], student_config_resolved)
                print("✅ Successfully retrieved config from SNNCore model.")
            else:
                raise ValueError("student_config was not provided and could not be retrieved from the model.")
        
        if student_config is None:
            raise ValueError("student_config is None, cannot proceed.")

        texts = []
        with open(unlabeled_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    texts.append(json.loads(line)['text'])
                except (json.JSONDecodeError, KeyError):
                    if line.strip():
                        texts.append(line.strip())
        
        if not texts:
            print("❌ No text found in the provided data file. Aborting.")
            return None

        # This part is now text-specific and needs adaptation for generic datasets
        raise NotImplementedError("run_on_demand_pipeline for text needs to be refactored to use the new prepare_dataset method.")

    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。
        """
        model_to_eval = self.distillation_trainer.model
        model_to_eval.eval()
        total_spikes = 0.0
        total_samples = 0
        num_neurons = 0
        if isinstance(model_to_eval, SNNCore):
            num_neurons = sum(p.numel() for p in model_to_eval.model.parameters())
        else:
            num_neurons = sum(p.numel() for p in model_to_eval.parameters())


        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model")
        for batch in progress_bar:
            inputs, labels, _ = batch
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = model_to_eval(inputs, return_spikes=True)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    _, avg_batch_spikes, _ = outputs
                else:
                    avg_batch_spikes = torch.zeros((), device=inputs.device)

            total_spikes += avg_batch_spikes.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0.0

        # Perplexity is not suitable for classification, so we skip it.
        # Accuracy is calculated within the trainer's evaluate method.
        energy = calculate_energy_consumption(avg_spikes_per_sample, num_neurons=num_neurons)

        return {
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }
