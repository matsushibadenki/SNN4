# ファイルパス: snn_research/distillation/knowledge_distillation_manager.py
# コードの最も最初には、ファイルパス、ファイルの内容を示したタイトル、機能の説明を詳細に記述してください。 修正内容は記載する必要はありません。
# タイトル: 知識蒸留マネージャー
# 機能説明: 知識蒸留プロセスを統括するマネージャークラス。
# BugFix: データセット側で入力とターゲットのペアを正しく作成するように修正し、
#         collate_fnを簡素化することで、学習データの不整合問題を解消。
# BugFix: ファイル内にあった不正な閉じ括弧を削除し、mypyの構文エラーを修正。
# 修正: mypyエラー `Name "Tuple" is not defined` を解消するため、Tupleをインポート。
# 修正(mypy): [annotation-unchecked] noteを解消するため、内部クラス・関数の
#             型ヒントを修正・追加。
# 改善点(v2): データセットの準備ロジックを汎用化し、画像データセットにも対応。
# 修正(v3): mypyエラー [arg-type] を解消するため、castを使用して型を明示。
# 改善点(v4): 画像データセットに対応するよう prepare_dataset と distillation_collate_fn を修正。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast, Tuple, Callable, Sized
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
        collate_fn: Callable, # 元のタスクのcollate_fnを受け取る
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        既存のデータセットをラップし、教師モデルのロジットを動的に付与するデータローダーを準備する。
        画像データセットにも対応。
        """
        class _DistillationWrapperDataset(Dataset):
            def __init__(self, original_dataset: Dataset, teacher_model: nn.Module, device: str):
                self.original_dataset = original_dataset
                self.teacher_model = teacher_model
                self.device = device

            def __len__(self) -> int:
                return len(cast(Sized, self.original_dataset))

            @torch.no_grad()
            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = self.original_dataset[idx] # original_datasetが返す形式 (画像なら(img, label)、テキストなら(input_ids, target_ids)など)

                # --- ▼ 画像/テキスト判定と入力形式統一 ▼ ---
                if isinstance(item[0], torch.Tensor) and item[0].ndim >= 2: # 画像データと仮定 (例: [C, H, W])
                    inputs = item[0].unsqueeze(0).to(self.device) # バッチ次元を追加してデバイスへ
                    label = item[1]
                    input_key = "input_images"
                elif isinstance(item[0], torch.Tensor) and item[0].ndim == 1: # テキストデータ (input_ids) と仮定
                    inputs = item[0].unsqueeze(0).to(self.device)
                    label = item[1] # target_ids
                    input_key = "input_ids"
                else:
                    raise TypeError(f"Unsupported data type from original_dataset: {type(item[0])}")
                # --- ▲ 画像/テキスト判定と入力形式統一 ▲ ---

                # 教師モデルでロジットを計算
                teacher_output = self.teacher_model(inputs)
                # 画像モデルはタプルを返さない場合がある
                teacher_logits_batch = (teacher_output.logits if hasattr(teacher_output, 'logits') else teacher_output)
                # バッチ次元を削除してCPUへ
                teacher_logits = teacher_logits_batch.squeeze(0).cpu()

                # collate_fnで扱える辞書形式で返す
                return {"inputs": item[0], "labels": label, "teacher_logits": teacher_logits, "input_key": input_key}

        train_wrapper = _DistillationWrapperDataset(train_dataset, self.teacher_model, self.device)
        val_wrapper = _DistillationWrapperDataset(val_dataset, self.teacher_model, self.device)

        def distillation_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            DistillationWrapperDatasetからの辞書をバッチ化し、
            DistillationTrainerが期待するタプル形式 (student_input, attention_mask, student_target, teacher_logits) に変換する。
            """
            input_key = batch[0]["input_key"] # バッチ内のinput_keyは同じと仮定

            if input_key == "input_images":
                student_input = torch.stack([item['inputs'] for item in batch])
                # 画像の場合、attention_maskは通常不要だが、Trainerのインターフェースに合わせてダミーを作成
                attention_mask = torch.ones(student_input.shape[0], student_input.shape[-1], dtype=torch.long) # 画像サイズに合わせたダミーマスク (より良い方法があれば修正)
                student_target = torch.tensor([item['labels'] for item in batch], dtype=torch.long) # 分類ラベル
                teacher_logits = torch.stack([item['teacher_logits'] for item in batch]) # 分類ロジット
            elif input_key == "input_ids":
                # 元のテキストcollate_fnを使用してinput_idsとattention_maskを作成
                # ただし、元のcollate_fnはテキストリストを期待するため、再構築が必要
                original_batch_for_collate = [{"text": self.tokenizer.decode(item['inputs']), "label": item['labels']} for item in batch] # ラベルはダミー
                collated_original = collate_fn(original_batch_for_collate) # 元のcollate_fnを呼び出し

                student_input = collated_original['input_ids']
                attention_mask = collated_original['attention_mask']
                student_target = torch.nn.utils.rnn.pad_sequence(
                    [item['labels'] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id or 0
                )
                teacher_logits = torch.nn.utils.rnn.pad_sequence(
                    [item['teacher_logits'] for item in batch], batch_first=True, padding_value=0.0
                )
                # Ensure teacher_logits has the same seq_len as student_input
                target_len = student_input.shape[1]
                current_len = teacher_logits.shape[1]
                if current_len < target_len:
                    padding = torch.zeros(teacher_logits.shape[0], target_len - current_len, teacher_logits.shape[2], device=teacher_logits.device)
                    teacher_logits = torch.cat([teacher_logits, padding], dim=1)
                elif current_len > target_len:
                     teacher_logits = teacher_logits[:, :target_len, :]

            else:
                 raise ValueError(f"Unknown input_key: {input_key}")


            return student_input, attention_mask, student_target, teacher_logits

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
        print(f"Step 1: Running distillation training for {epochs} epochs...")
        for epoch in range(epochs):
            self.distillation_trainer.train_epoch(train_loader, epoch)
            # 評価は最終エポック後のみ実施 (高速化のため)
            if epoch == epochs - 1:
                final_metrics = self.distillation_trainer.evaluate(val_loader, epoch)
        print("Distillation training finished.")

        # 2. モデルの評価 (最終) - trainer.evaluate内で実行済みのメトリクスを使用
        print("Step 2: Evaluating the distilled model...")
        # evaluation_results = await self.evaluate_model(val_loader) # 評価はtrain_epoch内で実施済
        # final_metrics.update(evaluation_results)
        print(f"Evaluation finished. Final Metrics: {final_metrics}")


        # 3. モデルの保存
        save_dir = os.path.join("runs", "specialists", safe_model_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        print(f"Step 3: Saving the model to {save_path}...")

        model_to_save = self.distillation_trainer.model.module if isinstance(self.distillation_trainer.model, nn.parallel.DistributedDataParallel) else self.distillation_trainer.model
        # SNNCoreラッパーの場合、中のモデルを取り出す
        if isinstance(model_to_save, SNNCore):
            model_to_save = model_to_save.model

        buffers_to_exclude = {
            name for name, _ in model_to_save.named_buffers()
            if any(keyword in name for keyword in ['mem', 'spikes', 'adaptive_threshold', 'pre_trace', 'post_trace', 'eligibility_trace', 'causal_contribution', 'v', 'u']) # SNN状態変数を除外
        }
        model_state_to_save = {k: v for k, v in model_to_save.state_dict().items() if k not in buffers_to_exclude}

        # 保存する辞書に student_config を含める
        save_dict = {
            'model_state_dict': model_state_to_save,
            'config': student_config # モデル設定を一緒に保存
        }
        torch.save(save_dict, save_path)
        print("Model saved.")


        # 4. モデルレジストリへの登録
        print("Step 4: Registering the model...")
        await self.model_registry.register_model(
            model_id=safe_model_id,
            task_description=task_description,
            metrics=final_metrics,
            model_path=save_path, # 保存パスを渡す
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
            if hasattr(self.student_model, 'config') and isinstance(self.student_model, SNNCore): # SNNCoreか確認
                student_config_resolved = OmegaConf.to_container(self.student_model.config, resolve=True)
                student_config = cast(Dict[str, Any], student_config_resolved)
                print("✅ Successfully retrieved config from SNNCore model.")
            else:
                raise ValueError("student_config was not provided and could not be retrieved from the model.")

        if student_config is None:
            raise ValueError("student_config is None, cannot proceed.")

        # --- ▼ 修正: 画像タスクかテキストタスクかを判定 ▼ ---
        # ここでは簡易的に、タスク記述に'image'や'cifar'が含まれるかで判定
        is_image_task = any(kw in task_description.lower() for kw in ['image', 'cifar', 'vision'])

        if is_image_task:
             # 画像タスク用のデータ準備 (例: CIFAR10Taskを使用)
             TaskClass = TASK_REGISTRY.get("cifar10") # 仮にCIFAR10とする
             if not TaskClass: raise ValueError("CIFAR10 task not found in registry.")
             task = TaskClass(tokenizer=self.tokenizer, device=self.device, hardware_profile={})
             train_dataset, val_dataset = task.prepare_data() # 画像データセットをロード
             collate_fn = task.get_collate_fn()
        else:
             # テキストタスク用のデータ準備
             from snn_research.data.datasets import SimpleTextDataset # テキスト用Datasetをインポート
             if not os.path.exists(unlabeled_data_path):
                 raise FileNotFoundError(f"Unlabeled data file not found: {unlabeled_data_path}")

             dataset = SimpleTextDataset(file_path=unlabeled_data_path, tokenizer=self.tokenizer, max_seq_len=student_config.get('time_steps', 128))
             # データセットを分割 (例: 90% train, 10% val)
             train_size = int(0.9 * len(dataset))
             val_size = len(dataset) - train_size
             train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

             from train import collate_fn as text_collate_fn # train.pyからcollate_fnをインポート
             collate_fn = text_collate_fn(self.tokenizer, is_distillation=True) # 蒸留用のcollate_fnを取得
        # --- ▲ 修正 ▲ ---

        # 知識蒸留用にデータセットをラップ
        train_loader, val_loader = self.prepare_dataset(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn, # 適切なcollate_fnを渡す
            batch_size=self.distillation_trainer.config.training.batch_size() if hasattr(self.distillation_trainer, 'config') else 8 # configがあればそこから、なければデフォルト
        )

        # 蒸留の実行
        result = await self.run_distillation(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.distillation_trainer.config.training.epochs() if hasattr(self.distillation_trainer, 'config') else 5, # configがあればそこから、なければデフォルト
            model_id=task_description,
            task_description=f"Expert for {task_description}",
            student_config=student_config
        )
        return result


    async def evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        蒸留済みモデルの性能を評価する。
        """
        model_to_eval = self.distillation_trainer.model
        model_to_eval.eval()
        total_spikes = 0.0
        total_samples = 0
        num_neurons = 0

        # SNNCoreラッパーの場合、中の実際のモデルのパラメータ数を計算
        if isinstance(model_to_eval, SNNCore):
            num_neurons = sum(p.numel() for p in model_to_eval.model.parameters() if p.requires_grad)
        else:
            num_neurons = sum(p.numel() for p in model_to_eval.parameters() if p.requires_grad)

        progress_bar = tqdm(dataloader, desc="Evaluating Distilled Model")
        all_logits = []
        all_labels = []

        for batch in progress_bar:
            # distillation_collate_fn からのタプルをアンパック
            inputs, attention_mask, labels, teacher_logits = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                # --- ▼ 修正: 入力キーを判定 ▼ ---
                input_key = "input_images" if inputs.ndim == 4 else "input_ids"
                model_input = {input_key: inputs}
                if input_key == "input_ids":
                    model_input["attention_mask"] = attention_mask.to(self.device)
                # --- ▲ 修正 ▲ ---

                outputs = model_to_eval(**model_input, return_spikes=True) # **で辞書を展開して渡す
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    logits, avg_batch_spikes, _ = outputs
                else:
                    logits = outputs # 画像分類などはタプルでない場合がある
                    avg_batch_spikes = torch.zeros((), device=inputs.device)

            total_spikes += avg_batch_spikes.item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0.0
        energy = calculate_energy_consumption(avg_spikes_per_sample, num_neurons=num_neurons)

        # AccuracyなどのメトリクスはTrainerのevaluateメソッドで計算されるため、ここではスパイクとエネルギーのみ返す
        # Note: もしTrainerのevaluateが呼ばれない場合は、ここでAccuracy計算を追加する必要がある
        metrics = {
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "estimated_energy_consumption": energy
        }
        # 他のメトリクス（例：accuracy）は trainer.evaluate の結果から取得される

        return metrics
