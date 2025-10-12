# matsushibadenki/snn_research/benchmark/tasks.py
# ベンチマークタスクの定義ファイル
#
# (省略)
# - ValueError修正: SNNClassifier.forwardの戻り値を3つのタプル(logits, spikes, mem)に修正し、
#                   Trainerが期待するインターフェースと一致させた。
# 改善点:
# - ROADMAPフェーズ4「ハードウェア展開」に基づき、ハードウェアプロファイルを導入。
# - evaluateメソッドを更新し、スパイク数から推定エネルギー消費量を計算するようにした。
#
# 修正点(v2):
# - RuntimeErrorを解消するため、SNNClassifierが使用するBreakthroughSNNの
#   d_stateパラメータを調整し、次元の不整合を修正。
#
# 修正点(v3):
# - 循環参照エラーを解消するため、TASK_REGISTRYの定義を __init__.py に移動。
#
# 修正点(v4):
# - mypyエラー[arg-type]を解消するため、SNNCoreに渡す設定をDictConfigに変換。

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable, Sized, cast
from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from omegaconf import OmegaConf # ◾️ 追加

from snn_research.core.snn_core import BreakthroughSNN, SNNCore
from snn_research.benchmark.ann_baseline import ANNBaselineModel
from snn_research.benchmark.metrics import calculate_accuracy, calculate_energy_consumption
from snn_research.hardware.profiles import get_hardware_profile

# --- 共通データセットクラス ---
class GenericDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

# --- ベンチマークタスクの基底クラス ---
class BenchmarkTask(ABC):
    """ベンチマークタスクの抽象基底クラス。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str, hardware_profile: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.device = device
        self.hardware_profile = hardware_profile

    @abstractmethod
    def prepare_data(self, data_dir: str) -> Tuple[Dataset, Dataset]:
        """データセットを準備し、train/validationのDatasetオブジェクトを返す。"""
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """タスク固有のcollate_fnを返す。"""
        pass

    @abstractmethod
    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        """タスクに適したSNNまたはANNモデルを構築する。"""
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        """モデルを評価し、結果を辞書で返す。"""
        pass

# --- 感情分析タスク (SST-2) ---
class SST2Task(BenchmarkTask):
    """GLUEベンチマークのSST-2 (感情分析) タスク。"""
    
    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        dataset = load_dataset("glue", "sst2")
        
        def _load_split(split):
            data = []
            for ex in dataset[split]:
                data.append({"text": ex['sentence'], "label": ex['label']})
            return GenericDataset(data)
            
        return _load_split("train"), _load_split("validation")

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Dict[str, Any]]):
            texts = [item['text'] for item in batch]
            targets = [item['label'] for item in batch]
            tokenized = self.tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            return {
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask'],
                "labels": torch.tensor(targets, dtype=torch.long)
            }
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        # 分類タスク用にモデルをラップする
        class SNNClassifier(nn.Module):
            def __init__(self, snn_backbone):
                super().__init__()
                self.snn_backbone = snn_backbone
                # BreakthroughSNNの隠れ層の次元に合わせて分類器を定義
                if isinstance(snn_backbone, SNNCore) and isinstance(snn_backbone.model, BreakthroughSNN):
                    in_features = snn_backbone.model.d_state * snn_backbone.model.num_layers
                elif isinstance(snn_backbone, BreakthroughSNN):
                    in_features = snn_backbone.d_state * snn_backbone.num_layers
                else: # ANNBaselineModel
                    in_features = snn_backbone.d_model
                self.classifier = nn.Linear(in_features, 2)
            
            def forward(self, input_ids, **kwargs):
                # SNNの出力から最後のタイムステップの特徴量を取得して分類
                hidden_states, spikes, mem = self.snn_backbone(
                    input_ids,
                    return_spikes=True,
                    output_hidden_states=True # このフラグで隠れ状態を取得
                )
                pooled_output = hidden_states[:, -1, :] # 最後のトークンの特徴量を使用
                logits = self.classifier(pooled_output)
                
                return logits, spikes, mem

        if model_type == 'SNN':
            # SNNCoreでラップしてSNNモデルを構築
            snn_config = {
                "architecture_type": "predictive_coding",
                "d_model": 64,
                "d_state": 16,
                "num_layers": 4,
                "time_steps": 64,
                "n_head": 2,
                "neuron": {'type': 'lif'}
            }
            # --- ◾️◾️◾️◾️◾️↓修正↓◾️◾️◾️◾️◾️ ---
            # DictConfigに変換
            backbone = SNNCore(config=OmegaConf.create(snn_config), vocab_size=vocab_size)
            # --- ◾️◾️◾️◾️◾️↑修正↑◾️◾️◾️◾️◾️ ---
            return SNNClassifier(backbone)
        else:
            ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2, 'num_classes': 2}
            return ANNBaselineModel(vocab_size=vocab_size, **ann_params)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0
        num_neurons = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating SST-2"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                outputs, spikes, _ = model(**inputs)
                if spikes is not None:
                    total_spikes += spikes.sum().item()
                
                preds = torch.argmax(outputs, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        dataset_size = len(cast(Sized, loader.dataset))
        avg_spikes = total_spikes / dataset_size if total_spikes > 0 else 0.0
        
        energy_j = calculate_energy_consumption(
            avg_spikes_per_sample=avg_spikes,
            num_neurons=num_neurons,
            energy_per_synop=self.hardware_profile["energy_per_synop"]
        )

        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes,
            "estimated_energy_j": energy_j,
        }
