# ファイルパス: snn_research/benchmark/__init__.py
# (修正)
# 修正: 循環参照を解消するため、TASK_REGISTRYをここで定義する。
# 改善(snn_4_ann_parity_plan): 新しいCIFAR10Taskをレジストリに追加。
# 修正(mypy): [abstract]エラーを解消するため、TASK_REGISTRYに型ヒントを追加。

from typing import Dict, Type

from .ann_baseline import ANNBaselineModel
from .tasks import SST2Task, CIFAR10Task, BenchmarkTask
from .metrics import calculate_accuracy

TASK_REGISTRY: Dict[str, Type[BenchmarkTask]] = {
    "sst2": SST2Task,
    "cifar10": CIFAR10Task,
}

__all__ = ["ANNBaselineModel", "SST2Task", "CIFAR10Task", "calculate_accuracy", "TASK_REGISTRY", "BenchmarkTask"]
