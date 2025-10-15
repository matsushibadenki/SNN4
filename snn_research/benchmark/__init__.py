# matsushibadenki/snn4/snn_research/benchmark/__init__.py
# (修正)
# 修正: 循環参照を解消するため、TASK_REGISTRYをここで定義する。
# 改善(snn_4_ann_parity_plan): 新しいCIFAR10Taskをレジストリに追加。

from .ann_baseline import ANNBaselineModel
from .tasks import SST2Task, CIFAR10Task
from .metrics import calculate_accuracy

TASK_REGISTRY = {
    "sst2": SST2Task,
    "cifar10": CIFAR10Task,
}

__all__ = ["ANNBaselineModel", "SST2Task", "CIFAR10Task", "calculate_accuracy", "TASK_REGISTRY"]
