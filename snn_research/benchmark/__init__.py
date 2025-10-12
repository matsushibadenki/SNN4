# matsushibadenki/snn4/snn_research/benchmark/__init__.py
# (修正)
# 修正: 循環参照を解消するため、TASK_REGISTRYをここで定義する。

from .ann_baseline import ANNBaselineModel
from .tasks import SST2Task
from .metrics import calculate_accuracy

TASK_REGISTRY = {
    "sst2": SST2Task,
}

__all__ = ["ANNBaselineModel", "SST2Task", "calculate_accuracy", "TASK_REGISTRY"]
