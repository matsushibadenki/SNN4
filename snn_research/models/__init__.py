# matsushibadenki/snn4/snn_research/models/__init__.py

# [!!DEPRECATION WARNING!!]
# このパッケージ内のモデルは旧バージョンであり、現在は snn_research.core.snn_core 内の
# 実装が主として利用されています。互換性のために残されていますが、
# 新規開発での利用は推奨されません。

from .attention import SpikeDrivenSelfAttention
from .spiking_transformer import SpikingTransformer, SpikingTransformerBlock

__all__ = ["SpikeDrivenSelfAttention", "SpikingTransformer", "SpikingTransformerBlock"]