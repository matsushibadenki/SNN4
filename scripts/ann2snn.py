# ファイルパス: scripts/ann2snn_cnn.py
# Title: ANN(SimpleCNN)からSNN(SpikingCNN)への重み変換スクリプト
# Description:
# snn_4_ann_parity_plan.mdに基づき、学習済みのSimpleCNNモデルの重みを、
# 対応するアーキテクチャを持つSpikingCNNモデルにコピーする。
# これにより、ANNの知識をSNNに転移させ、SNNの学習を効率化することを目的とする。

import argparse
import torch
from collections import OrderedDict
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.benchmark.ann_baseline import SimpleCNN
from snn_research.core.snn_core import SNNCore
from omegaconf import OmegaConf

def convert_cnn_weights(ann_model_path: str, snn_config_path: str, output_path: str):
    """
    学習済みSimpleCNNの重みをSpikingCNNにコピーする。
    """
    print("--- ANN (SimpleCNN) to SNN (SpikingCNN) Weight Conversion ---")

    # 1. 学習済みANNモデルをロード
    print(f"🔄 Loading trained ANN model from '{ann_model_path}'...")
    ann_model = SimpleCNN(num_classes=10)
    ann_state_dict = torch.load(ann_model_path)
    # DDPなどで保存された場合の'module.'プレフィックスを削除
    if list(ann_state_dict.keys())[0].startswith('module.'):
        ann_state_dict = OrderedDict((k[7:], v) for k, v in ann_state_dict.items())
    ann_model.load_state_dict(ann_state_dict)
    ann_model.eval()
    print("✅ ANN model loaded.")

    # 2. 変換先のSNNモデルを初期化
    print(f"🔄 Initializing target SNN model from '{snn_config_path}'...")
    snn_config = OmegaConf.load(snn_config_path)
    snn_model_container = SNNCore(config=snn_config.model, vocab_size=10) # vocab_sizeはnum_classesとして使用
    snn_model = snn_model_container.model
    snn_model.eval()
    print("✅ SNN model initialized.")

    # 3. 重みのコピー
    print("🔄 Copying weights from ANN to SNN...")
    snn_state_dict = snn_model.state_dict()
    
    # SimpleCNNとSpikingCNNの対応するレイヤー名をマッピング
    # (例: 'features.0.weight' -> 'features.0.weight')
    for name, param in ann_model.state_dict().items():
        if name in snn_state_dict:
            if snn_state_dict[name].shape == param.shape:
                snn_state_dict[name].copy_(param)
                print(f"  - Copied '{name}'")
            else:
                print(f"  - ⚠️ Shape mismatch for '{name}'. Skipping.")
        else:
            print(f"  - ⚠️ Layer '{name}' not found in SNN model. Skipping.")
            
    snn_model.load_state_dict(snn_state_dict)
    print("✅ Weight copy complete.")

    # 4. 変換済みSNNモデルを保存
    torch.save(snn_model.state_dict(), output_path)
    print(f"✅ Converted SNN model saved to '{output_path}'.")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert weights from a trained SimpleCNN to a SpikingCNN.")
    parser.add_argument(
        "--ann_model_path",
        type=str,
        required=True,
        help="Path to the trained SimpleCNN model checkpoint (.pth)."
    )
    parser.add_argument(
        "--snn_config_path",
        type=str,
        default="configs/cifar10_spikingcnn_config.yaml",
        help="Path to the SpikingCNN model configuration file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted SpikingCNN model (.pth)."
    )
    args = parser.parse_args()
    convert_cnn_weights(args.ann_model_path, args.snn_config_path, args.output_path)
