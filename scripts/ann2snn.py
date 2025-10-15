# ファイルパス: scripts/ann2snn.py
# Title: ANNからSNNへの変換スクリプト
# Description:
# snn_4_ann_parity_plan.mdに基づき、ANNモデルからSNNモデルへの変換を実行します。
# ANNで学習済みの重みをSNNアーキテクチャにマッピングすることで、
# SNNの学習を効率化し、高性能な初期モデルを生成することを目的とします。

import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.conversion import AnnToSnnConverter

def main():
    """
    ANNモデルの重みをSNNモデルに変換（コピー）する。
    """
    parser = argparse.ArgumentParser(
        description="ANN to SNN Conversion Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--ann_model_path",
        type=str,
        required=True,
        help="変換元となる学習済みANNモデルのパス (.safetensors, .gguf, .pth)。"
    )
    parser.add_argument(
        "--snn_model_config",
        type=str,
        default="configs/models/small.yaml",
        help="変換先となるSNNモデルのアーキテクチャ設定ファイル。"
    )
    parser.add_argument(
        "--output_snn_path",
        type=str,
        required=True,
        help="変換後にSNNモデルを保存するパス (.pth)。"
    )
    args = parser.parse_args()

    print("--- ANN to SNN Conversion Process ---")

    # 1. DIコンテナからSNNモデルのインスタンスと設定を取得
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(args.snn_model_config)
    snn_model = container.snn_model()
    snn_config = container.config.model.to_dict()

    # 2. コンバータを初期化
    converter = AnnToSnnConverter(snn_model=snn_model, model_config=snn_config)

    # 3. 重み変換を実行
    print(f"🔄 Converting weights from '{args.ann_model_path}'...")
    converter.convert_weights(
        ann_model_path=args.ann_model_path,
        output_path=args.output_snn_path,
        calibration_loader=None  # 必要に応じてキャリブレーションデータローダーを指定
    )

    print("\n✅ Conversion complete.")
    print(f"   - SNN model saved to: {args.output_snn_path}")
    print("---------------------------------------")

if __name__ == "__main__":
    main()