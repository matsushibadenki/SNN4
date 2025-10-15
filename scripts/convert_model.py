# scripts/convert_model.py
# ANNモデルからSNNモデルへの変換・蒸留を実行するためのスクリプト
#
# 変更点:
# - [改善] オンライン知識蒸留で、ダミーではなく指定された教師モデルをロードするように修正。
# - [修正] 基本設定ファイル(base_config.yaml)を読み込むように修正し、Tokenizerの読み込みエラーを解消。
# - [改善 v2] ann2snn_cnn.py の機能を統合し、CNNモデルの直接変換に対応。

import argparse
import sys
from pathlib import Path
import torch
from collections import OrderedDict

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.conversion import AnnToSnnConverter
from snn_research.benchmark.ann_baseline import ANNBaselineModel, SimpleCNN
from snn_research.core.snn_core import SNNCore
from omegaconf import OmegaConf

def convert_cnn_weights(ann_model_path: str, snn_config_path: str, output_path: str):
    """
    学習済みSimpleCNNの重みをSpikingCNNにコピーする。
    (旧 scripts/ann2snn_cnn.py の機能)
    """
    print("--- ANN (SimpleCNN) to SNN (SpikingCNN) Weight Conversion ---")
    ann_model = SimpleCNN(num_classes=10)
    ann_state_dict = torch.load(ann_model_path, map_location='cpu')
    if list(ann_state_dict.keys())[0].startswith('module.'):
        ann_state_dict = OrderedDict((k[7:], v) for k, v in ann_state_dict.items())
    ann_model.load_state_dict(ann_state_dict)
    ann_model.eval()
    print(f"✅ ANN model loaded from '{ann_model_path}'.")

    snn_config = OmegaConf.load(snn_config_path)
    snn_model_container = SNNCore(config=snn_config.model, vocab_size=10)
    snn_model = snn_model_container.model
    snn_model.eval()
    print(f"✅ SNN model initialized from '{snn_config_path}'.")

    snn_state_dict = snn_model.state_dict()
    for name, param in ann_model.state_dict().items():
        if name in snn_state_dict and snn_state_dict[name].shape == param.shape:
            snn_state_dict[name].copy_(param)
    snn_model.load_state_dict(snn_state_dict)
    print("✅ Weight copy complete.")

    torch.save(snn_model.state_dict(), output_path)
    print(f"✅ Converted SNN model saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(
        description="ANNモデルからSNNへの変換・蒸留ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["convert", "distill", "cnn-convert"],
        help="実行する手法を選択:\n"
             " - convert: ANNの重みをSNNに直接コピーします。\n"
             " - distill: ANNを教師役としてSNNをオンラインで蒸留学習させます。\n"
             " - cnn-convert: SimpleCNN(ANN)をSpikingCNN(SNN)に変換します。"
    )
    parser.add_argument(
        "--ann_model_path",
        type=str,
        required=True,
        help="変換元となるANNモデルのパス (.safetensors, .gguf, .pth)。"
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
        help="変換・学習後にSNNモデルを保存するパス。"
    )
    args = parser.parse_args()

    if args.method == "cnn-convert":
        convert_cnn_weights(args.ann_model_path, args.snn_model_config, args.output_snn_path)
        return

    # DIコンテナからSNNモデルのインスタンスと設定を取得
    container = TrainingContainer()
    container.config.from_yaml("configs/base_config.yaml")
    container.config.from_yaml(args.snn_model_config)
    snn_model = container.snn_model()
    snn_config = container.config.model.to_dict()

    # コンバータを初期化
    converter = AnnToSnnConverter(snn_model=snn_model, model_config=snn_config)

    if args.method == "convert":
        converter.convert_weights(
            ann_model_path=args.ann_model_path,
            output_path=args.output_snn_path
        )
    elif args.method == "distill":
        print(f"教師ANNモデルを {args.ann_model_path} からロードします。")
        teacher_model = ANNBaselineModel(
            vocab_size=container.tokenizer.provided.vocab_size(),
            d_model=snn_config['d_model'],
            nhead=snn_config['n_head'],
            d_hid=snn_config['d_model'] * 2,
            nlayers=snn_config['num_layers'],
            num_classes=container.tokenizer.provided.vocab_size()
        )
        try:
            ann_weights = converter._load_ann_weights(args.ann_model_path)
            teacher_model.load_state_dict(ann_weights, strict=False)
            print("✅ 教師モデルの重みを正常にロードしました。")
        except Exception as e:
            print(f"❌ 教師モデルの重みロードに失敗しました: {e}")
            return

        dummy_dataset = [torch.randint(0, container.tokenizer.provided.vocab_size(), (snn_config['time_steps'],)) for _ in range(32)]
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)

        converter.run_online_distillation(
            ann_teacher_model=teacher_model,
            dummy_data_loader=dummy_loader,
            output_path=args.output_snn_path,
            epochs=3
        )

if __name__ == "__main__":
    main()
