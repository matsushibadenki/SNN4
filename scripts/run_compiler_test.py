# ファイルパス: scripts/run_compiler_test.py
# (新規作成)
#
# Title: ニューロモーフィック・コンパイラ テストスクリプト
#
# Description:
# - ロードマップ「ニューロモーフィックハードウェアへの最適化」で実装した
#   NeuromorphicCompilerの動作を検証するためのスクリプト。
# - ダミーのBioSNNモデルを構築し、それをハードウェア構成ファイルに
#   コンパイルするプロセスを実行する。

import sys
from pathlib import Path
import os

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment
from snn_research.hardware.compiler import NeuromorphicCompiler

def main():
    """
    NeuromorphicCompilerのテストを実行する。
    """
    print("--- ニューロモーフィック・コンパイラ テスト開始 ---")

    # 1. コンパイル対象のダミーBioSNNモデルを構築
    learning_rule = CausalTraceCreditAssignment(
        learning_rate=0.005, a_plus=1.0, a_minus=1.0,
        tau_trace=20.0, tau_eligibility=50.0
    )
    model = BioSNN(
        layer_sizes=[10, 20, 5], # 3層のネットワーク
        neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0},
        learning_rule=learning_rule
    )
    print("✅ ダミーのBioSNNモデルを構築しました。")

    # 2. コンパイラの初期化
    compiler = NeuromorphicCompiler(hardware_profile_name="default")

    # 3. コンパイルの実行
    output_dir = "runs/compiler_tests"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "compiled_hardware_config.yaml")
    
    compiler.compile(model, output_path)

    # 4. 結果の確認
    if os.path.exists(output_path):
        print(f"\n✅ テスト成功: コンパイルされた設定ファイルが '{output_path}' に生成されました。")
    else:
        print(f"\n❌ テスト失敗: 設定ファイルが生成されませんでした。")

    print("--- ニューロモーフィック・コンパイラ テスト終了 ---")


if __name__ == "__main__":
    main()
