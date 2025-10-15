# ファイルパス: run_compiler_test.py
# (更新)
#
# Title: ニューロモーフィック・コンパイラ テストスクリプト
#
# Description:
# - ロードマップ「ニューロモーフィックハードウェアへの最適化」で実装した
#   NeuromorphicCompilerの動作を検証するためのスクリプト。
# - ダミーのBioSNNモデルを構築し、それをハードウェア構成ファイルに
#   コンパイルするプロセスを実行する。
#
# 改善点(v2):
# - ROADMAPフェーズ6に基づき、コンパイル後のハードウェア性能シミュレーションを実行する処理を追加。
# 改善点(v3): コンパイルされたファイルに学習則が含まれているか検証するテストを追加。
# 改善点(snn_4_ann_parity_plan):
# - 学習則のパラメータ検証をより厳密化。
# - 古いスクリプトを削除し、こちらに機能を統合。

import sys
from pathlib import Path
import os
import torch
import yaml

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from snn_research.bio_models.simple_network import BioSNN
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment
from snn_research.hardware.compiler import NeuromorphicCompiler

def main():
    """
    NeuromorphicCompilerのテストを実行する。
    """
    print("--- ニューロモーフィック・コンパイラ テスト開始 ---")

    # 1. コンパイル対象のダミーBioSNNモデルを構築
    learning_rate = 0.005
    learning_rule = CausalTraceCreditAssignment(
        learning_rate=learning_rate, a_plus=1.0, a_minus=1.0,
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
        print(f"\n✅ コンパイル成功: 設定ファイルが '{output_path}' に生成されました。")
        
        # 4.1. 学習則が正しくコンパイルされたか詳細に検証
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "learning_rule_config" in config, "学習則の設定がYAMLファイルにありません。"
        lr_config = config["learning_rule_config"]
        assert lr_config["rule_name"] == "CausalTraceCreditAssignment", "学習則の名前が一致しません。"
        assert "parameters" in lr_config, "学習則のパラメータがYAMLファイルにありません。"
        params = lr_config["parameters"]
        assert "learning_rate" in params, "学習率がコンパイルされていません。"
        assert abs(params["learning_rate"] - learning_rate) < 1e-6, "学習率の値が一致しません。"
        assert "tau_trace" in params, "tau_traceパラメータがコンパイルされていません。"
        print("  - 検証: 学習則のコンパイル結果は正常です。")

        # 5. ハードウェア上での性能シミュレーションを実行
        total_spikes_for_simulation = 15000
        time_steps_for_simulation = 100
        
        simulation_report = compiler.simulate_on_hardware(
            compiled_config_path=output_path,
            total_spikes=total_spikes_for_simulation,
            time_steps=time_steps_for_simulation
        )
        
        print("\n--- 📊 ハードウェアシミュレーション結果 ---")
        for key, value in simulation_report.items():
            print(f"  - {key}: {value:.4e}")
        print("------------------------------------------")
        
    else:
        print(f"\n❌ テスト失敗: 設定ファイルが生成されませんでした。")

    # 古いスクリプトを削除
    old_script_path = Path("scripts/run_compiler_test.py")
    if old_script_path.exists():
        os.remove(old_script_path)
        print(f"\n🗑️ 古いテストスクリプト '{old_script_path}' を削除しました。")


    print("\n--- ニューロモーフィック・コンパイラ テスト終了 ---")


if __name__ == "__main__":
    main()
