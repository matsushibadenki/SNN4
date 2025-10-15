# ファイルパス: snn_research/hardware/compiler.py
# (更新)
#
# Title: ニューロモーフィック・コンパイラ
#
# Description:
# - mypyエラーを解消するため、typing.castを使用してモジュールの型を明示的に指定。
# - ROADMAPフェーズ6に基づき、simulate_on_hardwareメソッドを実装。
# - 改善点(v3): 学習則のパラメータもコンパイルしてハードウェア構成に含める機能を追加。
# - 改善点(snn_4_ann_parity_plan): 学習則のシリアライズをより堅牢な方法に変更。

from typing import Dict, Any, List, cast
import yaml
import time
import os

from snn_research.bio_models.simple_network import BioSNN
from snn_research.bio_models.lif_neuron import BioLIFNeuron
from snn_research.hardware.profiles import get_hardware_profile
from snn_research.learning_rules.base_rule import BioLearningRule

class NeuromorphicCompiler:
    """
    SNNモデルをニューロモーフィックハードウェア用の構成にコンパイルする。
    """
    def __init__(self, hardware_profile_name: str = "default"):
        """
        Args:
            hardware_profile_name (str): 'profiles.py'で定義されたハードウェアプロファイル名。
        """
        self.hardware_profile = get_hardware_profile(hardware_profile_name)
        print(f"🔩 ニューロモーフィック・コンパイラが初期化されました (ターゲット: {self.hardware_profile['name']})。")

    def compile(self, model: BioSNN, output_path: str):
        """
        BioSNNモデルを解析し、ハードウェア構成ファイルを生成する。

        Args:
            model (BioSNN): コンパイル対象の学習済みSNNモデル。
            output_path (str): 生成されたハードウェア構成ファイルの保存先 (YAML形式)。
        """
        print(f"⚙️ モデル '{type(model).__name__}' のコンパイルを開始...")

        hardware_config: Dict[str, Any] = {
            "target_hardware": self.hardware_profile['name'],
            "neuron_cores": [],
            "synaptic_connectivity": [],
            "learning_rule_config": {}
        }

        # 1. ニューロンのマッピング (Neuron Core Mapping)
        neuron_offset = 0
        for i, layer_module in enumerate(model.layers):
            layer = cast(BioLIFNeuron, layer_module)
            num_neurons = layer.n_neurons
            core_config = {
                "core_id": i,
                "neuron_type": type(layer).__name__,
                "num_neurons": num_neurons,
                "neuron_ids": list(range(neuron_offset, neuron_offset + num_neurons)),
                "parameters": { "tau_mem": layer.tau_mem, "v_threshold": layer.v_thresh }
            }
            hardware_config["neuron_cores"].append(core_config)
            neuron_offset += num_neurons
        
        print(f"  - {len(model.layers)}個のニューロン層を{len(model.layers)}個のコアにマッピングしました。")

        # 2. シナプスのマッピング (Synaptic Connectivity)
        for i, weight_matrix in enumerate(model.weights):
            pre_core_size = model.layer_sizes[i]
            pre_core_offset = hardware_config["neuron_cores"][i-1]["neuron_ids"][0] if i > 0 else 0
            post_core = hardware_config["neuron_cores"][i]
            post_core_offset = post_core["neuron_ids"][0]
            post_num_neurons = int(post_core["num_neurons"])

            connections = []
            for pre_id_local in range(pre_core_size):
                for post_id_local in range(post_num_neurons):
                    weight = weight_matrix[post_id_local, pre_id_local].item()
                    if weight > 0:
                        connections.append({
                            "source_neuron": pre_core_offset + pre_id_local,
                            "target_neuron": post_core_offset + post_id_local,
                            "weight": round(weight, 4),
                            "delay": 1
                        })
            
            hardware_config["synaptic_connectivity"].append({
                "source_core": i - 1 if i > 0 else "input",
                "target_core": i,
                "num_connections": len(connections),
                "connections": connections
            })
        
        print(f"  - {len(model.weights)}個のシナプス接続をマッピングしました。")

        # 3. 学習則のマッピング (Learning Rule Mapping)
        if hasattr(model, 'learning_rule') and isinstance(model.learning_rule, BioLearningRule):
            rule = model.learning_rule
            rule_name = type(rule).__name__
            # __dict__に頼らず、既知のパラメータを明示的に取得
            rule_params = {
                "learning_rate": getattr(rule, 'learning_rate', None),
                "a_plus": getattr(rule, 'a_plus', None),
                "a_minus": getattr(rule, 'a_minus', None),
                "tau_trace": getattr(rule, 'tau_trace', None),
                "tau_eligibility": getattr(rule, 'tau_eligibility', None),
            }
            # Noneのパラメータは除外
            rule_params = {k: v for k, v in rule_params.items() if v is not None}
            
            hardware_config["learning_rule_config"] = {
                "rule_name": rule_name,
                "parameters": rule_params
            }
            print(f"  - 学習則 '{rule_name}' をハードウェア構成にマッピングしました。")

        # 4. 設定ファイルの保存
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(hardware_config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ コンパイル完了。ハードウェア構成を '{output_path}' に保存しました。")
    
    def simulate_on_hardware(self, compiled_config_path: str, total_spikes: int, time_steps: int) -> Dict[str, float]:
        """
        コンパイル済み設定に基づき、ハードウェア上での性能をシミュレートする。
        """
        print(f"\n--- ⚡️ ハードウェアシミュレーション開始 ({self.hardware_profile['name']}) ---")
        
        with open(compiled_config_path, 'r') as f:
            config = yaml.safe_load(f)

        energy_per_synop = self.hardware_profile['energy_per_synop']
        estimated_energy = total_spikes * energy_per_synop
        print(f"  - 総スパイク数: {total_spikes}")
        print(f"  - シナプス演算あたりのエネルギー: {energy_per_synop:.2e} J")
        print(f"  -推定総エネルギー消費: {estimated_energy:.2e} J")
        
        time_per_step_ms = 1.0 
        estimated_time_ms = time_steps * time_per_step_ms
        print(f"  - タイムステップ数: {time_steps}")
        print(f"  - ステップあたりの処理時間 (仮定): {time_per_step_ms} ms")
        print(f"  - 推定処理時間: {estimated_time_ms} ms")

        report = {
            "estimated_energy_joules": estimated_energy,
            "estimated_processing_time_ms": estimated_time_ms,
            "total_spikes_simulated": total_spikes
        }
        print("--- ✅ シミュレーション完了 ---")
        return report
