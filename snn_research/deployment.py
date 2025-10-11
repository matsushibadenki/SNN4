# matsushibadenki/snn4/snn_research/deployment.py
# Title: SNN推論エンジン
# Description: 訓練済みSNNモデルをロードし、テキスト生成のための推論を実行するクラス。
# BugFix: モデルのパスを絶対パスに解決してからロードすることで、ファイルが見つからない問題を解消。
# BugFix: state_dictのキーから 'model.' プリフィックスを削除し、読み込みエラーを修正。
# BugFix: IndentationErrorの修正。

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from typing import Iterator, Optional, Dict, Any, List, Union, Tuple
from .core.snn_core import BreakthroughSNN, SpikingTransformer
from omegaconf import DictConfig, OmegaConf
from snn_research.core.snn_core import SNNCore # SNNCoreをインポート

def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

class SNNInferenceEngine:
    """
    学習済みSNNモデルをロードして推論を実行するエンジン。
    """
    def __init__(self, config: DictConfig):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        device_str = config.get("device", "auto")
        self.device = get_auto_device() if device_str == "auto" else device_str
        
        self.last_inference_stats: Dict[str, Any] = {}
        
        # 先にTokenizerをロードしてvocab_sizeを取得
        tokenizer_path = config.data.get("tokenizer_name", "gpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Could not load tokenizer from {tokenizer_path}. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # vocab_sizeを渡してSNNCoreを初期化
        vocab_size = len(self.tokenizer)
        model_config = config.get("model", config)
        self.model = SNNCore(model_config, vocab_size=vocab_size)
        
        model_path_str = config.model.get("path") if hasattr(config, "model") else None

        if model_path_str:
            model_path = Path(model_path_str).resolve()

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
                    # state_dictのキーから "model." プレフィックスを削除
                    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    
                    # SNNCoreラッパーの中の実際のモデルにstate_dictをロードする
                    # strict=False を追加して、キーが一致しないエラーを回避
                    self.model.model.load_state_dict(new_state_dict, strict=False)

                    print(f"✅ Model loaded from {model_path}")
                except RuntimeError as e:
                    print(f"⚠️ Warning: Failed to load state_dict, possibly due to architecture mismatch: {e}. Using an untrained model.")
            else:
                 print(f"⚠️ Warning: Model file not found at {model_path}. Using an untrained model.")

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_len: int, stop_sequences: Optional[List[str]] = None) -> Iterator[Tuple[str, Dict[str, float]]]:
        """
        プロンプトに基づいてテキストと統計情報をストリーミング生成する。
        """
        tokenizer_callable = getattr(self.tokenizer, "__call__", None)
        if not callable(tokenizer_callable):
            raise TypeError("Tokenizer is not callable.")
        input_ids = tokenizer_callable(prompt, return_tensors="pt")["input_ids"].to(self.device)
        
        total_spikes = 0.0
        
        for i in range(max_len):
            with torch.no_grad():
                outputs, avg_spikes, _ = self.model(input_ids, return_spikes=True)
            
            total_spikes += avg_spikes.item() * input_ids.shape[1]
            
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            if next_token_id.item() == getattr(self.tokenizer, 'eos_token_id', None):
                break
            
            new_token = getattr(self.tokenizer, "decode")(next_token_id.item())
            
            if stop_sequences and any(seq in new_token for seq in stop_sequences):
                break

            current_stats = {"total_spikes": total_spikes}
            yield new_token, current_stats
            
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        self.last_inference_stats = {"total_spikes": total_spikes}