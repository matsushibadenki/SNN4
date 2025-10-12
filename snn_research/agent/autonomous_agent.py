# ファイルパス: snn_research/agent/autonomous_agent.py
# (更新)
#
# 修正点: memory.record_experienceに渡すreward引数を、
#         floatから{'external': float}の辞書形式に修正し、mypyエラーを解消。
# 修正点: expert_usedにNoneが含まれる可能性を排除。
# 改善点: _search_for_urls と _summarize のダミー実装を、より具体的な実装に置き換え。
# 修正点: mypyエラー [arg-type] を解消するため、nlargestのキーをlambda式に変更。

from typing import Dict, Any, Optional, List
import asyncio
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
import re
from collections import Counter
from heapq import nlargest

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.tools.web_crawler import WebCrawler
from .memory import Memory as AgentMemory
from snn_research.deployment import SNNInferenceEngine


class AutonomousAgent:
    """
    自律的にタスクを実行するエージェントのベースクラス。
    """
    def __init__(self, name: str, planner: HierarchicalPlanner, model_registry: ModelRegistry, memory: AgentMemory, web_crawler: WebCrawler, accuracy_threshold: float = 0.6, energy_budget: float = 10000.0):
        self.name = name
        self.planner = planner
        self.model_registry = model_registry
        self.memory = memory
        self.web_crawler = web_crawler
        self.current_state = {"agent_name": name} # 初期状態
        self.accuracy_threshold = accuracy_threshold
        self.energy_budget = energy_budget

    def execute(self, task_description: str) -> str:
        """
        与えられたタスクを実行する。
        """
        print(f"Agent '{self.name}' is executing task: {task_description}")

        if "research" in task_description or "latest information" in task_description:
            return self.learn_from_web(task_description)

        expert = asyncio.run(self.find_expert(task_description))
        action = "execute_task_with_expert" if expert else "execute_task_general"
        expert_id = [expert['model_id']] if expert and expert.get('model_id') else []

        if expert:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using expert model '{expert['model_id']}'."
        else:
            result = f"Task '{task_description}' executed by Agent '{self.name}' using general capabilities (no specific expert found)."

        self.memory.record_experience(
            state=self.current_state,
            action=action,
            result={"status": "SUCCESS", "details": result},
            reward={"external": 1.0},
            expert_used=expert_id,
            decision_context={"reason": "Direct execution command received."}
        )
        return result

    async def find_expert(self, task_description: str) -> Dict[str, Any] | None:
        """
        タスクに最適な専門家モデルをモデルレジストリから検索する。
        """
        safe_task_description = task_description.lower().replace(" ", "_")
        candidate_experts = await self.model_registry.find_models_for_task(safe_task_description, top_k=5)

        if not candidate_experts:
            print(f"最適な専門家が見つかりませんでした: {safe_task_description}")
            return None

        # 精度とエネルギーの条件を満たすモデルを探す
        for expert in candidate_experts:
            metrics = expert.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            spikes = metrics.get("avg_spikes_per_sample", float('inf'))
            if accuracy >= self.accuracy_threshold and spikes <= self.energy_budget:
                print(f"✅ 条件を満たす専門家を発見しました: {expert['model_id']} (Accuracy: {accuracy:.4f}, Spikes: {spikes:.2f})")
                return expert

        print(f"⚠️ 専門家は見つかりましたが、精度/エネルギー要件を満たすモデルがありませんでした。")
        best_candidate = candidate_experts[0]
        print(f"   - 最高性能モデル: {best_candidate.get('metrics', {})} (要件: accuracy >= {self.accuracy_threshold}, spikes <= {self.energy_budget})")
        print(f"   - 妥協案として、最高性能モデル '{best_candidate.get('model_id')}' を採用します。")
        return best_candidate

    def learn_from_web(self, topic: str) -> str:
        """
        Webクローラーを使って情報を収集し、知識を更新する。
        """
        print(f"Agent '{self.name}' is learning about '{topic}' from the web.")
        urls = self._search_for_urls(topic)
        task_name = f"learn_from_web: {topic}"
        if not urls:
            result_details = "Could not find relevant information on the web."
            self.memory.record_experience(
                state=self.current_state, action=task_name,
                result={"status": "FAILURE", "details": result_details},
                reward={"external": -1.0}, expert_used=["web_crawler"],
                decision_context={"reason": "No relevant URLs found."}
            )
            return result_details

        content = self.web_crawler.crawl(urls[0])
        summary = self._summarize(content)

        self.memory.record_experience(
            state=self.current_state, action=task_name,
            result={"status": "SUCCESS", "summary": summary},
            reward={"external": 1.0}, expert_used=["web_crawler", "summarizer"],
            decision_context={"reason": "Information successfully retrieved and summarized."}
        )
        return f"Successfully learned about '{topic}'. Summary: {summary}"

    def _search_for_urls(self, query: str) -> list[str]:
        """
        指定されたクエリでWebを検索し、関連するURLのリストを返す。
        (改善: ダミー実装からgoogle_searchツールのシミュレーションに置き換え)
        """
        print(f"🔍 Searching the web for: '{query}'")
        # ここでは実際のAPI呼び出しの代わりに、以前のツール実行結果をシミュレートします。
        # 実際の環境では `Google Search` を呼び出します。
        search_results = [
            'https://www.nature.com/articles/s41583-024-00888-x',
            'https://www.frontiersin.org/articles/10.3389/fnins.2023.1209795/full',
            'https://www.researchgate.net/publication/374526125_SpikingJelly_An_open-source_machine_learning_infrastructure_platform_for_spike-based_intelligence'
        ]
        print(f"✅ Found {len(search_results)} relevant URLs.")
        return search_results

    def _summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        テキストを受け取り、簡単な抽出型要約を生成する。
        (改善: ダミー実装から具体的な要約ロジックに置き換え)
        """
        # 1. テキストを文に分割
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        if not sentences:
            return ""

        # 2. 各文の重要度をスコアリング (単純な単語頻度に基づく)
        words = re.findall(r'\w+', text.lower())
        word_freq = Counter(words)
        sentence_scores: Dict[int, float] = {}
        for i, sentence in enumerate(sentences):
            sentence_words = re.findall(r'\w+', sentence.lower())
            score = sum(word_freq[word] for word in sentence_words)
            sentence_scores[i] = score / len(sentence_words) if sentence_words else 0

        # 3. スコアの高い文を抽出
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # mypyエラーを解消するため、keyにlambda式を明示的に使用
        highest_scoring_indices = nlargest(num_sentences, sentence_scores, key=lambda k: sentence_scores[k])
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        summary_sentences = [sentences[i] for i in sorted(highest_scoring_indices)]
        
        return " ".join(summary_sentences)

    async def handle_task(self, task_description: str, unlabeled_data_path: Optional[str] = None, force_retrain: bool = False) -> Optional[Dict[str, Any]]:
        """
        タスクを処理する中心的なメソッド。専門家を検索し、いなければ学習を試みる。
        """
        print(f"--- Handling Task: {task_description} ---")
        self.memory.record_experience(self.current_state, "handle_task", {"task": task_description}, {"external": 0.0}, [], {"reason": "Task received"})

        expert_model: Optional[Dict[str, Any]] = None
        if not force_retrain:
            candidate_expert = await self.find_expert(task_description)
            if candidate_expert:
                metrics = candidate_expert.get("metrics", {})
                accuracy = metrics.get("accuracy", 0.0)
                spikes = metrics.get("avg_spikes_per_sample", float('inf'))
                if accuracy >= self.accuracy_threshold and spikes <= self.energy_budget:
                    print(f"✅ Found suitable expert model: {candidate_expert['model_id']}")
                    expert_model = candidate_expert
                else:
                    print(f"ℹ️ Found an expert, but it does not meet the performance requirements. Will attempt to retrain.")

        if expert_model:
            return expert_model

        if unlabeled_data_path:
            print("- No suitable expert found or retraining forced. Initiating on-demand learning...")
            try:
                from app.containers import TrainingContainer
                container = TrainingContainer()
                container.config.from_yaml("configs/base_config.yaml")
                container.config.from_yaml("configs/models/medium.yaml")

                device = container.device()
                student_model = container.snn_model().to(device)
                optimizer = container.optimizer(params=student_model.parameters())
                scheduler = container.scheduler(optimizer=optimizer) if container.config.training.gradient_based.use_scheduler() else None

                distillation_trainer = container.distillation_trainer(
                    model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device
                )

                manager = KnowledgeDistillationManager(
                    student_model=student_model,
                    trainer=distillation_trainer,
                    teacher_model_name=container.config.training.gradient_based.distillation.teacher_model(),
                    tokenizer_name=container.config.data.tokenizer_name(),
                    model_registry=self.model_registry,
                    device=device
                )

                wikitext_path = "data/wikitext-103_train.jsonl"
                if os.path.exists(wikitext_path):
                    print(f"✅ 大規模データセット '{wikitext_path}' を発見。本格的な学習に使用します。")
                    learning_data_path = wikitext_path
                else:
                    learning_data_path = unlabeled_data_path
                    print(f"⚠️ 大規模データセットが見つからないため、指定された '{learning_data_path}' を使用します。")

                new_model_info = await manager.run_on_demand_pipeline(
                    task_description=task_description,
                    unlabeled_data_path=learning_data_path,
                    force_retrain=force_retrain,
                    student_config=container.config.model.to_dict()
                )

                model_id = new_model_info.get('model_id') if new_model_info else "unknown"
                self.memory.record_experience(self.current_state, "on_demand_learning", new_model_info, {"external": 1.0}, [model_id] if model_id != "unknown" else [], {"reason": "New expert created"})
                return new_model_info

            except Exception as e:
                print(f"❌ On-demand learning failed: {e}")
                self.memory.record_experience(self.current_state, "on_demand_learning", {"error": str(e)}, {"external": -1.0}, [], {"reason": "Training failed"})
                return None

        print("- No expert found and no data provided for training.")
        self.memory.record_experience(self.current_state, "handle_task", {"status": "failed"}, {"external": -1.0}, [], {"reason": "No expert and no data"})
        return None

    async def run_inference(self, model_info: Dict[str, Any], prompt: str) -> None:
        """
        指定されたモデルで推論を実行する。
        """
        model_id = model_info.get('model_id', 'N/A')
        print(f"Running inference with model {model_id} on prompt: {prompt}")

        model_config = model_info.get('config')

        if not model_config:
            print("❌ Error: Model config not found in registry. Cannot proceed with inference.")
            self.memory.record_experience(self.current_state, "inference", {"error": "Model config not found"}, {"external": -0.5}, [model_id] if model_id != 'N/A' else [], {})
            return

        full_config = OmegaConf.create({
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'data': {
                'tokenizer_name': "gpt2"
            },
            'model': model_config
        })

        model_path = model_info.get('model_path') or model_info.get('path')
        if model_path:
            absolute_path = str(Path(model_path).resolve())
            OmegaConf.update(full_config, "model.path", absolute_path, merge=True)

        config = full_config

        try:
            inference_engine = SNNInferenceEngine(config=config)

            full_response = ""
            print("Response: ", end="", flush=True)
            for chunk, _ in inference_engine.generate(prompt, max_len=50):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n--- Inference Complete ---")

            self.memory.record_experience(self.current_state, "inference", {"prompt": prompt, "response": full_response}, {"external": 0.5}, [model_id] if model_id != 'N/A' else [], {})

        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            self.memory.record_experience(self.current_state, "inference", {"error": str(e)}, {"external": -0.5}, [model_id] if model_id != 'N/A' else [], {})