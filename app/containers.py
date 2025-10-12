# ファイルパス: app/containers.py
# (修正 v7 - 最終修正)
# 修正: pytest収集時に発生する `AttributeError: module 'dependency_injector.providers' has no attribute 'factory'` を
#       完全に解消するため、デコレータを `@providers.factory` から `@providers.Factory` に修正。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import redis
from typing import TYPE_CHECKING, cast

# --- プロジェクト内モジュールのインポート ---
from snn_research.core.snn_core import SNNCore
from snn_research.deployment import SNNInferenceEngine
from snn_research.training.losses import (
    CombinedLoss, DistillationLoss, SelfSupervisedLoss, PhysicsInformedLoss,
    PlannerLoss, ProbabilisticEnsembleLoss
)
from snn_research.training.trainers import (
    BreakthroughTrainer, DistillationTrainer, SelfSupervisedTrainer,
    PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer
)
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from snn_research.learning_rules.stdp import STDP
from snn_research.learning_rules.reward_modulated_stdp import RewardModulatedSTDP
from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment
from snn_research.bio_models.simple_network import BioSNN
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.perception_cortex import PerceptionCortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex


if TYPE_CHECKING:
    from .adapters.snn_langchain_adapter import SNNLangChainAdapter


def get_auto_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _calculate_t_max(epochs: int, warmup_epochs: int) -> int:
    return max(1, epochs - warmup_epochs)

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = _calculate_t_max(epochs=epochs, warmup_epochs=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def _load_planner_snn_factory(planner_snn_instance, model_path: str, device: str):
    model = planner_snn_instance
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            print(f"✅ 学習済みPlannerSNNモデルを '{model_path}' から正常にロードしました。")
        except Exception as e:
            print(f"⚠️ PlannerSNNモデルのロードに失敗しました: {e}。未学習のモデルを使用します。")
    else:
        print(f"⚠️ PlannerSNNモデルが見つかりません: {model_path}。未学習のモデルを使用します。")
    return model.to(device)


class TrainingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    device = providers.Factory(get_auto_device)
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)

    snn_model = providers.Factory(
        SNNCore,
        config=config.model,
        vocab_size=providers.Callable(lambda t: len(t) if t else 50257, tokenizer.provided),
    )

    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn = providers.Factory(MetaCognitiveSNN, **(config.training.meta_cognition.as_dict() or {}))

    # === Losses ===
    standard_loss = providers.Factory(CombinedLoss, tokenizer=tokenizer, **(config.training.gradient_based.loss.as_dict() or {}))
    distillation_loss = providers.Factory(DistillationLoss, tokenizer=tokenizer, **(config.training.gradient_based.distillation.loss.as_dict() or {}))
    self_supervised_loss = providers.Factory(SelfSupervisedLoss, tokenizer=tokenizer, **(config.training.self_supervised.loss.as_dict() or {}))
    physics_informed_loss = providers.Factory(PhysicsInformedLoss, tokenizer=tokenizer, **(config.training.physics_informed.loss.as_dict() or {}))
    probabilistic_ensemble_loss = providers.Factory(ProbabilisticEnsembleLoss, tokenizer=tokenizer, **(config.training.probabilistic_ensemble.loss.as_dict() or {}))

    # === Trainers (with dedicated optimizers and schedulers) ===
    @providers.Factory
    def standard_trainer(self):
        optimizer = AdamW(self.snn_model().parameters(), lr=self.config.training.gradient_based.learning_rate())
        scheduler = _create_scheduler(optimizer, self.config.training.epochs(), self.config.training.gradient_based.warmup_epochs()) if self.config.training.gradient_based.use_scheduler() else None
        return BreakthroughTrainer(
            model=self.snn_model(),
            optimizer=optimizer,
            criterion=self.standard_loss(),
            scheduler=scheduler,
            device=self.device(),
            grad_clip_norm=self.config.training.gradient_based.grad_clip_norm(),
            rank=-1,
            use_amp=self.config.training.gradient_based.use_amp(),
            log_dir=self.config.training.log_dir(),
            astrocyte_network=self.astrocyte_network(),
            meta_cognitive_snn=self.meta_cognitive_snn()
        )

    @providers.Factory
    def distillation_trainer(self):
        optimizer = AdamW(self.snn_model().parameters(), lr=self.config.training.gradient_based.learning_rate())
        scheduler = _create_scheduler(optimizer, self.config.training.epochs(), self.config.training.gradient_based.warmup_epochs()) if self.config.training.gradient_based.use_scheduler() else None
        return DistillationTrainer(
            model=self.snn_model(),
            optimizer=optimizer,
            criterion=self.distillation_loss(),
            scheduler=scheduler,
            device=self.device(),
            grad_clip_norm=self.config.training.gradient_based.grad_clip_norm(),
            rank=-1,
            use_amp=self.config.training.gradient_based.use_amp(),
            log_dir=self.config.training.log_dir(),
            astrocyte_network=self.astrocyte_network(),
            meta_cognitive_snn=self.meta_cognitive_snn()
        )

    @providers.Factory
    def physics_informed_trainer(self):
        optimizer = AdamW(self.snn_model().parameters(), lr=self.config.training.physics_informed.learning_rate())
        scheduler = _create_scheduler(optimizer, self.config.training.epochs(), self.config.training.physics_informed.warmup_epochs()) if self.config.training.physics_informed.use_scheduler() else None
        return PhysicsInformedTrainer(
            model=self.snn_model(),
            optimizer=optimizer,
            criterion=self.physics_informed_loss(),
            scheduler=scheduler,
            device=self.device(),
            grad_clip_norm=self.config.training.physics_informed.grad_clip_norm(),
            rank=-1,
            use_amp=self.config.training.physics_informed.use_amp(),
            log_dir=self.config.training.log_dir(),
            astrocyte_network=self.astrocyte_network(),
            meta_cognitive_snn=self.meta_cognitive_snn()
        )

    # ... (rest of the trainers can be defined similarly if needed) ...

    bio_learning_rule = providers.Selector(config.training.biologically_plausible.learning_rule,
        STDP=providers.Factory(STDP, **(config.training.biologically_plausible.stdp.as_dict() or {})),
        REWARD_MODULATED_STDP=providers.Factory(RewardModulatedSTDP, **(config.training.biologically_plausible.reward_modulated_stdp.as_dict() or {})),
        CAUSAL_TRACE=providers.Factory(CausalTraceCreditAssignment, **(config.training.biologically_plausible.causal_trace.as_dict() or {}))
    )
    bio_snn_model = providers.Factory(BioSNN, layer_sizes=[10, 50, 2], neuron_params=config.training.biologically_plausible.neuron.as_dict(), learning_rule=bio_learning_rule, sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification.as_dict())
    rl_environment = providers.Factory(GridWorldEnv, device=device)
    rl_agent = providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=rl_agent, env=rl_environment)
    particle_filter_trainer = providers.Factory(ParticleFilterTrainer, base_model=bio_snn_model, config=config.training.biologically_plausible.particle_filter.as_dict(), device=device)

    planner_snn = providers.Factory(PlannerSNN, vocab_size=providers.Callable(lambda t: len(t) if t else 50257, tokenizer.provided), d_model=config.model.d_model, d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps, n_head=config.model.n_head, num_skills=10)
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)
    
    redis_client = providers.Singleton(redis.Redis, **(config.model_registry.redis.as_dict() or {}))
    model_registry = providers.Selector(config.model_registry.provider,
        file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path),
        distributed=providers.Singleton(DistributedModelRegistry, registry_path=config.model_registry.file.path)
    )

class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)

    device = providers.Factory(get_auto_device)
    model_registry = training_container.model_registry
    web_crawler = providers.Singleton(WebCrawler)
    rag_system = providers.Factory(RAGSystem, vector_store_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "vector_store") if log_dir else "runs/vector_store", log_dir=config.training.log_dir))
    memory = providers.Factory(Memory, memory_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "agent_memory.jsonl") if log_dir else "runs/agent_memory.jsonl", log_dir=config.training.log_dir))
    
    loaded_planner_snn = providers.Singleton(_load_planner_snn_factory, planner_snn_instance=training_container.planner_snn, model_path=config.training.planner.model_path, device=device)
    hierarchical_planner = providers.Factory(HierarchicalPlanner, model_registry=model_registry, rag_system=rag_system, planner_model=loaded_planner_snn, tokenizer_name=config.data.tokenizer_name, device=device)

class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)
    agent_container = providers.Container(AgentContainer, config=config)

    snn_inference_engine = providers.Factory(SNNInferenceEngine, config=config)
    chat_service = providers.Factory(ChatService, snn_engine=snn_inference_engine, max_len=config.app.max_len)
    langchain_adapter = providers.Factory(SNNLangChainAdapter, snn_engine=snn_inference_engine)

class BrainContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    agent_container = providers.Container(AgentContainer, config=config)

    num_neurons = providers.Factory(lambda: 256)
    sensory_receptor = providers.Singleton(SensoryReceptor)
    spike_encoder = providers.Singleton(SpikeEncoder, num_neurons=num_neurons)
    actuator = providers.Singleton(Actuator, actuator_name="voice_synthesizer")
    perception_cortex = providers.Singleton(HybridPerceptionCortex, num_neurons=num_neurons, feature_dim=64, som_map_size=(8, 8), stdp_params=config.training.biologically_plausible.stdp.as_dict())
    prefrontal_cortex = providers.Singleton(PrefrontalCortex)
    hierarchical_planner = agent_container.hierarchical_planner
    hippocampus = providers.Singleton(Hippocampus, capacity=50)
    cortex = providers.Singleton(Cortex)
    amygdala = providers.Singleton(Amygdala)
    basal_ganglia = providers.Singleton(BasalGanglia)
    cerebellum = providers.Singleton(Cerebellum)
    motor_cortex = providers.Singleton(MotorCortex, actuators=['voice_synthesizer'])

    artificial_brain = providers.Singleton(
        ArtificialBrain,
        sensory_receptor=sensory_receptor,
        spike_encoder=spike_encoder,
        actuator=actuator,
        perception_cortex=perception_cortex,
        prefrontal_cortex=prefrontal_cortex,
        hierarchical_planner=hierarchical_planner,
        hippocampus=hippocampus,
        cortex=cortex,
        amygdala=amygdala,
        basal_ganglia=basal_ganglia,
        cerebellum=cerebellum,
        motor_cortex=motor_cortex
    )
