# ファイルパス: app/containers.py
# (修正 v13 - 最終修正)
# 根本修正: pytest収集時に発生するすべての `TypeError` 及び `AttributeError` を
#           完全に解消するため、DIコンテナの定義方法を全面的に見直し、
#           ライブラリのベストプラクティスに沿った形に修正。
#           すべての設定値アクセスをオブジェクト生成時まで遅延させる記述に統一した。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer
import os
import redis
from typing import TYPE_CHECKING

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
from app.services.chat_service import ChatService
from app.adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry
from snn_research.tools.web_crawler import WebCrawler
from snn_research.learning_rules import get_bio_learning_rule
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
    from app.adapters.snn_langchain_adapter import SNNLangChainAdapter


def get_auto_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _create_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LRScheduler:
    if epochs <= warmup_epochs: warmup_epochs = 0
    warmup_scheduler = LinearLR(optimizer=optimizer, start_factor=1e-3, total_iters=warmup_epochs)
    main_scheduler_t_max = max(1, epochs - warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=main_scheduler_t_max)
    return SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

def _load_planner_snn_factory(planner_snn_instance, model_path: str, device: str):
    model = planner_snn_instance
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
        except Exception: pass
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
    meta_cognitive_snn = providers.Factory(MetaCognitiveSNN)

    # === Losses (using delayed config access) ===
    standard_loss = providers.Factory(
        CombinedLoss, tokenizer=tokenizer, 
        ce_weight=config.training.gradient_based.loss.ce_weight,
        spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight,
        mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight,
    )
    distillation_loss = providers.Factory(
        DistillationLoss, tokenizer=tokenizer,
        ce_weight=config.training.gradient_based.distillation.loss.ce_weight,
        distill_weight=config.training.gradient_based.distillation.loss.distill_weight,
        spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight,
        mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight,
        temperature=config.training.gradient_based.distillation.loss.temperature,
    )
    physics_informed_loss = providers.Factory(
        PhysicsInformedLoss, tokenizer=tokenizer,
        ce_weight=config.training.physics_informed.loss.ce_weight,
        spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight,
        mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight,
    )

    # === Trainers (with dedicated optimizers and schedulers) ===
    grad_optimizer = providers.Factory(AdamW, params=snn_model.provided.parameters.call(), lr=config.training.gradient_based.learning_rate)
    grad_scheduler = providers.Factory(_create_scheduler, optimizer=grad_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    
    standard_trainer = providers.Factory(
        BreakthroughTrainer, model=snn_model, optimizer=grad_optimizer, criterion=standard_loss, scheduler=grad_scheduler, device=device,
        grad_clip_norm=config.training.gradient_based.grad_clip_norm, rank=-1, use_amp=config.training.gradient_based.use_amp,
        log_dir=config.training.log_dir, astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn
    )
    distillation_trainer = providers.Factory(
        DistillationTrainer, model=snn_model, optimizer=grad_optimizer, criterion=distillation_loss, scheduler=grad_scheduler, device=device,
        grad_clip_norm=config.training.gradient_based.grad_clip_norm, rank=-1, use_amp=config.training.gradient_based.use_amp,
        log_dir=config.training.log_dir, astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn
    )

    pi_optimizer = providers.Factory(AdamW, params=snn_model.provided.parameters.call(), lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    
    physics_informed_trainer = providers.Factory(
        PhysicsInformedTrainer, model=snn_model, optimizer=pi_optimizer, criterion=physics_informed_loss, scheduler=pi_scheduler, device=device,
        grad_clip_norm=config.training.physics_informed.grad_clip_norm, rank=-1, use_amp=config.training.physics_informed.use_amp,
        log_dir=config.training.log_dir, astrocyte_network=astrocyte_network, meta_cognitive_snn=meta_cognitive_snn
    )

    # === Bio Learning ===
    bio_learning_rule = providers.Factory(
        get_bio_learning_rule,
        name=config.training.biologically_plausible.learning_rule,
        params=config.training.biologically_plausible,
    )
    bio_snn_model = providers.Factory(
        BioSNN, layer_sizes=[10, 50, 2],
        neuron_params=config.training.biologically_plausible.neuron,
        learning_rule=bio_learning_rule,
        sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification
    )
    particle_filter_trainer = providers.Factory(
        ParticleFilterTrainer, base_model=bio_snn_model,
        config=config.training.biologically_plausible.particle_filter,
        device=device
    )
    rl_agent = providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device)
    rl_environment = providers.Factory(GridWorldEnv, device=device)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=rl_agent, env=rl_environment)
    
    # === Etc ===
    model_registry = providers.Selector(
        config.model_registry.provider,
        file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path),
    )

class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)
    device = providers.Factory(get_auto_device)
    web_crawler = providers.Singleton(WebCrawler)
    rag_system = providers.Factory(RAGSystem, vector_store_path=providers.Callable(os.path.join, config.training.log_dir, "vector_store"))
    memory = providers.Factory(Memory, memory_path=providers.Callable(os.path.join, config.training.log_dir, "agent_memory.jsonl"))
    
    planner_snn_unloaded = providers.Factory(
        PlannerSNN, vocab_size=providers.Callable(len, training_container.tokenizer),
        d_model=config.model.d_model, d_state=config.model.d_state, num_layers=config.model.num_layers,
        time_steps=config.model.time_steps, n_head=config.model.n_head, num_skills=10
    )
    loaded_planner_snn = providers.Singleton(
        _load_planner_snn_factory,
        planner_snn_instance=planner_snn_unloaded,
        model_path=config.training.planner.model_path,
        device=device,
    )
    hierarchical_planner = providers.Factory(
        HierarchicalPlanner, model_registry=training_container.model_registry, rag_system=rag_system,
        planner_model=loaded_planner_snn, tokenizer_name=config.data.tokenizer_name, device=device
    )

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
    perception_cortex = providers.Singleton(
        HybridPerceptionCortex, num_neurons=num_neurons, feature_dim=64, som_map_size=(8, 8),
        stdp_params=config.training.biologically_plausible.stdp,
    )
    prefrontal_cortex = providers.Singleton(PrefrontalCortex)
    hippocampus = providers.Singleton(Hippocampus, capacity=50)
    cortex = providers.Singleton(Cortex)
    amygdala = providers.Singleton(Amygdala)
    basal_ganglia = providers.Singleton(BasalGanglia)
    cerebellum = providers.Singleton(Cerebellum)
    motor_cortex = providers.Singleton(MotorCortex, actuators=['voice_synthesizer'])

    artificial_brain = providers.Singleton(
        ArtificialBrain,
        sensory_receptor=sensory_receptor, spike_encoder=spike_encoder, actuator=actuator,
        perception_cortex=perception_cortex, prefrontal_cortex=prefrontal_cortex,
        hierarchical_planner=agent_container.hierarchical_planner,
        hippocampus=hippocampus, cortex=cortex, amygdala=amygdala,
        basal_ganglia=basal_ganglia, cerebellum=cerebellum, motor_cortex=motor_cortex
    )
