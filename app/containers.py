# ファイルパス: app/containers.py
# (修正)
# - mypyエラー[no-redef]を解消するため、重複していたBrainContainerの定義を統合。
# - mypyエラー[name-defined]を解消するため、コンテナ間の参照を修正。

import torch
from dependency_injector import containers, providers
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from transformers import AutoTokenizer
import os
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
    PhysicsInformedTrainer, ProbabilisticEnsembleTrainer, ParticleFilterTrainer,
    PlannerTrainer
)
from snn_research.training.bio_trainer import BioRLTrainer
from snn_research.cognitive_architecture.astrocyte_network import AstrocyteNetwork
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.planner_snn import PlannerSNN
from .services.chat_service import ChatService
from .adapters.snn_langchain_adapter import SNNLangChainAdapter
from snn_research.distillation.model_registry import SimpleModelRegistry, DistributedModelRegistry
from snn_research.tools.web_crawler import WebCrawler

from snn_research.learning_rules.causal_trace import CausalTraceCreditAssignment
from snn_research.bio_models.simple_network import BioSNN
from snn_research.rl_env.grid_world import GridWorldEnv
from snn_research.agent.reinforcement_learner_agent import ReinforcementLearnerAgent

from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.agent.memory import Memory
from snn_research.cognitive_architecture.causal_inference_engine import CausalInferenceEngine
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.hybrid_perception_cortex import HybridPerceptionCortex
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace

from snn_research.benchmark import TASK_REGISTRY
from .utils import get_auto_device

from snn_research.agent.digital_life_form import DigitalLifeForm
from snn_research.agent.autonomous_agent import AutonomousAgent
from snn_research.agent.self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator
from snn_research.cognitive_architecture.symbol_grounding import SymbolGrounding

from snn_research.learning_rules import ProbabilisticHebbian, get_bio_learning_rule
from snn_research.core.neurons import ProbabilisticLIFNeuron
from snn_research.training.bio_trainer import BioRLTrainer

if TYPE_CHECKING:
    from .adapters.snn_langchain_adapter import SNNLangChainAdapter


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
    task_registry = providers.Object(TASK_REGISTRY)
    device = providers.Factory(get_auto_device)
    tokenizer = providers.Factory(AutoTokenizer.from_pretrained, pretrained_model_name_or_path=config.data.tokenizer_name)
    snn_model = providers.Factory(SNNCore, config=config.model, vocab_size=tokenizer.provided.vocab_size)
    astrocyte_network = providers.Factory(AstrocyteNetwork, snn_model=snn_model)
    meta_cognitive_snn = providers.Factory(MetaCognitiveSNN, **(config.training.meta_cognition.to_dict() or {}))
    optimizer = providers.Factory(AdamW, lr=config.training.gradient_based.learning_rate)
    scheduler = providers.Factory(_create_scheduler, optimizer=optimizer, epochs=config.training.epochs, warmup_epochs=config.training.gradient_based.warmup_epochs)
    standard_trainer = providers.Factory(BreakthroughTrainer, criterion=providers.Factory(CombinedLoss, ce_weight=config.training.gradient_based.loss.ce_weight, spike_reg_weight=config.training.gradient_based.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.loss.sparsity_reg_weight, tokenizer=tokenizer, ewc_weight=config.training.gradient_based.loss.ewc_weight), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    distillation_trainer = providers.Factory(DistillationTrainer, criterion=providers.Factory(DistillationLoss, tokenizer=tokenizer, ce_weight=config.training.gradient_based.distillation.loss.ce_weight, distill_weight=config.training.gradient_based.distillation.loss.distill_weight, spike_reg_weight=config.training.gradient_based.distillation.loss.spike_reg_weight, mem_reg_weight=config.training.gradient_based.distillation.loss.mem_reg_weight, sparsity_reg_weight=config.training.gradient_based.distillation.loss.sparsity_reg_weight, temperature=config.training.gradient_based.distillation.loss.temperature), grad_clip_norm=config.training.gradient_based.grad_clip_norm, use_amp=config.training.gradient_based.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    pi_optimizer = providers.Factory(AdamW, lr=config.training.physics_informed.learning_rate)
    pi_scheduler = providers.Factory(_create_scheduler, optimizer=pi_optimizer, epochs=config.training.epochs, warmup_epochs=config.training.physics_informed.warmup_epochs)
    physics_informed_trainer = providers.Factory(PhysicsInformedTrainer, criterion=providers.Factory(PhysicsInformedLoss, ce_weight=config.training.physics_informed.loss.ce_weight, spike_reg_weight=config.training.physics_informed.loss.spike_reg_weight, mem_smoothness_weight=config.training.physics_informed.loss.mem_smoothness_weight, tokenizer=tokenizer), grad_clip_norm=config.training.physics_informed.grad_clip_norm, use_amp=config.training.physics_informed.use_amp, log_dir=config.training.log_dir, meta_cognitive_snn=meta_cognitive_snn)
    bio_rl_trainer = providers.Factory(BioRLTrainer, agent=providers.Factory(ReinforcementLearnerAgent, input_size=4, output_size=4, device=device), env=providers.Factory(GridWorldEnv, device=device))
    particle_filter_trainer = providers.Factory(ParticleFilterTrainer, base_model=providers.Factory(BioSNN, layer_sizes=[10, 5, 2], neuron_params={'tau_mem': 10.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'v_rest': 0.0}, learning_rule=providers.Object(None), sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification), config=config, device=device)
    planner_snn = providers.Factory(PlannerSNN, vocab_size=providers.Callable(len, tokenizer), d_model=config.model.d_model, d_state=config.model.d_state, num_layers=config.model.num_layers, time_steps=config.model.time_steps, n_head=config.model.n_head, num_skills=10, neuron_config=config.model.neuron)
    planner_optimizer = providers.Factory(AdamW, lr=config.training.planner.learning_rate)
    planner_loss = providers.Factory(PlannerLoss)
    model_registry = providers.Selector(providers.Callable(lambda cfg: cfg.get("model_registry", {}).get("provider"), config.provided), file=providers.Singleton(SimpleModelRegistry, registry_path=config.model_registry.file.path), distributed=providers.Singleton(DistributedModelRegistry, registry_path=config.model_registry.file.path))


class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    training_container = providers.Container(TrainingContainer, config=config)
    device = providers.Factory(get_auto_device)
    model_registry = training_container.model_registry
    web_crawler = providers.Singleton(WebCrawler)
    rag_system = providers.Factory(RAGSystem, vector_store_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "vector_store") if log_dir else "runs/vector_store", log_dir=config.training.log_dir))
    memory = providers.Factory(Memory, rag_system=rag_system, memory_path=providers.Callable(lambda log_dir: os.path.join(log_dir, "agent_memory.jsonl") if log_dir else "runs/agent_memory.jsonl", log_dir=config.training.log_dir))
    loaded_planner_snn = providers.Singleton(_load_planner_snn_factory, planner_snn_instance=training_container.planner_snn, model_path=config.training.planner.model_path, device=device)
    hierarchical_planner = providers.Factory(HierarchicalPlanner, model_registry=model_registry, rag_system=rag_system, memory=memory, planner_model=loaded_planner_snn, tokenizer_name=config.data.tokenizer_name, device=device)


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
    app_container = providers.Container(AppContainer, config=config)

    global_workspace = providers.Singleton(GlobalWorkspace, model_registry=agent_container.model_registry)
    motivation_system = providers.Singleton(IntrinsicMotivationSystem)
    
    num_neurons = providers.Factory(lambda: 256)
    sensory_receptor = providers.Singleton(SensoryReceptor)
    spike_encoder = providers.Singleton(SpikeEncoder, num_neurons=num_neurons)
    actuator = providers.Singleton(Actuator, actuator_name="voice_synthesizer")

    perception_cortex = providers.Singleton(HybridPerceptionCortex, workspace=global_workspace, num_neurons=num_neurons, feature_dim=64, som_map_size=(8, 8), stdp_params=config.training.biologically_plausible.stdp.to_dict())
    prefrontal_cortex = providers.Singleton(PrefrontalCortex, workspace=global_workspace, motivation_system=motivation_system)
    
    hippocampus = providers.Singleton(Hippocampus, workspace=global_workspace, capacity=50)
    cortex = providers.Singleton(Cortex)

    amygdala = providers.Singleton(Amygdala, workspace=global_workspace)
    basal_ganglia = providers.Singleton(BasalGanglia, workspace=global_workspace)

    cerebellum = providers.Singleton(Cerebellum)
    motor_cortex = providers.Singleton(MotorCortex, actuators=['voice_synthesizer'])

    causal_inference_engine = providers.Singleton(CausalInferenceEngine, rag_system=agent_container.rag_system, workspace=global_workspace)

    artificial_brain = providers.Singleton(
        ArtificialBrain,
        global_workspace=global_workspace,
        motivation_system=motivation_system,
        sensory_receptor=sensory_receptor,
        spike_encoder=spike_encoder,
        actuator=actuator,
        perception_cortex=perception_cortex,
        prefrontal_cortex=prefrontal_cortex,
        hippocampus=hippocampus,
        cortex=cortex,
        amygdala=amygdala,
        basal_ganglia=basal_ganglia,
        cerebellum=cerebellum,
        motor_cortex=motor_cortex,
        causal_inference_engine=causal_inference_engine
    )
    
    autonomous_agent = providers.Singleton(
        AutonomousAgent,
        name="AutonomousAgent",
        planner=agent_container.hierarchical_planner,
        model_registry=agent_container.model_registry,
        memory=agent_container.memory,
        web_crawler=agent_container.web_crawler
    )

    rl_agent = providers.Singleton(
        ReinforcementLearnerAgent,
        input_size=4,
        output_size=4,
        device=agent_container.device
    )

    self_evolving_agent = providers.Singleton(
        SelfEvolvingAgent,
        name="SelfEvolvingAgent",
        planner=agent_container.hierarchical_planner,
        model_registry=agent_container.model_registry,
        memory=agent_container.memory,
        web_crawler=agent_container.web_crawler,
        model_config_path="configs/models/small.yaml",
        training_config_path="configs/base_config.yaml"
    )

    digital_life_form = providers.Singleton(
        DigitalLifeForm,
        planner=agent_container.hierarchical_planner,
        autonomous_agent=autonomous_agent,
        rl_agent=rl_agent,
        self_evolving_agent=self_evolving_agent,
        motivation_system=motivation_system,
        meta_cognitive_snn=providers.Singleton(MetaCognitiveSNN),
        memory=agent_container.memory,
        physics_evaluator=providers.Singleton(PhysicsEvaluator),
        symbol_grounding=providers.Singleton(SymbolGrounding, rag_system=agent_container.rag_system),
        langchain_adapter=app_container.langchain_adapter,
        global_workspace=global_workspace
    )
    
# --- 確率的ヘブ学習用のコンポーネント ---
    probabilistic_neuron = providers.Factory(
        ProbabilisticLIFNeuron,
        neuron_params=config.training.biologically_plausible.probabilistic_neuron # 新しい設定パス
    )
    probabilistic_learning_rule = providers.Factory(
        ProbabilisticHebbian,
        **config.training.biologically_plausible.probabilistic_hebbian.to_dict() # 新しい設定パス
    )
    probabilistic_model = providers.Factory(
        BioSNN,
        layer_sizes=[10, 5, 2], # ダミーのサイズ, 実際にはタスクに応じて設定
        neuron_params=config.training.biologically_plausible.probabilistic_neuron, # 新しい設定パス
        learning_rule=probabilistic_learning_rule,
        sparsification_config=config.training.biologically_plausible.adaptive_causal_sparsification
    )
    # 既存の BioRLTrainer を流用する例 (必要に応じて専用トレーナーを作成)
    # 注意: BioRLTrainer は強化学習環境を前提としているため、論文の教師なし学習とは異なる可能性がある
    probabilistic_trainer = providers.Factory(
        BioRLTrainer, # または BioProbabilisticTrainer (新規作成)
        agent=providers.Factory( # Agent も専用のものが必要になる可能性
            ReinforcementLearnerAgent, # または ProbabilisticHebbianAgent (新規作成)
            input_size=4, output_size=4, device=device,
            # model=probabilistic_model # Agent内でモデルを初期化するなら不要
        ),
        env=providers.Factory(GridWorldEnv, device=device) # 環境もダミー
    )
