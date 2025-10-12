# **Project SNN: A Predictive Digital Life Form (v6.0)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とし、**自律的デジタル生命体 (Autonomous Digital Life Form)** の創造を目指す研究開発フレームワークです。

最終目標は、静的なパターン認識の限界を超え、世界の動的なモデルを内的に構築し、**未来を予測し、その予測誤差を最小化する**という自己の存在理由に基づき、自律的に思考し、学習し、自己を改良するAIを実現することです。

このシステムは、単なるチャットボットではなく、以下の高度な認知能力を備えた自律エージェントとして動作します。

* **先進的なSNNアーキテクチャ:** 脳の働きに着想を得た**予測符号化モデル**、時間情報処理能力を最大化する**スパイキングトランスフォーマー**、長文脈処理に革命をもたらす**スパイキングMAMBA**に加え、脳の情報処理階層を模倣する\*\*階層的循環記憶（Spiking-HRM）\*\*モデルを導入。これにより、短期的な情報と長期的な文脈を統合し、より深い未来予測を実現します。  
* **オンデマンド学習:** 未知のタスクに直面した際、Web検索でリアルタイムに情報を収集したり、大規模言語モデルから知識を蒸留したりして、タスクに特化した超省エネルギーな「専門家SNN」を自律的に生成します。  
* **自己認識と計画立案:** 自身の能力（学習済み専門家モデル）と記憶（過去の経験）を把握し、学習済みのプランナーSNNを用いて、複雑なタスクをサブタスクに分解し、最適な実行計画を動的に推論します。  
* **アーキテクチャレベルの自己進化:** 自らの性能を評価し、表現力不足と判断した場合、自身のソースコード（モデルの層数や次元数、さらには学習パラダイム自体）を自律的に修正し、より強力なアーキテクチャへと進化します。  
* **行動を通じた学習（強化学習）:** バックプロパゲーションを使わない生物学的な学習則（報酬変調型STDP）を用い、環境との試行錯誤から直接スキルを学習する能力を持ちます。

## **2\. システムアーキテクチャ**

本システムの認知アーキテクチャは、統合CLIツール snn-cli.py を起点とし、各コマンドが対応する認知機能を呼び出す形で構成されています。

```mermaid
graph TD  
    CLI["snn-cli.py<br/>統合CLIツール"]
    
    subgraph Orchestration["自律実行と思考のオーケストレーション"]
        LifeForm["DigitalLifeForm<br/>(デジタル生命体)"]  
        Motivation["内発的動機システム"]  
        MetaCognition["メタ認知SNN"]  
        Agent_Autonomous["AutonomousAgent<br/>(タスク実行・Web学習)"]
        Agent_SelfEvolving["SelfEvolvingAgent<br/>(自己進化)"]
        Agent_RL["ReinforcementLearnerAgent<br/>(強化学習)"]
        
        LifeForm -->|"思考・行動を<br/>オーケストレーション"| Agent_Autonomous
        LifeForm -->|"思考・行動を<br/>オーケストレーション"| Agent_SelfEvolving
        LifeForm -->|"思考・行動を<br/>オーケストレーション"| Agent_RL
        LifeForm -->|"内部状態を監視"| Motivation
        LifeForm -->|"パフォーマンスを評価"| MetaCognition
    end
    
    subgraph Cognition["高次認知機能"]
        Planner["HierarchicalPlanner<br/>(階層プランナー)"]  
        Memory["Memory<br/>(RAG連携 長期記憶)"]  
        RAG["RAGSystem<br/>(ベクトル検索)"]
        
        Planner -->|"記憶を検索"| Memory
        Memory -->|"ベクトル検索を利用"| RAG
    end
    
    subgraph Engine["中核エンジン：学習と推論"]
        TrainPy["train.py<br/>(学習パイプライン)"]  
        SNN_Core["SNN_Core<br/>(Transformer, Mamba, HRM等)"]  
        BioRules["生物学的学習則<br/>(STDP, 因果追跡)"]  
        InferenceEngine["SNNInferenceEngine<br/>(推論エンジン)"]
        
        TrainPy -->|"モデルを訓練"| SNN_Core
        TrainPy -->|"学習則を利用"| BioRules
        InferenceEngine -->|"モデルをロード"| SNN_Core
    end
    
    CLI -->|"life-form start"| LifeForm
    CLI -->|"agent solve"| Agent_Autonomous
    CLI -->|"evolve run"| Agent_SelfEvolving
    CLI -->|"rl run"| Agent_RL
    CLI -->|"planner execute"| Planner
    CLI -->|"gradient-train"| TrainPy
    
    classDef cli fill:#b39ddb,stroke:#333,stroke-width:3px,color:#fff
    classDef orchestrator fill:#cde4ff,stroke:#333,stroke-width:2px
    classDef agent fill:#d4edda,stroke:#333,stroke-width:1px
    classDef cognition fill:#ffe4c4,stroke:#333,stroke-width:1px
    classDef engine fill:#f8d7da,stroke:#333,stroke-width:1px
    
    class CLI cli
    class LifeForm,Motivation,MetaCognition orchestrator
    class Agent_Autonomous,Agent_SelfEvolving,Agent_RL agent
    class Planner,Memory,RAG cognition
    class TrainPy,SNN_Core,BioRules,InferenceEngine engine
```


## **3\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なPythonライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: システム健全性チェック（推奨）**

プロジェクト全体のユニットテストおよび統合テストを実行し、すべてのコンポーネントが正しく動作することを確認します。

pytest \-v

### **ステップ3: クイックスタート \- オンデマンド学習と推論（数分で完了）**

**目的:** システム全体（オンデマンド学習〜推論）のパイプラインが正常に完了することを確認します。

python snn-cli.py agent solve \\  
    \--task "高速テスト" \\  
    \--prompt "これはテストです。" \\  
    \--unlabeled-data data/sample\_data.jsonl \\  
    \--force-retrain

**Note:** このテストはあくまでシステムの動作確認用です。**使用するデータが非常に小さいため、AIは意味のある応答を生成できません。** 応答が空になるのは、学習データ不足による正常な動作です。

### **ステップ4: 本格実行 \- 大規模学習と対話**

**目的:** 大規模データセットでAIを本格的に学習させ、意味のある応答を生成できるようにします。

#### **ステップ4-1: 大規模データセットの準備（初回のみ）**

wikitext-103（100万行以上のテキスト）をダウンロードし、学習用に整形します。

python scripts/data\_preparation.py

#### **ステップ4-2: 本格的な学習の実行**

準備した大規模データセットを使い、汎用的な言語能力を持つ専門家モデル「汎用言語モデル」を育成します。

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

**Note:** この学習はマシンスペックにより数時間以上かかる可能性があります。

#### **ステップ4-3: 学習済みモデルとの対話**

上記の本格的な学習で育成した「汎用言語モデル」を呼び出して対話します。

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

### **ステップ5: その他の高度な機能**

その他の高度な機能（Webからの自律学習、自己進化、強化学習など）については、doc/SNN開発：プロジェクト機能テスト コマンド一覧.md をご参照ください。

## **4\. プロジェクト構造**

snn4/  
├── app/                  \# UIアプリケーションとDIコンテナ  
├── configs/              \# 設定ファイル (base, models/\*.yaml)  
├── data/                 \# 学習用データセット  
├── doc/                  \# ドキュメント  
├── runs/                 \# (自動生成) 学習ログ、チェックポイント、モデル登録簿  
├── scripts/              \# データ準備やベンチマークなどの補助スクリプト  
├── snn\_research/         \# SNNコア研究開発コード  
│   ├── agent/            \# 各種エージェント (自律、自己進化、生命体、強化学習)  
│   ├── benchmark/        \# SNN vs ANN 性能評価  
│   ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、人工脳など)  
│   ├── communication/    \# エージェント間通信  
│   ├── conversion/       \# ANN-SNNモデル変換  
│   ├── core/             \# SNNモデル (BreakthroughSNN, SpikingTransformer, Mamba, HRM)  
│   ├── data/             \# データセット定義  
│   ├── deployment.py     \# 推論エンジン  
│   ├── distillation/     \# 知識蒸留とモデル登録簿  
│   ├── hardware/         \# ニューロモーフィックハードウェア関連  
│   ├── io/               \# 感覚入力・運動出力  
│   ├── learning\_rules/   \# 生物学的学習則 (STDPなど)  
│   ├── metrics/          \# 評価指標  
│   ├── models/           \# (旧) モデルアーキテクチャ  
│   ├── rl\_env/           \# 強化学習環境  
│   ├── tools/            \# 外部ツール (Webクローラーなど)  
│   └── training/         \# Trainerと損失関数  
├── tests/                \# テストコード  
├── snn-cli.py            \# ✨ 統合CLIツール  
├── train.py              \# 勾配ベース学習の実行スクリプト (CLIから呼び出される)  
├── run\_\*.py              \# 各機能の実行スクリプト  
└── requirements.txt      \# 必要なライブラリ  
