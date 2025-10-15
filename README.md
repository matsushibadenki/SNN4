# **Project SNN: 自己進化するデジタル生命体の構築**

## **1\. 思想：予測する存在としてのAI**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とし、**自律的デジタル生命体 (Autonomous Digital Life Form)** の創造を目指す、次世代のAI研究開発フレームワークです。

我々のビジョンは、静的なパターン認識の限界を超え、世界の動的なモデルを内的に構築することにあります。その根本原理は**予測符号化**、すなわち\*\*「未来を予測し、その予測誤差を最小化する」\*\*という自己の存在理由に基づき、自律的に思考し、学習し、さらには自らのアーキテクチャや学習戦略さえも進化させるAIの実現です。

これは単なるチャットボットではありません。脳に着想を得た認知アーキテクチャを通じて、より汎用的で、エネルギー効率が高く、そして真に自律的な知性の形を追求する試みです。

## **2\. 主な特徴**

* **🧠 脳型認知アーキテクチャ:** 知覚、記憶、情動、意思決定、行動までの一貫した認知サイクルをシミュレートする「人工脳」を実装しています。  
* **🚀 最先端SNNモデル群:** Spiking Transformer, Spiking Mamba, Hybrid CNN-SNNなど、複数の先進的SNNアーキテクチャを実装。タスクに応じて最適なモデルを選択・生成します。  
* **📚 オンデマンド学習と知識蒸留:** 未知のタスクに直面した際、Web検索や大規模言語モデルからの知識蒸留により、タスク特化型の超省エネルギーな「専門家SNN」を自律的に生成します。  
* **🧬 自己進化するアーキテクチャ:** 自身の性能をメタ認知的に評価し、モデルの層数や次元数、さらには**学習パラダイム自体**をも自律的に修正し、より強力なアーキテクチャへと進化します。  
* **📊 ANN vs SNN 統合ベンチマーク:** 統一された環境でANNとSNNの性能（精度、速度、エネルギー効率）を直接比較し、レポートを自動生成するベンチマークスイートを搭載しています。  
* **🔧 統合CLIツール (snn-cli.py):** 学習、推論、自己進化、人工脳シミュレーションまで、プロジェクトの全機能を単一のインターフェースから制御可能です。

## **3\. システムアーキテクチャ**

本システムは、ユーザーのコマンドを起点に、エージェント層が認知・実行層をオーケストレーションする階層構造になっています。

```mermaid
graph TD  
    subgraph UI["User Interface"]
        CLI["snn-cli.py 統合CLIツール"]
    end

    subgraph OL["Orchestration Layer エージェント層"]
        LifeForm["DigitalLifeForm デジタル生命体"]
        Autonomous["AutonomousAgent タスク実行・Web学習"]
        Evolving["SelfEvolvingAgent 自己進化"]
        RL["ReinforcementLearnerAgent 強化学習"]
    end

    subgraph CL["Cognitive Layer 高次認知層"]
        Planner["HierarchicalPlanner 階層プランナー"]
        Brain["ArtificialBrain 人工脳シミュレータ"]
        Memory["Memory and RAG 長期記憶・知識検索"]
    end

    subgraph EL["Execution Layer 実行層"]
        Training["train.py 学習パイプライン"]
        Inference["SNNInferenceEngine 推論エンジン"]
        Benchmark["Benchmark Suite 性能評価スイート"]
    end

    subgraph FL["Foundation Layer 基盤層"]
        Core["SNN_Core Transformer, Mamba, Hybrid等"]
        Rules["BioLearningRules STDP, 因果追跡"]
    end

    CLI -->|"solve"| Autonomous
    CLI -->|"evolve"| Evolving
    CLI -->|"rl"| RL
    CLI -->|"life-form"| LifeForm
    CLI -->|"planner"| Planner
    CLI -->|"brain"| Brain
    CLI -->|"gradient-train"| Training
    CLI -->|"benchmark"| Benchmark

    LifeForm -->|"指示"| Autonomous
    LifeForm -->|"指示"| Evolving
    LifeForm -->|"指示"| RL

    Autonomous -->|"計画を要求"| Planner
    Autonomous -->|"学習/推論を実行"| Training
    Autonomous -->|"学習/推論を実行"| Inference

    Planner -->|"記憶を検索"| Memory
    Training -->|"モデルを構築"| Core
    Inference -->|"モデルを利用"| Core
    Training -->|"学習則を利用"| Rules
```


## **4\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なPythonライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: システム健全性チェック（推奨）**

プロジェクト全体のユニットテストおよび統合テストを実行し、すべてのコンポーネントが正しく動作することを確認します。

pytest \-v

### **ステップ3: ANN vs SNN 性能比較ベンチマークの実行**

**目的:** snn\_4\_ann\_parity\_plan.mdに基づき、標準的な画像分類タスク（CIFAR-10）でANNとSNNの性能（精度、速度、エネルギー効率）を直接比較します。

python scripts/run\_benchmark\_suite.py \--experiment cifar10\_comparison \--epochs 5

実験完了後、結果はbenchmarks/cifar10\_ann\_vs\_snn\_report.mdにMarkdown形式のレポートとして自動的に保存されます。このレポートが、ANN性能パリティに向けたプロジェクトの進捗を示す重要な指標となります。

### **ステップ4: 本格実行 \- 大規模学習と対話**

**目的:** 大規模データセットでAIを本格的に学習させ、意味のある応答を生成できるようにします。

#### **4-1: 大規模データセットの準備（初回のみ）**

wikitext-103（100万行以上のテキスト）をダウンロードし、学習用に整形します。

python scripts/data\_preparation.py

#### **4-2: 本格的な学習の実行**

**A) 推奨：最強エンジン（Ultraモデル）の学習**

以下のコマンド一つで、データ準備から最大規模のSpiking Transformerモデルの学習までを自動的に実行します。

python snn-cli.py train-ultra \--override\_config "training.epochs=50"

**B) 通常モデルの学習**

特定のモデル構成で学習させたい場合は、agent solveコマンドを使用します。

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

**Note:** これらの学習はマシンスペックにより数時間以上かかる可能性があります。

#### **4-3: 学習済みモデルとの対話**

学習済みのモデルを呼び出して対話します。--model\_configで使用したいモデルの設定ファイルを指定してください。

\# Ultraモデルとの対話  
python snn-cli.py ui start \--model\_config configs/models/ultra.yaml

\# もしくは、特定のタスクを実行  
python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

### **ステップ5: 高度な機能の探求**

その他の高度な機能（Webからの自律学習、自己進化、人工脳シミュレーションなど）については、doc/SNN開発：プロジェクト機能テスト コマンド一覧.mdをご参照ください。

## **5\. プロジェクト構造**

snn4/  
├── app/                  \# UIアプリケーションとDIコンテナ  
├── benchmarks/           \# (自動生成) ベンチマーク結果レポート  
├── configs/              \# 設定ファイル (base, models/\*.yaml)  
├── data/                 \# 学習用データセット  
├── doc/                  \# ドキュメント  
├── runs/                 \# (自動生成) 学習ログ、チェックポイント、モデル登録簿  
├── scripts/              \# データ準備やベンチマークなどの補助スクリプト  
├── snn\_research/         \# SNNコア研究開発コード  
│   ├── agent/            \# 各種エージェント (自律、自己進化、生命体、強化学習)  
│   ├── benchmark/        \# SNN vs ANN 性能評価タスク定義  
│   ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、人工脳など)  
│   ├── communication/    \# エージェント間通信  
│   ├── conversion/       \# ANN-SNNモデル変換  
│   ├── core/             \# SNNモデル (BreakthroughSNN, SpikingTransformer, Hybrid)  
│   ├── data/             \# データセット定義  
│   ├── deployment.py     \# 推論エンジン  
│   ├── distillation/     \# 知識蒸留とモデル登録簿  
│   ├── hardware/         \# ニューロモーフィックハードウェア関連  
│   ├── io/               \# 感覚入力・運動出力  
│   ├── learning\_rules/   \# 生物学的学習則 (STDPなど)  
│   ├── models/           \# (旧) モデルアーキテクチャ  
│   ├── rl\_env/           \# 強化学習環境  
│   ├── tools/            \# 外部ツール (Webクローラーなど)  
│   └── training/         \# Trainer、損失関数、量子化、プルーニング  
├── tests/                \# テストコード  
├── snn-cli.py            \# ✨ 統合CLIツール  
├── train.py              \# 勾配ベース学習の実行スクリプト (CLIから呼び出される)  
└── requirements.txt      \# 必要なライブラリ  
