# **SNN開発：機能別コマンドリファレンス v8.0**

## **1\. はじめに**

このドキュメントは、統合CLIツール snn-cli.py を用いてSNNプロジェクトの全機能を実行するためのリファレンスです。コマンドを**作業の目的**に基づき5つの主要なカテゴリーに分類し、各機能へのアクセスを明確化しました。

## **2\. 統合CLIツールの基本**

すべてのコマンドは python snn-cli.py \<カテゴリー\> \<アクション\> の形式で実行されます。各コマンドやサブコマンドの詳細なヘルプは \--help フラグで確認できます。

| 目的 | コマンド例 |
| :---- | :---- |
| **全体のヘルプ** | python snn-cli.py \--help |
| **カテゴリーのヘルプ** | python snn-cli.py agent \--help |

## **3\. 機能別コマンドリファレンス**

### **3.1. 🧩 システム基盤の管理**

プロジェクト環境の初期セットアップ、健全性確認、大規模なデータ準備に関するコマンドです。

| アクション | コマンド | 説明 |
| :---- | :---- | :---- |
| **健全性チェック** | pytest \-v | すべてのユニットテストおよび統合テストを実行し、システムの基盤的な動作を確認します。 |
| **煙テスト** | pytest \-v tests/test\_smoke\_all\_paradigms.py | 主要な学習パラダイム（勾配、物理、生物学）が最小データでエラーなく動作するかを検証します。 |
| **大規模データ準備** | python scripts/data\_preparation.py | WikiText-103のような大規模コーパスをダウンロードし、学習用に前処理します。 |
| **知識ベース構築** | python scripts/build\_knowledge\_base.py | RAGシステム用のベクトルストアを、プロジェクト内のドキュメントから構築します。 |

### **3.2. 🧠 モデルの学習と最適化**

SNNモデルの訓練、教師モデルからの知識転移、モデルの軽量化に関するコマンドです。

| コマンド | アクション | 説明 |
| :---- | :---- | :---- |
| gradient-train | **直接学習** | 勾配ベースの学習（代理勾配法）を実行します。設定ファイルで学習パラダイム、エポック数、モデル構成を指定します。 |
| train-ultra | **大規模学習** | データ準備から最大規模のUltraモデルの学習までを自動で実行するパイプラインです。 |
| train-planner | **プランナー訓練** | 階層的思考プランナー (PlannerSNN) をタスク分解データで訓練します。（個別スクリプト） |
| run\_distillation.py | **知識蒸留** | ANN教師モデルからの知識蒸留を行い、高性能なSNN専門家モデルを育成します。（個別スクリプト） |
| convert ann2snn-cnn | **CNN変換** | 学習済みCNN (ANN) の重みを読み込み、BatchNorm Foldingと閾値キャリブレーションを経てSpikingCNN (SNN) に変換します。 |

#### **コマンド事例**

**1\. 基本的な勾配ベース学習**

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/medium.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=5"

**2\. Spiking Transformerモデルでの学習**

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/spiking\_transformer.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=10" \\  
    \--override\_config "training.log\_dir=runs/transformer\_test"

**3\. Spiking Mambaモデルでの学習**

python snn-cli.py gradient-train \\  
    \--model\_config configs/models/spiking\_mamba.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=10" \\  
    \--override\_config "training.log\_dir=runs/mamba\_test"

**4\. ANNモデルからSNNモデルへの変換**

python snn-cli.py convert ann2snn-cnn \\  
    \--ann-model-path runs/ann\_cifar\_baseline/cifar10/best\_model.pth \\  
    \--snn-model-config configs/cifar10\_spikingcnn\_config.yaml \\  
    \--output-snn-path runs/converted/spiking\_cnn\_from\_ann.pth

**5\. 知識蒸留の実行（個別スクリプト）**

python run\_distillation.py \\  
    \--task cifar10 \\  
    \--teacher\_model resnet18 \\  
    \--model\_config configs/cifar10\_spikingcnn\_config.yaml

### **3.3. 📊 性能評価と効率検証**

ANNとの直接比較、継続学習の検証、ハードウェア性能のシミュレーションに関するコマンドです。

| コマンド | アクション | 説明 |
| :---- | :---- | :---- |
| benchmark run | **ANN vs SNN比較** | CIFAR-10、SST-2、MRPCなどのタスクで、SNNとANNの**精度、レイテンシ、エネルギー効率**を比較します。 |
| benchmark continual | **継続学習検証** | EWCなどの継続学習メカニズムが「破局的忘却」をどれだけ抑制できるかを、逐次タスク学習で検証します。 |
| run-compiler-test | **ハードウェアシミュレーション** | 訓練済みSNNをニューロモーフィックコンパイラで処理し、ターゲットHW（例：Loihi）上での**推定エネルギー消費と処理時間**をシミュレートします。（個別スクリプト） |

#### **コマンド事例**

**1\. CIFAR-10でのANN/SNN性能比較**

python snn-cli.py benchmark run \\  
    \--experiment cifar10\_comparison \\  
    \--epochs 5 \\  
    \--tag "AccuracyTest\_CIFAR10"

**2\. SST-2（感情分析タスク）での比較**

python snn-cli.py benchmark run \\  
    \--experiment sst2\_comparison \\  
    \--epochs 5 \\  
    \--tag "AccuracyTest\_SST2"

### **3.4. 🤖 自律的知能の実行**

エージェントや認知アーキテクチャ全体を動作させ、自律的な思考や学習のサイクルを実行します。

| コマンド | アクション | 説明 |
| :---- | :---- | :---- |
| agent solve | **オンデマンド学習** | エージェントにタスクを与え、最適な専門家モデルを**自律的に検索・学習**させ、必要に応じて推論を実行します。 |
| agent evolve | **自己進化** | エージェントが自身の性能を評価し、モデルのアーキテクチャや学習パラダイムを**自律的に改善**するサイクルを実行します。 |
| agent rl | **強化学習** | 生物学的学習則（R-STDP、因果追跡）を持つエージェントをGridWorldなどの環境で訓練します。 |
| planner | **計画推論** | 複雑な要求に対し、知識ベースと専門家スキルマップに基づき、最適な実行ステップを立案させます。（個別スクリプト） |
| brain | **人工脳シミュレーション** | 統合された認知アーキテクチャ全体 (ArtificialBrain) を起動し、知覚、情動、記憶、行動のサイクルを観察します。 |
| life-form | **デジタル生命体** | AIを内発的動機に基づき、無限（または指定時間）に自律活動（思考、学習、進化）させるループを実行します。 |

#### **コマンド事例**

**1\. 対話形式で人工脳を起動し、その思考プロセスを観察**

python snn-cli.py brain \--loop

**2\. 単一の入力で人工脳を実行**

python snn-cli.py brain \--prompt "エラーが発生した。対応策を考えよ。"

### **3.5. 🖥️ UIとデプロイ**

ユーザーインターフェース（Gradio）の起動と、外部システムへの連携に関するコマンドです。

| コマンド | アクション | 説明 |
| :---- | :---- | :---- |
| ui | **標準UI起動** | SNNモデルとのリアルタイム対話UI (Gradio) を起動します。 |
| ui \--start-langchain | **LangChain UI起動** | SNNモデルをLangChainアダプタ経由で利用するUIを起動し、外部エコシステムとの連携をテストします。 |

#### **コマンド事例**

**1\. 学習済みの小規模モデルでUIを起動**

python snn-cli.py ui \--model\_config configs/models/small.yaml  
