# **プロジェクト機能テスト コマンド一覧 v6.0**

## **1\. 概要**

このドキュメントは、SNN4プロジェクトの全機能をテスト・実行するためのコマンドを体系的にまとめたものです。v6.0では、プロジェクトの司令塔である統合CLIツール snn-cli.py にすべての機能呼び出しを集約し、コマンド体系を全面的に刷新しました。

## **2\. 推奨テストフロー**

プロジェクトを初めて触る方や、変更後に全体的な動作確認を行いたい場合は、以下の順序でコマンドを実行することを推奨します。

1. **システム全体の健全性チェック (pytest)**  
   * すべての単体・統合テストを実行し、基本的な動作に問題がないことを確認します。  
2. **オンデマンド学習のクイックスタート (snn-cli agent solve)**  
   * 小規模データで学習から推論までの一連のパイプラインが正常に完了することを確認します。  
3. **人工脳シミュレーションの対話実行 (snn-cli brain \--loop)**  
   * 統合された認知アーキテクチャが、対話形式の入力に対してどのように応答するかを体感します。

## **3\. コマンドリファレンス**

### **A) システム健全性チェック**

**目的:** プロジェクトの基本的な健全性を確認します。

#### **A-1. 全テストスイートの実行**

プロジェクト全体のユニットテストおよび統合テストを実行します。

pytest \-v

#### **A-2. 学習パラダイムの煙テスト**

主要な学習パラダイム（勾配ベース、物理法則、生物学的学習則など）が、ごく小規模なデータでエラーなく実行できるかを個別に検証します。

pytest \-v tests/test\_smoke\_all\_paradigms.py

### **B) 学習とデータ準備**

**目的:** SNNモデルの学習や、そのために必要なデータセットを準備します。

#### **B-1. 勾配ベース学習 (gradient-train)**

train.py を直接呼び出し、指定した設定ファイルに基づいてモデルを学習させます。

\# 使用例: mediumサイズのモデルをsnnTorchバックエンドで学習  
snn-cli gradient-train \\  
    \--model\_config configs/models/medium.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3" \\  
    \--backend snntorch

#### **B-2. Ultraモデルの学習パイプライン (train-ultra)**

大規模データセットの準備から、最大規模のUltraモデルの学習までを自動で実行します。

snn-cli train-ultra \--override\_config "training.epochs=10"

#### **B-3. 大規模データセットの準備**

この機能は train-ultra コマンドに統合されましたが、個別実行も可能です。  
大規模公開コーパス (WikiText-103) をダウンロードし、前処理します。  
python scripts/data\_preparation.py

### **C) 高度な認知・自律機能**

**目的:** 学習済みのモデルや認知コンポーネントを連携させ、高度なタスクを実行します。

#### **C-1. 専門家モデルの学習と実行 (agent solve)**

未知のタスクを与え、専門家モデルをオンデマンドで学習させるか、既存のモデルで推論を実行します。

\# 学習例: 小規模データで「高速テスト」モデルを育成  
snn-cli agent solve "高速テスト" \--unlabeled\_data\_path data/sample\_data.jsonl \--force\_retrain

\# 推論例: 学習済みのモデルに質問する  
snn-cli agent solve "高速テスト" \--prompt "SNNとは何ですか？"

#### **C-2. 自己進化 (agent evolve)**

エージェントが自身の性能を評価し、アーキテクチャや学習パラメータ、学習パラダイム自体を改善するプロセスを実行します。

snn-cli agent evolve "高難度タスク" \\  
    \--model\_config configs/models/small.yaml \\  
    \--training\_config configs/base\_config.yaml

#### **C-3. 強化学習 (agent rl)**

生物学的学習則を用いるエージェントが、GridWorld環境内でタスクを学習するプロセスを開始します。

snn-cli agent rl \--episodes 1000

#### **C-4. 階層プランナー (planner)**

複雑なタスクを分解し、最適な実行計画を立てさせます。

snn-cli planner \\  
    "この記事を要約して、その内容の感情を分析してください。" \\  
    "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

#### **C-5. 人工脳シミュレーション (brain)**

統合された認知アーキテクチャ ArtificialBrain 全体を動作させます。

\# 単一の入力で1サイクルだけ実行  
snn-cli brain \--prompt "素晴らしい成功体験でした。"

\# 対話形式で繰り返し実行  
snn-cli brain \--loop

#### **C-6. デジタル生命体 (life-form)**

AIが内発的動機に基づいて自律的に思考・学習・自己改善するループを開始します。

\# 60秒間、自律的に活動させる  
snn-cli life-form \--duration 60

#### **C-7. Webからの自律学習**

この機能は現在、個別スクリプトとして提供されています。  
指定したトピックについてWebをクロールし、専門家モデルを自律的に生成します。  
python run\_web\_learning.py \\  
    \--topic "最新のAI技術" \\  
    \--start\_url "\[https://www.itmedia.co.jp/news/subtop/aiplus/\](https://www.itmedia.co.jp/news/subtop/aiplus/)" \\  
    \--max\_pages 5

### **D) 評価・分析・ハードウェア連携**

**目的:** モデルの性能を定量的に評価し、将来のハードウェア展開を見据えたテストを実行します。

#### **D-1. SNN vs ANN ベンチマーク (benchmark run)**

SNNとANNの性能（精度、速度、エネルギー効率）を、標準的なベンチマークタスクで比較評価します。

\# 使用例: MRPCタスクで比較  
snn-cli benchmark run \--experiment mrpc\_comparison \--epochs 3

#### **D-2. 継続学習ベンチマーク (benchmark continual)**

ANNの弱点である「破局的忘却」の克服を実証する実験を実行します。

snn-cli benchmark continual \--epochs\_task\_a 3 \--epochs\_task\_b 3

#### **D-3. ニューロモーフィック・コンパイラテスト**

この機能は現在、個別スクリプトとして提供されています。  
学習済みSNNをニューロモーフィックハードウェア向けの構成に変換し、性能をシミュレートします。  
python scripts/run\_compiler\_test.py

### **E) モデル変換**

**目的:** 既存のANNモデル資産をSNNに変換します。

#### **E-1. CNNモデルの変換 (convert ann2snn-cnn)**

学習済みのSimpleCNN (ANN) モデルを、SpikingCNN (SNN) に変換します。

snn-cli convert ann2snn-cnn \\  
    path/to/your/ann\_model.pth \\  
    path/to/your/output\_snn.pth \\  
    \--snn\_model\_config configs/cifar10\_spikingcnn\_config.yaml

### **F) 対話UI**

**目的:** GradioベースのWeb UIを起動し、モデルと対話します。

#### **F-1. 標準UI・LangChain連携版UI**

\# 標準UIの起動  
snn-cli ui \--model\_config configs/models/medium.yaml

\# LangChain連携版UIの起動  
snn-cli ui \--model\_config configs/models/medium.yaml \--start-langchain

### **G) 知識ベース管理**

**目的:** RAGシステムの知識ベースを管理します。

#### **G-1. 知識ベースの構築**

この機能は現在、個別スクリプトとして提供されています。  
プロジェクト内のドキュメントからベクトルストアを構築します。  
python scripts/build\_knowledge\_base.py  
