# **プロジェクト機能テスト コマンド一覧 (v3.0)**

このドキュメントは、プロジェクトの各機能をテストするための主要なコマンドをまとめたものです。コマンドは、専用の実行スクリプトまたは統合CLIツール snn-cli.py を通じて実行します。

## **A) システム健全性チェック**

**目的:** プロジェクトの基本的な健全性を迅速に確認するためのテスト。

### **0\. クイックテスト・簡易テストの実行**

全ての簡易テスト。

pytest -v tests/test_smoke_all_paradigms.py

### **1\. ユニットテスト・統合テストの実行**

プロジェクト全体のユニットテストおよび統合テストを実行し、すべてのコンポーネントが個別に、また連携して正しく動作することを確認します。これは、変更を加えた際の最も基本的な健全性チェックです。

pytest \-v

### **2\. オンデマンド学習のクイックテスト**

agent solveコマンドが、小規模なサンプルデータを使って、専門家モデルの（再）学習から推論までの一連のサイクルをエラーなく完了できるかを確認します。

python snn-cli.py agent solve \\  
    \--task "高速テスト" \\  
    \--prompt "これはテストです。" \\  
    \--unlabeled-data data/sample\_data.jsonl \\  
    \--force-retrain

**Note:** このテストはシステムの動作確認用です。小規模データのため、AIは意味のある応答を生成できません。

## **B) 主要機能・学習パイプラインテスト**

**目的:** 主要な学習パラダイムやデータ処理パイプラインが正しく機能することを確認します。

### **3\. 標準的な勾配ベース学習**

train.py を直接呼び出し、指定された設定で短時間の学習を完了できるかを確認します。

python train.py \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3"

### **4\. ANN-SNNモデル変換と知識蒸留**

scripts/convert\_model.py を使用して、既存のANNモデルからSNNモデルを生成する2つの主要な手法をテストします。

#### **4.1 重み変換テスト**

Safetensors形式のモデルから重みを直接コピーする機能をテストします。（事前にダミーのsafetensorsファイルが必要です）

\# 事前にダミーのANNモデルが必要です  
\# python scripts/convert\_model.py \--method convert \--ann\_model\_path dummy\_ann.safetensors \--output\_snn\_path runs/converted\_model.pth

#### **4.2 知識蒸留テスト**

ANNモデルを教師役として、SNNモデルを蒸留学習させる機能をテストします。（事前にダミーのANNモデルが必要です）

\# python scripts/convert\_model.py \--method distill \--ann\_model\_path dummy\_ann.safetensors \--output\_snn\_path runs/distilled\_model.pth

### **5\. 思考プランナーの学習**

PlannerSNNを学習させるためのスクリプトを実行します。（現在はダミーデータで動作確認）

python train\_planner.py

### **6\. 大規模データセットによるオンデマンド学習**

wikitext-103を使い、汎用的な言語能力を持つ専門家モデルを育成します。AIの応答品質を本格的に向上させるには、このコマンドの実行が必要です。

**ステップ 1: 大規模データセットの準備（初回のみ）**

python scripts/data\_preparation.py

**ステップ 2: 本格的な学習の実行**

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--force-retrain

### **7\. 学習済みモデルとの対話**

上記の学習で育成した「汎用言語モデル」を呼び出して対話します。

python snn-cli.py agent solve \\  
    \--task "汎用言語モデル" \\  
    \--prompt "SNNとは何ですか？"

## **C) 高度な生物学的学習則の検証**

**目的:** バックプロパゲーションに依存しない、脳に着想を得た学習アルゴリズムの動作を検証します。

### **8\. 基本的な強化学習**

エージェントが複数ステップのGridWorld環境でタスクを学習できることを検証します。学習後、学習曲線グラフと訓練済みモデルがruns/rl\_results/に保存されます。

python run\_rl\_agent.py \--episodes 1000 \--grid\_size 5 \--max\_steps 50

### **9\. 適応的因果スパース化を有効にした強化学習**

貢献度の低いシナプスの学習を抑制する「適応的因果スパース化」を有効にして、CausalTraceCreditAssignment学習則を実行します。

python train.py \--paradigm bio-causal-sparse \--config configs/base\_config.yaml

### **10\. パーティクルフィルタによる確率的学習**

微分不可能なSNNを確率的なアンサンブルとして学習させるParticleFilterTrainerを実行します。

python train.py \--paradigm bio-particle-filter \--config configs/base\_config.yaml

## **D) 高度な認知・自律機能テスト**

**目的:** 自己進化やマルチエージェント協調など、より高度なシステムの動作を確認します。

### **11\. Webからの自律学習**

AIに新しいトピックをWebから自律的に学習させ、その知識に基づいた専門家モデルを生成させます。

python run\_web\_learning.py \\  
    \--topic "最新の半導体技術" \\  
    \--start\_url "\[https://pc.watch.impress.co.jp/\](https://pc.watch.impress.co.jp/)" \\  
    \--max\_pages 10

### **12\. 自己進化**

エージェントが自身の性能を評価し、アーキテクチャや学習パラメータ、さらには学習パラダイム自体を改善するプロセスをテストします。

python snn-cli.py evolve run \\  
    \--task\_description "高難度タスク" \\  
    \--initial\_accuracy 0.4 \\  
    \--model\_config "configs/models/small.yaml" \\  
    \--training\_config "configs/base\_config.yaml"

### **13\. デジタル生命体の自律ループ**

AIが外部からの指示なしに、内発的動機に基づいて自律的に思考・学習するループを開始します。

python snn-cli.py life-form start \--cycles 10

### **14\. マルチエージェントによる協調的タスク解決**

複数のエージェントが協調して単一の目標を解決する創発的システムを起動します。

python snn-cli.py emergent-system execute \\  
    \--goal "最新のAIトレンドを調査し、その内容を要約する"

## **E) 統合アーキテクチャ・シミュレーション**

### **15\. 人工脳 全体シミュレーション**

これまでに実装された全ての認知コンポーネントを統合したArtificialBrainを起動し、感覚入力から思考、行動出力までの一連の認知サイクルをシミュレートします。

python run\_brain\_simulation.py

## **F) 将来機能・ハードウェア連携テスト**

### **16\. ニューロモーフィック・コンパイラテスト**

学習済みのBioSNNモデルを、Intel Loihiのようなニューロモーフィックハードウェア向けの構成ファイルに変換（コンパイル）する機能をテストします。

python scripts/run\_compiler\_test.py

## **G) 対話UIの起動**

### **17\. 標準UI**

Gradioベースの標準的な対話UIを起動します。

python snn-cli.py ui start \--model\_config configs/models/medium.yaml

### **18\. LangChain連携版UI**

SNNモデルをLangChainエージェントとしてラップした対話UIを起動します。

python snn-cli.py ui start-langchain \--model\_config configs/models/medium.yaml  
