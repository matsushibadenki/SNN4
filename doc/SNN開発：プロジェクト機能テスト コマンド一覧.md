# **プロジェクト機能テスト コマンド一覧 (v4.0)**

## **1\. 概要**

このドキュメントは、プロジェクトの全機能をテスト・実行するためのコマンドを体系的にまとめたものです。ほとんどの機能は、統合CLIツール snn-cli.py を通じて実行されます。

## **2\. 推奨テストフロー**

プロジェクトを初めて触る方や、変更後に全体的な動作確認を行いたい場合は、以下の順序でコマンドを実行することを推奨します。

1. **システム全体の健全性チェック (pytest)**  
   * すべての単体・統合テストを実行し、基本的な動作に問題がないことを確認します。  
2. **オンデマンド学習のクイックスタート (agent solve \--force-retrain)**  
   * 小規模データで学習から推論までの一連のパイプラインが正常に完了することを確認します。  
3. **人工脳シミュレーションの対話実行 (brain loop)**  
   * 統合された認知アーキテクチャが、対話形式の入力に対してどのように応答するかを体感します。

## **3\. コマンドリファレンス**

### **A) システム健全性チェック**

**目的:** プロジェクトの基本的な健全性を確認するためのテスト。

#### **A-1. 全テストスイートの実行**

プロジェクト全体のユニットテストおよび統合テストを実行し、すべてのコンポーネントが正しく動作することを確認します。

pytest \-v

#### **A-2. 学習パラダイムの煙テスト**

主要な学習パラダイム（勾配ベース、物理法則ベース、生物学的学習則など）が、ごく小規模なデータでエラーなく実行できるかを個別に検証します。

pytest \-v tests/test\_smoke\_all\_paradigms.py

### **B) 学習とデータ準備**

**目的:** SNNモデルの学習や、そのために必要なデータセットを準備します。

#### **B-1. 手動での勾配ベース学習**

train.py を直接呼び出し、指定した設定ファイルに基づいてモデルを学習させます。詳細な学習設定の調整やデバッグに有用です。

\# 使用例: mediumサイズのモデルをsample\_data.jsonlで3エポック学習  
python snn-cli.py gradient-train \\  
    \--model\_config configs/models/medium.yaml \\  
    \--data\_path data/sample\_data.jsonl \\  
    \--override\_config "training.epochs=3"

#### **B-2. 大規模データセットの準備**

モデルの汎用的な言語能力を向上させるため、大規模な公開コーパス (WikiText-103) をダウンロードし、学習に適した形式に前処理します。

python scripts/data\_preparation.py

#### **B-3. オンデマンド学習（知識蒸留）**

自律エージェントに未知のタスクを与え、Webから収集した情報や既存のデータを用いて、タスク特化型の「専門家モデル」を自動的に生成（知識蒸留）させます。

\# 使用例: wikitext-103を使って「汎用言語モデル」を育成  
snn-cli.py agent solve \--task "汎用言語モデル" \--force-retrain

\# 使用例: 小規模データで「高速テスト」モデルを育成  
snn-cli.py agent solve \--task "高速テスト" \--unlabeled-data data/sample\_data.jsonl \--force-retrain

#### **B-4. 思考プランナーの学習**

PlannerSNN モデルを学習させます。これにより、プランナーは与えられた目標に対し、どの専門家スキルを選択すべきかをより正確に予測できるようになります。

python train\_planner.py

#### **B-5. ANNからSNNへのモデル変換**

既存のANNモデル（.safetensors, .gguf形式）からSNNモデルを生成します。

\# 重みを直接コピーする手法  
python scripts/convert\_model.py \--method convert \--ann\_model\_path dummy\_ann.safetensors \--output\_snn\_path runs/converted\_model.pth

\# 知識蒸留を行う手法  
python scripts/convert\_model.py \--method distill \--ann\_model\_path dummy\_ann.safetensors \--output\_snn\_path runs/distilled\_model.pth

### **C) 高度な認知・自律機能**

**目的:** 学習済みのモデルや認知コンポーネントを連携させ、高度なタスクを実行します。

#### **C-1. 学習済みモデルによるタスク解決**

オンデマンド学習などで育成した専門家モデルを呼び出して、特定のタスクを実行させます。

\# 使用例: 学習済みの「汎用言語モデル」に質問する  
snn-cli.py agent solve \--task "汎用言語モデル" \--prompt "SNNとは何ですか？"

#### **C-2. 階層プランナーによる複雑なタスクの実行**

複数の専門家スキルを組み合わせる必要がある複雑なタスクを、階層プランナーに依頼します。プランナーはタスクを分解し、最適な実行計画を立てます。

snn-cli.py planner execute \\  
    \--request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context\_data "SNNは非常にエネルギー効率が高いことで知られている。"

#### **C-3. 人工脳シミュレーション**

統合された認知アーキテクチャ (ArtificialBrain) 全体を動作させます。

\# 単一の入力で1サイクルだけ実行  
snn-cli.py brain run \--input\_text "素晴らしい成功体験でした。"

\# 対話形式で繰り返し実行  
snn-cli.py brain loop

#### **C-4. デジタル生命体の自律ループ**

AIが外部からの指示なしに、内発的動機（好奇心、退屈など）に基づいて自律的に思考・学習・自己改善するループを開始します。

\# 5サイクルの間、自律的に活動させる  
snn-cli.py life-form start \--cycles 5

\# 直前の行動理由をAI自身に説明させる  
snn-cli.py life-form explain-last-action

#### **C-5. 自己進化**

エージェントが自身の性能を評価し、アーキテクチャや学習パラメータ、学習パラダイム自体を改善するプロセスを1サイクル実行します。

snn-cli.py evolve run \\  
    \--task\_description "高難度タスク" \\  
    \--initial\_accuracy 0.4

#### **C-6. 強化学習**

生物学的学習則（報酬変調型STDP）を用いるエージェントが、GridWorld環境内で試行錯誤を通じてタスクを学習するプロセスを開始します。

snn-cli.py rl run \--episodes 1000

#### **C-7. マルチエージェントによる協調的タスク解決**

複数のエージェントが協調して単一の目標を解決する創発的システムを起動します。

snn-cli.py emergent-system execute \\  
    \--goal "最新のAIトレンドを調査し、その内容を要約する"

### **D) 評価・分析・ハードウェア連携**

**目的:** モデルの性能を定量的に評価し、将来のハードウェア展開を見据えたテストを実行します。

#### **D-1. SNN vs ANN ベンチマーク**

SNNとANNの性能（精度、速度、エネルギー効率）を、標準的なベンチマークタスクで比較評価します。

python scripts/run\_benchmark.py \--task sst2

#### **D-2. ニューロモーフィック・コンパイラテスト**

学習済みのSNNモデルを、ニューロモーフィックハードウェア向けの構成ファイルに変換（コンパイル）し、その性能をシミュレートします。

python scripts/run\_compiler\_test.py

### **E) 対話UI**

**目的:** GradioベースのWeb UIを起動し、モデルと対話します。

#### **E-1. 標準UI**

snn-cli.py ui start \--model\_config configs/models/medium.yaml

#### **E-2. LangChain連携版UI**

snn-cli.py ui start-langchain \--model\_config configs/models/medium.yaml

### **F) 知識ベース管理**

**目的:** 階層プランナーや自己参照に使用されるRAGシステムの知識ベースを管理します。

#### **F-1. 知識ベースの構築**

プロジェクト内のドキュメント (doc/ ディレクトリ) とエージェントの記憶ログから、ベクトルストアを構築します。

python scripts/build\_knowledge\_base.py  
