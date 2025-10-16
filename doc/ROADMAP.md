# ROADMAP v10.1 — 継続学習

> この追加は「ANN を超える SNN 系 AI を実現する」ための継続学習要素（Meta-Plasticity、Neuromodulation、Replay、Adapter 層など）を、既存 ROADMAP v10.0 のフェーズに**スケジュールとして組み込み**たものです。

---

## 目的（Why）
- SNN の**継続学習能力**と**低エネルギー利点**を活かして、ANN 系の静的モデル（LLM 等）では実現しにくい「逐次適応」「オンデバイス更新」「忘却耐性」を持つ AI を構築する。
- 短期で ANN 性能に追随しつつ、中長期で ANN の限界を超える運用上の優位性（省エネ・適応性）を示す。

---

## 要点（What）
1. **短期（0–6週）**：ANN→SNN の精度ギャップを潰す基盤（Distillation, ann2snn, threshold balancing）を固める。
2. **中期（6–24週）**：継続学習モジュール（MetaPlasticity、Neuromodulated Plasticity、Replay Memory、Adapter 層）を実装し、オンライン更新パイプラインを確立する。
3. **長期（24週〜12ヶ月）**：大規模スパイキングTransformer・ニューロモルフィック実機統合（Loihi 等）、運用でのエネルギー優位性、自己進化/社会性実験。

---

## フェーズ別スケジュール（Who / When / How）

### フェーズ 0 — 準備（Week - / 0–2）
**目的**: ベースライン整備
- タスク
  - ベンチセット確定（CIFAR-10/100, ImageNet-subset, DVS dataset, NLP token tasks小規模）
  - 教師モデル準備（ResNet / ViT / 小型Transformer）
  - CI パイプライン（benchmarks/run_bench.py）作成
- 成果物: ベースラインレポート、ベンチスイート

### フェーズ 1 — 短期実装（Week 0–6）
**目的**: ANN→SNN 蒸留で迅速に精度差を縮める
- 主要タスク（並列実施）
  - Distillation 拡張（`snn4/distillation/`）: `teacher.py`, `losses.py`, `projector.py` を追加
  - ANN→SNN 変換ツール（`snn4/tools/ann2snn.py`）: BN folding, threshold balancing
  - ハイパーパラ探索自動化（Optuna）で T, temp, loss weights を最適化
  - 評価: top1/top5, spike count, latency, energy-estimate
- マイルストーン（Week 6）: CIFAR/小型 ImageNet で ANN 比 ≤2% を目指す

### フェーズ 2 — 継続学習プロトタイプ（Week 6–16）
**目的**: 継続学習の基本ユニットを実装・評価
- 主要タスク
  - `snn4/modules/adaptive_learning.py` を追加（MetaPlasticityLayer, Neuromodulator）
  - Replay Buffer（スパイク列の保存・再生）と Replay Scheduler 実装
  - Adapter 層（局所的ファインチューニング用 1x1 conv）を設計・追加
  - 小規模オンライン学習実験（逐次データ流での学習と忘却評価）
- マイルストーン（Week 12）: 新データを逐次追加しても既存性能の低下（forgetting）を 30% 以下に抑える

### フェーズ 3 — Surfacing LLM-like 機能（Week 16–36）
**目的**: 大規模言語処理に近い用途へ適用するための構造適応
- 主要タスク
  - Spiking Transformer の中核実装（`snn4/models/spiking_transformer.py`）
  - Spiking Embedding / Token Encoding（時間符号化）を設計
  - 継続学習モジュールをトランスフォーマーに統合
  - オフライン蒸留 + オンライン適応ハイブリッド戦略を確立
- マイルストーン（Week 36）: 小型言語タスク（次単語予測等）での意味的整合性確認

### フェーズ 4 — ニューロモルフィック統合 & 運用（Week 36–52+）
**目的**: 実機でのエネルギー優位性とオンライン学習を実証
- 主要タスク
  - Loihi 等へのモデル変換・最適化（sparser weights, quantization, delay handling）
  - 実機で Joules/inference を計測し、同クラス ANN と比較
  - 自律運用シナリオでの長期学習実験（数週間〜数ヶ月）
- マイルストーン（Month 12）: 実機で Joules/inference を ANN 比で 3x 以上改善（目標例）

---

## タスク詳細マトリクス（短期〜中期の Sprint 単位）
- Sprint 0 (Week 0–2): ベンチ整備、teacher モデル準備、CI
- Sprint 1 (Week 2–4): Distillation コード化、ann2snn PoC
- Sprint 2 (Week 4–6): ハイパーパラ最適化、精度チェック、レポート
- Sprint 3 (Week 6–8): MetaPlasticity 基礎モジュール実装
- Sprint 4 (Week 8–12): Replay/Adapter 実験、忘却試験
- Sprint 5 (Week 12–16): 基本オンライン学習パイプラインの完成
- Sprint 6–9 (Week 16–36): SpikingTransformer 構築と統合試験
- Sprint 10–? (Week 36+): ニューロモルフィック最適化と実機評価

---

## リポジトリとの対応（Suggested file map）
- `snn4/distillation/teacher.py` — ANN teacher extractor
- `snn4/distillation/losses.py` — logits/feature/timing/spike sparsity losses
- `snn4/tools/ann2snn.py` — conversion utilities (BN fold, threshold balancing)
- `snn4/modules/adaptive_learning.py` — MetaPlasticityLayer, Neuromodulation
- `snn4/modules/replay.py` — Spike Replay Buffer & Scheduler
- `snn4/models/spiking_transformer.py` — Spiking Transformer prototype
- `benchmarks/run_bench.py` — 自動ベンチスイート

---

## 評価指標（KPI）
- 精度: top1 / top5（各データセット）
- 忘却率: 新知識学習後の既存タスク性能低下（%）
- スパイク効率: 平均スパイク数 / inference
- レイテンシ: ms / inference
- エネルギー: J / inference（実測 or 見積）
- 総合指標: 性能（精度）/消費エネルギー 比

---

## リスクと対策
- **リスク**: 追加学習で精度が崩れる（忘却）
  - **対策**: EWC 相当の重要度固定 + replay + adapter 層で局所更新
- **リスク**: 実機移行時に大幅な精度低下
  - **対策**: 早期に quantization/delay-aware の PoC を行う
- **リスク**: スパイク表現が NLP トークンに馴染まない
  - **対策**: hybrid embedding（ANN 埋め込み → スパイク変換）で橋渡し

---

## 成功定義（Success Criteria）
1. 短期: CIFAR/ImageNet-subset で ANN 比 ≤ 2%（Week 6）
2. 中期: 継続学習で忘却率 ≤ 30%（新データ追加後 3 サイクル）
3. 長期: 実機で J/inference が ANN 比で 2–3x 改善（Month 12）

---

## 次アクション（短く）
1. Sprint 0 を開始：ベンチセット/teacher を定める（2 週間）
2. Distillation + ann2snn の PoC を Sprint 1 にアサイン
3. Sprint 2 でハイパーパラ最適化と結果レポート


---

_この文書は ROADMAP v10.0 に対する追記であり、既存のゴールや構成を変更することなく「継続学習要素」をスケジュールにブレンドしています。必要であれば、各 Sprint に対する担当者割り当て、見積り (man-weeks)、CI ジョブ設定などさらに詳細化します。_

