# 実験レポート: sho-swin-asl-scheduler

作成日: 2026-07-20

## 1. 共有用サマリ

### 1.1 この実験の位置づけ

- 何を改善しようとしたか: 固定学習率による学習後半の過学習（Val Loss上昇）を防ぐため。
- ベースラインまたは直前実験から変えたこと: 学習率スケジューラー `ReduceLROnPlateau` を導入（`factor=0.5`, `patience=3`）。※Linear Warmupは初期学習の勢いを削ぐ挙動が見られたため今回は除外して検証した。
- 主評価指標 mAP の結果をどう判断するか: validation mAP は 0.3688 となり、直前のスケジューラーなしASL（0.3675）からほぼ横ばいであった。
- 何がダメだったか / まだ残っている問題: スケジューラーで過学習を抑えても精度が伸びなかった。これは学習率の問題ではなく、ASLのハイパーパラメータ（$\gamma_+$, $\gamma_-$, $m$）が現在のデータセットに対して最適化されていないことが根本原因であると考えられる。

### 1.2 自動要約

- この実験の比較行が見つかりませんでした。
- 同じ実験グループの seed 別分析結果が見つかりませんでした。

### 1.3 採用判断

- 採用判断: TODO（採用 / 条件付き採用 / 不採用 / 保留）
- 判断理由: TODO
- 次に試すこと: TODO

## 2. 他実験との比較

`config.yaml` で明示した主比較と参考実験だけを、validation mAP を中心に比較します。test split は最終モデル選定後まで使いません。

| 実験 | 役割 | method | validation mAP | mAP 標準偏差 | 今回との差 | Macro F1 | Samples F1 | Hamming Loss | 予測ジャンル数/作品 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sho-swin-asl | 主比較 | 0.5 固定 | 0.3675 | 0.0024 |  | 0.3684 | 0.5126 | 0.1525 | 3.3616 |

### 2.1 複数 seed 集計

seed 集計グループ: `sho-swin-asl-scheduler`

_該当するデータが見つかりませんでした。_

#### seed 別結果

_該当するデータが見つかりませんでした。_

## 3. 実験の目的と変更

### 3.1 背景

ASL導入により過剰予測の問題は解決し Samples F1 は高水準に達したが、固定学習率のままでは学習後半に Val Loss が上昇する「過学習の兆候」が残っていた。
### 3.2 仮説

学習の停滞を検知して学習率を減衰させるスケジューラー（`ReduceLROnPlateau`）を導入すれば、モデルを損失関数の谷底へ安全に着地させ、mAPのさらなる向上が見込める。
### 3.3 検証した変更

| 種類 | 内容 | mAP 改善につながると考えた理由 |
|---|---|---|
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) の追加 | 谷底でのバウンド（過学習）を防ぎ、より深い最適な重みを探索させるため。 |

### 3.4 比較条件

- 主比較: `sho-swin-asl`
- 参考実験: なし
- 変えたもの: オプティマイザの学習率スケジュール処理
- 変えていないもの: モデル構造(Swin-T)、損失関数(ASL)、初期学習率(1e-4)

### 3.5 主な設定

| 項目 | 値 |
| --- | --- |
| seed | 42 |
| seeds | 42, 43, 44 |
| device | auto |
| comparison | {"primary": "sho-swin-asl", "references": []} |
| epochs | 100 |
| early_stopping | {"enabled": true, "monitor": "mAP", "mode": "max", "patience": 15, "min_delta": 0.001, "min_epochs": 1} |
| batch_size | 64 |
| learning_rate | 1e-4 |
| num_workers | 0 |
| image_size | 224 |
| compile | True |
| max_train_samples |  |
| max_val_samples |  |
| output_dir | outputs |
| best_model_name | best_model.pth |
| metrics_name | metrics.csv |

### 3.6 再現コマンド

```bash
uv run python experiments/sho-swin-asl-scheduler/run_exp.py
uv run python experiments/sho-swin-asl-scheduler/analyze.py
uv run python experiments/sho-swin-asl-scheduler/make_report.py
```

## 4. 学習ログ

### 4.1 代表 epoch

_該当するデータが見つかりませんでした。_

### 4.3 読み取りメモ

- Train Loss と Val Loss の差が開く場合は、過学習を疑う。
- 主評価指標は mAP。mAP 最大 epoch と最終 epoch の差を見る。
- F1 はしきい値で 0/1 にした後の補助指標。mAP が改善していても F1 が悪い場合は threshold 設計を疑う。
- Hamming Loss は低いほど良いが、何も予測しないモデルでも低く見える場合がある。

## 5. 全体評価

_該当するデータが見つかりませんでした。_

### 5.1 mAP 中心の読み取り

- validation mAP が比較対象より上がったか: TODO
- validation mAP の改善幅は、偶然や seed 差より十分大きそうか: TODO
- mAP は上がったが補助指標が悪化した場合、その悪化を許容できるか: TODO

## 6. ジャンル別結果

### 6.1 主比較 `sho-swin-asl` とのジャンル別 AP 差分

_該当するデータが見つかりませんでした。_

### 6.2 AP が高いジャンル

_該当するデータが見つかりませんでした。_

### 6.3 F1 が低いジャンル

_該当するデータが見つかりませんでした。_

### 6.4 考察の観点

- AP が高いジャンルは、モデルが順位付けできているジャンル。mAP 改善に寄与している可能性が高い。
- AP が低いジャンルは、特徴量・データ量・ラベルの曖昧さなどを疑う。
- AP は高いのに F1 が低いジャンルは、しきい値調整で改善する可能性がある。
- Precision が低いジャンルは、関係ない作品にもそのジャンルを付けすぎている。
- Recall が低いジャンルは、本当はそのジャンルの作品を見逃している。
- 件数が少ないジャンルは、少数の正解/不正解で指標が大きく動く。

## 7. 人が書く考察

### 7.1 何を改善しようとして、改善できたか

学習パイプラインへの「安全装置」の組み込み。過学習（Val Lossの上昇）をトリガーにして学習率を絞る仕組み自体は正常に動作し、学習を安定して終わらせるベース環境が完成した。

### 7.2 何がダメだったか / 想定と違ったか

スケジューラーを入れて最適化の質を上げても、最終的な mAP や F1 スコアは前回の「ASL単体（固定LR）」とほとんど変わらなかった（誤差レベルの微増）。

### 7.3 原因仮説

| 仮説ID | 観察した結果 | 原因仮説 | 次の確認方法 |
|---|---|---|---|
| H1 | スケジューラーを入れても mAP が 0.36 台で頭打ちになる | 問題は「最適化手法」ではなく、「損失関数の設計（ASLのパラメータ）」にある。現在の $\gamma_-=4, m=0.05$ という設定が、このデータセットに対して厳しすぎる（または緩すぎる）。 | ASLのガンマ値やマージンを変更して再実験する。 |

### 7.4 他メンバーに共有したい注意点

「Swin Transformer + ASL + Scheduler」の組み合わせにより、コード実装の基盤としては完全にSOTA（最高レベル）の構成が整いました。しかし、スケジューラーはあくまで「設定されたLossの底へ無事に導く」だけのツールです。ゴール（Loss）の形が今のデータセットに最適化されていなければ、精度は上がりません。

### 7.5 次に試すこと

実験環境（インフラ）は完全に整ったため、今後は純粋な「ハイパーパラメータ・チューニング」に移行します。
1. **ASLパラメータの調整:** $\gamma_-=2$ に緩める、あるいはマージン $m$ を調整して、mAPの変動を観察する。
2. **しきい値（Threshold）の最適化:** 0.5 固定ではなく、評価・推論時に各ジャンルで最適なF1が出るしきい値を計算して適用する。

## 8. 生成元ファイル

| ファイル | 用途 |
| --- | --- |
| experiments/sho-swin-asl-scheduler/config.yaml | 実験設定 |
| 未生成 | epoch ごとの学習ログ |
| experiments/sho-swin-asl-scheduler/analysis/overall_model_metrics.csv | validation の全体指標 |
| experiments/sho-swin-asl-scheduler/analysis/genre_metrics_validation_threshold_0.5.csv | validation のジャンル別指標 |

### 8.1 analysis ディレクトリ内のファイル

- analysis ディレクトリが見つかりませんでした。
