# 実験レポート: riku-lab

作成日: 2026-07-24

> **数値について**
>
> このレポートは、2026-07-24に共有された `Validation mAP` と
> `Train / Validation Loss` のグラフを基に作成した。
> グラフの元になった `metrics.csv` は現在のローカル環境に存在しないため、
> epochごとの数値は画像から読み取った概算値である。

## 1. 共有用サマリ

### 1.1 この実験の位置づけ

- 何を改善しようとしたか: ImageNet事前学習済みConvNeXt-Tinyを19ジャンルのマルチラベル分類へ転移学習し、validation mAPを改善する。
- ベースラインまたは直前実験から変えたこと: ConvNeXt-Tinyの最後のCNBlockと分類器を学習対象とし、その他の層を固定する。
- 主評価指標mAPの結果をどう判断するか: mAPはepoch 1の約0.255から上昇し、epoch 18付近で最大約0.384に到達した。その後は約0.381前後で横ばいになった。
- 何がダメだったか / まだ残っている問題: epoch 15前後からTrain Lossだけが下がり続ける一方、Validation Lossは下げ止まって上昇傾向になった。後半は過学習が始まっており、学習を続けてもvalidation mAPはほとんど改善していない。

### 1.2 自動要約

- 学習は28 epochまで記録されている。
- validation mAPは序盤に大きく改善した。
- epoch 10で約0.378、epoch 14以降はおおむね0.380以上。
- 最大validation mAPはepoch 18付近の約0.384。
- epoch 19以降のmAPは約0.381前後で横ばい。
- Validation Lossはepoch 15付近で最小約0.273。
- 最終epochではTrain Loss約0.228、Validation Loss約0.277となり、両者の差が広がった。

### 1.3 採用判断

- 採用判断: 条件付き採用
- 判断理由: mAPは安定して約0.38まで改善しており、グループ共通の`baseline`のvalidation mAP 0.2876を約0.096上回った。一方、後半には過学習が見られ、今回の結果は1実行分だけなので再現性が未確認である。
- 採用するcheckpoint: validation mAPが最大になったepoch 18付近のcheckpoint。
- 次に試すこと: early stoppingを使い、epoch 18前後で学習を終了する。正確なbest epochは元の`metrics.csv`で確認する。

## 2. 他実験との比較

test splitは使用せず、validation mAPを中心に比較する。

| 実験 | 役割 | validation mAP | mAP標準偏差 | 今回との差 | 備考 |
| --- | --- | ---: | ---: | ---: | --- |
| riku-lab | 今回 | 約0.384 | 未算出 | — | 画像からの概算、bestはepoch 18付近 |
| baseline | 主比較 | 0.2876 | 0.0005 | 約+0.0964 | 3 seed平均 |

今回のmAPは学習前半から着実に上昇し、主比較を約0.096上回った。
ただし、今回の値は1実行のグラフから読み取った概算値なので、
複数seedで改善を再現できるまでは置き換えを確定しない。

### 2.1 複数seed集計

seed集計グループ: `riku-lab`

*共有されたグラフは1回分だけなので、複数seedの平均・標準偏差は算出できない。*

## 3. 実験の目的と変更

### 3.1 背景

画像ジャンル分類では、分類器だけを学習するとImageNet特徴を十分にタスクへ適応できない可能性がある。
一方、ConvNeXt全体を更新すると、学習データに過剰適合して事前学習済み特徴を壊す危険がある。

### 3.2 仮説

ConvNeXt-Tinyの最後のCNBlockと分類器だけを更新すれば、事前学習済みの汎用的な画像特徴を残しながら、アニメ画像のジャンル判定に必要な高水準特徴を学習できると考えた。

### 3.3 検証した変更

| 種類 | 内容 | mAP改善につながると考えた理由 |
| --- | --- | --- |
| モデル | ImageNet事前学習済みConvNeXt-Tiny | 事前学習済みの画像特徴を利用できるため |
| 学習範囲 | 最後のCNBlockとclassifier | 全層更新より過学習を抑えながら、タスク固有特徴を学習するため |
| loss | BCEWithLogitsLoss | 19ジャンルを独立した二値分類として扱えるため |
| augmentation | crop、左右反転、弱い色変化、回転 | ポスター画像の軽微な変化への頑健性を高めるため |
| モデル選択 | validation mAP最大 | 主評価指標に直接基づいてcheckpointを選ぶため |

### 3.4 比較条件

- 主比較: `baseline`
- 変えたもの: モデルと学習対象層
- 変えていないもの: 19ジャンルのマルチラベル分類、validationによるモデル選択
- 主評価指標: validation mAP
- 補助指標: Train Loss、Validation Loss
- test split: 最終モデル選定後まで使用しない

### 3.5 主な設定

以下は現在の `experiments/riku-lab` の設定。共有画像を生成した実行と完全に一致するかは、元のconfigで再確認する必要がある。

| 項目 | 値 |
| --- | --- |
| seed | 42 |
| model | ConvNeXt-Tiny |
| pretrained | ImageNet DEFAULT |
| trainable part | 最後のCNBlock + classifier |
| trainable parameters | 4,779,283 / 27,834,739（17.17%） |
| batch size | 48 |
| image size | 224 |
| head learning rate | 0.0003 |
| backbone learning rate | 0.00001 |
| weight decay | 0.0001 |
| loss | BCEWithLogitsLoss |
| augmentation | 有効 |
| best metric | validation mAP |
| threshold | 0.5 |

### 3.6 再現コマンド

```bash
USE_SYNCLR_DISTILLATION=0 uv run python experiments/riku-lab/run.py
```

## 4. 学習ログ

### 4.1 代表epoch

| Epoch | Train Loss | Validation Loss | validation mAP | 読み取り |
| ---: | ---: | ---: | ---: | --- |
| 1 | 約0.364 | 約0.307 | 約0.255 | 学習開始 |
| 5 | 約0.273 | 約0.278 | 約0.353 | mAPが大きく改善 |
| 10 | 約0.256 | 約0.274 | 約0.378 | 改善が緩やかになる |
| 14 | 約0.248 | 約0.274 | 約0.381 | mAPがほぼ飽和 |
| 18 | 約0.241 | 約0.274 | **約0.384** | mAP最大付近 |
| 20 | 約0.238 | 約0.275 | 約0.382 | Validation Lossが上昇傾向 |
| 28 | 約0.228 | 約0.277 | 約0.381 | 過学習傾向、mAP改善なし |

*すべて共有画像からの概算値。正確な値は元の`metrics.csv`で置き換える。*

### 4.2 mAPの推移

1. epoch 1〜10では約0.255から約0.378まで大きく改善した。
2. epoch 10〜18では改善幅が小さくなり、最大約0.384に到達した。
3. epoch 19〜28では約0.381前後を上下し、新しい改善はほぼ見られなかった。

### 4.3 Lossの推移

- Train Lossはepoch 1の約0.364からepoch 28の約0.228まで一貫して低下した。
- Validation Lossはepoch 1の約0.307からepoch 15付近の約0.273まで低下した。
- epoch 15以降、Validation Lossは約0.274〜0.279で上下し、後半はわずかに悪化した。
- Train LossとValidation Lossの差が広がっているため、epoch 15前後から過学習が始まったと考えられる。

## 5. 全体評価

| 観点 | 評価 |
| --- | --- |
| mAPの改善 | 良い。約0.255から約0.384へ上昇 |
| 学習の安定性 | epoch 14以降は約0.38で安定 |
| 汎化性能 | 後半に過学習傾向あり |
| 主比較との関係 | `baseline`より約0.096高い |
| seed再現性 | 未確認 |
| 総合判断 | 条件付き採用 |

### 5.1 mAP中心の読み取り

- validation mAPが学習によって改善したか: 改善した。約0.255から最大約0.384まで上昇した。
- validation mAPが比較対象より上がったか: 上がった。`baseline`の0.2876より約0.096高い。
- 学習を長く続ける価値があるか: 現設定のままepoch数だけ増やす価値は低い。epoch 18以降はmAPが改善せず、Validation Lossも悪化傾向にある。
- 改善幅がseed差より十分大きいか: 今回は1実行分だけなので判断できない。

## 6. ジャンル別結果

### 6.1 主比較とのジャンル別AP差分

*共有画像にはジャンル別APが含まれていないため、算出できない。*

### 6.2 APが高いジャンル

*ジャンル別評価結果がないため確認できない。*

### 6.3 F1が低いジャンル

*ジャンル別評価結果がないため確認できない。*

### 6.4 次回必要な分析

- ジャンル別AP、Precision、Recall、F1を保存する。
- 各ジャンルのpositive件数と併記する。
- threshold 0.5固定だけでなく、validationでジャンル別thresholdを検証する。
- 1作品あたりの予測ジャンル数を保存する。

## 7. 人が書く考察

### 7.1 何を改善しようとして、改善できたか

ConvNeXt-Tinyの最後のCNBlockと分類器を学習することで、validation mAPは約0.255から約0.384まで改善した。特に最初の10 epochの改善が大きく、転移学習によってアニメ画像のジャンル分類に必要な特徴を学べたと考えられる。

### 7.2 何がダメだったか / 想定と違ったか

epoch 14以降はmAPがほぼ横ばいになった。Train Lossはその後も低下したが、Validation Lossは低下せず、学習データへの適合だけが進んだ。最大mAPは主比較の`baseline`を上回ったが、後半の追加学習は性能改善につながらなかった。

### 7.3 原因仮説

| 仮説ID | 観察した結果 | 原因仮説 | 次の確認方法 |
| --- | --- | --- | --- |
| H1 | epoch 15以降にTrain/Validation Lossの差が拡大 | 学習データへの過学習 | augmentation、weight decay、dropoutを比較する |
| H2 | epoch 18以降mAPが改善しない | 現在のモデル・lossで到達できる性能が飽和 | loss、学習率、解凍範囲を1項目ずつ比較する |
| H3 | baselineよりmAPが約0.096高い | 最後のCNBlockまで微調整したことが、分類器のみのbaselineより有効だった可能性 | 同じ3 seedで再実行し、学習範囲以外の条件をそろえて比較する |
| H4 | 1実行分しかない | 結果がseedに依存している可能性 | seed 42、43、44で再実行する |

### 7.4 他メンバーに共有したい注意点

最大mAPは最終epochではなくepoch 18付近で得られている。最終epochのモデルではなく、validation mAPが最大になったcheckpointを使う必要がある。また、約0.384は画像からの概算なので、発表資料へ載せる前に元の`metrics.csv`で正確な値を確認する。

## 8. 生成元ファイル

| ファイル | 用途 | 状態 |
| --- | --- | --- |
| 共有画像 `Validation mAP` | epochごとのvalidation mAP | 確認済み |
| 共有画像 `Train / Validation Loss` | epochごとのTrain/Validation Loss | 確認済み |
| 画像生成元の`metrics.csv` | 正確なepoch別指標 | ローカル未同期 |
| `experiments/riku-lab/run.py` | 現在の実験実行コード | ローカルに存在 |
| `experiments/riku-lab/model.py` | ConvNeXt-Tinyと学習範囲 | ローカルに存在 |
| `experiments/baseline/analysis/overall_model_metrics.csv` | baselineの3 seed集計 | ローカルに存在 |

### 8.1 不足しているファイル

- 共有画像を生成した実行の`metrics.csv`
- 同じ実行の`config.json`
- Macro F1、Samples F1、Hamming Lossの推移
- ジャンル別評価結果
- 複数seedの実行結果

これらを同期できれば、概算値を実測値へ置き換え、採用判断を最終確定できる。
