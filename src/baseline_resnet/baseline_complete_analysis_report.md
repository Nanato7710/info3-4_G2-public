# ResNet18 ベースライン完全詳細分析レポート

作成日: 2026-06-13

このレポートは、`src/baseline_resnet` の ResNet18 ベースラインについて、データセット、学習方法、評価指標、validation / test 結果、単純ベースライン比較、既知の課題、次の改善案を 1 つにまとめた完全詳細版である。

元になった追加調査は `playground/kazusa/` 配下で実施した。主な参照元は次の通りである。

| 参照元 | 内容 |
| --- | --- |
| `playground/kazusa/baseline/baseline_report.md` | ベースライン結果の技術寄り要約 |
| `playground/kazusa/baseline/baseline_report_for_members.md` | 評価指標と学習方法の初学者向け説明 |
| `playground/kazusa/series_split/dataset_report.md` | `data/series_split_outputs` を対象にしたデータセット分析 |
| `playground/kazusa/baseline/analysis/` | test 評価、ジャンル別指標、しきい値分析、単純ベースライン比較 |
| `playground/kazusa/series_split/analysis/` | dataset 図表、画像チェック、旧 split リーク比較 |

## 1. 要約

今回のベースラインは、アニメ作品のカバー画像から 19 種類のジャンルを同時に予測するマルチラベル画像分類モデルである。モデルは ResNet18 で、ImageNet 事前学習済み重みは使わず、`weights=None` でスクラッチ学習している。

現行コードは `data/series_split_outputs/` のシリーズ単位分割済み CSV を読み込む。同一シリーズが train / validation / test にまたがらないため、旧ランダム分割より評価は厳しく、未知シリーズへの汎化性能を見やすい。

保存済み checkpoint `src/baseline_resnet/model/resnet18_best.pth` は、100 epoch の最終モデルではない。`run_baseline.py` は validation loss が過去最小を更新したときだけ checkpoint を上書き保存するため、最終的に validation loss 最小の epoch 8 の重みが残っている。

test split における主な結果は次の通り。

| 評価設定 | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, threshold 0.5 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| model, validation 最適しきい値 | 0.3252 | 0.4318 | 0.2178 | 0.2875 | 4.685 |

結論として、モデルは単純ベースラインより有意な順位情報を学習している。特に mAP は単純ベースラインの約 0.129 に対して 0.2875 である。一方、0.5 固定しきい値では陽性予測が少なすぎる。19 ジャンル中 10 ジャンルでは test で陽性予測が 0 件であり、少数ジャンルをほとんど拾えていない。

## 2. タスク定義

入力はアニメ作品のカバー画像である。出力は 19 個のジャンルそれぞれについて「その作品に該当するかどうか」を表す 0/1 ラベルである。

対象ジャンルは次の 19 種類である。

`Action`, `Adventure`, `Comedy`, `Drama`, `Ecchi`, `Fantasy`, `Hentai`, `Horror`, `Mahou Shoujo`, `Mecha`, `Music`, `Mystery`, `Psychological`, `Romance`, `Sci-Fi`, `Slice of Life`, `Sports`, `Supernatural`, `Thriller`

1 作品には複数のジャンルが付く。例えば、ある作品が `Action`, `Fantasy`, `Adventure` を同時に持つことがある。このため、今回の問題は 1 つだけのクラスを選ぶマルチクラス分類ではなく、各ジャンルについて独立に 0/1 を判定するマルチラベル分類として扱う。

## 3. データセット

### 3.1 正式に使う split

現行のベースライン学習コード `src/preprocessing/dataset_utils.py` は、次の 3 ファイルを読み込む。

| split | ファイル |
| --- | --- |
| train | `data/series_split_outputs/training_data_grouped.csv` |
| validation | `data/series_split_outputs/validation_data_grouped.csv` |
| test | `data/series_split_outputs/test_data_grouped.csv` |

`playground/kazusa/series_split/outputs/` にも作業用出力があるが、現行ベースラインの正式な入力は `data/series_split_outputs/` である。

### 3.2 データ件数

`data/series_split_outputs/preprocessed_with_series_group.csv` には 11,199 件の作品がある。ID 重複はなく、`Title` と `ImageUrl` の欠損もない。

| 項目 | 値 |
| --- | ---: |
| 作品数 | 11,199 |
| ユニーク ID 数 | 11,199 |
| ジャンル数 | 19 |
| SeriesGroup 数 | 6,447 |
| 複数作品を含む SeriesGroup 数 | 1,748 |
| 最大 SeriesGroup サイズ | 67 |
| relation edges total | 14,804 |
| relation edges used for grouping | 10,502 |

split の行数は次の通りである。

| split | rows | row ratio | series groups | 平均ジャンル数 | ジャンル数中央値 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8,957 | 0.7998 | 5,153 | 2.478 | 2 |
| validation | 1,121 | 0.1001 | 617 | 2.411 | 2 |
| test | 1,121 | 0.1001 | 677 | 2.459 | 2 |

### 3.3 シリーズ単位分割

通常のランダム分割では、同じシリーズの作品が train と test に分かれることがある。これは画像分類では特に問題になる。同じシリーズではキャラクター、絵柄、ロゴ、構図、色使いが似やすいため、モデルがジャンルではなくシリーズ固有の見た目を覚えてしまっても、test で良いスコアが出る可能性がある。

このリークを避けるため、AniList relations を使って作品同士を結び、同じ系列と判断された作品を `SeriesGroup` にまとめている。そのうえで `SeriesGroup` 単位で train / validation / test に割り当てる。

実装上は、作品をノード、relation を edge とするグラフを作り、Union-Find で連結成分を求める。例えば A が B の `SEQUEL`、B が C の `SIDE_STORY` なら、A, B, C は同じ連結成分に入るため同じ `SeriesGroup` になる。

現在の `data/series_split_outputs` では、split 間の `SeriesGroup` 重複は 0 件である。

| 比較 | 重複 SeriesGroup 数 |
| --- | ---: |
| train vs validation | 0 |
| train vs test | 0 |
| validation vs test | 0 |

一方、旧 split である `data/training_data.csv`, `data/validation_data.csv`, `data/test_data.csv` に現在の SeriesGroup を対応付けると、多数のリークが見つかる。

| 比較 | 重複 SeriesGroup 数 | 左 split の重複行数 | 右 split の重複行数 | 合計重複行数 |
| --- | ---: | ---: | ---: | ---: |
| train vs validation | 502 | 1,961 | 624 | 2,585 |
| train vs test | 468 | 2,020 | 588 | 2,608 |
| validation vs test | 169 | 252 | 247 | 499 |

旧 split では、ポケモン、プリキュア、アイドルマスター、Fate、ラブライブ、NARUTO などの大きなシリーズが複数 split にまたがっていた。したがって、今回のシリーズ単位分割は評価の信頼性を上げるために重要である。

### 3.4 ジャンル不均衡

ジャンルごとの件数は大きく偏っている。最多の `Comedy` は 4,820 件、最少の `Thriller` は 189 件であり、約 25.5 倍の差がある。

| genre | total | 全体割合 | train | validation | test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comedy | 4,820 | 43.04% | 3,848 | 487 | 485 |
| Action | 3,083 | 27.53% | 2,452 | 305 | 326 |
| Fantasy | 2,652 | 23.68% | 2,107 | 274 | 271 |
| Drama | 2,114 | 18.88% | 1,689 | 222 | 203 |
| Slice of Life | 2,043 | 18.24% | 1,629 | 195 | 219 |
| Romance | 1,940 | 17.32% | 1,574 | 167 | 199 |
| Adventure | 1,878 | 16.77% | 1,530 | 175 | 173 |
| Sci-Fi | 1,759 | 15.71% | 1,414 | 157 | 188 |
| Hentai | 1,423 | 12.71% | 1,142 | 137 | 144 |
| Supernatural | 1,361 | 12.15% | 1,106 | 133 | 122 |
| Ecchi | 806 | 7.20% | 645 | 80 | 81 |
| Mystery | 718 | 6.41% | 591 | 70 | 57 |
| Mecha | 634 | 5.66% | 520 | 49 | 65 |
| Music | 582 | 5.20% | 479 | 58 | 45 |
| Sports | 551 | 4.92% | 437 | 61 | 53 |
| Psychological | 441 | 3.94% | 339 | 54 | 48 |
| Horror | 334 | 2.98% | 270 | 39 | 25 |
| Mahou Shoujo | 328 | 2.93% | 269 | 22 | 37 |
| Thriller | 189 | 1.69% | 156 | 18 | 15 |

この不均衡は Macro F1 を下げやすい。Macro F1 は各ジャンルを同じ重みで平均するため、少数ジャンルが 0 点に近いと全体値も低くなる。

### 3.5 画像キャッシュ品質

`data/series_split_outputs/preprocessed_with_series_group.csv` に含まれる 11,199 件について、`data/images/{ID}.jpg` の存在と読み込み可否を確認した。

| 項目 | 値 |
| --- | ---: |
| 期待画像数 | 11,199 |
| 存在する画像数 | 11,199 |
| 欠損画像数 | 0 |
| 読み込み可能画像数 | 11,199 |
| 読み込み不可画像数 | 0 |
| 画像フォーマット | JPEG 11,199 件 |
| 画像モード | RGB 11,199 件 |
| 幅の中央値 | 460 px |
| 高さの中央値 | 640 px |

少なくとも現時点では、画像欠損や破損によって学習・評価が失敗するリスクは低い。

## 4. モデル実装

### 4.1 モデル構造

モデル定義は `src/baseline_resnet/model.py` の `AnimeResNet` である。

| 項目 | 内容 |
| --- | --- |
| backbone | ResNet18 |
| torchvision 呼び出し | `models.resnet18(weights=None)` |
| 事前学習 | なし |
| 最終層 | `nn.Linear(num_ftrs, 19)` |
| 出力 | 19 次元 logits |

`weights=None` なので、ImageNet 事前学習済み重みは使っていない。したがって、画像特徴抽出も今回のアニメ画像データだけから学習する。

### 4.2 入力前処理

`src/baseline_resnet/run_baseline.py` では、画像に次の変換を適用する。

| 処理 | 内容 |
| --- | --- |
| Resize | 224 x 224 |
| ToTensor | PyTorch tensor へ変換 |
| Normalize | ImageNet mean / std |

Normalize には `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` を使っている。これは ResNet 系でよく使われる ImageNet 標準値である。ただし、今回のモデルは事前学習なしなので、この正規化が最適かどうかは今後検証余地がある。

### 4.3 学習設定

| 項目 | 値 |
| --- | --- |
| batch size | 64 |
| epochs | 100 |
| optimizer | Adam |
| learning rate | 1e-3 |
| loss | `BCEWithLogitsLoss` |
| device | MPS -> CUDA -> CPU の順に選択 |
| torch.compile | device が MPS でない場合に使用 |

現行の学習では、weight decay、learning rate scheduler、early stopping、画像データ拡張は使っていない。また、`train.py` には `calculate_pos_weights` が定義されているが、`run_baseline.py` の `criterion = nn.BCEWithLogitsLoss()` には `pos_weight` が渡されていない。したがって、クラス不均衡を損失関数側で補正していない。

### 4.4 checkpoint 保存

`run_baseline.py` では、各 epoch の validation loss が過去最小を下回ったときだけ `src/baseline_resnet/model/resnet18_best.pth` を保存する。

checkpoint が更新された epoch は次の通りである。

| epoch | Train Loss | Val Loss | Macro F1 | Samples F1 | Hamming Loss | mAP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3264 | 0.3257 | 0.0751 | 0.2081 | 0.1240 | 0.2330 |
| 2 | 0.3135 | 0.3227 | 0.0649 | 0.2563 | 0.1241 | 0.2402 |
| 4 | 0.3062 | 0.3080 | 0.0988 | 0.2697 | 0.1203 | 0.2627 |
| 7 | 0.2974 | 0.3067 | 0.1244 | 0.3136 | 0.1212 | 0.2757 |
| 8 | 0.2932 | 0.3021 | 0.1024 | 0.3106 | 0.1193 | 0.2864 |

最終的に epoch 8 が validation loss 最小であり、保存済み checkpoint は epoch 8 のモデルである。

## 5. 評価指標

### 5.1 記号

| 記号 | 意味 |
| --- | --- |
| $N$ | 作品数 |
| $C$ | ジャンル数。今回は 19 |
| $y_{i,c}$ | 作品 $i$ のジャンル $c$ の正解ラベル |
| $\hat{y}_{i,c}$ | 作品 $i$ のジャンル $c$ の予測ラベル |
| $p_{i,c}$ | 作品 $i$ のジャンル $c$ の予測確率 |
| TP | 正解 1、予測 1 |
| FP | 正解 0、予測 1 |
| FN | 正解 1、予測 0 |
| TN | 正解 0、予測 0 |

### 5.2 Loss

学習では `BCEWithLogitsLoss` を使う。1 つの 0/1 ラベルに対する binary cross entropy は次の形である。

$$
\mathrm{BCE}(y, p) = -\{y \log p + (1-y)\log(1-p)\}
$$

実際の `BCEWithLogitsLoss` は、確率 $p$ ではなく sigmoid 前の logit を受け取り、数値的に安定な形で同じ目的の計算を行う。

### 5.3 Precision, Recall, F1

Precision は、陽性予測のうち正しかった割合である。

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

Recall は、正解陽性のうち拾えた割合である。

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

F1 は Precision と Recall の調和平均である。

$$
\mathrm{F1} =
\frac{2 \times \mathrm{Precision} \times \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$

### 5.4 Macro F1

各ジャンルの F1 を単純平均する。

$$
\mathrm{Macro\ F1} =
\frac{1}{C}\sum_{c=1}^{C}\mathrm{F1}_c
$$

頻出ジャンルも少数ジャンルも同じ重みで平均されるため、少数ジャンルに厳しい指標である。

### 5.5 Samples F1

作品ごとに正解ジャンル集合と予測ジャンル集合の一致度を見る。作品 $i$ の正解集合を $Y_i$、予測集合を $\hat{Y}_i$ とすると、作品ごとの F1 は次である。

$$
\mathrm{F1}_i =
\frac{2|Y_i \cap \hat{Y}_i|}
{|Y_i| + |\hat{Y}_i|}
$$

Samples F1 はこれを全作品で平均する。

$$
\mathrm{Samples\ F1} =
\frac{1}{N}\sum_{i=1}^{N}\mathrm{F1}_i
$$

### 5.6 Hamming Loss

全作品・全ジャンルの 0/1 判定のうち、間違えた割合である。

$$
\mathrm{Hamming\ Loss} =
\frac{1}{NC}
\sum_{i=1}^{N}\sum_{c=1}^{C}
\mathbf{1}(y_{i,c} \ne \hat{y}_{i,c})
$$

小さいほど良い。ただし、今回のようにラベル 0 が多いタスクでは、何も予測しないモデルでも Hamming Loss が低く見えることがある。そのため、Hamming Loss 単独では判断しない。

### 5.7 mAP

mAP は mean Average Precision であり、しきい値で 0/1 にする前のスコアランキングの品質を見る。1 つのジャンル $c$ について、スコア順に並べた上位 $k$ までの Precision を $P_c(k)$、$k$ 番目の作品が正解なら 1 になる値を $\mathrm{rel}_c(k)$ とすると、AP は次のように書ける。

$$
\mathrm{AP}_c =
\frac{1}{\text{ジャンル }c\text{ の正解数}}
\sum_{k=1}^{N} P_c(k)\mathrm{rel}_c(k)
$$

mAP はジャンルごとの AP の平均である。

$$
\mathrm{mAP} =
\frac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_c
$$

F1 や Hamming Loss はしきい値後の 0/1 結果を見るが、mAP はスコアの順位情報を見る。そのため、しきい値が悪くても mAP にはモデルのランキング能力が表れる。

## 6. 学習ログ分析

保存済みの `src/baseline_resnet/model/baseline_full_metrics.csv` には 100 epoch 分の train / validation 指標がある。

代表 epoch は次の通りである。

| 観点 | Epoch | Train Loss | Val Loss | Macro F1 | Samples F1 | Hamming Loss | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 初回 | 1 | 0.3264 | 0.3257 | 0.0751 | 0.2081 | 0.1240 | 0.2330 |
| Val Loss 最小 | 8 | 0.2932 | 0.3021 | 0.1024 | 0.3106 | 0.1193 | 0.2864 |
| mAP 最大 | 11 | 0.2812 | 0.3118 | 0.1759 | 0.3531 | 0.1237 | 0.2868 |
| Samples F1 最大 | 28 | 0.0082 | 0.7534 | 0.2379 | 0.3906 | 0.1484 | 0.2552 |
| Macro F1 最大 | 34 | 0.0258 | 0.7961 | 0.2413 | 0.3470 | 0.1513 | 0.2507 |
| 最終 | 100 | 0.0007 | 1.0294 | 0.2241 | 0.3559 | 0.1373 | 0.2516 |

学習曲線は次の追加分析画像に保存されている。

![Learning curves](../../playground/kazusa/baseline/analysis/learning_curves.png)

train loss は 100 epoch にわたって下がり続ける。一方、validation loss は epoch 8 で最小になった後、長期的には悪化している。これは過学習を示している。したがって、現行の checkpoint 保存規則によって epoch 8 のモデルが残ることは妥当である。

ただし、validation loss 最小と F1 最大は一致していない。もし最終目的を F1 最大化に置くなら、checkpoint 選択基準を validation loss から Macro F1、Samples F1、mAP、または複合指標へ変更する選択肢もある。

## 7. Test 評価

保存済み checkpoint、つまり epoch 8 のモデルを test split で評価した結果は次の通りである。

| split | しきい値 | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5 固定 | 0.1024 | 0.3106 | 0.1193 | 0.2864 | 0.847 |
| test | 0.5 固定 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| validation | validation 最適 | 0.3473 | 0.4352 | 0.2147 | 0.2864 | 4.725 |
| test | validation 最適 | 0.3252 | 0.4318 | 0.2178 | 0.2875 | 4.685 |

validation と test の mAP はほぼ同じであり、スコア順位の品質は大きく崩れていない。一方で、0.5 固定しきい値では 1 作品あたり平均 0.857 ジャンルしか予測しない。実データの平均ジャンル数は test で 2.459 なので、かなり保守的である。

validation 上でジャンル別に F1 最大化しきい値を選ぶと、test Macro F1 は 0.1133 から 0.3252 に上がる。しかし、予測ジャンル数は 4.685 まで増え、Hamming Loss は 0.1180 から 0.2178 に悪化する。これは、Recall を増やした代わりに false positive が増えたことを意味する。

しきい値最適化の統計は次の通りである。

| 項目 | 値 |
| --- | ---: |
| 最小しきい値 | 0.05 |
| 中央値 | 0.19 |
| 最大しきい値 | 0.40 |

0.5 固定は高すぎる可能性が高いが、単純に validation F1 最適しきい値を採用すると陽性を出しすぎる。実運用や最終評価では、目的に応じて top-k、クラス別しきい値、または予測ジャンル数制約を設計する必要がある。

## 8. ジャンル別評価

test split、しきい値 0.5 で F1 が高いジャンルは次の通りである。

| genre | support | predicted positive | Precision | Recall | F1 | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hentai | 144 | 84 | 0.9048 | 0.5278 | 0.6667 | 0.8066 |
| Comedy | 485 | 586 | 0.6075 | 0.7340 | 0.6648 | 0.6872 |
| Action | 326 | 221 | 0.6063 | 0.4110 | 0.4899 | 0.5933 |
| Slice of Life | 219 | 48 | 0.5417 | 0.1187 | 0.1948 | 0.4043 |

`Hentai`, `Comedy`, `Action` は比較的よく識別できている。特に `Hentai` は Precision 0.9048、AP 0.8066 と高い。

一方で、しきい値 0.5 では次の 10 ジャンルで陽性予測が 0 件だった。

`Drama`, `Ecchi`, `Horror`, `Mecha`, `Music`, `Mystery`, `Psychological`, `Sports`, `Supernatural`, `Thriller`

このうち AP が完全に 0 ではないジャンルもある。つまり、モデルのスコア順位には多少の情報があるが、0.5 を超えるほど強いスコアが出ていない。例として、`Mecha` は AP 0.2371、`Ecchi` は AP 0.2228 だが、0.5 しきい値では陽性予測 0 件である。

AP が低いジャンルは次の通りである。

| genre | support | predicted positive | F1 | AP |
| --- | ---: | ---: | ---: | ---: |
| Thriller | 15 | 0 | 0.0000 | 0.0549 |
| Sports | 53 | 0 | 0.0000 | 0.0641 |
| Music | 45 | 0 | 0.0000 | 0.0898 |
| Horror | 25 | 0 | 0.0000 | 0.0972 |
| Psychological | 48 | 0 | 0.0000 | 0.1196 |
| Mystery | 57 | 0 | 0.0000 | 0.1247 |

これらは少数ジャンルが多く、学習データ数も評価データ数も少ない。今後の改善では、ジャンル不均衡対策とクラス別しきい値が重要になる。

## 9. 単純ベースライン比較

学習済みモデルが本当に画像から情報を得ているかを見るため、機械学習モデルを使わない単純ベースラインと比較した。

比較対象は次の 4 つである。

| method | 内容 |
| --- | --- |
| always none | すべての作品にジャンルを 1 つも付けない |
| always top 2 train genres | train で最頻の 2 ジャンル、`Comedy`, `Action` を全作品に付ける |
| always top 3 train genres | train で最頻の 3 ジャンル、`Comedy`, `Action`, `Fantasy` を全作品に付ける |
| Bernoulli by train prevalence | train のジャンル出現率に従ってランダムにジャンルを付ける |

test split 上の比較は次の通りである。

| method | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, threshold 0.5 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| always none | 0.0000 | 0.0000 | 0.1294 | 0.1294 | 0.000 |
| always top 2 train genres | 0.0555 | 0.3151 | 0.1585 | 0.1294 | 2.000 |
| always top 3 train genres | 0.0760 | 0.3341 | 0.1857 | 0.1294 | 3.000 |
| Bernoulli by train prevalence | 0.1292 | 0.1876 | 0.2045 | 0.1301 | 2.478 |

モデルは mAP と Hamming Loss で単純ベースラインを明確に上回る。Samples F1 は always top 3 が少し高いが、これは頻出ジャンルを広く付けることで部分一致を稼いでいるためであり、Hamming Loss と mAP は悪い。Bernoulli baseline は Macro F1 だけモデルを上回るが、ランダムに少数ジャンルも出すため Macro F1 が上がっているだけで、Samples F1、Hamming Loss、mAP は大きく劣る。

したがって、モデルは画像から何らかの有用なランキング情報を学習している。ただし、しきい値後のラベル選択が弱い。

## 10. 主要な解釈

### 10.1 モデルはランキング情報を持っている

mAP が単純ベースラインより高いため、モデルは正解ジャンルの作品に相対的に高いスコアを付ける能力をある程度持っている。完全なランダムや頻度固定ではない。

### 10.2 0.5 しきい値が保守的すぎる

test での予測ジャンル数は 0.857 個/作品である。一方、正解の平均ジャンル数は 2.459 個/作品である。したがって、多くの正解ジャンルを見逃している。

### 10.3 少数ジャンルが弱い

`Thriller`, `Horror`, `Music`, `Psychological`, `Sports`, `Mystery` などは AP も F1 も低い。これらはデータ件数が少なく、標準の BCE だけでは学習が難しい。

### 10.4 過学習している

train loss は 0.0007 付近まで下がるが、validation loss は epoch 8 以降に悪化する。データ拡張、正則化、early stopping、事前学習がないことが影響している可能性が高い。

### 10.5 checkpoint 基準と最終指標のずれ

checkpoint は validation loss 最小で保存される。一方、Macro F1 最大は epoch 34、Samples F1 最大は epoch 28、mAP 最大は epoch 11 である。どの指標を最終目的にするかによって、保存すべきモデルが変わる可能性がある。

## 11. 現状の限界

- 事前学習なし ResNet18 であり、データ量に対して画像特徴学習の負荷が大きい。
- `pos_weight` は実装されているが、現行学習では `BCEWithLogitsLoss` に渡していない。
- class-balanced loss、focal loss、weighted sampler などの不均衡対策がない。
- RandomResizedCrop、HorizontalFlip、ColorJitter などのデータ拡張がない。
- weight decay と learning rate scheduler がない。
- checkpoint は validation loss 基準であり、F1 や mAP 基準ではない。
- 0.5 固定しきい値では Recall が不足する。
- validation 最適しきい値は F1 を上げるが、Hamming Loss を悪化させる。
- test 評価は追加スクリプトによる後処理であり、学習スクリプト本体には組み込まれていない。
- seed 固定が明示されていないため、再学習時に完全再現できるとは限らない。

## 12. 改善案

優先度が高い改善は次の通りである。

1. ImageNet 事前学習済み ResNet18 / ResNet50 を使う  
   スクラッチ学習よりも少ないデータで安定しやすい。まず最も効果が見込める。

2. 不均衡対策を入れる  
   `calculate_pos_weights` を実際に `BCEWithLogitsLoss(pos_weight=...)` に渡す。あわせて focal loss や class-balanced loss も比較する。

3. データ拡張を入れる  
   RandomResizedCrop、HorizontalFlip、ColorJitter などを入れ、train loss だけが下がる過学習を抑える。

4. checkpoint 選択基準を比較する  
   validation loss、mAP、Macro F1、Samples F1 のどれで保存するのが良いか比較する。

5. ラベル選択戦略を設計する  
   0.5 固定、ジャンル別しきい値、top-k、予測ジャンル数制約、動的しきい値を比較する。

6. test 評価を正式パイプラインに入れる  
   `run_baseline.py` または別の `evaluate_checkpoint.py` として、best checkpoint の test 指標とジャンル別指標を毎回保存できるようにする。

7. ジャンル別の失敗分析を行う  
   AP が低いジャンル、F1 が 0 のジャンルについて、画像例と予測スコア分布を確認する。

## 13. 再現コマンド

学習:

```bash
uv run python src/baseline_resnet/run_baseline.py
```

ベースライン追加評価:

```bash
.venv/bin/python playground/kazusa/baseline/analyze_baseline.py
```

データセット追加分析:

```bash
.venv/bin/python playground/kazusa/series_split/analyze_dataset.py
```

## 14. 関連ファイル

| ファイル | 内容 |
| --- | --- |
| `src/baseline_resnet/run_baseline.py` | 学習パイプライン |
| `src/baseline_resnet/model.py` | ResNet18 モデル定義 |
| `src/baseline_resnet/train.py` | 1 epoch の学習処理 |
| `src/baseline_resnet/evaluate.py` | validation 指標計算 |
| `src/baseline_resnet/model/resnet18_best.pth` | validation loss 最小 epoch 8 の checkpoint |
| `src/baseline_resnet/model/baseline_full_metrics.csv` | 100 epoch の学習ログ |
| `data/series_split_outputs/` | 現行ベースラインが読む正式 split |
| `playground/kazusa/baseline/analyze_baseline.py` | best checkpoint の test 評価としきい値分析 |
| `playground/kazusa/series_split/analyze_dataset.py` | データセット分析、画像チェック、旧 split リーク比較 |

## 15. まとめ

このベースラインは、シリーズリークを避けたデータ分割上で、カバー画像から 19 ジャンルを予測するマルチラベル分類の最小実装として機能している。ResNet18 は単純ベースラインより良い mAP を出しており、画像から一定の情報を学習している。

しかし、現行設定はまだ実用的な分類器としては弱い。特に、少数ジャンルの Recall が低く、0.5 固定しきい値では多くのジャンルを 1 回も予測しない。さらに、100 epoch 学習では強い過学習が見られる。

次の段階では、事前学習済みモデル、不均衡対策、データ拡張、checkpoint 基準、しきい値戦略を系統的に比較するべきである。現在のレポートと追加分析成果物は、その比較実験の基準点として使える。
