# 共有用ベースラインレポート

作成日: 2026-06-13

このレポートは、機械学習に詳しくない情報系の学部生でも読めるように、今回のベースラインが何をしているか、どのように学習されているか、評価指標が何を意味するかを説明するための資料である。

## まず結論

今回のベースラインは、アニメのカバー画像から 19 種類のジャンルを予測する画像分類モデルである。1 作品に複数ジャンルが付くため、「1 つだけ選ぶ分類」ではなく、「19 ジャンルそれぞれについて付くか付かないかを判定する分類」として扱っている。

保存されている checkpoint は、100 epoch の学習中で validation loss が最も小さかった epoch 8 のモデルである。`run_baseline.py` では、validation loss が過去最小を更新したときだけ `resnet18_best.pth` を上書き保存しているため、学習終了時点では epoch 8 の重みが残っている。

test split での主な結果は次の通り。

| 評価方法 | Macro F1 | Samples F1 | Hamming Loss | mAP |
| --- | ---: | ---: | ---: | ---: |
| モデル、しきい値 0.5 | 0.1133 | 0.3240 | 0.1180 | 0.2875 |
| モデル、validation で調整したしきい値 | 0.3252 | 0.4318 | 0.2178 | 0.2875 |

0.5 固定しきい値では、モデルはかなり慎重で、1 作品あたり平均 0.857 個のジャンルしか予測していない。実データでは 1 作品あたり平均で約 2.46 個のジャンルが付いているため、現状は「ジャンルを出し渋る」モデルになっている。

## 今回のタスク

入力はアニメ作品のカバー画像である。出力は次の 19 ジャンルそれぞれについて、その作品に当てはまるかどうかである。

`Action`, `Adventure`, `Comedy`, `Drama`, `Ecchi`, `Fantasy`, `Hentai`, `Horror`, `Mahou Shoujo`, `Mecha`, `Music`, `Mystery`, `Psychological`, `Romance`, `Sci-Fi`, `Slice of Life`, `Sports`, `Supernatural`, `Thriller`

例えば、ある作品の正解ラベルが `Action`, `Fantasy`, `Adventure` だった場合、モデルは 19 個のジャンル全部に対してスコアを出し、そのうち該当しそうなジャンルを選ぶ。

このように、1 つの入力に対して複数の正解ラベルがあり得る分類を **マルチラベル分類** と呼ぶ。犬・猫・鳥のどれか 1 つだけを選ぶような分類は **マルチクラス分類** であり、今回の問題とは少し違う。

## データ分割

データは train / validation / test に分かれている。

| split | 役割 | 件数 |
| --- | --- | ---: |
| train | モデルの重みを学習するために使う | 8,957 |
| validation | 学習中に、モデルの良し悪しを確認するために使う | 1,121 |
| test | 最後に、未知データへの性能を見るために使う | 1,121 |

今回は AniList の relations を使って、同じシリーズの作品が train / validation / test にまたがらないように分割している。例えば、同じシリーズの第 1 期が train、第 2 期が test に入ると、画像の雰囲気やキャラクターが似ているため、実力より良く見える可能性がある。これを避けるために、シリーズ単位で分割している。

## ベースラインモデル

モデルには ResNet18 を使っている。ResNet18 は画像分類でよく使われるニューラルネットワークで、画像から特徴を取り出す部分を持っている。

今回の設定は次の通り。

| 項目 | 内容 |
| --- | --- |
| モデル | ResNet18 |
| 事前学習 | 使っていない。`weights=None` でスクラッチ学習 |
| 入力画像サイズ | 224 x 224 |
| 出力 | 19 ジャンル分のスコア |
| 損失関数 | `BCEWithLogitsLoss` |
| optimizer | Adam |
| epoch 数 | 100 |
| checkpoint | validation loss が過去最小を更新したら保存 |

「スクラッチ学習」とは、最初から今回のデータだけで学習するという意味である。ImageNet などで事前に学習された重みは使っていない。そのため、データ量に対して学習が難しくなりやすい。

## 学習の流れ

学習はおおまかに次の手順で行われる。

1. train 用 CSV からアニメ ID、画像 URL、19 ジャンルの 0/1 ラベルを読む。
2. 画像を `data/images/{ID}.jpg` から読み込む。なければ URL から取得して保存する。
3. 画像を 224 x 224 にリサイズし、モデルに入れやすい形式に変換する。
4. ResNet18 に画像を入力し、19 個のスコアを出す。
5. 正解ラベルと予測スコアを `BCEWithLogitsLoss` で比較する。
6. Adam optimizer でモデルの重みを更新する。
7. 1 epoch ごとに validation データで性能を測る。
8. validation loss がこれまでで最も小さければ、`resnet18_best.pth` に checkpoint を保存する。

ここで重要なのは、モデルが最終 epoch の重みを使っているわけではない、という点である。保存されるのは validation loss が最も良かった時点の重みである。今回のログでは validation loss の最小値は epoch 8 だったため、保存済み checkpoint は epoch 8 のモデルである。

## 損失関数の考え方

学習では `BCEWithLogitsLoss` という損失関数を使っている。損失関数とは、モデルの予測が正解からどれくらい外れているかを数値化する関数である。値が小さいほど、モデルの予測が正解に近いと考える。

今回のようなマルチラベル分類では、19 ジャンルをまとめて 1 つのクラスとして扱うのではなく、各ジャンルについて個別に「あり / なし」を判定する。例えば `Action` についてありかなし、`Comedy` についてありかなし、という 2 値分類を 19 個同時に行っている。

`BCEWithLogitsLoss` は、このような「あり / なし」の判定を複数まとめて学習するのに使いやすい損失関数である。`logits` は sigmoid を通す前の生のスコアを意味する。PyTorch の `BCEWithLogitsLoss` は内部で sigmoid 相当の処理も含めて安定に計算するため、モデル側では sigmoid を最後に付けず、生スコアをそのまま出している。

## 予測スコアとしきい値

モデルは各ジャンルについて「このジャンルらしさ」のスコアを出す。最終的にジャンルを付けるには、そのスコアを 0/1 に変換する必要がある。

現行の標準評価では、しきい値を 0.5 に固定している。つまり、あるジャンルの予測確率が 0.5 以上なら「そのジャンルあり」、0.5 未満なら「そのジャンルなし」と判定する。

ただし今回のモデルでは、0.5 固定だと予測ジャンル数が少なすぎる。test では 1 作品あたり平均 0.857 個しかジャンルを出していない。一方、実データの平均ジャンル数は約 2.46 個である。このため、多くのジャンルを見逃している。

validation データを使ってジャンルごとにしきい値を調整すると、Macro F1 と Samples F1 は上がる。しかし、予測ジャンル数が 4.685 個まで増え、Hamming Loss は悪化する。つまり、しきい値を下げれば見逃しは減るが、余計なジャンルを付ける誤りが増える。

## 評価指標の説明

この節では、今回使っている評価指標を順番に説明する。

### 記号

数式では、次の記号を使う。

| 記号 | 意味 |
| --- | --- |
| $N$ | 作品数 |
| $C$ | ジャンル数。今回は 19 |
| $i$ | 作品の番号 |
| $c$ | ジャンルの番号 |
| $y_{i,c}$ | 正解ラベル。作品 $i$ にジャンル $c$ が付いていれば 1、なければ 0 |
| $\hat{y}_{i,c}$ | 予測ラベル。モデルがジャンル $c$ を「あり」と判定すれば 1、なければ 0 |
| $p_{i,c}$ | モデルが出したジャンル $c$ の予測確率 |

また、ある 1 つのジャンルに注目したとき、予測結果は次の 4 種類に分けられる。

| 記号 | 意味 |
| --- | --- |
| TP | True Positive。正解も 1、予測も 1 |
| FP | False Positive。正解は 0、予測は 1 |
| FN | False Negative。正解は 1、予測は 0 |
| TN | True Negative。正解も 0、予測も 0 |

Precision, Recall, F1 は主に TP, FP, FN から計算する。TN は、Hamming Loss のように全ラベルの正誤を見る指標では効くが、F1 には直接入らない。

### Loss

Loss は、学習中にモデルの予測が正解からどれくらい外れているかを見る値である。小さいほど良い。

今回の 1 作品・1 ジャンルに対する binary cross entropy は、予測確率を $p$、正解を $y$ とすると次の形になる。

$$
\mathrm{BCE}(y, p) = -\{y \log p + (1-y)\log(1-p)\}
$$

正解が 1 のときに $p$ が小さい、または正解が 0 のときに $p$ が大きいと、loss は大きくなる。実際の `BCEWithLogitsLoss` は、確率 $p$ ではなく sigmoid 前の logit を受け取り、数値的に安定する形で同じ目的の計算を行う。

今回の checkpoint は validation loss が最も小さい epoch 8 で保存されている。train loss は epoch が進むほど下がっているが、validation loss は途中から悪化している。これは、train データには合わせ込めているが、未知に近い validation データにはうまく一般化できていないことを示している。

ただし loss は F1 や mAP と完全に同じ目的の指標ではない。loss が最小の epoch と F1 が最大の epoch がずれることは普通にある。今回も validation loss の最小は epoch 8 だが、Macro F1 の最大はもっと後の epoch 34 である。

### Precision

Precision は、「モデルがこのジャンルだと予測したもののうち、実際に正解だった割合」である。

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

例えば `Action` と予測した作品が 100 件あり、そのうち本当に `Action` だったものが 60 件なら、Precision は 0.60 である。

Precision が高いモデルは、ジャンルを付けるときの誤爆が少ない。

### Recall

Recall は、「実際にそのジャンルである作品のうち、モデルが拾えた割合」である。

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

例えば本当は `Action` の作品が 100 件あり、そのうちモデルが `Action` と予測できたものが 40 件なら、Recall は 0.40 である。

Recall が高いモデルは、見逃しが少ない。

### F1

F1 は Precision と Recall のバランスを見る指標である。Precision だけ高くても Recall が低いと、ほとんど予測しないモデルになってしまう。逆に Recall だけ高くても Precision が低いと、何でもかんでもジャンルを付けるモデルになってしまう。

$$
\mathrm{F1} =
\frac{2 \times \mathrm{Precision} \times \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$

F1 はこの 2 つのバランスを 1 つの値で見るために使う。最大値は 1.0 で、高いほど良い。

### Macro F1

Macro F1 は、まずジャンルごとに F1 を計算し、その 19 ジャンル分を単純平均した値である。

$$
\mathrm{Macro\ F1} =
\frac{1}{C}\sum_{c=1}^{C}\mathrm{F1}_c
$$

重要なのは、データ数の多い `Comedy` も、データ数の少ない `Thriller` も、同じ重みで平均されるという点である。そのため、少数ジャンルを全然当てられないと Macro F1 は低くなる。

今回の test Macro F1 は 0.1133 で低い。これは、`Comedy`, `Action`, `Hentai` など一部のジャンルは予測できているが、多くのジャンルではほとんど陽性予測できていないためである。

### Samples F1

Samples F1 は、作品ごとに「予測したジャンル集合」と「正解ジャンル集合」がどれくらい重なっているかを見る指標である。

作品 $i$ について、正解ジャンル集合を $Y_i$、予測ジャンル集合を $\hat{Y}_i$ とすると、作品ごとの F1 は次のように書ける。

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

例えば、ある作品の正解が `Action`, `Fantasy`, `Adventure` で、モデルの予測が `Action`, `Comedy` だった場合、`Action` は合っているが、`Fantasy` と `Adventure` は見逃しており、`Comedy` は余計に付けている。このような作品ごとの一致度を平均したものが Samples F1 である。

Samples F1 は、ユーザー目線で「この作品に付いたジャンルのセットがどれくらいそれっぽいか」を見るのに近い。

### Hamming Loss

Hamming Loss は、全作品と全ジャンルの組み合わせを 1 個ずつ見て、0/1 判定をどれくらい間違えたかを見る指標である。

$$
\mathrm{Hamming\ Loss} =
\frac{1}{NC}
\sum_{i=1}^{N}\sum_{c=1}^{C}
\mathbf{1}(y_{i,c} \ne \hat{y}_{i,c})
$$

$\mathbf{1}(\cdot)$ は、条件が正しければ 1、そうでなければ 0 になる関数である。つまり、全作品・全ジャンルの 0/1 判定のうち、間違えた割合を数えている。

今回の test は 1,121 作品、ジャンルは 19 個なので、合計で 1,121 x 19 個の 0/1 判定がある。このうち何割を間違えたかが Hamming Loss である。小さいほど良い。

ただし、今回のように 1 作品あたり平均 2.46 ジャンルしか付かない場合、多くのラベルは 0 である。そのため、何もジャンルを予測しないモデルでも Hamming Loss がそこそこ低く見えてしまうことがある。実際、`always none` ベースラインの Hamming Loss は 0.1294 で、モデルの 0.1180 と大きく離れていない。

したがって、Hamming Loss だけを見るのは危険である。

### mAP

mAP は mean Average Precision の略である。ざっくり言うと、「正解ジャンルの作品を、モデルがどれくらい上位に並べられているか」を見る指標である。

1 つのジャンル $c$ について、モデルのスコアが高い順に作品を並べる。そのランキングの上から $k$ 番目までを見たときの Precision を $P_c(k)$、$k$ 番目の作品が正解なら 1、そうでなければ 0 になる値を $\mathrm{rel}_c(k)$ とする。このとき AP は次の形で書ける。

$$
\mathrm{AP}_c =
\frac{1}{\text{ジャンル }c\text{ の正解数}}
\sum_{k=1}^{N} P_c(k)\mathrm{rel}_c(k)
$$

mAP は、ジャンルごとの AP を平均した値である。

$$
\mathrm{mAP} =
\frac{1}{C}\sum_{c=1}^{C}\mathrm{AP}_c
$$

F1 や Hamming Loss は、しきい値 0.5 などで 0/1 に変換した後の結果を見る。一方、mAP はしきい値を決める前のスコアの並び方を見る。

例えば `Action` について、モデルが `Action` らしい作品に高いスコアを付け、`Action` でない作品に低いスコアを付けられていれば、AP は高くなる。これを 19 ジャンルで平均したものが mAP である。

今回の test mAP は 0.2875 である。単純ベースラインの mAP はおよそ 0.129 前後なので、モデルは少なくともランダムや頻度固定よりは意味のある順位情報を持っている。

## 現在の結果

best checkpoint、つまり validation loss 最小 epoch 8 のモデルを test で評価した結果は次の通りである。

| split | しきい値 | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5 固定 | 0.1024 | 0.3106 | 0.1193 | 0.2864 | 0.847 |
| test | 0.5 固定 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| test | validation 最適しきい値 | 0.3252 | 0.4318 | 0.2178 | 0.2875 | 4.685 |

validation と test の mAP がほぼ同じなので、順位付けの性能は validation と test で大きくは変わっていない。一方で、0.5 固定しきい値では予測ジャンル数が少なすぎる。

ジャンル別に見ると、test では次のジャンルが比較的よく当たっている。

ここで `support` は、そのジャンルが正解として付いている作品数を表す。`predicted positive` は、モデルがそのジャンルを「あり」と予測した作品数を表す。

| genre | support | predicted positive | Precision | Recall | F1 | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hentai | 144 | 84 | 0.9048 | 0.5278 | 0.6667 | 0.8066 |
| Comedy | 485 | 586 | 0.6075 | 0.7340 | 0.6648 | 0.6872 |
| Action | 326 | 221 | 0.6063 | 0.4110 | 0.4899 | 0.5933 |

一方で、しきい値 0.5 では 19 ジャンル中 10 ジャンルで陽性予測が 0 件だった。つまり、そのジャンルを 1 回も予測していない。

陽性予測が 0 件だったジャンルは、`Drama`, `Ecchi`, `Horror`, `Mecha`, `Music`, `Mystery`, `Psychological`, `Sports`, `Supernatural`, `Thriller` である。

この結果から、現状のモデルは `Comedy`, `Action`, `Hentai` などの一部のジャンルには反応できるが、多くのジャンルを見逃していることが分かる。

## 学習曲線から分かること

![Learning curves](analysis/learning_curves.png)

train loss は学習が進むほど下がり続けている。一方で、validation loss は epoch 8 で最小になり、その後は悪化している。

これは、モデルが train データにはどんどん合わせ込めているが、validation データへの性能は良くなっていないことを意味する。このような状態を **過学習** と呼ぶ。

そのため、今回の checkpoint 保存方法は妥当である。最終 epoch 100 のモデルではなく、validation loss が最も良かった epoch 8 のモデルを保存しているからである。

## 単純ベースラインとは何か

単純ベースラインは、「機械学習モデルを使わない、とても単純な予測方法」である。これと比較することで、ResNet18 が本当に意味のある情報を学習しているのかを確認できる。

今回比較した単純ベースラインは次の 4 つである。

### always none

全作品に対して、どのジャンルも付けない方法である。

これは明らかに役に立たないが、Hamming Loss では意外と悪く見えない。なぜなら、19 ジャンルのうち多くは 0 なので、「全部なし」と言うだけでも、多くの 0 は当たるからである。

### always top 2 train genres

train データで最も多かった 2 ジャンルを、全作品に必ず付ける方法である。

今回の train データでは、上位 2 ジャンルは `Comedy` と `Action` である。つまり、どんな画像が来ても `Comedy` と `Action` を予測する。

### always top 3 train genres

train データで最も多かった 3 ジャンルを、全作品に必ず付ける方法である。

今回の上位 3 ジャンルは `Comedy`, `Action`, `Fantasy` である。画像はまったく見ず、全作品にこの 3 つを付ける。

### Bernoulli by train prevalence

train データでのジャンル出現率に従って、ランダムにジャンルを付ける方法である。

例えば train データで `Comedy` が約 43% の作品に付いているなら、新しい作品にも約 43% の確率で `Comedy` を付ける。`Thriller` のように少ないジャンルは低い確率で付ける。

これは画像を見ないランダム予測だが、少数ジャンルもたまに予測するため、Macro F1 だけは少し高く出ることがある。

## 単純ベースラインとの比較

test split で比較した結果は次の通りである。

| method | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, threshold 0.5 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| always none | 0.0000 | 0.0000 | 0.1294 | 0.1294 | 0.000 |
| always top 2 train genres | 0.0555 | 0.3151 | 0.1585 | 0.1294 | 2.000 |
| always top 3 train genres | 0.0760 | 0.3341 | 0.1857 | 0.1294 | 3.000 |
| Bernoulli by train prevalence | 0.1292 | 0.1876 | 0.2045 | 0.1301 | 2.478 |

この表から分かることは次の通りである。

- モデルは mAP で単純ベースラインを大きく上回っている。つまり、画像から何らかの有用な順位情報を学習している。
- モデルは Hamming Loss でも単純ベースラインより良い。
- `always top 3 train genres` は Samples F1 だけモデルを少し上回るが、これは頻出ジャンルを常に多めに付けて部分一致を稼いでいるためである。Hamming Loss と mAP はモデルより悪い。
- Bernoulli baseline は Macro F1 だけモデルより高いが、これはランダムに少数ジャンルも出すためであり、Samples F1、Hamming Loss、mAP は悪い。

したがって、ResNet18 モデルは単純ベースラインより意味のある情報を学習している。ただし、0.5 しきい値で最終的なジャンルを選ぶ方法が弱く、少数ジャンルをかなり見逃している。

## 現状の課題

現状の主な課題は次の通りである。

1. 0.5 固定しきい値では、予測ジャンル数が少なすぎる。
2. 多くの少数ジャンルで Recall が 0 に近い。
3. ResNet18 を事前学習なしで学習しているため、画像特徴を十分に学びにくい。
4. train loss は下がるが validation loss は悪化しており、過学習している。
5. クラス不均衡への対策がまだ入っていない。

## 次に試す価値があること

次に試すなら、優先度が高いのは次の改善である。

1. ImageNet 事前学習済み ResNet18 または ResNet50 を使う。
2. `pos_weight` や class-balanced loss を使って、少数ジャンルを学習しやすくする。
3. 画像拡張を入れて、過学習を減らす。
4. validation loss または mAP を見て early stopping する。
5. 0.5 固定ではなく、ジャンル別しきい値や top-k 方式を検討する。
6. ジャンル別の AP / F1 を毎回出力し、どのジャンルが改善したかを確認する。

## ファイル

関連ファイルは次の通り。

| ファイル | 内容 |
| --- | --- |
| `src/baseline_resnet/run_baseline.py` | ベースライン学習の入口 |
| `src/baseline_resnet/model.py` | ResNet18 モデル定義 |
| `src/baseline_resnet/train.py` | 1 epoch の学習処理 |
| `src/baseline_resnet/evaluate.py` | 評価指標の計算 |
| `src/baseline_resnet/model/resnet18_best.pth` | validation loss 最小 epoch 8 の checkpoint |
| `src/baseline_resnet/model/baseline_full_metrics.csv` | 100 epoch 分の学習ログ |
| `playground/kazusa/baseline/analyze_baseline.py` | 追加評価スクリプト |
| `playground/kazusa/baseline/analysis/` | 追加評価の CSV、JSON、学習曲線 |

## 再現コマンド

```bash
# ベースライン学習
uv run python src/baseline_resnet/run_baseline.py

# 追加評価
.venv/bin/python playground/kazusa/baseline/analyze_baseline.py
```
