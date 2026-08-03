# takato最終モデルレポート

作成日: 2026-07-04

## 技術サマリ

この実験の目的は、今までの実験の成果からもっとも精度の良いモデルを作成することである。
モデルと学習方法は[General Multi-label Image Classification with Transformers](https://arxiv.org/abs/2011.14027)と[著者の公式実装](https://github.com/QData/C-Tran)を参考にした。
具体的にはImageNet-1K事前学習済みResNet-101,transformer,Adam,576px入力,データ拡張,ReduceLROnPlateauを組み合わせている。

## 1. 目的と比較の位置づけ

### 1.1 実験目的

今までの実験の成果からもっとも精度の良いモデルを作成することである。比較対象はkazusa-baselineとする

### 1.2 評価条件

| 項目 | 定義 |
| --- | --- |
| 予測対象 | 1作品につき19ジャンルの有無 |
| 主評価 split | validation、1,121作品 |
| 主評価指標 | クラス別 Average Precision の単純平均である mAP |
| 補助指標 | Macro F1、Samples F1、Hamming Loss、ジャンル別 AP / Precision / Recall / F1 |
| ラベル決定 | sigmoid確率 0.5以上を陽性 |
| seed | 42、43、44 |
| 主比較 | `kazusa-baseline` |
| testの扱い | モデル構成を確定するまで未使用 |

mAPは各ジャンル内で正例を上位に並べる能力を測り、0.5というしきい値には依存しない。F1とHamming Lossは0.5で二値化した結果である。このため、本レポートではmAPをモデル比較の主指標とし、F1とHamming Lossはラベル決定の性質を確認する補助指標として扱う。

## 2. 実装したモデル

### 2.1 モデルと学習設定

| 項目 | 設定 |
| --- | --- |
| backbone | ResNet-101 |
| 事前学習 | ImageNet1K |
| transoformer | 3層、4ヘッド |
| 出力層 | biasあり19次元線形層 |
| 入力解像度 | 576 × 576 |
| train augmentation | RandomCrop + RandomHorizontalFlip(640pxへresize後、複数サイズからRandomCropし、576pxへ再resizeしてから水平反転) |
| validation前処理 | 576 × 576へのresize |
| 損失 | BCEWithLogitsLoss (pos_weight=0.5,) |
| optimizer | Adam、learning rate 1e-5 weight_decay 0.0004 |
| scheduler | ReduceLROnPlateau |
| batch size | 32 |
| 最大epoch | 100 |
| early stopping | map、patience 5、min_delta=1e-4 |

### 2.2 論文・著者実装との対応と相違点

| 項目 | 論文 / 公開train code | 本実装 |
| --- | --- | --- |
| loss | cross entropy loss | BCEWithLogitsLoss(pos_weight=0.5) |
| batch処理 | 16 | 32 |
| データ・クラス数 | MS-COCOなど | アニメカバー画像、19ジャンル |

## 3. 全体結果

### 3.1 kazusa-baselineとの比較

| 実験 | validation mAP | Macro F1 | Samples F1 | Hamming Loss |
| --- | ---: | ---: | ---: | ---: |
| `kazusa-baseline` | 0.4392 ± 0.0058 | 0.4220 ± 0.0093 | **0.5504 ± 0.0038** | 0.1374 ± 0.0017 |
| 本モデル（C-Tran v2） | **0.4431 ± 0.0046** | **0.4338 ± 0.0099** | 0.5333 ± 0.0087 | **0.1363 ± 0.0056** |

本モデルは `kazusa-baseline` と比べて、主評価指標のvalidation mAPが0.0039、Macro F1が0.0118高かった。また、Hamming Lossは0.0011低かった。一方、Samples F1は0.0172低かった。

### 3.2 seed間の再現性

| seed | best epoch | 実行epoch数 | validation mAP | Macro F1 | Samples F1 | Hamming Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 11 | 16 | 0.4402 | 0.4237 | 0.5233 | 0.1343 |
| 43 | 10 | 15 | 0.4485 | 0.4436 | 0.5374 | 0.1321 |
| 44 | 6 | 11 | 0.4407 | 0.4342 | 0.5391 | 0.1426 |

mAPの範囲は0.4402–0.4485で、seed間の最大差は0.0082だった。seed 43がmAPとMacro F1で最も高く、seed 44がSamples F1で最も高かった。一方、Hamming Lossはseed 43が最も低く、seed 44が最も高かった。3 seedのmAPは近い性能帯に収まっているが、best epochは6–11と差があるため、今後の比較でも複数seedの平均とばらつきを確認する必要がある。


## 4. ジャンル別分析

### 4.1 APは19ジャンル中7ジャンルで改善

| ジャンル | support | `kazusa-baseline` AP | 本モデル AP | AP差 | 本モデル F1 | 本モデル Precision | 本モデル Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mahou Shoujo | 22 | 0.1406 | **0.2649** | **+0.1243** | 0.2711 | 0.2703 | 0.2727 |
| Thriller | 18 | 0.0332 | **0.1230** | **+0.0898** | 0.0899 | 0.0778 | 0.1111 |
| Supernatural | 133 | 0.2632 | **0.3070** | **+0.0438** | 0.3132 | 0.2987 | 0.3308 |
| Mystery | 70 | 0.1651 | **0.2057** | **+0.0406** | 0.2561 | 0.2189 | 0.3095 |
| Comedy | 487 | 0.7479 | **0.7647** | **+0.0168** | 0.7011 | 0.6767 | 0.7303 |
| Slice of Life | 195 | 0.4516 | **0.4573** | **+0.0057** | 0.4823 | 0.4071 | 0.6000 |
| Ecchi | 80 | 0.3184 | **0.3212** | **+0.0028** | 0.3286 | 0.2743 | 0.4167 |
| Romance | 167 | **0.4558** | 0.4490 | -0.0067 | 0.4572 | 0.3888 | 0.5569 |
| Hentai | 137 | **0.9353** | 0.9240 | -0.0113 | 0.8392 | 0.7804 | 0.9075 |
| Adventure | 175 | **0.4430** | 0.4316 | -0.0114 | 0.4421 | 0.3881 | 0.5219 |
| Mecha | 49 | **0.6876** | 0.6758 | -0.0118 | 0.6201 | 0.5453 | 0.7211 |
| Psychological | 54 | **0.1440** | 0.1319 | -0.0121 | 0.1848 | 0.1685 | 0.2099 |
| Sci-Fi | 157 | **0.4962** | 0.4833 | -0.0129 | 0.4802 | 0.4576 | 0.5053 |
| Music | 58 | **0.5191** | 0.5057 | -0.0134 | 0.4752 | 0.4167 | 0.5690 |
| Drama | 222 | **0.3974** | 0.3790 | -0.0184 | 0.4074 | 0.3607 | 0.4700 |
| Fantasy | 274 | **0.6126** | 0.5854 | -0.0272 | 0.5712 | 0.5440 | 0.6046 |
| Action | 305 | **0.6504** | 0.6214 | -0.0289 | 0.6244 | 0.5864 | 0.6699 |
| Horror | 39 | **0.1659** | 0.1256 | -0.0403 | 0.1609 | 0.1620 | 0.1624 |
| Sports | 61 | **0.7177** | 0.6638 | -0.0539 | 0.5437 | 0.4677 | 0.6776 |

本モデルは19ジャンル中7ジャンルで `kazusa-baseline` より高いAPを示した。改善幅が大きかったのは `Mahou Shoujo`（+0.1243）、`Thriller`（+0.0898）、`Supernatural`（+0.0438）、`Mystery`（+0.0406）である。一方、`Sports`（-0.0539）、`Horror`（-0.0403）、`Action`（-0.0289）、`Fantasy`（-0.0272）ではAPが低下した。

改善したジャンル数は7に留まるが、改善幅の大きいジャンルがあるため、19ジャンルの単純平均であるmAPは `kazusa-baseline` の0.4392から本モデルの0.4431へわずかに上昇している。ただし、最大の改善が見られた `Mahou Shoujo` と `Thriller` のsupportはそれぞれ22件と18件であり、少数の予測順位の変化によってAPが大きく変動しやすい。このため、これらの改善が安定して再現されるかは別splitでも確認する必要がある。

本モデルの値は、`ctran_v2_best_model_genre_ap.csv`、`seed43_ctran_v2_best_model_genre_ap.csv`、`seed44_ctran_v2_best_model_genre_ap.csv` の3 seed平均である。F1は各seedのPrecisionとRecallから算出した後に平均した。

## 5. 再現方法

```bash
uv run python experiments/takato-competitions/exp_run.py
uv run python experiments/takato-competitions/seed43_exp_run.py
uv run python experiments/takato-competitions/seed44_exp_run.py
uv run python experiments/takato-competitions/analyze.py --checkpoint experiments/takato-competitions/outputs/seed42_best_model.pth
uv run python experiments/takato-competitions/analyze.py --checkpoint experiments/takato-competitions/outputs/seed43_best_model.pth
uv run python experiments/takato-competitions/analyze.py --checkpoint experiments/takato-competitions/outputs/seed44_best_model.pth
```
