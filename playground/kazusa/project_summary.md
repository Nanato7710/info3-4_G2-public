# プロジェクト概要

このプロジェクトは、AniList から取得したアニメ作品データとカバー画像を使い、画像からジャンルを推定するマルチラベル分類モデルを作るための実験環境である。

1 作品に複数ジャンルが付くため、通常の「1 クラスだけを選ぶ分類」ではなく、19 ジャンルそれぞれについて該当するかどうかを予測する。

## 対象ジャンル

対象ジャンルは次の 19 種類である。

- Action
- Adventure
- Comedy
- Drama
- Ecchi
- Fantasy
- Hentai
- Horror
- Mahou Shoujo
- Mecha
- Music
- Mystery
- Psychological
- Romance
- Sci-Fi
- Slice of Life
- Sports
- Supernatural
- Thriller

## 現在のデータフロー

1. AniList GraphQL API からアニメ作品の ID、タイトル、ジャンル、カバー画像 URL を取得する。
2. ジャンルのリストを one-hot 形式の 19 列に変換し、`data/anime_data.csv` として保存する。
3. 欠損値、ジャンルなしデータ、無効な画像 URL を取り除き、`data/preprocessed_anime_data.csv` を作る。
4. AniList relations を取得し、同一シリーズまたは強い派生関係にある作品を `SeriesGroup` としてまとめる。
5. `SeriesGroup` 単位で train / validation / test に分割する。
6. 学習時に画像を `ImageUrl` から取得し、`data/images/` にキャッシュしながらモデルへ入力する。
7. ResNet18 ベースラインまたは `experiments/` 以下の各実験で、19 ジャンルの同時予測を行う。

現行の学習コード `src/preprocessing/dataset_utils.py` は、次のシリーズ単位分割済み CSV を読み込む。

```text
data/series_split_outputs/training_data_grouped.csv
data/series_split_outputs/validation_data_grouped.csv
data/series_split_outputs/test_data_grouped.csv
```

## データセット

`data/series_split_outputs/` の概要は次の通りである。

| 項目 | 値 |
| --- | ---: |
| 作品数 | 11,199 |
| ユニーク ID 数 | 11,199 |
| ジャンル数 | 19 |
| シリーズグループ数 | 6,447 |
| 複数作品を含むシリーズグループ数 | 1,748 |
| 最大シリーズグループの作品数 | 67 |
| relation edges total | 14,804 |
| relation edges inside dataset | 12,539 |
| relation edges used for grouping | 10,502 |
| リーク検査 | passed |

split ごとの行数は次の通りである。

| split | rows | row ratio | series groups | 平均ジャンル数 |
| --- | ---: | ---: | ---: | ---: |
| train | 8,957 | 0.7998 | 5,153 | 2.478 |
| validation | 1,121 | 0.1001 | 617 | 2.411 |
| test | 1,121 | 0.1001 | 677 | 2.459 |

同じ `SeriesGroup` は必ず同じ split に入るため、train / validation / test 間のシリーズリークは検出されていない。

## シリーズ単位分割の目的

通常のランダム分割では、同じシリーズの別作品が train と test に分かれる可能性がある。例えば第 1 期が train、第 2 期が test に入ると、キャラクター、構図、ロゴ、画風などが似ているため、モデルがジャンルを理解していなくても test で良いスコアが出る可能性がある。

この問題を避けるため、AniList relations から作品間の edge を作り、Union-Find で同じ系列の作品を 1 つの `SeriesGroup` にまとめている。そのうえで、行単位ではなく `SeriesGroup` 単位で train / validation / test に割り当てている。

詳細は `playground/kazusa/series_split/dataset_report.md` を参照する。

## 主要ディレクトリ

```text
src/
  scraiping/
    scraiping.py
    extraLarge_scraiping.py
  preprocessing/
    data_check.py
    genre.py
    anilist_group_split.py
    dataset_utils.py
    glaf.py
  baseline_resnet/
    model.py
    train.py
    evaluate.py
    run_baseline.py
    model/
      resnet18_best.pth
      baseline_full_metrics.csv
experiments/
  template/
data/
  anime_data.csv
  preprocessed_anime_data.csv
  series_split_outputs/
playground/
  kazusa/
    baseline/
    series_split/
```

## ベースラインモデル

`src/baseline_resnet/run_baseline.py` は ResNet18 ベースの学習パイプラインである。

主な設定は次の通り。

| 項目 | 内容 |
| --- | --- |
| モデル | ResNet18 |
| 初期重み | 事前学習なし、`weights=None` |
| 出力 | 19 次元 logits |
| 入力画像 | 224 x 224 |
| 正規化 | ImageNet mean / std |
| 損失関数 | `BCEWithLogitsLoss` |
| optimizer | Adam |
| learning rate | 1e-3 |
| batch size | 64 |
| epoch 数 | 100 |
| checkpoint | validation loss が改善するたびに `resnet18_best.pth` を保存 |

保存済みの `src/baseline_resnet/model/baseline_full_metrics.csv` には 100 epoch 分の validation 指標が記録されている。validation loss が最小だったのは epoch 8 であり、学習終了後の `resnet18_best.pth` は epoch 8 の checkpoint である。

代表点は次の通り。

| 観点 | Epoch | Train Loss | Val Loss | Macro F1 | Samples F1 | Hamming Loss | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 初回 | 1 | 0.3264 | 0.3257 | 0.0751 | 0.2081 | 0.1240 | 0.2330 |
| Val Loss 最小 | 8 | 0.2932 | 0.3021 | 0.1024 | 0.3106 | 0.1193 | 0.2864 |
| mAP 最大 | 11 | 0.2812 | 0.3118 | 0.1759 | 0.3531 | 0.1237 | 0.2868 |
| Macro F1 最大 | 34 | 0.0258 | 0.7961 | 0.2413 | 0.3470 | 0.1513 | 0.2507 |
| 最終 | 100 | 0.0007 | 1.0294 | 0.2241 | 0.3559 | 0.1373 | 0.2516 |

train loss は下がり続ける一方、validation loss は epoch 8 以降で悪化している。現行の 100 epoch 学習は過学習が強く、early stopping を入れるなら epoch 8 から 11 付近が候補になる。

## Test 評価

`playground/kazusa/baseline/analyze_baseline.py` で保存済み checkpoint を追加評価した結果は次の通りである。

| 評価方法 | Macro F1 | Samples F1 | Hamming Loss | mAP | 予測ジャンル数/作品 |
| --- | ---: | ---: | ---: | ---: | ---: |
| しきい値 0.5 固定 | 0.1133 | 0.3240 | 0.1180 | 0.2875 | 0.857 |
| validation で調整したしきい値 | 0.3252 | 0.4318 | 0.2178 | 0.2875 | 4.685 |

0.5 固定しきい値では予測ジャンル数が少なく、実データの平均ジャンル数である約 2.46 よりかなり低い。このため Recall 不足になりやすい。

validation でジャンル別しきい値を最適化すると Macro F1 と Samples F1 は上がるが、予測ジャンル数が多くなり、Hamming Loss は悪化する。最終的な評価では、目的指標を先に決めたうえで、クラス別しきい値または top-k 制約を設計する必要がある。

詳細は `playground/kazusa/baseline/baseline_report.md` を参照する。

## 実験の進め方

実験は `make_exp.py` で `experiments/template/` をコピーして作る。

```bash
uv run python make_exp.py --name <your_name>-<experiment_name>
```

例:

```bash
uv run python make_exp.py --name kazusa-resnet50
```

作成された `experiments/<your_name>-<experiment_name>/` 内で、主に次のファイルを編集する。

| ファイル | 役割 |
| --- | --- |
| `config.yaml` | epoch 数、batch size、learning rate、画像サイズ、出力先など |
| `model.py` | モデル定義 |
| `criterion.py` | 損失関数 |
| `optimizer.py` | 最適化手法 |
| `train.py` | 学習ループ |
| `evaluate.py` | 評価処理 |
| `report.md` | 実験の仮説、変更内容、結果、採用判断の記録 |

## 実行コマンド

```bash
# データ取得
uv run python src/scraiping/scraiping.py

# データの欠損・URL確認
uv run python src/preprocessing/data_check.py

# シリーズ単位分割
uv run python src/preprocessing/anilist_group_split.py

# ベースライン学習
uv run python src/baseline_resnet/run_baseline.py

# 実験テンプレートから作成した実験を実行
uv run python experiments/<your_name>-<experiment_name>/run_exp.py
```

## 注意点

- ディレクトリ名とファイル名に `scraiping` という綴りが使われている。一般的には `scraping` だが、現状のコードではこの名前で参照されている。
- `extraLarge_scraiping.py` は GraphQL クエリで `extraLarge` を取得しているが、保存時に `cover_image.get("large")` を参照しているため、意図通りに画像 URL が入らない可能性がある。
- 画像は学習中に URL から取得されるため、初回実行時はネットワーク接続と時間が必要である。
- `src/baseline_resnet/run_baseline.py` の現行学習では `pos_weight`、データ拡張、scheduler、early stopping は使っていない。
- `playground/kazusa/` は分析メモと生成物置き場であり、正式な学習入力は `data/series_split_outputs/` 側である。

## 次に試す改善案

1. ImageNet 事前学習済み ResNet18 / ResNet50 を使い、スクラッチ学習との差分を比較する。
2. `pos_weight`、focal loss、class-balanced loss などで少数ジャンルを補正する。
3. RandomResizedCrop、HorizontalFlip、ColorJitter などの軽い画像拡張を追加する。
4. validation loss または mAP による early stopping を導入する。
5. 0.5 固定しきい値だけでなく、クラス別しきい値や top-k 制約を比較する。
6. ジャンル別 AP / F1 を継続して出力し、`Thriller`, `Horror`, `Music`, `Psychological`, `Sports`, `Mystery` などの少数ジャンルを重点的に見る。
