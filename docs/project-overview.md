# プロジェクト概要

このプロジェクトは、アニメ作品のカバー画像から作品のジャンルを推定します。
正式な評価値は[正式な評価結果](results.md)、追加実験の進め方は[実験チュートリアル](experiment-tutorial.md)を参照してください。

## タスク

入力はアニメ作品のカバー画像です。
モデルは、19ジャンルそれぞれの該当度を表す**logit**（sigmoid適用前の値）を出力します。

対象ジャンルは次の19種類です。

`Action`、`Adventure`、`Comedy`、`Drama`、`Ecchi`、`Fantasy`、`Hentai`、`Horror`、`Mahou Shoujo`、`Mecha`、`Music`、`Mystery`、`Psychological`、`Romance`、`Sci-Fi`、`Slice of Life`、`Sports`、`Supernatural`、`Thriller`

一つの作品には複数のジャンルが付くため、一つのクラスだけを選ぶマルチクラス分類ではありません。
各ジャンルを独立した二値分類として扱う**マルチラベル分類**です。

## 正式なデータ

学習と評価には、`data/series_split_outputs/`にある次のCSVを使用します。

| 用途 | ファイル | 作品数 |
| --- | --- | ---: |
| 学習 | `training_data_grouped.csv` | 8,957 |
| モデル選択と実験比較 | `validation_data_grouped.csv` | 1,121 |
| 最終評価 | `test_data_grouped.csv` | 1,121 |

全11,199作品を8:1:1に近い割合で分割しています。
件数、ジャンル分布、リーク検査は[データ分割の集計結果](../data/series_split_outputs/split_summary.md)に記録しています。

CSVには画像URLが含まれます。
共通の読み込み処理は、画像を初回利用時に取得して`data/images/`へキャッシュします。

## シリーズ単位分割

同じシリーズの作品は、キャラクター、絵柄、ロゴ、構図が似ている場合があります。
通常のランダム分割では、同じシリーズの別作品が学習データと評価データに分かれ、未知シリーズに対する性能より評価が高く見える可能性があります。

この影響を抑えるため、AniListの作品関係を使って同じ系列の作品を`SeriesGroup`へまとめ、グループ単位でtrain、validation、testへ割り当てています。
現在の正式なsplitでは、split間で重複する`SeriesGroup`はありません。

## 評価方針

実験中のモデル選択、early stopping、実験間比較にはvalidationデータを使用します。
testデータは、モデルと評価方法を確定した後の最終評価にだけ使用します。

主評価指標は**mAP**です。
mAPはジャンルごとのAverage Precisionを平均し、しきい値を固定せずに予測順位を評価します。

Macro F1、Samples F1、Hamming Lossは補助指標です。
これらの値は、logitを陽性と陰性へ分けるしきい値の影響を受けます。
特に陽性ラベルが少ないデータでは、陽性をほとんど予測しないモデルでもHamming Lossが低く見える場合があります。

正式評価では、baselineと`final-tri-model`をseed 42、43、44で比較しました。
各seedのmAPを計算してから平均し、差の不確実性は`SeriesGroup`単位の対応ありBootstrapで確認しています。
モデル条件と数値は[正式な評価結果](results.md)に集約しています。

## 文書と成果物の関係

- 課題、正式データ、評価方針：[プロジェクト概要](project-overview.md)
- 確定したモデル条件と評価値：[正式な評価結果](results.md)
- 実験の作業手順：[実験チュートリアル](experiment-tutorial.md)
- 実験テンプレートの仕様：[実験ツールとテンプレート](experiment-tools.md)
- split件数とリーク検査：[データ分割の集計結果](../data/series_split_outputs/split_summary.md)

数値や挙動を更新する場合は、先に機械可読な成果物または実装を更新し、共通文書には判断に必要な要点と参照先を反映します。
