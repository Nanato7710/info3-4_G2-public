# AniList relations を使ったシリーズ単位分割

`anilist_group_split.py` は、`data/preprocessed_anime_data.csv` を入力として、AniList の relations からシリーズグループを作り、同じシリーズが train / validation / test にまたがらないように分割するスクリプトです。

## 実行コマンド

```bash
uv run python playground/kazusa/series_split/anilist_group_split.py
```

デフォルトでは成果物を `playground/kazusa/series_split/outputs/` に生成します。

## 生成されるファイル

| ファイル | 内容 |
| --- | --- |
| `anilist_relations_cache.json` | AniList API から取得した relations のキャッシュ。再実行時はこれを使う。 |
| `anilist_relation_edges.csv` | 取得した relation edge の一覧。`used_for_grouping` が `True` の edge がグループ化に使われたもの。 |
| `preprocessed_with_series_group.csv` | 元の前処理済みデータに `SeriesGroup` を追加したもの。 |
| `training_data_grouped.csv` | シリーズリークを避けた学習用データ。 |
| `validation_data_grouped.csv` | シリーズリークを避けた検証用データ。 |
| `test_data_grouped.csv` | シリーズリークを避けたテスト用データ。 |
| `split_summary.json` | 分割結果の機械可読な要約。 |
| `split_summary.md` | 分割結果の確認用レポート。 |
| `figures/genre_distribution_by_split.png` | split 内で各ジャンルが何%を占めるかを比較する横棒グラフ。 |
| `figures/genre_split_balance_heatmap.png` | 各ジャンルが 80% / 10% / 10% の目標分割比率からどれだけずれているかを見るヒートマップ。 |
| `figures/genre_distribution_chart_data.csv` | 可視化に使った集計済みデータ。 |

## グループ化に使う relationType

デフォルトでは、シリーズ関係として扱いやすい次の relationType だけをグループ化に使います。

- `PREQUEL`
- `SEQUEL`
- `SIDE_STORY`
- `PARENT`
- `SUMMARY`
- `ALTERNATIVE`
- `SPIN_OFF`
- `COMPILATION`
- `CONTAINS`

`CHARACTER` や `OTHER` なども含めると、クロスオーバーや関連作品経由でグループが広がりすぎることがあるため、デフォルトでは除外しています。

すべての anime relation を使いたい場合は次のように実行できます。

```bash
uv run python playground/kazusa/series_split/anilist_group_split.py --relation-types ALL
```

## 現在の生成結果

現在の成果物では、11,156 件のデータが 6,428 個の `SeriesGroup` にまとまりました。

| split | rows | row ratio | series groups |
| --- | ---: | ---: | ---: |
| train | 8,924 | 0.7999 | 5,128 |
| validation | 1,116 | 0.1000 | 634 |
| test | 1,116 | 0.1000 | 666 |

`SeriesGroup` の重複検査は通っており、train / validation / test 間のシリーズリークは 0 件です。

## ジャンル分布の可視化

ジャンル分布は次のコマンドで再生成できます。

```bash
uv run python playground/kazusa/series_split/plot_genre_distribution.py
```

`figures/genre_distribution_by_split.png` は、各 split の中でそのジャンルを持つ作品が何%あるかを示します。train / validation / test の棒が近いほど、split 間のジャンル分布が似ています。

`figures/genre_split_balance_heatmap.png` は、各ジャンルの全件数のうち train / validation / test に何%入ったかを、目標の 80% / 10% / 10% と比較したものです。値は percentage point です。例えば `+2.0` は、その split に目標より 2.0 percentage points 多く入っていることを意味します。
