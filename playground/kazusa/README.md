# kazusa playground

`playground/kazusa` は、データ分割と ResNet18 ベースラインの追加分析をまとめるための作業領域である。
ルート直下の正式な学習コードやデータは変更せず、調査用スクリプト、レポート、可視化、CSV/JSON の生成結果をここに置く。

このディレクトリは自由に試行錯誤する場所として使う。整理できた実験コードや、
他の実験と比較したい結果、再実行可能な設定、実験レポートは `experiments/` 配下の
各実験ディレクトリにまとめる。`playground/` は探索と下書き、`experiments/` は共有・比較用の実験記録という位置づけである。

## 要約

- `series_split/`: AniList relations を使ったシリーズ単位分割の検証とレポート。
- `baseline/`: ResNet18 ベースラインの追加評価、しきい値分析、単純ベースライン比較。
- `project_summary.md`: プロジェクト全体の現状を短く確認するための概要メモ。

現行の学習コード `src/preprocessing/dataset_utils.py` は、
`data/series_split_outputs/training_data_grouped.csv`、
`validation_data_grouped.csv`、`test_data_grouped.csv` を読み込む。
そのため、モデル評価で使う正式な split は `data/series_split_outputs/` 側であり、
`playground/kazusa/series_split/outputs/` は作業用の出力である。

## 構成

```text
playground/kazusa/
  README.md
  project_summary.md
  baseline/
    baseline_report.md
    analyze_baseline.py
    analysis/
  series_split/
    dataset_report.md
    analyze_dataset.py
    anilist_group_split.py
    plot_genre_distribution.py
    series_group_split_notes.md
    analysis/
    outputs/
```

## 主な結果

### シリーズ単位分割

`data/series_split_outputs/` の分割は、同一シリーズが train / validation / test にまたがらないように作成されている。

| 項目 | 値 |
| --- | ---: |
| 作品数 | 11,199 |
| シリーズグループ数 | 6,447 |
| 複数作品を含むシリーズグループ数 | 1,748 |
| train | 8,957 |
| validation | 1,121 |
| test | 1,121 |
| リーク検査 | passed |

詳細は `series_split/dataset_report.md` を参照する。

### ベースライン評価

保存済み checkpoint は、100 epoch 学習のうち validation loss が最小だった epoch 8 の ResNet18 である。
test split での追加評価は次の通り。

| 評価方法 | Macro F1 | Samples F1 | Hamming Loss | mAP |
| --- | ---: | ---: | ---: | ---: |
| しきい値 0.5 固定 | 0.1133 | 0.3240 | 0.1180 | 0.2875 |
| validation で調整したしきい値 | 0.3252 | 0.4318 | 0.2178 | 0.2875 |

0.5 固定しきい値では予測ジャンル数が少なく、Recall 不足になりやすい。
validation でしきい値を最適化すると F1 は上がるが、陽性予測が増えすぎて Hamming Loss は悪化する。
詳細は `baseline/baseline_report.md` と `baseline/baseline_report_for_members.md` を参照する。

## 再生成コマンド

追加分析を再生成する場合は、リポジトリルートから実行する。

```bash
# シリーズ単位分割データセットの追加分析
.venv/bin/python playground/kazusa/series_split/analyze_dataset.py

# ベースライン checkpoint の追加評価
.venv/bin/python playground/kazusa/baseline/analyze_baseline.py
```

## baseline レポート

- `baseline/baseline_report.md`: 実験結果と追加分析を簡潔にまとめた技術寄りレポート。
- `baseline/baseline_report_for_members.md`: 評価指標、学習方法、単純ベースラインを初学者向けに説明した共有用レポート。
- `baseline/analysis/`: 推論結果、ジャンル別指標、しきい値、単純ベースライン比較、学習曲線。

## series_split レポート

- `series_split/dataset_report.md`: `data/series_split_outputs/` を対象にしたデータセット構成、分割、ジャンル分布、画像品質、旧分割リーク比較のレポート。
- `series_split/series_group_split_notes.md`: シリーズ単位分割の設計メモ。
- `series_split/analysis/`: 追加分析の CSV、JSON、PNG、SVG。
- `series_split/outputs/`: `anilist_group_split.py` による作業用出力。
