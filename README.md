# info3-4_G2

アニメ作品のカバー画像から、19種類のジャンルを推定するマルチラベル画像分類プロジェクトです。
このリポジトリには、データ分割、実験コード、正式評価の成果物、ローカル推論アプリを保存しています。

## 最初に読む資料

- [プロジェクト概要](docs/project-overview.md)：課題、データ分割、評価方針
- [正式な評価結果](docs/results.md)：最終モデル、test結果、Bootstrap、再現コマンド
- [実験チュートリアル](docs/experiment-tutorial.md)：環境構築から実験共有までの手順
- [実験ツールとテンプレート](docs/experiment-tools.md)：設定、入出力、テンプレートの仕様
- [ドキュメント索引](docs/README.md)：目的別の資料案内

初めて実験する場合は、プロジェクト概要を読んでから実験チュートリアルを進めてください。
既存の最終結果を確認する場合は、正式な評価結果を参照してください。

## セットアップ

Python 3.13以上と`uv`を使用します。

```bash
git clone git@github.com:Nanato7710/info3-4_G2.git
cd info3-4_G2
uv sync
```

正式なデータが配置されていることを確認します。

```bash
ls data/series_split_outputs/training_data_grouped.csv
ls data/series_split_outputs/validation_data_grouped.csv
ls data/series_split_outputs/test_data_grouped.csv
```

## 新しい実験

実験用のブランチとディレクトリを作成します。

```bash
git switch -c exp/<user-name>-<experiment-name>
uv run python make_exp.py \
  --user-name <user-name> \
  --exp-name <experiment-name>
```

作成した実験をリポジトリのルートから実行します。

```bash
uv run python experiments/<user-name>-<experiment-name>/run_exp.py
uv run python experiments/<user-name>-<experiment-name>/analyze.py
uv run python experiments/<user-name>-<experiment-name>/make_report.py
```

`run_exp.py`は設定されたseedを順に学習し、`analyze.py`はbest checkpointをvalidationデータで再評価します。
`make_report.py`は設定と分析結果から実験用の`README.md`を生成します。

## ディレクトリの役割

| 場所 | 役割 |
| --- | --- |
| `data/series_split_outputs/` | シリーズ単位で分割した正式なtrain、validation、testデータ |
| `docs/` | プロジェクト全体で共有する背景、結果、手順、仕様 |
| `experiments/template/` | 新しい実験のひな型 |
| `experiments/<experiment-name>/` | 実験コード、設定、分析結果、実験別レポート |
| `outputs/` | 正式評価、Bootstrap、推論アプリと固定済み成果物 |
| `src/preprocessing/` | 実験間で共有するデータ読み込みと画像キャッシュ |
| `playground/<user-name>/` | 個人の調査、試作、整理前の出力 |
| `logs/` | SLURMジョブの標準出力と標準エラー |

## 記録の方針

`playground/`は自由な探索に使い、共有する実験は`experiments/`へ残します。
各実験には、再実行できるコードと設定、分析結果、採用判断をそろえます。
正式評価では、固定済みの`outputs/`とそのchecksumを根拠として使います。
大きなcheckpoint、画像キャッシュ、ログはGitへ追加しません。
