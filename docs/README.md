# ドキュメント

`docs/`には、プロジェクト全体で共有する背景、結果、実験手順、仕様を置きます。
個別実験の記録と固定済み評価成果物は、それぞれ`experiments/`と`outputs/`に保存します。

## プロジェクトを理解する

- [プロジェクト概要](project-overview.md)：課題、正式データ、シリーズ単位分割、評価方針
- [正式な評価結果](results.md)：ベースラインと最終モデルのtest結果、Bootstrap、再現コマンド
- [データ分割の集計結果](../data/series_split_outputs/split_summary.md)：splitの件数とリーク検査結果

最初にプロジェクト概要を読み、確定した数値が必要な場合は正式な評価結果を参照してください。

## 実験を行う

- [実験チュートリアル](experiment-tutorial.md)：環境構築、実験作成、学習、分析、共有の順序
- [実験ツールとテンプレート](experiment-tools.md)：`make_exp.py`、`config.yaml`、実験スクリプト、生成物の仕様
- [プロジェクトルートのREADME](../README.md)：最短セットアップと基本コマンド

作業の順序はチュートリアルで確認し、設定や内部契約はツール文書で確認します。

## 結果とアプリを確認する

- [`experiments/`](../experiments/)：実験ごとのコード、設定、分析結果、採用判断
- [`outputs/evaluation/`](../outputs/evaluation/)：validationとtestの正式評価
- [`outputs/bootstrap/`](../outputs/bootstrap/)：SeriesGroup Bootstrapの設定と成果物
- [推論アプリ](../outputs/app/README.md)：ローカルGradioアプリの起動、閾値、checkpoint取得

実験中のvalidation結果と、モデル確定後のtest結果は区別して扱います。
正式値は`docs/results.md`から固定済み成果物へたどって確認します。

## 配置ルール

| 場所 | 置く内容 |
| --- | --- |
| `docs/` | 全員が参照する背景、結果、手順、仕様 |
| `experiments/<experiment-name>/` | 実験時点のコード、設定、validation結果、判断 |
| `outputs/` | 固定済みモデル、正式評価、Bootstrap、推論アプリ |
| `playground/<user-name>/` | 個人の調査、試作、一時的なメモや出力 |

複数の実験に共通する説明は`docs/`へ置きます。
実験固有の変更と結果は各実験のレポートへ記録し、共通文書には結論を重複させず参照元を示します。
