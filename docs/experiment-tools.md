# 実験ツールとテンプレート

この文書は、実験テンプレートの設定、入出力、関数契約を説明します。
作業の順序は[実験チュートリアル](experiment-tutorial.md)、課題と評価方針は[プロジェクト概要](project-overview.md)を参照してください。

## 設計方針

共有する実験は`experiments/<experiment-name>/`だけで再実行できる形にします。
実験ごとにモデル、損失関数、optimizer、学習処理を複製するため、変更内容を他の実験から切り離せます。

共通化する範囲は、正式データの読み込み、画像キャッシュ、ジャンル順です。
これらは`src/preprocessing/dataset_utils.py`が提供します。

実験反復ではvalidationだけを使います。
testを使う正式評価は`outputs/evaluation/`の固定済み手順へ分離しています。

## データフロー

```text
make_exp.py
  └── experiments/template/を複製

run_exp.py
  ├── config.yaml
  ├── model.py
  ├── criterion.py
  ├── optimizer.py
  ├── train.py
  ├── evaluate.py
  └── outputs/

analyze.py
  ├── best checkpoint
  ├── validation_data_grouped.csv
  └── analysis/

make_report.py
  ├── config.yaml
  ├── metrics.csv
  ├── analysis/
  └── README.md
```

実験コードはリポジトリのルートから実行します。
データCSVと`data/images/`がカレントディレクトリ基準だからです。

## `make_exp.py`

`make_exp.py`は`experiments/template/`を新しい実験ディレクトリへ複製します。

入力方法は二つあります。

```bash
uv run python make_exp.py --name kazusa-resnet50
uv run python make_exp.py --user-name kazusa --exp-name resnet50
```

名前には英数字、`_`、`-`だけを使用できます。
生成先が存在する場合は終了し、既存ファイルを上書きしません。

複製時には`run_train.sbatch`のjob nameと`experiment_dir`を書き換えます。
それ以外のファイル内容はテンプレートと同じです。

## テンプレートの構成

| ファイル | 役割 | 主な変更場面 |
| --- | --- | --- |
| `config.yaml` | 実験条件、seed、出力先 | すべての実験 |
| `model.py` | `ExperimentModel` | backboneやheadの変更 |
| `criterion.py` | `build_criterion()` | 損失関数の変更 |
| `optimizer.py` | `build_optimizer(model, learning_rate)` | optimizerの変更 |
| `train.py` | 1 epochの学習 | 学習手順の変更 |
| `evaluate.py` | validation推論 | 評価処理の変更 |
| `metrics.py` | 指標計算 | 指標の追加や変更 |
| `run_exp.py` | 学習全体の制御 | transformやDataLoaderの変更 |
| `analyze.py` | best checkpointの詳細分析 | 分析項目の変更 |
| `make_report.py` | Markdown生成 | レポート形式の変更 |
| `run_train.sbatch` | SLURM実行 | GPU、時間、環境の変更 |
| `report.md` | 旧来の実験メモひな型 | 必要な場合だけ使用 |

## 部品の契約

### モデル

`run_exp.py`は`ExperimentModel(num_classes=19)`を作成します。
`forward()`は形状`[batch, 19]`のlogitを返す必要があります。

### 損失関数

テンプレートは`build_criterion()`を引数なしで呼び出します。
初期実装は`BCEWithLogitsLoss`です。

### optimizer

テンプレートは`build_optimizer(model, learning_rate)`を呼び出します。
初期実装はAdamです。

### 学習と評価

`train_one_epoch()`はサンプル数で重み付けしたepoch平均lossを返します。
`evaluate_model()`はvalidation全体のloss、Macro F1、Samples F1、Hamming Loss、mAPを返します。

テンプレートはlogitが0より大きい場合を陽性と判定します。
これはsigmoid後のscoreを0.5で判定する処理と同じです。
mAPはしきい値化前のlogitを使います。

## `config.yaml`

### seedとdevice

| キー | 意味 |
| --- | --- |
| `seed` | `seeds`が空の場合に使う単一seed |
| `seeds` | 順番に実行するseedの一覧。値があれば`seed`より優先 |
| `device` | `auto`、`cpu`、`cuda`、`mps`などのPyTorch device |
| `compile` | MPS以外で`torch.compile()`を使うか |

`set_seed()`はPythonの`random`とPyTorchのseedを設定します。
アルゴリズム全体の完全な決定性は強制しません。

### 比較設定

`comparison`は、レポートへ載せる比較対象をまとめたmappingです。

| キー | 意味 |
| --- | --- |
| `comparison.primary` | レポートの主比較に使う実験名 |
| `comparison.references` | 参考として併記する実験名の一覧 |

比較先の実験にも`analysis/overall_model_metrics.csv`が必要です。
比較にはvalidation結果を使用します。

### 学習設定

| キー | 意味 |
| --- | --- |
| `epochs` | 最大epoch数 |
| `batch_size` | trainとvalidationのbatch size |
| `learning_rate` | `build_optimizer()`へ渡す学習率 |
| `num_workers` | DataLoaderのworker数 |
| `image_size` | 入力画像を正方形へresizeするサイズ |
| `max_train_samples` | trainの先頭から使う最大件数。空なら全件 |
| `max_val_samples` | validationの先頭から使う最大件数。空なら全件 |

`max_*_samples`は動作確認用です。
ランダム抽出ではなくDataFrameの先頭を使うため、この設定の指標を正式な比較結果にしません。

### early stopping

`early_stopping`は、best checkpointの更新と学習停止条件をまとめたmappingです。

| キー | 意味 |
| --- | --- |
| `enabled` | early stoppingを使うか |
| `monitor` | 改善判定に使う`metrics.csv`の列名 |
| `mode` | 大きいほど良い場合は`max`、小さいほど良い場合は`min` |
| `patience` | 改善なしを許容するepoch数 |
| `min_delta` | 改善とみなす最小差 |
| `min_epochs` | 停止を許可する最小epoch |

monitorが`min_delta`を超えて改善した場合だけbest checkpointを更新します。

### 出力設定

| キー | 意味 |
| --- | --- |
| `output_dir` | `config.yaml`を基準に解決する出力ディレクトリ |
| `best_model_name` | best checkpointのファイル名 |
| `metrics_name` | epoch履歴のCSVファイル名 |

## `run_exp.py`の出力

`seeds`に複数の値がある場合は次の構成になります。

```text
outputs/
├── seed_training_summary.csv
├── seed_42/
│   ├── best_model.pth
│   └── metrics.csv
├── seed_43/
│   ├── best_model.pth
│   └── metrics.csv
└── seed_44/
    ├── best_model.pth
    └── metrics.csv
```

単一seedの場合はcheckpointと`metrics.csv`を`outputs/`直下へ保存します。
どちらの場合も`seed_training_summary.csv`を出力先の直下へ作ります。

`metrics.csv`にはepoch、seed、train loss、validation loss、Macro F1、Samples F1、Hamming Loss、mAPが入ります。
`seed_training_summary.csv`にはbest epoch、monitor値、実行epoch数、early stoppingの有無が入ります。

## `analyze.py`

`analyze.py`は保存済みbest checkpointを読み直し、validation全件を0.5固定しきい値で評価します。
testとしきい値最適化は使用しません。

複数seedの集約結果は実験直下の`analysis/`へ保存します。

| ファイル | 内容 |
| --- | --- |
| `analysis_summary.json` | 評価条件、run一覧、指標、生成物一覧 |
| `overall_model_metrics.csv` | seed平均と標本標準偏差 |
| `seed_overall_model_metrics.csv` | seedごとの全体指標 |
| `genre_metrics_validation_threshold_0.5.csv` | seed平均のジャンル別指標 |
| `seed_genre_metrics_validation_threshold_0.5.csv` | seedごとのジャンル別指標 |
| `learning_curves.png`、`.svg` | seed別曲線と平均曲線 |

各runの`analysis/`には、全体指標、ジャンル別指標、学習曲線、`model_predictions.npz`を保存します。
`model_predictions.npz`にはID、正解ラベル、logit、sigmoid後のscoreが入ります。

## `make_report.py`

`make_report.py`は`config.yaml`、学習履歴、`analysis/`を読み、実験用Markdownを生成します。

```bash
uv run python experiments/<experiment-name>/make_report.py
```

主なオプションは次のとおりです。

| オプション | 意味 |
| --- | --- |
| `--analysis-dir` | 分析結果の読み込み先 |
| `--metrics` | 学習履歴CSV |
| `--config` | 設定ファイル |
| `--output` | Markdownの出力先 |
| `--compare-to` | `comparison.primary`の上書き |
| `--reference` | 参考実験の上書き。複数回指定可能 |
| `--seed-group` | seed集約時の実験グループ名 |

既存の出力ファイルがある場合は対話的に上書きを確認します。
SLURMなどの非対話処理では、新しい`--output`を指定します。

生成されたレポートは、分析値と再現情報を埋めた下書きです。
仮説、変更点、結果の解釈、採用判断は開発者が記入します。

## `run_train.sbatch`

`make_exp.py`はテンプレートのjob nameと実験パスを複製先に合わせます。
GPU種別、時間制限、moduleや環境設定は実行先のSLURM環境に合わせて編集します。

テンプレートは次の処理を順に実行します。

1. `run_exp.py`
2. `analyze.py`
3. `make_report.py`

スクリプトには`set -e`がないため、途中の処理が失敗しても後続コマンドへ進み得ます。
完了後は`.log`と`.err`の両方を確認します。

## 評価指標

| 指標 | 意味 | 判定 |
| --- | --- | --- |
| mAP | ジャンルごとのAverage Precisionの平均 | 高いほど良い。主評価指標 |
| Macro F1 | ジャンルごとのF1を同じ重みで平均 | 高いほど良い |
| Samples F1 | 作品ごとのラベル集合のF1を平均 | 高いほど良い |
| Hamming Loss | 全作品と全ジャンルの0/1誤り率 | 低いほど良い |
| Precision | 陽性予測のうち正解だった割合 | 高いほど誤検出が少ない |
| Recall | 正解陽性のうち検出できた割合 | 高いほど見逃しが少ない |
| AP | 一つのジャンルの順位品質 | 高いほど良い |

Macro F1、Samples F1、Hamming Loss、Precision、Recallはしきい値に依存します。
mAPとAPはしきい値化前のscoreで順位を評価します。
