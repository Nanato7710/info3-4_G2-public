# 実験チュートリアル

この文書は、初めてこのリポジトリで実験する開発者が、環境構築から実験共有までを順番に進めるための手順です。
課題と評価方針は[プロジェクト概要](project-overview.md)、設定と生成物の仕様は[実験ツールとテンプレート](experiment-tools.md)を参照してください。

## 実験の流れ

```text
環境を準備する
  ↓
make_exp.pyでテンプレートを複製する
  ↓
仮説と比較条件を記録する
  ↓
少量データで動作確認する
  ↓
本学習を実行する
  ↓
validation結果を分析する
  ↓
レポートへ判断を書く
```

実験中のモデル選択にはvalidationを使います。
testは最終モデルと評価方法を確定するまで使用しません。

## 前提条件

- Gitを使用できること
- Python 3.13以上を実行できること
- `uv`を使用できること
- `data/series_split_outputs/`に正式な3個のCSVがあること
- 未取得画像がある場合はCSVの`ImageUrl`へアクセスできること
- 学科サーバーを使う場合はSSHとSLURMを利用できること

`device: auto`では、MPS、CUDA、CPUの順に利用可能なdeviceを選びます。
CPUでも動作しますが、本学習には時間がかかります。

## 環境を準備する

以降のコマンドは、特に断りがない限りリポジトリのルートで実行します。
データ読み込みと画像キャッシュがカレントディレクトリ基準であるため、実験ディレクトリへ移動して実行しません。

### リポジトリを取得する

```bash
git clone git@github.com:Nanato7710/info3-4_G2.git
cd info3-4_G2
```

学科サーバーからGitHubへ接続するためにagent forwardingを使う場合は、ローカルで鍵を追加してから接続します。

```bash
ssh-add ~/.ssh/<private-key>
ssh -A <server-host>
```

秘密鍵そのものはサーバーへコピーしません。

### 依存関係を準備する

`uv`が未導入の場合は、公式の手順に従ってインストールします。
リポジトリでは次のコマンドで依存関係を同期します。

```bash
uv sync
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

仮想環境を手動でactivateする必要はありません。
Pythonコマンドには`uv run`を付けます。

### 正式データを確認する

```bash
ls data/series_split_outputs/training_data_grouped.csv
ls data/series_split_outputs/validation_data_grouped.csv
ls data/series_split_outputs/test_data_grouped.csv
```

いずれかが欠けている場合は、実験を始める前に正式データの配置を確認します。

## 実験を作る

### ブランチを分ける

実験ごとにブランチを作ります。

```bash
git switch -c exp/<user-name>-<experiment-name>
```

実験名には、変更する条件が分かる名前を付けます。

### テンプレートを複製する

```bash
uv run python make_exp.py \
  --user-name <user-name> \
  --exp-name <experiment-name>
```

完成した名前を直接指定する場合は`--name`を使います。

```bash
uv run python make_exp.py --name <user-name>-<experiment-name>
```

名前に使える文字は英数字、`_`、`-`です。
同名のディレクトリが存在する場合は上書きしません。

生成先は`experiments/<user-name>-<experiment-name>/`です。
テンプレート自体を直接編集して実験しません。

## 実験条件を決める

コードを変更する前に、実験の`report.md`または作業メモへ次を記録します。

- 改善したい問題
- 変更する条件
- 改善すると考える理由
- 固定する条件
- 主比較と参考比較
- 採用判断に使う指標

原則として、一度の実験で変更する主要因は一つにします。
複数条件を同時に変えた場合は、改善を単一の変更へ帰属できません。

`config.yaml`では、少なくともseed、epoch数、batch size、学習率、画像サイズ、比較対象を確認します。
主評価指標はvalidation mAPです。

変更する部品とファイルの対応は次のとおりです。

| 変更 | ファイル |
| --- | --- |
| モデル構造 | `model.py` |
| 損失関数 | `criterion.py` |
| optimizer | `optimizer.py` |
| 学習処理 | `train.py` |
| 評価指標 | `evaluate.py`、`metrics.py` |
| データ変換とDataLoader | `run_exp.py` |

変更していない条件もレポートに残します。

## 少量データで動作確認する

本学習の前に、`config.yaml`を一時的に小さい設定へ変更します。

```yaml
seeds:
  - 42
epochs: 1
compile: false
max_train_samples: 100
max_val_samples: 50
output_dir: outputs/smoke
```

リポジトリのルートから実行します。

```bash
uv run python experiments/<experiment-name>/run_exp.py
```

次を確認します。

- trainとvalidationのCSVを読み込める
- 画像を取得またはキャッシュから読み込める
- forward、loss、backward、optimizer stepが完了する
- checkpoint、`metrics.csv`、`seed_training_summary.csv`が生成される
- lossと指標に`nan`が含まれない

動作確認後は、サンプル数制限、seed、epoch数、`output_dir`を本学習用に戻します。
動作確認の出力を正式結果として扱いません。

## 本学習を実行する

### 直接実行する

```bash
uv run python experiments/<experiment-name>/run_exp.py
```

複数seedを設定した場合は、seedごとに順番に学習します。
各seedのcheckpointと学習履歴は`outputs/seed_<seed>/`へ分かれます。

### SLURMで実行する

```bash
sbatch experiments/<experiment-name>/run_train.sbatch
```

状態とログを確認します。

```bash
squeue -u "$USER"
tail -f logs/<job-name>-<job-id>.log
tail -f logs/<job-name>-<job-id>.err
```

テンプレートの`run_train.sbatch`は、学習、分析、レポート生成を順に実行します。
途中のコマンドが失敗しても後続処理へ進み得るため、標準出力と標準エラーの両方を確認します。

## validation結果を分析する

```bash
uv run python experiments/<experiment-name>/analyze.py
```

`analyze.py`はbest checkpointを読み直し、validation全件を評価します。
集約結果は実験直下の`analysis/`、seed別の結果は各runの`analysis/`へ保存します。
この処理はtestを使用せず、しきい値最適化も行いません。

確認する項目は次のとおりです。

- `analysis/overall_model_metrics.csv`のvalidation mAP
- `analysis/seed_overall_model_metrics.csv`のseed間ばらつき
- `analysis/genre_metrics_validation_threshold_0.5.csv`のジャンル別APとF1
- `analysis/learning_curves.png`の過学習や未収束
- 主比較との差と、仮説に沿う変化かどうか

## 実験レポートを作る

```bash
uv run python experiments/<experiment-name>/make_report.py
```

既存の`README.md`がある場合は上書き確認が表示されます。
非対話のSLURMジョブでは確認へ回答できないため、別の出力先を指定します。

```bash
uv run python experiments/<experiment-name>/make_report.py \
  --output experiments/<experiment-name>/generated-report.md
```

自動生成されたMarkdownは下書きです。
背景、仮説、変更点、結果の解釈、採用判断、次の実験を人が追記します。

## 共有前の確認

```bash
git status --short
git diff --check
git diff --stat
```

共有する実験には次をそろえます。

- 実行に使ったコードと`config.yaml`
- 比較対象と固定条件
- seed別結果と集約結果
- 学習曲線とジャンル別分析
- 採用または不採用の判断
- 再実行コマンド

checkpoint、`data/images/`、ログなどの大きな生成物はGitへ追加しません。
正式評価へ進める実験は、validation結果だけで候補を決めてから固定済み評価手順へ渡します。

## よくある問題

### CSVが見つからない

コマンドをリポジトリのルートから実行しているか確認します。
データ読み込みは相対パスを使用します。

### 画像取得に失敗する

CSVの`ImageUrl`へ接続できるか確認します。
取得済み画像は`data/images/`から読み込まれます。

### GPUが使われない

`config.yaml`の`device`と、PyTorchが認識するdeviceを確認します。

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

### checkpointを分析できない

`config.yaml`の`output_dir`と`best_model_name`を確認します。
複数seedの場合は`output_dir/seed_<seed>/`以下にcheckpointが必要です。

### レポートに分析値が入らない

先に`analyze.py`を実行し、実験直下の`analysis/`にCSVとJSONが生成されていることを確認します。
