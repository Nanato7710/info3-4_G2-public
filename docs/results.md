# 正式な評価結果

この文書は、固定済みのbaselineと`final-tri-model`をtestデータで比較した結果を示します。
数値の正本は`outputs/evaluation/`と`outputs/bootstrap/`にある機械可読な成果物です。

## 比較したモデル

両モデルは同じtrain、validation、testのsplitを使い、seed 42、43、44の3回で評価しました。

| 条件 | baseline | `final-tri-model` |
| --- | --- | --- |
| backbone | ResNet18 | ConvNeXt-Base (`convnext_base.fb_in22k`) |
| 事前学習 | なし | ImageNet-22K |
| 損失関数 | `BCEWithLogitsLoss` | Asymmetric Loss |
| optimizer | Adam | AdamW |
| 入力サイズ | 224×224 | 384×384 |
| seed | 42、43、44 | 42、43、44 |

設定の正本は[`outputs/baseline/config.yaml`](../outputs/baseline/config.yaml)と[`outputs/final-tri-model/config.yaml`](../outputs/final-tri-model/config.yaml)です。
`final-tri-model`は一つのモデル構成を3 seedで学習したものであり、3モデルのアンサンブルではありません。

## test結果

主評価指標には、19ジャンルのAverage Precisionを平均したmAPを使います。
表の`±`は3 seed間の標本標準偏差です。

| モデル | test mAP |
| --- | ---: |
| baseline | 0.2958 ± 0.0045 |
| `final-tri-model` | 0.4456 ± 0.0034 |

3 seed平均の差は`+0.1497`です。
この差は、各seedのmAPを計算してからモデルごとに平均した値の差です。
予測scoreをseed間で平均して一つのmAPを計算した値ではありません。

各seedの値と補助指標は、[`test-summary.json`](../outputs/evaluation/results/test-summary.json)と[`test-per-seed.csv`](../outputs/evaluation/results/test-per-seed.csv)で確認できます。
validationの結果は[`validation-summary.json`](../outputs/evaluation/results/validation-summary.json)に分けて保存しています。

## SeriesGroup Bootstrap

差の不確実性は、testデータの677個の`SeriesGroup`を標本単位とする対応ありBootstrapで評価しました。
各反復では同じグループ抽出をbaselineと`final-tri-model`の6 runへ適用し、seedごとのmAP、モデル平均、モデル間差の順に計算しています。

| 項目 | 値 |
| --- | ---: |
| 有効反復数 | 10,000 |
| mAP差の点推定 | +0.1497 |
| 95% percentile CI | [+0.1249, +0.1707] |
| 差が正だった反復の割合 | 1.0000 |

95% CIは0をまたいでいません。
この結果は、指定したsplit、6個のcheckpoint、`SeriesGroup`単位の再標本化という条件に対する評価です。

Bootstrapの設定と完全な数値は、[`config.yaml`](../outputs/bootstrap/config.yaml)、[`summary.json`](../outputs/bootstrap/summary.json)、[`replicates.csv`](../outputs/bootstrap/replicates.csv)に保存しています。

## 成果物の対応

| 成果物 | 役割 |
| --- | --- |
| [`outputs/freeze-manifest.json`](../outputs/freeze-manifest.json) | 正式評価入力のパスとSHA-256を固定 |
| [`outputs/checksums.sha256`](../outputs/checksums.sha256) | freeze時点の評価成果物の整合性検査 |
| [`outputs/evaluation/results/`](../outputs/evaluation/results/) | seed別、モデル平均、ジャンル別の正式評価 |
| [`outputs/bootstrap/`](../outputs/bootstrap/) | mAP差のBootstrap設定、反復値、要約、図 |
| [`outputs/app/`](../outputs/app/) | seed 44を使うローカル推論アプリ |

## 主要成果物の再現手順

すべてのコマンドはリポジトリのルートから実行します。
学習、test評価、Bootstrapには既存成果物の上書き防止があります。
最初から再実行する場合は正式成果物を保持したまま、別の作業コピーで対象の出力先に同名成果物がない状態を用意します。

最初に依存関係を同期します。

```bash
uv sync
```

baselineと`final-tri-model`をseedごとに学習します。
次の例はseed 42です。

```bash
uv run python outputs/baseline/run_exp.py \
  --config outputs/baseline/config.yaml \
  --seed 42

uv run python outputs/final-tri-model/run_exp.py \
  --config outputs/final-tri-model/config.yaml \
  --seed 42
```

seed 43と44は、両方の学習コマンドの`--seed`を変更して個別に実行します。
6 runが完了すると、各runにbest checkpoint、学習履歴、validation予測、実行metadataが保存されます。

6 runのvalidation結果を集計し、test評価へ渡す入力をfreezeします。
freeze manifestはcheckpoint、validation予測、設定、評価コード、3個のsplitをSHA-256で固定します。

```bash
uv run python outputs/evaluation/aggregate.py --split validation
uv run python outputs/evaluation/create_freeze_manifest.py
```

freeze後に6 runをtestで評価し、validationとtestの集計を生成します。

```bash
uv run python outputs/evaluation/evaluate.py --split test
uv run python outputs/evaluation/aggregate.py --split all
```

test予測からSeriesGroup Bootstrapを再実行します。
正式成果物との混同を避けるため、再現結果は`/tmp/info3-bootstrap-repeat`へ出力します。

```bash
uv run python outputs/bootstrap/bootstrap.py \
  --config outputs/bootstrap/config.yaml \
  --output-dir /tmp/info3-bootstrap-repeat

uv run python outputs/bootstrap/verify.py \
  --config outputs/bootstrap/config.yaml \
  --result-dir /tmp/info3-bootstrap-repeat
```

再学習したseed 44のvalidation予測から、推論アプリ用のジャンル別閾値を生成します。

```bash
uv run python outputs/app/tune_threshold.py
uv run python -m unittest discover -s outputs/app/tests -v
uv run python outputs/app/app.py
```

`app.py`は`127.0.0.1`で起動します。
既存の正式なseed 44 checkpointと`threshold.json`を使う場合は、`tune_threshold.py`を省略できます。

## 再計算と検証

固定済み予測からvalidationとtestの集計を再生成する場合は、次を実行します。
このコマンドは`outputs/evaluation/results/`の集計ファイルを更新します。

```bash
uv run python outputs/evaluation/aggregate.py --split all
```

Bootstrapを正式成果物と分けて再実行する場合は、別の出力先を指定します。

```bash
uv run python outputs/bootstrap/bootstrap.py \
  --config outputs/bootstrap/config.yaml \
  --output-dir /tmp/info3-bootstrap-repeat
```

正式な反復値から選択した反復を独立計算で照合します。

```bash
uv run python outputs/bootstrap/verify.py \
  --config outputs/bootstrap/config.yaml \
  --result-dir outputs/bootstrap \
  --replicate-ids 1 137 10000
```

`outputs/checksums.sha256`は正式評価をfreezeした時点で作成されました。
その後のBootstrap実装で`outputs/artifact_common.py`に`calculate_map()`を追加したため、この一件だけは記録済みhashと現在のファイルが一致しません。
残る88件の固定済み成果物は次のコマンドで検査できます。

```bash
tail -n +2 outputs/checksums.sha256 | shasum -a 256 -c -
```
