# 実験ワークシート

このワークシートは、`experiments/template` をコピーして作った各実験の目的、仮説、変更内容、結果、採用判断を記録するためのテンプレートです。

`baseline_v1 -> exp_001`、`exp_001 -> exp_002` のように、毎回「旧モデル」と「新モデル」を比較する形式で使います。

---

## 0. 実験情報

| 項目 | 内容 |
|---|---|
| 実験ID | exp_001 |
| 日付 |  |
| 担当者 | 宇良俊飛 |
| タスク | データ拡張にモデルの精度の変化の確認 |
| 旧モデル | baseline_v1 |
| 新モデル | exp_001 |
| 今回の目的 | 過学習の改善 |
| Git branch | exp/takato-data_augmentation |
| Git commit |  |
| 実験ディレクトリ | `experiments/takato-data-augmentation` |
| config | `experiments/takato-data-augmentation/config.yaml` |
| metrics | `experiments/takato-data-augmentation/outputs/metrics.csv` |
| checkpoint | `experiments/takato-data-augmentation/outputs/best_model.pth` |

---

## 1. 実行メモ

### 1.1 実行コマンド

```bash
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_none.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_F.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_R.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_C.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_FR.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_FC.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_RC.yaml
uv run python experiments/takato-01/run_exp.py --config experiments/takato-01/config_FRC.yaml
```

### 1.2 主な設定

| 項目 | 値 |
|---|---|
| seed | 42 |
| device | auto |
| epochs | 40 |
| batch size | 64 |
| learning rate | 0.001 |
| num workers | 0 |
| image size | 224 |
| torch.compile | true |
| max train samples |  |
| max val samples |  |
| output dir | outpus |

### 1.3 出力ファイル

| 種類 | path | 備考 |
|---|---|---|
| best model |  | validation loss が最良の checkpoint |
| metrics CSV |  | epoch ごとの train/validation 指標 |
| 追加ログ |  |  |

---

## 2. 旧モデルの状況

### 2.1 旧モデルの構成

| 項目 | 内容 |
|---|---|
| モデル | AnimeResNet (ResNet18) |
| 画像エンコーダ | ResNet18 (weights=None) |
| 事前学習 |  |
| 分類ヘッド | nn.Linear(512, 19) |
| loss | BCEWithLogitsLoss (重みなし) |
| optimizer | Adam (lr=0.001) |
| scheduler | なし |
| batch size | 64 |
| epoch数 | 40 |
| learning rate | 0.01 |
| threshold | 0.5 (logit > 0) |
| augmentation | なし (Resize 224x224, Normalizeのみ) |
| その他 |  |

### 2.2 旧モデルのスコア

| 指標 | score |
|---|---:|
| train loss |  |
| validation loss |  |
| mAP |  |
| macro F1 |  |
| samples F1 |  |
| Hamming loss |  |

### 2.3 旧モデルで残っている問題

```markdown
例：
- レアラベルの recall が低い
- 「Slice of Life」と「Comedy」の混同が多い
- 高確率の false positive が多い
- 画像だけでは判断しにくいジャンルの AP が低い
```

---

## 3. 今回扱う問題

### 3.1 今回扱う問題

```markdown
データ不足による過学習
```

### 3.2 今回扱わない問題

```markdown

```

### 3.3 この問題を優先する理由

```markdown

```

---

## 4. 原因仮説

| 仮説ID | 観察された問題 | 原因仮説 | 根拠 | 検証方法 |
|---|---|---|---|---|
| H1 | データ不足による過学習 | データの偏りが大きい | レアラベルのAPが低い | データ拡張を試す |
| H2 |  |  |  |  |
| H3 |  |  |  |  |


---

## 5. 今回の変更

### 5.1 変更するもの

- [ ] データ
- [ ] split
- [ ] 前処理
- [x] augmentation
- [ ] モデル
- [ ] loss
- [ ] optimizer
- [ ] scheduler
- [ ] threshold
- [ ] 評価指標
- [ ] 入力情報
- [ ] その他

### 5.2 具体的な変更内容

```markdown
transformsにtransforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees = 50),transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),の組み合わせをすべて試す。
```

### 5.3 変更しないもの

```markdown
transforms以外
```

---

## 6. 期待する結果

### 6.1 期待する改善

```markdown
モデルの過学習の軽減
```

### 6.2 想定される副作用

```markdown

```

### 6.3 成功条件

```markdown
valが最小になるepoch数が上がる
旧モデルよりAPが上がる
```

### 6.4 採用しない条件

```markdown
例：
- 主指標は上がっても、重要ラベルの recall が大きく下がる場合は採用しない
- validation だけに強く、test で再現しない場合は採用しない
```

---

## 7. 実験結果

### 7.1 全体スコア比較

#### 7.1.1 前処理の記号
|F:HorizontalFlip|R:Rotation|C:ColorJitter|
|---|---|---|

#### val_loss最良時スコア

| モデル | 前処理 |epoch|train_loss| val_loss | mAP | macro F1 | samples F1 | Hamming loss | 備考 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 旧モデル | None | 10 | 0.2862 | 0.2991 | 0.2932 | 0.1391 | 0.2975 | 0.1186 |  |
| モデル1 | F | 11 (+1) | 0.2867 (+0.0005) | 0.3009 (+0.0018) | 0.2897 (-0.0035) | 0.1239 (-0.0152) | 0.2875 (-0.0100) | 0.1182 (-0.0004) |  |
| モデル2 | R | 18 (+8) | 0.2859 (-0.0003) | 0.2947 (-0.0045) | 0.2930 (-0.0001) | 0.1192 (-0.0199) | 0.3103 (+0.0128) | 0.1165 (-0.0021) |  |
| モデル3 | C | 8 (-2) | 0.3034 (+0.0172) | 0.3126 (+0.0135) | 0.2451 (-0.0480) | 0.0910 (-0.0482) | 0.2626 (-0.0349) | 0.1224 (+0.0038) |  |
| モデル4 | FR | 20 (+10) | 0.2852 (-0.0010) | 0.2947 (-0.0045) | 0.2968 (+0.0036) | 0.1483 (+0.0091) | 0.3388 (+0.0413) | 0.1164 (-0.0023) |  |
| モデル5 | RC | 30 (+20) | 0.2906 (+0.0044) | 0.3035 (+0.0044) | 0.2674 (-0.0257) | 0.1034 (-0.0358) | 0.2593 (-0.0382) | 0.1196 (+0.0009) |  |
| モデル6 | FC | 8 (-2) | 0.3093 (+0.0226) | 0.3104 (+0.0113) | 0.2423 (-0.0509) | 0.0892 (-0.0499) | 0.2576 (-0.0399) | 0.1199 (+0.0012) |  |
| モデル7 | FRC | 28 (+18) | 0.2956 (+0.0094) | 0.3025 (+0.0034) | 0.2648 (-0.0283) | 0.0999 (-0.0392) | 0.2494 (-0.0481) | 0.1193 (+0.0007) |  |


## 8. train / validation の差

| 観点 | 結果 | 解釈 |
|---|---|---|
| train は良いが validation が悪い | あり / なし | 過学習の可能性 |
| train も validation も悪い | あり / なし | 未学習・モデル不足・データ困難の可能性 |
| 特定ラベルだけ悪い | あり / なし | 不均衡・曖昧ラベル・ラベルノイズの可能性 |
| validation のばらつきが大きい | あり / なし | データ数不足・split 不安定の可能性 |


## 9. 仮説の判定

| 仮説ID | 仮説 | 結果 | 判定 |
|---|---|---|---|
| H1 |  |  | 支持 |
| H2 |  |  | 支持 / 棄却 / 保留 |
| H3 |  |  | 支持 / 棄却 / 保留 |

### 9.1 判定理由

```markdown
FlipとRotationを適用した場合過学習が改善されたうえスコアの改善も見られたため
```

---

## 10. 採用判断

### 10.1 採用判定

- [ ] 採用
- [x] 条件付き採用
- [ ] 不採用
- [ ] 保留

### 10.2 判断理由

```markdown
FlipとRotationを適用した場合を採用するのは良いがそれ以外の組み合わせの場合改善するとはいえないため
```

### 10.3 採用する場合に残す変更

```markdown

```

### 10.4 採用しない場合の理由

```markdown

```

---

## 11. validation 過適合チェック

- [ ] validation の失敗例だけに合わせた改善になっていない
- [ ] validation score の小さな差を過大評価していない
- [ ] 複数 seed で傾向を確認した
- [ ] test set は最後まで触っていない
- [ ] test set の結果を見てから再調整していない
- [ ] 改善理由が説明できる

---

## 12. 次の課題

### 12.1 今回解決したこと

```markdown
過学習を抑えたが解決したとは言えない
```

### 12.2 まだ残っていること

```markdown
過学習
```

### 12.3 次に試す候補

```markdown

```


## 13. レポート・発表用まとめ

### 13.1 背景

```markdown
旧モデルでは、過学習という問題が残っていた。
```

### 13.2 仮説

```markdown
この問題の原因は、データ不足であると考えた。
```

### 13.3 手法

```markdown
この仮説を検証するため、今回はデータ拡張を導入した。
```

### 13.4 結果

```markdown
新モデルは旧モデルと比較して、データ拡張の条件によっては全てのスコアが改善した。
```

### 13.5 考察

```markdown
以上より、Rotationは有効であると考えられる。
ただし、〇〇という副作用が確認されたため、次回は〇〇を検討する。
```

---

## 14. 新しい実験の作り方

1. リポジトリルートで `uv run python make_exp.py --user-name <your_name> --exp-name <experiment_name>` を実行します。
2. 作成された `experiments/<your_name>-<experiment_name>/config.yaml` を確認します。
3. `report.md` の `実験ID`、`旧モデル`、`新モデル`、`実験ディレクトリ` を更新します。
4. 必要に応じて `model.py`, `criterion.py`, `optimizer.py`, `metrics.py` を変更します。
5. `uv run python experiments/<your_name>-<experiment_name>/run_exp.py` で実験を実行します。
6. `outputs/metrics.csv` と `outputs/best_model.pth` を確認します。
7. このワークシートに結果、エラー分析、採用判断、次の課題を記録します。

重要なのは、毎回「何を変えたか」と「なぜ変えたか」を明確にすることです。

スコアが上がったかどうかだけでなく、どのラベル・どの失敗パターンで改善したかを記録します。
