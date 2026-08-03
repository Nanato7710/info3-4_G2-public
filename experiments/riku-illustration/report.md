# 実験ワークシート

このワークシートは、`experiments/template` をコピーして作った各実験の目的、仮説、変更内容、結果、採用判断を記録するためのテンプレートです。

`baseline_v1 -> exp_001`、`exp_001 -> exp_002` のように、毎回「旧モデル」と「新モデル」を比較する形式で使います。

---

## 0. 実験情報

| 項目 | 内容 |
|---|---|
| 実験ID | exp_001 |
| 日付 | 6/30 |
| 担当者 | 玉城俐空 |
| タスク | アニメのキービジュアルからジャンルを推定するマルチラベル分類 |
| 旧モデル | baseline_v1 |
| 新モデル | resnet50_360_body.pth |
| 今回の目的 | イラスト事前モデルとの比較 |
| Git branch | riku-illustration |
| Git commit |  |
| 実験ディレクトリ | `experiments/exp_001` |
| config | `experiments/exp_001/config.yaml` |
| metrics | `experiments/exp_001/outputs/metrics.csv` |
| checkpoint | `experiments/exp_001/outputs/best_model.pth` |
---

## 1. 実行メモ

### 1.1 実行コマンド

```bash
uv run python experiments/exp_001/run_exp.py --config experiments/exp_001/config.yaml
```

### 1.2 主な設定

| 項目 | 値 |
|---|---|
| seed |  |
| device |  |
| epochs | 50 |
| batch size | 32 |
| learning_rate | 0.001 |
| layer4_learning_rate | 0.0001 |
| body_learning_rate | 0.000001 |
| num workers | 0 |
| image size | 360 |
| torch.compile |  |
| max train samples |  |
| max val samples |  |
| output dir |  |

### 1.3 出力ファイル

| 種類 | path | 備考 |
|---|---|---|
| best model | resnet50_360_body.pth | validation loss が最良の checkpoint |
| metrics CSV | resnet50_360_body.csv | epoch ごとの train/validation 指標 |
| 追加ログ |  |  |

---

## 2. 旧モデルの状況

### 2.1 旧モデルの構成

| 項目 | 内容 |
|---|---|
| モデル |  |
| 画像エンコーダ |  |
| 事前学習 |  |
| 分類ヘッド |  |
| loss |  |
| optimizer |  |
| scheduler |  |
| batch size |  |
| epoch数 |  |
| learning rate |  |
| threshold |  |
| augmentation |  |
| その他 |  |

### 2.2 旧モデルのスコア

| 指標 | score |
|---|---:|
| train loss |  |
| validation loss |  |
| mAP | 0.3 |
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
事前学習をリアル画像をアニメ画像に変えた場合どうなるかを検証したいと思った。

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
| H1 |  |  |  |  |
| H2 |  |  |  |  |
| H3 |  |  |  |  |

### 記入例

| 仮説ID | 観察された問題 | 原因仮説 | 根拠 | 検証方法 |
|---|---|---|---|---|
| H1 | レアラベルの recall が低い | class imbalance の影響が大きい | ラベル頻度が低いほど AP が低い | class weight / focal loss を試す |
| H2 | 似たジャンルを混同する | ラベル間の共起関係を扱えていない | 両ラベルの同時出現が多い | label correlation を考慮する |
| H3 | 一部ジャンルが画像だけで当たらない | 入力情報が不足している | 画像から内容を推測しにくい | タイトル・あらすじを追加する |

---

## 5. 今回の変更

### 5.1 変更するもの

- [ ] データ
- [ ] split
- [ ] 前処理
- [ ] augmentation
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

```

### 5.3 変更しないもの

```markdown
例：
データ分割、画像サイズ、optimizer、learning rate は旧モデルと同じにする。
今回は loss の効果だけを見る。
```

---

## 6. 期待する結果

### 6.1 期待する改善

```markdown
mAPのスコアが高くなる。
```

### 6.2 想定される副作用

```markdown

```

### 6.3 成功条件

```markdown
例：
- 旧モデルより macro F1 が上がる
- 低頻度ラベルの recall が改善する
- mAP が大きく下がらない
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

| モデル | 変更点 | val loss | mAP | macro F1 | samples F1 | Hamming loss | 備考 |
|---|---|---:|---:|---:|---:|---:|---|
| 旧モデル |  |  |  |  |  |  |  |
| 新モデル |  |  |  |  |  |  |  |
| 差分 |  |  |  |  |  |  |  |

### 7.2 train / validation の差

| 観点 | 結果 | 解釈 |
|---|---|---|
| train は良いが validation が悪い | あり / なし | 過学習の可能性 |
| train も validation も悪い | あり / なし | 未学習・モデル不足・データ困難の可能性 |
| 特定ラベルだけ悪い | あり / なし | 不均衡・曖昧ラベル・ラベルノイズの可能性 |
| validation のばらつきが大きい | あり / なし | データ数不足・split 不安定の可能性 |

### 7.3 metrics.csv の最良 epoch

| 項目 | 値 |
|---|---:|
| best epoch |  |
| train loss |  |
| val loss |  |
| mAP |  |
| macro F1 |  |
| samples F1 |  |
| Hamming loss |  |

---

## 8. ラベル別比較

| ラベル | 件数 | 旧AP | 新AP | AP差分 | 旧F1 | 新F1 | F1差分 | 解釈 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Action |  |  |  |  |  |  |  |  |
| Adventure |  |  |  |  |  |  |  |  |
| Comedy |  |  |  |  |  |  |  |  |
| Drama |  |  |  |  |  |  |  |  |
| Fantasy |  |  |  |  |  |  |  |  |

---

## 9. 頻度別比較

| ラベル群 | 対象ラベル | 旧スコア | 新スコア | 差分 | 解釈 |
|---|---|---:|---:|---:|---|
| 高頻度ラベル |  |  |  |  |  |
| 中頻度ラベル |  |  |  |  |  |
| 低頻度ラベル |  |  |  |  |  |

---

## 10. エラー分析

### 10.1 失敗例の抽出条件

- [ ] 旧モデルでも新モデルでも外しているサンプル
- [ ] 旧モデルでは当たっていたが新モデルで外したサンプル
- [ ] 新モデルで confidence が高いのに誤っているサンプル
- [ ] 新モデルで正解ラベルの confidence が低いサンプル
- [ ] 特定ラベルで false positive が増えたサンプル
- [ ] 特定ラベルで false negative が残っているサンプル
- [ ] loss が大きいサンプル
- [ ] レアラベルを含むサンプル
- [ ] ラベル数が多いサンプル

抽出件数：

```markdown
例：validation から 100 件
```

### 10.2 改善した例

| ID | 正解ラベル | 旧モデル予測 | 新モデル予測 | 改善内容 | 推定理由 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |

### 10.3 悪化した例

| ID | 正解ラベル | 旧モデル予測 | 新モデル予測 | 悪化内容 | 推定理由 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |

### 10.4 まだ失敗している例

| ID | 正解ラベル | 新モデル予測 | 失敗内容 | 原因候補 |
|---|---|---|---|---|
|  |  |  | false positive / false negative |  |
|  |  |  | false positive / false negative |  |

### 10.5 残っている失敗パターン

| 優先度 | 失敗パターン | 影響するラベル | 頻度 | 重要度 | メモ |
|---|---|---|---:|---|---|
| 1 |  |  |  | 高 / 中 / 低 |  |
| 2 |  |  |  | 高 / 中 / 低 |  |
| 3 |  |  |  | 高 / 中 / 低 |  |

---

## 11. 仮説の判定

| 仮説ID | 仮説 | 結果 | 判定 |
|---|---|---|---|
| H1 |  |  | 支持 / 棄却 / 保留 |
| H2 |  |  | 支持 / 棄却 / 保留 |
| H3 |  |  | 支持 / 棄却 / 保留 |

### 11.1 判定理由

```markdown

```

---

## 12. 採用判断

### 12.1 採用判定

- [ ] 採用
- [ ] 条件付き採用
- [ ] 不採用
- [ ] 保留

### 12.2 判断理由

```markdown

```

### 12.3 採用する場合に残す変更

```markdown

```

### 12.4 採用しない場合の理由

```markdown

```

---

## 13. validation 過適合チェック

- [ ] validation の失敗例だけに合わせた改善になっていない
- [ ] validation score の小さな差を過大評価していない
- [ ] 複数 seed で傾向を確認した
- [ ] test set は最後まで触っていない
- [ ] test set の結果を見てから再調整していない
- [ ] 改善理由が説明できる

### 13.1 複数 seed 結果

| seed | mAP | macro F1 | samples F1 | Hamming loss | 備考 |
|---:|---:|---:|---:|---:|---|
| 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 平均 |  |  |  |  |  |
| 標準偏差 |  |  |  |  |  |

---

## 14. 次の課題

### 14.1 今回解決したこと

```markdown

```

### 14.2 まだ残っていること

```markdown

```

### 14.3 次に試す候補

```markdown

```

---

## 15. 実験ログ追記用

| 実験ID | 旧モデル | 新モデル | 目的 | 変更点 | 結果 | 採用判断 | 次の課題 |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

---

## 16. レポート・発表用まとめ

### 16.1 背景

```markdown
旧モデルでは、〇〇という問題が残っていた。
```

### 16.2 仮説

```markdown
この問題の原因は、〇〇であると考えた。
```

### 16.3 手法

```markdown
この仮説を検証するため、今回は〇〇を導入した。
```

### 16.4 結果

```markdown
新モデルは旧モデルと比較して、〇〇が改善した。
一方で、〇〇は悪化または維持にとどまった。
```

### 16.5 考察

```markdown
以上より、〇〇は有効であると考えられる。
ただし、〇〇という副作用が確認されたため、次回は〇〇を検討する。
```

---

## 17. 新しい実験の作り方

1. リポジトリルートで `uv run python make_exp.py --user-name <your_name> --exp-name <experiment_name>` を実行します。
2. 作成された `experiments/<your_name>-<experiment_name>/config.yaml` を確認します。
3. `report.md` の `実験ID`、`旧モデル`、`新モデル`、`実験ディレクトリ` を更新します。
4. 必要に応じて `model.py`, `criterion.py`, `optimizer.py`, `metrics.py` を変更します。
5. `uv run python experiments/<your_name>-<experiment_name>/run_exp.py` で実験を実行します。
6. `outputs/metrics.csv` と `outputs/best_model.pth` を確認します。
7. このワークシートに結果、エラー分析、採用判断、次の課題を記録します。

重要なのは、毎回「何を変えたか」と「なぜ変えたか」を明確にすることです。

スコアが上がったかどうかだけでなく、どのラベル・どの失敗パターンで改善したかを記録します。
