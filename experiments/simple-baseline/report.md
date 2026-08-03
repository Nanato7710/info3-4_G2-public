# 実験レポート: simple-baseline

## 目的

画像を使うモデルが、ジャンルの出現率だけを使う単純な予測より有用か確認するための基準です。

## 手法

- training split でジャンルごとの出現率を計算する
- すべての validation サンプルに同じ出現率を予測スコアとして割り当てる
- 主評価指標は validation mAP とする
- test split と閾値最適化は使用しない

この基準を比較に含める場合は、通常実験の `config.yaml` に次のように指定します。

```yaml
comparison:
  primary: kazusa-baseline
  references:
    - simple-baseline
```

分析結果は次のコマンドで更新します。

```bash
uv run python experiments/simple-baseline/analyze.py
```
