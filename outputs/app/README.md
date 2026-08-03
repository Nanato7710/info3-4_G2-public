# 推論アプリ

`final-tri-model`のseed 44 checkpointを使い、画像一枚に対する19ジャンルのscoreを表示するローカルGradioアプリである。

候補判定の既定値には、同じseedのvalidation予測でジャンル別F1を最大化した19個の閾値を使う。

UIで変更した閾値は現在のsession内の表示だけに反映し、`threshold.json`と評価成果物を更新しない。

## 閾値の再生成

リポジトリ内の正式checkpointとvalidation予測から次のcommandで再生成する。

```bash
uv run python outputs/app/tune_threshold.py
```

この処理はtest予測を読み込まない。

## 起動

リポジトリ内のcheckpointを直接使う場合は次のcommandで起動する。

```bash
uv run python outputs/app/app.py
```

CPUを明示する場合は`--device cpu`を指定する。

アプリは`127.0.0.1`だけで待ち受け、共有URLを作成しない。

報告書用画面などで起動時から確認用画像と推論結果を表示する場合は、`--example-image`を指定する。

```bash
uv run python outputs/app/app.py \
  --device cpu \
  --example-image outputs/app/assets/app-input-kyodai-robot.png
```

この確認用画像の出典は、いらすとやの[「巨大ロボットのイラスト」](https://www.irasutoya.com/2016/04/blog-post_776.html)である。

この引数を省略した場合は、従来どおり空の入力画面で起動する。

## Releaseからのcheckpoint取得

公開先は[info3-4_G2-publicのRelease](https://github.com/Nanato7710/info3-4_G2-public/releases/tag/final-report-2026-07-28)である。

Releaseへ手動で置くassetは、baselineと`final-tri-model`の3 seed分に相当する6個の`.pth`だけである。

manifestとRelease対象用のchecksum一覧はこのリポジトリで管理し、Release assetには含めない。

該当ファイルは`outputs/release-manifest.json`と`outputs/release-checksums.sha256`である。

`download_checkpoint.py`が読むmanifestは次のschemaを使用する。

```json
{
  "repository": "Nanato7710/info3-4_G2-public",
  "release_tag": "final-report-2026-07-28",
  "release_assets": [
    {
      "model_id": "final-tri-model",
      "seed": 44,
      "asset_name": "final-tri-model-seed-44.pth",
      "size_bytes": 350453737,
      "sha256": "64文字のSHA-256"
    }
  ]
}
```

取得commandは次のとおりである。

```bash
uv run python outputs/app/download_checkpoint.py \
  --manifest outputs/release-manifest.json \
  --model final-tri-model \
  --seed 44
```

既存ファイルまたはダウンロード結果のSHA-256が一致しない場合、checkpointとして受理しない。

manifestに`size_bytes`がある場合はファイルサイズも検査する。

Release上の6 checkpointを一時ディレクトリへ取得して一括検証するcommandは次のとおりである。

```bash
uv run python outputs/app/verify_release.py \
  --manifest outputs/release-manifest.json
```

一時ディレクトリと取得済みファイルは、検証終了後に自動で削除する。

2026年7月27日に公開Releaseの6 checkpointをこのcommandで再取得し、manifestに記載したファイルサイズとSHA-256へ一致することを確認した。

## テスト

```bash
uv run python -m unittest discover -s outputs/app/tests -v
```
