# 依存関係

依存関係はリポジトリ直下の`pyproject.toml`と`uv.lock`で管理する。

アプリが直接使用する主なpackageは次のとおりである。

- `gradio`
- `numpy`
- `pillow`
- `scikit-learn`
- `timm`
- `torch`
- `torchvision`

環境はリポジトリ直下で次のcommandにより作成する。

```bash
uv sync --locked
```
