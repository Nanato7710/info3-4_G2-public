from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from inference import sha256_file

APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReleaseAsset:
    model_id: str
    seed: int
    repository: str
    release_tag: str
    asset_name: str
    sha256: str
    size_bytes: int | None = None

    @property
    def url(self) -> str:
        return (
            f"https://github.com/{self.repository}/releases/download/"
            f"{quote(self.release_tag, safe='')}/{quote(self.asset_name)}"
        )


def load_release_assets(manifest_path: Path) -> list[ReleaseAsset]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Release manifestを読み込めません: {manifest_path}") from exc
    entries = manifest.get("release_assets", manifest.get("models"))
    if not isinstance(entries, list):
        raise ValueError("manifestにはrelease_assetsの配列が必要です")
    assets: list[ReleaseAsset] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("release_assetsの各要素はobjectである必要があります")
        try:
            entry_seed = int(entry.get("seed", -1))
            raw_size = entry.get("size_bytes")
            size_bytes = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError("seedまたはsize_bytesが不正です") from exc
        asset = ReleaseAsset(
            model_id=str(entry.get("model_id", "")),
            seed=entry_seed,
            repository=str(entry.get("repository", manifest.get("repository", ""))),
            release_tag=str(entry.get("release_tag", manifest.get("release_tag", ""))),
            asset_name=str(entry.get("asset_name", "")),
            sha256=str(entry.get("sha256", "")),
            size_bytes=size_bytes,
        )
        if (
            not asset.model_id
            or not asset.repository
            or not asset.release_tag
            or not asset.asset_name
        ):
            raise ValueError(
                "model_id、repository、release_tag、asset_nameが必要です"
            )
        if Path(asset.asset_name).name != asset.asset_name:
            raise ValueError("asset_nameにはファイル名だけを指定してください")
        if len(asset.sha256) != 64:
            raise ValueError("assetのSHA-256が不正です")
        if asset.size_bytes is not None and asset.size_bytes <= 0:
            raise ValueError("assetのsize_bytesが不正です")
        assets.append(asset)

    keys = [(asset.model_id, asset.seed) for asset in assets]
    if len(keys) != len(set(keys)):
        raise ValueError("model_idとseedの組が重複しています")
    return assets


def load_release_asset(manifest_path: Path, model_id: str, seed: int) -> ReleaseAsset:
    matches = [
        asset
        for asset in load_release_assets(manifest_path)
        if asset.model_id == model_id and asset.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"{model_id} seed {seed}のassetを一意に特定できません")
    return matches[0]


def download_asset(asset: ReleaseAsset, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if asset.size_bytes is not None and destination.stat().st_size != asset.size_bytes:
            raise ValueError(f"既存checkpointのファイルサイズが一致しません: {destination}")
        if sha256_file(destination) == asset.sha256:
            return False
        raise ValueError(f"既存checkpointのSHA-256が一致しません: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "info3-final-report-checkpoint-downloader"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if asset.size_bytes is not None and temporary.stat().st_size != asset.size_bytes:
            raise ValueError(
                "ダウンロードしたcheckpointのファイルサイズが一致しません"
            )
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != asset.sha256:
            raise ValueError(
                "ダウンロードしたcheckpointのSHA-256が一致しません"
            )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify a checkpoint from a GitHub Release."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="final-tri-model")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = load_release_asset(args.manifest.resolve(), args.model, args.seed)
    destination = (
        args.destination.resolve()
        if args.destination
        else APP_DIR / "checkpoints" / asset.asset_name
    )
    downloaded = download_asset(asset, destination)
    action = "downloaded" if downloaded else "already verified"
    print(f"{action}: {destination}")


if __name__ == "__main__":
    main()
