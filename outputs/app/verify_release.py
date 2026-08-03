from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from download_checkpoint import download_asset, load_release_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify every checkpoint in a release manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = load_release_assets(args.manifest.resolve())
    if not assets:
        raise ValueError("検証対象のRelease assetがありません")

    with tempfile.TemporaryDirectory(prefix="info3-release-verify-") as directory:
        temporary_directory = Path(directory)
        for asset in assets:
            destination = temporary_directory / asset.asset_name
            download_asset(asset, destination)
            print(
                f"verified: {asset.model_id} seed {asset.seed} "
                f"{destination.stat().st_size} bytes {asset.sha256}"
            )

    print(f"verified {len(assets)} release assets")


if __name__ == "__main__":
    main()
