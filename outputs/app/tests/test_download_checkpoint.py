from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from download_checkpoint import (  # noqa: E402
    download_asset,
    load_release_asset,
    load_release_assets,
)


class DownloadCheckpointTest(unittest.TestCase):
    def test_loads_unique_release_asset(self):
        document = {
            "release_assets": [
                {
                    "model_id": "final-tri-model",
                    "seed": 44,
                    "repository": "owner/repository",
                    "release_tag": "tag name",
                    "asset_name": "model seed 44.pth",
                    "sha256": "a" * 64,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            asset = load_release_asset(manifest, "final-tri-model", 44)
        self.assertEqual(
            asset.url,
            "https://github.com/owner/repository/releases/download/"
            "tag%20name/model%20seed%2044.pth",
        )

    def test_existing_valid_file_is_not_downloaded(self):
        payload = b"checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        document = {
            "release_assets": [
                {
                    "model_id": "final-tri-model",
                    "seed": 44,
                    "repository": "owner/repository",
                    "release_tag": "tag",
                    "asset_name": "model.pth",
                    "sha256": digest,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            destination = root / "model.pth"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            destination.write_bytes(payload)
            asset = load_release_asset(manifest, "final-tri-model", 44)
            self.assertFalse(download_asset(asset, destination))

    def test_reads_repository_and_tag_from_manifest_root(self):
        document = {
            "repository": "Nanato7710/info3-4_G2-public",
            "release_tag": "final-report-2026-07-28",
            "release_assets": [
                {
                    "model_id": "baseline",
                    "seed": 42,
                    "asset_name": "baseline-seed-42.pth",
                    "size_bytes": 12,
                    "sha256": "a" * 64,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            assets = load_release_assets(manifest)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].repository, document["repository"])
        self.assertEqual(assets[0].release_tag, document["release_tag"])
        self.assertEqual(assets[0].size_bytes, 12)

    def test_rejects_existing_file_with_wrong_size(self):
        payload = b"checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        document = {
            "release_assets": [
                {
                    "model_id": "final-tri-model",
                    "seed": 44,
                    "repository": "owner/repository",
                    "release_tag": "tag",
                    "asset_name": "model.pth",
                    "size_bytes": len(payload) + 1,
                    "sha256": digest,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            destination = root / "model.pth"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            destination.write_bytes(payload)
            asset = load_release_asset(manifest, "final-tri-model", 44)
            with self.assertRaisesRegex(ValueError, "ファイルサイズ"):
                download_asset(asset, destination)


if __name__ == "__main__":
    unittest.main()
