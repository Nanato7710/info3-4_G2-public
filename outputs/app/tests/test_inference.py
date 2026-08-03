from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from inference import (  # noqa: E402
    AppConfigurationError,
    UserInputError,
    format_results,
    load_genre_names,
    load_threshold_config,
    preprocess_image,
    select_device,
)


class InferenceHelpersTest(unittest.TestCase):
    def test_select_device_prefers_cuda_over_mps(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(select_device(), torch.device("cuda"))

    def test_select_device_uses_mps_then_cpu(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(select_device(), torch.device("mps"))
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(select_device(), torch.device("cpu"))

    def test_preprocess_converts_rgb_rgba_and_grayscale(self):
        for mode in ("RGB", "RGBA", "L"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (16, 12))
                tensor = preprocess_image(image)
                self.assertEqual(tuple(tensor.shape), (3, 384, 384))
                self.assertTrue(torch.isfinite(tensor).all())

    def test_preprocess_rejects_unsupported_format(self):
        image = Image.new("RGB", (8, 8))
        image.format = "GIF"
        with self.assertRaises(UserInputError):
            preprocess_image(image)

    def test_preprocess_rejects_unsupported_file_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.gif"
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaises(UserInputError):
                preprocess_image(path)

    def test_format_results_uses_per_genre_thresholds(self):
        top, candidates, rows = format_results(
            [0.4, 0.8, 0.2],
            [0.5, 0.7, 0.1],
            ["A", "B", "C"],
        )
        self.assertEqual([row[1] for row in top], ["B", "A", "C"])
        self.assertEqual([row[1] for row in candidates], ["B", "C"])
        by_genre = {row[1]: row for row in rows}
        self.assertEqual(by_genre["A"][4], "")
        self.assertEqual(by_genre["B"][4], "候補")
        self.assertEqual(by_genre["C"][4], "候補")

    def test_load_checked_threshold_config(self):
        names = load_genre_names(APP_DIR / "genres.json")
        config = load_threshold_config(APP_DIR / "threshold.json", names)
        self.assertEqual(len(config.thresholds), 19)
        self.assertEqual(config.thresholds[0], 0.33)
        self.assertEqual(config.thresholds[-1], 0.07)

    def test_threshold_config_rejects_genre_order_mismatch(self):
        document = json.loads(
            (APP_DIR / "threshold.json").read_text(encoding="utf-8")
        )
        document["genre_names"][0], document["genre_names"][1] = (
            document["genre_names"][1],
            document["genre_names"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            names = load_genre_names(APP_DIR / "genres.json")
            with self.assertRaises(AppConfigurationError):
                load_threshold_config(path, names)

    def test_threshold_config_rejects_test_derived_settings(self):
        document = json.loads(
            (APP_DIR / "threshold.json").read_text(encoding="utf-8")
        )
        document["source_split"] = "test"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            names = load_genre_names(APP_DIR / "genres.json")
            with self.assertRaises(AppConfigurationError):
                load_threshold_config(path, names)


if __name__ == "__main__":
    unittest.main()
