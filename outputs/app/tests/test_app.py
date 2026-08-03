from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
warnings.filterwarnings("ignore", category=ResourceWarning)

from app import build_demo  # noqa: E402
from inference import load_genre_names, load_threshold_config  # noqa: E402


class FakeEngine:
    def __init__(self, genre_names):
        self.genre_names = genre_names
        self.device = torch.device("cpu")
        self.prediction_calls = 0

    def predict(self, _image):
        self.prediction_calls += 1
        return np.linspace(0.05, 0.95, len(self.genre_names), dtype=np.float32)


class AppConstructionTest(unittest.TestCase):
    def test_builds_blocks_without_loading_checkpoint_or_starting_server(self):
        warnings.simplefilter("ignore", ResourceWarning)
        genre_names = load_genre_names(APP_DIR / "genres.json")
        config = load_threshold_config(APP_DIR / "threshold.json", genre_names)
        demo = build_demo(FakeEngine(genre_names), config)
        self.addCleanup(demo.close)
        component_types = [
            component["type"] for component in demo.get_config_file()["components"]
        ]
        self.assertEqual(component_types.count("slider"), 19)
        self.assertIn("image", component_types)
        self.assertIn("file", component_types)
        self.assertGreaterEqual(component_types.count("dataframe"), 3)

    def test_builds_with_an_initial_example_image_and_results(self):
        genre_names = load_genre_names(APP_DIR / "genres.json")
        config = load_threshold_config(APP_DIR / "threshold.json", genre_names)
        engine = FakeEngine(genre_names)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "example.png"
            Image.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)
            demo = build_demo(engine, config, example_image=image_path)
            self.addCleanup(demo.close)
            self.assertEqual(engine.prediction_calls, 1)


if __name__ == "__main__":
    unittest.main()
