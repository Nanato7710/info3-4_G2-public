from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from tune_threshold import optimize_thresholds  # noqa: E402


class ThresholdOptimizationTest(unittest.TestCase):
    def test_optimizes_each_genre_independently_and_breaks_ties_high(self):
        targets = np.array(
            [
                [0, 1],
                [1, 1],
                [1, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        )
        scores = np.array(
            [
                [0.1, 0.9],
                [0.8, 0.75],
                [0.6, 0.45],
                [0.2, 0.1],
            ],
            dtype=np.float32,
        )
        thresholds, per_genre_f1, macro_f1 = optimize_thresholds(
            targets,
            scores,
            ("A", "B"),
            step=0.1,
        )
        self.assertEqual(thresholds, {"A": 0.6, "B": 0.7})
        self.assertEqual(per_genre_f1, {"A": 1.0, "B": 1.0})
        self.assertEqual(macro_f1, 1.0)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            optimize_thresholds(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.float32),
                ("A", "B"),
            )


if __name__ == "__main__":
    unittest.main()
