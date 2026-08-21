import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import (
    load_json,
    load_submission_csv,
    save_json,
    save_lightgbm_model,
    save_submission_csv,
)


class FakeBooster:
    def model_to_string(self):
        return "tree\nмодель\n"


class FakeLightGBMModel:
    booster_ = FakeBooster()


class ArtifactTests(unittest.TestCase):
    def test_json_round_trip_supports_numpy_dates_and_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metadata.json"
            save_json(
                {
                    "score": np.float32(1.5),
                    "cutoff": date(2026, 1, 14),
                    "model": Path("models/model.joblib"),
                },
                path,
            )

            self.assertEqual(
                load_json(path),
                {
                    "score": 1.5,
                    "cutoff": "2026-01-14",
                    "model": "models\\model.joblib",
                },
            )

    def test_saves_lightgbm_model_as_utf8_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "модель.txt"
            save_lightgbm_model(FakeLightGBMModel(), path)

            self.assertEqual(path.read_text(encoding="utf-8"), "tree\nмодель\n")

    def test_submission_csv_round_trip_preserves_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "submission.csv"
            expected = pd.DataFrame({"user_id": [2, 7], "predict": [1.25, 0.0]})
            save_submission_csv(expected, path)

            pd.testing.assert_frame_equal(load_submission_csv(path), expected)


if __name__ == "__main__":
    unittest.main()
