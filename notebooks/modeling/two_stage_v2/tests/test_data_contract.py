import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import load_cutoff_frame, load_feature_columns
from src.validation import validate_feature_contract, validate_submission


class DataContractTests(unittest.TestCase):
    def test_feature_contract_requires_exact_ordered_feature_names(self):
        names = tuple(f"feature_{i}" for i in range(91))
        self.assertEqual(validate_feature_contract(names, names), names)
        with self.assertRaises(ValueError):
            validate_feature_contract(names[:-1], names)

    def test_cutoff_loader_returns_float32_features_and_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.parquet"
            pd.DataFrame({
                "user_id": ["u1"],
                "cutoff_date": [pd.Timestamp("2025-07-18")],
                "feature_0": [1.0],
                "target_gmv_30d": [2.0],
            }).to_parquet(path)
            frame = load_cutoff_frame(
                path, [pd.Timestamp("2025-07-18")],
                columns=("user_id", "cutoff_date", "feature_0", "target_gmv_30d"),
            )
            self.assertEqual(frame["feature_0"].dtype, np.dtype("float32"))
            self.assertEqual(frame["target_gmv_30d"].dtype, np.dtype("float32"))

    def test_submission_validation_enforces_order_and_nonnegative_finite_values(self):
        expected = pd.Series(["u1", "u2"], name="user_id")
        actual = pd.DataFrame({"user_id": ["u1", "u2"], "prediction": [0.0, 1.5]})
        self.assertIsNone(validate_submission(actual, expected, expected_rows=2))
        bad = actual.assign(prediction=[np.inf, 1.0])
        with self.assertRaises(ValueError):
            validate_submission(bad, expected, expected_rows=2)

    def test_feature_columns_are_loaded_as_a_stable_tuple(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "feature_columns.json"
            path.write_text(json.dumps({"feature_columns": ["a", "b"]}), encoding="utf-8")
            self.assertEqual(load_feature_columns(path), ("a", "b"))
