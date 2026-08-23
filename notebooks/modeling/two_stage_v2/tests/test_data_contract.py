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
        with self.assertRaises(ValueError):
            validate_feature_contract((names[1], *names[2:], names[0]), names)

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

    def test_cutoff_loader_filters_only_requested_dates_and_casts_binary_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.parquet"
            pd.DataFrame({"cutoff_date": pd.to_datetime(["2025-07-18", "2025-08-17"]),
                          "feature_0": [1.0, 2.0], "target_nonzero": [0, 1]}).to_parquet(path)
            frame = load_cutoff_frame(path, [pd.Timestamp("2025-07-18")],
                                      columns=("cutoff_date", "feature_0", "target_nonzero"))
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["cutoff_date"], pd.Timestamp("2025-07-18"))
            self.assertEqual(frame.iloc[0]["feature_0"], 1.0)
            self.assertEqual(frame["target_nonzero"].dtype, np.dtype("int8"))

    def test_submission_validation_enforces_order_and_nonnegative_finite_values(self):
        expected = pd.Series(["u1", "u2"], name="user_id")
        actual = pd.DataFrame({"user_id": ["u1", "u2"], "prediction": [0.0, 1.5]})
        self.assertIsNone(validate_submission(actual, expected, expected_rows=2))
        bad = actual.assign(prediction=[np.inf, 1.0])
        with self.assertRaises(ValueError):
            validate_submission(bad, expected, expected_rows=2)
        for prediction in ([-0.1, 1.0], [np.nan, 1.0]):
            with self.assertRaises(ValueError):
                validate_submission(actual.assign(prediction=prediction), expected, expected_rows=2)
        with self.assertRaises(ValueError):
            validate_submission(actual.iloc[:1], expected, expected_rows=2)
        with self.assertRaises(ValueError):
            validate_submission(actual.assign(user_id=["u1", "u1"]), expected, expected_rows=2)
        with self.assertRaises(ValueError):
            validate_submission(actual.iloc[::-1].reset_index(drop=True), expected, expected_rows=2)

    def test_feature_validation_allows_missing_values_but_rejects_infinity(self):
        from src.validation import validate_feature_matrix
        names = tuple(f"feature_{i}" for i in range(91))
        frame = pd.DataFrame(np.zeros((2, 91)), columns=names)
        frame.iloc[0, 0] = np.nan
        validate_feature_matrix(frame, names)
        frame.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_feature_matrix(frame, names)

    def test_feature_columns_are_loaded_as_a_stable_tuple(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "feature_columns.json"
            path.write_text(json.dumps({"feature_columns": ["a", "b"]}), encoding="utf-8")
            self.assertEqual(load_feature_columns(path), ("a", "b"))
