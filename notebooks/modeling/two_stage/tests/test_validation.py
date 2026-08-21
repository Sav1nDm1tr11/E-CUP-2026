import unittest

import numpy as np
import pandas as pd

from src.validation import (
    make_submission,
    validate_feature_matrix,
    validate_submission,
    validate_temporal_split,
)


class FeatureValidationTests(unittest.TestCase):
    def test_accepts_numeric_features_with_nan(self):
        features = pd.DataFrame({"a": [1.0, np.nan], "b": [0, 1]})
        validate_feature_matrix(features, ("a", "b"))

    def test_rejects_wrong_feature_order(self):
        features = pd.DataFrame({"b": [0], "a": [1.0]})
        with self.assertRaisesRegex(ValueError, "порядок"):
            validate_feature_matrix(features, ("a", "b"))

    def test_rejects_infinity(self):
        features = pd.DataFrame({"a": [np.inf]})
        with self.assertRaisesRegex(ValueError, "infinity"):
            validate_feature_matrix(features, ("a",))


class SubmissionValidationTests(unittest.TestCase):
    def test_make_submission_preserves_expected_user_order(self):
        submission = make_submission(
            expected_user_ids=np.array([7, 2]),
            predictions=np.array([1.5, 0.0]),
        )

        self.assertEqual(submission.columns.tolist(), ["user_id", "predict"])
        self.assertEqual(submission["user_id"].tolist(), [7, 2])
        self.assertEqual(submission["predict"].tolist(), [1.5, 0.0])

    def test_validate_submission_rejects_wrong_user_order(self):
        submission = pd.DataFrame({"user_id": [2, 7], "predict": [0.0, 1.5]})
        with self.assertRaisesRegex(ValueError, "порядок"):
            validate_submission(submission, expected_user_ids=np.array([7, 2]))

    def test_make_submission_rejects_negative_prediction(self):
        with self.assertRaisesRegex(ValueError, "отрицательные"):
            make_submission(np.array([2]), np.array([-0.01]))


class TemporalValidationTests(unittest.TestCase):
    def test_accepts_strictly_future_validation_period(self):
        validate_temporal_split(
            pd.to_datetime(["2025-01-01", "2025-02-01"]),
            pd.to_datetime(["2025-03-01"]),
        )

    def test_rejects_overlapping_periods(self):
        with self.assertRaisesRegex(ValueError, "позже"):
            validate_temporal_split(
                pd.to_datetime(["2025-01-01", "2025-02-01"]),
                pd.to_datetime(["2025-02-01"]),
            )


if __name__ == "__main__":
    unittest.main()
