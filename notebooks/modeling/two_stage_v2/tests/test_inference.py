import unittest

import numpy as np
import pandas as pd

from src.inference import predict_ensemble
from src.validation import validate_submission


class InferenceTests(unittest.TestCase):
    def test_inference_preserves_expected_user_order(self):
        frame = pd.DataFrame({"user_id": ["u2", "u1"], "feature_b": [2.0, 1.0], "feature_a": [1.0, 2.0]})
        classifiers = [RecordingClassifier(0.25), RecordingClassifier(0.5)]
        result = predict_ensemble(frame, classifiers=classifiers, positive_regressor=RecordingRegressor(),
                                  calibrator=RecordingCalibrator(), weights=np.array([0.5, 0.5]),
                                  feature_names=("feature_a", "feature_b"), expected_user_ids=["u2", "u1"])
        self.assertEqual(result.user_id.tolist(), ["u2", "u1"])
        self.assertEqual(classifiers[0].columns_seen, ["feature_a", "feature_b"])
        np.testing.assert_allclose(result.prediction, np.expm1(np.array([0.375, 0.375]) * 2.0))

    def test_inference_rejects_wrong_feature_order(self):
        frame = pd.DataFrame({"user_id": ["u1"], "feature_b": [2.0], "feature_a": [1.0]})
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[], positive_regressor=RecordingRegressor(),
                              calibrator=RecordingCalibrator(), weights=np.array([]),
                              feature_names=("feature_a", "feature_b"), expected_user_ids=["u1"])

    def test_submission_validation_accepts_250000_rows_without_feature_matrix(self):
        user_ids = pd.Series([f"u{i}" for i in range(250_000)], name="user_id")
        submission = pd.DataFrame({"user_id": user_ids, "prediction": np.zeros(250_000)})
        self.assertIsNone(validate_submission(submission, user_ids, expected_rows=250_000))


class RecordingClassifier:
    def __init__(self, probability):
        self.probability = probability
        self.columns_seen = None

    def predict_proba(self, frame):
        self.columns_seen = list(frame.columns)
        return np.column_stack([1 - np.full(len(frame), self.probability), np.full(len(frame), self.probability)])


class RecordingRegressor:
    def predict(self, frame):
        return np.full(len(frame), 2.0)


class RecordingCalibrator:
    def predict_proba(self, probability):
        return np.column_stack([1 - probability, probability])
