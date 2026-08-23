import unittest

import numpy as np
import pandas as pd

from src.inference import predict_ensemble
from src.validation import validate_submission


FEATURE_NAMES = ("feature_a", "feature_b", *(f"feature_{i}" for i in range(2, 91)))


def feature_frame(user_ids, *, reverse=False, extra=False):
    values = {"user_id": list(user_ids)}
    for index, name in enumerate(FEATURE_NAMES):
        values[name] = [float(index + 1)] * len(values["user_id"])
    frame = pd.DataFrame(values)
    if reverse:
        columns = ["user_id", "feature_b", "feature_a", *FEATURE_NAMES[2:]]
        frame = frame.loc[:, columns]
    if extra:
        frame["extra"] = 3.0
    return frame


class InferenceTests(unittest.TestCase):
    def test_inference_preserves_expected_user_order(self):
        frame = feature_frame(["u2", "u1"])
        classifiers = [RecordingClassifier(0.25), RecordingClassifier(0.5)]
        result = predict_ensemble(frame, classifiers=classifiers, positive_regressor=RecordingRegressor(),
                                  calibrator=RecordingCalibrator(), weights=np.array([0.5, 0.5]),
                                  feature_names=FEATURE_NAMES, expected_user_ids=["u2", "u1"])
        self.assertEqual(result.user_id.tolist(), ["u2", "u1"])
        self.assertEqual(classifiers[0].columns_seen, list(FEATURE_NAMES))
        np.testing.assert_allclose(result.prediction, np.expm1(np.array([0.375, 0.375]) * 2.0))
        for field in ("raw_base_probabilities", "blended_probability", "calibrated_probability",
                      "positive_log", "prediction_log", "prediction"):
            self.assertTrue(hasattr(result, field), field)

    def test_inference_rejects_wrong_feature_order(self):
        frame = feature_frame(["u1"], reverse=True)
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[RecordingClassifier(0.5)], positive_regressor=RecordingRegressor(),
                              calibrator=RecordingCalibrator(), weights=np.array([1.0]),
                              feature_names=FEATURE_NAMES, expected_user_ids=["u1"])

    def test_inference_rejects_extra_feature_and_out_of_range_probability(self):
        frame = feature_frame(["u1"], extra=True)
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[RecordingClassifier(0.5)], positive_regressor=RecordingRegressor(),
                              weights=[1.0], feature_names=FEATURE_NAMES)
        class BadClassifier(RecordingClassifier):
            def predict_proba(self, frame):
                return np.column_stack([np.zeros(len(frame)), np.full(len(frame), 1.1)])
        frame = frame.drop(columns="extra")
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[BadClassifier(0.5)], positive_regressor=RecordingRegressor(),
                              weights=[1.0], feature_names=FEATURE_NAMES)

    def test_inference_requires_91_unique_features(self):
        frame = feature_frame(["u1"])
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[RecordingClassifier(0.5)], positive_regressor=RecordingRegressor(),
                             weights=[1.0], feature_names=FEATURE_NAMES[:-1])
        duplicate_names = (*FEATURE_NAMES[:-1], FEATURE_NAMES[0])
        with self.assertRaises(ValueError):
            predict_ensemble(frame, classifiers=[RecordingClassifier(0.5)], positive_regressor=RecordingRegressor(),
                             weights=[1.0], feature_names=duplicate_names)

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
