import unittest

import numpy as np
import pandas as pd

from src.inference import predict_ensemble
from src.validation import validate_submission


class InferenceTests(unittest.TestCase):
    def test_inference_preserves_expected_user_order(self):
        frame = pd.DataFrame({"user_id": ["u2", "u1"], "feature_0": [2.0, 1.0]})
        result = predict_ensemble(frame, expected_user_ids=["u2", "u1"])
        self.assertEqual(result.user_id.tolist(), ["u2", "u1"])

    def test_submission_validation_accepts_250000_rows_without_feature_matrix(self):
        user_ids = pd.Series([f"u{i}" for i in range(250_000)], name="user_id")
        submission = pd.DataFrame({"user_id": user_ids, "prediction": np.zeros(250_000)})
        self.assertIsNone(validate_submission(submission, user_ids, expected_rows=250_000))

