import unittest

import numpy as np
import pandas as pd

from src.models import FittedFoldModel, fit_lgbm_classifier
from src.temporal_split import build_nested_folds


class RecordingEstimator:
    fit_calls = []

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((len(X), kwargs))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.75), np.full(len(X), 0.25)])


class ModelContractTests(unittest.TestCase):
    def test_two_phase_fit_excludes_outer_report_and_refit_has_more_rows(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"cutoff_date": dates.repeat(2), "target_nonzero": [0, 1] * 4})
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        RecordingEstimator.fit_calls.clear()
        result = fit_lgbm_classifier(
            frame.drop(columns="target_nonzero"), frame["target_nonzero"],
            fold=fold, estimator_factory=RecordingEstimator,
        )
        self.assertIsInstance(result, FittedFoldModel)
        self.assertEqual(len(RecordingEstimator.fit_calls), 2)
        inner_rows, refit_rows = [call[0] for call in RecordingEstimator.fit_calls]
        self.assertGreater(refit_rows, inner_rows)
        self.assertNotIn(pd.Timestamp("2025-07-18"), fold.inner_train_dates)

