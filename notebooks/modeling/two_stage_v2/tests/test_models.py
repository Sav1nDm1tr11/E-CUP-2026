import unittest

import numpy as np
import pandas as pd

from src.models import FittedFoldModel, fit_lgbm_classifier
from src.temporal_split import build_nested_folds


class RecordingEstimator:
    fit_calls = []
    init_calls = []

    def __init__(self, **kwargs):
        self.init_calls.append(kwargs)
        self.best_iteration_ = 7

    def fit(self, X, y, **kwargs):
        self.fit_calls.append((X.copy(), y.copy(), kwargs))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.75), np.full(len(X), 0.25)])


class ModelContractTests(unittest.TestCase):
    def test_two_phase_fit_excludes_outer_report_and_refit_has_more_rows(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float), "target_nonzero": [0, 1] * 4},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        RecordingEstimator.fit_calls.clear()
        RecordingEstimator.init_calls.clear()
        result = fit_lgbm_classifier(
            frame.drop(columns="target_nonzero"), frame["target_nonzero"],
            fold=fold, estimator_factory=RecordingEstimator,
        )
        self.assertIsInstance(result, FittedFoldModel)
        self.assertEqual(len(RecordingEstimator.fit_calls), 2)
        self.assertEqual(len(RecordingEstimator.init_calls), 2)
        inner_X, _inner_y, inner_kwargs = RecordingEstimator.fit_calls[0]
        refit_X, _refit_y, refit_kwargs = RecordingEstimator.fit_calls[1]
        self.assertEqual(set(inner_X.index), set(pd.to_datetime(["2025-04-19", "2025-05-19"])))
        self.assertIn("eval_set", inner_kwargs)
        eval_X, _eval_y = inner_kwargs["eval_set"][0]
        self.assertEqual(set(eval_X.index), {pd.Timestamp("2025-06-18")})
        self.assertEqual(set(refit_X.index), set(pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18"])))
        self.assertNotIn("eval_set", refit_kwargs)
        self.assertEqual(RecordingEstimator.init_calls[1].get("n_estimators"), result.best_iteration)
        self.assertNotIn(pd.Timestamp("2025-07-18"), refit_X.index.tolist())
