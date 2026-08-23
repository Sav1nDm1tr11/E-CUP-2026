import unittest

import numpy as np
import pandas as pd

from src.models import (
    FittedFoldModel,
    ModelFitRejection,
    ResourceRejection,
    fit_catboost_classifier,
    fit_ebm_classifier,
    fit_lgbm_classifier,
    fit_positive_lgbm_regressor,
)
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

    def predict(self, X):
        return np.full(len(X), 1.0)


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
        self.assertEqual(result.best_iteration, 7)
        self.assertGreater(result.best_iteration, 0)
        self.assertEqual(RecordingEstimator.init_calls[1]["n_estimators"], 7)
        inner_X, _inner_y, inner_kwargs = RecordingEstimator.fit_calls[0]
        refit_X, _refit_y, refit_kwargs = RecordingEstimator.fit_calls[1]
        self.assertEqual(set(inner_X.index), set(pd.to_datetime(["2025-04-19", "2025-05-19"])))
        self.assertEqual(len(inner_X), 4)
        self.assertIn("eval_set", inner_kwargs)
        eval_X, _eval_y = inner_kwargs["eval_set"][0]
        self.assertEqual(set(eval_X.index), {pd.Timestamp("2025-06-18")})
        self.assertEqual(len(eval_X), 2)
        self.assertEqual(set(refit_X.index), set(pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18"])))
        self.assertEqual(len(refit_X), 6)
        self.assertNotIn("eval_set", refit_kwargs)
        self.assertEqual(RecordingEstimator.init_calls[1].get("n_estimators"), result.best_iteration)
        self.assertNotIn(pd.Timestamp("2025-07-18"), refit_X.index.tolist())

    def test_lightgbm_refuses_estimator_that_drops_required_eval_set(self):
        class RejectingEstimator(RecordingEstimator):
            def fit(self, X, y, **kwargs):
                if "eval_set" in kwargs:
                    raise TypeError("eval_set unsupported")
                return super().fit(X, y, **kwargs)

        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        with self.assertRaises(ModelFitRejection):
            fit_lgbm_classifier(frame, [0, 1] * 4, fold, estimator_factory=RejectingEstimator)

    def test_positive_regressor_filters_each_phase_and_refits_fixed_iteration(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        target = np.array([0, 2, 0, 3, 0, 4, 0, 5], dtype=float)
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        RecordingEstimator.fit_calls.clear()
        RecordingEstimator.init_calls.clear()
        result = fit_positive_lgbm_regressor(frame, target, fold, estimator_factory=RecordingEstimator)
        inner_X, inner_y, _ = RecordingEstimator.fit_calls[0]
        refit_X, refit_y, refit_kwargs = RecordingEstimator.fit_calls[1]
        self.assertEqual(len(inner_X), 2)
        self.assertEqual(len(refit_X), 3)
        np.testing.assert_allclose(inner_y, np.log1p([2, 3]))
        np.testing.assert_allclose(refit_y, np.log1p([2, 3, 4]))
        self.assertNotIn("eval_set", refit_kwargs)
        self.assertEqual(RecordingEstimator.init_calls[1]["n_estimators"], result.best_iteration)

    def test_catboost_keeps_winning_params_from_end_to_end_inner_objective(self):
        calls = []
        class CatFake:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.best_iteration_ = 4
                calls.append(("init", kwargs))
            def fit(self, X, y, **kwargs):
                calls.append(("fit", self.kwargs, kwargs))
                return self
            def predict_proba(self, X):
                p = 0.8 if self.kwargs.get("depth") == 2 else 0.2
                return np.column_stack([np.full(len(X), 1 - p), np.full(len(X), p)])
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        result = fit_catboost_classifier(
            frame, [0, 1] * 4, fold, estimator_factory=CatFake,
            param_distributions={"depth": [1, 2]}, trials=2,
            inner_positive_log=np.full(len(frame), 2.0),
            actual_gmv=np.full(len(frame), np.expm1(1.6)),
        )
        self.assertIsInstance(result, FittedFoldModel)
        self.assertEqual(result.metadata["objective"], "inner_end_to_end_rmsle")
        self.assertEqual(result.metadata["selected_params"], {"depth": 2})
        self.assertEqual(result.estimator.kwargs["depth"], 2)

    def test_ebm_memory_gate_and_temporal_bags_are_structured(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        class EBM:
            fits = []
            def __init__(self, **kwargs): self.kwargs = kwargs; self.best_iteration_ = 3
            def estimate_mem(self, X, y, data_multiplier=1): return 500
            def fit(self, X, y, **kwargs): self.fits.append((X, y, kwargs)); return self
            def predict_proba(self, X): return np.column_stack([np.full(len(X), .5), np.full(len(X), .5)])
        rejected = fit_ebm_classifier(frame, [0, 1] * 4, fold, estimator_factory=EBM,
                                      available_memory=1000, measured_overhead=250)
        self.assertIsInstance(rejected, ResourceRejection)
        accepted = fit_ebm_classifier(frame, [0, 1] * 4, fold, estimator_factory=EBM,
                                      available_memory=1000, measured_overhead=100)
        self.assertIsInstance(accepted, FittedFoldModel)
        self.assertTrue(EBM.fits[0][2]["bags"].tolist().count(-1) > 0)

    def test_ebm_default_overhead_measurement_and_array_rounds_are_safe(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        class EBM:
            def __init__(self, **kwargs): self.best_iteration_ = np.array([[2, 4], [3, 5]])
            def estimate_mem(self, X, y, data_multiplier=1): return 1
            def fit(self, X, y, **kwargs): return self
            def predict_proba(self, X): return np.column_stack([np.full(len(X), .5), np.full(len(X), .5)])
        accepted = fit_ebm_classifier(frame, [0, 1] * 4, fold, estimator_factory=EBM,
                                      available_memory=1_000_000)
        self.assertIsInstance(accepted, FittedFoldModel)
        self.assertEqual(accepted.best_iteration, 5)
        self.assertGreaterEqual(accepted.metadata["measured_overhead"], 0)
