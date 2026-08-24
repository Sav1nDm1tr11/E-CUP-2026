import unittest

import numpy as np
import pandas as pd

from src.models import (
    FittedFoldModel,
    FittedProductionModel,
    ModelFitRejection,
    ResourceRejection,
    fit_catboost_classifier,
    fit_ebm_classifier,
    fit_lgbm_classifier,
    fit_positive_lgbm_regressor,
    refit_catboost_classifier,
    refit_ebm_classifier,
    refit_lgbm_classifier,
    refit_positive_lgbm_regressor,
    _rss_tree_bytes,
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
    def test_fixed_production_refits_use_all_rows_and_no_selection_arguments(self):
        frame = pd.DataFrame({"user_id": [2, 1, 2], "cutoff_date": pd.to_datetime(["2025-03-01", "2025-02-01", "2025-03-01"]),
                              "feature_0": [2.0, 1.0, 3.0]})
        y = np.array([0, 1, 1])
        RecordingEstimator.fit_calls.clear()
        result = refit_lgbm_classifier(frame, y, params={"learning_rate": 0.1}, n_estimators=17,
                                       estimator_factory=RecordingEstimator)
        self.assertIsInstance(result, FittedProductionModel)
        self.assertEqual(result.rounds, 17)
        self.assertEqual(len(RecordingEstimator.fit_calls[-1][0]), 3)
        self.assertEqual(RecordingEstimator.fit_calls[-1][2], {})
        self.assertNotIn("cutoff_date", RecordingEstimator.fit_calls[-1][0].columns)
        self.assertNotIn("user_id", RecordingEstimator.fit_calls[-1][0].columns)

        target_result = refit_positive_lgbm_regressor(frame, np.array([0.0, 2.0, 3.0]), params={}, n_estimators=9,
                                                      estimator_factory=RecordingEstimator)
        self.assertEqual(target_result.rounds, 9)
        self.assertEqual(len(RecordingEstimator.fit_calls[-1][0]), 2)
        np.testing.assert_allclose(RecordingEstimator.fit_calls[-1][1], np.log1p([2.0, 3.0]))

    def test_fixed_catboost_refit_forces_iterations_and_no_eval_set(self):
        frame = pd.DataFrame({"user_id": ["z", "a", "b"], "cutoff_date": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]),
                              "feature_0": [3.0, 1.0, 2.0]})
        RecordingEstimator.fit_calls.clear()
        result = refit_catboost_classifier(frame, [1, 0, 1], params={"depth": 6}, iterations=23,
                                            estimator_factory=RecordingEstimator)
        self.assertEqual(result.rounds, 23)
        self.assertEqual(RecordingEstimator.fit_calls[-1][2], {})
        self.assertEqual(len(RecordingEstimator.fit_calls[-1][0]), 3)
        self.assertEqual(RecordingEstimator.init_calls[-1]["iterations"], 23)
        self.assertFalse(RecordingEstimator.init_calls[-1]["use_best_model"])

    def test_fixed_ebm_refit_uses_production_contract_and_gate(self):
        calls = []
        class EBM:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
            def estimate_mem(self, X, y, data_multiplier=1):
                calls.append(("estimate", X.copy(), np.asarray(y).copy(), data_multiplier))
                return 100
            def fit(self, X, y, **kwargs):
                calls.append(("fit", X.copy(), np.asarray(y).copy(), kwargs))
                return self
        accepted = refit_ebm_classifier(
            pd.DataFrame({"cutoff_date": pd.to_datetime(["2025-01-01", "2025-02-01"]), "feature_0": [1., 2.]}),
            [0, 1], params={}, max_rounds=31, available_memory=1000,
            measured_overhead=10, estimator_factory=EBM,
        )
        self.assertIsInstance(accepted, FittedProductionModel)
        self.assertEqual(accepted.rounds, 31)
        init = calls[0][1]
        self.assertEqual({key: init[key] for key in ("validation_size", "outer_bags", "inner_bags", "early_stopping_rounds")},
                         {"validation_size": 0, "outer_bags": 1, "inner_bags": 0, "early_stopping_rounds": 0})
        self.assertNotIn("callback", calls[-1][3])
        rejected = refit_ebm_classifier(pd.DataFrame({"feature_0": [1.]}), [1], params={}, max_rounds=2,
                                        available_memory=100, measured_overhead=1, estimator_factory=EBM)
        self.assertIsInstance(rejected, ResourceRejection)
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

    def test_ordered_requires_finite_plain_governance_metrics(self):
        class OrderedFake(RecordingEstimator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.best_iteration_ = 2
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"feature_0": np.arange(8, dtype=float)},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        kwargs = dict(estimator_factory=OrderedFake, param_distributions={"depth": [6]}, trials=1,
                      inner_positive_log=np.ones(len(frame)), actual_gmv=np.ones(len(frame)))
        self.assertIsInstance(fit_catboost_classifier(frame, [0, 1] * 4, fold, ordered=True, **kwargs), ResourceRejection)
        result = fit_catboost_classifier(frame, [0, 1] * 4, fold, ordered=True,
                                         plain_metrics={"rmsle": 1.0, "fit_seconds": 1.0,
                                                        "peak_memory_bytes": 10_000}, **kwargs)
        self.assertIsInstance(result, FittedFoldModel)

    def test_catboost_report_prediction_restores_original_outer_row_order(self):
        class Keyed(RecordingEstimator):
            def predict_proba(self, X):
                p = np.asarray(X["feature_0"], dtype=float) / 10.0
                return np.column_stack([1 - p, p])
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        frame = pd.DataFrame({"user_id": ["z", "a", "z", "a", "z", "a", "z", "a"],
                              "feature_0": [1., 2., 3., 4., 5., 6., 9., 8.]},
                             index=pd.DatetimeIndex(dates.repeat(2), name="cutoff_date"))
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        result = fit_catboost_classifier(frame, [0, 1] * 4, fold, estimator_factory=Keyed,
                                         param_distributions={"depth": [6]}, trials=1,
                                         inner_positive_log=np.ones(len(frame)), actual_gmv=np.ones(len(frame)))
        np.testing.assert_allclose(result.report_prediction, [0.9, 0.8])

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
                                      available_memory=1000, measured_overhead=250,
                                      inner_positive_log=np.ones(len(frame)), actual_gmv=np.ones(len(frame)))
        self.assertIsInstance(rejected, ResourceRejection)
        accepted = fit_ebm_classifier(frame, [0, 1] * 4, fold, estimator_factory=EBM,
                                      available_memory=1000, measured_overhead=100,
                                      inner_positive_log=np.ones(len(frame)), actual_gmv=np.ones(len(frame)))
        self.assertIsInstance(accepted, FittedFoldModel)
        self.assertEqual(EBM.fits[0][2]["bags"].shape[1], 8)
        self.assertTrue((EBM.fits[0][2]["bags"] == -1).any())

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
                                      available_memory=1_000_000,
                                      inner_positive_log=np.ones(len(frame)), actual_gmv=np.ones(len(frame)))
        self.assertIsInstance(accepted, FittedFoldModel)
        self.assertEqual(accepted.best_iteration, 5)
        self.assertGreaterEqual(accepted.metadata["measured_overhead"], 0)

    def test_rss_tree_aggregates_parent_and_recursive_children(self):
        class Proc:
            def memory_info(self): return type("Info", (), {"rss": 100})()
            def children(self, recursive=True):
                return [type("Child", (), {"memory_info": lambda self: type("Info", (), {"rss": 20})()})(),
                        type("Child", (), {"memory_info": lambda self: type("Info", (), {"rss": 30})()})()]
        self.assertEqual(_rss_tree_bytes(Proc()), 150)
