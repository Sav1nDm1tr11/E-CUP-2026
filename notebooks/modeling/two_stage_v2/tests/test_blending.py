import unittest

import numpy as np
import pandas as pd

from src.blending import combine_predictions, fit_walk_forward_blend, simplex_grid


class BlendingTests(unittest.TestCase):
    def test_simplex_grid_is_unique_and_sums_to_one(self):
        weights = simplex_grid(model_count=3, step=0.5)
        self.assertEqual(len(weights), 6)
        self.assertEqual(len({tuple(row) for row in weights}), 6)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights, simplex_grid(model_count=3, step=0.5))
        self.assertIn((1.0, 0.0, 0.0), {tuple(row) for row in weights})
        self.assertIn((0.0, 0.0, 1.0), {tuple(row) for row in weights})

    def test_soft_log_prediction_matches_manual_formula(self):
        result = combine_predictions(
            probability=np.array([0.25, 1.0]), positive_log=np.array([2.0, -1.0])
        )
        np.testing.assert_allclose(result.prediction_log, [0.5, 0.0])
        np.testing.assert_allclose(result.prediction, np.expm1([0.5, 0.0]))

    def test_walk_forward_blend_uses_past_oof_only_and_records_cutoff(self):
        past = pd.DataFrame({
            "cutoff_date": pd.to_datetime(["2025-05-19", "2025-06-18"]),
            "p_lgbm": [0.2, 0.8], "p_catboost": [0.3, 0.7],
            "predicted_positive_log": [1.0, 2.0], "target_gmv_30d": [0.0, 3.0],
            "target_nonzero": [0, 1],
        })
        state = fit_walk_forward_blend(past, positive_log=past["predicted_positive_log"],
                                       actual_gmv=past["target_gmv_30d"],
                                       config={"trained_through": pd.Timestamp("2025-06-18")})
        self.assertEqual(pd.Timestamp(state.trained_through), pd.Timestamp("2025-06-18"))
        self.assertEqual(len(state.weights), 2)
        np.testing.assert_allclose(np.sum(state.weights), 1.0)
        self.assertTrue(np.isfinite([state.calibrator_a, state.calibrator_b, state.epsilon, state.objective]).all())
        self.assertNotIn(pd.Timestamp("2025-07-18"), getattr(state, "report_dates", ()))
        future = pd.concat([past, past.iloc[[0]].assign(cutoff_date=pd.Timestamp("2025-07-18"))])
        with self.assertRaises(ValueError):
            fit_walk_forward_blend(future, positive_log=future["predicted_positive_log"],
                                   actual_gmv=future["target_gmv_30d"],
                                   config={"trained_through": pd.Timestamp("2025-06-18")})

    def test_walk_forward_blend_requires_trusted_cutoff_and_all_required_models(self):
        past = pd.DataFrame({
            "cutoff_date": pd.to_datetime(["2025-05-19", "2025-06-18"]),
            "p_lgbm": [0.2, 0.8], "p_catboost": [0.3, 0.7],
            "predicted_positive_log": [1.0, 2.0], "target_gmv_30d": [0.0, 3.0],
            "target_nonzero": [0, 1],
        })
        with self.assertRaises(ValueError):
            fit_walk_forward_blend(past, past["predicted_positive_log"], past["target_gmv_30d"], {})
        with self.assertRaises(ValueError):
            fit_walk_forward_blend(past.drop(columns="cutoff_date"), past["predicted_positive_log"],
                                   past["target_gmv_30d"], {"trained_through": "2025-06-18"})
        with self.assertRaises(ValueError):
            fit_walk_forward_blend(past.drop(columns="p_catboost"), past["predicted_positive_log"],
                                   past["target_gmv_30d"], {"trained_through": "2025-06-18"})

    def test_blend_state_contains_serializable_metric_diagnostics(self):
        past = pd.DataFrame({
            "cutoff_date": pd.to_datetime(["2025-05-19", "2025-06-18"]),
            "p_lgbm": [0.2, 0.8], "p_catboost": [0.3, 0.7],
            "predicted_positive_log": [1.0, 2.0], "target_gmv_30d": [0.0, 3.0],
            "target_nonzero": [0, 1],
        })
        state = fit_walk_forward_blend(past, past["predicted_positive_log"], past["target_gmv_30d"],
                                       {"trained_through": pd.Timestamp("2025-06-18")})
        self.assertTrue(state.diagnostics)
        self.assertIn("fold_metrics", state.diagnostics[0])
        self.assertEqual(state.to_dict()["trained_through"], "2025-06-18T00:00:00")

    def test_governance_defaults_and_fold_stability_objective_are_exact(self):
        from src.config import ExperimentConfig
        self.assertEqual(ExperimentConfig.__dataclass_fields__["max_logloss_degradation"].default, 0.0010)
        self.assertEqual(ExperimentConfig.__dataclass_fields__["max_brier_degradation"].default, 0.0005)
        past = pd.DataFrame({
            "cutoff_date": pd.to_datetime(["2025-05-19", "2025-05-19", "2025-06-18", "2025-06-18"]),
            "p_lgbm": [0.1, 0.2, 0.8, 0.9], "p_catboost": [0.2, 0.3, 0.7, 0.8],
            "predicted_positive_log": [1.0, 1.0, 2.0, 2.0],
            "target_gmv_30d": [0.0, 0.0, 3.0, 4.0], "target_nonzero": [0, 0, 1, 1],
        })
        state = fit_walk_forward_blend(past, past["predicted_positive_log"], past["target_gmv_30d"],
                                       {"trained_through": pd.Timestamp("2025-06-18")})
        selected = next(item for item in state.diagnostics if tuple(item["weights"]) == state.weights)
        scores = np.asarray([item["rmsle"] for item in selected["fold_metrics"]])
        self.assertAlmostEqual(state.objective, scores.mean() + 0.25 * scores.std(ddof=1))
        self.assertTrue(all("logloss_current" in item and "brier_current" in item
                            for item in selected["fold_metrics"]))

    def test_governance_gate_rejects_candidate_on_one_cutoff(self):
        past = pd.DataFrame({
            "cutoff_date": pd.to_datetime(["2025-05-19", "2025-05-19", "2025-06-18", "2025-06-18"]),
            "p_lgbm": [0.01, 0.01, 0.99, 0.99], "p_catboost": [0.99, 0.99, 0.01, 0.01],
            "predicted_positive_log": [1.0] * 4,
            "target_gmv_30d": [0.0, 0.0, 3.0, 4.0], "target_nonzero": [0, 0, 1, 1],
        })
        state = fit_walk_forward_blend(past, past["predicted_positive_log"], past["target_gmv_30d"],
                                       {"trained_through": pd.Timestamp("2025-06-18")})
        self.assertTrue(any(not item["accepted"] for item in state.diagnostics))
        for item in state.diagnostics:
            if item["accepted"]:
                self.assertTrue(all(fold["accepted"] for fold in item["fold_metrics"]))
