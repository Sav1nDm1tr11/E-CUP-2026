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
