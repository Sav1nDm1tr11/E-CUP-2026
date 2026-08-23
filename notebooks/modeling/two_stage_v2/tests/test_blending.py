import unittest

import numpy as np

from src.blending import combine_predictions, simplex_grid


class BlendingTests(unittest.TestCase):
    def test_simplex_grid_is_unique_and_sums_to_one(self):
        weights = simplex_grid(model_count=3, step=0.5)
        self.assertEqual(len(weights), 6)
        self.assertEqual(len({tuple(row) for row in weights}), 6)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertTrue((weights >= 0).all())

    def test_soft_log_prediction_matches_manual_formula(self):
        result = combine_predictions(
            probability=np.array([0.25, 1.0]), positive_log=np.array([2.0, -1.0])
        )
        np.testing.assert_allclose(result.prediction_log, [0.5, 0.0])
        np.testing.assert_allclose(result.prediction, np.expm1([0.5, 0.0]))

