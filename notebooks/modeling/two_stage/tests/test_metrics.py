import unittest

import numpy as np

from src.metrics import (
    classification_metrics,
    make_soft_pipeline_rmsle_metric,
    pipeline_metrics,
    positive_regression_metrics,
    rmsle,
)


class RmsleTests(unittest.TestCase):
    def test_uses_log1p_distance(self):
        self.assertAlmostEqual(
            rmsle(np.array([0.0]), np.array([3.0])),
            np.log(4.0),
        )

    def test_rejects_negative_predictions(self):
        with self.assertRaisesRegex(ValueError, "prediction"):
            rmsle(np.array([0.0]), np.array([-0.01]))

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "одинаковую форму"):
            rmsle(np.array([0.0]), np.array([0.0, 1.0]))


class DiagnosticMetricTests(unittest.TestCase):
    def test_classification_metrics_return_f1_not_f2(self):
        result = classification_metrics(
            np.array([1, 1, 0, 0], dtype=np.int8),
            np.array([0.9, 0.8, 0.7, 0.1]),
            threshold=0.5,
        )

        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertNotIn("f2", result)

    def test_classification_metrics_match_notebook_contract(self):
        result = classification_metrics(
            np.array([0, 1, 1, 0], dtype=np.int8),
            np.array([0.1, 0.9, 0.4, 0.6]),
            threshold=0.5,
        )

        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["brier"], 0.185)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_positive_regression_metrics_are_exact_for_perfect_prediction(self):
        actual = np.array([3.0, 8.0])
        result = positive_regression_metrics(actual, np.log1p(actual))

        self.assertAlmostEqual(result["rmse_log"], 0.0)
        self.assertAlmostEqual(result["mae_log"], 0.0)
        self.assertAlmostEqual(result["sum_ratio"], 1.0)

    def test_pipeline_metrics_are_exact_for_perfect_prediction(self):
        actual = np.array([0.0, 3.0, 8.0])
        result = pipeline_metrics(actual, actual.copy())

        self.assertAlmostEqual(result["rmsle"], 0.0)
        self.assertAlmostEqual(result["sum_ratio"], 1.0)
        self.assertAlmostEqual(result["log_bias"], 0.0)

    def test_soft_pipeline_metric_matches_log_space_formula(self):
        metric = make_soft_pipeline_rmsle_metric(np.array([0.0, 0.5, 1.0]))
        name, value, greater_is_better = metric(
            np.array([0.0, 2.0, 3.0]),
            np.array([-1.0, 3.0, 2.5]),
        )

        expected = np.sqrt(
            np.mean(np.square(np.array([0.0, 1.5, 2.5]) - np.array([0.0, 2.0, 3.0])))
        )
        self.assertEqual(name, "soft_pipeline_rmsle")
        self.assertAlmostEqual(value, expected)
        self.assertFalse(greater_is_better)


if __name__ == "__main__":
    unittest.main()
