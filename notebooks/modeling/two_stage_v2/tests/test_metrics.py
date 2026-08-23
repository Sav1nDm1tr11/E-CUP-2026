import unittest

import numpy as np

from src.metrics import paired_cluster_bootstrap_delta, rmsle


class MetricsTests(unittest.TestCase):
    def test_rmsle_is_finite_and_nonnegative(self):
        self.assertGreaterEqual(rmsle(np.array([0.0, 2.0]), np.array([0.0, 1.0])), 0.0)

    def test_cluster_bootstrap_returns_delta_interval_and_count(self):
        result = paired_cluster_bootstrap_delta(
            actual=np.array([0.0, 1.0, 4.0, 2.0]),
            candidate=np.array([0.0, 2.0, 3.0, 2.0]),
            baseline=np.array([0.0, 1.5, 4.0, 2.5]),
            group=np.array(["a", "a", "b", "b"]),
            n_resamples=25,
            seed=42,
        )
        self.assertEqual(result.n_resamples, 25)
        self.assertLessEqual(result.ci_low, result.ci_high)
        self.assertEqual(result.delta, result.point_delta)

