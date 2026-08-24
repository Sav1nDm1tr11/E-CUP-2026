import unittest
import time

import numpy as np

from src.metrics import paired_cluster_bootstrap_delta, rmsle


class MetricsTests(unittest.TestCase):
    def test_rmsle_is_finite_and_nonnegative(self):
        expected = np.sqrt(np.mean((np.log1p([0.0, 2.0]) - np.log1p([0.0, 1.0])) ** 2))
        self.assertAlmostEqual(rmsle(np.array([0.0, 2.0]), np.array([0.0, 1.0])), expected)

    def test_cluster_bootstrap_returns_delta_interval_and_count(self):
        result = paired_cluster_bootstrap_delta(
            actual=np.array([0.0, 1.0, 4.0, 2.0]),
            candidate=np.array([0.0, 1.0, 4.0, 2.0]),
            baseline=np.array([0.0, 2.0, 5.0, 3.0]),
            group=np.array(["a", "a", "b", "b"]),
            fold=np.array([1, 1, 2, 2]),
            n_resamples=25,
            seed=42,
        )
        self.assertEqual(result.n_resamples, 25)
        self.assertLessEqual(result.ci_low, result.ci_high)
        self.assertEqual(result.delta, result.point_delta)
        self.assertLess(result.delta, 0.0)
        expected_delta = np.mean([
            rmsle(np.array([0.0, 1.0]), np.array([0.0, 1.0])) - rmsle(np.array([0.0, 1.0]), np.array([0.0, 2.0])),
            rmsle(np.array([4.0, 2.0]), np.array([4.0, 2.0])) - rmsle(np.array([4.0, 2.0]), np.array([5.0, 3.0])),
        ])
        self.assertAlmostEqual(result.delta, expected_delta)
        repeat = paired_cluster_bootstrap_delta(
            actual=np.array([0.0, 1.0, 4.0, 2.0]), candidate=np.array([0.0, 1.0, 4.0, 2.0]),
            baseline=np.array([0.0, 2.0, 5.0, 3.0]), group=np.array(["a", "a", "b", "b"]), fold=np.array([1, 1, 2, 2]),
            n_resamples=25, seed=42,
        )
        self.assertEqual(result, repeat)
        reassigned = paired_cluster_bootstrap_delta(
            actual=np.array([0.0, 1.0, 4.0, 2.0]), candidate=np.array([0.0, 1.0, 4.0, 2.0]),
            baseline=np.array([0.0, 2.0, 5.0, 3.0]), group=np.array(["a", "b", "a", "b"]),
            fold=np.array([1, 1, 2, 2]), n_resamples=25, seed=42,
        )
        self.assertNotEqual((result.ci_low, result.ci_high), (reassigned.ci_low, reassigned.ci_high))

    def test_cluster_bootstrap_large_row_scaling_uses_user_fold_sufficient_statistics(self):
        n_users = 250_000
        n_rows = n_users * 2
        actual = np.zeros(n_rows, dtype=float)
        candidate = np.ones(n_rows, dtype=float)
        baseline = np.full(n_rows, 2.0, dtype=float)
        groups = np.repeat(np.arange(n_users), 2)
        folds = np.tile([0, 1], n_users)
        started = time.perf_counter()
        result = paired_cluster_bootstrap_delta(actual, candidate, baseline, groups, folds,
                                                n_resamples=20, seed=17)
        elapsed = time.perf_counter() - started
        self.assertEqual(result.n_resamples, 20)
        self.assertLess(elapsed, 15.0)

    def test_cluster_bootstrap_supports_two_thousand_resamples(self):
        n_users = 1_000
        groups = np.repeat(np.arange(n_users), 2)
        folds = np.tile([0, 1], n_users)
        result = paired_cluster_bootstrap_delta(
            np.zeros(len(groups)), np.ones(len(groups)), np.full(len(groups), 2.0),
            groups, folds, n_resamples=2_000, seed=19,
        )
        self.assertEqual(result.n_resamples, 2_000)
        self.assertTrue(np.isfinite([result.ci_low, result.ci_high]).all())
