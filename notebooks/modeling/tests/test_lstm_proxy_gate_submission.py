import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from lstm_proxy_gate_submission import (  # noqa: E402
    build_submission,
    compose_proxy_log_prediction,
    load_project_inputs,
    passes_submission_gate,
    select_best_correction,
)


class ProxyGateCompositionTests(unittest.TestCase):
    def test_zero_blend_weight_preserves_baseline(self):
        baseline = np.array([0.0, 1.0, 4.0])

        actual = compose_proxy_log_prediction(
            baseline_log=baseline,
            baseline_probability=np.array([0.2, 0.5, 0.8]),
            oracle_probability=np.array([0.9, 0.1, 0.3]),
            probability_weight=0.0,
            correction_weight=1.0,
            max_ratio=2.0,
        )

        np.testing.assert_allclose(actual, baseline, atol=1e-12)

    def test_oracle_probability_changes_prediction_in_same_direction(self):
        baseline = np.array([2.0, 2.0])

        actual = compose_proxy_log_prediction(
            baseline_log=baseline,
            baseline_probability=np.array([0.5, 0.5]),
            oracle_probability=np.array([0.25, 0.75]),
            probability_weight=1.0,
            correction_weight=1.0,
            max_ratio=2.0,
        )

        np.testing.assert_allclose(actual, np.array([1.0, 3.0]), atol=1e-12)

    def test_probability_ratio_is_bounded_and_output_is_nonnegative(self):
        actual = compose_proxy_log_prediction(
            baseline_log=np.array([3.0, 3.0, 0.0]),
            baseline_probability=np.array([0.9, 0.1, 0.5]),
            oracle_probability=np.array([0.01, 0.99, 0.9]),
            probability_weight=1.0,
            correction_weight=1.0,
            max_ratio=2.0,
        )

        np.testing.assert_allclose(actual, np.array([1.5, 6.0, 0.0]), atol=1e-12)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0.0).all())

    def test_partial_correction_interpolates_in_log_ratio_space(self):
        actual = compose_proxy_log_prediction(
            baseline_log=np.array([4.0]),
            baseline_probability=np.array([0.5]),
            oracle_probability=np.array([0.125]),
            probability_weight=1.0,
            correction_weight=0.5,
            max_ratio=4.0,
        )

        np.testing.assert_allclose(actual, np.array([2.0]), atol=1e-12)

    def test_nonfinite_probability_is_rejected_immediately(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compose_proxy_log_prediction(
                baseline_log=np.array([1.0]),
                baseline_probability=np.array([0.5]),
                oracle_probability=np.array([np.nan]),
                probability_weight=1.0,
                correction_weight=0.2,
                max_ratio=2.0,
            )

    def test_parameter_selection_finds_exact_hand_checked_correction(self):
        choice = select_best_correction(
            baseline_log=np.array([1.0, 1.0]),
            target_log=np.array([0.5, 1.5]),
            baseline_probability=np.array([0.5, 0.5]),
            oracle_probability=np.array([0.25, 0.75]),
            probability_weights=np.array([0.0, 1.0]),
            correction_weights=np.array([0.0, 1.0]),
            max_ratios=np.array([2.0]),
        )

        self.assertEqual(choice["probability_weight"], 1.0)
        self.assertEqual(choice["correction_weight"], 1.0)
        self.assertEqual(choice["max_ratio"], 2.0)
        self.assertAlmostEqual(choice["rmsle"], 0.0, places=12)


class SubmissionContractTests(unittest.TestCase):
    def test_build_submission_preserves_expected_user_order(self):
        actual = build_submission(
            expected_user_ids=np.array([7, 2]),
            predictions=np.array([1.5, 0.0]),
        )

        expected = pd.DataFrame(
            {"user_id": [7, 2], "predict": [1.5, 0.0]}
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_build_submission_rejects_nonfinite_predictions(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            build_submission(
                expected_user_ids=np.array([2]),
                predictions=np.array([np.nan]),
            )

    def test_gate_accepts_small_but_significant_consistent_improvement(self):
        self.assertTrue(
            passes_submission_gate(
                delta=-0.0004,
                bootstrap_upper=-0.0001,
                better_folds=5,
                guardrails_pass=True,
            )
        )

    def test_gate_rejects_improvement_with_uncertain_bootstrap(self):
        self.assertFalse(
            passes_submission_gate(
                delta=-0.0004,
                bootstrap_upper=0.0001,
                better_folds=5,
                guardrails_pass=True,
            )
        )


class ProjectInputIntegrationTests(unittest.TestCase):
    def test_loader_aligns_january_and_february_users(self):
        project_root = Path(__file__).resolve().parents[3]
        inputs = load_project_inputs(project_root)

        self.assertEqual(inputs["jan_features"].shape, (250_000, 32))
        self.assertEqual(inputs["feb_features"].shape, (250_000, 32))
        self.assertEqual(inputs["target_log"].shape, (250_000,))
        np.testing.assert_array_equal(inputs["jan_user_ids"], inputs["feb_user_ids"])
        np.testing.assert_array_equal(
            inputs["feb_user_ids"], inputs["submission_user_ids"]
        )


if __name__ == "__main__":
    unittest.main()
