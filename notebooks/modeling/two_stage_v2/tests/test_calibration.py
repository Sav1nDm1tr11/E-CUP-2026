import unittest

import numpy as np

from src.calibration import IdentityCalibrator, SigmoidCalibrator
from sklearn.base import clone


class CalibrationTests(unittest.TestCase):
    def test_identity_calibrator_preserves_probabilities_and_is_cloneable(self):
        calibrator = IdentityCalibrator().fit(np.array([0.1, 0.8, 1.0]), np.array([0, 1, 1]))
        result = calibrator.predict_proba(np.array([0.0, 0.25, 1.0]))
        np.testing.assert_allclose(result, [[1.0, 0.0], [0.75, 0.25], [0.0, 1.0]])
        np.testing.assert_array_equal(calibrator.predict(np.array([0.49, 0.5, 0.9])), [0, 1, 1])
        self.assertEqual(clone(calibrator).get_params(), calibrator.get_params())

    def test_identity_calibrator_rejects_invalid_probability_or_labels(self):
        with self.assertRaises(ValueError):
            IdentityCalibrator().fit(np.array([-0.1, 0.5]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            IdentityCalibrator().fit(np.array([0.1, 0.5]), np.array([0, 2]))

    def test_sigmoid_calibrator_returns_two_class_probabilities(self):
        calibrator = SigmoidCalibrator().fit(
            np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])
        )
        result = calibrator.predict_proba(np.array([0.0, 0.5, 1.0]))
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.isfinite([calibrator.a, calibrator.b]).all())
        self.assertTrue((result >= 0).all() and (result <= 1).all())
        self.assertGreater(result[-1, 1], result[0, 1])
        self.assertNotEqual(result[0, 1], result[1, 1])
        self.assertEqual(clone(calibrator).get_params(), calibrator.get_params())

    def test_sigmoid_calibrator_requires_exact_binary_labels(self):
        with self.assertRaises(ValueError):
            SigmoidCalibrator().fit(np.array([0.1, 0.9]), np.array([1, 2]))
