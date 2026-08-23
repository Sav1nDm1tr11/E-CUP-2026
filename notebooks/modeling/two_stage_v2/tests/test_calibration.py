import unittest

import numpy as np

from src.calibration import SigmoidCalibrator
from sklearn.base import clone


class CalibrationTests(unittest.TestCase):
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
