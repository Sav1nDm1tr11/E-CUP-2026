import unittest

import numpy as np

from src.calibration import SigmoidCalibrator


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

