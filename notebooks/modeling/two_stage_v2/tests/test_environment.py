from importlib.metadata import PackageNotFoundError, version
import unittest


class EnvironmentTests(unittest.TestCase):
    def test_required_distributions_are_installed(self):
        required = {
            "lightgbm": "4.6.0",
            "catboost": "1.2.10",
            "interpret": "0.7.8",
            "scikit-learn": "1.8.0",
            "nbclient": "0.10.4",
            "nbformat": "5.10.4",
        }
        missing = []
        mismatched = []
        for distribution, expected in required.items():
            try:
                actual = version(distribution)
            except PackageNotFoundError:
                missing.append(distribution)
                continue
            if actual != expected:
                mismatched.append((distribution, expected, actual))
        self.assertEqual(missing, [])
        self.assertEqual(mismatched, [])
