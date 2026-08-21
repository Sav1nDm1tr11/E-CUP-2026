import unittest

import numpy as np

from src.inference import combine_two_stage_predictions, predict_two_stage


class FakeClassifier:
    def __init__(self, probability):
        self.probability = np.asarray(probability, dtype=np.float64)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.column_stack((1.0 - self.probability, self.probability))


class ReorderedClassifier(FakeClassifier):
    def __init__(self, probability):
        super().__init__(probability)
        self.classes_ = np.array([1, 0])

    def predict_proba(self, features):
        return np.column_stack((self.probability, 1.0 - self.probability))


class FakeRegressor:
    def __init__(self, predicted_log):
        self.predicted_log = np.asarray(predicted_log, dtype=np.float64)

    def predict(self, features):
        return self.predicted_log


class TwoStageInferenceTests(unittest.TestCase):
    def test_combines_stages_with_scale_and_gamma(self):
        result = combine_two_stage_predictions(
            probability=np.array([0.25, 1.0]),
            predicted_positive_log=np.array([2.0, 3.0]),
            scale=1.2,
            gamma=2.0,
        )

        expected_log = 1.2 * np.array([0.25, 1.0]) ** 2 * np.array([2.0, 3.0])
        np.testing.assert_allclose(result.prediction_log, expected_log)
        np.testing.assert_allclose(result.prediction, np.expm1(expected_log))

    def test_clips_negative_regressor_output_before_combining(self):
        result = combine_two_stage_predictions(
            probability=np.array([0.5]),
            predicted_positive_log=np.array([-1.0]),
            scale=1.0,
            gamma=1.0,
        )

        np.testing.assert_array_equal(result.predicted_positive_log, [0.0])
        np.testing.assert_array_equal(result.prediction, [0.0])

    def test_rejects_invalid_probability(self):
        with self.assertRaisesRegex(ValueError, "диапазоне"):
            combine_two_stage_predictions(
                probability=np.array([1.1]),
                predicted_positive_log=np.array([1.0]),
                scale=1.0,
                gamma=1.0,
            )

    def test_predict_two_stage_returns_all_components(self):
        result = predict_two_stage(
            classifier=FakeClassifier([0.5, 0.75]),
            regressor=FakeRegressor([2.0, 4.0]),
            features=np.zeros((2, 1)),
            scale=1.0,
            gamma=1.0,
        )

        np.testing.assert_allclose(result.probability, [0.5, 0.75])
        np.testing.assert_allclose(result.predicted_positive_log, [2.0, 4.0])
        np.testing.assert_allclose(result.prediction_log, [1.0, 3.0])
        np.testing.assert_allclose(result.prediction, np.expm1([1.0, 3.0]))

    def test_predict_uses_probability_column_for_positive_class(self):
        result = predict_two_stage(
            classifier=ReorderedClassifier([0.25, 0.75]),
            regressor=FakeRegressor([2.0, 4.0]),
            features=np.zeros((2, 1)),
            scale=1.0,
            gamma=1.0,
        )

        np.testing.assert_allclose(result.probability, [0.25, 0.75])

    def test_predict_rejects_classifier_without_classes(self):
        classifier = FakeClassifier([0.5])
        del classifier.classes_

        with self.assertRaisesRegex(ValueError, "classes_"):
            predict_two_stage(
                classifier=classifier,
                regressor=FakeRegressor([1.0]),
                features=np.zeros((1, 1)),
                scale=1.0,
                gamma=1.0,
            )

    def test_predict_rejects_classifier_without_single_positive_class(self):
        classifier = FakeClassifier([0.5])
        classifier.classes_ = np.array([0, 2])

        with self.assertRaisesRegex(ValueError, "класс 1"):
            predict_two_stage(
                classifier=classifier,
                regressor=FakeRegressor([1.0]),
                features=np.zeros((1, 1)),
                scale=1.0,
                gamma=1.0,
            )


if __name__ == "__main__":
    unittest.main()
