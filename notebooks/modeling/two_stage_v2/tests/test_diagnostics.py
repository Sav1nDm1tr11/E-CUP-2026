import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import numpy as np
import pandas as pd

from src import diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "cutoff_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"]),
                "fold": [1, 1, 2, 2],
                "user_id": [1, 2, 3, 4],
                "target_nonzero": [0, 1, 1, 0],
                "target_gmv_30d": [0.0, 2.0, 4.0, 0.0],
                "prediction": [0.0, 1.0, 2.0, 1.0],
                "p_lgbm": [0.1, 0.8, 0.7, 0.2],
                "p_catboost": [0.2, 0.7, 0.6, 0.3],
                "residual_a": [0.0, 1.0, 2.0, -1.0],
                "residual_b": [0.0, 0.5, 1.0, -0.5],
                "segment": ["A", "A", "B", "B"],
            }
        )

    def test_confusion_contribution_has_tidy_columns_and_exact_decomposition(self):
        result = d.build_confusion_contribution(
            actual=self.frame["target_gmv_30d"],
            predicted=self.frame["prediction"],
            y_true=self.frame["target_nonzero"],
            probability=self.frame["p_lgbm"],
            threshold=0.5,
        )
        self.assertEqual(
            list(result.columns),
            ["confusion", "count", "squared_log_error", "mean_squared_log_error", "share"],
        )
        self.assertEqual(result["confusion"].tolist(), ["TP", "FP", "FN", "TN"])
        expected = np.square(np.log1p(self.frame["target_gmv_30d"]) - np.log1p(self.frame["prediction"])).sum()
        self.assertAlmostEqual(result["squared_log_error"].sum(), expected)

    def test_probability_bins_are_deterministic_and_include_empty_bins(self):
        result = d.build_probability_bins(
            self.frame["target_nonzero"], self.frame["p_lgbm"], n_bins=4
        )
        self.assertEqual(list(result.columns), ["bin", "bin_left", "bin_right", "count", "positive_rate", "mean_probability", "mean_squared_log_error"])
        self.assertEqual(result["bin"].tolist(), [0, 1, 2, 3])
        self.assertEqual(result["count"].sum(), len(self.frame))

    def test_all_table_builders_are_robust_for_empty_and_constant_inputs(self):
        empty = pd.DataFrame(columns=self.frame.columns)
        self.assertEqual(list(d.build_cutoff_summary(empty).columns), ["cutoff_date", "rows", "positive_count", "positive_share", "total_gmv", "mean_gmv"])
        self.assertEqual(list(d.build_missing_summary(empty).columns), ["feature", "missing_count", "missing_share"])
        self.assertEqual(len(d.build_reliability_table([], [], n_bins=3)), 3)
        self.assertEqual(len(d.build_probability_bins([], [], n_bins=3)), 3)
        self.assertEqual(len(d.build_residual_correlations(pd.DataFrame({"x": [1, 1], "y": [1, 1]}))), 4)
        self.assertEqual(list(d.build_segment_heatmap(empty).columns), ["segment", "metric", "value", "count"])

    def test_required_plot_builders_return_figures_and_nonempty_png(self):
        calls = [
            (d.plot_cutoff_summary, (d.build_cutoff_summary(self.frame),)),
            (d.plot_missing_summary, (d.build_missing_summary(self.frame),)),
            (d.plot_fold_timeline, (d.build_fold_timeline(self.frame),)),
            (d.plot_model_fold_metrics, (d.build_model_fold_metrics(self.frame),)),
            (d.plot_reliability, (d.build_reliability_table(self.frame["target_nonzero"], self.frame["p_lgbm"]),)),
            (d.plot_residual_correlations, (d.build_residual_correlations(self.frame[["residual_a", "residual_b"]]),)),
            (d.plot_confusion_contribution, (d.build_confusion_contribution(self.frame["target_gmv_30d"], self.frame["prediction"], self.frame["target_nonzero"], self.frame["p_lgbm"]),)),
            (d.plot_probability_bins, (d.build_probability_bins(self.frame["target_nonzero"], self.frame["p_lgbm"]),)),
            (d.plot_blend_weights, (d.build_blend_weights({"weights": [0.7, 0.3], "model_names": ["A", "B"]}),)),
            (d.plot_simplex_landscape, (d.build_simplex_landscape([{"weights": [1.0, 0.0], "objective": 0.4}, {"weights": [0.0, 1.0], "objective": 0.5}]),)),
            (d.plot_bootstrap_interval, (d.build_bootstrap_interval({"point_delta": -0.1, "ci_low": -0.2, "ci_high": 0.0}),)),
            (d.plot_segment_heatmap, (d.build_segment_heatmap(self.frame),)),
            (d.plot_inference_comparison, (d.build_inference_comparison(self.frame["prediction"], self.frame["target_gmv_30d"]),)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, (function, args) in enumerate(calls):
                path = Path(tmp) / f"plot-{index}.png"
                figure = function(*args, path=path)
                self.assertIsInstance(figure, matplotlib.figure.Figure)
                self.assertTrue(path.exists() and path.stat().st_size > 0)
                # Library functions must return ownership to the caller; no implicit display.
                figure.clf()


if __name__ == "__main__":
    unittest.main()
