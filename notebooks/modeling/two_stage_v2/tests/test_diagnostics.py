import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
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
            self.frame["target_nonzero"], self.frame["p_lgbm"],
            actual=self.frame["target_gmv_30d"], predicted=self.frame["prediction"], n_bins=4
        )
        self.assertEqual(list(result.columns), ["bin", "bin_left", "bin_right", "count", "positive_rate", "mean_probability", "mean_squared_log_error", "squared_log_error"])
        self.assertEqual(result["bin"].tolist(), [0, 1, 2, 3])
        self.assertEqual(result["count"].sum(), len(self.frame))
        expected = np.square(np.log1p(self.frame["target_gmv_30d"]) - np.log1p(self.frame["prediction"])).sum()
        self.assertAlmostEqual(result["squared_log_error"].sum(), expected)

    def test_model_fold_metrics_preserve_model_fold_metric_mapping(self):
        wide = pd.DataFrame({
            "model": ["lgbm", "lgbm", "catboost", "catboost"],
            "fold": [1, 2, 1, 2],
            "cutoff_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01", "2025-02-01"]),
            "rmsle": [0.1, 0.2, 0.3, 0.4],
            "logloss": [0.5, 0.6, 0.7, 0.8],
        })
        result = d.build_model_fold_metrics(wide)
        observed = {(row.model, int(row.fold), row.metric): row.value for row in result.itertuples()}
        self.assertEqual(observed[("lgbm", 1, "rmsle")], 0.1)
        self.assertEqual(observed[("lgbm", 2, "logloss")], 0.6)
        self.assertEqual(observed[("catboost", 1, "rmsle")], 0.3)
        self.assertEqual(observed[("catboost", 2, "logloss")], 0.8)

    def test_duplicate_index_segment_groups_use_positions(self):
        duplicate = self.frame.copy()
        duplicate.index = [0, 0, 1, 1]
        result = d.build_segment_heatmap(duplicate)
        self.assertEqual(result["count"].sum(), len(duplicate))
        self.assertEqual(result["segment"].tolist(), ["A", "B"])

    def test_all_table_builders_are_robust_for_empty_and_constant_inputs(self):
        empty = pd.DataFrame(columns=self.frame.columns)
        self.assertEqual(list(d.build_cutoff_summary(empty).columns), ["cutoff_date", "rows", "positive_count", "positive_share", "total_gmv", "mean_gmv"])
        self.assertEqual(list(d.build_missing_summary(empty).columns), ["feature", "missing_count", "missing_share"])
        self.assertEqual(len(d.build_reliability_table([], [], n_bins=3)), 3)
        self.assertEqual(len(d.build_probability_bins([], [], actual=[], predicted=[], n_bins=3)), 3)
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
            (d.plot_probability_bins, (d.build_probability_bins(self.frame["target_nonzero"], self.frame["p_lgbm"], actual=self.frame["target_gmv_30d"], predicted=self.frame["prediction"]),)),
            (d.plot_blend_weights, (d.build_blend_weights({"weights": [0.7, 0.3], "model_names": ["A", "B"]}),)),
            (d.plot_simplex_landscape, (d.build_simplex_landscape([{"weights": [1.0, 0.0], "objective": 0.4}, {"weights": [0.0, 1.0], "objective": 0.5}]),)),
            (d.plot_bootstrap_interval, (d.build_bootstrap_interval({"point_delta": -0.1, "ci_low": -0.2, "ci_high": 0.0}),)),
            (d.plot_segment_heatmap, (d.build_segment_heatmap(self.frame),)),
            (d.plot_inference_comparison, (d.build_inference_comparison(self.frame["prediction"], self.frame["target_gmv_30d"]),)),
            (d.plot_probability_distribution, ({"raw": self.frame["p_lgbm"], "calibrated": self.frame["p_catboost"]},)),
            (d.plot_learning_curves, (pd.DataFrame({"model": ["A", "A", "B", "B"], "fold": [1, 1, 1, 1], "iteration": [1, 2, 1, 2], "metric": [0.9, 0.8, 0.95, 0.85]}),)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, (function, args) in enumerate(calls):
                path = Path(tmp) / f"plot-{index}.png"
                figure = function(*args, path=path)
                self.assertIsInstance(figure, matplotlib.figure.Figure)
                self.assertTrue(path.exists() and path.stat().st_size > 0)
                # Library functions must return ownership to the caller; no implicit display.
                figure.clf()
                plt.close(figure)

    def test_semantic_multiseries_plots_have_labels_and_artists(self):
        cutoff = d.plot_cutoff_summary(d.build_cutoff_summary(self.frame))
        self.assertGreaterEqual(len(cutoff.axes), 1)
        self.assertTrue(any(axis.get_legend() for axis in cutoff.axes))
        timeline = d.plot_fold_timeline(pd.DataFrame({
            "fold": [1, 2], "train_start": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "train_end": pd.to_datetime(["2025-01-10", "2025-02-10"]),
            "inner_start": pd.to_datetime(["2025-01-11", "2025-02-11"]),
            "inner_end": pd.to_datetime(["2025-01-15", "2025-02-15"]),
            "report_start": pd.to_datetime(["2025-01-20", "2025-02-20"]),
        }))
        self.assertGreaterEqual(len(timeline.axes[0].lines), 3)
        metrics = d.plot_model_fold_metrics(d.build_model_fold_metrics(pd.DataFrame({
            "model": ["A", "A", "B", "B"], "fold": [1, 1, 1, 1],
            "rmsle": [0.1, 0.2, 0.3, 0.4], "logloss": [0.5, 0.6, 0.7, 0.8],
        })))
        self.assertTrue(metrics.axes[0].get_legend())
        simplex = d.plot_simplex_landscape(d.build_simplex_landscape([
            {"weights": [1, 0, 0], "objective": 0.4}, {"weights": [0, 1, 0], "objective": 0.3},
            {"weights": [0, 0, 1], "objective": 0.2},
        ]))
        self.assertEqual(len(simplex.axes[0].collections), 1)
        self.assertEqual(simplex.axes[0].get_xlabel(), "w₁ + 0.5·w₂")
        segment_figure = d.plot_segment_heatmap(pd.DataFrame({"segment": ["A", "B"], "model": ["new", "new"], "target_gmv_30d": [1, 2], "prediction": [1, 1]}))
        self.assertEqual(len(segment_figure.axes[0].images), 1)
        for figure in (cutoff, timeline, metrics, simplex, segment_figure):
            plt.close(figure)

    def test_blend_history_uses_populated_trained_through_when_report_cutoff_empty(self):
        history = pd.DataFrame({
            "model": ["A", "A"], "weight": [0.6, 0.7],
            "trained_through": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "report_cutoff": [pd.NaT, pd.NaT],
        })
        figure = d.plot_blend_weights(history)
        self.assertEqual(len(figure.axes[0].lines), 1)
        self.assertTrue(pd.to_datetime(figure.axes[0].lines[0].get_xdata()).notna().all())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
