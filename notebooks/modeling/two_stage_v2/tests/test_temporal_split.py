import unittest

import pandas as pd

from src.temporal_split import ExpandingCutoffSplit, build_nested_folds


def toy_cutoff_frame():
    dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
    return pd.DataFrame({"cutoff_date": dates.repeat(2), "value": range(8)})


class TemporalSplitTests(unittest.TestCase):
    def test_nested_fold_uses_last_past_cutoff_as_inner_validation(self):
        dates = pd.to_datetime(["2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"])
        fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
        self.assertEqual(fold.inner_valid_date, pd.Timestamp("2025-06-18"))
        self.assertEqual(fold.outer_valid_date, pd.Timestamp("2025-07-18"))
        self.assertLess(max(fold.inner_train_dates), fold.inner_valid_date)

    def test_expanding_split_keeps_outer_report_labels_out_of_fit(self):
        frame = toy_cutoff_frame()
        split = ExpandingCutoffSplit(build_nested_folds(
            frame["cutoff_date"].unique(), [pd.Timestamp("2025-07-18")]
        ))
        train_idx, report_idx = next(split.split(frame, groups=frame["cutoff_date"]))
        self.assertEqual(set(frame.iloc[train_idx]["cutoff_date"]), set(frame["cutoff_date"].unique()[:-1]))
        self.assertEqual(set(frame.iloc[report_idx]["cutoff_date"]), {pd.Timestamp("2025-07-18")})

    def test_missing_two_past_cutoffs_is_rejected(self):
        with self.assertRaises(ValueError):
            build_nested_folds(pd.to_datetime(["2025-06-18"]), [pd.Timestamp("2025-07-18")])

