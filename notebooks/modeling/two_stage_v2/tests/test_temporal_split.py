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
        self.assertEqual(tuple(fold.train_dates), tuple(dates[:3]))
        self.assertEqual(max(fold.train_dates), fold.inner_valid_date)
        self.assertEqual(tuple(fold.train_dates[:-1]), tuple(dates[:2]))
        self.assertLess(max(fold.train_dates[:-1]), fold.inner_valid_date)

    def test_expanding_split_keeps_outer_report_labels_out_of_fit(self):
        frame = toy_cutoff_frame()
        split = ExpandingCutoffSplit(build_nested_folds(
            frame["cutoff_date"].unique(), [pd.Timestamp("2025-07-18")]
        ))
        train_idx, report_idx = next(split.split(frame, groups=frame["cutoff_date"]))
        self.assertEqual(len(train_idx), 6)
        self.assertEqual(len(report_idx), 2)
        self.assertTrue(set(train_idx).isdisjoint(report_idx))
        self.assertEqual(set(frame.iloc[train_idx]["cutoff_date"]), set(frame["cutoff_date"].unique()[:-1]))
        self.assertEqual(set(frame.iloc[report_idx]["cutoff_date"]), {pd.Timestamp("2025-07-18")})

    def test_split_is_cutoff_based_for_shuffled_rows_and_covers_each_row_once(self):
        frame = toy_cutoff_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)
        fold = build_nested_folds(frame["cutoff_date"].unique(), [pd.Timestamp("2025-07-18")])[0]
        train_idx, report_idx = next(ExpandingCutoffSplit([fold]).split(frame, groups=frame["cutoff_date"]))
        self.assertEqual(set(train_idx) | set(report_idx), set(range(len(frame))))
        self.assertEqual(len(set(train_idx) & set(report_idx)), 0)

    def test_missing_two_past_cutoffs_is_rejected(self):
        with self.assertRaises(ValueError):
            build_nested_folds(pd.to_datetime(["2025-06-18"]), [pd.Timestamp("2025-07-18")])
