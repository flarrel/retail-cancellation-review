"""Regression checks for source handling and point-in-time evaluation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from retail_review.data import RETAIL_COLUMNS, add_fx, prepare_retail
from retail_review.model import FEATURES, metrics, rank, weekly_panel


class SourceTests(unittest.TestCase):
    def test_retail_partitions_rows_and_preserves_missing_customers(self):
        raw = pd.DataFrame([
            ("536365", "85123A", "item", "6", "12/1/2010 8:26", "2.55", "", "United Kingdom"),
            ("C536366", "85123A", "item", "-2", "12/2/2010 8:26", "2.55", "17850", "France"),
            ("536367", "POST", "shipping", "1", "12/2/2010 8:26", "2", "17850", "France"),
            ("536368", "85123A", "adjustment", "-1", "12/2/2010 8:26", "0", "17850", "France"),
            ("536369", "85123A", "item", "6", "impossible", "2.55", "17850", "France"),
        ], columns=RETAIL_COLUMNS, dtype="string")
        rows, quality = prepare_retail(raw)
        self.assertEqual(quality["input"], 5)
        self.assertEqual(quality["retained"], 2)
        self.assertEqual(quality["retained_without_customer"], 1)
        self.assertEqual([quality[key] for key in ("invalid", "nonstandard_sku", "other_adjustment")], [1, 1, 1])
        self.assertEqual(rows["cancelled"].tolist(), [False, True])
        self.assertEqual(rows["day"].min(), pd.Timestamp("2010-12-01"))

    def test_fx_uses_prior_published_rate_and_counts_unmatched_dates(self):
        days = pd.DataFrame({"day": pd.to_datetime(["2011-04-21", "2011-04-25", "2011-05-20"])})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            path = Path(folder) / "fx.csv"
            path.write_text(
                "TIME_PERIOD,OBS_VALUE,OBS_STATUS\n2011-04-20,0.8,A\n2011-04-21,0.9,A\n2011-04-25,,H\n"
            )
            rows, quality = add_fx(days, path)
        self.assertEqual(rows["gbp_per_eur"].iloc[:2].tolist(), [0.8, 0.9])
        self.assertTrue(pd.isna(rows["gbp_per_eur"].iloc[2]))
        self.assertEqual(quality["holiday_rows"], 1)
        self.assertEqual(quality["unmatched_retail_dates"], 1)


class EvaluationTests(unittest.TestCase):
    def test_same_week_labels_cannot_change_features(self):
        weeks = pd.date_range("2011-01-03", periods=15, freq="W-MON")
        rows = pd.DataFrame({
            "sku": "10000", "week": weeks, "invoice": [f"S{i}" for i in range(len(weeks))], "cancelled": False,
        })
        week = weeks[11]
        changed = pd.concat([rows, pd.DataFrame({
            "sku": "10000", "week": week, "invoice": ["C1", "C2", "C3", "C3"], "cancelled": True,
        })], ignore_index=True)
        clean = weekly_panel(rows).set_index("week")
        altered = weekly_panel(changed).set_index("week")
        pd.testing.assert_frame_equal(clean.loc[:week, FEATURES], altered.loc[:week, FEATURES])
        self.assertEqual(altered.at[week, "cancel_docs"], 3)
        self.assertTrue(altered.at[week, "high_load"])
        self.assertEqual(altered.at[weeks[12], "cancels_1w"], 3)
        self.assertTrue(pd.isna(altered.at[weeks[-1], "high_load"]))

    def test_queue_ties_and_short_week_denominators(self):
        week = pd.Timestamp("2011-01-03")
        rows = pd.DataFrame({
            "week": week, "sku": ["B", "C", "A"], "cancel_docs": [4, 0, 0],
            "high_load": [True, False, False], "cancels_4w": 0, "sales_4w": 1,
        })
        ranked = rank(rows, np.array([0.2, 0.1, 0.2]))
        self.assertEqual(ranked["sku"].tolist(), ["A", "B", "C"])
        mean, weekly = metrics(ranked)
        self.assertEqual(weekly["flagged"].item(), 3)
        self.assertEqual(mean["precision_at_20"], 1 / 3)
        self.assertEqual(mean["recall_at_20"], 1)


if __name__ == "__main__":
    unittest.main()
