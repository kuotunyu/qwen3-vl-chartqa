"""Regression tests pinning the canonical ChartQA relaxed-accuracy semantics.

These tests exist to stop well-meaning "cleanups" from silently changing the
metric that produced every headline number in the README. In particular the
zero-target branch is a deliberate compatibility choice, not a bug: see
`src/relaxed_accuracy.py` and `docs/DESIGN_NOTES.md`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relaxed_accuracy import (  # noqa: E402
    normalize_prediction,
    relaxed_accuracy,
    relaxed_correctness,
)


class NumericToleranceTests(unittest.TestCase):
    def test_exact_numeric_match(self):
        self.assertTrue(relaxed_correctness("96", "96"))

    def test_within_five_percent_is_correct(self):
        # |104 - 100| / 100 = 0.04
        self.assertTrue(relaxed_correctness("104", "100"))

    def test_exactly_five_percent_is_correct(self):
        # boundary is inclusive (<=), matching the paper implementation
        self.assertTrue(relaxed_correctness("105", "100"))

    def test_beyond_five_percent_is_wrong(self):
        # |106 - 100| / 100 = 0.06
        self.assertFalse(relaxed_correctness("106", "100"))

    def test_tolerance_is_relative_not_absolute(self):
        self.assertTrue(relaxed_correctness("10400", "10000"))
        self.assertFalse(relaxed_correctness("1.06", "1.0"))

    def test_negative_targets_use_absolute_denominator(self):
        self.assertTrue(relaxed_correctness("-104", "-100"))
        self.assertFalse(relaxed_correctness("-106", "-100"))


class PercentageParsingTests(unittest.TestCase):
    def test_percent_suffix_is_stripped_on_both_sides(self):
        self.assertTrue(relaxed_correctness("51%", "50%"))
        self.assertFalse(relaxed_correctness("60%", "50%"))

    def test_percent_and_bare_decimal_are_comparable(self):
        # "50%" -> 0.5 and "0.5" -> 0.5
        self.assertTrue(relaxed_correctness("0.5", "50%"))
        self.assertTrue(relaxed_correctness("50%", "0.5"))

    def test_percent_sign_only_recognised_as_suffix(self):
        # not parseable as float -> falls back to string comparison
        self.assertFalse(relaxed_correctness("%50", "50"))


class ZeroTargetCanonicalTests(unittest.TestCase):
    """Canonical compatibility: a target of exactly 0 falls through to string match.

    `relaxed_correctness` tests the truthiness of `target_float`, not
    `is not None`. A zero target is therefore compared as a string. This
    mirrors the reference implementation (Masry et al. 2022 / Pix2Struct /
    lmms-eval) and additionally avoids a division-by-zero. Changing it would
    make our scores incomparable with published numbers.
    """

    def test_zero_target_exact_string_is_correct(self):
        self.assertTrue(relaxed_correctness("0", "0"))

    def test_zero_target_different_spelling_is_wrong(self):
        # numerically equal, but string comparison rejects it — canonical behaviour
        self.assertFalse(relaxed_correctness("0.0", "0"))
        self.assertFalse(relaxed_correctness("0%", "0"))

    def test_zero_target_near_miss_is_wrong(self):
        self.assertFalse(relaxed_correctness("0.01", "0"))

    def test_zero_target_never_raises_zero_division(self):
        for prediction in ("0", "0.0", "0.01", "-3", "not a number"):
            with self.subTest(prediction=prediction):
                relaxed_correctness(prediction, "0")

    def test_zero_prediction_against_nonzero_target_uses_tolerance(self):
        # only the *target* triggers the fall-through, not the prediction
        self.assertFalse(relaxed_correctness("0", "100"))


class NonNumericTests(unittest.TestCase):
    def test_case_insensitive_exact_match(self):
        self.assertTrue(relaxed_correctness("Yes", "yes"))
        self.assertTrue(relaxed_correctness("UNITED STATES", "united states"))

    def test_different_strings_are_wrong(self):
        self.assertFalse(relaxed_correctness("USA", "United States"))

    def test_numeric_prediction_against_text_target_is_string_compared(self):
        self.assertFalse(relaxed_correctness("5", "five"))

    def test_whitespace_is_not_normalised_by_the_metric(self):
        # cleanup belongs to normalize_prediction, deliberately not to the metric
        self.assertFalse(relaxed_correctness(" yes", "yes"))


class NormalizePredictionTests(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        self.assertEqual(normalize_prediction("  96  "), "96")

    def test_strips_single_trailing_period(self):
        self.assertEqual(normalize_prediction("96."), "96")

    def test_strips_trailing_period_then_whitespace(self):
        self.assertEqual(normalize_prediction(" 96. "), "96")

    def test_keeps_internal_periods(self):
        self.assertEqual(normalize_prediction("3.14"), "3.14")

    def test_removes_only_one_trailing_period(self):
        self.assertEqual(normalize_prediction("96.."), "96.")


class AggregateTests(unittest.TestCase):
    def test_accuracy_counts_normalised_predictions(self):
        predictions = ["96.", " yes ", "106"]
        targets = ["96", "yes", "100"]
        # first two become correct after normalisation, third is outside tolerance
        self.assertAlmostEqual(relaxed_accuracy(predictions, targets), 2 / 3)

    def test_empty_input_is_zero(self):
        self.assertEqual(relaxed_accuracy([], []), 0.0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            relaxed_accuracy(["96"], ["96", "97"])


if __name__ == "__main__":
    unittest.main()
