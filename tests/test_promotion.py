"""A modellváltási szabály tesztjei."""

import unittest

from src.promotion import should_promote


class TestPromotion(unittest.TestCase):
    """A jobb, azonos és rosszabb eredmények kezelését ellenőrzi."""

    def test_first_model_is_accepted(self) -> None:
        self.assertTrue(should_promote(0.8, None))

    def test_score_comparison(self) -> None:
        self.assertTrue(should_promote(0.9, 0.8))
        self.assertTrue(should_promote(0.8, 0.8))
        self.assertFalse(should_promote(0.7, 0.8))

    def test_required_improvement(self) -> None:
        self.assertTrue(should_promote(0.9, 0.8, 0.05))
        self.assertFalse(should_promote(0.82, 0.8, 0.05))

    def test_invalid_candidate_score(self) -> None:
        for score in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    should_promote(score, 0.8)

    def test_invalid_current_score(self) -> None:
        for score in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    should_promote(0.8, score)

    def test_invalid_required_improvement(self) -> None:
        for improvement in [-0.1, float("nan"), float("inf")]:
            with self.subTest(improvement=improvement):
                with self.assertRaises(ValueError):
                    should_promote(0.8, 0.7, improvement)
