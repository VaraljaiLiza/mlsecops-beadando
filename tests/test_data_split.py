"""A tanító–teszt felosztás tesztjei."""

import unittest

from src.data_split import split_records


class TestSplitRecords(unittest.TestCase):
    """A csoportok elkülönítését és a hibakezelést ellenőrzi."""

    def setUp(self) -> None:
        self.records = [
            {"id": "1", "brand": "Honda"},
            {"id": "2", "brand": "Honda"},
            {"id": "3", "brand": "Maruti"},
            {"id": "4", "brand": "Hyundai"},
            {"id": "5", "brand": "Nissan"},
        ]

    def test_groups_do_not_overlap(self) -> None:
        train, test = split_records(
            self.records, ["brand"], test_size=0.5
        )

        train_brands = {row["brand"] for row in train}
        test_brands = {row["brand"] for row in test}

        self.assertTrue(train)
        self.assertTrue(test)
        self.assertTrue(train_brands.isdisjoint(test_brands))
        self.assertEqual(
            sorted(row["id"] for row in train + test),
            ["1", "2", "3", "4", "5"],
        )

    def test_same_seed_produces_same_split(self) -> None:
        first = split_records(self.records, ["brand"], seed=42)
        second = split_records(self.records, ["brand"], seed=42)
        self.assertEqual(first, second)

    def test_empty_inputs(self) -> None:
        with self.assertRaises(ValueError):
            split_records([], ["brand"])

        with self.assertRaises(ValueError):
            split_records(self.records, [])

    def test_invalid_test_size(self) -> None:
        for size in [0, 1, -0.1, 1.1]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    split_records(self.records, ["brand"], test_size=size)

    def test_missing_feature(self) -> None:
        with self.assertRaises(ValueError):
            split_records(self.records, ["engine"])
