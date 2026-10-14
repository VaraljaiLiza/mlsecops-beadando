"""A teljes tanítási folyamat tesztje."""

import unittest

from src.training import train_model


class TestTraining(unittest.TestCase):
    """Az adatfelosztástól az értékelésig ellenőrzi a folyamatot."""

    def test_training_returns_model_and_metrics(self) -> None:
        records = [
            {
                "id": str(index),
                "vehicle_age": str(index),
                "brand": "Honda" if index < 20 else "Maruti",
                "selling_price": "900000" if index < 20 else "200000",
                "price_category": "1" if index < 20 else "0",
            }
            for index in range(40)
        ]

        pipeline, metrics = train_model(
            records,
            numeric_features=["vehicle_age"],
            categorical_features=["brand"],
            n_estimators=10,
            max_depth=3,
            model_seed=123,
        )

        self.assertEqual(
            set(metrics), {"accuracy", "balanced_accuracy"}
        )
        for value in metrics.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

        self.assertEqual(
            list(pipeline.named_steps["preprocessing"].feature_names_in_),
            ["vehicle_age", "brand"],
        )
        self.assertTrue(
            hasattr(pipeline.named_steps["classifier"], "estimators_")
        )
        self.assertEqual(
            pipeline.named_steps["classifier"].random_state, 123
        )
