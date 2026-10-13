"""A modell-pipeline működésének tesztjei."""

import unittest

import numpy as np
import pandas as pd

from src.model import build_pipeline


class TestModelPipeline(unittest.TestCase):
    """A taníthatóságot és az új bemenetek kezelését ellenőrzi."""

    def test_training_and_prediction(self) -> None:
        data = pd.DataFrame({
            "vehicle_age": [1, 2, 10, 12, 3, 15],
            "brand": ["Honda", "Honda", "Maruti", "Maruti", "Honda", "Maruti"],
            "selling_price": [900000, 800000, 200000, 150000, 700000, 100000],
        })
        target = [1, 1, 0, 0, 1, 0]
        original = data.copy(deep=True)

        pipeline = build_pipeline(
            numeric_features=["vehicle_age"],
            categorical_features=["brand"],
            n_estimators=10,
            max_depth=3,
            seed=42,
        )
        pipeline.fit(data, target)
        predictions = pipeline.predict(data)

        self.assertEqual(len(predictions), len(data))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertEqual(
            list(pipeline.named_steps["preprocessing"].feature_names_in_),
            ["vehicle_age", "brand", "selling_price"],
        )
        transformed_names = (
            pipeline.named_steps["preprocessing"].get_feature_names_out()
        )
        self.assertFalse(
            any("selling_price" in name for name in transformed_names)
        )
        self.assertEqual(
            pipeline.named_steps["classifier"].n_estimators, 10
        )
        pd.testing.assert_frame_equal(data, original)

    def test_unknown_category_and_missing_numeric_value(self) -> None:
        train = pd.DataFrame({
            "vehicle_age": [1, 3, 10, 15],
            "brand": ["Honda", "Honda", "Maruti", "Maruti"],
        })
        incoming = pd.DataFrame({
            "vehicle_age": [np.nan],
            "brand": ["Previously unseen brand"],
        })

        pipeline = build_pipeline(
            ["vehicle_age"], ["brand"], n_estimators=10
        )
        pipeline.fit(train, [1, 1, 0, 0])
        prediction = pipeline.predict(incoming)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])
