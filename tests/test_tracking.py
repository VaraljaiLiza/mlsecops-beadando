"""Az MLflow-mentés és az aktív modell kiválasztásának tesztjei."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.model import build_pipeline
from src.tracking import log_training_run, promote_version


class TestTracking(unittest.TestCase):
    """A mentést szerverkapcsolat nélkül ellenőrzi."""

    def setUp(self) -> None:
        self.data = pd.DataFrame({
            "vehicle_age": [1.0, 2.0, 10.0, 15.0],
            "brand": ["Honda", "Honda", "Maruti", "Maruti"],
        })
        self.pipeline = build_pipeline(
            ["vehicle_age"], ["brand"], n_estimators=5
        )
        self.pipeline.fit(self.data, [1, 1, 0, 0])
        self.arguments = {
            "pipeline": self.pipeline,
            "metrics": {"accuracy": 0.9, "balanced_accuracy": 0.9},
            "parameters": {"evaluation_id": "same-data", "n_estimators": 5},
            "input_example": self.data,
            "tracking_uri": "http://127.0.0.1:5000",
            "experiment_name": "test-experiment",
            "registered_model_name": "test-model",
            "config_path": Path("params.yaml"),
            "dataset_path": Path("sample.csv"),
        }

    def test_logging_with_and_without_promotion(self) -> None:
        for alias in [None, "champion"]:
            with self.subTest(alias=alias):
                with (
                    patch("src.tracking.mlflow") as tracking,
                    patch("src.tracking.mlflow_sklearn") as model_logging,
                    patch("src.tracking.MlflowClient") as client_factory,
                    patch("src.tracking.promote_version") as promotion,
                ):
                    run = tracking.start_run.return_value.__enter__.return_value
                    run.info.run_id = "test-run-id"
                    model_logging.log_model.return_value.registered_model_version = "2"
                    promotion.return_value = "promoted"

                    result = log_training_run(
                        **self.arguments, model_alias=alias
                    )

                    self.assertEqual(result, "test-run-id")
                    tracking.log_params.assert_called_once_with(
                        self.arguments["parameters"]
                    )
                    tracking.log_metrics.assert_called_once_with(
                        self.arguments["metrics"]
                    )
                    tracking.log_artifact.assert_any_call(
                        "params.yaml", artifact_path="configuration"
                    )
                    tracking.log_artifact.assert_any_call(
                        "sample.csv", artifact_path="data"
                    )
                    model_logging.log_model.assert_called_once()
                    kwargs = model_logging.log_model.call_args.kwargs
                    self.assertEqual(
                        kwargs["serialization_format"], "cloudpickle"
                    )
                    self.assertIs(kwargs["sk_model"], self.pipeline)

                    if alias is None:
                        promotion.assert_not_called()
                    else:
                        promotion.assert_called_once()
                        self.assertEqual(
                            client_factory.return_value.set_model_version_tag.call_count,
                            2,
                        )
                        tracking.set_tag.assert_called_once_with(
                            "promotion_decision", "promoted"
                        )

    def test_missing_registered_version(self) -> None:
        with (
            patch("src.tracking.mlflow"),
            patch("src.tracking.mlflow_sklearn") as model_logging,
        ):
            model_logging.log_model.return_value.registered_model_version = None

            with self.assertRaises(ValueError):
                log_training_run(**self.arguments, model_alias="champion")


class TestPromotionRegistry(unittest.TestCase):
    """Az alias módosítását és a rosszabb modellek elutasítását ellenőrzi."""

    def test_promotion_decisions(self) -> None:
        cases = [
            (False, "same-data", 0.8, 0.9, "promoted"),
            (True, "same-data", 0.8, 0.9, "promoted"),
            (True, "same-data", 0.9, 0.9, "promoted"),
            (True, "same-data", 0.9, 0.8, "kept_current_model"),
            (True, "different-data", 0.8, 0.9, "incompatible_evaluation"),
        ]

        for has_current, key, old_score, new_score, expected in cases:
            with self.subTest(expected=expected, old_score=old_score):
                client = MagicMock()
                client.get_registered_model.return_value.aliases = (
                    {"champion": "1"} if has_current else {}
                )
                client.get_model_version.return_value.tags = {
                    "evaluation_id": key,
                    "balanced_accuracy": str(old_score),
                }

                decision = promote_version(
                    client=client,
                    model_name="test-model",
                    version="2",
                    score=new_score,
                    evaluation_key="same-data",
                    metric="balanced_accuracy",
                    alias="champion",
                    minimum_improvement=0.0,
                )

                self.assertEqual(decision, expected)
                if expected == "promoted":
                    client.set_registered_model_alias.assert_called_once_with(
                        "test-model", "champion", "2"
                    )
                else:
                    client.set_registered_model_alias.assert_not_called()
