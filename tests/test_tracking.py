"""Az MLflow-mentési hívások tesztje."""

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.model import build_pipeline
from src.tracking import log_training_run


class TestTracking(unittest.TestCase):
    """Ellenőrzi, hogy minden szükséges eredményt elküldünk."""

    def test_logs_model_metrics_parameters_and_artifacts(self) -> None:
        data = pd.DataFrame({
            "vehicle_age": [1, 2, 10, 15],
            "brand": ["Honda", "Honda", "Maruti", "Maruti"],
        })
        pipeline = build_pipeline(
            ["vehicle_age"], ["brand"], n_estimators=5
        )
        pipeline.fit(data, [1, 1, 0, 0])
        metrics = {"accuracy": 0.9, "balanced_accuracy": 0.9}
        parameters = {"n_estimators": 5, "random_seed": 42}

        with (
            patch("src.tracking.mlflow") as tracking,
            patch("src.tracking.mlflow_sklearn") as model_logging,
        ):
            run = tracking.start_run.return_value.__enter__.return_value
            run.info.run_id = "test-run-id"

            run_id = log_training_run(
                pipeline=pipeline,
                metrics=metrics,
                parameters=parameters,
                input_example=data,
                tracking_uri="http://127.0.0.1:5000",
                experiment_name="test-experiment",
                registered_model_name="test-model",
                config_path=Path("params.yaml"),
                dataset_path=Path("sample.csv"),
            )

            self.assertEqual(run_id, "test-run-id")
            tracking.set_tracking_uri.assert_called_once_with(
                "http://127.0.0.1:5000"
            )
            tracking.set_experiment.assert_called_once_with(
                "test-experiment"
            )
            tracking.log_params.assert_called_once_with(parameters)
            tracking.log_metrics.assert_called_once_with(metrics)
            tracking.log_artifact.assert_any_call(
                "params.yaml", artifact_path="configuration"
            )
            tracking.log_artifact.assert_any_call(
                "sample.csv", artifact_path="data"
            )
            self.assertEqual(tracking.log_artifact.call_count, 2)
            model_logging.log_model.assert_called_once()

            arguments = model_logging.log_model.call_args.kwargs
            self.assertEqual(
                arguments["serialization_format"], "cloudpickle"
            )
            self.assertIs(arguments["sk_model"], pipeline)
            self.assertIsNotNone(arguments["signature"])
            pd.testing.assert_frame_equal(
                arguments["input_example"], data
            )
