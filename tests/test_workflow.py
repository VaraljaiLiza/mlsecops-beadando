"""A konfigurációból indított tanítási folyamat tesztje."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.workflow import run_training


class TestTrainingWorkflow(unittest.TestCase):
    """A beolvasást és a tanítási, mentési hívásokat ellenőrzi."""

    def test_runs_training_and_logging_from_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "params.yaml"
            dataset_path = root / "sample.csv"

            config = {
                "preprocessing": {"price_threshold": 600000},
                "training": {
                    "numeric_features": ["vehicle_age"],
                    "categorical_features": ["brand"],
                    "test_size": 0.2,
                    "random_seed": 42,
                    "model": {
                        "n_estimators": 10,
                        "max_depth": 3,
                        "random_seed": 42,
                    },
                },
                "mlflow": {
                    "tracking_uri": "http://127.0.0.1:5000",
                    "experiment_name": "test-experiment",
                    "registered_model_name": "test-model",
                },
            }
            config_path.write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )
            dataset_path.write_text(
                "id,vehicle_age,brand,selling_price,price_category\n"
                "1,2,Honda,900000,1\n"
                "2,12,Maruti,200000,0\n",
                encoding="utf-8",
            )
            metrics = {"accuracy": 0.9, "balanced_accuracy": 0.9}

            with (
                patch("src.workflow.train_model") as training,
                patch("src.workflow.log_training_run") as logging,
            ):
                pipeline = training.return_value[0]
                training.return_value = (pipeline, metrics)
                logging.return_value = "test-run-id"

                run_id, result = run_training(config_path, dataset_path)

                self.assertEqual(run_id, "test-run-id")
                self.assertEqual(result, metrics)
                training.assert_called_once()
                logging.assert_called_once()

                training_arguments = training.call_args.kwargs
                self.assertEqual(len(training_arguments["records"]), 2)
                self.assertEqual(
                    training_arguments["numeric_features"], ["vehicle_age"]
                )
                self.assertEqual(
                    training_arguments["categorical_features"], ["brand"]
                )
                self.assertEqual(training_arguments["test_size"], 0.2)
                self.assertEqual(training_arguments["seed"], 42)
                self.assertEqual(training_arguments["n_estimators"], 10)
                self.assertEqual(training_arguments["max_depth"], 3)

                logging_arguments = logging.call_args.kwargs
                self.assertIs(logging_arguments["pipeline"], pipeline)
                self.assertEqual(logging_arguments["metrics"], metrics)
                self.assertEqual(
                    logging_arguments["tracking_uri"],
                    "http://127.0.0.1:5000",
                )
                self.assertEqual(
                    logging_arguments["experiment_name"], "test-experiment"
                )
                self.assertEqual(
                    logging_arguments["registered_model_name"], "test-model"
                )
                self.assertEqual(
                    logging_arguments["config_path"], config_path
                )
                self.assertEqual(
                    logging_arguments["dataset_path"], dataset_path
                )
                self.assertEqual(
                    logging_arguments["parameters"]["price_threshold"],
                    600000,
                )
                self.assertEqual(
                    logging_arguments["parameters"]["sample_size"], 2
                )

                expected_example = pd.DataFrame({
                    "vehicle_age": [2.0, 12.0],
                    "brand": ["Honda", "Maruti"],
                })
                pd.testing.assert_frame_equal(
                    logging_arguments["input_example"],
                    expected_example,
                )
