"""Az API végpontjainak és naplózásának tesztjei."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from src.api import app


class TestAPI(unittest.TestCase):
    """Az API-t valódi MLflow-szerver nélkül ellenőrzi."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "params.yaml"

        config = {
            "training": {
                "numeric_features": [
                    "vehicle_age", "km_driven", "mileage",
                    "engine", "max_power", "seats",
                ],
                "categorical_features": [
                    "brand", "model", "seller_type",
                    "fuel_type", "transmission_type",
                ],
            },
            "mlflow": {
                "tracking_uri": "http://127.0.0.1:5000",
                "registered_model_name": "test-model",
            },
            "promotion": {"model_alias": "champion"},
        }
        self.config_path.write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )

        config_patch = patch("src.api.CONFIG_PATH", self.config_path)
        config_patch.start()
        self.addCleanup(config_patch.stop)

        log_patch = patch("src.api.LOG_DIR", self.root / "logs")
        log_patch.start()
        self.addCleanup(log_patch.stop)

        mlflow_patch = patch("src.api.mlflow")
        self.tracking = mlflow_patch.start()
        self.addCleanup(mlflow_patch.stop)
        run = self.tracking.start_run.return_value.__enter__.return_value
        run.info.run_id = "event-run"

        registry_patch = patch("src.api.MlflowClient")
        self.registry = registry_patch.start().return_value
        self.addCleanup(registry_patch.stop)
        self.registry.get_model_version_by_alias.return_value.version = "6"

        model_patch = patch("src.api.mlflow_sklearn.load_model")
        self.load_model = model_patch.start()
        self.addCleanup(model_patch.stop)
        self.load_model.return_value.predict.return_value = np.array([1])

        self.car = {
            "vehicle_age": 6,
            "km_driven": 70000,
            "mileage": 11.1,
            "engine": 2354,
            "max_power": 158.8,
            "seats": 5,
            "brand": "Honda",
            "model": "CR-V",
            "seller_type": "Dealer",
            "fuel_type": "Petrol",
            "transmission_type": "Automatic",
        }
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def read_events(self) -> list[dict]:
        path = self.root / "logs" / "events.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_prediction_and_artifact_logging(self) -> None:
        response = self.client.post("/predict", json={"cars": [self.car]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "model_version": "6",
            "predictions": [1],
            "run_id": "event-run",
        })
        self.registry.get_model_version_by_alias.assert_called_once_with(
            "test-model", "champion"
        )
        self.load_model.assert_called_once_with("models:/test-model/6")

        inputs = self.load_model.return_value.predict.call_args.args[0]
        self.assertNotIn("selling_price", inputs.columns)
        self.assertEqual(float(inputs.iloc[0]["vehicle_age"]), 6.0)

        events = self.read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["endpoint"], "predict")
        self.assertEqual(events[0]["request"]["cars"], [self.car])
        self.assertEqual(events[0]["response"]["predictions"], [1])
        self.tracking.log_artifact.assert_called_once()
        self.assertEqual(
            self.tracking.log_artifact.call_args.kwargs["artifact_path"],
            "api_events",
        )

    def test_training_and_logging(self) -> None:
        with patch("src.api.run_training") as training:
            training.return_value = (
                "training-run", {"balanced_accuracy": 0.9}
            )
            response = self.client.post(
                "/train", json={"reason": "API test"}
            )
            training.assert_called_once_with(
                self.config_path,
                Path("data/processed/cardekho_sample.csv"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "training-run")
        self.assertEqual(self.read_events()[0]["endpoint"], "train")
        self.tracking.start_run.assert_called_once_with(
            run_id="training-run", run_name="api-train"
        )

    def test_invalid_prediction_requests(self) -> None:
        invalid_cars = [
            {**self.car, "vehicle_age": -1},
            {**self.car, "seats": 0},
            {**self.car, "selling_price": 900000},
            {**self.car, "fuel_type": "Unknown"},
        ]
        for car in invalid_cars:
            with self.subTest(car=car):
                response = self.client.post(
                    "/predict", json={"cars": [car]}
                )
                self.assertEqual(response.status_code, 422)

        self.assertEqual(
            self.client.post("/predict", json={"cars": []}).status_code,
            422,
        )
        self.load_model.assert_not_called()

    def test_prediction_service_failure(self) -> None:
        self.load_model.side_effect = MlflowException("Model unavailable")
        response = self.client.post("/predict", json={"cars": [self.car]})
        self.assertEqual(response.status_code, 503)

    def test_training_service_failure(self) -> None:
        with patch("src.api.run_training", side_effect=ValueError("Bad data")):
            response = self.client.post("/train", json={})
        self.assertEqual(response.status_code, 503)

    def test_prediction_logging_failure(self) -> None:
        self.tracking.log_artifact.side_effect = MlflowException("Save failed")
        response = self.client.post("/predict", json={"cars": [self.car]})
        self.assertEqual(response.status_code, 503)

    def test_swagger_schema_contains_endpoints(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertIn("/train", paths)
        self.assertIn("/predict", paths)
