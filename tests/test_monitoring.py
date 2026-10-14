"""A naplófeldolgozás és a drift riport tesztjei."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.monitoring import build_drift_report, load_predictions


class TestMonitoring(unittest.TestCase):
    """A becslési eseményeket és a riportkészítést ellenőrzi."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "params.yaml"
        self.reference_path = self.root / "reference.csv"
        self.events_path = self.root / "events.jsonl"
        self.output_path = self.root / "reports" / "drift.html"

        self.config_path.write_text(
            yaml.safe_dump({
                "training": {
                    "numeric_features": ["vehicle_age"],
                    "categorical_features": ["brand"],
                },
            }),
            encoding="utf-8",
        )
        pd.DataFrame({
            "vehicle_age": [index % 15 for index in range(60)],
            "brand": ["Honda", "Maruti"] * 30,
        }).to_csv(self.reference_path, index=False)

        cars = [
            {
                "vehicle_age": index % 15,
                "brand": "Honda" if index % 2 == 0 else "Maruti",
            }
            for index in range(30)
        ]
        self.event = {
            "endpoint": "predict",
            "request": {"cars": cars},
            "response": {
                "predictions": [index % 2 for index in range(30)],
                "model_version": "6",
            },
        }
        self.events_path.write_text(
            "\n"
            + json.dumps({"endpoint": "train"})
            + "\n"
            + json.dumps(self.event)
            + "\n",
            encoding="utf-8",
        )

    def test_reads_prediction_events(self) -> None:
        data = load_predictions(
            self.events_path, ["vehicle_age", "brand"]
        )
        self.assertEqual(len(data), 30)
        self.assertEqual(list(data.columns), [
            "vehicle_age", "brand", "prediction", "model_version",
        ])
        self.assertEqual(data.iloc[0]["prediction"], 0)
        self.assertEqual(data.iloc[0]["model_version"], "6")

    def test_missing_log_returns_empty_table(self) -> None:
        data = load_predictions(
            self.root / "missing.jsonl", ["vehicle_age", "brand"]
        )
        self.assertTrue(data.empty)

    def test_mismatched_prediction_count(self) -> None:
        self.event["response"]["predictions"] = []
        self.events_path.write_text(
            json.dumps(self.event), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            load_predictions(
                self.events_path, ["vehicle_age", "brand"]
            )

    def test_creates_real_evidently_report(self) -> None:
        path, reference_count, current_count = build_drift_report(
            self.config_path,
            self.reference_path,
            self.events_path,
            self.output_path,
        )
        self.assertEqual(reference_count, 60)
        self.assertEqual(current_count, 30)
        self.assertTrue(path.is_file())
        self.assertIn("<html", path.read_text(encoding="utf-8").lower())

    def test_insufficient_current_data(self) -> None:
        with self.assertRaises(ValueError):
            build_drift_report(
                self.config_path,
                self.reference_path,
                self.events_path,
                self.output_path,
                minimum_rows=100,
            )

    def test_missing_reference_column(self) -> None:
        self.reference_path.write_text("brand\nHonda\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            build_drift_report(
                self.config_path,
                self.reference_path,
                self.events_path,
                self.output_path,
            )
