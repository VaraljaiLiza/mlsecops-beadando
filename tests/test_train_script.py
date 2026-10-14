"""A parancssori tanításindító tesztje."""

import contextlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train import main


class TestTrainScript(unittest.TestCase):
    """Az indítás paramétereit és a kiírt eredményt ellenőrzi."""

    def test_main_prints_training_results(self) -> None:
        output = io.StringIO()

        with patch("scripts.train.run_training") as training:
            training.return_value = (
                "test-run-id",
                {"accuracy": 0.9, "balanced_accuracy": 0.8},
            )

            with contextlib.redirect_stdout(output):
                main()

            training.assert_called_once_with(
                config_path=Path("params.yaml"),
                dataset_path=Path("data/processed/cardekho_sample.csv"),
            )

        text = output.getvalue()
        self.assertIn("MLflow run azonosító: test-run-id", text)
        self.assertIn("accuracy: 0.9000", text)
        self.assertIn("balanced_accuracy: 0.8000", text)

    def test_execution_as_main(self) -> None:
        output = io.StringIO()

        with patch("src.workflow.run_training") as training:
            training.return_value = ("test-run-id", {"accuracy": 0.9})

            with contextlib.redirect_stdout(output):
                runpy.run_path("scripts/train.py", run_name="__main__")

            training.assert_called_once_with(
                config_path=Path("params.yaml"),
                dataset_path=Path("data/processed/cardekho_sample.csv"),
            )

        self.assertIn(
            "MLflow run azonosító: test-run-id",
            output.getvalue(),
        )
