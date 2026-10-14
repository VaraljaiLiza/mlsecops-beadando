"""Konfigurációból indított tanítás és MLflow-mentés."""

import csv
from pathlib import Path

import pandas as pd
import yaml

from src.tracking import log_training_run
from src.training import train_model


def run_training(
    config_path: Path,
    dataset_path: Path,
) -> tuple[str, dict[str, float]]:
    """Tanít, MLflow-ba ment, majd visszaadja a run ID-t és a mérőszámokat."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    settings = config["training"]
    model_settings = settings["model"]
    tracking_settings = config["mlflow"]

    with dataset_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    pipeline, metrics = train_model(
        records=records,
        numeric_features=settings["numeric_features"],
        categorical_features=settings["categorical_features"],
        test_size=settings["test_size"],
        seed=settings["random_seed"],
        n_estimators=model_settings["n_estimators"],
        max_depth=model_settings["max_depth"],
    )

    features = (
        settings["numeric_features"] + settings["categorical_features"]
    )
    example = pd.DataFrame({
        feature: [row[feature] for row in records[:5]]
        for feature in features
    })

    for feature in settings["numeric_features"]:
        example[feature] = [
            float(row[feature]) for row in records[:5]
        ]

    parameters = {
        "n_estimators": model_settings["n_estimators"],
        "max_depth": model_settings["max_depth"],
        "split_seed": settings["random_seed"],
        "model_seed": settings["random_seed"],
        "test_size": settings["test_size"],
        "sample_size": len(records),
        "price_threshold": config["preprocessing"]["price_threshold"],
    }

    run_id = log_training_run(
        pipeline=pipeline,
        metrics=metrics,
        parameters=parameters,
        input_example=example,
        tracking_uri=tracking_settings["tracking_uri"],
        experiment_name=tracking_settings["experiment_name"],
        registered_model_name=tracking_settings["registered_model_name"],
        config_path=config_path,
        dataset_path=dataset_path,
    )
    return run_id, metrics
