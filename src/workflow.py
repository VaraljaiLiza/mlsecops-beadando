"""Konfigurációból indított tanítás és MLflow-mentés."""

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
import os


from src.tracking import log_training_run
from src.training import train_model


def run_training(
    config_path: Path,
    dataset_path: Path,
) -> tuple[str, dict[str, float]]:
    """Tanít, ment, és ellenőrzi az aktív modell lecserélhetőségét."""
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    settings = config["training"]
    model_settings = settings["model"]
    tracking_settings = config["mlflow"]
    tracking_settings["tracking_uri"] = os.environ.get(
        "MLFLOW_TRACKING_URI",
        tracking_settings["tracking_uri"],
    )
    promotion = config["promotion"]

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
        model_seed=model_settings["random_seed"],
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

    evaluation_settings = {
        "dataset": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "features": features,
        "test_size": settings["test_size"],
        "split_seed": settings["random_seed"],
        "target": "price_category",
        "split_method": "group-shuffle-v1",
    }
    evaluation_key = hashlib.sha256(
        json.dumps(evaluation_settings, sort_keys=True).encode("utf-8")
    ).hexdigest()

    parameters = {
        "n_estimators": model_settings["n_estimators"],
        "max_depth": model_settings["max_depth"],
        "split_seed": settings["random_seed"],
        "model_seed": model_settings["random_seed"],
        "test_size": settings["test_size"],
        "sample_size": len(records),
        "price_threshold": config["preprocessing"]["price_threshold"],
        "evaluation_id": evaluation_key,
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
        model_alias=promotion["model_alias"],
        promotion_metric=promotion["metric"],
        minimum_improvement=promotion["minimum_improvement"],
    )
    return run_id, metrics
