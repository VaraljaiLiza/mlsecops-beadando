"""Tanítási eredmények és modellek mentése MLflow-val."""

from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline


def log_training_run(
    pipeline: Pipeline,
    metrics: dict[str, float],
    parameters: dict[str, str | int | float],
    input_example: pd.DataFrame,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    config_path: Path,
    dataset_path: Path,
) -> str:
    """Elment egy tanítási eredményt, és visszaadja a run azonosítóját."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    signature = infer_signature(
        input_example,
        pipeline.predict(input_example),
    )

    with mlflow.start_run(run_name="random-forest-training") as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config_path), artifact_path="configuration")
        mlflow.log_artifact(str(dataset_path), artifact_path="data")

        mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        return str(run.info.run_id)
