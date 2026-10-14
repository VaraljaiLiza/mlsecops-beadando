"""Tanítási eredmények mentése és az aktív modell kiválasztása."""

from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.promotion import should_promote


def promote_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    score: float,
    evaluation_key: str,
    metric: str,
    alias: str,
    minimum_improvement: float,
) -> str:
    """Kiválasztja az új verziót, ha az összehasonlítás ezt indokolja."""
    should_promote(score, None, minimum_improvement)
    registered = client.get_registered_model(model_name)
    current_score = None

    if alias in registered.aliases:
        current = client.get_model_version(
            model_name, registered.aliases[alias]
        )
        if current.tags.get("evaluation_id") != evaluation_key:
            return "incompatible_evaluation"

        current_score = float(current.tags[metric])

    if should_promote(score, current_score, minimum_improvement):
        client.set_registered_model_alias(model_name, alias, version)
        return "promoted"

    return "kept_current_model"


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
    model_alias: str | None = None,
    promotion_metric: str = "balanced_accuracy",
    minimum_improvement: float = 0.0,
) -> str:
    """Elmenti a tanítást, és igény esetén kiválasztja az aktív modellt."""
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

        model_info = mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        if model_alias is not None:
            version = model_info.registered_model_version
            if version is None:
                raise ValueError("Nem jött létre regisztrált modellverzió.")

            client = MlflowClient(tracking_uri=tracking_uri)
            evaluation_key = str(parameters["evaluation_id"])
            score = metrics[promotion_metric]

            client.set_model_version_tag(
                registered_model_name, str(version),
                "evaluation_id", evaluation_key,
            )
            client.set_model_version_tag(
                registered_model_name, str(version),
                promotion_metric, str(score),
            )

            decision = promote_version(
                client=client,
                model_name=registered_model_name,
                version=str(version),
                score=score,
                evaluation_key=evaluation_key,
                metric=promotion_metric,
                alias=model_alias,
                minimum_improvement=minimum_improvement,
            )
            mlflow.set_tag("promotion_decision", decision)

        return str(run.info.run_id)
