"""REST API tanításhoz, becsléshez és naplózáshoz."""

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast
from uuid import uuid4

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline
import os

from src.workflow import run_training

CONFIG_PATH = Path("params.yaml")
DATASET_PATH = Path("data/processed/cardekho_sample.csv")
LOG_DIR = Path("logs")
LOG_LOCK = Lock()
TRAIN_LOCK = Lock()

app = FastAPI(
    title="CarDekho Price Classification API",
    version="1.0.0",
)


class Car(BaseModel):
    """Egy autó modellbemenete."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    vehicle_age: int = Field(ge=0)
    km_driven: float = Field(ge=0)
    mileage: float = Field(gt=0)
    engine: float = Field(gt=0)
    max_power: float = Field(gt=0)
    seats: int = Field(ge=1)
    brand: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"]
    fuel_type: Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    transmission_type: Literal["Manual", "Automatic"]


class PredictRequest(BaseModel):
    """Legfeljebb 500 autó adatait fogadja."""

    model_config = ConfigDict(extra="forbid")
    cars: list[Car] = Field(min_length=1, max_length=500)


class TrainRequest(BaseModel):
    """A tanítás indításának megjegyzése."""

    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="manual", min_length=1, max_length=200)


def read_config() -> dict[str, Any]:
    """Beolvassa a beállításokat, figyelembe véve a konténer címét."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = cast(dict[str, Any], yaml.safe_load(file))

    config["mlflow"]["tracking_uri"] = os.environ.get(
        "MLFLOW_TRACKING_URI",
        config["mlflow"]["tracking_uri"],
    )
    return config


def save_event(
    endpoint: str,
    request_data: dict[str, Any],
    response_data: dict[str, Any],
    tracking_uri: str,
    run_id: str | None = None,
) -> str:
    """Helyi naplót és MLflow-artifactot készít."""
    event_id = uuid4().hex
    event = {
        "event_id": event_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
    }
    serialized = json.dumps(event, ensure_ascii=False, allow_nan=False)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event_path = LOG_DIR / f"{event_id}.json"

    with LOG_LOCK:
        event_path.write_text(serialized + "\n", encoding="utf-8")
        with (LOG_DIR / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(serialized + "\n")

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id, run_name=f"api-{endpoint}") as run:
        mlflow.set_tag("endpoint", endpoint)
        mlflow.log_artifact(str(event_path), artifact_path="api_events")
        return str(run.info.run_id)


@app.get("/health")
def health() -> dict[str, str]:
    """Jelzi, hogy az API fut."""
    return {"status": "ok"}


@app.post("/train")
def train(request: TrainRequest) -> dict[str, Any]:
    """Elindítja és naplózza a tanítást."""
    try:
        config = read_config()
        with TRAIN_LOCK:
            run_id, metrics = run_training(CONFIG_PATH, DATASET_PATH)
            response = {"run_id": run_id, "metrics": metrics}
            save_event(
                endpoint="train",
                request_data=request.model_dump(),
                response_data=response,
                tracking_uri=config["mlflow"]["tracking_uri"],
                run_id=run_id,
            )
        return response
    except (MlflowException, OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="A tanítás vagy az eredmények mentése nem sikerült.",
        ) from error


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    """Az aktív modellverzióval készít és naplóz becslést."""
    try:
        config = read_config()
        tracking = config["mlflow"]
        settings = config["training"]
        model_name = tracking["registered_model_name"]
        alias = config["promotion"]["model_alias"]

        client = MlflowClient(tracking_uri=tracking["tracking_uri"])
        version = client.get_model_version_by_alias(model_name, alias)

        mlflow.set_tracking_uri(tracking["tracking_uri"])
        model_uri = f"models:/{model_name}/{version.version}"
        pipeline = cast(Pipeline, mlflow_sklearn.load_model(model_uri))

        cars = [car.model_dump() for car in request.cars]
        features = (
            settings["numeric_features"] + settings["categorical_features"]
        )
        inputs = pd.DataFrame({
            feature: [car[feature] for car in cars]
            for feature in features
        })
        for feature in settings["numeric_features"]:
            inputs[feature] = [float(car[feature]) for car in cars]

        predictions = [
            int(value) for value in pipeline.predict(inputs)
        ]
        response = {
            "model_version": str(version.version),
            "predictions": predictions,
        }
        prediction_run_id = save_event(
            endpoint="predict",
            request_data=request.model_dump(),
            response_data=response,
            tracking_uri=tracking["tracking_uri"],
        )
        response["run_id"] = prediction_run_id
        return response
    except (MlflowException, OSError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="A modell betöltése, a becslés vagy a mentés nem sikerült.",
        ) from error
