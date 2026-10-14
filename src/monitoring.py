"""Becslési naplók feldolgozása és Evidently drift riport."""

import json
from pathlib import Path

import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


def load_predictions(
    events_path: Path,
    features: list[str],
) -> pd.DataFrame:
    """A predict eseményeket autóként egy-egy táblázatsorrá alakítja."""
    columns = pd.Index(features + ["prediction", "model_version"])
    rows: list[dict] = []

    if not events_path.exists():
        return pd.DataFrame(columns=columns)

    with events_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            event = json.loads(line)
            if event["endpoint"] != "predict":
                continue

            cars = event["request"]["cars"]
            response = event["response"]
            predictions = response["predictions"]

            if len(cars) != len(predictions):
                raise ValueError("Eltér az autók és a becslések száma.")

            for car, prediction in zip(cars, predictions):
                row = {feature: car[feature] for feature in features}
                row["prediction"] = int(prediction)
                row["model_version"] = str(response["model_version"])
                rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_drift_report(
    config_path: Path,
    reference_path: Path,
    events_path: Path,
    output_path: Path,
    minimum_rows: int = 20,
) -> tuple[Path, int, int]:
    """HTML-riportot készít a referencia és a beküldött autók adataiból."""
    with config_path.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)["training"]

    numeric = settings["numeric_features"]
    categorical = settings["categorical_features"]
    features = numeric + categorical

    current = load_predictions(events_path, features)
    if len(current) < minimum_rows:
        raise ValueError(
            f"Még kevés a naplózott autó: {len(current)}. "
            f"A riporthoz legalább {minimum_rows} szükséges."
        )

    reference = pd.read_csv(reference_path)
    if not set(features).issubset(reference.columns):
        raise ValueError("Hiányzik egy bemeneti oszlop a referenciaadatból.")

    reference = pd.DataFrame({
        feature: reference[feature] for feature in features
    })
    current = pd.DataFrame({
        feature: current[feature] for feature in features
    })

    numeric_types = {feature: "float64" for feature in numeric}
    reference = reference.astype(numeric_types)
    current = current.astype(numeric_types)

    definition = DataDefinition(
        numerical_columns=numeric,
        categorical_columns=categorical,
    )
    reference_dataset = Dataset.from_pandas(
        reference, data_definition=definition
    )
    current_dataset = Dataset.from_pandas(
        current, data_definition=definition
    )

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    return output_path, len(reference), len(current)
