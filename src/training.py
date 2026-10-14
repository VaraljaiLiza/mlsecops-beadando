"""Modelltanítás és értékelés külön tesztadatokon."""

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

from src.data_split import split_records
from src.model import build_pipeline


def train_model(
    records: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    model_seed: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Betanított pipeline-t és tesztadatokon mért eredményeket ad vissza."""
    features = numeric_features + categorical_features
    train, test = split_records(
        records, features, test_size=test_size, seed=seed
    )

    train_data = pd.DataFrame(train)
    test_data = pd.DataFrame(test)

    for feature in numeric_features:
        train_data[feature] = pd.to_numeric(
            train_data[feature], errors="raise"
        )
        test_data[feature] = pd.to_numeric(
            test_data[feature], errors="raise"
        )

    train_target = train_data["price_category"].astype(int)
    test_target = test_data["price_category"].astype(int)

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=model_seed,
    )
    pipeline.fit(train_data[features], train_target)
    predictions = pipeline.predict(test_data[features])

    metrics = {
        "accuracy": float(accuracy_score(test_target, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_target, predictions)
        ),
    }
    return pipeline, metrics
