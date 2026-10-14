"""Adatfelosztás az azonos bemeneti rekordok együtt tartásával."""

import json
import numpy as np

from sklearn.model_selection import GroupShuffleSplit


def split_records(
    records: list[dict[str, str]],
    features: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Külön tanító- és tesztadatokat ad vissza."""
    if not records or not features:
        raise ValueError("Az adatok és a bemeneti oszlopok nem lehetnek üresek.")

    if not 0 < test_size < 1:
        raise ValueError("A tesztarány legyen 0 és 1 között.")

    if any(feature not in row for row in records for feature in features):
        raise ValueError("Hiányzik egy szükséges bemeneti oszlop.")

    groups = [
        json.dumps([row[feature] for feature in features])
        for row in records
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    row_positions = np.arange(len(records)).reshape(-1, 1)
    train_indices, test_indices = next(
        splitter.split(row_positions, groups=groups)
    )

    train = [records[int(index)] for index in train_indices]
    test = [records[int(index)] for index in test_indices]
    return train, test
