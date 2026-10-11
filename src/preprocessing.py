"""Az eredeti autóadatok tisztítása és reprodukálható mintavételezése."""

import csv
import random
from pathlib import Path


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    sample_size: int = 3000,
    seed: int = 42,
) -> int:
    """Tisztított mintát ment, és visszaadja a mentett sorok számát."""
    if sample_size <= 0:
        raise ValueError("A mintaméretnek pozitívnak kell lennie.")

    with input_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if columns is None or "" not in columns or "seats" not in columns:
            raise ValueError("Hiányzik a sorszám vagy a seats oszlop.")

        output_columns = ["id" if name == "" else name for name in columns]
        records: list[dict[str, str]] = []

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Hibás szerkezetű CSV-sor.")

            if float(row["seats"]) == 0:
                continue

            record = {
                name: row[source]
                for source, name in zip(columns, output_columns)
            }
            records.append(record)

    if sample_size > len(records):
        raise ValueError("Nincs elegendő sor a kért mintához.")

    sample = random.Random(seed).sample(records, sample_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(sample)

    return len(sample)


if __name__ == "__main__":
    count = prepare_dataset(
        input_path=Path("data/raw/cardekho_dataset.csv"),
        output_path=Path("data/processed/cardekho_sample.csv"),
    )
    print(f"Elmentett adatsorok száma: {count}")