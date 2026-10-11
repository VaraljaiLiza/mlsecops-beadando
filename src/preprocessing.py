"""Az eredeti autóadatok tisztítása és reprodukálható mintavételezése."""

import yaml
import csv
import random
from pathlib import Path


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    sample_size: int = 3000,
    seed: int = 42,
    price_threshold: float = 600000,
) -> int:
    """Tisztított mintát ment, és visszaadja a mentett sorok számát."""
    if sample_size <= 0:
        raise ValueError("A mintaméretnek pozitívnak kell lennie.")

    if price_threshold <= 0:
        raise ValueError("Az árhatárnak pozitívnak kell lennie.")

    with input_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if columns is None or not {"", "seats", "selling_price"}.issubset(columns):
            raise ValueError("Hiányzik a sorszám, a seats vagy a selling_price oszlop.")

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
    output_columns.append("price_category")

    for record in sample:
        price = float(record["selling_price"])
        record["price_category"] = str(int(price >= price_threshold))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(sample)

    return len(sample)


if __name__ == "__main__":
    with Path("params.yaml").open(encoding="utf-8") as file:
        parameters = yaml.safe_load(file)["preprocessing"]

    count = prepare_dataset(
        input_path=Path("data/raw/cardekho_dataset.csv"),
        output_path=Path("data/processed/cardekho_sample.csv"),
        sample_size=parameters["sample_size"],
        seed=parameters["random_seed"],
        price_threshold=parameters["price_threshold"],
    )
    print(f"Elmentett adatsorok száma: {count}")
