"""A konfigurációból vezérelt tanítás parancssori indítása."""

from pathlib import Path

from src.workflow import run_training


def main() -> None:
    """Elindítja a tanítást, és kiírja az eredményeket."""
    run_id, metrics = run_training(
        config_path=Path("params.yaml"),
        dataset_path=Path("data/processed/cardekho_sample.csv"),
    )
    print("MLflow run azonosító:", run_id)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
