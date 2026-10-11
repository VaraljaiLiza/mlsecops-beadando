"""Az előfeldolgozás automatikus tesztjei."""

import contextlib
import io
import os
import runpy
import csv
import tempfile
import unittest
from pathlib import Path

from src.preprocessing import prepare_dataset


class TestPrepareDataset(unittest.TestCase):
    """A mintavétel és az adatellenőrzés tesztjei."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input_path = self.root / "input.csv"
        self.output_path = self.root / "processed" / "sample.csv"
        self.input_path.write_text(
            ",car_name,seats\n"
            "10,Honda City,5\n"
            "11,Nissan Kicks,0\n"
            "12,Maruti Alto,5\n"
            "13,Hyundai i20,5\n",
            encoding="utf-8",
        )

    def test_cleaning_and_id_preservation(self) -> None:
        count = prepare_dataset(self.input_path, self.output_path, 3)

        with self.output_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            columns = reader.fieldnames

        self.assertEqual(count, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(columns, ["id", "car_name", "seats"])
        self.assertEqual({row["id"] for row in rows}, {"10", "12", "13"})
        self.assertTrue(all(row["seats"] == "5" for row in rows))

    def test_same_seed_produces_same_file(self) -> None:
        other_path = self.root / "other.csv"
        prepare_dataset(self.input_path, self.output_path, 2, seed=42)
        prepare_dataset(self.input_path, other_path, 2, seed=42)

        self.assertEqual(
            self.output_path.read_bytes(),
            other_path.read_bytes(),
        )

    def test_invalid_sample_size(self) -> None:
        for size in [0, -1, 4]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    prepare_dataset(self.input_path, self.output_path, size)

    def test_missing_required_column(self) -> None:
        self.input_path.write_text("car_name,seats\nHonda,5\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            prepare_dataset(self.input_path, self.output_path, 1)

    def test_malformed_row(self) -> None:
        self.input_path.write_text(",car_name,seats\n10,Honda\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            prepare_dataset(self.input_path, self.output_path, 1)

    def test_direct_execution(self) -> None:
        raw_path = self.root / "data" / "raw" / "cardekho_dataset.csv"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text(
            ",car_name,seats\n"
            + "".join(f"{index},Honda,5\n" for index in range(3000)),
            encoding="utf-8",
        )
        (self.root / "params.yaml").write_text(
            "preprocessing:\n"
            "  sample_size: 3000\n"
            "  random_seed: 42\n"
            "  price_threshold: 600000\n",
            encoding="utf-8",
        )
        script_path = Path("src/preprocessing.py").resolve()
        original_directory = Path.cwd()
        output = io.StringIO()

        try:
            os.chdir(self.root)
            with contextlib.redirect_stdout(output):
                runpy.run_path(str(script_path), run_name="__main__")
        finally:
            os.chdir(original_directory)

        self.assertIn("Elmentett adatsorok száma: 3000", output.getvalue())
        result_path = self.root / "data" / "processed" / "cardekho_sample.csv"
        self.assertTrue(result_path.is_file())