from pathlib import Path
from zipfile import ZipFile

import requests

DATA_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"

RAW_DATA_DIR = Path("data/raw")
ZIP_PATH = RAW_DATA_DIR / "vehicles.csv.zip"
CSV_PATH = RAW_DATA_DIR / "vehicles.csv"


def download_dataset():
    """Download and extract the official FuelEconomy.gov vehicles dataset."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from:\n{DATA_URL}")

    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()

    ZIP_PATH.write_bytes(response.content)
    print(f"Saved ZIP file to: {ZIP_PATH}")

    with ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(RAW_DATA_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected file {CSV_PATH} was not found after extraction."
        )

    print(f"Extracted dataset to: {CSV_PATH}")


if __name__ == "__main__":
    download_dataset()