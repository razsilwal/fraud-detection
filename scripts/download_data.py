"""Download the public ULB credit-card fraud dataset into data/raw/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import DATASET_URL, RAW_DATASET_PATH, ensure_project_dirs  # noqa: E402


def download_dataset(force: bool = False) -> Path:
    ensure_project_dirs()
    if RAW_DATASET_PATH.exists() and not force:
        print(f"Dataset already present: {RAW_DATASET_PATH}")
        return RAW_DATASET_PATH
    print(f"Downloading creditcard.csv from:\n  {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, RAW_DATASET_PATH)
    size_mb = RAW_DATASET_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {RAW_DATASET_PATH} ({size_mb:.1f} MB)")
    return RAW_DATASET_PATH


if __name__ == "__main__":
    download_dataset(force="--force" in sys.argv)
