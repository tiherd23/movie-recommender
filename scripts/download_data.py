"""Tải MovieLens 1M từ trang chính thức của GroupLens và giải nén vào data/raw/ml-1m.

Chạy:  python scripts/download_data.py
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "ml-1m"
    if (target / "ratings.dat").exists():
        print(f"Da co du lieu tai {target}, bo qua.")
        return
    zip_path = RAW_DIR / "ml-1m.zip"
    print(f"Dang tai {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"Xong. Du lieu nam o {target}")


if __name__ == "__main__":
    main()
