"""Huấn luyện mô hình phục vụ web trên toàn bộ dữ liệu và lưu vào artifacts/service.pkl.

Chạy: python scripts/build_service.py   (web cũng tự làm việc này ở lần chạy đầu)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, RecommenderService  # noqa: E402


def main() -> None:
    best = ROOT / "results" / "best_params.json"
    params = json.loads(best.read_text()) if best.exists() else None
    t0 = time.perf_counter()
    try:
        import lightgbm  # noqa: F401
        with_ranker = True
    except ImportError:
        with_ranker = False
        print("Chua cai lightgbm -> web chi dung cong thuc tron. Cai bang: pip install lightgbm")
    svc = RecommenderService.build(build_dataset(), params, with_ranker=with_ranker)
    print("Bo xep hang hai tang:", "co" if svc.ranker else "khong")
    svc.save(DEFAULT_ARTIFACT)
    size = DEFAULT_ARTIFACT.stat().st_size / 1e6
    print(f"Da luu {DEFAULT_ARTIFACT} ({size:.1f} MB) sau {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
