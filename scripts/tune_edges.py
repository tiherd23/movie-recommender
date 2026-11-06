"""Dò thêm siêu tham số ở những chỗ lưới cũ (scripts/tune.py) có giá trị tốt nhất nằm ngay ở biên.

Chạy: python scripts/tune_edges.py   (khoảng 10-15 phút, CPU)
- Implicit ALS: alpha tốt nhất (1,0) là giá trị nhỏ nhất của lưới -> thử 0,1 / 0,25 / 0,5; reg chưa từng được dò
  (cố định 0,1) -> thử 0,01 / 1 / 10 quanh cấu hình tốt nhất.
- Hybrid: trọng số ItemKNN (1,0) và Popularity (0,25) tốt nhất đều là giá trị lớn nhất của lưới -> thử lớn hơn.
Vẫn chọn trên tập val, huấn luyện trên train, giống hệt scripts/tune.py. Kết quả ghi riêng vào
results/tuning_edges.csv (không sửa results/tuning.csv).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import ContentBased, Hybrid, ImplicitALS, ItemKNN, Popularity  # noqa: E402

OUT = ROOT / "results" / "tuning_edges.csv"


def main() -> None:
    ds = build_dataset()
    best = json.loads((ROOT / "results" / "best_params.json").read_text())
    rows = []

    def log(model: str, params: dict, res: dict, secs: float) -> None:
        rows.append({"model": model, "params": json.dumps(params), "NDCG@10": res["NDCG@10"],
                     "Recall@10": res["Recall@10"], "Coverage": res["Coverage"], "seconds": round(secs, 1)})
        print(f"{model:12s} {json.dumps(params):60s} NDCG={res['NDCG@10']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    # ---- Implicit ALS ----
    trials = [{"factors": f, "alpha": a, "reg": 0.1, "min_rating": 4.0}
              for f, a in itertools.product([16, 32, 64], [0.1, 0.25, 0.5, 1.0])]
    trials += [{"factors": 32, "alpha": a, "reg": r, "min_rating": 4.0}
               for a, r in itertools.product([0.5, 1.0], [0.01, 1.0, 10.0])]
    for params in trials:
        t0 = time.perf_counter()
        log("ImplicitALS", params, evaluate(ImplicitALS(**params).fit(ds), ds, "val"), time.perf_counter() - t0)
    als_rows = pd.DataFrame([r for r in rows if r["model"] == "ImplicitALS"])
    best_als = json.loads(als_rows.loc[als_rows["NDCG@10"].idxmax(), "params"])
    print("ALS tốt nhất sau khi dò thêm:", best_als, flush=True)

    # ---- Hybrid (dùng ALS tốt nhất mới) ----
    parts = [ImplicitALS(**best_als).fit(ds), ItemKNN(**best["ItemKNN"]).fit(ds),
             ContentBased().fit(ds), Popularity().fit(ds)]
    for w_knn, w_cb, w_pop in itertools.product([0.5, 1.0, 1.5, 2.0, 3.0], [0.0, 0.1, 0.25], [0.1, 0.25, 0.5, 1.0]):
        weights = [1.0, w_knn, w_cb, w_pop]
        t0 = time.perf_counter()
        log("Hybrid", {"weights": weights}, evaluate(Hybrid(parts, weights, prefit=True), ds, "val"),
            time.perf_counter() - t0)
    hy = pd.DataFrame([r for r in rows if r["model"] == "Hybrid"])
    print("Hybrid tốt nhất sau khi dò thêm:", hy.loc[hy["NDCG@10"].idxmax(), "params"])
    print("TUNE_EDGES_DONE")


if __name__ == "__main__":
    main()
