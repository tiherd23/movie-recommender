"""Đánh giá cuối trên tập test với siêu tham số tốt nhất (results/best_params.json).

Mỗi mô hình được huấn luyện lại trên train + val rồi mới chấm trên test.
Chạy: python scripts/run_final.py   (sau khi đã chạy scripts/tune.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import (SVD, BiasBaseline, ContentBased, Hybrid, ImplicitALS,  # noqa: E402
                             ItemKNN, Popularity)


def main() -> None:
    ds = build_dataset()
    best = json.loads((ROOT / "results" / "best_params.json").read_text())
    full = pd.concat([ds.train, ds.val], ignore_index=True)

    als, knn = ImplicitALS(**best["ImplicitALS"]), ItemKNN(**best["ItemKNN"])
    content, pop = ContentBased(), Popularity()
    models = [pop, BiasBaseline(), content, knn, SVD(**best["SVD"]), als]

    rows = []
    for model in models:
        t0 = time.perf_counter()
        model.fit(ds, full)
        rows.append({"Model": model.name, **evaluate(model, ds, "test"),
                     "fit_s": round(time.perf_counter() - t0, 1)})
        print(f"xong {model.name}", flush=True)

    hybrid = Hybrid([als, knn, content, pop], best["Hybrid"]["weights"], prefit=True)
    rows.append({"Model": "Hybrid", **evaluate(hybrid, ds, "test"), "fit_s": 0.0})

    table = pd.DataFrame(rows).set_index("Model").drop(columns="n_eval_users").round(4)
    print("\n" + table.to_string())
    table.to_csv(ROOT / "results" / "final.csv")
    print("FINAL_DONE")


if __name__ == "__main__":
    main()
