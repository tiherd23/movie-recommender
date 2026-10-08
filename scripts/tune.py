"""Chỉnh siêu tham số trên tập val (huấn luyện trên train).

Chạy toàn bộ:      python scripts/tune.py
Chạy từng phần:    python scripts/tune.py --stage als --only factors=64
Kết quả: results/tuning.csv (mọi lần thử, cộng dồn) và results/best_params.json.
Tiêu chí chọn: SVD theo RMSE thấp nhất, các mô hình xếp hạng theo NDCG@10 cao nhất.
"""
from __future__ import annotations

import argparse
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
from movierec.models import SVD, ContentBased, Hybrid, ImplicitALS, ItemKNN, Popularity  # noqa: E402

GRIDS = {
    "knn": {"k": [20, 50, 100, 200], "shrink": [0.0, 20.0, 100.0]},
    "als": {"factors": [16, 32, 64, 128], "alpha": [1.0, 2.0, 5.0, 20.0], "min_rating": [1.0, 4.0]},
    "svd": {"factors": [32, 64], "reg": [0.03, 0.06, 0.1]},
}
NAMES = {"knn": "ItemKNN", "als": "ImplicitALS", "svd": "SVD", "hybrid": "Hybrid"}
CSV, BEST = ROOT / "results" / "tuning.csv", ROOT / "results" / "best_params.json"


def grid(space: dict, only: dict) -> list[dict]:
    keys = list(space)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]
    return [c for c in combos if all(str(c[k]) == v for k, v in only.items())]


def best_params() -> dict:
    """Đọc tuning.csv và lấy cấu hình tốt nhất của từng mô hình."""
    df = pd.read_csv(CSV)
    best = {}
    for model, part in df.groupby("model"):
        row = part.loc[part["RMSE"].idxmin()] if model == "SVD" else part.loc[part["NDCG@10"].idxmax()]
        best[model] = json.loads(row["params"])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["knn", "als", "svd", "hybrid", "all"], default="all")
    ap.add_argument("--only", nargs="*", default=[], help="lọc lưới, ví dụ: factors=64 reg=0.06")
    ap.add_argument("--reset", action="store_true", help="xóa kết quả cũ trong tuning.csv")
    args = ap.parse_args()
    only = dict(kv.split("=") for kv in args.only)
    stages = ["knn", "als", "svd", "hybrid"] if args.stage == "all" else [args.stage]

    ds = build_dataset()
    rows = [] if args.reset or not CSV.exists() else pd.read_csv(CSV).to_dict("records")

    def log(model: str, params: dict, res: dict, secs: float) -> None:
        key = json.dumps(params)
        rows[:] = [r for r in rows if not (r["model"] == model and r["params"] == key)]  # ghi đè lần thử trùng
        rows.append({"model": model, "params": key, "NDCG@10": res["NDCG@10"],
                     "Recall@10": res["Recall@10"], "Coverage": res["Coverage"],
                     "RMSE": res["RMSE"], "seconds": round(secs, 1)})
        print(f"{model:12s} {key:58s} NDCG={res['NDCG@10']:.4f} RMSE={res['RMSE']:.4f} ({secs:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    for stage in stages:
        if stage in ("knn", "als"):
            cls = ItemKNN if stage == "knn" else ImplicitALS
            for params in grid(GRIDS[stage], only):
                t0 = time.perf_counter()
                res = evaluate(cls(**params).fit(ds), ds, "val")
                log(NAMES[stage], params, res, time.perf_counter() - t0)
        elif stage == "svd":
            for params in grid(GRIDS["svd"], only):
                t0 = time.perf_counter()
                model = SVD(epochs=40, **params).fit(ds, val=ds.val)
                res = evaluate(model, ds, "val")
                # ghi lại số vòng tốt nhất do dừng sớm tìm ra
                log("SVD", {**params, "epochs": model.best_epoch_}, res, time.perf_counter() - t0)
        else:
            # Hybrid: ALS làm gốc (trọng số 1), dò trọng số của 3 thành phần còn lại
            best = best_params()
            parts = [ImplicitALS(**best["ImplicitALS"]).fit(ds), ItemKNN(**best["ItemKNN"]).fit(ds),
                     ContentBased().fit(ds), Popularity().fit(ds)]
            for w_knn, w_cb, w_pop in itertools.product([0.0, 0.5, 1.0], [0.0, 0.1, 0.25], [0.0, 0.1, 0.25]):
                weights = [1.0, w_knn, w_cb, w_pop]
                t0 = time.perf_counter()
                res = evaluate(Hybrid(parts, weights, prefit=True), ds, "val")
                log("Hybrid", {"weights": weights}, res, time.perf_counter() - t0)

    best = best_params()
    BEST.write_text(json.dumps(best, indent=2))
    print("\nTot nhat hien tai:\n" + json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
