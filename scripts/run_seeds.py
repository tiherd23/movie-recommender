"""Chạy lại các mô hình có tính ngẫu nhiên với nhiều hạt giống để biết kết quả dao động bao nhiêu.

Chạy: python scripts/run_seeds.py --models ImplicitALS TwoStage            (CPU, khoảng 20-30 phút)
      python scripts/run_seeds.py --models NeuMF SASRec --seeds 1 2 3      (nên chạy trên GPU; SASRec ~25 phút/lần)
Mỗi lần: học trên train+val với cấu hình và số vòng đã chọn trước đây (không chỉnh lại gì), chấm trên test.
TwoStage học lại cả bộ xếp hạng LightGBM (tầng 1 trên train, nhãn từ val) rồi học lại tầng 1 trên train+val.
Kết quả cộng dồn vào results/seeds.csv (mỗi dòng một mô hình x một hạt giống); in trung bình ± độ lệch chuẩn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402

OUT = ROOT / "results" / "seeds.csv"
METRICS = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10", "Coverage"]


def build(name: str, seed: int, ds, full, best: dict):
    if name == "ImplicitALS":
        from movierec.models import ImplicitALS
        return ImplicitALS(**best["ImplicitALS"], seed=seed).fit(ds, full)
    if name == "TwoStage":
        from movierec.ranker import TwoStageRanker
        ranker = TwoStageRanker({**best["ImplicitALS"], "seed": seed}, best["ItemKNN"], seed=seed)
        ranker.fit_ranker(ds, hist=ds.train, label=ds.val)
        return ranker.fit(ds, full)
    if name == "NeuMF":
        from movierec.models.ncf import NeuMF
        epochs = int(pd.read_csv(ROOT / "results" / "ncf.csv")["best_epoch"].iloc[0])
        return NeuMF(epochs=epochs, seed=seed).fit(ds, full)
    if name == "SASRec":
        from movierec.models.sasrec import SASRec
        epochs = int(pd.read_csv(ROOT / "results" / "sasrec.csv")["best_epoch"].iloc[0])
        return SASRec(epochs=epochs, seed=seed).fit(ds, full)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ImplicitALS", "TwoStage"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    ds = build_dataset()
    full = pd.concat([ds.train, ds.val], ignore_index=True)
    best = json.loads((ROOT / "results" / "best_params.json").read_text())
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    for name in args.models:
        for seed in args.seeds:
            t0 = time.perf_counter()
            res = evaluate(build(name, seed, ds, full, best), ds, "test")
            rows = [r for r in rows if not (r["Model"] == name and r["seed"] == seed)]
            rows.append({"Model": name, "seed": seed, **{m: res[m] for m in METRICS},
                         "seconds": round(time.perf_counter() - t0, 1)})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[{name} seed={seed}] NDCG@10 {res['NDCG@10']:.4f} | {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("Model")[METRICS].agg(["mean", "std"]).round(4)
    print("\n" + summary.to_string())
    print("SEEDS_DONE")


if __name__ == "__main__":
    main()
