"""Huấn luyện NeuMF (Neural Collaborative Filtering) trên GPU và đánh giá.

Chạy: python scripts/run_ncf.py
Bước 1: học trên train, theo dõi NDCG@10 trên val sau mỗi vòng, dừng sớm -> biết số vòng tốt nhất
Bước 2: học lại trên train+val với đúng số vòng đó, chấm trên test -> results/ncf.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models.ncf import NeuMF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-neg", type=int, default=4)
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (chua co CUDA)"
    print(f"PyTorch {torch.__version__} | thiet bi: {name}")
    ds = build_dataset()
    kw = dict(factors=args.factors, epochs=args.epochs, lr=args.lr, n_neg=args.n_neg)

    print("\n[1/2] Hoc tren train, dung som theo NDCG@10 tren val")
    t0 = time.perf_counter()
    model = NeuMF(**kw).fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"], verbose=True)
    best_epoch, best_val = model.best_epoch_, max(h["val"] for h in model.history_)
    print(f"Vong tot nhat: {best_epoch} | NDCG@10 val: {best_val:.4f} | {time.perf_counter() - t0:.0f}s")
    pd.DataFrame(model.history_).to_csv(ROOT / "results" / "ncf_history.csv", index=False)

    print(f"\n[2/2] Hoc lai tren train+val voi {best_epoch} vong, cham tren test")
    t0 = time.perf_counter()
    full = pd.concat([ds.train, ds.val], ignore_index=True)
    final = NeuMF(**{**kw, "epochs": best_epoch}).fit(ds, full)
    res = {"Model": "NeuMF", **evaluate(final, ds, "test"), "fit_s": round(time.perf_counter() - t0, 1)}
    table = pd.DataFrame([res]).set_index("Model").drop(columns="n_eval_users").round(4)
    table["best_epoch"], table["val_NDCG@10"] = best_epoch, round(best_val, 4)
    print("\n" + table.to_string())
    table.to_csv(ROOT / "results" / "ncf.csv")
    print("NCF_DONE")


if __name__ == "__main__":
    main()
