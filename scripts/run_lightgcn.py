"""Huấn luyện LightGCN (lọc cộng tác bằng mạng nơ-ron đồ thị) và đánh giá, cùng quy trình với các mô hình khác.

Chạy: python scripts/run_lightgcn.py
Bước 1: học trên train, đánh giá NDCG@10 trên val mỗi vài vòng, dừng sớm -> biết số vòng tốt nhất
Bước 2: học lại trên train+val với đúng số vòng đó, chấm trên test -> results/lightgcn.csv
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
from movierec.models.lightgcn import LightGCN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default=None, help="cuda hoặc cpu (mặc định: tự chọn)")
    ap.add_argument("--final-only", action="store_true",
                    help="bỏ qua bước 1, lấy số vòng tốt nhất từ results/lightgcn_history.csv đã có")
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (chua co CUDA)"
    print(f"PyTorch {torch.__version__} | thiet bi: {args.device or name}")
    ds = build_dataset()
    kw = dict(dim=args.dim, layers=args.layers, lr=args.lr, reg=args.reg, epochs=args.epochs, device=args.device)

    hist_path = ROOT / "results" / "lightgcn_history.csv"
    if args.final_only:
        hist = pd.read_csv(hist_path).dropna(subset=["val"])
        best = hist.loc[hist["val"].idxmax()]
        best_epoch, best_val = int(best["epoch"]), float(best["val"])
        print(f"\n[1/2] Bo qua, dung ket qua cu: vong tot nhat {best_epoch} | NDCG@10 val: {best_val:.4f}")
    else:
        print("\n[1/2] Hoc tren train, dung som theo NDCG@10 tren val")
        t0 = time.perf_counter()
        model = LightGCN(**kw).fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"], verbose=True)
        best_epoch = model.best_epoch_
        best_val = max(h["val"] for h in model.history_ if "val" in h)
        print(f"Vong tot nhat: {best_epoch} | NDCG@10 val: {best_val:.4f} | {time.perf_counter() - t0:.0f}s")
        (ROOT / "results").mkdir(exist_ok=True)
        pd.DataFrame(model.history_).to_csv(hist_path, index=False)

    print(f"\n[2/2] Hoc lai tren train+val voi {best_epoch} vong, cham tren test")
    t0 = time.perf_counter()
    full = pd.concat([ds.train, ds.val], ignore_index=True)
    final = LightGCN(**{**kw, "epochs": best_epoch}).fit(ds, full, verbose=True)
    res = {"Model": "LightGCN", **evaluate(final, ds, "test"), "fit_s": round(time.perf_counter() - t0, 1)}
    table = pd.DataFrame([res]).set_index("Model").drop(columns="n_eval_users").round(4)
    table["best_epoch"], table["val_NDCG@10"] = best_epoch, round(best_val, 4)
    print("\n" + table.to_string())
    table.to_csv(ROOT / "results" / "lightgcn.csv")
    print("LIGHTGCN_DONE")


if __name__ == "__main__":
    main()
