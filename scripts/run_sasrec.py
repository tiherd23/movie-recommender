"""Huấn luyện SASRec (gợi ý theo chuỗi) và đánh giá, cùng quy trình với các mô hình khác.

Chạy: python scripts/run_sasrec.py
Bước 1: học trên train, đánh giá NDCG@10 trên val mỗi vài vòng, dừng sớm -> biết số vòng tốt nhất
Bước 2: học lại trên train+val với đúng số vòng đó, chấm trên test -> results/sasrec.csv

Thêm --shuffle-order để huấn luyện và chấm với thứ tự lịch sử bị xáo trộn -> results/sasrec_shuffled.csv
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
from movierec.models.sasrec import SASRec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda hoặc cpu (mặc định: tự chọn)")
    ap.add_argument("--shuffle-order", action="store_true",
                    help="xáo trộn thứ tự phim trong lịch sử (cả lúc học lẫn lúc gợi ý) -> results/sasrec_shuffled.csv")
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (chua co CUDA)"
    print(f"PyTorch {torch.__version__} | thiet bi: {args.device or name}")
    ds = build_dataset()
    kw = dict(dim=args.dim, max_len=args.max_len, layers=args.layers, dropout=args.dropout, lr=args.lr,
              epochs=args.epochs, batch=args.batch, device=args.device, shuffle_order=args.shuffle_order)
    tag = "_shuffled" if args.shuffle_order else ""
    label = "SASRec (xáo trộn thứ tự)" if args.shuffle_order else "SASRec"

    print("\n[1/2] Hoc tren train, dung som theo NDCG@10 tren val")
    t0 = time.perf_counter()
    model = SASRec(**kw).fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"], verbose=True)
    vals = [h["val"] for h in model.history_ if "val" in h]
    best_epoch, best_val = model.best_epoch_, max(vals)
    print(f"Vong tot nhat: {best_epoch} | NDCG@10 val: {best_val:.4f} | {time.perf_counter() - t0:.0f}s")
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(model.history_).to_csv(ROOT / "results" / f"sasrec{tag}_history.csv", index=False)

    print(f"\n[2/2] Hoc lai tren train+val voi {best_epoch} vong, cham tren test")
    t0 = time.perf_counter()
    full = pd.concat([ds.train, ds.val], ignore_index=True)
    final = SASRec(**{**kw, "epochs": best_epoch}).fit(ds, full, verbose=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    final.save(ROOT / "artifacts" / f"sasrec{tag}.pt")
    res = {"Model": label, **evaluate(final, ds, "test"), "fit_s": round(time.perf_counter() - t0, 1)}
    table = pd.DataFrame([res]).set_index("Model").drop(columns="n_eval_users").round(4)
    table["best_epoch"], table["val_NDCG@10"] = best_epoch, round(best_val, 4)
    print("\n" + table.to_string())
    table.to_csv(ROOT / "results" / f"sasrec{tag}.csv")
    print("SASREC_DONE")


if __name__ == "__main__":
    main()
