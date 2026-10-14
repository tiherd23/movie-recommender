"""SASRec dựa vào THỨ TỰ xem phim đến mức nào?

Học lại SASRec trên train+val với số vòng tốt nhất đã tìm (results/sasrec.csv), lưu vào artifacts/sasrec.pt,
rồi chấm trên test hai lần với cùng trọng số:
    1. lịch sử đúng thứ tự thời gian (như bình thường)
    2. lịch sử bị xáo trộn thứ tự (cùng TẬP phim) -> mô hình chỉ còn thông tin "đã xem những phim nào"
Điểm giảm càng nhiều thì mô hình càng dựa vào thứ tự xem. Kết quả: results/sasrec_order.csv

Chạy: python scripts/sasrec_order_check.py            (học lại, khoảng 25 phút trên RTX 3050 đang hạ xung)
      python scripts/sasrec_order_check.py --reuse    (dùng artifacts/sasrec.pt có sẵn, chỉ chấm điểm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models.sasrec import SASRec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    ds = build_dataset()
    path = ROOT / "artifacts" / "sasrec.pt"
    if args.reuse and path.exists():
        model = SASRec.load(path)
    else:
        epochs = int(pd.read_csv(ROOT / "results" / "sasrec.csv")["best_epoch"].iloc[0])
        print(f"Hoc lai SASRec tren train+val voi {epochs} vong")
        model = SASRec(epochs=epochs).fit(ds, pd.concat([ds.train, ds.val], ignore_index=True), verbose=True)
        path.parent.mkdir(exist_ok=True)
        model.save(path)
    rows = [{"history": "đúng thứ tự", **evaluate(model, ds, "test")}]
    for seed in (0, 1, 2):
        rows.append({"history": f"xáo trộn (seed {seed})", **evaluate(model.shuffled_copy(seed), ds, "test")})
    table = pd.DataFrame(rows).set_index("history").drop(columns=["n_eval_users", "RMSE"]).round(4)
    print("\n" + table.to_string())
    table.to_csv(ROOT / "results" / "sasrec_order.csv")
    print("ORDER_DONE")


if __name__ == "__main__":
    main()
