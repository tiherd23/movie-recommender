"""Gộp các nhóm kiểm tra của scripts/cold_start.py (seed 0, 2, 3, 4; seed 1 là nhóm phát triển, không gộp) và kiểm định.

Chỉ số chính (đặt trước): NDCG@10 trung bình của mỗi người trên các mức n = 1, 2, 3, 5, 7, 10, 15, 20, 30 phim thích.
Phép so sánh chính, hiệu chỉnh Holm: công thức trộn với trọng số phổ biến w(n) (mô hình chọn cho web)
  - so với Hai tầng học cả hồ sơ ngắn;  - so với Phổ biến nhất.
So sánh phụ (mô tả): với công thức trộn gốc và Hai tầng gốc. Bảng theo từng n chỉ để mô tả.
Chạy: python scripts/cold_start_summary.py [--pool nang|nhe]  -> results/cold_start_pooled*.csv, cold_start_tests*.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FULL = 10_000
CHOSEN = "Trộn, w(n)=60·n^-1"  # w(n) = 60/n, chọn trên nhóm phát triển (seed 1, hai loại người dùng)
PRIMARY = ["Hai tầng (học cả hồ sơ ngắn)", "Phổ biến nhất"]
SECONDARY = ["Công thức trộn", "Hai tầng", "Trộn, w(n)=60·n^-0.8"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["nang", "nhe"], default="nang",
                    help="nang: người có ≥ 30 phim thích; nhe: người có ≥ 10 phim thích (file *_min10)")
    args = ap.parse_args()
    suf = "_min10" if args.pool == "nhe" else ""
    files = {0: f"cold_start_users{suf}.csv", 2: f"cold_start_users_seed2{suf}.csv",
             3: f"cold_start_users_seed3{suf}.csv", 4: f"cold_start_users_seed4{suf}.csv"}
    df = pd.concat([pd.read_csv(ROOT / "results" / f).assign(seed=s) for s, f in files.items()], ignore_index=True)
    df["uid"] = df["seed"].astype(str) + "_" + df["user"].astype(str)
    rng = np.random.default_rng(0)

    # bảng theo n (mô tả): trung bình, CI95 bootstrap, độ trùng với top 10 phổ biến
    rows = []
    for (n, m), g in df.groupby(["n", "model"]):
        x = g["ndcg"].to_numpy()
        boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(axis=1)
        rows.append({"số phim thích": "đầy đủ" if n == FULL else n, "mô hình": m, "NDCG@10": round(x.mean(), 4),
                     "CI95 thấp": round(np.percentile(boot, 2.5), 4), "CI95 cao": round(np.percentile(boot, 97.5), 4),
                     "trùng top phổ biến": round(g["overlap_pop"].mean(), 3), "số người": len(g)})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(ROOT / "results" / f"cold_start_pooled{suf}.csv", index=False)
    print(pooled.pivot(index="số phim thích", columns="mô hình", values="NDCG@10").to_string())

    # chỉ số chính theo người
    short = df[df["n"] != FULL]
    if args.pool == "nhe":  # nhóm nhẹ: ai cũng có ít nhất 10 phim thích, nên chỉ số chính tính trên n = 1..10
        short = short[short["n"] <= 10]
    per_user = short.groupby(["uid", "model"])["ndcg"].mean().unstack()
    tests = []
    for other in PRIMARY + SECONDARY:
        d = (per_user[CHOSEN] - per_user[other]).dropna().to_numpy()
        boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
        tests.append({"so sánh": f"{CHOSEN} − {other}", "loại": "chính" if other in PRIMARY else "phụ",
                      "chênh lệch": d.mean(), "tương đối %": 100 * d.mean() / per_user[other].mean(),
                      "CI95 thấp": np.percentile(boot, 2.5), "CI95 cao": np.percentile(boot, 97.5),
                      "p (Wilcoxon)": wilcoxon(d).pvalue, "số người": len(d)})
    t = pd.DataFrame(tests)
    prim = t["loại"] == "chính"
    order = t[prim].sort_values("p (Wilcoxon)").index  # Holm cho các phép so sánh chính
    m = len(order)
    adj, running = {}, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * t.loc[i, "p (Wilcoxon)"]))
        adj[i] = running
    t["p Holm"] = [adj.get(i, np.nan) for i in t.index]
    t.round(5).to_csv(ROOT / "results" / f"cold_start_tests{suf}.csv", index=False)
    print(f"\nChỉ số chính: NDCG@10 trung bình trên n = 1..{30 if args.pool == 'nang' else 10}, mỗi người một giá trị")
    print(per_user.mean().sort_values(ascending=False).round(4).to_string())
    print()
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
