"""Phân tích SAU thí nghiệm người dùng mới: danh sách của từng kiểu gợi ý bám gu của hồ sơ tới đâu?

Thí nghiệm người dùng mới (scripts/cold_start.py) chọn trọng số độ phổ biến 120/n theo NDCG@10. Khi dùng thử web, danh
sách của hồ sơ ngắn có nhiều phim nổi tiếng ngoài gu (vd. 6 phim hoạt hình -> 2/18 phim hoạt hình). Tập lệnh này đo
điều đó trên hồ sơ thật, chỉ để MÔ TẢ sự đánh đổi; nó không dùng để chọn lại mô hình mặc định.

Người dùng: người dùng nặng (≥ 30 phim thích ở train+val, ≥ 1 phim thích ở test) thuộc 4 phần kiểm tra của
cold_start.py (cùng cách chia, hạt giống 2026). Mỗi người tạo hồ sơ n phim thích lấy ngẫu nhiên trong lịch sử, n = 1, 3,
5, 7, 10, 20. Dịch vụ là bản web (artifacts/service.pkl, học trên toàn bộ dữ liệu, gồm cả những người này): chỉ số ở đây
không dùng nhãn tập kiểm tra nên điều đó không làm lợi cho kiểu nào, nhưng con số không so trực tiếp với Bảng 4.11.
Mỗi danh sách là 10 phim đầu theo đúng cách web gợi ý (bỏ các phim trong hồ sơ, không dùng độ đa dạng). Chỉ số:
  - khớp thể loại: cosine giữa tổng vector thể loại của hồ sơ và tổng vector thể loại của 10 phim (1 = cùng tỉ lệ thể loại);
  - trùng top phổ biến: tỉ lệ 10 phim nằm trong 10 phim được thích nhiều nhất (chưa bị bỏ vì thuộc hồ sơ);
  - ILD: độ khác nhau trung bình về thể loại giữa các cặp phim trong danh sách (cao = đa dạng).
Chạy: python scripts/taste_check.py [--users 500]   -> results/taste_check.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.beyond import genre_vectors  # noqa: E402
from movierec.data import build_dataset  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, RecommenderService  # noqa: E402

LEVELS = [1, 3, 5, 7, 10, 20]
VARIANTS = {"Trộn 120/n (Phổ biến, mặc định)": ("blend", 120.0), "Trộn 60/n (Cân bằng)": ("blend", 60.0),
            "Trộn 30/n": ("blend", 30.0), "Hai tầng (Sát gu nhất)": ("two_stage", None), "Phổ biến nhất": ("pop", None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=500)
    args = ap.parse_args()
    ds = build_dataset()
    svc = RecommenderService.load(DEFAULT_ARTIFACT)
    hist = pd.concat([ds.train, ds.val])
    liked_hist = hist[hist["rating"] >= 4].groupby("user")["item"].apply(list)
    test_rel = ds.test[ds.test["rating"] >= 4].groupby("user")["item"].apply(set)
    base = sorted(u for u in test_rel.index if len(liked_hist.get(u, [])) >= 10)
    fold_of = {u: i for i, part in enumerate(np.array_split(np.random.default_rng(2026).permutation(base), 5))
               for u in part}
    users = [u for u in base if fold_of[u] != 1 and len(liked_hist[u]) >= 30]  # phần 1 là tập phát triển
    rng = np.random.default_rng(7)
    users = rng.permutation(users)[: args.users]
    gvec = genre_vectors(svc.movies["genres"])
    pop_order = np.argsort(-svc.pop)
    rows = []
    for u in users:
        order = rng.permutation(liked_hist[u])
        for n in LEVELS:
            prof = [int(i) for i in order[:n]]
            want = gvec[prof].sum(axis=0)
            pop_list = [int(j) for j in pop_order[: 10 + n] if j not in set(prof)][:10]
            top_pop = set(pop_list)
            for name, (mode, c) in VARIANTS.items():
                if mode == "pop":
                    top = pop_list
                else:
                    top = [m["item"] for m in svc.recommend({i: 5 for i in prof}, k=10, mode=mode, pop_c=c)]
                got = gvec[top].sum(axis=0)
                cos = float(want @ got / (np.linalg.norm(want) * np.linalg.norm(got) + 1e-12))
                v = gvec[top]
                sim = v @ v.T
                ild = float(1 - (sim.sum() - np.trace(sim)) / (len(top) * (len(top) - 1)))
                rows.append({"user": int(u), "n": n, "kiểu": name, "khớp thể loại": cos,
                             "trùng top phổ biến": len(set(top) & top_pop) / 10, "ILD": ild})
    df = pd.DataFrame(rows)
    out = df.groupby(["n", "kiểu"])[["khớp thể loại", "trùng top phổ biến", "ILD"]].mean().round(3).reset_index()
    out.to_csv(ROOT / "results" / "taste_check.csv", index=False)
    print(f"{len(users)} người dùng nặng của 4 phần kiểm tra")
    print(out.pivot(index="n", columns="kiểu", values="khớp thể loại").to_string())
    print(out.pivot(index="n", columns="kiểu", values="trùng top phổ biến").to_string())
    print(out.pivot(index="n", columns="kiểu", values="ILD").to_string())


if __name__ == "__main__":
    main()
