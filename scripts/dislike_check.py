"""Kiểm tra trên dịch vụ web: chấm thấp một phim có kéo các phim giống nó lên danh sách gợi ý không?

Lo ngại: đặc trưng ItemKNN của kiến trúc hai tầng cộng độ tương đồng với MỌI phim đã chấm (đúng như lúc huấn luyện,
vì trong MovieLens việc đã xem một phim cũng là tín hiệu quan tâm), nên về lý thuyết chấm 1★ cũng có thể đẩy phim
giống phim đó lên. Script đo trực tiếp điều này trên người dùng MovieLens thật:
  - Gốc: mỗi người, lấy ngẫu nhiên tối đa 10 phim họ chấm 4-5★ làm phim đã thích (đủ để dùng kiến trúc hai tầng).
  - Thêm 3 phim chính họ chấm 1-2★ vào hồ sơ, lần lượt với 1★ (đúng như họ đã chấm) và 5★ (đối chứng).
  - Đếm số phim trong top 18 thuộc "vùng lân cận" của 3 phim đó (20 phim ItemKNN giống nhất với mỗi phim).
Nếu chấm 1★ gần như không làm tăng số đó, còn chấm 5★ làm tăng rõ, thì hệ thống phân biệt được thích và không thích.
Chạy: python scripts/dislike_check.py [--users 300]   -> results/dislike_check.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, RecommenderService  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ds = build_dataset()
    svc = RecommenderService.load(DEFAULT_ARTIFACT)
    full = pd.concat([ds.train, ds.val, ds.test], ignore_index=True)
    sim = svc.sim.tocsr()
    rows = []
    users = rng.permutation(full["user"].unique())
    for u in users:
        h = full[full["user"] == u]
        good, bad = h.loc[h["rating"] >= 4, "item"].to_numpy(), h.loc[h["rating"] <= 2, "item"].to_numpy()
        if len(good) < 5 or len(bad) < 3:
            continue
        liked = rng.choice(good, min(10, len(good)), replace=False)
        disliked = rng.choice(bad, 3, replace=False)
        near = set()
        for j in disliked:
            row = sim[int(j)]
            near |= set(row.indices[np.argsort(-row.data)[:20]].tolist())
        near -= set(liked.tolist())
        base = {int(i): 5 for i in liked}
        out = {"user": int(u)}
        for label, extra in [("goc", {}), ("them_1sao", {int(j): 1 for j in disliked}),
                             ("them_5sao", {int(j): 5 for j in disliked})]:
            recs = svc.recommend({**base, **extra}, k=18)
            out[label] = sum(m["item"] in near for m in recs)
        rows.append(out)
        if len(rows) >= args.users:
            break
    df = pd.DataFrame(rows)
    path = ROOT / "results" / "dislike_check.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} người dùng. Số phim (trong top 18) thuộc vùng lân cận của 3 phim họ chấm thấp:")
    for c, name in [("goc", "Gốc (chưa đưa 3 phim vào)"), ("them_1sao", "Thêm 3 phim với 1★"),
                    ("them_5sao", "Thêm 3 phim với 5★ (đối chứng)")]:
        print(f"  {name:32s} trung bình {df[c].mean():.2f}")
    print(f"  Người có số phim lân cận tăng khi chấm 1★: {(df['them_1sao'] > df['goc']).mean():.1%}, "
          f"khi chấm 5★: {(df['them_5sao'] > df['goc']).mean():.1%}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
