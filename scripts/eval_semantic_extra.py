"""Chấm chỉ mục tìm theo mô tả ĐÃ CHỌN (artifacts/semantic.npz) trên bộ câu hỏi mới, không chỉnh lại gì.

Bộ câu mới data/eval/semantic_queries_moi.csv (43 câu) được viết sau khi đã chốt mô hình và trọng số độ phổ biến,
gồm cả phim ít nổi tiếng hơn (hạng độ phổ biến tới khoảng 730), để kiểm tra kết quả ở Bảng 4.12 có giữ được không.
Chạy: python scripts/eval_semantic_extra.py   (lần đầu ~1,5 phút để nạp BGE-M3)
Kết quả -> results/semantic_eval_moi.csv (từng câu) và in Hit@1/5/10, MRR, kèm khoảng tin cậy bootstrap 95%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.models import Popularity  # noqa: E402
from movierec.semantic import SemanticIndex  # noqa: E402


def main() -> None:
    ds = build_dataset()
    full = pd.concat([ds.train, ds.val, ds.test], ignore_index=True)
    pop = Popularity().fit(ds, full).scores_          # giống scripts/eval_semantic.py
    pop_rank = pd.Series(pop).rank(ascending=False, method="min").astype(int).to_numpy()
    index = SemanticIndex.load()
    print(f"Chỉ mục: {index.model_name} | trọng số độ phổ biến {index.pop_weight}")
    title_to_item = {t: i for i, t in enumerate(ds.movies["title"])}
    rows = []
    for name in ["semantic_queries.csv", "semantic_queries_moi.csv"]:
        q = pd.read_csv(ROOT / "data" / "eval" / name)
        for _, r in q.iterrows():
            item = title_to_item[r["target"]]
            for weight, label in [(0.0, "chỉ nội dung"), (index.pop_weight, "nội dung + độ phổ biến")]:
                ranked = [i for i, _ in index.search(r["query"], k=len(ds.movies), popularity=pop, pop_weight=weight)]
                rank = ranked.index(item) + 1
                rows.append({"set": r["split"], "config": label, "query": r["query"], "target": r["target"],
                             "pop_rank": int(pop_rank[item]), "rank": rank})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "semantic_eval_moi.csv", index=False)
    rng = np.random.default_rng(0)
    out = []
    for (s, c), part in df.groupby(["set", "config"], sort=False):
        r = part["rank"].to_numpy()
        stats = {"Hit@1": (r <= 1), "Hit@5": (r <= 5), "Hit@10": (r <= 10), "MRR": 1.0 / r}
        row = {"set": s, "config": c, "n": len(r)}
        for k, v in stats.items():
            v = v.astype(float)
            boot = v[rng.integers(0, len(v), size=(5000, len(v)))].mean(axis=1)
            row[k] = f"{v.mean():.3f} [{np.percentile(boot, 2.5):.2f}–{np.percentile(boot, 97.5):.2f}]"
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))
    new = df[(df["set"] == "test2") & (df["config"] != "chỉ nội dung")]
    print("\nBộ mới, theo độ nổi tiếng của phim đúng (hạng độ phổ biến):")
    for lo, hi in [(1, 100), (101, 300), (301, 10_000)]:
        part = new[(new["pop_rank"] >= lo) & (new["pop_rank"] <= hi)]
        if len(part):
            print(f"  hạng {lo}-{hi if hi < 10_000 else '...'}: {len(part)} câu | Hit@1 {np.mean(part['rank'] <= 1):.2f}"
                  f" | Hit@10 {np.mean(part['rank'] <= 10):.2f} | MRR {np.mean(1 / part['rank']):.3f}")
    print("SEMANTIC_EXTRA_DONE")


if __name__ == "__main__":
    main()
