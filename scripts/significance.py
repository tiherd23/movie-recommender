"""Kiểm định ý nghĩa thống kê cho bảng kết quả chính (11 mô hình, tập kiểm tra).

Chạy: python scripts/significance.py   (cần artifacts/topk/*.npy do scripts/beyond_accuracy.py tạo ra; vài phút, CPU)

- Tính NDCG@10, Recall@10, HitRate@10 cho TỪNG người dùng từ danh sách top-10 đã lưu (khớp với bảng chính).
- Khoảng tin cậy 95% bằng bootstrap theo người dùng (10.000 lần lấy mẫu lại).
- So sánh ghép cặp các cặp mô hình quan trọng: chênh lệch trung bình kèm khoảng tin cậy bootstrap ghép cặp, kiểm
  định Wilcoxon ký hạng (bỏ các người dùng hai mô hình bằng điểm nhau), hiệu chỉnh Holm cho nhiều phép so sánh.
Kết quả -> results/significance_ci.csv, results/significance_pairs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402

ORDER = ["Popularity", "BiasBaseline", "ContentBased", "ItemKNN", "SVD", "ImplicitALS", "Hybrid",
         "TwoStage", "NeuMF", "LightGCN", "SASRec"]
# (A, B): A có tốt hơn B thật không. Chọn các cặp mà báo cáo đưa ra kết luận.
PAIRS = [("TwoStage", "ImplicitALS"), ("TwoStage", "Hybrid"), ("ImplicitALS", "Hybrid"), ("ImplicitALS", "LightGCN"),
         ("LightGCN", "NeuMF"), ("NeuMF", "ItemKNN"), ("ItemKNN", "Popularity"), ("TwoStage", "Popularity"),
         ("TwoStage", "LightGCN"), ("SASRec", "TwoStage")]
B = 10_000
SEED = 2026


def per_user(topk: np.ndarray, relevant: list[set[int]], users: np.ndarray) -> pd.DataFrame:
    """NDCG@K, Recall@K, Hit@K của từng người dùng (cùng công thức với movierec.metrics.ranking_metrics)."""
    k = topk.shape[1]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rows = []
    for u in users:
        rel = relevant[u]
        hits = np.fromiter((i in rel for i in topk[u]), dtype=np.float64, count=k)
        rows.append((hits @ disc / disc[: min(len(rel), k)].sum(), hits.sum() / len(rel), float(hits.any())))
    return pd.DataFrame(rows, columns=["NDCG@10", "Recall@10", "HitRate@10"], index=users)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ds = build_dataset()
    relevant: list[set[int]] = [set() for _ in range(ds.n_users)]
    liked = ds.test[ds.test["rating"] >= 4.0]
    for u, i in zip(liked["user"].to_numpy(), liked["item"].to_numpy()):
        relevant[u].add(int(i))
    users = np.array([u for u, r in enumerate(relevant) if r])
    scores = {m: per_user(np.load(ROOT / "artifacts" / "topk" / f"{m}.npy"), relevant, users) for m in ORDER}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(users), size=(B, len(users)))  # dùng CHUNG các mẫu bootstrap cho mọi mô hình

    ci_rows = []
    for m in ORDER:
        for metric in ["NDCG@10", "Recall@10", "HitRate@10"]:
            v = scores[m][metric].to_numpy()
            boot = v[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ci_rows.append({"Model": m, "metric": metric, "mean": v.mean(), "ci_low": lo, "ci_high": hi})
    ci = pd.DataFrame(ci_rows)
    ci.round(4).to_csv(ROOT / "results" / "significance_ci.csv", index=False)

    pair_rows = []
    for a, b in PAIRS:
        va, vb = scores[a]["NDCG@10"].to_numpy(), scores[b]["NDCG@10"].to_numpy()
        d = va - vb
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        nz = d != 0
        p = wilcoxon(va[nz], vb[nz]).pvalue if nz.sum() > 0 else 1.0
        pair_rows.append({"A": a, "B": b, "NDCG_A": va.mean(), "NDCG_B": vb.mean(), "diff": d.mean(),
                          "rel_diff": d.mean() / vb.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                          "users_A_better": int((d > 0).sum()), "users_B_better": int((d < 0).sum()),
                          "ties": int((d == 0).sum()), "p_wilcoxon": p})
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = holm(pairs["p_wilcoxon"].to_numpy())
    pairs.to_csv(ROOT / "results" / "significance_pairs.csv", index=False)

    print(f"{len(users)} người dùng có phim thích trong tập kiểm tra; bootstrap {B} lần\n")
    nd = ci[ci.metric == "NDCG@10"].set_index("Model")
    print(nd[["mean", "ci_low", "ci_high"]].round(4).to_string())
    print()
    show = pairs.copy()
    for c in ["NDCG_A", "NDCG_B", "diff", "diff_ci_low", "diff_ci_high"]:
        show[c] = show[c].round(4)
    show["rel_diff"] = (show["rel_diff"] * 100).round(1)
    show["p_wilcoxon"] = show["p_wilcoxon"].map(lambda x: f"{x:.2e}")
    show["p_holm"] = show["p_holm"].map(lambda x: f"{x:.2e}")
    print(show.to_string(index=False))
    print("SIGNIFICANCE_DONE")


if __name__ == "__main__":
    main()
