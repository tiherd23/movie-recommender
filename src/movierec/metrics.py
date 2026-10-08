"""Các độ đo đánh giá.

- RMSE: sai số dự đoán điểm (càng thấp càng tốt).
- Precision@K, Recall@K, NDCG@K, HitRate@K: chất lượng danh sách top-K (càng cao càng tốt).
- Coverage: tỉ lệ phim trong kho từng được gợi ý cho ít nhất một người.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def ranking_metrics(topk: np.ndarray, relevant: list[set[int]], n_items: int) -> dict[str, float]:
    """topk: mảng (n_users, K) chỉ số phim được gợi ý, đã xếp theo điểm giảm dần.
    relevant[u]: tập phim người dùng u thật sự thích trong tập kiểm tra.
    Người dùng không có phim thích nào trong tập kiểm tra bị bỏ qua.
    """
    k = topk.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 1/log2(vị trí + 1)
    prec, rec, ndcg, hit = [], [], [], []
    for u, rel in enumerate(relevant):
        if not rel:
            continue
        hits = np.fromiter((i in rel for i in topk[u]), dtype=np.float64, count=k)
        n_hit = hits.sum()
        prec.append(n_hit / k)
        rec.append(n_hit / len(rel))
        hit.append(float(n_hit > 0))
        idcg = discounts[: min(len(rel), k)].sum()  # DCG của danh sách lý tưởng
        ndcg.append((hits * discounts).sum() / idcg)
    return {
        f"Precision@{k}": float(np.mean(prec)),
        f"Recall@{k}": float(np.mean(rec)),
        f"NDCG@{k}": float(np.mean(ndcg)),
        f"HitRate@{k}": float(np.mean(hit)),
        "Coverage": float(len(np.unique(topk)) / n_items),
        "n_eval_users": float(len(prec)),
    }
