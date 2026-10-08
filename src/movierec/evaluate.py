"""Quy trình đánh giá dùng chung cho mọi mô hình."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .metrics import ranking_metrics, rmse
from .models.base import Recommender


def recommend_topk(model: Recommender, seen, n_users: int, k: int = 10, batch: int = 1024) -> np.ndarray:
    """Top-k cho mọi người dùng, loại các phim đã xem (ma trận thưa `seen`)."""
    out = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users))
        scores = model.score_users(users).astype(np.float32, copy=True)
        scores[seen[users].nonzero()] = -np.inf  # không gợi ý lại phim đã xem
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]  # lấy k phim điểm cao nhất
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
        out[users] = np.take_along_axis(part, order, axis=1)
    return out


def evaluate(model: Recommender, ds: Dataset, split: str = "test", k: int = 10,
             like_threshold: float = 4.0) -> dict[str, float]:
    """Đánh giá trên tập val hoặc test.

    Phim "liên quan" = phim người dùng chấm >= like_threshold trong tập đánh giá.
    """
    target = ds.val if split == "val" else ds.test
    seen_df = ds.train if split == "val" else pd.concat([ds.train, ds.val])
    seen = ds.matrix(seen_df, binary=True)

    relevant: list[set[int]] = [set() for _ in range(ds.n_users)]
    liked = target[target["rating"] >= like_threshold]
    for u, i in zip(liked["user"].to_numpy(), liked["item"].to_numpy()):
        relevant[u].add(int(i))

    result = ranking_metrics(recommend_topk(model, seen, ds.n_users, k), relevant, ds.n_items)
    pred = model.predict(target["user"].to_numpy(), target["item"].to_numpy())
    result["RMSE"] = rmse(target["rating"].to_numpy(), pred) if pred is not None else float("nan")
    return result
