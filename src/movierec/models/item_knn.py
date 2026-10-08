"""Lọc cộng tác dựa trên phim (Item-based kNN).

Ý tưởng: hai phim "giống nhau" nếu cùng được một nhóm người xem.
Điểm gợi ý phim j cho người dùng u = tổng độ tương đồng giữa j và các phim u đã xem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


class ItemKNN(Recommender):
    name = "ItemKNN"

    def __init__(self, k: int = 50, shrink: float = 20.0):
        self.k = k  # số phim láng giềng giữ lại cho mỗi phim
        self.shrink = shrink  # giảm độ tin cậy của cặp phim có ít người xem chung

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "ItemKNN":
        self.X_ = ds.matrix(df, binary=True)  # user x item, 1 = đã xem
        co = (self.X_.T @ self.X_).toarray().astype(np.float32)  # số người xem chung
        norm = np.sqrt(np.diag(co))
        # cosine có shrinkage: co_ij / (|i| * |j| + shrink)
        sim = co / (np.outer(norm, norm) + self.shrink + 1e-9)  # 1e-9: tránh chia 0 với phim chưa ai xem
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0] - 1:
            # mỗi hàng chỉ giữ k giá trị lớn nhất, còn lại gán 0 để bớt nhiễu
            cut = np.partition(sim, -self.k, axis=1)[:, -self.k][:, None]
            sim[sim < cut] = 0.0
        self.sim_ = sim
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X_[users] @ self.sim_, dtype=np.float32)

    def similar_items(self, item: int, n: int = 10) -> np.ndarray:
        """Top-n phim giống phim `item` nhất (dùng cho trang 'phim tương tự')."""
        return np.argsort(-self.sim_[item])[:n]
