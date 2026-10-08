"""Hai mô hình mốc. Mọi mô hình phức tạp hơn phải thắng được chúng."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


class Popularity(Recommender):
    """Gợi ý phim được nhiều người thích nhất (số lượt chấm >= like_threshold).
    Không cá nhân hóa: mọi người dùng nhận cùng một danh sách."""

    name = "Popularity"

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "Popularity":
        df = ds.train if df is None else df
        liked = df.loc[df["rating"] >= self.like_threshold, "item"].to_numpy()
        self.scores_ = np.bincount(liked, minlength=ds.n_items).astype(np.float32)
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.scores_, (len(users), 1))


class BiasBaseline(Recommender):
    """Dự đoán điểm = mu + b_u + b_i.

    mu : điểm trung bình toàn cục
    b_i: độ lệch của phim (phim hay thì dương)
    b_u: độ lệch của người dùng (người dễ tính thì dương)
    reg: hệ số co (shrinkage), kéo độ lệch của phim/người có ít dữ liệu về 0.
    """

    name = "BiasBaseline"

    def __init__(self, reg: float = 10.0):
        self.reg = reg

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "BiasBaseline":
        df = ds.train if df is None else df
        u, i = df["user"].to_numpy(), df["item"].to_numpy()
        r = df["rating"].to_numpy(np.float64)
        self.mu_ = r.mean()
        res = r - self.mu_
        self.b_i_ = np.bincount(i, res, ds.n_items) / (np.bincount(i, minlength=ds.n_items) + self.reg)
        res = res - self.b_i_[i]
        self.b_u_ = np.bincount(u, res, ds.n_users) / (np.bincount(u, minlength=ds.n_users) + self.reg)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.clip(self.mu_ + self.b_u_[users] + self.b_i_[items], 1.0, 5.0)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return (self.mu_ + self.b_u_[users][:, None] + self.b_i_[None, :]).astype(np.float32)
