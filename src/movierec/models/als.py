"""Phân rã ma trận cho phản hồi ngầm (implicit ALS, Hu-Koren-Volinsky 2008).

Khác SVD ở trên: không dự đoán số sao mà học từ việc "đã xem / chưa xem".
    p_ui = 1 nếu u đã xem i, ngược lại 0        (sở thích)
    c_ui = 1 + alpha nếu đã xem, ngược lại 1    (độ tin cậy)
Tối thiểu  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + reg (|x_u|^2 + |y_i|^2)
bằng bình phương tối thiểu luân phiên: cố định Y giải X (có nghiệm đóng), rồi đổi lại.
Vì học trên MỌI ô (kể cả ô chưa xem) nên mô hình này xếp hạng top-K tốt hơn SVD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


def _solve_side(X, other: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """Giải vector ẩn cho từng hàng của ma trận thưa X, giữ nguyên `other`."""
    f = other.shape[1]
    gram = other.T @ other + reg * np.eye(f, dtype=np.float32)  # phần chung cho mọi hàng
    out = np.zeros((X.shape[0], f), dtype=np.float32)
    indptr, indices = X.indptr, X.indices
    for row in range(X.shape[0]):
        idx = indices[indptr[row]:indptr[row + 1]]
        if len(idx) == 0:
            continue
        Y = other[idx]
        A = gram + alpha * (Y.T @ Y)
        b = (1.0 + alpha) * Y.sum(axis=0)
        out[row] = np.linalg.solve(A, b)
    return out


class ImplicitALS(Recommender):
    name = "ImplicitALS"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 10.0,
                 iters: int = 10, min_rating: float = 1.0, seed: int = 42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iters, self.min_rating, self.seed = iters, min_rating, seed

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "ImplicitALS":
        df = ds.train if df is None else df
        X = ds.matrix(df[df["rating"] >= self.min_rating], binary=True)
        Xt = X.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.Y_ = rng.normal(0, 0.01, (ds.n_items, self.factors)).astype(np.float32)
        for _ in range(self.iters):
            self.X_ = _solve_side(X, self.Y_, self.reg, self.alpha)
            self.Y_ = _solve_side(Xt, self.X_, self.reg, self.alpha)
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.X_[users] @ self.Y_.T
