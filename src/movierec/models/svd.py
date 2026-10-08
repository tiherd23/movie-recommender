"""SVD kiểu Funk (phân rã ma trận có độ lệch), tự cài đặt bằng NumPy.

Dự đoán:  r_hat(u, i) = mu + b_u + b_i + p_u . q_i
    p_u: vector ẩn của người dùng (gu), q_i: vector ẩn của phim (đặc điểm).
Huấn luyện: giảm dần ngẫu nhiên theo lô (mini-batch SGD) để tối thiểu
    sum (r - r_hat)^2 + reg * (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


class SVD(Recommender):
    name = "SVD"

    def __init__(self, factors: int = 64, lr: float = 0.01, reg: float = 0.05,
                 epochs: int = 30, batch: int = 2048, seed: int = 42):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.batch, self.seed = epochs, batch, seed

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None, val: pd.DataFrame | None = None,
            patience: int = 2) -> "SVD":
        """Nếu truyền `val`, dừng sớm khi RMSE trên val không giảm sau `patience` vòng."""
        df = ds.train if df is None else df
        rng = np.random.default_rng(self.seed)
        u_all, i_all = df["user"].to_numpy(), df["item"].to_numpy()
        r_all = df["rating"].to_numpy(np.float32)

        self.mu_ = float(r_all.mean())
        self.bu_ = np.zeros(ds.n_users, np.float32)
        self.bi_ = np.zeros(ds.n_items, np.float32)
        self.P_ = rng.normal(0, 0.05, (ds.n_users, self.factors)).astype(np.float32)
        self.Q_ = rng.normal(0, 0.05, (ds.n_items, self.factors)).astype(np.float32)

        best, best_state, bad = np.inf, None, 0
        self.history_: list[float] = []
        for epoch in range(self.epochs):
            perm = rng.permutation(len(r_all))
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                u, i, r = u_all[b], i_all[b], r_all[b]
                pu, qi = self.P_[u], self.Q_[i]
                err = r - (self.mu_ + self.bu_[u] + self.bi_[i] + np.einsum("ij,ij->i", pu, qi))
                # np.add.at cộng dồn đúng cả khi một user/phim xuất hiện nhiều lần trong lô
                np.add.at(self.bu_, u, self.lr * (err - self.reg * self.bu_[u]))
                np.add.at(self.bi_, i, self.lr * (err - self.reg * self.bi_[i]))
                np.add.at(self.P_, u, self.lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q_, i, self.lr * (err[:, None] * pu - self.reg * qi))
            if val is not None:
                pred = self.predict(val["user"].to_numpy(), val["item"].to_numpy())
                score = float(np.sqrt(np.mean((val["rating"].to_numpy() - pred) ** 2)))
                self.history_.append(score)
                if score < best - 1e-4:
                    best, bad = score, 0
                    best_state = (self.bu_.copy(), self.bi_.copy(), self.P_.copy(), self.Q_.copy())
                    self.best_epoch_ = epoch + 1
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.bu_, self.bi_, self.P_, self.Q_ = best_state
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        dot = np.einsum("ij,ij->i", self.P_[users], self.Q_[items])
        return np.clip(self.mu_ + self.bu_[users] + self.bi_[items] + dot, 1.0, 5.0)

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.mu_ + self.bu_[users][:, None] + self.bi_[None, :] + self.P_[users] @ self.Q_.T
