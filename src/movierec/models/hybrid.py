"""Mô hình lai: cộng có trọng số điểm của nhiều mô hình thành phần.

Điểm của mỗi mô hình có thang đo khác nhau, nên trước khi cộng phải chuẩn hóa
z-score theo từng người dùng: (điểm - trung bình) / độ lệch chuẩn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


def zscore_rows(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.float32)
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-8)


class Hybrid(Recommender):
    name = "Hybrid"

    def __init__(self, models: list[Recommender], weights: list[float], prefit: bool = False):
        assert len(models) == len(weights)
        self.models, self.weights, self.prefit = models, weights, prefit

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "Hybrid":
        if not self.prefit:  # prefit=True: dùng lại các mô hình đã huấn luyện
            for m in self.models:
                m.fit(ds, df)
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        total = None
        for m, w in zip(self.models, self.weights):
            if w == 0:
                continue
            part = w * zscore_rows(m.score_users(users))
            total = part if total is None else total + part
        return total
