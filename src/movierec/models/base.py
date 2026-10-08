"""Giao diện chung: mọi mô hình đều có fit() và score_users()."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset


class Recommender:
    name = "base"

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "Recommender":
        """Học từ df (mặc định là ds.train)."""
        raise NotImplementedError

    def score_users(self, users: np.ndarray) -> np.ndarray:
        """Trả về mảng (len(users), n_items): điểm càng cao càng nên gợi ý."""
        raise NotImplementedError

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray | None:
        """Dự đoán điểm 1-5 cho từng cặp (user, item). None nếu mô hình không hỗ trợ."""
        return None
