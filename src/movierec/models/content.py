"""Gợi ý dựa trên nội dung (content-based).

Mỗi phim -> vector TF-IDF từ thể loại và thập niên phát hành.
Hồ sơ người dùng = trung bình vector các phim họ thích (chấm >= like_threshold).
Điểm gợi ý = độ tương đồng cosine giữa hồ sơ và vector phim.
Ưu điểm: gợi ý được phim mới chưa ai chấm (cold-start phía phim).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data import Dataset
from .base import Recommender


def item_documents(movies: pd.DataFrame) -> pd.Series:
    """'Toy Story (1995)', 'Animation|Comedy' -> 'Animation Comedy d1990'."""
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    decade = (year // 10 * 10).fillna(0).astype(int).astype(str)
    genres = movies["genres"].str.replace("-", "", regex=False).str.replace("|", " ", regex=False)
    return genres + " d" + decade


class ContentBased(Recommender):
    name = "ContentBased"

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "ContentBased":
        df = ds.train if df is None else df
        self.vectorizer_ = TfidfVectorizer(token_pattern=r"[^ ]+")
        self.item_vecs_ = normalize(self.vectorizer_.fit_transform(item_documents(ds.movies))).toarray()
        liked = ds.matrix(df[df["rating"] >= self.like_threshold], binary=True)
        # người chưa thích phim nào: dùng toàn bộ lịch sử xem để có hồ sơ
        empty = np.asarray(liked.sum(axis=1)).ravel() == 0
        if empty.any():
            allx = ds.matrix(df, binary=True).tolil()
            liked = liked.tolil()
            liked[empty] = allx[empty]
            liked = liked.tocsr()
        self.profiles_ = normalize(liked @ self.item_vecs_).astype(np.float32)
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.profiles_[users] @ self.item_vecs_.T.astype(np.float32)

    def similar_items(self, item: int, n: int = 10) -> np.ndarray:
        sims = self.item_vecs_ @ self.item_vecs_[item]
        sims[item] = -np.inf
        return np.argsort(-sims)[:n]
