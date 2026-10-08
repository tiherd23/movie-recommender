"""Gợi ý dựa trên nội dung (content-based).

Mỗi phim -> một vector nội dung, ghép từ 4 khối TF-IDF:
    1. thể loại + thập niên phát hành     (luôn có, từ MovieLens)
    2. từ khóa (keywords)                  (TMDB)
    3. diễn viên chính + đạo diễn          (TMDB)
    4. mô tả nội dung phim (overview)      (TMDB)
Nếu có dữ liệu TMDB, vector ghép (hàng chục nghìn chiều, thưa) được nén xuống
n_components chiều bằng TruncatedSVD (kỹ thuật LSA) để gọn và bắt được từ đồng nghĩa.
Hồ sơ người dùng = tổng vector các phim họ thích; điểm gợi ý = cosine(hồ sơ, phim).
Ưu điểm: gợi ý được phim mới chưa ai chấm (cold-start phía phim).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
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


def _as_tokens(col: pd.Series) -> pd.Series:
    """'Tom Hanks|Tim Allen' -> 'tom_hanks tim_allen': mỗi tên/từ khóa là một token nguyên vẹn."""
    return col.fillna("").str.lower().str.replace(r"[^a-z0-9|]+", "_", regex=True).str.replace("|", " ", regex=False)


class ContentBased(Recommender):
    name = "ContentBased"

    def __init__(self, like_threshold: float = 4.0, use_tmdb: bool = True, n_components: int = 64,
                 block_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.7, 1.0), seed: int = 42):
        self.like_threshold, self.use_tmdb = like_threshold, use_tmdb
        self.n_components, self.block_weights, self.seed = n_components, block_weights, seed

    # ---------- dựng vector phim ----------
    def _fit_items(self, movies: pd.DataFrame) -> None:
        self.genre_vec_ = TfidfVectorizer(token_pattern=r"[^ ]+")
        blocks = [normalize(self.genre_vec_.fit_transform(item_documents(movies)))]
        self.has_tmdb_ = self.use_tmdb and "overview" in movies.columns
        if self.has_tmdb_:
            people = _as_tokens(movies["cast"]) + " " + _as_tokens(movies["director"])
            texts = [
                (_as_tokens(movies["keywords"]), TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2)),
                (people, TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2)),
                (movies["overview"].fillna(""), TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)),
            ]
            for text, vec in texts:
                blocks.append(normalize(vec.fit_transform(text)))
        weights = self.block_weights[: len(blocks)]
        self.block_dims_ = [b.shape[1] for b in blocks]
        X = sparse.hstack([w * b for w, b in zip(weights, blocks)]).tocsr()
        if self.has_tmdb_:
            self.svd_ = TruncatedSVD(min(self.n_components, X.shape[1] - 1), random_state=self.seed)
            self.item_vecs_ = normalize(self.svd_.fit_transform(X)).astype(np.float32)
        else:
            self.svd_ = None
            self.item_vecs_ = normalize(X).toarray().astype(np.float32)

    def genre_vector(self, genres: list[str]) -> np.ndarray:
        """Vector nội dung của một danh sách thể loại (dùng cho người mới chỉ chọn thể loại)."""
        if not genres:
            return np.zeros(self.item_vecs_.shape[1], dtype=np.float32)
        g = normalize(self.genre_vec_.transform([" ".join(x.replace("-", "") for x in genres)]))
        g = self.block_weights[0] * g
        if self.svd_ is None:
            return np.asarray(g.todense()).ravel().astype(np.float32)
        pad = sparse.csr_matrix((1, sum(self.block_dims_[1:])))
        return self.svd_.transform(sparse.hstack([g, pad]).tocsr()).ravel().astype(np.float32)

    # ---------- giao diện Recommender ----------
    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "ContentBased":
        df = ds.train if df is None else df
        self._fit_items(ds.movies)
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
        return self.profiles_[users] @ self.item_vecs_.T

    def similar_items(self, item: int, n: int = 10) -> np.ndarray:
        sims = self.item_vecs_ @ self.item_vecs_[item]
        sims[item] = -np.inf
        return np.argsort(-sims)[:n]
