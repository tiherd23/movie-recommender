"""Kiến trúc hai tầng (two-stage): truy hồi ứng viên rồi xếp hạng lại bằng LightGBM.

Tầng 1 - truy hồi (candidate generation):
    Mỗi mô hình (ALS, ItemKNN, Popularity, Content) đề cử top-N phim cho người dùng.
    Gộp lại được khoảng 150-250 ứng viên / người, thay vì phải xét cả 3.700 phim.
Tầng 2 - xếp hạng (ranking):
    Với mỗi cặp (người dùng, phim ứng viên), tạo vector đặc trưng gồm điểm và thứ hạng
    từ từng mô hình, thống kê của người dùng, thống kê của phim, độ hợp thể loại.
    LightGBM (LambdaRank) học cách kết hợp các đặc trưng đó để đưa phim người dùng
    thật sự thích lên đầu. Đây là "hybrid có huấn luyện", thay cho cộng trọng số bằng tay.

Tránh rò rỉ dữ liệu: khi huấn luyện bộ xếp hạng, các mô hình tầng 1 chỉ được học trên
phần lịch sử (hist), còn nhãn lấy từ giai đoạn SAU đó (label).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .data import Dataset
from .models import ContentBased, ImplicitALS, ItemKNN, Popularity
from .models.base import Recommender

LIKE = 4.0


def _zrows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-8)


def candidate_features(scores: dict[str, np.ndarray], seen: np.ndarray, n_candidates: dict[str, int],
                       user_feats: np.ndarray, user_genre: np.ndarray,
                       item_feats: np.ndarray, item_genre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chọn ứng viên và dựng ma trận đặc trưng cho một lô người dùng.

    scores[name]: (B, n_items) điểm thô của từng mô hình tầng 1 (theo đúng thứ tự trong dict)
    seen        : (B, n_items) True nếu người dùng đã chấm phim đó
    user_feats  : (B, k) thống kê người dùng; user_genre: (B, số thể loại) hồ sơ thể loại
    Trả về (hàng trong lô, chỉ số phim ứng viên, X). Dùng chung cho đánh giá offline và web,
    để đặc trưng lúc phục vụ giống hệt lúc huấn luyện.
    """
    n_users, n_items = seen.shape
    cand = np.zeros((n_users, n_items), dtype=bool)
    sources = np.zeros((n_users, n_items), dtype=np.float32)
    rows_idx = np.arange(n_users)[:, None]
    per_model = []
    for name, s in scores.items():
        s = s.astype(np.float32)
        z = _zrows(s)
        order = np.argsort(-np.where(seen, -np.inf, s), axis=1, kind="stable")
        rank = np.empty_like(order)
        rank[rows_idx, order] = np.arange(n_items)[None, :]
        hit = np.zeros_like(cand)
        hit[rows_idx, order[:, : n_candidates[name]]] = True
        hit &= ~seen
        cand |= hit
        sources += hit
        per_model.append((z, np.log1p(rank).astype(np.float32)))
    rows, items = np.nonzero(cand)
    cols = []
    for z, logrank in per_model:
        cols += [z[rows, items], logrank[rows, items]]
    cols.append(sources[rows, items])
    cols.append(np.einsum("ij,ij->i", user_genre[rows], item_genre[items]))
    X = np.column_stack(cols + [user_feats[rows], item_feats[items]]).astype(np.float32)
    return rows, items, X


class TwoStageRanker(Recommender):
    name = "TwoStage"

    def __init__(self, als_params: dict | None = None, knn_params: dict | None = None,
                 n_candidates: dict | None = None, lgb_params: dict | None = None, seed: int = 42,
                 occupation_categorical: bool = False, use_demographics: bool = True):
        self.occupation_categorical = occupation_categorical and use_demographics
        self.use_demographics = use_demographics
        self.als_params = als_params or {"factors": 32, "alpha": 1.0, "min_rating": 4.0}
        self.knn_params = knn_params or {"k": 100, "shrink": 0.0}
        self.n_candidates = n_candidates or {"als": 100, "knn": 100, "pop": 50, "content": 30}
        self.lgb_params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
                           "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 100,
                           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                           "lambda_l2": 1.0, "verbose": -1, "seed": seed, **(lgb_params or {})}
        self.seed = seed
        self.booster_ = None

    # ---------- tầng 1: các mô hình truy hồi + thống kê ----------
    def _fit_base(self, ds: Dataset, df: pd.DataFrame) -> None:
        self.n_items_ = ds.n_items
        self.models_ = {
            "als": ImplicitALS(**self.als_params).fit(ds, df),
            "knn": ItemKNN(**self.knn_params).fit(ds, df),
            "pop": Popularity().fit(ds, df),
            "content": ContentBased().fit(ds, df),
        }
        self.seen_ = ds.matrix(df, binary=True)
        u, i = df["user"].to_numpy(), df["item"].to_numpy()
        r = df["rating"].to_numpy(np.float64)
        liked = (r >= LIKE).astype(np.float64)
        n_u = np.bincount(u, minlength=ds.n_users)
        n_i = np.bincount(i, minlength=ds.n_items)
        mu = r.mean()
        self.user_feats_ = np.column_stack([
            np.log1p(n_u),
            np.bincount(u, r, ds.n_users) / np.maximum(n_u, 1),
            np.bincount(u, liked, ds.n_users) / np.maximum(n_u, 1),
        ] + ([
            (ds.users["gender"].to_numpy() == "M").astype(float),
            ds.users["age"].to_numpy(float),
            ds.users["occupation"].to_numpy(float),
        ] if self.use_demographics else [])).astype(np.float32)
        year = ds.movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float).fillna(0).to_numpy()
        self.genre_names_ = sorted({g for gs in ds.movies["genres"] for g in gs.split("|")})
        G = np.array([[g in gs.split("|") for g in self.genre_names_] for gs in ds.movies["genres"]], dtype=np.float32)
        self.item_feats_ = np.column_stack([
            np.log1p(n_i),
            (np.bincount(i, r, ds.n_items) + 10 * mu) / (n_i + 10),  # điểm trung bình có co về mu
            np.bincount(i, liked, ds.n_items) / np.maximum(n_i, 1),
            year,
            G,
        ]).astype(np.float32)
        # hồ sơ thể loại của người dùng: tỉ lệ phim đã thích thuộc từng thể loại
        like_mat = ds.matrix(df[df["rating"] >= LIKE], binary=True)
        prof = like_mat @ G
        self.user_genre_ = (prof / np.maximum(prof.sum(axis=1, keepdims=True), 1)).astype(np.float32)
        self.item_genre_ = G / np.maximum(G.sum(axis=1, keepdims=True), 1)
        self.feature_names_ = (
            [f"{m}_{k}" for m in self.models_ for k in ("z", "logrank")]
            + ["n_sources", "genre_affinity"]
            + ["u_log_n", "u_mean", "u_like_frac"]
            + (["u_male", "u_age", "u_occupation"] if self.use_demographics else [])
            + ["i_log_n", "i_mean", "i_like_frac", "i_year"] + [f"g_{g}" for g in self.genre_names_]
        )

    def _features(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trả về (chỉ số hàng trong lô, chỉ số phim ứng viên, ma trận đặc trưng)."""
        scores = {name: model.score_users(users) for name, model in self.models_.items()}
        seen = self.seen_[users].toarray().astype(bool)
        return candidate_features(scores, seen, self.n_candidates, self.user_feats_[users],
                                  self.user_genre_[users], self.item_feats_, self.item_genre_)

    # ---------- tầng 2: huấn luyện bộ xếp hạng ----------
    def fit_ranker(self, ds: Dataset, hist: pd.DataFrame, label: pd.DataFrame,
                   num_rounds: int = 600, holdout: float = 0.15, batch: int = 512) -> dict:
        """Mô hình tầng 1 học trên `hist`; nhãn = phim được chấm >= 4 sao trong `label`."""
        import lightgbm as lgb

        self._fit_base(ds, hist)
        liked = label[label["rating"] >= LIKE]
        Y = sparse.csr_matrix((np.ones(len(liked), np.float32), (liked["user"], liked["item"])),
                              shape=(ds.n_users, ds.n_items))
        Xs, ys, groups, group_users, recalls = [], [], [], [], []
        for start in range(0, ds.n_users, batch):
            users = np.arange(start, min(start + batch, ds.n_users))
            rows, items, X = self._features(users)
            y = np.asarray(Y[users][rows, items]).ravel()
            counts = np.bincount(rows, minlength=len(users))
            pos = np.bincount(rows, y, minlength=len(users))
            n_liked = np.asarray(Y[users].sum(axis=1)).ravel()
            recalls.append(pos[n_liked > 0] / n_liked[n_liked > 0])
            keep_user = pos > 0  # người dùng không có ứng viên đúng nào thì không giúp gì cho việc học
            keep = keep_user[rows]
            Xs.append(X[keep]); ys.append(y[keep])
            groups.append(counts[keep_user]); group_users.append(users[keep_user])
        X, y = np.vstack(Xs), np.concatenate(ys)
        groups, group_users = np.concatenate(groups), np.concatenate(group_users)

        # chia theo NGƯỜI DÙNG để dừng sớm: 85% học, 15% kiểm tra
        rng = np.random.default_rng(self.seed)
        is_val = rng.random(len(groups)) < holdout
        row_is_val = np.repeat(is_val, groups)
        cat = [self.feature_names_.index("u_occupation")] if self.occupation_categorical else []
        dtrain = lgb.Dataset(X[~row_is_val], y[~row_is_val], group=groups[~is_val],
                             feature_name=self.feature_names_, categorical_feature=cat)
        dval = lgb.Dataset(X[row_is_val], y[row_is_val], group=groups[is_val], reference=dtrain)
        self.booster_ = lgb.train(self.lgb_params, dtrain, num_rounds, valid_sets=[dval],
                                  callbacks=[lgb.early_stopping(40, verbose=False)])
        gain = self.booster_.feature_importance("gain")
        self.importance_ = pd.Series(gain / gain.sum(), index=self.feature_names_).sort_values(ascending=False)
        return {"rows": int(len(y)), "users": int(len(groups)), "positives": int(y.sum()),
                "avg_candidates": float(groups.mean()), "candidate_recall": float(np.concatenate(recalls).mean()),
                "best_iteration": int(self.booster_.best_iteration),
                "holdout_ndcg@10": float(self.booster_.best_score["valid_0"]["ndcg@10"])}

    # ---------- giao diện Recommender (dùng khi đánh giá / phục vụ) ----------
    def fit(self, ds: Dataset, df: pd.DataFrame | None = None) -> "TwoStageRanker":
        """Huấn luyện lại tầng 1 trên df; bộ xếp hạng (booster) đã học giữ nguyên."""
        if self.booster_ is None:
            raise RuntimeError("Chưa huấn luyện bộ xếp hạng. Gọi fit_ranker() trước.")
        self._fit_base(ds, ds.train if df is None else df)
        return self

    def score_users(self, users: np.ndarray) -> np.ndarray:
        rows, items, X = self._features(users)
        out = np.full((len(users), self.n_items_), -1e9, dtype=np.float32)  # phim ngoài ứng viên: loại
        out[rows, items] = self.booster_.predict(X, num_iteration=self.booster_.best_iteration)
        return out
