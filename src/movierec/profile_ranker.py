"""Huấn luyện bộ xếp hạng hai tầng cho web bằng hồ sơ mô phỏng, gồm cả hồ sơ ngắn của người dùng mới.

Vấn đề (scripts/cold_start.py): bộ xếp hạng gốc học từ người dùng MovieLens có lịch sử đầy đủ (trung bình hơn 100 lượt
chấm), với điểm tầng 1 lấy từ vector người dùng ALS đã học. Trên web, người dùng mới chỉ có vài lượt chấm và điểm ALS
lấy bằng fold-in, nên đặc trưng (số phim đã chấm, điểm trung bình, điểm z của các mô hình...) lệch khỏi phân phối lúc
huấn luyện, và bộ xếp hạng kém hơn cả công thức trộn.

Cách làm: dựng dữ liệu huấn luyện bằng chính đường tính của web (RecommenderService.profile_features):
  - mô hình tầng 1 học trên `hist` (train), nhãn = phim chấm >= 4 sao trong `label` (val), như bộ xếp hạng gốc;
  - mỗi người dùng đóng góp hồ sơ đầy đủ (mọi lượt chấm trong train) và `short` hồ sơ ngắn, mỗi hồ sơ gồm m lượt chấm
    lấy ngẫu nhiên trong train (m chọn đều trong SHORT_SIZES, giữ nguyên số sao thật nên có cả phim không thích);
  - ứng viên loại toàn bộ phim người đó đã chấm trong train (phim chưa nằm trong hồ sơ cũng không phải phim "mới").
Tham số LightGBM, số ứng viên và cách dừng sớm giữ như TwoStageRanker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

SHORT_SIZES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
LIKE = 4


def build_groups(svc, hist: pd.DataFrame, label: pd.DataFrame, rng: np.random.Generator,
                 short: int = 1, full: bool = True, users: np.ndarray | None = None):
    """Trả về X, y, kích thước nhóm, người dùng của nhóm. svc: RecommenderService có mô hình tầng 1 học trên hist."""
    pos = label[label["rating"] >= LIKE].groupby("user")["item"].apply(set)
    by_user = {u: (g["item"].to_numpy(), g["rating"].to_numpy()) for u, g in hist.groupby("user")}
    users = pos.index.to_numpy() if users is None else users
    Xs, ys, groups, owners = [], [], [], []
    for u in users:
        if u not in by_user or u not in pos.index:
            continue
        items, ratings = by_user[u]
        profiles = []
        if full:
            profiles.append(np.arange(len(items)))
        for _ in range(short):  # short: số hồ sơ ngắn mỗi người dùng
            m = min(int(rng.choice(SHORT_SIZES)), len(items))
            profiles.append(rng.choice(len(items), m, replace=False))
        for idx in profiles:
            prof = {int(i): int(r) for i, r in zip(items[idx], ratings[idx])}
            liked = [i for i, r in prof.items() if r >= LIKE]
            if not liked:
                continue
            als = svc.Y @ svc._fold_in(liked)
            cand, X = svc.profile_features(prof, liked, als, seen_items=items)
            y = np.fromiter((j in pos[u] for j in cand), dtype=np.float32, count=len(cand))
            if y.sum() == 0:  # không có ứng viên đúng nào thì nhóm này không giúp gì cho việc học
                continue
            Xs.append(X); ys.append(y); groups.append(len(cand)); owners.append(u)
    return np.vstack(Xs), np.concatenate(ys), np.array(groups), np.array(owners)


def train_booster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, owners: np.ndarray, lgb_params: dict,
                  feature_names: list[str], seed: int = 42, holdout: float = 0.15, num_rounds: int = 600):
    """LambdaRank, dừng sớm trên 15% NGƯỜI DÙNG giữ lại (mọi nhóm của một người nằm cùng một phía)."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    uniq = np.unique(owners)
    val_users = set(uniq[rng.random(len(uniq)) < holdout].tolist())
    is_val = np.array([o in val_users for o in owners])
    row_is_val = np.repeat(is_val, groups)
    dtrain = lgb.Dataset(X[~row_is_val], y[~row_is_val], group=groups[~is_val], feature_name=feature_names)
    dval = lgb.Dataset(X[row_is_val], y[row_is_val], group=groups[is_val], reference=dtrain)
    booster = lgb.train(lgb_params, dtrain, num_rounds, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    info = {"rows": int(len(y)), "groups": int(len(groups)), "best_iteration": int(booster.best_iteration),
            "holdout_ndcg@10": float(booster.best_score["valid_0"]["ndcg@10"])}
    return booster, info


def fit_profile_ranker(ds: Dataset, als_p: dict, knn_p: dict, weights: list[float], seed: int = 42,
                       short: int = 3, full: bool = True):
    """Học tầng 1 trên ds.train, tạo nhóm hồ sơ (đầy đủ + ngắn) với nhãn ds.val, huấn luyện LightGBM.
    Trả về (booster, thông tin, TwoStageRanker chứa tầng 1 và thống kê phim)."""
    from scipy import sparse

    from .ranker import TwoStageRanker
    from .service import RecommenderService

    two = TwoStageRanker(als_p, knn_p, use_demographics=False, seed=seed)
    two._fit_base(ds, ds.train)
    m = two.models_
    tmp = RecommenderService(ds.movies, m["als"].Y_, sparse.csr_matrix(m["knn"].sim_), m["content"],
                             m["pop"].scores_, m["als"].reg, m["als"].alpha, weights,
                             {"booster": None, "n_candidates": two.n_candidates, "item_feats": two.item_feats_,
                              "item_genre": two.item_genre_})
    X, y, groups, owners = build_groups(tmp, ds.train, ds.val, np.random.default_rng(seed), short=short, full=full)
    booster, info = train_booster(X, y, groups, owners, two.lgb_params, two.feature_names_, seed=seed)
    return booster, info, two
