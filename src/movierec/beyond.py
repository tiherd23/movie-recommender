"""Độ đo ngoài độ chính xác (beyond-accuracy) và xếp hạng lại để tăng độ đa dạng (MMR).

Các độ đo, tính trên danh sách top-K của mọi người dùng:
- ILD (intra-list diversity): trung bình độ khác nhau (1 - cosine của vector thể loại) giữa các cặp phim
  trong cùng một danh sách. 0 = mọi phim cùng thể loại, càng cao càng đa dạng.
- Genres: số thể loại khác nhau trung bình trong một danh sách.
- Novelty: trung bình -log2(p_i), p_i = tỉ lệ người dùng đã chấm phim i trong dữ liệu huấn luyện.
  Phim càng ít người biết thì càng "mới" với người dùng.
- ARP (average recommendation popularity): số lượt chấm trung bình của các phim được gợi ý.
- LongTail: tỉ lệ vị trí gợi ý dành cho phim ngoài nhóm phổ biến nhất (nhóm phim chiếm 80% lượt chấm).
- Gini: độ tập trung của số lần mỗi phim được gợi ý trên toàn hệ thống (0 = đều, 1 = dồn vào một phim).

MMR (Carbonell & Goldstein, 1998): chọn lần lượt từng phim, mỗi bước lấy phim có
lam * độ liên quan - (1 - lam) * độ giống lớn nhất với các phim đã chọn. lam = 1 giữ nguyên thứ tự gốc.
Tùy chọn beta > 0 trừ thêm beta * độ phổ biến (log số lượt chấm, chuẩn hóa trong nhóm ứng viên) để đẩy phim ít
người biết lên, nhắm vào thiên lệch độ phổ biến thay vì độ đa dạng thể loại.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def genre_vectors(genres: pd.Series) -> np.ndarray:
    """Vector thể loại (one-hot nhiều nhãn) đã chuẩn hóa độ dài 1, mỗi dòng một phim."""
    names = sorted({g for gs in genres for g in gs.split("|") if g})
    col = {g: n for n, g in enumerate(names)}
    vecs = np.zeros((len(genres), len(names)), dtype=np.float32)
    for row, gs in enumerate(genres):
        for g in gs.split("|"):
            if g:
                vecs[row, col[g]] = 1.0
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norm > 0, norm, 1.0)


def head_items(pop: np.ndarray, share: float = 0.8) -> np.ndarray:
    """Nhóm phim phổ biến nhất chiếm `share` tổng số lượt chấm (mảng bool theo phim)."""
    order = np.argsort(-pop, kind="stable")
    cum = np.cumsum(pop[order]) / max(pop.sum(), 1)
    head = np.zeros(len(pop), dtype=bool)
    head[order[: int(np.searchsorted(cum, share)) + 1]] = True
    return head


def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def beyond_metrics(topk: np.ndarray, pop: np.ndarray, n_users: int, gvec: np.ndarray,
                   users: np.ndarray | None = None) -> dict[str, float]:
    """topk: (n_users, K) chỉ số phim; pop[i]: số người dùng đã chấm phim i trong dữ liệu huấn luyện.
    users: chỉ tính trên các người dùng này (mặc định: tất cả)."""
    lists = topk if users is None else topk[users]
    k = lists.shape[1]
    v = gvec[lists]                                    # (U, K, G)
    sim = np.einsum("ukg,ulg->ukl", v, v)              # cosine giữa các cặp phim trong danh sách
    off = ~np.eye(k, dtype=bool)
    ild = (1.0 - sim[:, off]).mean(axis=1)
    n_genres = (v > 0).any(axis=1).sum(axis=1)
    prob = np.clip(pop, 1, None) / n_users
    head = head_items(pop)
    counts = np.bincount(lists.ravel(), minlength=len(pop))
    return {"ILD": float(ild.mean()), "Genres": float(n_genres.mean()),
            "Novelty": float((-np.log2(prob[lists])).mean()), "ARP": float(pop[lists].mean()),
            "LongTail": float((~head[lists]).mean()), "Gini": gini(counts)}


def _unit(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.ones_like(x)


def mmr_rerank(scores: np.ndarray, gvec: np.ndarray, lam: float, k: int = 10, n_cand: int = 50,
               pop: np.ndarray | None = None, beta: float = 0.0) -> np.ndarray:
    """Xếp hạng lại cho MỘT người dùng. scores: điểm mọi phim (-inf = loại). Trả về k chỉ số phim."""
    finite = np.flatnonzero(np.isfinite(scores))
    n_cand = min(n_cand, len(finite))
    cand = finite[np.argpartition(-scores[finite], n_cand - 1)[:n_cand]]
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    use_pop = pop is not None and beta > 0
    if (lam >= 1.0 and not use_pop) or len(cand) <= k:
        return cand[:k]
    rel = _unit(scores[cand].astype(np.float64))      # đưa về [0, 1] để cùng thang với độ giống
    if use_pop:
        rel = rel - beta * _unit(np.log1p(pop[cand].astype(np.float64)))
    if lam >= 1.0:
        return cand[np.argsort(-rel, kind="stable")[:k]]
    sim = gvec[cand] @ gvec[cand].T
    first = int(np.argmax(rel))                        # phim điểm (sau phạt độ phổ biến) cao nhất đứng đầu
    chosen = [first]
    max_sim = sim[first].copy()
    free = np.ones(len(cand), dtype=bool)
    free[first] = False
    for _ in range(k - 1):
        gain = np.where(free, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(gain))
        chosen.append(j)
        free[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return cand[chosen]


def recommend_mmr(model, seen, n_users: int, gvec: np.ndarray, lam: float, k: int = 10, n_cand: int = 50,
                  pop: np.ndarray | None = None, beta: float = 0.0, batch: int = 1024) -> np.ndarray:
    """Như evaluate.recommend_topk nhưng xếp hạng lại top-n_cand của mỗi người bằng MMR."""
    out = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users))
        scores = model.score_users(users).astype(np.float32, copy=True)
        scores[seen[users].nonzero()] = -np.inf
        for row, u in enumerate(users):
            out[u] = mmr_rerank(scores[row], gvec, lam, k, n_cand, pop, beta)
    return out
