"""Luật "bỏ phim giống phim bị chấm thấp" (RecommenderService._drop_disliked_neighbours) có làm giảm độ chính xác?

Huấn luyện kiến trúc hai tầng như bản web (không dùng giới tính, tuổi, nghề nghiệp), chấm trên tập test hai lần:
không luật và có luật, với lịch sử train+val của từng người làm hồ sơ (phim ≤ 2★ là không thích, ≥ 4★ là thích).
So NDCG@10 từng người bằng kiểm định Wilcoxon cặp đôi, cho toàn bộ người dùng và riêng nhóm bị luật tác động.
Chạy: python scripts/dislike_rule_eval.py   (CPU, khoảng 10-15 phút)   -> results/dislike_rule_eval.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.ranker import TwoStageRanker  # noqa: E402

K = 10


def ndcg_rows(top: np.ndarray, relevant: list[set[int]]) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    out = np.full(len(top), np.nan)
    for u, rel in enumerate(relevant):
        if rel:
            gain = np.array([j in rel for j in top[u]], dtype=float)
            out[u] = (gain * disc).sum() / disc[: min(len(rel), K)].sum()
    return out


def main() -> None:
    ds = build_dataset()
    best = json.loads((ROOT / "results" / "best_params.json").read_text(encoding="utf-8"))
    hist = pd.concat([ds.train, ds.val], ignore_index=True)
    ranker = TwoStageRanker(best["ImplicitALS"], best["ItemKNN"], use_demographics=False)
    ranker.fit_ranker(ds, hist=ds.train, label=ds.val)
    ranker.fit(ds, hist)
    sim = ranker.models_["knn"].sim_
    seen = ds.matrix(hist, binary=True)
    relevant = [set() for _ in range(ds.n_users)]
    t = ds.test[ds.test["rating"] >= 4]
    for u, i in zip(t["user"], t["item"]):
        relevant[u].add(int(i))
    by_user = {u: g for u, g in hist.groupby("user")}
    tops = {"không luật": np.zeros((ds.n_users, K), int), "có luật": np.zeros((ds.n_users, K), int)}
    touched = np.zeros(ds.n_users, bool)
    for start in range(0, ds.n_users, 1024):
        users = np.arange(start, min(start + 1024, ds.n_users))
        scores = ranker.score_users(users)
        scores[seen[users].nonzero()] = -np.inf
        scores[scores <= -1e8] = -np.inf  # ngoài ứng viên
        for r, u in enumerate(users):
            s = scores[r]
            tops["không luật"][u] = np.argsort(-s)[:K]
            g = by_user[u]
            bad = g.loc[g["rating"] <= 2, "item"].to_numpy()
            good = g.loc[g["rating"] >= 4, "item"].to_numpy()
            s2 = s.copy()
            if len(bad):
                near_bad = sim[bad].max(axis=0)
                near_good = sim[good].max(axis=0) if len(good) else np.zeros_like(near_bad)
                drop = (near_bad > near_good) & np.isfinite(s2)
                if np.isfinite(s2).sum() - drop.sum() >= K:
                    s2[drop] = -np.inf
            tops["có luật"][u] = np.argsort(-s2)[:K]
            touched[u] = not np.array_equal(tops["có luật"][u], tops["không luật"][u])
    a, b = ndcg_rows(tops["không luật"], relevant), ndcg_rows(tops["có luật"], relevant)
    rows = []
    for name, mask in [("Mọi người dùng", np.isfinite(a)), ("Người có danh sách bị luật thay đổi", np.isfinite(a) & touched)]:
        x, y = a[mask], b[mask]
        diff = y - x
        p = wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
        rows.append({"nhóm": name, "số người": int(mask.sum()), "NDCG@10 không luật": round(x.mean(), 4),
                     "NDCG@10 có luật": round(y.mean(), 4), "chênh lệch %": round(100 * (y.mean() / x.mean() - 1), 2),
                     "số người tốt hơn": int((diff > 0).sum()), "số người kém hơn": int((diff < 0).sum()),
                     "p (Wilcoxon)": f"{p:.2g}"})
    cov = {n: len(np.unique(v)) / ds.n_items for n, v in tops.items()}
    df = pd.DataFrame(rows)
    df["độ phủ không luật"], df["độ phủ có luật"] = round(cov["không luật"], 4), round(cov["có luật"], 4)
    df.to_csv(ROOT / "results" / "dislike_rule_eval.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
