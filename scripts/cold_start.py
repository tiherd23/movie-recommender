"""Thí nghiệm người dùng mới (cold-start): độ chính xác theo số phim đã thích, để chọn ngưỡng chuyển sang hai tầng.

Mô phỏng đúng tình huống trên web:
  1. Chọn ngẫu nhiên N người dùng MovieLens làm "người dùng mới" và LOẠI HẲN họ khỏi dữ liệu huấn luyện.
  2. Huấn luyện lại toàn bộ dịch vụ web (RecommenderService.build, như scripts/build_service.py) trên những người
     còn lại, nên mô hình chưa từng thấy bất kỳ lượt chấm nào của người dùng mới.
  3. Mỗi người dùng mới "đăng ký" với n phim thích (n = 1, 2, 3, 5, 7, 10, 15, 20, 30) lấy ngẫu nhiên trong các phim họ chấm
     4-5★ ở giai đoạn train + val; n lớn hơn chứa n nhỏ hơn (cùng một thứ tự ngẫu nhiên).
  4. Mỗi phương pháp xếp toàn bộ kho phim (bỏ các phim họ đã chấm ở train + val), lấy top 10, so với các phim họ
     chấm 4-5★ trong tập test (10% lượt chấm cuối theo thời gian): NDCG@10, Recall@10, khoảng tin cậy bootstrap 95%.
Phương pháp: Phổ biến nhất; Implicit ALS (fold-in); Công thức trộn (web dùng khi < 5 phim thích); Hai tầng (web dùng
khi ≥ 5 phim thích). Không dùng độ đa dạng, không chọn thể loại.
Chạy: python scripts/cold_start.py [--users 1000]   (CPU, vài phút)   -> results/cold_start.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.profile_ranker import fit_profile_ranker  # noqa: E402
from movierec.service import RecommenderService  # noqa: E402

K = 10
NS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
NS_RATED = [5, 10, 20, 30, 50]
FULL = 10_000  # mã của "toàn bộ lịch sử"
DISC = 1.0 / np.log2(np.arange(2, K + 2))


def metrics(top: np.ndarray, rel: set[int]) -> tuple[float, float]:
    hit = np.array([j in rel for j in top], dtype=float)
    return float((hit * DISC).sum() / DISC[: min(len(rel), K)].sum()), float(hit.sum() / min(len(rel), K))


def ci(x: np.ndarray, rng, b: int = 2000) -> tuple[float, float]:
    idx = rng.integers(0, len(x), (b, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rated", action="store_true",
                    help="hồ sơ = n phim đã chấm bất kỳ với số sao thật (gồm cả phim không thích), thay vì n phim thích")
    ap.add_argument("--schedules", nargs="*", default=[],
                    help="trọng số phổ biến giảm theo số phim thích: w(n) = c * n^-p, mỗi mục dạng c,p")
    ap.add_argument("--exclude", choices=["history", "profile"], default="history",
                    help="history: bỏ mọi phim người đó đã chấm ở train+val khỏi danh sách; profile: chỉ bỏ n phim trong "
                         "hồ sơ (đúng như web, phim đã xem ngoài hồ sơ vẫn được xếp và tính là không liên quan)")
    ap.add_argument("--no-profile-ranker", action="store_true", help="bỏ qua bộ xếp hạng học hồ sơ ngắn (chạy nhanh)")
    ap.add_argument("--min-liked", type=int, default=0, help="ngưỡng phim thích để chọn người dùng (mặc định max n)")
    ap.add_argument("--short", type=int, default=1, help="số hồ sơ ngắn mỗi người khi huấn luyện bộ xếp hạng mới")
    ap.add_argument("--pop-weights", type=float, nargs="*", default=[],
                    help="thử thêm công thức trộn với trọng số độ phổ biến khác (mặc định của web: trong best_params)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ds = build_dataset()
    hist = pd.concat([ds.train, ds.val], ignore_index=True)
    liked_hist = hist[hist["rating"] >= 4].groupby("user")["item"].apply(list)
    test_rel = ds.test[ds.test["rating"] >= 4].groupby("user")["item"].apply(set)
    # người dùng mới: có ít nhất max(NS) phim thích trước đó (để thử đủ mọi n) và ít nhất 1 phim thích trong test
    rated_hist = hist.groupby("user")[["item", "rating"]].apply(lambda g: list(zip(g["item"], g["rating"])))
    need = args.min_liked or max(NS)
    pool = [u for u in test_rel.index if len(liked_hist.get(u, [])) >= need]
    new_users = rng.choice(pool, min(args.users, len(pool)), replace=False)
    keep = lambda df: df[~df["user"].isin(set(new_users))].reset_index(drop=True)  # noqa: E731
    reduced = dataclasses.replace(ds, train=keep(ds.train), val=keep(ds.val), test=keep(ds.test))
    best = json.loads((ROOT / "results" / "best_params.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    svc = RecommenderService.build(reduced, best, with_ranker=True, profile_ranker=False)
    print(f"Huấn luyện lại dịch vụ web không có {len(new_users)} người dùng mới: {time.perf_counter() - t0:.0f}s")
    # bộ xếp hạng học từ hồ sơ mô phỏng (gồm hồ sơ ngắn), cùng tầng 1 và thống kê phim với svc
    rankers = {"Hai tầng": svc.ranker}
    if not args.no_profile_ranker:
        t0 = time.perf_counter()
        profile_booster, info, _ = fit_profile_ranker(reduced, best["ImplicitALS"], best["ItemKNN"], svc.weights,
                                                       short=args.short)
        print(f"Bộ xếp hạng theo hồ sơ: {info} ({time.perf_counter() - t0:.0f}s)")
        rankers["Hai tầng (học cả hồ sơ ngắn)"] = {**svc.ranker, "booster": profile_booster}
    top_pop = set(np.argsort(-np.asarray(svc.pop))[:K].tolist())

    seen_by_user = hist.groupby("user")["item"].apply(list)
    rows = []
    for u in new_users:
        order = rng.permutation(liked_hist[u])
        order_rated = [rated_hist[u][i] for i in rng.permutation(len(rated_hist[u]))]
        seen = seen_by_user[u]
        rel = test_rel[u]
        for n in [m for m in (NS_RATED if args.rated else NS) if m <= len(order)] + [FULL]:
            if n == FULL:  # toàn bộ lịch sử train + val, như người dùng MovieLens trong thực nghiệm chính
                ratings = {int(i): int(r) for i, r in rated_hist[u]}
                liked = [i for i, r in ratings.items() if r >= 4]
            elif args.rated:
                ratings = {int(i): int(r) for i, r in order_rated[:n]}
                liked = [i for i, r in ratings.items() if r >= 4]
                if not liked:
                    continue
            else:
                liked = [int(i) for i in order[:n]]
                ratings = {i: 5 for i in liked}
            als = svc.Y @ svc._fold_in(liked)
            knn_part = svc.sim[liked].toarray()
            scores = {
                "Phổ biến nhất": np.asarray(svc.pop, dtype=np.float64).copy(),
                "Implicit ALS": als.astype(np.float64),
                "Công thức trộn": svc._blend_score(liked, [], als, knn_part).astype(np.float64),
            }
            for name, rk in rankers.items():
                svc.ranker = rk
                scores[name] = svc._rank_two_stage(ratings, liked, als)[0].astype(np.float64)
            for w in args.pop_weights:  # biến thể: đổi trọng số độ phổ biến của công thức trộn
                parts = svc._blend_parts(liked, [], als, knn_part)
                parts["pop"] = parts["pop"] / svc.weights[3] * w
                scores[f"Trộn, phổ biến {w:g}"] = sum(parts.values()).astype(np.float64)
            for sched in args.schedules:
                c, pw = map(float, sched.split(","))
                parts = svc._blend_parts(liked, [], als, knn_part)
                parts["pop"] = parts["pop"] / svc.weights[3] * c * len(liked) ** -pw
                scores[f"Trộn, w(n)={c:g}·n^-{pw:g}"] = sum(parts.values()).astype(np.float64)
            for name, s in scores.items():
                s[seen if args.exclude == "history" or n == FULL else list(ratings)] = -np.inf
                top = np.argpartition(-s, K)[:K]
                top = top[np.argsort(-s[top])]
                nd, rc = metrics(top, rel)  # phim test không nằm trong train+val nên không bị loại ở cả hai giao thức
                rows.append({"user": int(u), "n": n, "model": name, "ndcg": nd, "recall": rc,
                             "overlap_pop": len(top_pop & set(top.tolist())) / K, "top": top.tolist()})
    df = pd.DataFrame(rows)
    out = []
    for (n, name), g in df.groupby(["n", "model"], sort=False):
        lo, hi = ci(g["ndcg"].to_numpy(), rng)
        cov = len({j for t in g["top"] for j in t}) / len(svc.pop)
        out.append({"số phim thích": n, "mô hình": name, "NDCG@10": round(g["ndcg"].mean(), 4),
                    "CI95 thấp": round(lo, 4), "CI95 cao": round(hi, 4), "Recall@10": round(g["recall"].mean(), 4),
                    "trùng top phổ biến": round(g["overlap_pop"].mean(), 3), "độ phủ": round(cov, 4),
                    "số người": len(g)})
    res = pd.DataFrame(out)
    tag = ("" if args.seed == 0 and not args.pop_weights else f"_seed{args.seed}" + ("_pop" if args.pop_weights else "")) \
        + ("_rated" if args.rated else "") + ("_excl-profile" if args.exclude == "profile" else "") \
        + (f"_min{args.min_liked}" if args.min_liked else "")
    res.to_csv(ROOT / "results" / f"cold_start{tag}.csv", index=False)
    df.drop(columns="top").to_csv(ROOT / "results" / f"cold_start_users{tag}.csv", index=False)
    table = res.pivot(index="số phim thích", columns="mô hình", values="NDCG@10")
    print(table.to_string())
    # so cặp Hai tầng - Công thức trộn ở từng n (cùng người dùng): chênh lệch và khoảng tin cậy
    print("Trung bình trên các mức n (không tính đầy đủ):")
    print(res[res["số phim thích"] != FULL].groupby("mô hình")["NDCG@10"].mean().sort_values(ascending=False).round(4)
          .to_string())
    for n in ([] if args.no_profile_ranker else sorted(df.n.unique())):
        a = df[(df.n == n) & (df.model == "Hai tầng (học cả hồ sơ ngắn)")].set_index("user")["ndcg"]
        b = df[(df.n == n) & (df.model == "Công thức trộn")].set_index("user")["ndcg"].loc[a.index]
        d = (a - b).to_numpy()
        lo, hi = ci(d, rng)
        label = "đầy đủ" if n == FULL else f"{n:2d}"
        print(f"n={label}: Hai tầng (hồ sơ ngắn) - Công thức trộn = {d.mean():+.4f} [{lo:+.4f}; {hi:+.4f}]")
    print("-> results/cold_start.csv, results/cold_start_users.csv")


if __name__ == "__main__":
    main()
