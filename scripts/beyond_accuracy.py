"""Đo độ đa dạng, độ mới và thiên lệch độ phổ biến của 11 mô hình, rồi thử xếp hạng lại bằng MMR.

Chạy: python scripts/beyond_accuracy.py                 # cả hai phần
      python scripts/beyond_accuracy.py --reuse         # dùng lại danh sách top-10 đã lưu, không huấn luyện lại
      python scripts/beyond_accuracy.py --skip LightGCN # bỏ qua mô hình chậm (LightGCN ~40 phút trên GPU hạ xung)
Phần 1: mỗi mô hình học lại trên train+val với cấu hình tốt nhất, lấy top-10 trên tập test
        -> artifacts/topk/<mô hình>.npy và results/beyond.csv
Phần 2: xếp hạng lại 50 ứng viên đầu của TwoStage và Implicit ALS bằng MMR với nhiều mức lam -> results/mmr.csv
Phần 3: xếp hạng lại 50 ứng viên đầu của TwoStage, trừ điểm theo độ phổ biến với nhiều mức beta -> results/pop_penalty.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.beyond import beyond_metrics, genre_vectors, recommend_mmr  # noqa: E402
from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import recommend_topk  # noqa: E402
from movierec.metrics import ranking_metrics  # noqa: E402

ORDER = ["Popularity", "BiasBaseline", "ContentBased", "ItemKNN", "SVD", "ImplicitALS", "Hybrid",
         "TwoStage", "NeuMF", "LightGCN", "SASRec"]
LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


def best_epoch(csv: str) -> int:
    return int(pd.read_csv(ROOT / "results" / csv)["best_epoch"].iloc[0])


class Builder:
    """Huấn luyện từng mô hình đúng như các tập lệnh run_*.py (cấu hình tốt nhất, học trên train+val)."""

    def __init__(self, ds):
        self.ds, self.full = ds, pd.concat([ds.train, ds.val], ignore_index=True)
        self.best = json.loads((ROOT / "results" / "best_params.json").read_text())
        self.cache: dict = {}

    def get(self, name: str):
        if name not in self.cache:
            self.cache[name] = self._build(name)
        return self.cache[name]

    def _build(self, name: str):
        from movierec.models import SVD, BiasBaseline, ContentBased, Hybrid, ImplicitALS, ItemKNN, Popularity

        ds, full, best = self.ds, self.full, self.best
        simple = {"Popularity": Popularity, "BiasBaseline": BiasBaseline, "ContentBased": ContentBased}
        if name in simple:
            return simple[name]().fit(ds, full)
        if name == "ItemKNN":
            return ItemKNN(**best["ItemKNN"]).fit(ds, full)
        if name == "SVD":
            return SVD(**best["SVD"]).fit(ds, full)
        if name == "ImplicitALS":
            return ImplicitALS(**best["ImplicitALS"]).fit(ds, full)
        if name == "Hybrid":
            parts = [self.get(n) for n in ("ImplicitALS", "ItemKNN", "ContentBased", "Popularity")]
            return Hybrid(parts, best["Hybrid"]["weights"], prefit=True)
        if name == "TwoStage":
            from movierec.ranker import TwoStageRanker
            keep = pickle.loads((ROOT / "artifacts" / "ranker.pkl").read_bytes())
            ranker = TwoStageRanker(keep["als_params"], keep["knn_params"], keep["n_candidates"],
                                    keep["lgb_params"], keep["seed"])
            ranker.booster_, ranker.importance_ = keep["booster_"], keep["importance_"]
            return ranker.fit(ds, full)
        if name == "NeuMF":
            from movierec.models.ncf import NeuMF
            return NeuMF(epochs=best_epoch("ncf.csv")).fit(ds, full)
        if name == "LightGCN":
            from movierec.models.lightgcn import LightGCN
            return LightGCN(epochs=best_epoch("lightgcn.csv")).fit(ds, full, verbose=True)
        if name == "SASRec":
            from movierec.models.sasrec import SASRec
            return SASRec.load(ROOT / "artifacts" / "sasrec.pt")
        raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="dùng lại artifacts/topk/*.npy nếu đã có")
    ap.add_argument("--skip", nargs="*", default=[], help="tên mô hình bỏ qua")
    ap.add_argument("--mmr-models", nargs="*", default=["TwoStage", "ImplicitALS"])
    ap.add_argument("--n-cand", type=int, default=50)
    ap.add_argument("--pop-models", nargs="*", default=["TwoStage"])
    args = ap.parse_args()

    ds = build_dataset()
    builder = Builder(ds)
    seen = ds.matrix(builder.full, binary=True)
    pop = np.asarray(seen.sum(axis=0)).ravel()            # số người đã chấm mỗi phim (train+val)
    gvec = genre_vectors(ds.movies["genres"].reset_index(drop=True))
    relevant: list[set[int]] = [set() for _ in range(ds.n_users)]
    liked = ds.test[ds.test["rating"] >= 4.0]
    for u, i in zip(liked["user"].to_numpy(), liked["item"].to_numpy()):
        relevant[u].add(int(i))
    users = np.array([u for u, r in enumerate(relevant) if r])  # cùng tập người dùng với các độ đo chính xác

    def score(topk: np.ndarray) -> dict:
        acc = ranking_metrics(topk, relevant, ds.n_items)
        return {"NDCG@10": acc["NDCG@10"], "HitRate@10": acc["HitRate@10"], "Coverage": acc["Coverage"],
                **beyond_metrics(topk, pop, ds.n_users, gvec, users)}

    # ---------- phần 1: 11 mô hình ----------
    cache_dir = ROOT / "artifacts" / "topk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ORDER:
        if name in args.skip:
            continue
        path = cache_dir / f"{name}.npy"
        t0 = time.perf_counter()
        if args.reuse and path.exists():
            topk = np.load(path)
            how = "dùng lại"
        else:
            print(f"[{name}] đang huấn luyện...", flush=True)
            topk = recommend_topk(builder.get(name), seen, ds.n_users, 10)
            np.save(path, topk)
            how = f"{time.perf_counter() - t0:.0f}s"
        rows.append({"Model": name, **score(topk)})
        r = rows[-1]
        print(f"[{name}] {how} | NDCG {r['NDCG@10']:.4f} | ILD {r['ILD']:.3f} | Novelty {r['Novelty']:.2f} | "
              f"LongTail {r['LongTail']:.3f} | Gini {r['Gini']:.3f}", flush=True)
    table = pd.DataFrame(rows).set_index("Model").round(4)
    table.to_csv(ROOT / "results" / "beyond.csv")
    print("\n" + table.to_string())

    # ---------- phần 2: MMR ----------
    mmr = []
    for name in args.mmr_models:
        if name in args.skip:
            continue
        model = builder.get(name)
        for lam in LAMBDAS:
            t0 = time.perf_counter()
            topk = recommend_mmr(model, seen, ds.n_users, gvec, lam, 10, args.n_cand)
            mmr.append({"Model": name, "lambda": lam, **score(topk)})
            r = mmr[-1]
            print(f"[MMR {name} lam={lam}] NDCG {r['NDCG@10']:.4f} | ILD {r['ILD']:.3f} | Genres {r['Genres']:.2f} "
                  f"| Coverage {r['Coverage']:.3f} | {time.perf_counter() - t0:.0f}s", flush=True)
    if mmr:
        pd.DataFrame(mmr).round(4).to_csv(ROOT / "results" / "mmr.csv", index=False)

    # ---------- phần 3: phạt độ phổ biến ----------
    pen = []
    for name in args.pop_models:
        if name in args.skip:
            continue
        model = builder.get(name)
        for beta in BETAS:
            topk = recommend_mmr(model, seen, ds.n_users, gvec, 1.0, 10, args.n_cand, pop, beta)
            pen.append({"Model": name, "beta": beta, **score(topk)})
            r = pen[-1]
            print(f"[Pop {name} beta={beta}] NDCG {r['NDCG@10']:.4f} | Novelty {r['Novelty']:.2f} | "
                  f"LongTail {r['LongTail']:.3f} | Coverage {r['Coverage']:.3f}", flush=True)
    if pen:
        pd.DataFrame(pen).round(4).to_csv(ROOT / "results" / "pop_penalty.csv", index=False)
    print("BEYOND_DONE")


if __name__ == "__main__":
    main()
