"""Huấn luyện và đánh giá kiến trúc hai tầng (truy hồi + LightGBM).

Chạy: python scripts/run_two_stage.py            (cả hai bước, vài phút)
      python scripts/run_two_stage.py --stage train | test
Bước train: tầng 1 học trên train, nhãn xếp hạng lấy từ val  -> artifacts/ranker.pkl
Bước test : tầng 1 học lại trên train+val, chấm trên test    -> results/two_stage.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.ranker import TwoStageRanker  # noqa: E402

MODEL = ROOT / "artifacts" / "ranker.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "train", "test"], default="all")
    args = ap.parse_args()
    ds = build_dataset()
    best = json.loads((ROOT / "results" / "best_params.json").read_text())

    if args.stage in ("all", "train"):
        t0 = time.perf_counter()
        ranker = TwoStageRanker(best["ImplicitALS"], best["ItemKNN"])
        info = ranker.fit_ranker(ds, hist=ds.train, label=ds.val)
        info["train_seconds"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(info, indent=2))
        print("\nDo quan trong cua dac trung (ti le gain):")
        print(ranker.importance_.head(12).round(4).to_string())
        MODEL.parent.mkdir(exist_ok=True)
        keep = {k: getattr(ranker, k) for k in ("booster_", "importance_", "als_params", "knn_params",
                                                "n_candidates", "lgb_params", "seed")}
        MODEL.write_bytes(pickle.dumps(keep))
        (ROOT / "results" / "two_stage_info.json").write_text(json.dumps(info, indent=2))
        ranker.importance_.round(5).to_csv(ROOT / "results" / "feature_importance.csv", header=["gain_share"])

    if args.stage in ("all", "test"):
        keep = pickle.loads(MODEL.read_bytes())
        ranker = TwoStageRanker(keep["als_params"], keep["knn_params"], keep["n_candidates"], keep["lgb_params"], keep["seed"])
        ranker.booster_, ranker.importance_ = keep["booster_"], keep["importance_"]
        t0 = time.perf_counter()
        ranker.fit(ds, pd.concat([ds.train, ds.val], ignore_index=True))
        res = {"Model": "TwoStage", **evaluate(ranker, ds, "test"), "fit_s": round(time.perf_counter() - t0, 1)}
        table = pd.DataFrame([res]).set_index("Model").drop(columns="n_eval_users").round(4)
        print("\n" + table.to_string())
        table.to_csv(ROOT / "results" / "two_stage.csv")
        print("TWO_STAGE_DONE")


if __name__ == "__main__":
    main()
