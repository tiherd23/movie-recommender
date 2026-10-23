"""Giải thích bộ xếp hạng hai tầng bằng SHAP (TreeSHAP có sẵn trong LightGBM, pred_contrib=True).

Chạy: python scripts/explain_ranker.py   (vài phút, chỉ dùng CPU)
Tầng 1 học lại trên train+val như scripts/run_two_stage.py, rồi với mỗi người dùng lấy 10 phim được xếp cao nhất
trên tập test và tính giá trị SHAP của từng nhóm đặc trưng cho các phim đó.
Kết quả -> results/shap_groups.csv: mỗi nhóm có
  mean_abs    : trung bình |SHAP| trên các phim được gợi ý (mức ảnh hưởng)
  mean        : trung bình SHAP (dấu cho biết thường đẩy lên hay kéo xuống)
  top_share   : tỉ lệ phim được gợi ý mà nhóm này là lý do đẩy lên mạnh nhất
  gain_share  : tỉ lệ "gain" của LightGBM (độ quan trọng toàn cục, để so sánh)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.ranker import FEATURE_GROUPS, TwoStageRanker, feature_group, grouped_contributions  # noqa: E402


def main() -> None:
    ds = build_dataset()
    keep = pickle.loads((ROOT / "artifacts" / "ranker.pkl").read_bytes())
    ranker = TwoStageRanker(keep["als_params"], keep["knn_params"], keep["n_candidates"], keep["lgb_params"], keep["seed"])
    ranker.booster_ = keep["booster_"]
    ranker.fit(ds, pd.concat([ds.train, ds.val], ignore_index=True))
    booster = ranker.booster_

    parts = []
    for start in range(0, ds.n_users, 512):
        users = np.arange(start, min(start + 512, ds.n_users))
        rows, items, X = ranker._features(users)
        score = booster.predict(X, num_iteration=booster.best_iteration)
        # 10 ứng viên điểm cao nhất của mỗi người (đúng danh sách được gợi ý khi đánh giá)
        order = np.lexsort((-score, rows))
        first = np.r_[0, np.flatnonzero(np.diff(rows[order])) + 1]
        pos_in_user = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        pick = order[pos_in_user < 10]
        parts.append(grouped_contributions(booster, X[pick]))
        print(f"{min(start + 512, ds.n_users)}/{ds.n_users} người dùng", flush=True)
    shap = pd.concat(parts, ignore_index=True)
    groups = [g for g in FEATURE_GROUPS if g in shap.columns]

    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = gain.groupby([feature_group(n) for n in gain.index]).sum()
    top = shap[[g for g in groups if g != "user"]].idxmax(axis=1).value_counts(normalize=True)
    table = pd.DataFrame({
        "label": [FEATURE_GROUPS[g] for g in groups],
        "mean_abs": shap[groups].abs().mean().to_numpy(),
        "mean": shap[groups].mean().to_numpy(),
        "top_share": [top.get(g, 0.0) for g in groups],
        "gain_share": (gain.reindex(groups).fillna(0) / gain.sum()).to_numpy(),
    }, index=pd.Index(groups, name="group")).sort_values("mean_abs", ascending=False).round(4)
    table.to_csv(ROOT / "results" / "shap_groups.csv")
    print(f"\n{len(shap)} cặp (người dùng, phim được gợi ý); base = {shap['base'].iloc[0]:.4f}")
    print(table.to_string())
    print("SHAP_DONE")


if __name__ == "__main__":
    main()
