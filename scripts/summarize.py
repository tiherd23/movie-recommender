"""Gộp kết quả của mọi mô hình thành một bảng: results/summary.csv và results/summary.md.

Chạy: python scripts/summarize.py   (sau run_final.py, run_two_stage.py, run_ncf.py)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10", "Coverage", "RMSE"]


def main() -> None:
    parts = [pd.read_csv(ROOT / "results" / f, index_col="Model")
             for f in ("final.csv", "two_stage.csv", "ncf.csv") if (ROOT / "results" / f).exists()]
    table = pd.concat(parts)[COLS].sort_values("NDCG@10")
    base = table.loc["Popularity", "NDCG@10"]
    table["vs Popularity"] = ((table["NDCG@10"] / base - 1) * 100).round(1).map(lambda v: f"{v:+.1f}%")
    table.to_csv(ROOT / "results" / "summary.csv")
    lines = ["| Mô hình | " + " | ".join(table.columns) + " |", "|---|" + "---|" * len(table.columns)]
    for name, row in table.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string())


if __name__ == "__main__":
    main()
