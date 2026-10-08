"""Huấn luyện và so sánh các mô hình mốc. Chạy: python scripts/run_baselines.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import BiasBaseline, ItemKNN, Popularity  # noqa: E402


def main() -> None:
    ds = build_dataset()
    print(f"Nguoi dung: {ds.n_users} | Phim: {ds.n_items} | "
          f"train/val/test: {len(ds.train)}/{len(ds.val)}/{len(ds.test)}")

    rows = []
    for model in [Popularity(), BiasBaseline(), ItemKNN(k=50)]:
        t0 = time.perf_counter()
        model.fit(ds)
        fit_s = time.perf_counter() - t0
        row = {"Model": model.name, **evaluate(model, ds, split="test", k=10), "fit_s": round(fit_s, 2)}
        rows.append(row)
        print(f"  xong {model.name} ({fit_s:.1f}s)")

    table = pd.DataFrame(rows).set_index("Model").drop(columns="n_eval_users").round(4)
    print("\n" + table.to_string())
    out = ROOT / "results" / "baselines.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(f"\nDa luu {out}")


if __name__ == "__main__":
    main()
