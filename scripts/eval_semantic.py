"""So sánh các mô hình nhúng cho tìm kiếm tiếng Việt và chọn cấu hình tốt nhất.

Bộ câu hỏi: data/eval/semantic_queries.csv gồm 40 câu tiếng Việt mô tả nội dung một phim nổi tiếng
(không nhắc tên phim), kèm tên phim đúng. 20 câu "dev" dùng để chọn mô hình và trọng số độ phổ biến,
20 câu "test" chỉ dùng để báo cáo kết quả cuối, tránh chọn cấu hình theo chính các câu dùng để chấm.

Độ đo: Hit@k = tỉ lệ câu hỏi có phim đúng trong k kết quả đầu; MRR = trung bình của 1/thứ hạng phim đúng.
Chạy:  python scripts/eval_semantic.py
Kết quả: results/semantic_eval.csv; cấu hình tốt nhất được lưu thành artifacts/semantic.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.models import Popularity  # noqa: E402
from movierec.semantic import DEFAULT_INDEX, SemanticIndex, _z, load_encoder  # noqa: E402

MODELS = ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "BAAI/bge-m3"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def attempts(device: str) -> list[tuple[str, int, bool]]:
    """Các cách chạy thử lần lượt: (thiết bị, cỡ lô, dùng số 16 bit). GPU lỗi thì chuyển sang CPU."""
    if device == "cpu":
        return [("cpu", 16, False)]
    try:
        import torch
        if torch.cuda.is_available():
            # không dùng số 16 bit: thử nghiệm cho thấy bge-m3 trả về NaN ở chế độ đó
            return [("cuda", 8, False), ("cpu", 16, False)]
    except Exception:
        pass
    return [("cpu", 16, False)]


def metrics(ranks: np.ndarray) -> dict[str, float]:
    return {"Hit@1": float((ranks <= 1).mean()), "Hit@5": float((ranks <= 5).mean()),
            "Hit@10": float((ranks <= 10).mean()), "MRR": float((1.0 / ranks).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto", help="cpu: không dùng GPU")
    ap.add_argument("--rebuild", action="store_true", help="nhúng lại toàn bộ phim, bỏ qua chỉ mục đã lưu")
    args = ap.parse_args()

    ds = build_dataset()
    full = pd.concat([ds.train, ds.val, ds.test], ignore_index=True)
    pop_z = _z(np.log1p(Popularity().fit(ds, full).scores_))
    title_to_item = {t: i for i, t in enumerate(ds.movies["title"])}
    queries = pd.read_csv(ROOT / "data" / "eval" / "semantic_queries.csv")
    queries["item"] = queries["target"].map(title_to_item)
    assert queries["item"].notna().all(), "Có tên phim trong bộ câu hỏi không khớp dữ liệu"

    rows, indexes = [], {}
    for name in args.models:
        slug = name.split("/")[-1]
        path = ROOT / "artifacts" / f"semantic_{slug}.npz"
        index = sims = None
        for device, batch, half in attempts(args.device):
            try:
                t0 = time.perf_counter()
                encoder = load_encoder(name, device=device, batch_size=batch, half=half)
                index = None
                if path.exists() and not args.rebuild:
                    try:
                        index = SemanticIndex.load(path).set_encoder(encoder)
                    except ValueError as err:  # chỉ mục cũ bị hỏng (NaN): dựng lại
                        print(f"{slug}: {err}", flush=True)
                if index is None:
                    index = SemanticIndex.build(ds.movies, encoder, name)
                    index.save(path)
                sims = np.vstack([index.similarities(q) for q in queries["query"]])  # (số câu hỏi, số phim)
                print(f"{slug}: xong tren {device} sau {time.perf_counter() - t0:.0f}s", flush=True)
                break
            except Exception as err:  # lỗi GPU, hết bộ nhớ...: thử cách kế tiếp
                print(f"{slug}: loi tren {device} ({type(err).__name__}: {str(err)[:90]}), thu cach khac", flush=True)
                index = sims = None
        if sims is None:
            print(f"{slug}: BO QUA", flush=True)
            continue
        indexes[name] = index
        assert np.isfinite(sims).all(), "độ tương đồng chứa NaN: kết quả sẽ sai"
        sims_z = (sims - sims.mean(axis=1, keepdims=True)) / (sims.std(axis=1, keepdims=True) + 1e-8)
        target = queries["item"].to_numpy(int)
        for w in WEIGHTS:
            score = sims_z + w * pop_z[None, :]
            # thứ hạng của phim đúng = 1 + số phim có điểm cao hơn
            ranks = 1 + (score > score[np.arange(len(target)), target][:, None]).sum(axis=1)
            for split in ("dev", "test"):
                mask = (queries["split"] == split).to_numpy()
                rows.append({"model": slug, "pop_weight": w, "split": split, **metrics(ranks[mask]),
                             "median_rank": float(np.median(ranks[mask]))})
        del encoder
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    table = pd.DataFrame(rows)
    table.round(4).to_csv(ROOT / "results" / "semantic_eval.csv", index=False)
    dev = table[table["split"] == "dev"].sort_values(["MRR", "Hit@10"], ascending=False)
    print("\nTap dev (dung de chon cau hinh):")
    print(dev.drop(columns="split").round(3).to_string(index=False))
    best = dev.iloc[0]
    name = next(n for n in indexes if n.split("/")[-1] == best["model"])
    chosen = indexes[name]
    chosen.pop_weight = float(best["pop_weight"])
    chosen.save(DEFAULT_INDEX)
    test = table[(table["split"] == "test") & (table["model"] == best["model"])]
    print(f"\nDa chon: {best['model']}, pop_weight = {best['pop_weight']} -> {DEFAULT_INDEX}")
    print("Ket qua tren tap test cua mo hinh da chon, theo tung trong so:")
    print(test.drop(columns=["split", "model"]).round(3).to_string(index=False))
    final = test[test["pop_weight"] == best["pop_weight"]].iloc[0]
    base = test[test["pop_weight"] == 0.0].iloc[0]
    print(f"\nTEST  cau hinh da chon: Hit@1={final['Hit@1']:.2f} Hit@5={final['Hit@5']:.2f} "
          f"Hit@10={final['Hit@10']:.2f} MRR={final['MRR']:.3f} | khong tron do pho bien: "
          f"Hit@10={base['Hit@10']:.2f} MRR={base['MRR']:.3f}")
    print("SEMANTIC_EVAL_DONE")


if __name__ == "__main__":
    main()
