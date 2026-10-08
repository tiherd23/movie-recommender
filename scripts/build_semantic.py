"""Nhúng mô tả của toàn bộ phim để phục vụ tìm kiếm bằng câu tiếng Việt.

Cài thêm:  pip install sentence-transformers      (cần PyTorch; lần đầu tải mô hình khoảng 470 MB)
Chạy:      python scripts/build_semantic.py
           python scripts/build_semantic.py --query "phim hoạt hình vui nhộn cho trẻ em"
Kết quả:   artifacts/semantic.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import build_dataset  # noqa: E402
from movierec.semantic import DEFAULT_INDEX, DEFAULT_MODEL, SemanticIndex, load_encoder  # noqa: E402

DEMO = ["phim hoạt hình vui nhộn cho trẻ em", "phi hành gia bị mắc kẹt ngoài không gian",
        "chuyện tình lãng mạn trên con tàu", "thám tử điều tra kẻ giết người hàng loạt",
        "robot từ tương lai quay về quá khứ", "phim chiến tranh thế giới thứ hai"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--query", action="append", help="câu hỏi để thử (có thể lặp lại nhiều lần)")
    ap.add_argument("--rebuild", action="store_true", help="nhúng lại dù đã có tệp chỉ mục")
    args = ap.parse_args()

    ds = build_dataset()
    if "overview" not in ds.movies.columns:
        raise SystemExit("Chưa có dữ liệu TMDB. Chạy trước: python scripts/fetch_tmdb.py")
    from movierec.models import Popularity
    import pandas as pd
    pop = Popularity().fit(ds, pd.concat([ds.train, ds.val, ds.test], ignore_index=True)).scores_
    encoder = load_encoder(args.model)
    if DEFAULT_INDEX.exists() and not args.rebuild:
        index = SemanticIndex.load().set_encoder(encoder)
        print(f"Da nap chi muc co san: {index.embeddings.shape}")
    else:
        t0 = time.perf_counter()
        index = SemanticIndex.build(ds.movies, encoder, args.model)
        index.save()
        print(f"Da nhung {index.embeddings.shape[0]} phim, {index.embeddings.shape[1]} chieu, "
              f"{time.perf_counter() - t0:.0f}s -> {DEFAULT_INDEX}")

    for q in args.query or DEMO:
        t0 = time.perf_counter()
        hits = index.search(q, k=5, popularity=pop)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n> {q}   ({ms:.0f} ms)")
        for i, s in hits:
            print(f"   {s:.3f}  {ds.movies.iloc[i]['title']}  [{ds.movies.iloc[i]['genres']}]")
    print("\nSEMANTIC_DONE")


if __name__ == "__main__":
    main()
