"""Lấy poster, mô tả, từ khóa, diễn viên từ TMDB cho toàn bộ phim MovieLens 1M.

Chạy:  python scripts/fetch_tmdb.py            (toàn bộ, khoảng 5-10 phút)
       python scripts/fetch_tmdb.py --limit 20 (thử nhanh 20 phim đầu)
Cần file .env chứa TMDB_API_KEY=...
Chạy lại được nhiều lần: phim đã lấy rồi sẽ bỏ qua (lưu tạm ở data/processed/tmdb_raw.jsonl).
Kết quả: data/processed/tmdb.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.data import load_ml1m  # noqa: E402
from movierec.tmdb import TMDBClient  # noqa: E402

RAW = ROOT / "data" / "processed" / "tmdb_raw.jsonl"
OUT = ROOT / "data" / "processed" / "tmdb.csv"


def read_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("TMDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise SystemExit("Chưa có khóa. Tạo file .env với dòng: TMDB_API_KEY=<khóa của bạn>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N phim đầu (để thử)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retry-below", type=float, default=0.0,
                    help="lấy lại phim chưa tìm thấy hoặc có match_score dưới ngưỡng này")
    args = ap.parse_args()

    client = TMDBClient(read_key())
    _, movies, _ = load_ml1m()
    if args.limit:
        movies = movies.head(args.limit)

    RAW.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["movie_id"]] = row
    if args.retry_below > 0:
        # lấy lại các phim chưa tìm thấy hoặc khớp kém (sau khi cải tiến thuật toán ghép tên)
        done = {k: v for k, v in done.items()
                if v.get("tmdb_id") is not None and v.get("match_score", 0) >= args.retry_below}
        RAW.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values()), encoding="utf-8")
    todo = [(int(r.movie_id), r.title) for r in movies.itertuples() if int(r.movie_id) not in done]
    print(f"Tong: {len(movies)} phim | da co: {len(movies) - len(todo)} | can lay: {len(todo)}")

    lock, t0 = threading.Lock(), time.perf_counter()

    def work(movie_id: int, title: str) -> dict:
        return {"movie_id": movie_id, "ml_title": title, **client.match(title)}

    with ThreadPoolExecutor(args.workers) as pool, RAW.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(work, mid, title) for mid, title in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            with lock:
                done[row["movie_id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 200 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    table = pd.DataFrame([done[int(m)] for m in movies["movie_id"] if int(m) in done])
    table.to_csv(OUT, index=False, encoding="utf-8")
    found = table["tmdb_id"].notna()
    print(f"\nDa luu {OUT}")
    print(f"Tim thay: {found.sum()}/{len(table)} ({found.mean():.1%}) | "
          f"co poster: {table.get('poster_path', pd.Series(dtype=object)).notna().sum()} | "
          f"co mo ta: {(table.get('overview', pd.Series(dtype=object)).fillna('').str.len() > 0).sum()}")
    low = table[found & (table["match_score"] < 0.7)]
    if len(low):
        print(f"Can xem lai {len(low)} phim co diem khop thap (< 0.7), vi du:")
        print(low[["ml_title", "tmdb_title", "match_score"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
