"""Lấy mã trailer YouTube của từng phim từ TMDB (/movie/{id}/videos), để xem trailer ngay trên web.

Chạy:  python scripts/fetch_trailers.py            (khoảng 3.800 phim, vài phút)
       python scripts/fetch_trailers.py --limit 20 (thử nhanh)
Cần file .env chứa TMDB_API_KEY=... và đã chạy scripts/fetch_tmdb.py (để có tmdb_id).
Chạy lại được nhiều lần: phim đã lấy rồi sẽ bỏ qua.
Kết quả: data/processed/trailers.csv (movie_id, tmdb_id, youtube_key, trailer_name, trailer_type)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_tmdb import read_key  # noqa: E402
from movierec.extras import pick_trailer  # noqa: E402
from movierec.tmdb import TMDBClient  # noqa: E402

TMDB = ROOT / "data" / "processed" / "tmdb.csv"
OUT = ROOT / "data" / "processed" / "trailers.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not TMDB.exists():
        raise SystemExit("Chưa có data/processed/tmdb.csv. Chạy python scripts/fetch_tmdb.py trước.")

    client = TMDBClient(read_key())
    movies = pd.read_csv(TMDB).dropna(subset=["tmdb_id"]).drop_duplicates("movie_id")
    if args.limit:
        movies = movies.head(args.limit)
    done = pd.read_csv(OUT, dtype={"youtube_key": str}) if OUT.exists() else pd.DataFrame(
        columns=["movie_id", "tmdb_id", "youtube_key", "trailer_name", "trailer_type"])
    todo = movies[~movies["movie_id"].isin(done["movie_id"])]
    print(f"Tong: {len(movies)} phim co tmdb_id | da co: {len(movies) - len(todo)} | can lay: {len(todo)}", flush=True)

    def work(movie_id: int, tmdb_id: int) -> dict:
        videos = client.get(f"/movie/{tmdb_id}/videos").get("results", [])
        best = pick_trailer(videos) or {}
        return {"movie_id": movie_id, "tmdb_id": tmdb_id, "youtube_key": best.get("key"),
                "trailer_name": best.get("name"), "trailer_type": best.get("type")}

    rows, t0 = [], time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, int(r.movie_id), int(r.tmdb_id)) for r in todo.itertuples()]
        for n, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if n % 200 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} ({time.perf_counter() - t0:.0f}s)", flush=True)
                # lưu dần: dừng giữa chừng thì lần sau làm tiếp
                pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT, index=False)
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(OUT, index=False)
    print(f"\nDa luu {OUT}: {table['youtube_key'].notna().sum()}/{len(table)} phim co trailer")
    print("TRAILERS_DONE")


if __name__ == "__main__":
    main()
