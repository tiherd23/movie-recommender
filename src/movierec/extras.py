"""Thông tin hiển thị thêm cho từng phim trên web: thời lượng, điểm TMDB, đạo diễn, diễn viên, trailer.

Các thông tin này không dùng để huấn luyện mô hình nên được nạp riêng từ data/processed/tmdb.csv
(scripts/fetch_tmdb.py) và data/processed/trailers.csv (scripts/fetch_trailers.py), không cần dựng lại
artifacts/service.pkl. Thiếu tệp nào thì các cột tương ứng để trống.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRAILER_TYPES = ("Trailer", "Teaser", "Clip")  # thứ tự ưu tiên khi chọn video


def pick_trailer(videos: list[dict]) -> dict | None:
    """Chọn một video YouTube từ kết quả /movie/{id}/videos của TMDB.
    Ưu tiên trailer chính thức, rồi trailer bất kỳ, teaser, clip; cùng loại thì video tiếng Anh, nhiều độ phân giải hơn."""
    def rank(v: dict) -> tuple:
        return (TRAILER_TYPES.index(v.get("type")), not v.get("official", False),
                v.get("iso_639_1") != "en", -(v.get("size") or 0))

    usable = [v for v in videos if v.get("site") == "YouTube" and v.get("key") and v.get("type") in TRAILER_TYPES]
    return min(usable, key=rank) if usable else None


def load_movie_extras(tmdb_csv: Path | str, trailers_csv: Path | str) -> pd.DataFrame:
    """Bảng theo movie_id (mã MovieLens): runtime, vote_average, vote_count, release_date, director, cast,
    youtube_key, trailer_name."""
    cols = ["runtime", "vote_average", "vote_count", "release_date", "director", "cast"]
    out = pd.DataFrame(columns=cols + ["youtube_key", "trailer_name"])
    if Path(tmdb_csv).exists():
        tmdb = pd.read_csv(tmdb_csv).drop_duplicates("movie_id").set_index("movie_id")
        out = tmdb.reindex(columns=cols)
    if Path(trailers_csv).exists():
        tr = pd.read_csv(trailers_csv, dtype={"youtube_key": str}).drop_duplicates("movie_id").set_index("movie_id")
        out = out.join(tr[["youtube_key", "trailer_name"]], how="outer")
    for c in ("youtube_key", "trailer_name"):
        if c not in out.columns:
            out[c] = None
    return out
