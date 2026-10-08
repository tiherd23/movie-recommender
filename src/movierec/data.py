"""Nạp MovieLens 1M, đánh lại chỉ số và chia train/val/test theo thời gian.

Luồng xử lý:
    load_ml1m()      -> đọc 3 file .dat thành DataFrame
    build_dataset()  -> ánh xạ userId/movieId sang chỉ số liên tục 0..n-1,
                        rồi chia dữ liệu của TỪNG người dùng theo thứ tự thời gian
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "raw" / "ml-1m"


@dataclass
class Dataset:
    """Gói dữ liệu đã xử lý, dùng chung cho mọi mô hình."""

    train: pd.DataFrame  # cột: user, item, rating, timestamp (user/item là chỉ số)
    val: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame  # index = item, cột: movie_id, title, genres
    users: pd.DataFrame  # index = user, cột: user_id, gender, age, occupation
    n_users: int
    n_items: int

    def matrix(self, df: pd.DataFrame | None = None, binary: bool = False) -> sparse.csr_matrix:
        """Ma trận thưa user x item. binary=True: 1 nếu có tương tác, ngược lại là điểm."""
        df = self.train if df is None else df
        vals = np.ones(len(df), dtype=np.float32) if binary else df["rating"].to_numpy(np.float32)
        return sparse.csr_matrix(
            (vals, (df["user"].to_numpy(), df["item"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )


def load_ml1m(data_dir: Path | str = DEFAULT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc ratings.dat, movies.dat, users.dat (phân tách bằng '::')."""
    data_dir = Path(data_dir)
    if not (data_dir / "ratings.dat").exists():
        raise FileNotFoundError(
            f"Khong thay {data_dir / 'ratings.dat'}. Hay chay: python scripts/download_data.py"
        )
    # '::' gồm 2 ký tự nên tách bằng ':' rồi lấy các cột chẵn -> dùng được engine C (nhanh).
    ratings = pd.read_csv(
        data_dir / "ratings.dat", sep=":", header=None, usecols=[0, 2, 4, 6],
        names=["user_id", "_a", "movie_id", "_b", "rating", "_c", "timestamp"],
    )
    users = pd.read_csv(
        data_dir / "users.dat", sep=":", header=None, usecols=[0, 2, 4, 6],
        names=["user_id", "_a", "gender", "_b", "age", "_c", "occupation", "_d", "zip"],
    )
    # Tên phim có thể chứa ':' nên file này phải tách đúng bằng '::'.
    movies = pd.read_csv(
        data_dir / "movies.dat", sep="::", header=None, engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    return ratings, movies, users


TMDB_CSV = ROOT / "data" / "processed" / "tmdb.csv"
TMDB_TEXT_COLS = ["overview", "keywords", "cast", "director", "poster_path"]


def attach_tmdb(movies: pd.DataFrame, path: Path | str | None = None) -> pd.DataFrame:
    """Thêm mô tả, từ khóa, diễn viên, đạo diễn, poster từ TMDB (nếu đã chạy scripts/fetch_tmdb.py).
    Phim không ghép được thì các cột này là chuỗi rỗng."""
    path = TMDB_CSV if path is None else Path(path)
    if not path.exists():
        return movies
    tmdb = pd.read_csv(path).drop_duplicates("movie_id").set_index("movie_id")
    out = movies.copy()
    for col in TMDB_TEXT_COLS:
        out[col] = out["movie_id"].map(tmdb[col]).fillna("").astype(str).to_numpy()
    out["tmdb_id"] = out["movie_id"].map(tmdb["tmdb_id"]).to_numpy()
    return out


def build_dataset(
    data_dir: Path | str = DEFAULT_DIR, val_frac: float = 0.1, test_frac: float = 0.1,
    tmdb_path: Path | str | None = None,
) -> Dataset:
    """Chia theo thời gian cho từng người dùng: 80% đầu train, 10% val, 10% cuối test.

    Lý do không chia ngẫu nhiên: khi triển khai thật, mô hình chỉ biết quá khứ và
    phải dự đoán tương lai. Chia ngẫu nhiên cho mô hình "nhìn trước" tương lai
    nên kết quả đẹp hơn thực tế.
    """
    ratings, movies, users = load_ml1m(data_dir)

    # Chỉ giữ phim/người dùng có xuất hiện trong ratings, đánh lại chỉ số 0..n-1.
    user_ids = np.sort(ratings["user_id"].unique())
    movie_ids = np.sort(ratings["movie_id"].unique())
    u_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_map = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    ratings["user"] = ratings["user_id"].map(u_map).to_numpy()
    ratings["item"] = ratings["movie_id"].map(i_map).to_numpy()

    ratings = ratings.sort_values(["user", "timestamp"], kind="stable")
    grp = ratings.groupby("user", sort=False)
    pos = grp.cumcount().to_numpy()  # vị trí của lượt chấm trong lịch sử người dùng
    size = grp["item"].transform("size").to_numpy()
    frac = pos / size

    cols = ["user", "item", "rating", "timestamp"]
    train = ratings.loc[frac < 1 - val_frac - test_frac, cols].reset_index(drop=True)
    val = ratings.loc[(frac >= 1 - val_frac - test_frac) & (frac < 1 - test_frac), cols].reset_index(drop=True)
    test = ratings.loc[frac >= 1 - test_frac, cols].reset_index(drop=True)

    movies = movies[movies["movie_id"].isin(movie_ids)].copy()
    movies.index = movies["movie_id"].map(i_map).to_numpy()
    movies = movies.sort_index()
    if tmdb_path is None and Path(data_dir) == DEFAULT_DIR:
        tmdb_path = TMDB_CSV  # dữ liệu TMDB chỉ khớp với bộ MovieLens thật
    if tmdb_path is not None:
        movies = attach_tmdb(movies, tmdb_path)
    users = users[users["user_id"].isin(user_ids)].copy()
    users.index = users["user_id"].map(u_map).to_numpy()
    users = users.sort_index()

    return Dataset(train, val, test, movies, users, len(user_ids), len(movie_ids))
