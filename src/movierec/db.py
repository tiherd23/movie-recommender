"""Lưu tài khoản và đánh giá của người dùng web bằng SQLite (một file, không cần server)."""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path


def _hash(password: str, salt: bytes) -> bytes:
    # Không lưu mật khẩu gốc: chỉ lưu kết quả băm PBKDF2 kèm salt ngẫu nhiên.
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)


class Database:
    def __init__(self, path: Path | str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt BLOB NOT NULL,
                pw_hash BLOB NOT NULL,
                genres TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL REFERENCES users(id),
                item INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                rated_at REAL NOT NULL,
                PRIMARY KEY (user_id, item)
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id INTEGER NOT NULL REFERENCES users(id),
                item INTEGER NOT NULL,
                added_at REAL NOT NULL,
                PRIMARY KEY (user_id, item)
            );
            """
        )

    def register(self, username: str, password: str) -> int:
        username = username.strip()
        if len(username) < 3 or len(password) < 4:
            raise ValueError("Tên đăng nhập cần ít nhất 3 ký tự, mật khẩu ít nhất 4 ký tự.")
        salt = secrets.token_bytes(16)
        try:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
                    (username, salt, _hash(password, salt), time.time()),
                )
        except sqlite3.IntegrityError:
            raise ValueError("Tên đăng nhập đã tồn tại.") from None
        return int(cur.lastrowid)

    def login(self, username: str, password: str) -> int | None:
        row = self.conn.execute(
            "SELECT id, salt, pw_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        if row and secrets.compare_digest(_hash(password, row[1]), row[2]):
            return int(row[0])
        return None

    def set_genres(self, user_id: int, genres: list[str]) -> None:
        with self.conn:
            self.conn.execute("UPDATE users SET genres = ? WHERE id = ?", ("|".join(genres), user_id))

    def get_genres(self, user_id: int) -> list[str]:
        row = self.conn.execute("SELECT genres FROM users WHERE id = ?", (user_id,)).fetchone()
        return [g for g in (row[0] if row else "").split("|") if g]

    def rate(self, user_id: int, item: int, rating: int | None) -> None:
        """rating=None: xóa đánh giá."""
        with self.conn:
            if rating is None:
                self.conn.execute("DELETE FROM ratings WHERE user_id = ? AND item = ?", (user_id, item))
            else:
                self.conn.execute(
                    "INSERT INTO ratings (user_id, item, rating, rated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(user_id, item) DO UPDATE SET rating = excluded.rating, rated_at = excluded.rated_at",
                    (user_id, item, int(rating), time.time()),
                )

    def ratings(self, user_id: int) -> dict[int, int]:
        rows = self.conn.execute(
            "SELECT item, rating FROM ratings WHERE user_id = ? ORDER BY rated_at DESC", (user_id,)
        ).fetchall()
        return {int(i): int(r) for i, r in rows}

    # ---------- danh sách "muốn xem" ----------
    def toggle_watch(self, user_id: int, item: int) -> bool:
        """Thêm phim vào danh sách muốn xem, hoặc bỏ ra nếu đã có. Trả về True nếu phim đang ở trong danh sách."""
        with self.conn:
            if self.conn.execute("DELETE FROM watchlist WHERE user_id = ? AND item = ?", (user_id, item)).rowcount:
                return False
            self.conn.execute("INSERT INTO watchlist (user_id, item, added_at) VALUES (?, ?, ?)",
                              (user_id, item, time.time()))
            return True

    def watchlist(self, user_id: int) -> list[int]:
        rows = self.conn.execute("SELECT item FROM watchlist WHERE user_id = ? ORDER BY added_at DESC",
                                 (user_id,)).fetchall()
        return [int(r[0]) for r in rows]

    # ---------- quản trị ----------
    def stats(self) -> dict:
        """Số liệu tổng quan cho trang quản trị."""
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        week = time.time() - 7 * 24 * 3600
        return {"users": q("SELECT COUNT(*) FROM users"), "ratings": q("SELECT COUNT(*) FROM ratings"),
                "avg_rating": q("SELECT AVG(rating) FROM ratings"), "watchlist": q("SELECT COUNT(*) FROM watchlist"),
                "active_7d": self.conn.execute("SELECT COUNT(DISTINCT user_id) FROM ratings WHERE rated_at >= ?",
                                               (week,)).fetchone()[0]}

    def users_overview(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT u.id, u.username, u.created_at, COUNT(r.item), AVG(r.rating), MAX(r.rated_at) "
            "FROM users u LEFT JOIN ratings r ON r.user_id = u.id GROUP BY u.id ORDER BY u.id").fetchall()
        return [{"id": r[0], "username": r[1], "created_at": r[2], "n_ratings": r[3], "avg_rating": r[4],
                 "last_rated": r[5]} for r in rows]

    def all_ratings(self) -> list[tuple[int, int, int, float]]:
        """(user_id, item, rating, rated_at) của mọi người dùng web, để thống kê hoặc xuất ra CSV."""
        return [tuple(r) for r in self.conn.execute("SELECT user_id, item, rating, rated_at FROM ratings")]

    def delete_user(self, user_id: int) -> None:
        with self.conn:
            for table in ("ratings", "watchlist"):
                self.conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
