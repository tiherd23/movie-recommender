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
