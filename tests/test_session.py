"""Mã phiên giữ đăng nhập khi tải lại trang."""
from movierec import session as sess
from movierec.db import Database

import pytest

KEY = b"k" * 32


def test_token_roundtrip_and_tampering():
    pw = {1: b"hash-1"}
    tok = sess.make_token(KEY, 1, pw[1], now=1000)
    assert sess.read_token(KEY, tok, pw.get, now=1001) == (True, 1)
    assert sess.read_token(KEY, tok.replace("1.", "2.", 1), pw.get, now=1001)[0] is False  # sửa id: chữ ký sai
    assert sess.read_token(b"x" * 32, tok, pw.get, now=1001)[0] is False                  # khóa khác
    assert sess.read_token(KEY, tok, {1: b"doi-mat-khau"}.get, now=1001)[0] is False      # đổi mật khẩu
    assert sess.read_token(KEY, tok, {}.get, now=1001)[0] is False                        # tài khoản đã xóa
    assert sess.read_token(KEY, tok, pw.get, now=1000 + sess.MAX_AGE + 1)[0] is False     # hết hạn
    for bad in (None, "", "abc", "1.x.y", "a.b.c.d"):
        assert sess.read_token(KEY, bad, pw.get)[0] is False
    guest = sess.make_token(KEY, None, now=1000)
    assert sess.read_token(KEY, guest, pw.get, now=1001) == (True, None)


def test_key_is_created_once(tmp_path):
    key = sess.load_key(tmp_path / ".session_key")
    assert len(key) == 32 and sess.load_key(tmp_path / ".session_key") == key


def test_register_validates_username_and_password(tmp_path):
    db = Database(tmp_path / "a.db")
    for name, pw in [("ab", "12345"), ("abc", "    "), ("Phú Nguyễn", "12345"), ("<b>x</b>", "12345"),
                     ("x" * 31, "12345"), ("vui😀", "12345")]:
        with pytest.raises(ValueError):
            db.register(name, pw)
    uid = db.register("phu_nguyen.2", "12345")
    assert db.user(uid)[0] == "phu_nguyen.2" and db.user(999) is None
    assert db.register("Phú", "12345")  # chữ tiếng Việt có dấu vẫn được


def test_diversity_saved_and_old_db_migrated(tmp_path):
    import sqlite3

    old = tmp_path / "old.db"  # cơ sở dữ liệu tạo từ bản cũ, chưa có cột diversity
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, "
                "salt BLOB NOT NULL, pw_hash BLOB NOT NULL, genres TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)")
    con.execute("INSERT INTO users (username, salt, pw_hash, created_at) VALUES ('cu', x'00', x'00', 0)")
    con.commit(); con.close()
    db = Database(old)
    assert db.get_diversity(1) == ""  # tài khoản cũ còn nguyên, chưa chọn mức nào
    db.set_diversity(1, "Mạnh")
    assert Database(old).get_diversity(1) == "Mạnh"
