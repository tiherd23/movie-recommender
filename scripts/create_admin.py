"""Tạo (hoặc đặt lại mật khẩu) tài khoản quản trị. Trên web không ai tự đăng ký được các tên quản trị.

Chạy: python scripts/create_admin.py admin            # hỏi mật khẩu (không hiện khi gõ)
Tên quản trị lấy từ biến môi trường MOVIEREC_ADMINS (mặc định "admin"), cơ sở dữ liệu từ MOVIEREC_DB.
"""
from __future__ import annotations

import getpass
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.db import Database, _hash  # noqa: E402


def main() -> None:
    admins = {u.strip() for u in os.environ.get("MOVIEREC_ADMINS", "admin").split(",") if u.strip()}
    name = sys.argv[1] if len(sys.argv) > 1 else "admin"
    if name not in admins:
        sys.exit(f"'{name}' không nằm trong MOVIEREC_ADMINS ({', '.join(sorted(admins))}).")
    pw = getpass.getpass(f"Mật khẩu cho {name}: ")
    if len(pw) < 8 or pw != getpass.getpass("Nhập lại: "):
        sys.exit("Mật khẩu cần ít nhất 8 ký tự và hai lần nhập phải giống nhau.")
    db = Database(os.environ.get("MOVIEREC_DB", str(ROOT / "data" / "app.db")))
    row = db.conn.execute("SELECT id FROM users WHERE username = ?", (name,)).fetchone()
    if row:
        salt = secrets.token_bytes(16)
        with db.conn:
            db.conn.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE id = ?", (salt, _hash(pw, salt), row[0]))
        print(f"Đã đặt lại mật khẩu cho '{name}'.")
    else:
        db.register(name, pw)
        print(f"Đã tạo tài khoản quản trị '{name}'.")


if __name__ == "__main__":
    main()
