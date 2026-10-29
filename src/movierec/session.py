"""Giữ đăng nhập qua các lần tải lại trang: mã phiên có chữ ký HMAC lưu trong cookie trình duyệt.

Mã = "<id người dùng hoặc g (khách)>.<hạn dùng, giây Unix>.<chữ ký>". Chữ ký tính bằng khóa bí mật của máy chủ
(tệp data/.session_key, tự tạo, không đưa lên git) trên id, hạn dùng và một phần mã băm mật khẩu: đổi mật khẩu
hoặc xóa tài khoản thì mọi mã cũ hết hiệu lực. Không ai sửa được id trong mã vì không có khóa để ký lại.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path

COOKIE = "mr_session"
MAX_AGE = 7 * 24 * 3600  # 7 ngày


def load_key(path: Path | str) -> bytes:
    path = Path(path)
    if path.exists() and len(path.read_bytes()) >= 32:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    return key


def _sign(key: bytes, who: str, exp: int, pw_hash: bytes) -> str:
    msg = f"{who}.{exp}.".encode() + hashlib.sha256(pw_hash).digest()[:8]
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[:32]


def make_token(key: bytes, user_id: int | None, pw_hash: bytes = b"", now: float | None = None) -> str:
    """user_id=None: phiên khách."""
    who = "g" if user_id is None else str(int(user_id))
    exp = int((now or time.time()) + MAX_AGE)
    return f"{who}.{exp}.{_sign(key, who, exp, pw_hash)}"


def read_token(key: bytes, token: str | None, pw_hash_of, now: float | None = None) -> tuple[bool, int | None]:
    """(hợp lệ?, id người dùng hoặc None nếu là khách). pw_hash_of(id) -> mã băm mật khẩu, None nếu không còn tài khoản."""
    try:
        who, exp_s, sig = str(token).split(".")
        exp = int(exp_s)
    except ValueError:
        return False, None
    if exp < (now or time.time()):
        return False, None
    if who == "g":
        return hmac.compare_digest(sig, _sign(key, who, exp, b"")), None
    if not who.isdigit():
        return False, None
    pw_hash = pw_hash_of(int(who))
    if pw_hash is None:
        return False, None
    return hmac.compare_digest(sig, _sign(key, who, exp, pw_hash)), int(who)


def cookie_script(token: str | None) -> str:
    """Đoạn JavaScript đặt (token) hoặc xóa (None) cookie phiên trên trang chính."""
    if token is None:
        return f"<script>document.cookie='{COOKIE}=; path=/; max-age=0; SameSite=Lax';</script>"
    return f"<script>document.cookie='{COOKIE}={token}; path=/; max-age={MAX_AGE}; SameSite=Lax';</script>"
