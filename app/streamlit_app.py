"""Web demo: đăng ký/đăng nhập, chấm điểm phim, nhận gợi ý cá nhân hóa.

Chạy:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.db import Database  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, load_or_build  # noqa: E402

DB_PATH = os.environ.get("MOVIEREC_DB", str(ROOT / "data" / "app.db"))
ARTIFACT = os.environ.get("MOVIEREC_ARTIFACT", str(DEFAULT_ARTIFACT))
MIN_RATINGS = 5  # số phim nên chấm trước khi gợi ý đủ tin cậy
STARS = ["Chưa chấm", "★", "★★", "★★★", "★★★★", "★★★★★"]

st.set_page_config(page_title="MovieRec", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình gợi ý...")
def get_service(path: str):
    return load_or_build(path)  # cache_resource: chỉ nạp một lần cho mọi phiên


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


svc, db = get_service(ARTIFACT), get_db(DB_PATH)


# ---------- thành phần dùng chung ----------
def _save_rating(item: int, key: str) -> None:
    value = st.session_state[key]
    db.rate(st.session_state.user_id, item, value or None)


def rating_box(item: int, prefix: str, ratings: dict[int, int]) -> None:
    """Ô chọn số sao; đổi giá trị là lưu ngay vào SQLite rồi trang tự chạy lại."""
    key = f"{prefix}_{item}"
    st.selectbox("Điểm", range(6), index=ratings.get(item, 0), format_func=lambda v: STARS[v],
                 key=key, label_visibility="collapsed", on_change=_save_rating, args=(item, key))


def movie_grid(movies: list[dict], prefix: str, ratings: dict[int, int], cols: int = 4) -> None:
    columns = st.columns(cols)
    for n, m in enumerate(movies):
        with columns[n % cols], st.container(border=True):
            st.markdown(f"**{m['title']}**")
            st.caption(m["genres"].replace("|", " · "))
            if m.get("reason"):
                st.caption(f"💡 {m['reason']}")
            rating_box(m["item"], prefix, ratings)


# ---------- các trang ----------
def page_auth() -> None:
    st.title("🎬 MovieRec")
    st.write("Hệ thống gợi ý phim cá nhân hóa. Đăng nhập hoặc tạo tài khoản để bắt đầu.")
    tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
    # st.form: gửi cả tên và mật khẩu cùng lúc khi bấm nút, kể cả khi ô nhập chưa mất tiêu điểm
    with tab_login, st.form("login_form", border=False):
        user = st.text_input("Tên đăng nhập", key="login_user")
        pw = st.text_input("Mật khẩu", type="password", key="login_pw")
        if st.form_submit_button("Đăng nhập", key="login_btn", type="primary"):
            uid = db.login(user, pw)
            if uid is None:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")
            else:
                st.session_state.update(user_id=uid, username=user.strip())
                st.rerun()
    with tab_register, st.form("register_form", border=False):
        user = st.text_input("Tên đăng nhập", key="reg_user")
        pw = st.text_input("Mật khẩu", type="password", key="reg_pw")
        if st.form_submit_button("Tạo tài khoản", key="reg_btn", type="primary"):
            try:
                uid = db.register(user, pw)
            except ValueError as err:
                st.error(str(err))
            else:
                st.session_state.update(user_id=uid, username=user.strip())
                st.rerun()


def _save_genres() -> None:
    db.set_genres(st.session_state.user_id, st.session_state.genre_select)


def page_home(ratings: dict[int, int]) -> None:
    st.title("Gợi ý cho bạn")
    genres = db.get_genres(st.session_state.user_id)
    st.multiselect("Thể loại yêu thích", svc.genres, default=genres, key="genre_select",
                   on_change=_save_genres, placeholder="Chọn vài thể loại bạn thích")

    if len(ratings) < MIN_RATINGS:
        # Cold-start: người mới chưa có lịch sử, hỏi sở thích bằng các phim nổi tiếng
        st.info(f"Hãy chấm ít nhất {MIN_RATINGS} phim bạn đã xem để gợi ý chính xác hơn "
                f"(đã chấm {len(ratings)}).")
        st.subheader("Bạn đã xem phim nào dưới đây?")
        movie_grid(svc.popular(24), "onb", ratings)
        st.divider()

    st.subheader("Top phim dành cho bạn")
    movie_grid(svc.recommend(ratings, genres, k=12), "rec", ratings)


def page_search(ratings: dict[int, int]) -> None:
    st.title("Tìm phim")
    query = st.text_input("Tên phim (tiếng Anh)", key="search_q", placeholder="ví dụ: matrix")
    results = svc.search(query)
    if query and not results:
        st.warning("Không tìm thấy phim nào.")
    for m in results:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"**{m['title']}**")
            left.caption(m["genres"].replace("|", " · "))
            with right:
                rating_box(m["item"], "search", ratings)
            with st.expander("Phim tương tự"):
                for s in svc.similar(m["item"]):
                    st.write(f"{s['title']} · {s['genres'].replace('|', ', ')}")


def page_history(ratings: dict[int, int]) -> None:
    st.title("Phim đã chấm")
    if not ratings:
        st.info("Bạn chưa chấm phim nào.")
        return
    rows = [{"Phim": svc.movies.iloc[i]["title"], "Thể loại": svc.movies.iloc[i]["genres"].replace("|", ", "),
             "Điểm": STARS[r]} for i, r in ratings.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    if "user_id" not in st.session_state:
        page_auth()
        return
    ratings = db.ratings(st.session_state.user_id)
    with st.sidebar:
        st.markdown(f"### 🎬 MovieRec\nXin chào, **{st.session_state.username}**")
        page = st.radio("Trang", ["Gợi ý cho bạn", "Tìm phim", "Phim đã chấm"], key="nav",
                        label_visibility="collapsed")
        st.metric("Số phim đã chấm", len(ratings))
        if st.button("Đăng xuất", key="logout_btn"):
            st.session_state.clear()
            st.rerun()
    {"Gợi ý cho bạn": page_home, "Tìm phim": page_search, "Phim đã chấm": page_history}[page](ratings)


main()
