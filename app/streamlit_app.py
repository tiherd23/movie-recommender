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
SEMANTIC = os.environ.get("MOVIEREC_SEMANTIC", str(ROOT / "artifacts" / "semantic.npz"))
LLM_MODEL = os.environ.get("MOVIEREC_LLM", "qwen3:4b")
MIN_RATINGS = 5  # số phim nên chấm trước khi gợi ý đủ tin cậy
STARS = ["Chưa chấm", "★", "★★", "★★★", "★★★★", "★★★★★"]

st.set_page_config(page_title="MovieRec", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình gợi ý...")
def get_service(path: str):
    return load_or_build(path)  # cache_resource: chỉ nạp một lần cho mọi phiên


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_resource(show_spinner="Đang nạp mô hình tìm kiếm ngữ nghĩa...")
def get_semantic(path: str):
    """Chỉ mục tìm kiếm ngữ nghĩa; None nếu chưa dựng hoặc chưa cài sentence-transformers."""
    from movierec.semantic import SemanticIndex, load_encoder

    if not Path(path).exists():
        return None
    try:
        index = SemanticIndex.load(path)
        # nhúng một câu hỏi rất nhẹ nên dùng CPU, để GPU không ảnh hưởng tới độ ổn định của web
        return index.set_encoder(load_encoder(index.model_name, device="cpu"))
    except (ImportError, ValueError):  # chưa cài thư viện, hoặc chỉ mục bị hỏng
        return None


@st.cache_resource
def get_assistant(artifact: str, semantic_path: str, model: str):
    """Trợ lý hội thoại. Không có Ollama thì llm=None: vẫn tìm được phim, chỉ thiếu phần hiểu câu và lời đáp."""
    from movierec.assistant import Assistant, OllamaClient

    client = OllamaClient(model)
    return Assistant(get_service(artifact), get_semantic(semantic_path), client if client.available() else None)


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
            if m.get("poster_url"):
                st.image(m["poster_url"], width="stretch")
            st.markdown(f"**{m['title']}**")
            st.caption(m["genres"].replace("|", " · "))
            if m.get("overview"):
                st.caption(m["overview"][:140] + ("…" if len(m["overview"]) > 140 else ""))
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
    recs = svc.recommend(ratings, genres, k=12)
    if recs and recs[0].get("method") == "two_stage":
        st.caption("Mô hình đang dùng: kiến trúc hai tầng (truy hồi ứng viên + xếp hạng lại bằng LightGBM).")
    else:
        st.caption("Mô hình đang dùng: công thức trộn cho người mới. Thích từ 5 phim trở lên "
                   "(chấm 4-5 sao) để chuyển sang kiến trúc hai tầng.")
    movie_grid(recs, "rec", ratings)


def page_search(ratings: dict[int, int]) -> None:
    st.title("Tìm phim")
    query = st.text_input("Tên phim (tiếng Anh)", key="search_q", placeholder="ví dụ: matrix")
    results = svc.search(query)
    if query and not results:
        st.warning("Không tìm thấy phim nào.")
    for m in results:
        with st.container(border=True):
            poster, left, right = st.columns([1, 5, 2])
            if m.get("poster_url"):
                poster.image(m["poster_url"], width="stretch")
            left.markdown(f"**{m['title']}**")
            left.caption(m["genres"].replace("|", " · "))
            if m.get("overview"):
                left.write(m["overview"])
            with right:
                rating_box(m["item"], "search", ratings)
            with st.expander("Phim tương tự"):
                sim_cols = st.columns(6)
                for n, s in enumerate(svc.similar(m["item"], k=6)):
                    with sim_cols[n]:
                        if s.get("poster_url"):
                            st.image(s["poster_url"], width="stretch")
                        st.caption(s["title"])


def page_semantic(ratings: dict[int, int]) -> None:
    st.title("Tìm theo mô tả")
    st.write("Mô tả bộ phim bạn muốn xem bằng một câu tiếng Việt. Hệ thống so nghĩa của câu đó với nội dung "
             "từng phim, không cần trùng từ khóa hay biết tên phim.")
    index = get_semantic(SEMANTIC)
    if index is None:
        st.info("Tính năng này chưa được bật. Cài thư viện bằng `pip install sentence-transformers` "
                "rồi chạy `python scripts/build_semantic.py`, sau đó khởi động lại ứng dụng.")
        return
    query = st.text_input("Bạn muốn xem phim như thế nào?", key="semantic_q",
                          placeholder="ví dụ: phim hoạt hình vui nhộn cho trẻ em")
    hide_rated = st.checkbox("Ẩn phim tôi đã chấm", value=True, key="semantic_hide")
    if not query.strip():
        st.caption("Gợi ý: “phi hành gia bị mắc kẹt ngoài không gian”, “thám tử điều tra kẻ giết người hàng loạt”, "
                   "“chuyện tình lãng mạn trên con tàu”.")
        return
    hits = index.search(query, k=12, exclude=set(ratings) if hide_rated else None, popularity=svc.pop)
    movies = [svc._movie(i, reason=f"Độ khớp nội dung: {score:.0%}") for i, score in hits]
    movie_grid(movies, "sem", ratings)


def page_assistant(ratings: dict[int, int]) -> None:
    st.title("Trợ lý phim")
    bot = get_assistant(ARTIFACT, SEMANTIC, LLM_MODEL)
    if bot.llm is None:
        st.warning(f"Chưa kết nối được Ollama với mô hình `{LLM_MODEL}`. Trợ lý vẫn tìm phim theo mô tả, "
                   "nhưng chưa hiểu được điều kiện (thể loại, năm) và chưa viết lời giải thích. "
                   f"Hãy mở Ollama, chạy `ollama pull {LLM_MODEL}` rồi khởi động lại ứng dụng.")
    else:
        st.caption(f"Mô hình ngôn ngữ: {LLM_MODEL} (chạy cục bộ qua Ollama). Phim do hệ thống gợi ý chọn, "
                   "mô hình ngôn ngữ chỉ hiểu yêu cầu và viết lời giải thích.")
    chat = st.session_state.setdefault("chat", [])        # các lượt để hiển thị
    state = st.session_state.setdefault("chat_state", {})  # ngữ cảnh của trợ lý giữa các lượt
    if st.button("Xóa hội thoại", key="chat_clear") and chat:
        chat.clear(); state.clear()
    if not chat:
        st.caption("Thử: “phim kinh dị thập niên 90 về ngôi nhà ma ám”, “phim nào giống Toy Story”, "
                   "“gợi ý phim hài, đừng có lãng mạn”, rồi hỏi tiếp “còn phim nào khác không”.")
    for n, turn in enumerate(chat):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("movies"):
                movie_grid(turn["movies"], f"chat{n}", ratings, cols=5)
    message = st.chat_input("Bạn muốn xem phim gì?", key="chat_input")
    if message:
        with st.chat_message("user"):
            st.markdown(message)
        with st.chat_message("assistant"):
            cards = st.container()  # thẻ phim hiện ngay khi chọn xong, lời giải thích tới sau

            def show(movies: list[dict]) -> None:
                with cards:
                    movie_grid(movies, f"chat{len(chat) + 1}", ratings, cols=5)

            with st.spinner("Đang tìm phim và viết lời giải thích..."):
                out = bot.reply(message, ratings, state, on_movies=show)
        chat.append({"role": "user", "text": message})
        chat.append({"role": "assistant", "text": out["text"], "movies": out["movies"]})
        st.rerun()


def page_history(ratings: dict[int, int]) -> None:
    st.title("Phim đã chấm")
    if not ratings:
        st.info("Bạn chưa chấm phim nào.")
        return
    rows = [{"Phim": svc.movies.iloc[i]["title"], "Thể loại": svc.movies.iloc[i]["genres"].replace("|", ", "),
             "Điểm": STARS[r]} for i, r in ratings.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main() -> None:
    if "user_id" not in st.session_state:
        page_auth()
        return
    ratings = db.ratings(st.session_state.user_id)
    with st.sidebar:
        st.markdown(f"### 🎬 MovieRec\nXin chào, **{st.session_state.username}**")
        page = st.radio("Trang", ["Gợi ý cho bạn", "Tìm phim", "Tìm theo mô tả", "Trợ lý phim", "Phim đã chấm"], key="nav",
                        label_visibility="collapsed")
        st.metric("Số phim đã chấm", len(ratings))
        if st.button("Đăng xuất", key="logout_btn"):
            st.session_state.clear()
            st.rerun()
        st.caption("Poster và mô tả phim lấy từ TMDB. This product uses the TMDB API "
                   "but is not endorsed or certified by TMDB.")
    pages = {"Gợi ý cho bạn": page_home, "Tìm phim": page_search, "Tìm theo mô tả": page_semantic,
             "Trợ lý phim": page_assistant,              "Phim đã chấm": page_history}
    pages[page](ratings)


main()
