"""Web demo: đăng ký/đăng nhập, chấm điểm phim, nhận gợi ý cá nhân hóa.

Chạy:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
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
BLURBS_PATH = os.environ.get("MOVIEREC_BLURBS", str(ROOT / "artifacts" / "blurbs_vi.json"))
MIN_RATINGS = 5  # số phim nên chấm trước khi gợi ý đủ tin cậy
DIVERSITY = {"Không": 0.0, "Nhẹ": 0.2, "Vừa": 0.4, "Mạnh": 0.6}  # 1 - lam của MMR
STARS = ["Chưa chấm", "★", "★★", "★★★", "★★★★", "★★★★★"]
PAGES = ["⭐ Gợi ý cho bạn", "🔎 Tìm phim", "💬 Tìm theo mô tả", "🤖 Trợ lý phim", "🎞️ Phim đã chấm"]

# Thẻ phim cùng kích thước: poster tỉ lệ 2:3, tên tối đa 2 dòng, mô tả tối đa 3 dòng, để lưới phim thẳng hàng
CSS = """<style>
[data-testid="stMainBlockContainer"]{padding-top:2.2rem}
.mr-poster{width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:8px;display:block;background:#eceae6}
.mr-noposter{width:100%;aspect-ratio:2/3;border-radius:8px;background:#eceae6;display:flex;align-items:center;
  justify-content:center;color:#8a8780;font-size:2rem}
.mr-title{font-weight:600;font-size:.92rem;line-height:1.25;margin-top:.5rem;height:2.5em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.mr-genres{color:#6b6963;font-size:.74rem;margin:.2rem 0 .35rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mr-desc{color:#47453f;font-size:.79rem;line-height:1.35;height:4.05em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
.mr-reason{color:#b4540c;font-size:.74rem;margin-top:.35rem;height:1.3em;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
</style>"""

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


def _mtime(path: str) -> float:
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


@st.cache_resource
def get_blurbs(path: str, mtime: float) -> dict[str, str]:
    """Câu giới thiệu tiếng Việt viết sẵn (scripts/build_blurbs.py). Tham số mtime: tệp được cập nhật thì tự nạp lại."""
    from movierec.assistant import load_blurbs

    return load_blurbs(path)


@st.cache_resource
def get_assistant(artifact: str, semantic_path: str, model: str, blurbs_path: str, blurbs_mtime: float):
    """Trợ lý hội thoại. Không có Ollama thì llm=None: vẫn tìm được phim, chỉ thiếu phần hiểu câu và lời đáp."""
    from movierec.assistant import Assistant, OllamaClient

    client = OllamaClient(model)
    return Assistant(get_service(artifact), get_semantic(semantic_path), client if client.available() else None,
                     blurbs=get_blurbs(blurbs_path, blurbs_mtime))


svc, db = get_service(ARTIFACT), get_db(DB_PATH)
BLURBS = get_blurbs(BLURBS_PATH, _mtime(BLURBS_PATH))


# ---------- thành phần dùng chung ----------
def _save_rating(item: int, key: str) -> None:
    """st.feedback trả 0..4 (None khi bấm lại ngôi sao đang chọn để bỏ chấm); lưu thành 1..5 sao."""
    value = st.session_state[key]
    db.rate(st.session_state.user_id, item, None if value is None else value + 1)


def rating_stars(item: int, prefix: str, ratings: dict[int, int]) -> None:
    """Hàng 5 ngôi sao, bấm một lần là lưu ngay vào SQLite rồi trang tự chạy lại."""
    key = f"{prefix}_{item}"
    st.feedback("stars", key=key, default=ratings[item] - 1 if item in ratings else None,
                on_change=_save_rating, args=(item, key))


def why_box(factors: list[tuple[str, float]]) -> None:
    """Các nhóm đặc trưng ảnh hưởng mạnh nhất đến điểm của bộ xếp hạng (giá trị SHAP). Thanh dài theo độ lớn."""
    top = max(abs(v) for _, v in factors)
    for label, v in factors:
        bar = "█" * max(1, round(6 * abs(v) / top))
        color = "green" if v > 0 else "red"
        st.markdown(f"<span style='font-size:0.8rem'>{'▲' if v > 0 else '▼'} {label}<br>"
                    f"<span style='color:{color}'>{bar}</span> {v:+.2f}</span>", unsafe_allow_html=True)
    st.caption("Giá trị SHAP của bộ xếp hạng LightGBM: dương là đẩy phim lên, âm là kéo xuống.")


def description(m: dict) -> str:
    """Câu giới thiệu tiếng Việt viết sẵn nếu có, không thì mô tả gốc tiếng Anh."""
    return BLURBS.get(m["title"]) or str(m.get("overview") or "")


def card_html(m: dict) -> str:
    e = html.escape
    poster = (f'<img class="mr-poster" src="{e(m["poster_url"])}" alt="" loading="lazy">' if m.get("poster_url")
              else '<div class="mr-noposter">🎬</div>')
    reason = f'<div class="mr-reason">💡 {e(m["reason"])}</div>' if m.get("reason") else ""
    return (f'{poster}<div class="mr-title" title="{e(m["title"])}">{e(m["title"])}</div>'
            f'<div class="mr-genres">{e(m["genres"].replace("|", " · "))}</div>'
            f'<div class="mr-desc">{e(description(m))}</div>{reason}')


@st.dialog("Thông tin phim", width="large")
def movie_dialog(item: int, ratings: dict[int, int]) -> None:
    """Cửa sổ chi tiết: poster lớn, mô tả đầy đủ (tiếng Việt và tiếng Anh), chấm sao, phim tương tự."""
    m = svc._movie(item)
    left, right = st.columns([1, 3])
    if m.get("poster_url"):
        left.image(m["poster_url"].replace("/w342", "/w500"), width="stretch")
    with right:
        st.subheader(m["title"])
        st.caption(m["genres"].replace("|", " · "))
        if m["title"] in BLURBS:
            st.write(BLURBS[m["title"]])
        if m.get("overview"):
            st.caption(m["overview"])
        st.caption("Điểm của bạn")
        rating_stars(item, "dlg", ratings)
    st.markdown("**Phim tương tự**")
    cols = st.columns(6)
    for col, s in zip(cols, svc.similar(item, k=6)):
        with col:
            st.markdown(card_html({**s, "overview": ""}).split('<div class="mr-desc">')[0], unsafe_allow_html=True)


def movie_card(m: dict, prefix: str, ratings: dict[int, int]) -> None:
    with st.container(border=True):
        st.markdown(card_html(m), unsafe_allow_html=True)
        rating_stars(m["item"], prefix, ratings)
        with st.container(horizontal=True, gap="small"):
            if st.button("Chi tiết", key=f"{prefix}_info_{m['item']}", type="tertiary"):
                movie_dialog(m["item"], ratings)
            if m.get("why"):
                with st.popover("Vì sao?", type="tertiary"):
                    why_box(m["why"])


def movie_grid(movies: list[dict], prefix: str, ratings: dict[int, int], cols: int = 5) -> None:
    for start in range(0, len(movies), cols):  # mỗi hàng một st.columns riêng để các thẻ cùng hàng thẳng nhau
        for col, m in zip(st.columns(cols), movies[start:start + cols]):
            with col:
                movie_card(m, prefix, ratings)


# ---------- các trang ----------
def page_auth() -> None:
    _, mid, _ = st.columns([1, 1.25, 1])  # form hẹp ở giữa trang thay vì kéo dài hết chiều ngang
    with mid:
        st.title("🎬 MovieRec")
        st.write("Hệ thống gợi ý phim cá nhân hóa dựa trên học máy, trên 3.706 phim của bộ dữ liệu MovieLens 1M.")
        st.caption("⭐ Gợi ý theo sở thích, có giải thích lý do · 💬 Tìm phim bằng câu tiếng Việt · "
                   "🤖 Trợ lý hội thoại chạy ngay trên máy")
        tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
        # st.form: gửi cả tên và mật khẩu cùng lúc khi bấm nút, kể cả khi ô nhập chưa mất tiêu điểm
        with tab_login, st.form("login_form"):
            user = st.text_input("Tên đăng nhập", key="login_user")
            pw = st.text_input("Mật khẩu", type="password", key="login_pw")
            if st.form_submit_button("Đăng nhập", key="login_btn", type="primary", width="stretch"):
                uid = db.login(user, pw)
                if uid is None:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")
                else:
                    st.session_state.update(user_id=uid, username=user.strip())
                    st.rerun()
        with tab_register, st.form("register_form"):
            user = st.text_input("Tên đăng nhập", key="reg_user")
            pw = st.text_input("Mật khẩu", type="password", key="reg_pw")
            if st.form_submit_button("Tạo tài khoản", key="reg_btn", type="primary", width="stretch"):
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
    cold = len(ratings) < MIN_RATINGS
    tab_rec, tab_rate = st.tabs(["⭐ Phim dành cho bạn", "🎞️ Chấm thêm phim"], key="home_tab",
                                default="🎞️ Chấm thêm phim" if cold else "⭐ Phim dành cho bạn")

    # Phim nổi tiếng chưa chấm, mỗi lần 15 phim. Người mới dùng tab này để cho hệ thống biết sở thích.
    with tab_rate:
        if cold:
            st.progress(len(ratings) / MIN_RATINGS,
                        text=f"Đã chấm {len(ratings)}/{MIN_RATINGS} phim. Chấm những phim bạn đã xem để nhận gợi ý "
                             "cá nhân hóa; thích từ 5 phim (4-5 sao) thì hệ thống chuyển sang kiến trúc hai tầng.")
        else:
            st.caption("Phim nổi tiếng bạn chưa chấm. Chấm càng nhiều, gợi ý càng sát sở thích.")
        page = st.session_state.setdefault("rate_page", 0)
        pool = [m for m in svc.popular(300) if m["item"] not in ratings]
        to_rate = pool[page * 15:(page + 1) * 15] or pool[:15]
        movie_grid(to_rate, "onb", ratings)
        if st.button("Xem phim khác", key="more_to_rate", icon=":material/refresh:"):
            st.session_state.rate_page = (page + 1) % max(1, len(pool) // 15)
            st.rerun()

    with tab_rec:
        genres = db.get_genres(st.session_state.user_id)
        left, right = st.columns([3, 2], vertical_alignment="bottom")
        left.multiselect("Thể loại yêu thích", svc.genres, default=genres, key="genre_select",
                         on_change=_save_genres, placeholder="Chọn vài thể loại bạn thích")
        # Mức đa dạng = 1 - lam của MMR. Khi đánh giá trên tập test, "Nhẹ" (lam 0,8) chỉ giảm NDCG@10 của kiến trúc
        # hai tầng 0,6% nhưng thêm 0,8 thể loại cho mỗi danh sách 10 phim (scripts/beyond_accuracy.py)
        level = right.segmented_control("Độ đa dạng thể loại", list(DIVERSITY), default="Nhẹ", key="diversity",
                                        required=True,
                                        help="Cao hơn: danh sách trải đều nhiều thể loại hơn, đổi lại độ chính xác "
                                             "giảm nhẹ.")
        # Người mới: gợi ý chủ yếu là phim phổ biến, nên bỏ các phim đang hiện ở tab chấm phim cho khỏi lặp
        shown = {m["item"] for m in to_rate} if cold else set()
        recs = [m for m in svc.recommend(ratings, genres, k=15 + len(shown), diversity=DIVERSITY[level or "Nhẹ"])
                if m["item"] not in shown][:15]
        if recs and recs[0].get("method") == "two_stage":
            st.caption("Mô hình đang dùng: kiến trúc hai tầng (truy hồi ứng viên + xếp hạng lại bằng LightGBM). "
                       "Bấm “Vì sao?” để xem yếu tố nào đưa phim lên.")
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
            poster, body = st.columns([1, 6])
            if m.get("poster_url"):
                poster.image(m["poster_url"], width="stretch")
            with body:
                st.markdown(f"**{m['title']}**")
                st.caption(m["genres"].replace("|", " · "))
                if m["title"] in BLURBS:
                    st.write(BLURBS[m["title"]])
                if m.get("overview"):
                    st.caption(m["overview"])
                rating_stars(m["item"], "search", ratings)
            with st.expander("Phim tương tự"):
                movie_grid(svc.similar(m["item"], k=6), f"sim{m['item']}", ratings, cols=6)


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
    hits = index.search(query, k=15, exclude=set(ratings) if hide_rated else None, popularity=svc.pop)
    st.caption("Thứ tự kết hợp độ khớp nội dung với độ phổ biến của phim (phim quen thuộc hơn được ưu tiên khi nội dung "
               "gần giống nhau), nên đôi khi phim có độ khớp thấp hơn một chút vẫn đứng trước.")
    movies = [svc._movie(i, reason=f"Độ khớp nội dung: {score:.0%}") for i, score in hits]
    movie_grid(movies, "sem", ratings)


def page_assistant(ratings: dict[int, int]) -> None:
    st.title("Trợ lý phim")
    bot = get_assistant(ARTIFACT, SEMANTIC, LLM_MODEL, BLURBS_PATH, _mtime(BLURBS_PATH))
    if bot.llm is None:  # Ollama có thể được mở sau khi web đã chạy: thử kết nối lại, không cần khởi động lại web
        from movierec.assistant import OllamaClient

        client = OllamaClient(LLM_MODEL)
        if client.available():
            bot.llm = client
    if bot.llm is None:
        st.warning(f"Chưa kết nối được Ollama với mô hình `{LLM_MODEL}`. Trợ lý vẫn tìm phim theo mô tả, "
                   "nhưng chưa hiểu được điều kiện (thể loại, năm) và chưa viết lời giải thích. "
                   f"Hãy mở Ollama (và chạy `ollama pull {LLM_MODEL}` nếu chưa tải mô hình), rồi tải lại trang này.")
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
        st.info("Bạn chưa chấm phim nào. Vào trang “Gợi ý cho bạn”, tab “Chấm thêm phim” để bắt đầu.")
        return
    genre_count = pd.Series([g for i in ratings for g in svc.movies.iloc[i]["genres"].split("|")]).value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số phim đã chấm", len(ratings))
    c2.metric("Điểm trung bình", f"{sum(ratings.values()) / len(ratings):.1f} ★".replace(".", ","))
    c3.metric("Số phim thích (4-5 sao)", sum(r >= 4 for r in ratings.values()))
    c4.metric("Thể loại chấm nhiều nhất", genre_count.index[0])
    order = st.segmented_control("Sắp xếp", ["Điểm cao trước", "Điểm thấp trước", "Tên phim"],
                                 default="Điểm cao trước", key="hist_order", required=True)
    items = list(ratings)
    if order == "Tên phim":
        items.sort(key=lambda i: svc.movies.iloc[i]["title"])
    else:
        items.sort(key=lambda i: ratings[i], reverse=order != "Điểm thấp trước")
    st.caption("Bấm vào ngôi sao để chấm lại; bấm lại đúng ngôi sao đang chọn để bỏ chấm phim đó.")
    movie_grid([svc._movie(i) for i in items], "hist", ratings, cols=6)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "user_id" not in st.session_state:
        page_auth()
        return
    ratings = db.ratings(st.session_state.user_id)
    liked = sum(r >= 4 for r in ratings.values())
    with st.sidebar:
        st.markdown(f"### 🎬 MovieRec\nXin chào, **{st.session_state.username}**")
        page = st.radio("Trang", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.metric("Số phim đã chấm", len(ratings))
        if liked >= MIN_RATINGS:
            st.caption("Chế độ gợi ý: **kiến trúc hai tầng**")
        else:
            st.caption(f"Chế độ gợi ý: **người mới**. Thích thêm {MIN_RATINGS - liked} phim (4-5 sao) "
                       "để dùng kiến trúc hai tầng.")
        if st.button("Đăng xuất", key="logout_btn", icon=":material/logout:"):
            st.session_state.clear()
            st.rerun()
        st.caption("Poster và mô tả phim lấy từ TMDB. This product uses the TMDB API "
                   "but is not endorsed or certified by TMDB.")
    pages = dict(zip(PAGES, [page_home, page_search, page_semantic, page_assistant, page_history]))
    pages[page](ratings)


main()
