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
TMDB_PATH = os.environ.get("MOVIEREC_TMDB", str(ROOT / "data" / "processed" / "tmdb.csv"))
TRAILERS_PATH = os.environ.get("MOVIEREC_TRAILERS", str(ROOT / "data" / "processed" / "trailers.csv"))
# tài khoản quản trị: đặt tên đăng nhập trong biến môi trường, mặc định là "admin"
ADMINS = {u.strip() for u in os.environ.get("MOVIEREC_ADMINS", "admin").split(",") if u.strip()}
MIN_RATINGS = 5  # số phim nên chấm trước khi gợi ý đủ tin cậy
DIVERSITY = {"Không": 0.0, "Nhẹ": 0.2, "Vừa": 0.4, "Mạnh": 0.6}  # 1 - lam của MMR
STARS = ["Chưa chấm", "★", "★★", "★★★", "★★★★", "★★★★★"]

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
.mr-badge{position:absolute;top:6px;left:6px;background:rgba(217,72,15,.92);color:#fff;font-size:.7rem;
  padding:2px 7px;border-radius:10px}
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


@st.cache_resource
def get_extras(tmdb_path: str, trailers_path: str, m1: float, m2: float):
    """Thời lượng, điểm TMDB, đạo diễn, diễn viên, trailer theo mã phim MovieLens (chỉ để hiển thị)."""
    from movierec.extras import load_movie_extras

    return load_movie_extras(tmdb_path, trailers_path)


svc, db = get_service(ARTIFACT), get_db(DB_PATH)
BLURBS = get_blurbs(BLURBS_PATH, _mtime(BLURBS_PATH))
EXTRAS = get_extras(TMDB_PATH, TRAILERS_PATH, _mtime(TMDB_PATH), _mtime(TRAILERS_PATH))


def is_guest() -> bool:
    return st.session_state.get("guest", False)


def is_admin() -> bool:
    return not is_guest() and st.session_state.get("username") in ADMINS


# ---------- thành phần dùng chung ----------
def _save_rating(item: int, key: str) -> None:
    """st.feedback trả 0..4 (None khi bấm lại ngôi sao đang chọn để bỏ chấm); lưu thành 1..5 sao."""
    value = st.session_state[key]
    db.rate(st.session_state.user_id, item, None if value is None else value + 1)


def rating_stars(item: int, prefix: str, ratings: dict[int, int]) -> None:
    """Hàng 5 ngôi sao, bấm một lần là lưu ngay vào SQLite rồi trang tự chạy lại. Khách chưa đăng nhập: bị khóa."""
    key = f"{prefix}_{item}"
    st.feedback("stars", key=key, default=ratings[item] - 1 if item in ratings else None,
                on_change=_save_rating, args=(item, key), disabled=is_guest())


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
    badge = '<div class="mr-badge">🔖 Muốn xem</div>' if m["item"] in st.session_state.get("watch_set", set()) else ""
    poster = f'<div style="position:relative">{poster}{badge}</div>'
    return (f'{poster}<div class="mr-title" title="{e(m["title"])}">{e(m["title"])}</div>'
            f'<div class="mr-genres">{e(m["genres"].replace("|", " · "))}</div>'
            f'<div class="mr-desc">{e(description(m))}</div>{reason}')


def _info_line(movie_id: int) -> str:
    """Năm · thời lượng · điểm TMDB, từ bảng thông tin thêm (bỏ qua mục không có)."""
    if movie_id not in EXTRAS.index:
        return ""
    e = EXTRAS.loc[movie_id]
    parts = []
    if isinstance(e.get("release_date"), str) and e["release_date"]:
        parts.append(f"Khởi chiếu {e['release_date'][:4]}")
    if pd.notna(e.get("runtime")) and e["runtime"] > 0:
        parts.append(f"{int(e['runtime'])} phút")
    if pd.notna(e.get("vote_average")) and pd.notna(e.get("vote_count")) and e["vote_count"] > 0:
        parts.append(f"TMDB {e['vote_average']:.1f}/10 ({int(e['vote_count']):,} lượt)".replace(",", "."))
    return " · ".join(parts)


@st.dialog("Thông tin phim", width="large")
def movie_dialog(item: int, ratings: dict[int, int]) -> None:
    """Cửa sổ chi tiết: poster, thông tin, mô tả (tiếng Việt và tiếng Anh), trailer, chấm sao, muốn xem, phim tương tự."""
    m = svc._movie(item)
    movie_id = int(svc.movies.iloc[item]["movie_id"])
    extra = EXTRAS.loc[movie_id] if movie_id in EXTRAS.index else {}
    left, right = st.columns([1, 3])
    if m.get("poster_url"):
        left.image(m["poster_url"].replace("/w342", "/w500"), width="stretch")
    with right:
        st.subheader(m["title"])
        st.caption(" · ".join(x for x in [m["genres"].replace("|", ", "), _info_line(movie_id)] if x))
        people = []
        if isinstance(extra.get("director"), str) and extra["director"]:
            people.append(f"**Đạo diễn:** {extra['director']}")
        if isinstance(extra.get("cast"), str) and extra["cast"]:
            people.append(f"**Diễn viên:** {', '.join(extra['cast'].split('|')[:5])}")
        if people:
            st.markdown("  \n".join(people))
        if m["title"] in BLURBS:
            st.write(BLURBS[m["title"]])
        if m.get("overview"):
            st.caption(m["overview"])
        if is_guest():
            st.caption("Đăng nhập để chấm điểm và lưu phim vào danh sách muốn xem.")
        else:
            st.caption("Điểm của bạn")
            rating_stars(item, "dlg", ratings)
            watching = item in st.session_state.get("watch_set", set())
            if st.button("Bỏ khỏi danh sách muốn xem" if watching else "Thêm vào danh sách muốn xem",
                         key=f"watch_{item}", icon=":material/bookmark:"):
                db.toggle_watch(st.session_state.user_id, item)
                st.rerun()
    key = extra.get("youtube_key") if hasattr(extra, "get") else None
    if isinstance(key, str) and key:
        st.markdown("**Trailer**")
        st.video(f"https://www.youtube.com/watch?v={key}")
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


def movie_grid(movies: list[dict], prefix: str, ratings: dict[int, int], cols: int = 6) -> None:
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
        st.divider()
        if st.button("Xem thử không cần đăng nhập", key="guest_btn", width="stretch", icon=":material/visibility:"):
            st.session_state.update(guest=True, user_id=None, username="Khách")
            st.rerun()
        st.caption("Khách xem được gợi ý theo thể loại và độ phổ biến, tìm phim, tìm theo mô tả và trợ lý; "
                   "đăng nhập để chấm điểm và nhận gợi ý cá nhân hóa.")


def _save_genres() -> None:
    if not is_guest():  # khách: chỉ giữ trong phiên làm việc
        db.set_genres(st.session_state.user_id, st.session_state.genre_select)


def page_home(ratings: dict[int, int]) -> None:
    cold = len(ratings) < MIN_RATINGS
    tab_rec, tab_rate = st.tabs(["⭐ Phim dành cho bạn", "🎞️ Chấm thêm phim"], key="home_tab",
                                default="🎞️ Chấm thêm phim" if cold else "⭐ Phim dành cho bạn")

    # Phim nổi tiếng chưa chấm, mỗi lần 18 phim. Người mới dùng tab này để cho hệ thống biết sở thích.
    with tab_rate:
        if is_guest():
            st.info("Bạn đang xem thử với tư cách khách. Đăng nhập (nút ở thanh bên) để chấm điểm các phim dưới đây.")
        elif cold:
            st.progress(len(ratings) / MIN_RATINGS,
                        text=f"Đã chấm {len(ratings)}/{MIN_RATINGS} phim. Chấm những phim bạn đã xem để nhận gợi ý "
                             "cá nhân hóa; thích từ 5 phim (4-5 sao) thì hệ thống chuyển sang kiến trúc hai tầng.")
        else:
            st.caption("Phim nổi tiếng bạn chưa chấm. Chấm càng nhiều, gợi ý càng sát sở thích.")
        page = st.session_state.setdefault("rate_page", 0)
        pool = [m for m in svc.popular(300) if m["item"] not in ratings]
        to_rate = pool[page * 18:(page + 1) * 18] or pool[:18]
        movie_grid(to_rate, "onb", ratings)
        if st.button("Xem phim khác", key="more_to_rate", icon=":material/refresh:"):
            st.session_state.rate_page = (page + 1) % max(1, len(pool) // 18)
            st.rerun()

    with tab_rec:
        genres = (st.session_state.get("genre_select", []) if is_guest()
                  else db.get_genres(st.session_state.user_id))
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
        recs = [m for m in svc.recommend(ratings, genres, k=18 + len(shown), diversity=DIVERSITY[level or "Nhẹ"])
                if m["item"] not in shown][:18]
        if recs and recs[0].get("method") == "two_stage":
            st.caption("Mô hình đang dùng: kiến trúc hai tầng (truy hồi ứng viên + xếp hạng lại bằng LightGBM). "
                       "Bấm “Vì sao?” để xem yếu tố nào đưa phim lên.")
        else:
            st.caption("Mô hình đang dùng: công thức trộn cho người mới. Thích từ 5 phim trở lên "
                       "(chấm 4-5 sao) để chuyển sang kiến trúc hai tầng.")
        movie_grid(recs, "rec", ratings)


def page_search(ratings: dict[int, int]) -> None:
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
                movie_grid(svc.similar(m["item"], k=6), f"sim{m['item']}", ratings)


def page_semantic(ratings: dict[int, int]) -> None:
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
    hits = index.search(query, k=18, exclude=set(ratings) if hide_rated else None, popularity=svc.pop)
    st.caption("Thứ tự kết hợp độ khớp nội dung với độ phổ biến của phim (phim quen thuộc hơn được ưu tiên khi nội dung "
               "gần giống nhau), nên đôi khi phim có độ khớp thấp hơn một chút vẫn đứng trước.")
    movies = [svc._movie(i, reason=f"Độ khớp nội dung: {score:.0%}") for i, score in hits]
    movie_grid(movies, "sem", ratings)


def page_assistant(ratings: dict[int, int]) -> None:
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
    tab_rated, tab_watch = st.tabs([f"Đã chấm ({len(ratings)})",
                                    f"Muốn xem ({len(st.session_state.watch_set)})"])
    with tab_watch:
        watch = db.watchlist(st.session_state.user_id)
        if not watch:
            st.info("Chưa có phim nào. Mở “Chi tiết” của một phim rồi bấm “Thêm vào danh sách muốn xem”.")
        else:
            st.caption("Phim bạn đã lưu để xem sau, mới lưu nằm trước.")
            movie_grid([svc._movie(i) for i in watch], "watch", ratings)
    with tab_rated:
        _rated_tab(ratings)


def _rated_tab(ratings: dict[int, int]) -> None:
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
    movie_grid([svc._movie(i) for i in items], "hist", ratings)


def page_admin(ratings: dict[int, int]) -> None:
    """Trang quản trị: số liệu sử dụng, người dùng, tình trạng các thành phần, nạp lại mô hình, xuất dữ liệu."""
    stats = db.stats()
    c = st.columns(5)
    c[0].metric("Người dùng", stats["users"])
    c[1].metric("Lượt chấm điểm", stats["ratings"])
    c[2].metric("Điểm trung bình", "–" if stats["avg_rating"] is None else f"{stats['avg_rating']:.2f}".replace(".", ","))
    c[3].metric("Hoạt động 7 ngày qua", stats["active_7d"])
    c[4].metric("Lượt lưu muốn xem", stats["watchlist"])

    all_r = pd.DataFrame(db.all_ratings(), columns=["user_id", "item", "rating", "rated_at"])
    tab_use, tab_users, tab_sys = st.tabs(["Hoạt động", "Người dùng", "Hệ thống"])
    with tab_use:
        if all_r.empty:
            st.info("Chưa có lượt chấm điểm nào trên ứng dụng.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Phân bố điểm**")
                dist = all_r["rating"].value_counts().reindex(range(1, 6), fill_value=0)
                st.bar_chart(pd.DataFrame({"Số lượt": dist.to_numpy()}, index=[f"{k} sao" for k in range(1, 6)]),
                             color="#d9480f", height=260)
            with right:
                st.markdown("**Phim được chấm nhiều nhất**")
                top = all_r.groupby("item")["rating"].agg(["count", "mean"]).sort_values("count", ascending=False)[:10]
                st.dataframe(pd.DataFrame({"Phim": [svc.movies.iloc[i]["title"] for i in top.index],
                                           "Lượt chấm": top["count"].to_numpy(),
                                           "Điểm TB": top["mean"].round(2).to_numpy()}),
                             hide_index=True, width="stretch")
            st.download_button("Tải toàn bộ lượt chấm (CSV)", all_r.to_csv(index=False).encode("utf-8"),
                               "movierec_ratings.csv", "text/csv", icon=":material/download:")
    with tab_users:
        users = pd.DataFrame(db.users_overview())
        for col in ("avg_rating", "last_rated"):  # người chưa chấm phim nào: SQLite trả None
            users[col] = pd.to_numeric(users[col], errors="coerce")
        show = users.assign(
            created_at=pd.to_datetime(users["created_at"], unit="s").dt.strftime("%d/%m/%Y"),
            avg_rating=users["avg_rating"].round(2),
            last_rated=pd.to_datetime(users["last_rated"], unit="s").dt.strftime("%d/%m/%Y %H:%M"))
        st.dataframe(show.rename(columns={"id": "ID", "username": "Tên đăng nhập", "created_at": "Ngày tạo",
                                          "n_ratings": "Số phim đã chấm", "avg_rating": "Điểm TB",
                                          "last_rated": "Lần chấm gần nhất"}), hide_index=True, width="stretch")
        others = users[users["id"] != st.session_state.user_id]
        if not others.empty:
            with st.form("delete_user_form"):
                who = st.selectbox("Xóa tài khoản (kèm toàn bộ lượt chấm và danh sách muốn xem)",
                                   others["id"], format_func=lambda i: users.set_index("id").loc[i, "username"])
                sure = st.checkbox("Tôi chắc chắn muốn xóa tài khoản này")
                if st.form_submit_button("Xóa tài khoản", type="primary") and sure:
                    db.delete_user(int(who))
                    st.rerun()
    with tab_sys:
        semantic_ok = Path(SEMANTIC).exists()
        trailers = int(EXTRAS["youtube_key"].notna().sum()) if "youtube_key" in EXTRAS else 0
        from movierec.assistant import OllamaClient

        rows = [("Số phim trong kho", f"{len(svc.movies):,}".replace(",", ".")),
                ("Bộ xếp hạng hai tầng", "đã nạp" if svc.ranker is not None else "chưa có (chỉ dùng công thức trộn)"),
                ("Mô hình gợi ý cập nhật lúc", pd.Timestamp(_mtime(ARTIFACT), unit="s").strftime("%d/%m/%Y %H:%M")
                 if _mtime(ARTIFACT) else "–"),
                ("Chỉ mục tìm theo mô tả", "có" if semantic_ok else "chưa dựng (scripts/build_semantic.py)"),
                ("Lời giới thiệu tiếng Việt", f"{len(BLURBS):,} phim".replace(",", ".")),
                ("Trailer", f"{trailers:,} phim".replace(",", ".") if trailers else "chưa lấy (scripts/fetch_trailers.py)"),
                ("Mô hình ngôn ngữ (Ollama)", f"{LLM_MODEL}: " + ("đang chạy" if OllamaClient(LLM_MODEL).available()
                                                                   else "chưa kết nối"))]
        st.dataframe(pd.DataFrame(rows, columns=["Thành phần", "Tình trạng"]), hide_index=True, width="stretch")
        st.caption("Sau khi chạy lại các tập lệnh huấn luyện hoặc lấy dữ liệu, bấm nút dưới đây để ứng dụng nạp phiên "
                   "bản mới mà không cần khởi động lại.")
        if st.button("Nạp lại mô hình và dữ liệu", icon=":material/refresh:", key="reload_btn"):
            st.cache_resource.clear()
            st.rerun()


def _header(title: str, ratings: dict[int, int]) -> None:
    """Dòng đầu trang: tiêu đề bên trái, menu tài khoản bên phải (thay cho thanh bên)."""
    liked = sum(r >= 4 for r in ratings.values())
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.title(title)
    with right:
        if is_guest():
            if st.button("Đăng nhập", key="login_from_guest", icon=":material/login:", type="primary", width="stretch"):
                st.session_state.clear()
                st.rerun()
            return
        with st.popover(st.session_state.username, icon=":material/account_circle:", width="stretch"):
            st.metric("Số phim đã chấm", len(ratings))
            if liked >= MIN_RATINGS:
                st.caption("Chế độ gợi ý: **kiến trúc hai tầng**")
            else:
                st.caption(f"Chế độ gợi ý: **người mới**. Thích thêm {MIN_RATINGS - liked} phim (4-5 sao) "
                           "để dùng kiến trúc hai tầng.")
            if st.button("Đăng xuất", key="logout_btn", icon=":material/logout:", width="stretch"):
                st.session_state.clear()
                st.rerun()


def main() -> None:
    st.logo(str(ROOT / "app" / "assets" / "logo.svg"), size="large")
    st.markdown(CSS, unsafe_allow_html=True)
    if "user_id" not in st.session_state:
        page_auth()
        return
    guest = is_guest()
    ratings = {} if guest else db.ratings(st.session_state.user_id)
    st.session_state.watch_set = set() if guest else set(db.watchlist(st.session_state.user_id))

    # Thanh điều hướng ngang ở đầu trang. Mỗi trang là một hàm không tham số bọc hàm page_* tương ứng.
    specs = [(page_home, "Gợi ý cho bạn", ":material/recommend:", "goi-y"),
             (page_search, "Tìm phim", ":material/search:", "tim-phim"),
             (page_semantic, "Tìm theo mô tả", ":material/manage_search:", "tim-theo-mo-ta"),
             (page_assistant, "Trợ lý phim", ":material/smart_toy:", "tro-ly")]
    if not guest:
        specs.append((page_history, "Phim của tôi", ":material/bookmarks:", "phim-cua-toi"))
    if is_admin():
        specs.append((page_admin, "Quản trị", ":material/admin_panel_settings:", "quan-tri"))
    # trang mở khi vào địa chỉ gốc; start_page trong session_state cho phép mở thẳng một trang (dùng khi kiểm thử)
    start = st.session_state.get("start_page", "goi-y")
    start = start if start in {url for *_, url in specs} else "goi-y"

    def page(fn, title, icon, url):
        run = lambda: fn(ratings)  # noqa: E731
        run.__name__ = url.replace("-", "_")
        return st.Page(run, title=title, icon=icon, url_path=url, default=url == start)

    current = st.navigation([page(*spec) for spec in specs], position="top")
    _header("Quản trị hệ thống" if current.title == "Quản trị" else current.title, ratings)
    current.run()
    st.divider()
    st.caption("Poster và mô tả phim lấy từ TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")

main()
