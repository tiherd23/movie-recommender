"""Web demo: đăng ký/đăng nhập, chấm điểm phim, nhận gợi ý cá nhân hóa.

Chạy:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import math
import os
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec import session as sess  # noqa: E402
from movierec.db import Database  # noqa: E402
from movierec.ranker import FEATURE_GROUPS  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, GENRE_SHARE, GENRE_VI, display_title, genre_vi, load_or_build  # noqa: E402

DB_PATH = os.environ.get("MOVIEREC_DB", str(ROOT / "data" / "app.db"))
ARTIFACT = os.environ.get("MOVIEREC_ARTIFACT", str(DEFAULT_ARTIFACT))
SEMANTIC = os.environ.get("MOVIEREC_SEMANTIC", str(ROOT / "artifacts" / "semantic.npz"))
LLM_MODEL = os.environ.get("MOVIEREC_LLM", "qwen3:4b")
BLURBS_PATH = os.environ.get("MOVIEREC_BLURBS", str(ROOT / "artifacts" / "blurbs_vi.json"))
TMDB_PATH = os.environ.get("MOVIEREC_TMDB", str(ROOT / "data" / "processed" / "tmdb.csv"))
TRAILERS_PATH = os.environ.get("MOVIEREC_TRAILERS", str(ROOT / "data" / "processed" / "trailers.csv"))
# tài khoản quản trị: đặt tên đăng nhập trong biến môi trường, mặc định là "admin"
ADMINS = {u.strip() for u in os.environ.get("MOVIEREC_ADMINS", "admin").split(",") if u.strip()}
MIN_RATINGS = 5  # số phim nên thích (4-5★) khi mới dùng: độ chính xác tăng nhanh nhất ở 5 phim đầu (thí nghiệm cold-start)
DIVERSITY = {"Không": 0.0, "Nhẹ": 0.2, "Vừa": 0.4, "Mạnh": 0.6}  # 1 - lam của MMR
DEFAULT_DIVERSITY = "Không"  # mức đã dùng khi đo độ chính xác của công thức trộn (thí nghiệm người dùng mới)
DIVERSITY_HELP = ("Trộn thêm phim thuộc thể loại khác vào danh sách để đỡ nhàm. **Không** (mặc định): chỉ xếp theo "
                  "mức độ hợp với bạn. **Nhẹ**, **Vừa**, **Mạnh**: đa dạng dần, đổi lại có thể có thêm vài phim ít "
                  "hợp hơn.")
# Số liệu đo trên tập kiểm tra (results/mmr.csv, scripts/beyond_accuracy.py): đo với KIẾN TRÚC HAI TẦNG, chưa đo với công thức trộn
DIVERSITY_NOTE = ("Độ đa dạng (xếp lại 50 phim điểm cao nhất bằng MMR) mới được đo với kiến trúc hai tầng trên tập kiểm "
                  "tra, so với mức “Không”: Nhẹ thêm 0,8 thể loại mỗi danh sách 10 phim mà NDCG@10 không giảm; Vừa thêm "
                  "1,8 thể loại, giảm 2,8%; Mạnh thêm 2,8 thể loại, giảm 8,1%. Với công thức trộn (mặc định trên web), "
                  "ảnh hưởng của độ đa dạng chưa được đo.")
# Lựa chọn "Kiểu gợi ý" (đưa ra SAU thí nghiệm người dùng mới, là quyết định về sản phẩm): mặc định là công thức được
# chọn theo quy tắc đặt trước (120/n). Mức "Sát gu nhất" là kiến trúc hai tầng: kém hơn về NDCG@10 nhưng danh sách bám
# thể loại của hồ sơ hơn hẳn (scripts/taste_check.py, results/taste_check.csv).
STYLES = {"Phổ biến": ("blend", None), "Cân bằng": ("blend", 60.0), "Sát gu nhất": ("two_stage", None)}  # None: POP_C
DEFAULT_STYLE = "Phổ biến"
STYLE_KEY = {"Phổ biến": "Trộn, w(n)=120·n^-1", "Cân bằng": "Trộn, w(n)=60·n^-1",
             "Sát gu nhất": "Hai tầng (học cả hồ sơ ngắn)"}
STYLE_HELP = ("Khi mới thích vài phim, hệ thống còn biết ít về bạn nên dựa nhiều vào độ phổ biến. **Phổ biến** (mặc "
              "định): chính xác nhất trong thí nghiệm người dùng mới, nhưng danh sách có nhiều phim nổi tiếng ngoài gu "
              "của bạn. **Cân bằng**: giảm một nửa trọng số độ phổ biến, độ chính xác gần như bằng. **Sát gu nhất**: "
              "kiến trúc hai tầng (LightGBM), danh sách bám thể loại bạn thích nhất nhưng độ chính xác thấp hơn "
              "khoảng 13% trong thí nghiệm.")
RESULTS_PATH = os.environ.get("MOVIEREC_RESULTS", str(ROOT / "results" / "summary.csv"))
COLD_PATH = str(ROOT / "results" / "cold_start_pooled.csv")  # thí nghiệm người dùng mới (scripts/cold_start*.py)
COLD_LEVELS = [1, 2, 3, 5, 7, 10, 15, 20, 30]
# tên mô hình trong results/cold_start_pooled.csv -> tên hiển thị
COLD_MODELS = {"Trộn, w(n)=120·n^-1": "Công thức trộn, trọng số phổ biến 120/n (mặc định)",
               "Trộn, w(n)=60·n^-1": "Công thức trộn, trọng số phổ biến 60/n (“Cân bằng”)",
               "Hai tầng (học cả hồ sơ ngắn)": "Hai tầng, học cả hồ sơ ngắn (“Sát gu nhất”)",
               "Công thức trộn": "Công thức trộn, trọng số cố định",
               "Hai tầng": "Hai tầng, bản của thực nghiệm chính",
               "Implicit ALS": "Implicit ALS",
               "Phổ biến nhất": "Phổ biến nhất"}
# Tên tiếng Việt và mô tả ngắn của các mô hình trong bảng kết quả thực nghiệm (results/summary.csv)
MODEL_VI = {
    "TwoStage": "Hai tầng (ALS + ItemKNN + nội dung + phổ biến → LightGBM)",
    "ImplicitALS": "Implicit ALS (phân rã ma trận)",
    "Hybrid": "Công thức trộn (ALS + ItemKNN + nội dung + phổ biến)",
    "LightGCN": "LightGCN (mạng nơ-ron đồ thị)",
    "NeuMF": "NeuMF (mạng nơ-ron)",
    "ItemKNN": "ItemKNN (phim tương tự)",
    "Popularity": "Phổ biến nhất",
    "SVD": "SVD (dự đoán số sao)",
    "BiasBaseline": "Điểm trung bình có hiệu chỉnh",
    "ContentBased": "Theo nội dung (thể loại, năm)",
    "SASRec": "SASRec (Transformer theo thứ tự chấm)",
}
TZ = "Asia/Ho_Chi_Minh"
SESSION_KEY_PATH = os.environ.get("MOVIEREC_SESSION_KEY", str(Path(DB_PATH).parent / ".session_key"))
PAGE_URLS = {"goi-y", "tim-phim", "tim-theo-mo-ta", "tro-ly", "phim-cua-toi", "quan-tri"}
STARS = ["Chưa chấm", "★", "★★", "★★★", "★★★★", "★★★★★"]

# Thẻ phim cùng kích thước: poster tỉ lệ 2:3, tên tối đa 2 dòng, mô tả tối đa 3 dòng, để lưới phim thẳng hàng
CSS = """<style>
[data-testid="stMainBlockContainer"]{padding-top:3.3rem}
.mr-poster{width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:8px;display:block;background:#2c3440;box-shadow:0 0 0 1px rgba(255,255,255,.08) inset}
.mr-noposter{width:100%;aspect-ratio:2/3;border-radius:8px;background:#2c3440;display:flex;align-items:center;
  justify-content:center;color:#678;font-size:2rem}
.mr-poster:hover{transform:translateY(-2px);box-shadow:0 0 0 2px #ff8000;transition:all .15s}
.mr-title{font-weight:600;font-size:.92rem;line-height:1.25;margin-top:.5rem;height:2.5em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.mr-genres{color:#789;font-size:.74rem;margin:.2rem 0 .35rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mr-stats{color:#9ab;font-size:.74rem;margin:-.2rem 0 .35rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mr-desc{color:#9ab;font-size:.79rem;line-height:1.35;height:4.05em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
.mr-badge{position:absolute;top:6px;left:6px;background:rgba(255,128,0,.95);color:#fff;font-size:.7rem;
  padding:2px 7px;border-radius:10px}
[data-testid="stElementToolbar"]{display:none}
[role="listbox"][data-empty="true"] [role="option"] span{font-size:0}
[role="listbox"][data-empty="true"] [role="option"] span::after{content:"Không tìm thấy";font-size:.9rem}
.mr-change{position:absolute;top:6px;right:6px;background:#00a83a;color:#fff;font-size:.7rem;font-weight:600;
  padding:2px 7px;border-radius:10px}
.mr-change.up{background:#2f6fd6}
.mr-reason{color:#ff9d3c;font-size:.74rem;line-height:1.3;margin-top:.35rem;height:2.6em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
/* phần tử chỉ chứa CSS/JS (cao 0) vẫn chiếm một khoảng cách giữa các phần tử: ẩn hẳn phần bao ngoài */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style),
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > script),
[data-testid="stLayoutWrapper"]:has(> .st-key-mobile_nav){display:none!important}
/* màn hình vừa (laptop nhỏ, máy tính bảng ngang): 6 cột hẹp, thu nhỏ hàng sao để không tràn ra ngoài thẻ */
@media (min-width: 641px) and (max-width: 1280px){
  [data-testid="stColumn"]:has(.mr-poster, .mr-noposter) [data-testid="stFeedback"]{overflow:hidden}
  [data-testid="stColumn"]:has(.mr-poster, .mr-noposter) [data-testid="stFeedback"] button{padding:0!important;min-width:0!important}
  [data-testid="stColumn"]:has(.mr-poster, .mr-noposter) [data-testid="stFeedback"] svg,
  [data-testid="stColumn"]:has(.mr-poster, .mr-noposter) [data-testid="stFeedback"] span{font-size:1rem!important;
    width:1rem!important;height:1rem!important}
}
/* màn hình hẹp (điện thoại): lưới phim 2 cột thay vì mỗi thẻ chiếm cả màn hình */
@media (max-width: 640px){
  [data-testid="stHorizontalBlock"]:has(.mr-poster, .mr-noposter){flex-wrap:wrap!important;gap:.6rem!important}
  [data-testid="stHorizontalBlock"]:has(.mr-poster, .mr-noposter) > [data-testid="stColumn"]{
    width:calc(50% - .3rem)!important;flex:1 1 calc(50% - .3rem)!important;min-width:calc(50% - .3rem)!important}
  .mr-desc{display:none}
  /* điện thoại: bỏ menu trượt bên trái (không tự đóng sau khi chọn trang), dùng hàng nút trang cuộn ngang */
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"]{display:none!important}
  [data-testid="stLayoutWrapper"]:has(> .st-key-mobile_nav){display:flex!important}
  .st-key-mobile_nav{display:flex!important;flex-wrap:nowrap!important;overflow-x:auto;gap:.4rem!important;
    margin-top:.8rem;padding:.15rem 0 .3rem;scrollbar-width:none}
  .st-key-mobile_nav > div{flex:0 0 auto!important;width:auto!important}
  .st-key-mobile_nav a{white-space:nowrap;border:1px solid #2c3440;border-radius:16px;padding:.15rem .7rem!important}
}
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


@st.cache_data
def get_results(path: str, mtime: float) -> pd.DataFrame | None:
    """Bảng kết quả thực nghiệm trên tập test (scripts/run_experiments.py), để hiện cạnh gợi ý."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path).set_index("Model")


@st.cache_data
def get_cold(path: str, mtime: float) -> pd.DataFrame | None:
    """NDCG@10 theo số phim đã thích của thí nghiệm người dùng mới, dạng bảng: hàng = số phim, cột = mô hình."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df = df[df["số phim thích"].astype(str) != "đầy đủ"]
    df["số phim thích"] = df["số phim thích"].astype(int)
    return df.pivot(index="số phim thích", columns="mô hình", values="NDCG@10")


svc, db = get_service(ARTIFACT), get_db(DB_PATH)
SESSION_KEY = sess.load_key(SESSION_KEY_PATH)
BLURBS = get_blurbs(BLURBS_PATH, _mtime(BLURBS_PATH))
EXTRAS = get_extras(TMDB_PATH, TRAILERS_PATH, _mtime(TMDB_PATH), _mtime(TRAILERS_PATH))


def is_guest() -> bool:
    return st.session_state.get("guest", False)


def is_admin() -> bool:
    return not is_guest() and st.session_state.get("username") in ADMINS


# ---------- thành phần dùng chung ----------
def fmt(x: float, digits: int = 0) -> str:
    """Số kiểu Việt Nam: 1.234,5 (dấu chấm hàng nghìn, dấu phẩy thập phân)."""
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def to_local(seconds: pd.Series | float):
    """Thời điểm Unix (giây, UTC) -> giờ Việt Nam."""
    if isinstance(seconds, pd.Series):
        return pd.to_datetime(seconds, unit="s", utc=True).dt.tz_convert(TZ)
    return pd.Timestamp(seconds, unit="s", tz="UTC").tz_convert(TZ)


def sign_in(user_id: int | None, username: str) -> None:
    """Vào phiên (user_id=None: khách) và hẹn đặt cookie để tải lại trang không phải đăng nhập lại."""
    info = db.user(user_id) if user_id is not None else None
    st.session_state.update(user_id=user_id, username=username, guest=user_id is None,
                            set_cookie=sess.make_token(SESSION_KEY, user_id, info[1] if info else b""))
    if user_id is not None and (pending := st.session_state.pop("pending_genres", None)) and not db.get_genres(user_id):
        db.set_genres(user_id, pending)  # thể loại đã chọn lúc là khách được giữ khi đăng ký/đăng nhập
    st.rerun()


def login_again() -> None:
    """Đăng xuất (hoặc khách bấm Đăng nhập): xóa phiên và cookie, giữ thể loại khách đã chọn."""
    keep = {k: st.session_state[k] for k in ("genre_select",) if is_guest() and k in st.session_state}
    st.session_state.clear()
    st.session_state.update(logged_out=True, clear_cookie=True)
    if keep.get("genre_select"):
        st.session_state.pending_genres = keep["genre_select"]
    st.rerun()


def restore_session() -> None:
    """Tải lại trang (F5), mở tab mới: đọc cookie phiên, hợp lệ thì vào lại đúng tài khoản."""
    if st.session_state.get("logged_out"):
        return
    ok, uid = sess.read_token(SESSION_KEY, st.context.cookies.get(sess.COOKIE),
                              lambda i: (db.user(i) or (None, None))[1])
    if ok:
        st.session_state.update(user_id=uid, guest=uid is None,
                                username="Khách" if uid is None else db.user(uid)[0])


def _save_rating(item: int, key: str) -> None:
    """st.feedback trả 0..4 (None khi bấm lại ngôi sao đang chọn để bỏ chấm); lưu thành 1..5 sao."""
    value = st.session_state[key]
    db.rate(st.session_state.user_id, item, None if value is None else value + 1)
    if key.startswith("rec_") and "rec_shown" in st.session_state:
        # Chấm ngay trên danh sách gợi ý: giữ nguyên danh sách đang thấy đến khi người dùng bấm "Cập nhật", để các
        # thẻ không đổi chỗ giữa hai lần bấm (bấm nhanh liên tục sẽ rơi nhầm vào phim khác).
        st.session_state.setdefault("rec_hold", st.session_state.rec_shown)
    name = display_title(svc.movies.iloc[item]["title"])
    st.toast(f"Đã bỏ chấm “{name}”" if value is None else f"Đã lưu {value + 1}★ cho “{name}”",
             icon=":material/star:")


def rating_stars(item: int, prefix: str, ratings: dict[int, int]) -> None:
    """Hàng 5 ngôi sao, bấm một lần là lưu ngay vào SQLite rồi trang tự chạy lại. Khách chưa đăng nhập: bị khóa."""
    key = f"{prefix}_{item}"
    if is_guest():
        st.caption("☆ Đăng nhập để chấm điểm")
        return
    st.feedback("stars", key=key, default=ratings[item] - 1 if item in ratings else None,
                on_change=_save_rating, args=(item, key), disabled=is_guest())


GROUP_OF_LABEL = {label: group for group, label in FEATURE_GROUPS.items()}
# Tên hiển thị trong ô "Vì sao?": trung tính, vì ▲/▼ chỉ là hướng tác động so với một phim ứng viên trung bình (vd.
# "▼ điểm trung bình" ở một phim 4,5★ nghĩa là điểm cao này đẩy phim lên ít hơn ở các ứng viên khác, không phải phim dở).
WHY_LABEL = {
    "als": "Người có gu giống bạn cũng thích",
    "knn": "Hay được xem cùng phim bạn đã chấm",
    "content": "Nội dung gần với phim bạn thích",
    "pop": "Độ phổ biến của phim",
    "genre": "Thể loại so với gu của bạn",
    "quality": "Điểm trung bình và tỉ lệ thích của phim",
    "year": "Năm phát hành",
    "item_genre": "Thể loại của phim",
    "sources": "Số mô hình cùng đề cử",
    "user": "Thói quen chấm điểm của bạn",
}


def _round_percent(shares: list[float]) -> list[int]:
    """Làm tròn các tỉ lệ thành phần trăm nguyên có tổng đúng 100 (phương pháp phần dư lớn nhất)."""
    raw = [100 * s for s in shares]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: out[i] - raw[i])[: 100 - sum(out)]:
        out[i] += 1
    return out


def why_box(factors: list[tuple[str, float]], src: dict[str, str] | None = None, m: dict | None = None) -> None:
    """Các yếu tố ảnh hưởng mạnh nhất đến thứ hạng của phim (giá trị SHAP của bộ xếp hạng, gộp theo nhóm đặc trưng),
    hiện thành phần trăm đóng góp để người dùng không phải đọc giá trị SHAP thô. src: phim đã thích đóng góp nhiều
    nhất cho từng nhóm (ALS, ItemKNN, nội dung), cùng nguồn với dòng lý do trên thẻ phim."""
    src, m = src or {}, m or {}
    if m.get("added"):
        st.caption(f"Phim này được thêm vào vì bạn đã chọn thể loại: ít nhất {math.ceil(18 * GENRE_SHARE)}/18 phim "
                   "phải thuộc các thể loại đó.")
    info = m.get("stats") or {}
    value = {"quality": f"★ {fmt(info['mean'], 1)} · {info['like_frac']:.0%} thích" if "mean" in info else "",
             "pop": f"{fmt(info['n'])} lượt chấm" if "n" in info else "",
             "year": "".join(re.findall(r"\((\d{4})\)\s*$", m.get("title", "")))}
    total = sum(abs(v) for _, v in factors) or 1.0
    pct = _round_percent([abs(v) / total for _, v in factors])
    for (label, v), p in zip(factors, pct):
        if p == 0:  # yếu tố gần như không đóng góp: bỏ, không hiện "▲ … 0%"
            continue
        share = p / 100
        group = GROUP_OF_LABEL.get(label)
        label = WHY_LABEL.get(group) or re.sub(r"\s*\([^)]*\)$", "", label)
        if value.get(group):
            label += f" ({value[group]})"
        color = "#00e054" if v > 0 else "#ff5c5c"
        bar = "█" * max(1, round(8 * share))
        note = (f"<br><span style='color:#9ab'>chủ yếu từ: {html.escape(src[group])}</span>"
                if v > 0 and group in src else "")
        st.markdown(f"<span style='font-size:0.8rem'>{'▲' if v > 0 else '▼'} {html.escape(label)}<br>"
                    f"<span style='color:{color}'>{bar}</span> {p}%{note}</span>", unsafe_allow_html=True)
    if m.get("method") == "two_stage":
        st.caption(f"Phần trăm: tỉ lệ của mỗi yếu tố trong {len(factors)} yếu tố ảnh hưởng mạnh nhất tới thứ hạng của "
                   "phim này (giá trị SHAP của bộ xếp hạng LightGBM). ▲ đẩy phim lên, ▼ kéo xuống, so với một phim ứng "
                   "viên trung bình; ▼ không có nghĩa là phim dở.")
    else:
        st.caption("Điểm gợi ý là tổng của bốn thành phần (ALS, ItemKNN, nội dung, độ phổ biến; phần gần 0% không "
                   "hiện), nên phần trăm là tỉ lệ đóng góp đúng của từng thành phần. ▲ đẩy phim lên, ▼ kéo xuống, so với một phim trung bình trong kho; ▼ không có nghĩa là phim dở.")


def overview_block(m: dict) -> None:
    """Câu giới thiệu tiếng Việt nếu có; mô tả gốc tiếng Anh của TMDB chỉ hiện khi chưa có câu tiếng Việt, hoặc thu
    gọn trong mục "Mô tả gốc" để giao diện không lẫn tiếng Anh."""
    overview = str(m.get("overview") or "").strip()
    if m["title"] in BLURBS:
        st.write(BLURBS[m["title"]])
        if overview:
            with st.expander("Mô tả gốc (tiếng Anh, từ TMDB)"):
                st.caption(overview)
    elif overview:
        st.caption(overview)


def description(m: dict) -> str:
    """Câu giới thiệu tiếng Việt viết sẵn nếu có, không thì mô tả gốc tiếng Anh."""
    return BLURBS.get(m["title"]) or str(m.get("overview") or "")


def stats_line(m: dict) -> str:
    """'★ 4,1 · 2.077 lượt · 80% thích' (số liệu MovieLens 1M); thiếu số liệu thì chỉ ghi lượt thích."""
    st_ = m.get("stats") or {}
    if "n" not in st_:
        return f"♥ {st_['likes']:,} lượt thích".replace(",", ".") if st_ else ""
    if st_["n"] == 0:
        return "Chưa có lượt chấm"
    return (f"★ {st_['mean']:.1f}".replace(".", ",") + f" · {st_['n']:,} lượt".replace(",", ".")
            + f" · {st_['like_frac']:.0%} thích")


def card_html(m: dict) -> str:
    e = html.escape
    poster = (f'<img class="mr-poster" src="{e(m["poster_url"])}" alt="" loading="lazy">' if m.get("poster_url")
              else '<div class="mr-noposter">🎬</div>')
    reason = f'<div class="mr-reason" title="{e(m["reason"])}">💡 {e(m["reason"])}</div>' if m.get("reason") else ""
    badge = '<div class="mr-badge">🔖 Muốn xem</div>' if m["item"] in st.session_state.get("watch_set", set()) else ""
    change = m.get("change")  # thay đổi so với danh sách trước lần chấm gần nhất: "new" hoặc số hạng tăng
    if change == "new":
        badge += '<div class="mr-change" title="Mới vào danh sách sau lần chấm vừa rồi">Mới</div>'
    elif change:
        badge += f'<div class="mr-change up" title="Lên {change} hạng sau lần chấm vừa rồi">▲{change}</div>'
    poster = f'<div style="position:relative">{poster}{badge}</div>'
    return (f'{poster}<div class="mr-title" title="{e(display_title(m["title"]))}">{e(display_title(m["title"]))}</div>'
            f'<div class="mr-genres">{e(genre_vi(m["genres"], " · "))}</div>'
            f'<div class="mr-stats" title="Số liệu MovieLens 1M: điểm trung bình · số lượt chấm · tỉ lệ chấm ≥ 4 sao">'
            f'{e(stats_line(m))}</div>'
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
        parts.append(f"TMDB {fmt(e['vote_average'], 1)}/10 ({fmt(e['vote_count'])} lượt)")
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
        st.subheader(display_title(m["title"]))
        st.caption(" · ".join(x for x in [genre_vi(m["genres"]), _info_line(movie_id)] if x))
        people = []
        if isinstance(extra.get("director"), str) and extra["director"]:
            people.append(f"**Đạo diễn:** {', '.join(extra['director'].split('|'))}")
        if isinstance(extra.get("cast"), str) and extra["cast"]:
            people.append(f"**Diễn viên:** {', '.join(extra['cast'].split('|')[:5])}")
        line = stats_line(m)
        if line:
            st_ = m["stats"]
            people.append(f"**MovieLens:** {line} · hạng {fmt(st_['rank'])}/{fmt(len(svc.pop))} về lượt thích")
        if people:
            st.markdown("  \n".join(people))
        overview_block(m)
        if is_guest():
            st.caption("Đăng nhập để chấm điểm và lưu phim vào danh sách muốn xem.")
            if st.button("Đăng nhập", key=f"dlg_login_{item}", icon=":material/login:", type="primary"):
                login_again()
        else:
            st.caption("Điểm của bạn")
            rating_stars(item, "dlg", ratings)
            watch_set = st.session_state.setdefault("watch_set", set())
            if st.button("Bỏ khỏi danh sách muốn xem" if item in watch_set else "Thêm vào danh sách muốn xem",
                         key=f"dlg_watch_{item}", icon=":material/bookmark:"):
                added = db.toggle_watch(st.session_state.user_id, item)
                (watch_set.add if added else watch_set.discard)(item)
                name = display_title(m["title"])
                st.toast(f"Đã thêm “{name}” vào Muốn xem" if added else f"Đã bỏ “{name}” khỏi Muốn xem",
                         icon=":material/bookmark_added:" if added else ":material/bookmark_remove:")
                st.rerun(scope="fragment")  # chỉ chạy lại cửa sổ chi tiết, cửa sổ không bị đóng
    key = extra.get("youtube_key") if hasattr(extra, "get") else None
    if isinstance(key, str) and key:
        st.markdown("**Trailer**")
        st.video(f"https://www.youtube.com/watch?v={key}")
    st.markdown("**Phim tương tự**")
    cols = st.columns(6)
    for col, s in zip(cols, [s for s in svc.similar(item, k=6 + len(ratings)) if s["item"] not in ratings][:6]):
        with col:
            st.markdown(card_html({**s, "overview": ""}).split('<div class="mr-desc">')[0], unsafe_allow_html=True)
            if st.button("Chi tiết", key=f"dlg_sim_{item}_{s['item']}", type="tertiary"):
                st.session_state.open_item = s["item"]  # main() mở cửa sổ của phim này sau khi chạy lại
                st.rerun()


def movie_card(m: dict, prefix: str, ratings: dict[int, int]) -> None:
    with st.container(border=True):
        st.markdown(card_html(m), unsafe_allow_html=True)
        rating_stars(m["item"], prefix, ratings)
        with st.container(horizontal=True, gap="small"):
            if st.button("Chi tiết", key=f"{prefix}_info_{m['item']}", type="tertiary"):
                movie_dialog(m["item"], ratings)
            if m.get("why"):
                with st.popover("Vì sao?", type="tertiary"):
                    why_box(m["why"], m.get("why_src"), m)


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
        features = ["⭐ Gợi ý theo sở thích, có giải thích lý do", "🔎 Tìm phim theo tên"]
        if Path(SEMANTIC).exists():
            features.append("💬 Tìm phim bằng câu tiếng Việt")
        features.append("🤖 Trợ lý hội thoại")
        st.caption(" · ".join(features))
        tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
        # st.form: gửi cả tên và mật khẩu cùng lúc khi bấm nút, kể cả khi ô nhập chưa mất tiêu điểm
        with tab_login, st.form("login_form"):
            user = st.text_input("Tên đăng nhập", key="login_user")
            pw = st.text_input("Mật khẩu", type="password", key="login_pw")
            if st.form_submit_button("Đăng nhập", key="login_btn", type="primary", width="stretch"):
                uid = db.login(user, pw) if user.strip() and pw else None
                if not user.strip() or not pw:
                    st.error("Bạn chưa nhập tên đăng nhập hoặc mật khẩu.")
                elif uid is None:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")
                else:
                    sign_in(uid, db.user(uid)[0])  # tên đã lưu, không phải chữ hoa/thường người dùng vừa gõ
        with tab_register, st.form("register_form"):
            user = st.text_input("Tên đăng nhập", key="reg_user")
            pw = st.text_input("Mật khẩu", type="password", key="reg_pw")
            if st.form_submit_button("Tạo tài khoản", key="reg_btn", type="primary", width="stretch"):
                try:
                    uid = db.register(user, pw, reserved=ADMINS)
                except ValueError as err:
                    st.error(str(err))
                else:
                    sign_in(uid, user.strip())
        st.divider()
        if st.button("Xem thử không cần đăng nhập", key="guest_btn", width="stretch", icon=":material/visibility:"):
            sign_in(None, "Khách")
        st.caption("Khách xem được gợi ý theo thể loại và độ phổ biến, tìm phim, tìm theo mô tả và trợ lý; "
                   "đăng nhập để chấm điểm và nhận gợi ý cá nhân hóa.")


def _save_diversity() -> None:
    if is_guest():  # khách: chỉ giữ trong phiên làm việc
        st.session_state.guest_diversity = st.session_state.get("diversity")
    elif st.session_state.get("diversity"):
        db.set_diversity(st.session_state.user_id, st.session_state.diversity)


def _save_genres() -> None:
    if is_guest():  # khách: chỉ giữ trong phiên làm việc
        st.session_state.guest_genres = st.session_state.genre_select
        st.session_state.pending_genres = st.session_state.genre_select  # giữ khi khách đăng ký/đăng nhập
    else:
        db.set_genres(st.session_state.user_id, st.session_state.genre_select)


def _rate_batch(ratings: dict[int, int], new: bool = False) -> list[int]:
    """18 phim nổi tiếng đang cho chấm. Giữ nguyên tới khi bấm "Xem phim khác", nên phim vừa chấm vẫn nằm tại chỗ
    (hiện số sao đã chấm) thay vì biến mất và để phim khác trượt vào đúng chỗ con trỏ."""
    batch = st.session_state.get("rate_batch")
    if new or batch is None:
        page = st.session_state.get("rate_page", 0) + (1 if new else 0)
        pool = [m["item"] for m in svc.popular(300) if m["item"] not in ratings]
        page = page % max(1, len(pool) // 18)
        batch = pool[page * 18:(page + 1) * 18] or pool[:18]
        st.session_state.update(rate_page=page, rate_batch=batch)
    return batch


def _track_changes(recs: list[dict], ratings: dict[int, int], ctx: tuple) -> str:
    """Minh họa việc gợi ý được tính lại ngay sau mỗi lần chấm (fold-in, không cần huấn luyện lại): so danh sách hiện
    tại với danh sách trước lần chấm gần nhất, gắn nhãn "Mới" hoặc "▲n" lên thẻ phim và trả về một câu tóm tắt.
    Phim vừa chấm tự rời khỏi danh sách, nên thứ hạng cũ được tính sau khi bỏ các phim đó để các phim phía sau
    không bị tính là "lên hạng" chỉ vì có một chỗ trống."""
    items = [m["item"] for m in recs]
    method = sum(r >= 4 for r in ratings.values())  # số phim thích, để nhận ra mốc MIN_RATINGS
    sig = tuple(sorted(ratings.items()))
    prev = st.session_state.get("rec_track")
    if prev is not None and prev["ctx"] == ctx and prev["sig"] == sig:
        changes, summary = prev["changes"], prev["summary"]  # chưa chấm thêm: giữ nhãn của lần chấm trước
    else:
        changes, summary = {}, ""
        prev_rated = {j for j, _ in prev["sig"]} if prev is not None else set()
        if prev is not None and prev["ctx"] == ctx and not prev_rated - set(ratings):  # chỉ khi có thêm lượt chấm
            old = {j: r for r, j in enumerate([j for j in prev["items"] if j not in ratings])}
            for r, j in enumerate(items):
                if j not in old:
                    changes[j] = "new"
                elif old[j] - r >= 1:
                    changes[j] = old[j] - r
            n_new = sum(v == "new" for v in changes.values())
            n_up = len(changes) - n_new
            if prev["method"] < MIN_RATINGS <= method:
                summary = (f"🎉 Bạn đã thích {method} phim. Lần chấm vừa rồi đưa {n_new} phim mới vào danh sách, "
                           f"{n_up} phim lên hạng (nhãn trên poster)")
            elif changes:
                summary = (f"Sau lần chấm vừa rồi: {n_new} phim mới vào danh sách, {n_up} phim lên hạng "
                           "(nhãn trên poster)")
            else:
                summary = "Sau lần chấm vừa rồi: thứ tự gợi ý gần như giữ nguyên"
        st.session_state.rec_track = {"ctx": ctx, "sig": sig, "items": items, "method": method,
                                      "changes": changes, "summary": summary}
    for m in recs:
        if m["item"] in changes:
            m["change"] = changes[m["item"]]
    return summary


def _cold_level(liked: int) -> int:
    """Mức gần nhất (không vượt quá) trong thí nghiệm người dùng mới ứng với số phim đã thích."""
    return max([n for n in COLD_LEVELS if n <= liked] or [COLD_LEVELS[0]])


def model_info(method: str, liked: int, style: str = DEFAULT_STYLE) -> None:
    """Một dòng cho biết mô hình nào đang tạo danh sách và độ chính xác của nó với người có số phim thích tương tự
    (thí nghiệm người dùng mới), mở ra là chế độ thử nghiệm và các bảng kết quả thực nghiệm."""
    cold = get_cold(COLD_PATH, _mtime(COLD_PATH))
    level = _cold_level(liked)
    if method == "two_stage":
        key, short = "Hai tầng (học cả hồ sơ ngắn)", "Kiến trúc hai tầng, LightGBM (“Sát gu nhất”)"
        detail = ("Tầng 1: bốn mô hình (Implicit ALS, ItemKNN, nội dung, độ phổ biến) cùng chọn khoảng 190 phim ứng "
                  "viên. Tầng 2: LightGBM (LambdaRank) xếp hạng lại các ứng viên; bản này được huấn luyện thêm với hồ "
                  "sơ ngắn mô phỏng người dùng mới. Bấm “Vì sao?” để xem đóng góp của từng yếu tố (giá trị SHAP).")
    elif liked:
        c = STYLES.get(style, STYLES[DEFAULT_STYLE])[1] or svc.pop_c
        key, short = STYLE_KEY.get(style, STYLE_KEY[DEFAULT_STYLE]), "Công thức trộn ALS + ItemKNN + nội dung + phổ biến"
        detail = (f"Điểm của Implicit ALS, ItemKNN, nội dung và độ phổ biến được chuẩn hóa rồi cộng có trọng số. "
                  f"Trọng số độ phổ biến bằng {c:g}/n với n là số phim bạn đã thích (bạn thích {liked} phim nên trọng "
                  f"số là {c:g}/{liked} = {fmt(c / liked, 1)}): càng biết nhiều về bạn, gợi ý càng dựa vào "
                  "sở thích riêng thay vì độ phổ biến. Vì vậy khi mới thích vài phim, danh sách còn nhiều phim nổi "
                  "tiếng chưa hẳn đúng gu; chấm thêm phim sẽ thấy rõ phần sở thích riêng. Trong thí nghiệm người dùng "
                  "mới bên dưới, cách này chính xác hơn kiến trúc hai tầng và gợi ý theo độ phổ biến; muốn danh sách "
                  "bám gu hơn, chọn “Sát gu nhất” ở mục “Kiểu gợi ý”. Bấm “Vì sao?” để xem đóng góp đúng của từng "
                  "thành phần.")
    else:
        key, short = "Phổ biến nhất", "Độ phổ biến theo thể loại (chưa có phim thích)"
        detail = ("Chưa có phim nào được chấm 4-5★ nên chưa học được sở thích riêng: hệ thống chọn phim được thích "
                  "nhiều nhất, trong các thể loại đã chọn nếu có.")
    ndcg = ""
    if cold is not None and key in cold.columns:
        where = (f"{level} phim trở lên" if liked > COLD_LEVELS[-1] else
                 f"{level} phim" if liked == level else f"{level} phim, mức đo gần nhất với {liked} phim của bạn")
        ndcg = f" · NDCG@10 {fmt(cold.loc[level, key], 4)} khi người mới thích {where}"
    with st.expander(f"🧠 Mô hình: {short}{ndcg}", key="model_info"):
        st.write(detail)
        if cold is not None:
            st.markdown("**Người dùng mới:** NDCG@10 theo số phim đã thích")
            cols = [c for c in COLD_MODELS if c in cold.columns]
            tab = cold.loc[[n for n in COLD_LEVELS if n in cold.index], cols]
            st.dataframe(pd.DataFrame({"Mô hình": [COLD_MODELS[c] for c in cols],
                                       **{f"{n} phim": [fmt(tab.loc[n, c], 4) for c in cols] for n in tab.index}}),
                         hide_index=True, width="stretch")
            st.caption("Người dùng MovieLens có từ 30 phim thích được chia thành 5 phần rời nhau; mỗi lần, một phần bị loại "
                       "hẳn khỏi dữ liệu huấn luyện rồi “đăng ký mới” với n phim thích lấy từ lịch sử của họ, và được đo trên "
                       "các phim họ thích về sau (10% lượt chấm cuối). Trọng số 120/n được chọn trên một phần riêng; bảng "
                       "là kết quả trên 4 phần còn lại (3.307 người). Trung bình trên các mức n, công thức trộn này hơn kiến "
                       "trúc hai tầng 12,9% và hơn độ phổ biến 14,9% (có ý nghĩa thống kê). Ở nhóm 4.539 người có từ 10 phim thích "
                       "(n = 1–10), nó hơn độ phổ biến 16,5% nhưng hơn kiến trúc hai tầng 6,8% chưa có ý nghĩa thống kê, "
                       "và kém trọng số 60/n 3,2%.")
        res = get_results(RESULTS_PATH, _mtime(RESULTS_PATH))
        if res is None:
            return
        st.markdown("**Người dùng có lịch sử đầy đủ** (thực nghiệm chính, tập kiểm tra)")
        table = res.sort_values("NDCG@10", ascending=False)
        st.dataframe(pd.DataFrame({
            "Mô hình": [MODEL_VI.get(i, i) for i in table.index],
            "NDCG@10": [fmt(v, 4) for v in table["NDCG@10"]],
            "Recall@10": [fmt(v, 4) for v in table["Recall@10"]],
            "Độ phủ": [f"{fmt(100 * v, 1)}%" for v in table["Coverage"]],
            "So với Phổ biến": [str(v).replace(".", ",") for v in table["vs Popularity"]],
        }), hide_index=True, width="stretch")
        st.caption("Tập kiểm tra: 10% lượt chấm cuối cùng (theo thời gian) của mỗi người dùng MovieLens 1M; điểm tính "
                   "trên 5.741 người có ít nhất một phim chấm 4-5★ trong phần đó, mỗi người đã có lịch sử chấm đầy đủ "
                   "phía trước (trung bình hơn 100 lượt). Ở đây kiến trúc hai tầng đứng đầu nhóm mô hình dùng được "
                   "trên web, nhưng lợi thế đó đến từ lịch sử dài và không còn với người dùng mới. NDCG@10: các phim "
                   "họ thực sự thích có nằm trong top 10 không, nằm càng cao càng tốt. Độ phủ: tỉ lệ kho phim từng "
                   "được gợi ý cho ít nhất một người.")
        st.caption(DIVERSITY_NOTE)
        st.caption("SASRec cao nhất nhưng không dùng trên web: gần như toàn bộ lợi thế của nó đến từ thứ tự chấm điểm "
                   "(xáo trộn thứ tự thì NDCG@10 còn 0,0663, thấp hơn kiến trúc hai tầng). Trên web, người dùng chấm "
                   "theo thứ tự phim hiện ra trên trang chứ không theo thứ tự đã xem, nên thứ tự đó không mang thông tin.")


def page_home(ratings: dict[int, int]) -> None:
    liked = sum(r >= 4 for r in ratings.values())
    cold = liked < MIN_RATINGS  # chưa đủ 5 phim THÍCH (4-5 sao): mở sẵn tab chấm phim
    guest = is_guest()
    shown: set[int] = set()
    if guest:  # khách không chấm được nên không có tab chấm phim
        tab_rec = st.container()
    else:
        tab_rec, tab_rate = st.tabs(["⭐ Phim dành cho bạn", "🎞️ Chấm thêm phim"], key="home_tab",
                                    default="🎞️ Chấm thêm phim" if cold else "⭐ Phim dành cho bạn")
        # Phim nổi tiếng, mỗi lần 18 phim. Người mới dùng tab này để cho hệ thống biết sở thích.
        with tab_rate:
            left, right = st.columns([4, 1], vertical_alignment="center")
            with left:
                if cold:
                    st.progress(liked / MIN_RATINGS,
                                text=f"Đã thích {liked}/{MIN_RATINGS} phim (chấm 4-5★), đã chấm tổng cộng "
                                     f"{len(ratings)} phim. Thích thêm {MIN_RATINGS - liked} phim nữa để gợi ý bám sát "
                                     "sở thích của bạn.")
                else:
                    st.caption("Phim nổi tiếng. Chấm những phim bạn đã xem, chấm càng nhiều gợi ý càng sát sở thích.")
            if right.button("Xem phim khác", key="more_to_rate", icon=":material/refresh:", width="stretch"):
                _rate_batch(ratings, new=True)
            batch = _rate_batch(ratings)
            movie_grid([svc._movie(i) for i in batch], "onb", ratings)

    with tab_rec:
        genres = (st.session_state.get("guest_genres", []) if guest
                  else db.get_genres(st.session_state.user_id))
        left, mid, right = st.columns([3, 2, 2], vertical_alignment="bottom")
        left.multiselect("Thể loại yêu thích", svc.genres, default=genres, key="genre_select",
                         on_change=_save_genres, placeholder="Chọn vài thể loại bạn thích", select_all=False,
                         format_func=lambda g: GENRE_VI.get(g, g))
        # Mức đa dạng = 1 - lam của MMR. Khách: giữ trong phiên (bản sao ngoài widget, vì Streamlit xóa giá trị của
        # widget khi chuyển sang trang khác)
        saved = st.session_state.get("guest_diversity", "") if guest else db.get_diversity(st.session_state.user_id)
        level = right.segmented_control("Độ đa dạng thể loại", list(DIVERSITY),
                                        default=saved if saved in DIVERSITY else DEFAULT_DIVERSITY, key="diversity",
                                        required=True, on_change=_save_diversity, help=DIVERSITY_HELP)
        style = DEFAULT_STYLE
        if not guest and liked:
            options = [k for k in STYLES if svc.ranker is not None or STYLES[k][0] == "blend"]
            saved_style = st.session_state.get("style_saved", DEFAULT_STYLE)
            style = mid.segmented_control("Kiểu gợi ý", options, key="style",
                                          default=saved_style if saved_style in options else DEFAULT_STYLE,
                                          required=True, help=STYLE_HELP,
                                          on_change=lambda: st.session_state.update(style_saved=st.session_state.style)
                                          ) or DEFAULT_STYLE
        mode, pop_c = STYLES[style] if liked and not guest else ("blend", None)
        ctx = (tuple(genres), level, mode, pop_c)
        hold = st.session_state.get("rec_hold")
        if hold is not None and hold["ctx"] == ctx:
            recs, change = hold["recs"], ""
        else:
            st.session_state.pop("rec_hold", None)
            hold = None
            recs = [m for m in svc.recommend(ratings, genres, k=18 + len(shown), diversity=DIVERSITY[level or DEFAULT_DIVERSITY],
                                             mode=mode, pop_c=pop_c) if m["item"] not in shown][:18]
            change = "" if guest else _track_changes(recs, ratings, ctx)
            st.session_state.rec_shown = {"ctx": ctx, "recs": recs}
        method = recs[0].get("method", "blend") if recs else "blend"
        chosen = f"thể loại bạn chọn ({genre_vi(genres)})" if genres else ""
        if guest:
            st.caption((f"Phim được thích nhiều nhất trên MovieLens thuộc {chosen}. " if genres else
                        "Phim được thích nhiều nhất trên MovieLens. Chọn thể loại ở trên để lọc. ")
                       + "Đăng nhập để chấm phim và nhận gợi ý riêng cho bạn.")
        elif liked:
            note = (f" Ít nhất {math.ceil(18 * GENRE_SHARE)}/18 phim thuộc {chosen}." if genres else "")
            more = (f" Thích thêm {MIN_RATINGS - liked} phim (chấm 4-5★) để gợi ý bám sát hơn." if liked < MIN_RATINGS
                    else "")
            st.caption(f"Gợi ý dựa trên {liked} phim bạn thích.{note}{more} Bấm “Vì sao?” để xem điều gì đưa phim lên.")
        else:
            st.caption((f"Phim được thích nhiều nhất thuộc {chosen}. " if genres else "Phim được thích nhiều nhất. ")
                       + f"Thích vài phim (chấm 4-5★) để nhận gợi ý riêng; nên thích ít nhất {MIN_RATINGS} phim.")
        n_genres = len({g for m in recs for g in m["genres"].split("|")})
        st.caption(f"{len(recs)} phim · {n_genres} thể loại khác nhau")
        if not guest:  # khách chưa có gợi ý riêng, không cần thông tin mô hình
            model_info(method, liked, style)
            if style == DEFAULT_STYLE and liked < 20:
                st.caption("Hồ sơ còn ngắn nên gợi ý đang nghiêng về phim nổi tiếng. Chọn “Cân bằng” hoặc “Sát gu nhất” "
                           "ở mục “Kiểu gợi ý” nếu muốn danh sách bám gu hơn.")
        if hold is not None:
            n_new = sum(m["item"] in ratings for m in recs)
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                st.markdown(f"Bạn vừa chấm {n_new} phim trong danh sách này. Danh sách được giữ nguyên để bạn chấm "
                            "tiếp; bấm cập nhật để xem gợi ý mới tính theo các lần chấm đó.")
                st.button("Cập nhật gợi ý", key="rec_refresh", icon=":material/autorenew:", type="primary",
                          on_click=lambda: st.session_state.pop("rec_hold", None))
        elif change:
            st.info(change, icon=":material/autorenew:")
        movie_grid(recs, "rec", ratings)


def page_search(ratings: dict[int, int]) -> None:
    query = st.text_input("Tên phim (tiếng Anh)", key="search_q", placeholder="ví dụ: matrix")[:100]
    results = svc.search(query)
    if query.strip() and not results:
        shown_q = query.strip() if len(query.strip()) <= 40 else query.strip()[:40] + "…"
        st.warning(f"Không có phim nào tên khớp “{shown_q}”. Hãy thử tên tiếng Anh gốc (vd. “matrix”, "
                   "“star wars”), bớt vài từ, hoặc dùng trang Tìm theo mô tả nếu chỉ nhớ nội dung phim.")
    elif query.strip():
        total = svc.search_all(query)[1]
        st.caption(f"{total:,} phim khớp".replace(",", ".") + (f", hiện {len(results)} phim phổ biến nhất." if total > len(results)
                                                               else "."))
    for m in results:
        with st.container(border=True):
            poster, body = st.columns([1, 6])
            if m.get("poster_url"):
                poster.image(m["poster_url"], width="stretch")
            with body:
                st.markdown(f"**{display_title(m['title'])}**")
                st.caption(genre_vi(m["genres"], " · ") + " · " + stats_line(m))
                overview_block(m)
                rating_stars(m["item"], "search", ratings)
                if st.button("Chi tiết · trailer · Muốn xem", key=f"search_info_{m['item']}", type="tertiary",
                             icon=":material/info:"):
                    movie_dialog(m["item"], ratings)
            with st.expander("Phim tương tự"):
                movie_grid([s for s in svc.similar(m["item"], k=6 + len(ratings)) if s["item"] not in ratings][:6],
                           f"sim{m['item']}", ratings)


def page_semantic(ratings: dict[int, int]) -> None:
    st.write("Mô tả bộ phim bạn muốn xem bằng một câu tiếng Việt. Hệ thống so nghĩa của câu đó với nội dung "
             "từng phim, không cần trùng từ khóa hay biết tên phim.")
    index = get_semantic(SEMANTIC)
    if index is None:
        st.info("Tính năng tìm theo mô tả tạm thời chưa sẵn sàng. Bạn có thể tìm theo tên ở trang “Tìm phim”, "
                "hoặc hỏi “Trợ lý phim”.")
        if is_admin():
            st.caption("Quản trị: cài `pip install sentence-transformers`, chạy `python scripts/build_semantic.py` "
                       "rồi bấm “Nạp lại mô hình và dữ liệu” ở trang Quản trị.")
        return
    query = st.text_input("Bạn muốn xem phim như thế nào?", key="semantic_q",
                          placeholder="ví dụ: phim hoạt hình vui nhộn cho trẻ em")
    hide_rated = not is_guest() and st.checkbox("Ẩn phim tôi đã chấm", value=True, key="semantic_hide")
    if not query.strip():
        st.caption("Gợi ý: “phi hành gia bị mắc kẹt ngoài không gian”, “thám tử điều tra kẻ giết người hàng loạt”, "
                   "“chuyện tình lãng mạn trên con tàu”.")
        return
    hits = index.search(query, k=18, exclude=set(ratings) if hide_rated else None, popularity=svc.pop)
    st.caption("Thứ tự kết hợp độ khớp nội dung với độ phổ biến của phim (phim quen thuộc hơn được ưu tiên khi nội dung "
               "gần giống nhau), nên đôi khi phim có độ khớp thấp hơn một chút vẫn đứng trước.")
    st.caption("“Độ khớp” cho biết nội dung phim gần với câu của bạn tới đâu, từ 0% đến 100%; từ khoảng 50% trở lên "
               "là nội dung rất gần.")
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
        can = ("Trợ lý vẫn tìm phim theo nội dung mô tả, nhưng chỉ hiểu các yêu cầu đơn giản (thể loại, năm, “phim "
               "giống …”)." if bot.semantic is not None else
               "Trợ lý chỉ hiểu các yêu cầu đơn giản (thể loại, năm, “phim giống …”) và chưa tìm được theo nội dung "
               "mô tả.")
        st.warning(f"Trợ lý đang chạy ở chế độ rút gọn. {can}")
        if is_admin():
            st.caption(f"Quản trị: mở Ollama (và chạy `ollama pull {LLM_MODEL}` nếu chưa tải mô hình), rồi tải lại trang.")
    else:
        st.caption("Phim do hệ thống gợi ý chọn từ kho phim; một mô hình ngôn ngữ nhỏ chạy ngay trên máy chỉ giúp hiểu "
                   "câu hỏi. Mỗi câu trả lời có thể mất 15–30 giây.")
    chat = st.session_state.setdefault("chat", [])        # các lượt để hiển thị
    state = st.session_state.setdefault("chat_state", {})  # ngữ cảnh của trợ lý giữa các lượt
    if chat and st.button("Xóa hội thoại", key="chat_clear", icon=":material/delete_sweep:"):
        chat.clear(); state.clear()
        st.rerun()
    if not chat:
        st.caption("Thử: “phim kinh dị thập niên 90 về ngôi nhà ma ám”, “phim nào giống Toy Story”, "
                   "“gợi ý phim hài, đừng có lãng mạn”, rồi hỏi tiếp “còn phim nào khác không”.")
    for n, turn in enumerate(chat):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("movies"):
                movie_grid(turn["movies"], f"chat{n}", ratings, cols=5)
    message = (st.chat_input("Bạn muốn xem phim gì?", key="chat_input") or "").strip()
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
    c2.metric("Điểm trung bình", f"{fmt(sum(ratings.values()) / len(ratings), 1)} ★")
    c3.metric("Số phim thích (4-5 sao)", sum(r >= 4 for r in ratings.values()))
    c4.metric("Thể loại chấm nhiều nhất", GENRE_VI.get(genre_count.index[0], genre_count.index[0]),
              help=f"{genre_count.iloc[0]}/{len(ratings)} phim bạn đã chấm thuộc thể loại này")
    order = st.segmented_control("Sắp xếp", ["Điểm cao trước", "Điểm thấp trước", "Tên phim"],
                                 default="Điểm cao trước", key="hist_order", required=True)
    items = list(ratings)
    if order == "Tên phim":
        items.sort(key=lambda i: display_title(svc.movies.iloc[i]["title"]).lower())
    else:
        items.sort(key=lambda i: ratings[i], reverse=order != "Điểm thấp trước")
    st.caption("Bấm vào ngôi sao để chấm lại; bấm lại đúng ngôi sao đang chọn để bỏ chấm phim đó.")
    movie_grid([svc._movie(i) for i in items], "hist", ratings)


def page_admin(ratings: dict[int, int]) -> None:
    """Trang quản trị: số liệu sử dụng, người dùng, tình trạng các thành phần, nạp lại mô hình, xuất dữ liệu."""
    if msg := st.session_state.pop("admin_msg", None):
        st.success(msg)
    stats = db.stats()
    c = st.columns(5)
    c[0].metric("Tài khoản", stats["users"], help="Số tài khoản đã đăng ký trên web, kể cả tài khoản quản trị.")
    c[1].metric("Lượt chấm trên web", fmt(stats["ratings"]),
                help="Chỉ tính lượt chấm của người dùng web, không gồm 1 triệu lượt của MovieLens.")
    c[2].metric("Điểm trung bình", "–" if stats["avg_rating"] is None else f"{fmt(stats['avg_rating'], 2)} ★ / 5",
                help="Trung bình mọi lượt chấm trên web.")
    c[3].metric("Người chấm phim trong 7 ngày", stats["active_7d"],
                help="Số tài khoản có ít nhất một lượt chấm trong 7 ngày gần nhất.")
    c[4].metric("Lượt lưu Muốn xem", stats["watchlist"], help="Tổng số phim được lưu vào danh sách Muốn xem.")

    all_r = pd.DataFrame(db.all_ratings(), columns=["user_id", "item", "rating", "rated_at"])
    users = pd.DataFrame(db.users_overview())
    for col in ("avg_rating", "last_rated"):  # người chưa chấm phim nào: SQLite trả None
        users[col] = pd.to_numeric(users[col], errors="coerce")
    names = users.set_index("id")["username"]
    tab_use, tab_users, tab_sys = st.tabs(["Hoạt động", "Người dùng", "Hệ thống"], key="admin_tab")
    with tab_use:
        if all_r.empty:
            st.info("Chưa có lượt chấm điểm nào trên ứng dụng.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Phân bố điểm**")
                dist = all_r["rating"].value_counts().reindex(range(1, 6), fill_value=0)
                st.bar_chart(pd.DataFrame({"Số lượt": dist.to_numpy()}, index=[f"{k} sao" for k in range(1, 6)]),
                             color="#ff8000", height=260)
            with right:
                st.markdown("**Phim được chấm nhiều nhất**")
                top = all_r.groupby("item")["rating"].agg(["count", "mean"]).sort_values("count", ascending=False)[:10]
                st.dataframe(pd.DataFrame({"Phim": [display_title(svc.movies.iloc[i]["title"]) for i in top.index],
                                           "Lượt chấm": top["count"].to_numpy(),
                                           "Điểm TB": [f"{fmt(v, 2)} ★" for v in top["mean"]]}),
                             hide_index=True, width="stretch")
            export = pd.DataFrame({
                "username": all_r["user_id"].map(names),
                "movie_id": svc.movies["movie_id"].to_numpy()[all_r["item"]],
                "title": svc.movies["title"].to_numpy()[all_r["item"]],
                "rating": all_r["rating"],
                "rated_at": to_local(all_r["rated_at"]).dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            })
            st.download_button("Tải toàn bộ lượt chấm (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                               "movierec_ratings.csv", "text/csv", icon=":material/download:",
                               help="Cột: tên đăng nhập, mã phim MovieLens, tên phim, số sao, thời điểm (giờ Việt Nam).")
    with tab_users:
        show = pd.DataFrame({
            "ID": users["id"], "Tên đăng nhập": users["username"],
            "Ngày tạo": to_local(users["created_at"]).dt.strftime("%d/%m/%Y"),
            "Số phim đã chấm": users["n_ratings"],
            "Điểm TB": [f"{fmt(v, 2)} ★" if pd.notna(v) else "–" for v in users["avg_rating"]],
            "Lần chấm gần nhất": to_local(users["last_rated"]).dt.strftime("%d/%m/%Y %H:%M").fillna("Chưa chấm"),
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Giờ Việt Nam (UTC+7).")
        others = users[users["id"] != st.session_state.user_id]
        if not others.empty:
            with st.form("delete_user_form", clear_on_submit=True):  # gửi xong thì bỏ chọn và bỏ đánh dấu
                who = st.selectbox("Xóa tài khoản (kèm toàn bộ lượt chấm và danh sách muốn xem)",
                                   others["id"], format_func=lambda i: names.loc[i], index=None,
                                   placeholder="Chọn tài khoản cần xóa")
                sure = st.checkbox("Tôi chắc chắn muốn xóa tài khoản này")
                if st.form_submit_button("Xóa tài khoản", type="primary"):
                    if who is None:
                        st.warning("Hãy chọn tài khoản cần xóa.")
                    elif not sure:
                        st.warning("Hãy đánh dấu “Tôi chắc chắn muốn xóa tài khoản này” để xác nhận.")
                    else:
                        n = int(users.set_index("id").loc[who, "n_ratings"])
                        db.delete_user(int(who))
                        st.session_state.admin_msg = f"Đã xóa tài khoản “{names.loc[who]}” và {n} lượt chấm của tài khoản này."
                        st.rerun()
    with tab_sys:
        semantic_ok = Path(SEMANTIC).exists()
        trailers = int(EXTRAS["youtube_key"].notna().sum()) if "youtube_key" in EXTRAS else 0
        from movierec.assistant import OllamaClient

        rows = [("Số phim trong kho", f"{len(svc.movies):,}".replace(",", ".")),
                ("Bộ xếp hạng hai tầng", "đã nạp" if svc.ranker is not None else "chưa có (chỉ dùng công thức trộn)"),
                ("Mô hình gợi ý cập nhật lúc", to_local(_mtime(ARTIFACT)).strftime("%d/%m/%Y %H:%M (giờ VN)")
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


def page_need_login(ratings: dict[int, int]) -> None:
    """Khách mở /phim-cua-toi bằng đường dẫn."""
    st.info("Trang này lưu các phim bạn đã chấm và danh sách muốn xem, nên cần đăng nhập.")
    if st.button("Đăng nhập hoặc tạo tài khoản", key="need_login_btn", icon=":material/login:", type="primary"):
        login_again()


def page_not_admin(ratings: dict[int, int]) -> None:
    """Người dùng thường mở /quan-tri bằng đường dẫn."""
    st.warning("Trang quản trị chỉ dành cho tài khoản quản trị. Bạn có thể quay lại trang “Gợi ý cho bạn” "
               "trên thanh điều hướng.")


def _header(title: str, ratings: dict[int, int]) -> None:
    """Dòng đầu trang: tiêu đề bên trái, menu tài khoản bên phải (thay cho thanh bên)."""
    liked = sum(r >= 4 for r in ratings.values())
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.title(title)
    with right:
        if is_guest():
            if st.button("Đăng nhập", key="login_from_guest", icon=":material/login:", type="primary", width="stretch"):
                login_again()
            return
        with st.popover(st.session_state.username, icon=":material/account_circle:", width="stretch"):
            st.metric("Số phim đã chấm", len(ratings))
            if liked >= MIN_RATINGS:
                st.caption(f"Đã thích {liked} phim")
            else:
                st.caption(f"Đã thích {liked}/{MIN_RATINGS} phim. Thích thêm {MIN_RATINGS - liked} phim (chấm 4-5★) "
                           "để gợi ý bám sát sở thích hơn.")
            if st.button("Đăng xuất", key="logout_btn", icon=":material/logout:", width="stretch"):
                login_again()


HOME_ALIAS = "trang-mac-dinh-movierec"  # đường dẫn nội bộ của trang mặc định (trang này luôn mở ở "/")


def _unknown_path(known: set[str]) -> str:
    """Đoạn cuối đường dẫn đang mở nếu không thuộc trang nào của ứng dụng (vd. /abc, /Tim-Phim, /goi-y khi đã đăng
    nhập vì trang mặc định nằm ở "/"), để đăng ký một trang ẩn đúng đường dẫn đó."""
    path = (st.context.url or "").split("?")[0].split("#")[0]
    path = path.split("://", 1)[-1].split("/", 1)[1] if "/" in path.split("://", 1)[-1] else ""
    path = path.strip("/")
    path = urllib.parse.unquote(path)
    return path if path and path not in known | {HOME_ALIAS} and "/" not in path and len(path) <= 200 else ""


def _go_home() -> None:
    if st.session_state.get("unknown_from") != "goi-y":
        st.toast("Không có trang này, đã chuyển về trang Gợi ý cho bạn.", icon=":material/info:")
    st.switch_page(st.session_state.home_page)


def main() -> None:
    st.logo(str(ROOT / "app" / "assets" / "logo.svg"), size="large")
    st.markdown(CSS, unsafe_allow_html=True)
    if "asked_page" not in st.session_state:  # trang người dùng mở bằng đường dẫn (vd. /phim-cua-toi)
        path = (st.context.url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        st.session_state.asked_page = path if path in PAGE_URLS - {"goi-y"} else ""
    if st.session_state.pop("clear_cookie", False):
        st.html(sess.cookie_script(None), unsafe_allow_javascript=True)
    if "user_id" not in st.session_state:
        restore_session()
    if "user_id" not in st.session_state:
        st.session_state.go_asked = True  # đăng nhập xong thì mở trang đã yêu cầu thay vì trang chủ
        # Trang đăng nhập có điều hướng riêng (ẩn): xóa thanh điều hướng của phiên trước (sau khi đăng xuất) và
        # nhận mọi đường dẫn của ứng dụng, để mở /phim-cua-toi khi chưa đăng nhập không bị báo "Page not found".
        auth = []
        for url in ["goi-y", *sorted(PAGE_URLS - {"goi-y"})]:
            run = lambda: page_auth()  # noqa: E731
            run.__name__ = "dang_nhap_" + url.replace("-", "_")
            # trang mặc định nằm ở "/", đường dẫn riêng của nó chỉ dùng nội bộ để không trùng với /goi-y
            auth.append(st.Page(run, title="Đăng nhập", url_path=HOME_ALIAS if url == "goi-y" else url,
                                default=url == "goi-y"))
        if extra := _unknown_path(PAGE_URLS - {"goi-y"}):
            auth.append(st.Page(lambda: page_auth(), title="Đăng nhập", url_path=extra, visibility="hidden"))
        st.navigation(auth, position="hidden").run()
        return
    guest = is_guest()
    ratings = {} if guest else db.ratings(st.session_state.user_id)
    st.session_state.watch_set = set() if guest else set(db.watchlist(st.session_state.user_id))

    # Thanh điều hướng ngang ở đầu trang. Mỗi trang là một hàm không tham số bọc hàm page_* tương ứng.
    specs = [(page_home, "Gợi ý cho bạn", ":material/recommend:", "goi-y"),
             (page_search, "Tìm phim", ":material/search:", "tim-phim"),
             (page_semantic, "Tìm theo mô tả", ":material/manage_search:", "tim-theo-mo-ta"),
             (page_assistant, "Trợ lý phim", ":material/smart_toy:", "tro-ly")]
    # Trang không có quyền vẫn được đăng ký (ẩn khỏi thanh điều hướng bằng CSS) để mở bằng đường dẫn thì hiện
    # lời giải thích, thay vì hộp "Page not found" tiếng Anh của Streamlit.
    hidden = []
    specs.append((page_need_login if guest else page_history, "Phim của tôi", ":material/bookmarks:", "phim-cua-toi"))
    hidden += ["phim-cua-toi"] if guest else []
    specs.append((page_admin if is_admin() else page_not_admin, "Quản trị", ":material/admin_panel_settings:",
                  "quan-tri"))
    hidden += [] if is_admin() else ["quan-tri"]
    if hidden:
        st.html("<style>" + ",".join(f'a[href$="/{u}"]' for u in hidden) + "{display:none!important}</style>")
    # trang mở khi vào địa chỉ gốc; start_page trong session_state cho phép mở thẳng một trang (dùng khi kiểm thử)
    start = st.session_state.get("start_page", "goi-y")
    allowed = {url for *_, url in specs}
    asked = st.session_state.get("asked_page", "")
    start = start if start in allowed else "goi-y"

    def page(fn, title, icon, url):
        run = lambda: fn(ratings)  # noqa: E731
        run.__name__ = url.replace("-", "_")
        return st.Page(run, title=title, icon=icon, url_path=HOME_ALIAS if url == start else url,
                       default=url == start)

    pages = [page(*spec) for spec in specs]
    st.session_state.home_page = next(p for p in pages if p._default)
    nav = list(pages)
    if extra := _unknown_path({p.url_path for p in pages if not p._default}):
        st.session_state.unknown_from = extra
        nav.append(st.Page(_go_home, title="Gợi ý cho bạn", url_path=extra, visibility="hidden"))
    current = st.navigation(nav, position="top")
    if st.session_state.pop("go_asked", False) and asked in allowed - set(hidden) and current.url_path != asked:
        st.switch_page(next(p for p in pages if p.url_path == asked))
    with st.container(key="mobile_nav", horizontal=True):  # chỉ hiện trên màn hình hẹp (xem CSS)
        for p in pages:
            if p.url_path not in hidden:
                st.page_link(p, label=p.title, icon=p.icon)
    if token := st.session_state.pop("set_cookie", None):  # sau switch_page để lần chạy bị bỏ dở không làm mất cookie
        st.html(sess.cookie_script(token), unsafe_allow_javascript=True)
    _header("Quản trị hệ thống" if current.title == "Quản trị" else current.title, ratings)
    current.run()
    if (item := st.session_state.pop("open_item", None)) is not None:  # bấm "Chi tiết" ở phim tương tự trong cửa sổ
        movie_dialog(item, ratings)
    st.divider()
    st.caption("Poster và mô tả phim lấy từ TMDB; sản phẩm dùng API của TMDB nhưng không được TMDB xác nhận hay chứng "
               "nhận. (This product uses the TMDB API but is not endorsed or certified by TMDB.)")

main()
