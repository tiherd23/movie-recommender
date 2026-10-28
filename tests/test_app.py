"""Kiểm thử CSDL, lớp phục vụ gợi ý và giao diện web (chạy không cần trình duyệt)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movierec.data import build_dataset
from movierec.db import Database
from movierec.service import RecommenderService


@pytest.fixture(scope="module")
def service(fake_ml1m, tmp_path_factory):
    svc = RecommenderService.build(build_dataset(fake_ml1m),
                                   {"ImplicitALS": {"factors": 8, "alpha": 1.0, "min_rating": 4.0, "iters": 5},
                                    "ItemKNN": {"k": 20, "shrink": 0.0}})
    path = tmp_path_factory.mktemp("art") / "service.pkl"
    svc.save(path)
    return RecommenderService.load(path), path


def test_service_uses_two_stage_when_enough_likes(service):
    svc, _ = service
    assert svc.ranker is not None
    group = lambda item: (item + 1) % 4
    liked = dict(list({i: 5 for i in range(len(svc.movies)) if group(i) == 2}.items())[:8])
    recs = svc.recommend(liked, k=10)
    assert recs and all(r["method"] == "two_stage" for r in recs)
    assert all(r["item"] not in liked for r in recs)
    assert np.mean([group(r["item"]) == 2 for r in recs]) >= 0.8
    few = svc.recommend(dict(list(liked.items())[:2]), k=5)  # ít phim thích -> công thức trộn
    assert all(r["method"] == "blend" for r in few)
    assert all(1 <= len(r["why"]) <= 4 for r in recs) and not any("why" in r for r in few)


def test_shap_contributions_add_up_to_ranker_score(service):
    """Tổng giá trị SHAP theo nhóm (kể cả 'base') phải bằng đúng điểm của bộ xếp hạng."""
    from movierec.ranker import grouped_contributions

    svc, _ = service
    liked = {i: 5 for i in range(8)}
    als = svc.Y @ svc._fold_in(list(liked))
    score, items, X = svc._rank_two_stage(liked, list(liked), als)
    contrib = grouped_contributions(svc.ranker["booster"], X)
    assert np.allclose(contrib.sum(axis=1).to_numpy(), score[items], atol=1e-4)


def test_service_diversity_option(service):
    svc, _ = service
    liked = {i: 5 for i in range(8)}
    base, div = svc.recommend(liked, k=8), svc.recommend(liked, k=8, diversity=0.4)
    assert len(div) == 8 and div[0]["item"] == base[0]["item"]
    assert all(r["item"] not in liked for r in div)


def test_database_accounts_and_ratings(tmp_path):
    db = Database(tmp_path / "app.db")
    uid = db.register("phu", "matkhau")
    assert db.login("phu", "matkhau") == uid and db.login("phu", "sai") is None
    with pytest.raises(ValueError):
        db.register("phu", "khac1")  # trùng tên
    db.rate(uid, 3, 5); db.rate(uid, 3, 2); db.rate(uid, 7, 4)
    assert db.ratings(uid) == {7: 4, 3: 2} or db.ratings(uid) == {3: 2, 7: 4}
    db.rate(uid, 3, None)
    assert db.ratings(uid) == {7: 4}
    db.set_genres(uid, ["Drama", "Sci-Fi"])
    assert db.get_genres(uid) == ["Drama", "Sci-Fi"]


def test_service_personalizes_for_new_user(service):
    svc, _ = service
    # Dữ liệu giả: phim có movie_id % 4 == g thuộc nhóm g. Chỉ số item = movie_id - 1.
    group = lambda item: (item + 1) % 4
    liked = {i: 5 for i in range(len(svc.movies)) if group(i) == 1}
    liked = dict(list(liked.items())[:6])
    recs = svc.recommend(liked, k=10)
    assert all(r["item"] not in liked for r in recs)  # không gợi ý lại phim đã chấm
    assert np.mean([group(r["item"]) == 1 for r in recs]) >= 0.8  # đúng nhóm sở thích
    assert recs[0]["reason"].startswith("Vì bạn thích")
    assert len(svc.recommend({}, k=5)) == 5  # người chưa chấm gì vẫn có gợi ý
    cold = svc.recommend({}, ["Drama"], k=5)  # cold-start theo thể loại
    assert all("Drama" in r["genres"] for r in cold)
    assert all(r["reason"] == "Hợp thể loại bạn chọn: Chính kịch" for r in cold)
    top = svc.recommend({}, k=1)[0]
    assert top["reason"].startswith("Hạng 1/")  # lý do phổ biến có số liệu
    assert svc.search("movie 12")[0]["title"].lower().startswith("movie 12")
    assert all(s["item"] != 0 for s in svc.similar(0))


def test_web_register_rate_and_recommend(service, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _, artifact = service
    monkeypatch.setenv("MOVIEREC_DB", str(tmp_path / "web.db"))
    monkeypatch.setenv("MOVIEREC_ARTIFACT", str(artifact))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=30).run()
    assert not at.exception and at.title[0].value.endswith("MovieRec")

    at.text_input(key="reg_user").set_value("tester")
    at.text_input(key="reg_pw").set_value("12345")
    at.button(key="reg_btn").click().run()
    assert not at.exception and at.title[0].value == "Gợi ý cho bạn"
    assert [t.label for t in at.tabs] == ["⭐ Phim dành cho bạn", "🎞️ Chấm thêm phim"]  # người mới: có tab chấm phim

    rec_before = [f.key for f in at.feedback if f.key.startswith("rec_")]
    onboarding = [f for f in at.feedback if f.key.startswith("onb_")]
    assert len(onboarding) == 18 and len(rec_before) == 18
    first = onboarding[0].key.split("_")[1]
    onboarding[0].set_value(4).run()  # bấm ngôi sao thứ 5 của phim đầu tiên
    assert not at.exception
    db = Database(tmp_path / "web.db")
    assert db.ratings(1) == {int(first): 5}  # đã lưu 5 sao vào SQLite
    assert f"onb_{first}" not in [f.key for f in at.feedback]  # phim vừa chấm rời khỏi danh sách cần chấm
    assert f"rec_{first}" not in [f.key for f in at.feedback if f.key.startswith("rec_")]

    at.session_state["start_page"] = "phim-cua-toi"
    at.run()
    assert at.title[0].value == "Phim của tôi"
    hist = [f for f in at.feedback if f.key.startswith("hist_")]
    assert not at.exception and len(hist) == 1 and hist[0].value == 4
    hist[0].set_value(None).run()  # bấm lại ngôi sao đang chọn: bỏ chấm
    assert not at.exception and db.ratings(1) == {}

    # trang tìm theo mô tả: chưa có chỉ mục thì hiện hướng dẫn, không báo lỗi
    monkeypatch.setenv("MOVIEREC_SEMANTIC", str(tmp_path / "khong_ton_tai.npz"))
    at2 = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=30).run()
    at2.text_input(key="login_user").set_value("tester")
    at2.text_input(key="login_pw").set_value("12345")
    at2.button(key="login_btn").click().run()
    at2.session_state["start_page"] = "tim-theo-mo-ta"
    at2.run()
    assert not at2.exception and any("chưa sẵn sàng" in i.value for i in at2.info)


def test_watchlist_and_admin_queries(tmp_path):
    db = Database(tmp_path / "x.db")
    a, b = db.register("alice", "1234"), db.register("bob", "1234")
    assert db.toggle_watch(a, 7) and db.toggle_watch(a, 3) and db.watchlist(a) == [3, 7]
    assert not db.toggle_watch(a, 7) and db.watchlist(a) == [3]  # bấm lần hai: bỏ khỏi danh sách
    db.rate(a, 1, 5); db.rate(b, 1, 3); db.rate(b, 2, 4)
    st = db.stats()
    assert st["users"] == 2 and st["ratings"] == 3 and st["watchlist"] == 1 and st["active_7d"] == 2
    assert [u["n_ratings"] for u in db.users_overview()] == [1, 2]
    db.delete_user(a)
    assert db.stats()["users"] == 1 and db.watchlist(a) == [] and len(db.all_ratings()) == 2


def _open_app(service, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _, artifact = service
    monkeypatch.setenv("MOVIEREC_DB", str(tmp_path / "web.db"))
    monkeypatch.setenv("MOVIEREC_ARTIFACT", str(artifact))
    monkeypatch.setenv("MOVIEREC_SEMANTIC", str(tmp_path / "khong_ton_tai.npz"))
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
                             default_timeout=30).run()


def test_guest_can_browse_but_not_rate(service, tmp_path, monkeypatch):
    at = _open_app(service, tmp_path, monkeypatch)
    at.button(key="guest_btn").click().run()
    assert not at.exception and at.title[0].value == "Gợi ý cho bạn"
    assert len(at.feedback) == 0  # khách không chấm được: không có hàng sao
    assert any("Đăng nhập để chấm điểm" in c.value for c in at.caption)
    at.session_state["start_page"] = "phim-cua-toi"  # khách không có trang này: quay về trang gợi ý
    at.run()
    assert at.title[0].value == "Gợi ý cho bạn"
    at.session_state["start_page"] = "tim-phim"
    at.run()
    assert at.title[0].value == "Tìm phim"
    at.text_input(key="search_q").set_value("movie 1").run()
    assert not at.exception
    assert Database(tmp_path / "web.db").stats()["users"] == 0  # không tạo tài khoản nào


def test_admin_page_only_for_admin(service, tmp_path, monkeypatch):
    at = _open_app(service, tmp_path, monkeypatch)
    at.text_input(key="reg_user").set_value("tester")
    at.text_input(key="reg_pw").set_value("12345")
    at.button(key="reg_btn").click().run()
    at.session_state["start_page"] = "quan-tri"  # người dùng thường không có trang quản trị
    at.run()
    assert at.title[0].value == "Gợi ý cho bạn"
    at.button(key="logout_btn").click().run()
    at.text_input(key="reg_user").set_value("admin")  # tên quản trị không tự đăng ký được trên web
    at.text_input(key="reg_pw").set_value("12345")
    at.button(key="reg_btn").click().run()
    assert "dành riêng" in at.error[0].value and Database(tmp_path / "web.db").stats()["users"] == 1
    Database(tmp_path / "web.db").register("admin", "12345")  # như scripts/create_admin.py
    at.text_input(key="login_user").set_value("admin")
    at.text_input(key="login_pw").set_value("12345")
    at.button(key="login_btn").click().run()
    at.session_state["start_page"] = "quan-tri"
    at.run()
    assert not at.exception and at.title[0].value == "Quản trị hệ thống"
    assert [m.value for m in at.metric if m.label == "Tài khoản"] == ["2"]  # hai tài khoản: tester và admin
    at.button(key="FormSubmitter:delete_user_form-Xóa tài khoản").click().run()  # chưa xác nhận: chỉ cảnh báo
    assert any("Tôi chắc chắn" in w.value for w in at.warning) and Database(tmp_path / "web.db").stats()["users"] == 2
    at.checkbox[0].check()
    at.button(key="FormSubmitter:delete_user_form-Xóa tài khoản").click().run()
    assert not at.exception and Database(tmp_path / "web.db").stats()["users"] == 1
    assert any("Đã xóa tài khoản “tester”" in m.value for m in at.success)


def test_item_stats_match_ratings(service, fake_ml1m):
    """Số lượt chấm, điểm TB, tỉ lệ thích hiển thị trên thẻ phim khớp đúng dữ liệu gốc."""
    svc, _ = service
    ds = build_dataset(fake_ml1m)
    full = pd.concat([ds.train, ds.val, ds.test])
    g = full.groupby("item")["rating"].agg(["count", "mean"])
    like = full.assign(l=full["rating"] >= 4).groupby("item")["l"].mean()
    for j in g.index[:20]:
        st = svc.stats(int(j))
        assert st["n"] == g.loc[j, "count"]
        assert abs(st["mean"] - g.loc[j, "mean"]) < 1e-3
        assert abs(st["like_frac"] - like.loc[j]) < 1e-4
    assert svc.stats(int(np.argmax(svc.pop)))["rank"] == 1


def test_genres_count_in_two_stage_mode(service):
    """Đã đủ phim thích (kiến trúc hai tầng) mà chọn thể loại thì ít nhất một nửa danh sách thuộc thể loại đó."""
    import copy

    svc = copy.copy(service[0])  # dữ liệu giả chỉ có Drama|Comedy: gán "Horror" cho nhóm phim người này không thích
    group = lambda item: (item + 1) % 4
    svc.movies = svc.movies.copy()
    svc.movies["genres"] = ["Horror" if group(i) == 3 else g for i, g in enumerate(svc.movies["genres"])]
    liked = dict(list({i: 5 for i in range(len(svc.movies)) if group(i) == 1}.items())[:6])
    base = svc.recommend(liked, k=10)
    assert base[0]["method"] == "two_stage"
    target = "Horror"
    assert sum(target in r["genres"] for r in base) < 5
    recs = svc.recommend(liked, [target], k=10)
    assert sum(target in r["genres"].split("|") for r in recs) >= 5
    assert len({r["item"] for r in recs}) == 10 and not set(liked) & {r["item"] for r in recs}
    assert recs[0]["item"] == base[0]["item"]  # phim mô hình chấm cao nhất vẫn đứng đầu
    assert any(r["reason"] == "Hợp thể loại bạn chọn: Kinh dị" for r in recs)


def test_search_ignores_case_articles_and_year():
    from movierec.service import _norm_text, display_title

    assert display_title("Matrix, The (1999)") == "The Matrix (1999)"
    assert display_title("Usual Suspects, The (1995)", year=False) == "The Usual Suspects"
    assert display_title("Toy Story (1995)") == "Toy Story (1995)"
    assert _norm_text("Amélie, the MATRIX!") == ["amelie", "the", "matrix"]


def test_search_matches_words(service):
    svc, _ = service
    first = svc.movies["title"].iloc[11]  # dữ liệu giả: "Movie 12 (...)"
    assert svc.search(first.upper().replace("(", ""))[0]["title"] == first
    assert svc.search("zzzz khong co") == [] and svc.search_all("   ")[1] == 0
