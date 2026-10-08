"""Kiểm thử CSDL, lớp phục vụ gợi ý và giao diện web (chạy không cần trình duyệt)."""
from pathlib import Path

import numpy as np
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
    assert any("Hãy chấm ít nhất" in i.value for i in at.info)  # màn hình cho người mới

    first_before = [s.key for s in at.selectbox if s.key.startswith("rec_")][0]
    onboarding = [s for s in at.selectbox if s.key.startswith("onb_")]
    onboarding[0].set_value(5).run()  # chấm 5 sao phim đầu tiên
    assert not at.exception
    db = Database(tmp_path / "web.db")
    assert list(db.ratings(1).values()) == [5]  # đã lưu vào SQLite
    rec_keys = [s.key for s in at.selectbox if s.key.startswith("rec_")]
    assert len(rec_keys) == 12 and f"rec_{onboarding[0].key.split('_')[1]}" not in rec_keys
    assert first_before  # danh sách gợi ý tồn tại trước và sau khi chấm

    at.sidebar.radio(key="nav").set_value("Phim đã chấm").run()
    assert not at.exception and len(at.dataframe) == 1

    # trang tìm theo mô tả: chưa có chỉ mục thì hiện hướng dẫn, không báo lỗi
    monkeypatch.setenv("MOVIEREC_SEMANTIC", str(tmp_path / "khong_ton_tai.npz"))
    at2 = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=30).run()
    at2.text_input(key="login_user").set_value("tester")
    at2.text_input(key="login_pw").set_value("12345")
    at2.button(key="login_btn").click().run()
    at2.sidebar.radio(key="nav").set_value("Tìm theo mô tả").run()
    assert not at2.exception and any("chưa được bật" in i.value for i in at2.info)
