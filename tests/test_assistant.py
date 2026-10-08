"""Kiểm thử trợ lý hội thoại bằng LLM giả (không cần Ollama)."""
import json

import numpy as np
import pandas as pd

from movierec.assistant import Assistant, clean_intent, intent_schema

GENRES = ["Action", "Comedy", "Horror"]


class FakeService:
    """Kho 6 phim, đủ các phương thức mà trợ lý dùng."""
    genres = GENRES

    def __init__(self):
        self.movies = pd.DataFrame({
            "title": ["Alien Night (1979)", "Laugh Out (1995)", "Ghost Hall (1999)",
                      "Speed Run (1994)", "Funny Ghost (1988)", "Old Fear (1960)"],
            "genres": ["Horror|Action", "Comedy", "Horror", "Action", "Comedy|Horror", "Horror"]})
        self.pop = np.array([50.0, 40.0, 30.0, 20.0, 10.0, 5.0])

    def _movie(self, j, **extra):
        row = self.movies.iloc[j]
        return {"item": int(j), "title": row["title"], "genres": row["genres"], "overview": "mo ta", **extra}

    def search(self, query, limit=20):
        hits = [j for j, t in enumerate(self.movies["title"]) if query.lower() in t.lower()]
        return [self._movie(j) for j in hits[:limit]]

    def similar(self, item, k=8):
        return [self._movie(j) for j in range(len(self.movies)) if j != item][:k]

    def recommend(self, ratings, genres=None, k=12):
        return [self._movie(j, reason="pop") for j in range(len(self.movies)) if j not in ratings][:k]


class FakeSemantic:
    def search(self, query, k=12, exclude=None, popularity=None, pop_weight=None):
        order = [2, 5, 0, 4, 1, 3] if "ghost" in query.lower() else [1, 4, 3, 0, 2, 5]
        return [(i, 0.9 - 0.1 * n) for n, i in enumerate(order)][:k]


def make_llm(intent):
    """LLM giả: có schema thì trả ý định cho trước, không có thì trả lời đáp cố định."""
    calls = []

    def llm(messages, schema=None):
        calls.append((messages, schema))
        props = schema["properties"]
        if "action" in props:
            return json.dumps(intent)
        if "reply" in props:
            return json.dumps({"reply": "CHAO"})
        return json.dumps({"reason": "LY_DO"})
    llm.calls = calls
    return llm


def base(**kw):
    return {"action": "describe", "description": "", "genres": [], "exclude_genres": [], "similar_to": "",
            "year_from": 0, "year_to": 0, "more": False, **kw}


def test_clean_intent_drops_invalid_values():
    out = clean_intent({"action": "hack", "genres": ["Horror", "Nope"], "year_from": "1990", "year_to": 5,
                        "exclude_genres": "Comedy"}, GENRES)
    assert out["action"] == "describe" and out["genres"] == ["Horror"] and out["exclude_genres"] == []
    assert out["year_from"] == 1990 and out["year_to"] == 0 and out["more"] is False
    assert set(intent_schema(GENRES)["required"]) == set(out)


def test_describe_uses_semantic_and_filters():
    llm = make_llm(base(description="a ghost in a house", year_from=1980, exclude_genres=["Comedy"]))
    bot = Assistant(FakeService(), FakeSemantic(), llm, k=3)
    out = bot.reply("phim ma sau năm 1980, đừng hài", ratings={})
    assert out["source"] == "semantic"
    assert out["text"].startswith("Mình tìm được 2 phim") and "- **Ghost Hall (1999)**: LY_DO" in out["text"]
    assert [m["item"] for m in out["movies"]] == [2, 3]  # 5 (1960), 0 (1979) bị lọc năm; 4, 1 bị lọc thể loại
    seen = " ".join(c[0][-1]["content"] for c in llm.calls[1:])
    bot.write("x", out["movies"])  # lần hai lấy từ bộ nhớ, không gọi LLM
    assert len(llm.calls) == 3 and "Ghost Hall (1999)" in seen and "Old Fear" not in seen  # mỗi phim một lần gọi, chỉ phim đã chọn


def test_similar_recommend_chat_and_more():
    svc = FakeService()
    out = Assistant(svc, FakeSemantic(), make_llm(base(action="similar", similar_to="Alien Night")), k=2).reply("x", {})
    assert out["source"] == "similar" and [m["item"] for m in out["movies"]] == [1, 2]

    bot = Assistant(svc, FakeSemantic(), make_llm(base(action="recommend", genres=["Horror"])), k=2)
    state = {}
    first = bot.reply("gợi ý phim kinh dị", {0: 5}, state)
    assert first["source"] == "recommend" and [m["item"] for m in first["movies"]] == [2, 4]  # bỏ phim đã chấm
    bot.llm = make_llm(base(action="chat", more=True))
    second = bot.reply("còn phim nào khác không", {0: 5}, state)
    assert [m["item"] for m in second["movies"]] == [5]  # lặp lại yêu cầu trước, bỏ phim đã đưa ra
    assert len(state["history"]) == 4

    chat = Assistant(svc, FakeSemantic(), make_llm(base(action="chat"))).reply("chào bạn", {})
    assert chat["movies"] == [] and chat["source"] == "chat" and chat["text"] == "CHAO"
    seen = []
    Assistant(svc, FakeSemantic(), make_llm(base(description="ghost")), k=2).reply("x", {}, on_movies=seen.append)
    assert [m["item"] for m in seen[0]] == [2, 5]  # giao diện nhận phim trước khi LLM viết lời đáp


def test_works_without_llm_or_when_llm_fails():
    svc = FakeService()
    out = Assistant(svc, FakeSemantic(), None, k=2).reply("ghost", {})
    assert [m["item"] for m in out["movies"]] == [2, 5] and "Ghost Hall" in out["text"]

    def broken(messages, schema=None):
        raise ConnectionError("Ollama chưa chạy")
    out = Assistant(svc, FakeSemantic(), broken, k=2).reply("ghost", {})
    assert [m["item"] for m in out["movies"]] == [2, 5] and "Ghost Hall" in out["text"]
    none = Assistant(svc, None, None, k=2).reply("ghost", {})  # chưa có chỉ mục ngữ nghĩa: lùi về gợi ý thường
    assert none["source"] == "recommend" and len(none["movies"]) == 2


def test_description_always_searches_by_meaning():
    svc = FakeService()
    out = Assistant(svc, FakeSemantic(), make_llm(base(action="recommend", description="ghost story", genres=["Horror"])), k=2).reply("x", {})
    assert out["source"] == "semantic" and [m["item"] for m in out["movies"]] == [2, 5]
    out = Assistant(svc, FakeSemantic(), make_llm(base(action="recommend", genres=["Comedy"])), k=2).reply("x", {})
    assert out["source"] == "recommend" and [m["item"] for m in out["movies"]] == [1, 4]  # không mô tả: gợi ý cá nhân hóa + lọc thể loại
