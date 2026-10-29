"""Kiểm thử trợ lý hội thoại bằng LLM giả (không cần Ollama)."""
import json

import numpy as np
import pandas as pd

from movierec.assistant import Assistant, clean_intent, intent_schema, load_blurbs, save_blurbs, valid_blurb

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
    assert "hợp với yêu cầu" not in none["text"] and "chưa chắc khớp mô tả" in none["text"]  # không khẳng định sai


def test_description_always_searches_by_meaning():
    svc = FakeService()
    out = Assistant(svc, FakeSemantic(), make_llm(base(action="recommend", description="ghost story", genres=["Horror"])), k=2).reply("x", {})
    assert out["source"] == "semantic" and [m["item"] for m in out["movies"]] == [2, 5]
    out = Assistant(svc, FakeSemantic(), make_llm(base(action="recommend", genres=["Comedy"])), k=2).reply("x", {})
    assert out["source"] == "recommend" and [m["item"] for m in out["movies"]] == [1, 4]  # không mô tả: gợi ý cá nhân hóa + lọc thể loại
    assert out["text"].startswith("Đây là 2 phim được yêu thích nhất trên MovieLens thuộc thể loại Hài")  # chưa chấm gì


def test_ollama_client_retries_once_on_connection_error():
    from movierec.assistant import OllamaClient

    client = OllamaClient("fake"); client.retry_wait = 0
    calls = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise ConnectionResetError("Ollama đang nạp mô hình")
        return {"message": {"content": '{"ok": true}'}}
    client._post = flaky
    assert client([{"role": "user", "content": "x"}], schema={"type": "object"}) == '{"ok": true}'
    assert len(calls) == 2


def test_explain_rejects_echoed_instruction():
    """Mô hình chép lại câu hướng dẫn thì thử lại một lần; vẫn sai thì bỏ câu giới thiệu, không lưu vào bộ nhớ."""
    echo = "MỘT câu (dưới 25 từ) tóm tắt nội dung bộ phim, chỉ dùng thông tin có trong mô tả"
    assert valid_blurb("Một phi hành gia bị kẹt lại trên sao Hỏa.") and not valid_blurb(echo) and not valid_blurb("")
    assert not valid_blurb("Tình yêu cấm và sự thay đổi từ hài đến 悲剧.")
    assert not valid_blurb("Một người đàn ông trở về quê hương tìm lại người con gái bị bỏ rơi, giữa lúc thị trấn "
                           "cố cứu ngọn hải đăng cũ.")  # chép câu mẫu trong lời nhắc
    good = json.dumps({"reason": "Ba anh em xoay xở với tình yêu."})
    answers = iter([json.dumps({"reason": echo}), '{"reason": "lặp lặp lặp', json.dumps({"reason": "Một cựu杀手."}), good])
    temps = []

    def llm(m, schema=None, temperature=None):
        temps.append(temperature)
        return next(answers)
    bot = Assistant(FakeService(), FakeSemantic(), llm)
    movie = FakeService()._movie(1)
    assert bot.explain(movie) == ""  # chép lời nhắc, JSON bị cắt, chữ Hán: cả ba lần đều hỏng
    assert len(bot.rejected) == 3 and temps == [None, 0.6, 0.9]  # lần thử lại tăng temperature
    assert bot.explain(movie) == "Ba anh em xoay xở với tình yêu."  # lần sau gọi lại thì được
    bad = Assistant(FakeService(), FakeSemantic(), lambda m, schema=None, **kw: json.dumps({"reason": echo}))
    assert bad.explain(movie) == "" and movie["title"] not in bad._blurbs


def test_prewritten_blurbs_skip_llm(tmp_path):
    """Câu viết sẵn được dùng ngay, kể cả khi không có LLM; câu hỏng trong tệp bị bỏ khi nạp."""
    path = tmp_path / "blurbs.json"
    save_blurbs(path, {"Laugh Out (1995)": "Một danh hài cố gượng cười.", "Ghost Hall (1999)": "dưới 25 từ"})
    blurbs = load_blurbs(path)
    assert blurbs == {"Laugh Out (1995)": "Một danh hài cố gượng cười."} and load_blurbs(tmp_path / "x.json") == {}
    svc = FakeService()
    bot = Assistant(svc, FakeSemantic(), None, blurbs=blurbs)
    assert bot.explain(svc._movie(1)) == "Một danh hài cố gượng cười." and bot.explain(svc._movie(2)) == ""
    calls = []
    with_llm = Assistant(svc, FakeSemantic(), lambda m, schema=None, **kw: calls.append(1) or '{"reason": "X"}',
                         blurbs=blurbs)
    assert with_llm.explain(svc._movie(2)) == "" and calls == []  # có bộ câu viết sẵn: phim thiếu câu không gọi LLM


def test_rule_intent_without_llm():
    from movierec.assistant import rule_intent

    genres = ["Comedy", "Horror", "Romance", "Sci-Fi", "Children's"]
    assert rule_intent("xin chào", genres)["action"] == "chat"
    sim = rule_intent("phim giống The Matrix nhé", genres)
    assert sim["action"] == "similar" and sim["similar_to"] == "The Matrix"
    more = rule_intent("còn phim nào khác không", genres)
    assert more["more"] and more["description"] == ""
    mix = rule_intent("phim hài thập niên 90, đừng có lãng mạn", genres)
    assert mix["genres"] == ["Comedy"] and mix["exclude_genres"] == ["Romance"]
    assert (mix["year_from"], mix["year_to"]) == (1990, 1999)
    assert rule_intent("phim kinh dị sau 1995", genres)["year_from"] == 1995
    assert rule_intent("phim hoạt hình cho trẻ em", genres)["genres"] == ["Children's"]


def test_more_and_similar_work_without_llm():
    svc = FakeService()
    bot, state = Assistant(svc, None, None, k=2), {}
    first = bot.reply("phim hài", {}, state)
    again = bot.reply("còn phim nào khác không", {}, state)
    assert first["movies"] and not {m["item"] for m in first["movies"]} & {m["item"] for m in again["movies"]}
    assert bot.reply("chào bạn", {}, {})["source"] == "chat"
