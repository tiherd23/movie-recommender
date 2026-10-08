"""Trợ lý hội thoại: người dùng chat tiếng Việt, hệ thống trả lời kèm danh sách phim.

Mô hình ngôn ngữ (LLM, chạy cục bộ qua Ollama) KHÔNG tự nghĩ ra phim. Nó chỉ làm hai việc hẹp:

    1. Hiểu câu chat  -> ý định có cấu trúc (JSON): hành động, mô tả, thể loại, khoảng năm, phim mẫu
    2. Viết lời đáp   -> diễn giải bằng tiếng Việt vì sao các phim ĐÃ ĐƯỢC CHỌN hợp với yêu cầu

Việc chọn phim do các mô hình gợi ý sẵn có đảm nhận (tìm theo ngữ nghĩa, phim tương tự, gợi ý cá nhân hóa),
nên danh sách phim luôn nằm trong kho dữ liệu và không bị LLM bịa ra.

    câu chat -> [LLM: rút ý định] -> [truy hồi: semantic / similar / recommend] -> [lọc thể loại, năm]
             -> [LLM: viết lời đáp dựa trên danh sách phim] -> lời đáp + thẻ phim
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

DEFAULT_LLM = os.environ.get("MOVIEREC_LLM", "qwen3:4b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ACTIONS = ["describe", "similar", "recommend", "chat"]
LLM = Callable[..., str]  # llm(messages, schema=None) -> nội dung trả lời

INTENT_PROMPT = """You turn a Vietnamese chat message about movies into a JSON request.
Fields:
- action: "similar" if the user names a specific movie and wants movies like it;
  "describe" if the user describes plot, content, theme or mood of the movie they want;
  "recommend" if the user only asks for suggestions (maybe by genre or year) without describing content;
  "chat" for greetings, thanks, or anything that is not a request for movies.
- description: short ENGLISH description of the wanted content (empty string if none).
- genres: genres the user explicitly wants, only from this list: {genres}. Empty list if none.
- exclude_genres: genres the user explicitly does NOT want, from the same list.
- similar_to: English title of the movie the user named (empty string if none).
- year_from, year_to: release year range, 0 when not mentioned ("thập niên 90" -> 1990 and 1999).
- more: true if the user asks for more/other results of the previous request.
Use the earlier conversation only to resolve references. Return JSON only."""

REPLY_PROMPT = """Bạn dịch và rút gọn mô tả phim sang tiếng Việt tự nhiên cho trợ lý MovieRec.
Trả về JSON với "reason": MỘT câu (dưới 25 từ) tóm tắt nội dung bộ phim, chỉ dùng thông tin có trong mô tả,
không thêm chi tiết, không nhắc lại tên phim, không nhận xét."""

CHAT_PROMPT = """Bạn là trợ lý gợi ý phim của MovieRec. Trả về JSON với "reply" là một hoặc hai câu tiếng Việt.
Không tự nêu tên phim. Hãy mời người dùng mô tả nội dung, thể loại hoặc một phim họ thích để bạn tìm giúp."""

CHAT_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
ASK_MORE = "Bạn muốn xem phim như thế nào? Hãy mô tả nội dung hoặc kể một phim bạn thích."


# Lời đáp cũng ép theo JSON vì Qwen3 không tắt được chế độ suy luận khi viết tự do.
REASON_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}


class OllamaClient:
    """Gọi API /api/chat của Ollama bằng thư viện chuẩn (không thêm phụ thuộc)."""

    def __init__(self, model: str = DEFAULT_LLM, host: str = DEFAULT_HOST, timeout: float = 120.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        gpu = os.environ.get("MOVIEREC_LLM_NUM_GPU", "")
        self.num_gpu: int | None = int(gpu) if gpu else None  # số lớp đưa lên GPU; 0 = chỉ CPU, None = Ollama tự chọn
        self._think = False  # tắt chế độ suy luận của Qwen3 để trả lời nhanh; None = mô hình không hỗ trợ tham số này

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 400:
                raise
            body = err.read().decode("utf-8", "replace")[:500]  # Ollama ghi nguyên nhân trong thân phản hồi
            raise OSError(f"Ollama trả lỗi {err.code} với mô hình {self.model}: {body}") from None

    def available(self) -> bool:
        """Ollama đang chạy và đã tải mô hình cần dùng."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                names = [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
        except (OSError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    def __call__(self, messages: list[dict], schema: dict | None = None) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                   "options": {"temperature": 0.0 if schema else 0.4, "num_ctx": 4096,
                               "num_predict": 400}}  # chặn độ dài để giới hạn độ trễ trên CPU
        if schema:
            payload["format"] = schema  # structured output: Ollama ép câu trả lời theo JSON schema
        if self._think is not None:
            payload["think"] = self._think
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        try:
            data = self._post(payload)
        except urllib.error.HTTPError as err:
            if err.code != 400 or "think" not in payload:
                raise
            self._think = None  # mô hình không có chế độ suy luận (ví dụ qwen2.5): gọi lại không kèm tham số
            payload.pop("think")
            data = self._post(payload)
        except OSError as err:
            if "CUDA" not in str(err) or self.num_gpu == 0:
                raise
            self.num_gpu = 0  # GPU lỗi khi suy luận: từ giờ chạy bằng CPU
            payload["options"]["num_gpu"] = 0
            data = self._post(payload)
        text = data.get("message", {}).get("content", "")
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def intent_schema(genres: list[str]) -> dict:
    genre_list = {"type": "array", "items": {"type": "string", "enum": genres}}
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": ACTIONS}, "description": {"type": "string"},
                           "genres": genre_list, "exclude_genres": genre_list, "similar_to": {"type": "string"},
                           "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
                           "more": {"type": "boolean"}},
            "required": ["action", "description", "genres", "exclude_genres", "similar_to",
                         "year_from", "year_to", "more"]}


def clean_intent(raw: dict, genres: list[str]) -> dict:
    """Chuẩn hóa JSON từ LLM: bỏ giá trị lạ, để phần còn lại của hệ thống không phải tin vào LLM."""
    def year(v) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return v if 1900 <= v <= 2100 else 0

    def keep(v) -> list[str]:
        return [g for g in (v if isinstance(v, list) else []) if g in genres]

    action = raw.get("action")
    return {"action": action if action in ACTIONS else "describe",
            "description": str(raw.get("description") or "").strip(),
            "genres": keep(raw.get("genres")), "exclude_genres": keep(raw.get("exclude_genres")),
            "similar_to": str(raw.get("similar_to") or "").strip(),
            "year_from": year(raw.get("year_from")), "year_to": year(raw.get("year_to")),
            "more": bool(raw.get("more"))}


class Assistant:
    def __init__(self, service, semantic=None, llm: LLM | None = None, k: int = 5):
        self.svc, self.semantic, self.llm, self.k = service, semantic, llm, k
        years = service.movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
        self.years = years.fillna(0).astype(int).to_numpy()
        self.genre_sets = [set(g.split("|")) for g in service.movies["genres"]]
        self._blurbs: dict[int, str] = {}  # câu giới thiệu đã viết, theo chỉ số phim

    # ---------- bước 1: hiểu câu chat ----------
    def parse(self, message: str, history: list[dict] | None = None) -> dict:
        """Ý định của câu chat. Không có LLM (hoặc LLM lỗi): coi cả câu là một mô tả nội dung."""
        fallback = clean_intent({"action": "describe", "description": message}, self.svc.genres)
        if self.llm is None:
            return fallback
        messages = [{"role": "system", "content": INTENT_PROMPT.format(genres=", ".join(self.svc.genres))}]
        messages += [{"role": m["role"], "content": m["text"]} for m in (history or [])[-6:]]
        messages.append({"role": "user", "content": message})
        try:
            return clean_intent(json.loads(self.llm(messages, schema=intent_schema(self.svc.genres))), self.svc.genres)
        except (OSError, ValueError, KeyError):
            return fallback

    # ---------- bước 2: chọn phim bằng các mô hình gợi ý ----------
    def _allowed(self, intent: dict, strict_genres: bool) -> np.ndarray:
        want, avoid = set(intent["genres"]), set(intent["exclude_genres"])
        ok = np.ones(len(self.years), dtype=bool)
        if intent["year_from"]:
            ok &= self.years >= intent["year_from"]
        if intent["year_to"]:
            ok &= (self.years <= intent["year_to"]) & (self.years > 0)
        if avoid:
            ok &= np.array([not (avoid & g) for g in self.genre_sets])
        if want and strict_genres:
            ok &= np.array([bool(want & g) for g in self.genre_sets])
        return ok

    def retrieve(self, intent: dict, message: str, ratings: dict[int, int], shown: set[int]) -> tuple[list[dict], str]:
        """Trả về (danh sách phim, nguồn). Nguồn: semantic | similar | recommend."""
        skip = set(ratings) | shown
        pool = len(self.years)  # lấy toàn bộ bảng xếp hạng rồi mới lọc, để bộ lọc năm/thể loại không làm rỗng kết quả
        action = intent["action"]
        cands: list[dict] = []
        source = "recommend"
        if action == "similar" and intent["similar_to"]:
            hits = self.svc.search(intent["similar_to"], limit=1)
            if hits:
                base = hits[0]
                skip.add(base["item"])
                cands = [dict(m, reason=f"Tương tự {base['title']}") for m in self.svc.similar(base["item"], k=pool)]
                source = "similar"
        # LLM nhỏ đôi khi gán "recommend" cho câu có mô tả nội dung; cứ có mô tả là tìm theo nghĩa (thể loại, năm vẫn được lọc)
        by_meaning = action in ("describe", "similar") or bool(intent["description"])
        if not cands and by_meaning and self.semantic is not None:
            query = intent["description"] or message
            hits = self.semantic.search(query, k=pool, popularity=self.svc.pop)
            cands = [self.svc._movie(i, reason=f"Độ khớp nội dung: {s:.0%}") for i, s in hits]
            source = "semantic"
        if not cands:
            cands = self.svc.recommend(ratings, intent["genres"], k=pool)
            source = "recommend"
        ok = self._allowed(intent, strict_genres=True)
        picked = [m for m in cands if m["item"] not in skip and ok[m["item"]]]
        return picked[: self.k], source

    # ---------- bước 3: viết lời đáp ----------
    def _fallback_text(self, movies: list[dict]) -> str:
        return "Mình chưa tìm được phim nào khớp với yêu cầu này. Bạn thử mô tả khác đi hoặc nới điều kiện nhé."

    def explain(self, movie: dict) -> str:
        """Một câu giới thiệu tiếng Việt cho MỘT phim, chỉ dựa trên mô tả của chính phim đó.

        Mỗi phim một lần gọi LLM và KHÔNG đưa câu hỏi của người dùng vào: khi thử, mô hình nhỏ nhận cả danh sách
        thì gán nhầm nội dung giữa các phim, còn khi biết yêu cầu thì bịa thêm chi tiết cho khớp yêu cầu.
        Kết quả được nhớ lại theo phim nên lần sau không phải gọi LLM nữa."""
        overview = str(movie.get("overview") or "").strip()
        if self.llm is None or not overview:
            return ""
        if movie["item"] in self._blurbs:
            return self._blurbs[movie["item"]]
        info = f"Phim: {movie['title']}\nThể loại: {movie['genres'].replace('|', ', ')}\nMô tả: {overview[:400]}"
        try:
            raw = json.loads(self.llm([{"role": "system", "content": REPLY_PROMPT}, {"role": "user", "content": info}],
                                      schema=REASON_SCHEMA))
            text = str(raw.get("reason") or "").strip()
        except (OSError, ValueError, AttributeError):
            return ""
        if text:
            self._blurbs[movie["item"]] = text
        return text

    def write(self, message: str, movies: list[dict]) -> str:
        """Lời đáp: câu mở đầu cố định + mỗi phim một dòng. Tên phim lấy từ dữ liệu, LLM chỉ viết câu giới thiệu."""
        if not movies:
            return self._fallback_text(movies)
        lines = []
        for m in movies:
            reason = self.explain(m)
            lines.append(f"- **{m['title']}**" + (f": {reason}" if reason else ""))
        return "\n".join([f"Mình tìm được {len(movies)} phim hợp với yêu cầu của bạn:", ""] + lines)

    # ---------- một lượt hội thoại ----------
    def reply(self, message: str, ratings: dict[int, int], state: dict | None = None,
              on_movies: Callable[[list[dict]], None] | None = None) -> dict:
        """on_movies: được gọi ngay khi đã chọn xong phim, trước bước viết lời đáp (bước chậm nhất),
        để giao diện hiện thẻ phim sớm.
        state (lưu giữa các lượt): history = các lượt trước, shown = phim đã đưa ra, intent = ý định gần nhất."""
        state = state if state is not None else {}
        history, shown = state.setdefault("history", []), state.setdefault("shown", set())
        intent = self.parse(message, history)
        last = state.get("intent")
        if intent["more"] and last:  # "còn phim nào khác không": lặp lại yêu cầu trước, bỏ các phim đã đưa ra
            intent = dict(last, more=True)
        if intent["action"] == "chat":
            movies, source = [], "chat"
            text = ASK_MORE if self.llm is None else self._chat(message)
        else:
            if not intent["more"]:
                shown.clear()
            movies, source = self.retrieve(intent, message, ratings, shown)
            if on_movies and movies:
                on_movies(movies)
            text = self.write(message, movies)
            state["intent"] = intent
            shown.update(m["item"] for m in movies)
        history += [{"role": "user", "text": message}, {"role": "assistant", "text": text}]
        return {"text": text, "movies": movies, "intent": intent, "source": source}

    def _chat(self, message: str) -> str:
        try:
            raw = json.loads(self.llm([{"role": "system", "content": CHAT_PROMPT}, {"role": "user", "content": message}],
                                      schema=CHAT_SCHEMA))
            return str(raw.get("reply") or "").strip() or ASK_MORE
        except (OSError, ValueError, AttributeError):
            return ASK_MORE
