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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from .service import display_title, genre_vi

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

# Một ví dụ mẫu (phim bịa) đặt trước phim thật: có ví dụ thì mô hình viết đúng dạng câu tóm tắt thay vì chép lại
# lời dặn ở trên, lỗi gặp ở 395/473 câu bị loại trong lần chạy đầu không có ví dụ.
REPLY_EXAMPLE = [
    {"role": "user", "content": "Phim: Harbor Lights (1990)\nThể loại: Drama\nMô tả: A retired sailor returns to his "
                                "hometown to reconnect with the daughter he abandoned decades ago, while the town fights "
                                "to save its old lighthouse."},
    {"role": "assistant", "content": '{"reason": "Một thủy thủ về hưu trở về quê tìm lại người con gái bị bỏ rơi, '
                                     'giữa lúc thị trấn cố cứu ngọn hải đăng cũ."}'},
]
# temperature cho từng lần thử: lần đầu ổn định; nếu bị loại thì thêm ngẫu nhiên, vì với temperature 0
# mô hình trả lại đúng câu hỏng cũ (đã thấy trong log lần chạy đầu)
BLURB_TEMPERATURES = (0.0, 0.6, 0.9)

CHAT_PROMPT = """Bạn là trợ lý gợi ý phim của MovieRec. Trả về JSON với "reply" là một hoặc hai câu tiếng Việt.
Không tự nêu tên phim. Hãy mời người dùng mô tả nội dung, thể loại hoặc một phim họ thích để bạn tìm giúp."""

CHAT_SCHEMA = {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
ASK_MORE = "Bạn muốn xem phim như thế nào? Hãy mô tả nội dung hoặc kể một phim bạn thích."


# Lời đáp cũng ép theo JSON vì Qwen3 không tắt được chế độ suy luận khi viết tự do.
# Mảnh của câu hướng dẫn: câu trả về chứa chúng là mô hình chép lại lời dặn chứ không tóm tắt phim
# Chữ Hán/Nhật/Hàn: Qwen đôi khi chen một từ tiếng Trung vào câu tiếng Việt (ví dụ "từ hài đến 悲剧")
_FOREIGN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_PROMPT_ECHO = ("dưới 25 từ", "tóm tắt nội dung", "không nhắc lại tên phim", "không thêm chi tiết", "json")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


# câu trả lời mẫu trong REPLY_EXAMPLE: mô hình đôi khi chép nguyên câu này cho một phim khác (12/3.629 câu ở lần chạy
# thứ hai, ví dụ Robocop 2 và East of Eden được tả là "một thủy thủ về hưu... ngọn hải đăng cũ")
_EXAMPLE_WORDS = _words(json.loads(REPLY_EXAMPLE[1]["content"])["reason"])


def valid_blurb(text: str) -> bool:
    """Câu giới thiệu dùng được: không rỗng, không quá dài, không chép lại lời dặn hay câu mẫu, không chen chữ Hán."""
    low, words = text.lower(), _words(text)
    copies_example = len(words & _EXAMPLE_WORDS) / max(len(words | _EXAMPLE_WORDS), 1) > 0.4
    return (bool(text) and len(text.split()) <= 60 and not any(f in low for f in _PROMPT_ECHO)
            and not _FOREIGN.search(text) and not copies_example)


def load_blurbs(path) -> dict[str, str]:
    """Câu giới thiệu tiếng Việt viết sẵn (scripts/build_blurbs.py), theo tên phim. Chưa có tệp thì trả về rỗng."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items() if valid_blurb(str(v))}


def save_blurbs(path, blurbs: dict[str, str]) -> None:
    """Ghi ra tệp tạm rồi đổi tên, để lỡ dừng giữa chừng cũng không làm hỏng tệp cũ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blurbs, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, path)


REASON_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}


class OllamaClient:
    """Gọi API /api/chat của Ollama bằng thư viện chuẩn (không thêm phụ thuộc)."""

    def __init__(self, model: str = DEFAULT_LLM, host: str = DEFAULT_HOST, timeout: float = 120.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.retry_wait = 2.0
        # Số lớp đưa lên GPU; 0 = chỉ CPU, đặt biến thành "auto" để Ollama tự chọn.
        # Mặc định CPU: trên RTX 3050 4 GB, qwen3:4b không vừa VRAM nên Ollama chia ~27% CPU / 73% GPU,
        # và cách chia này đo được chậm hơn chạy thuần CPU (10.2 s so với 7.7 s mỗi câu, cùng buổi 13/10).
        gpu = os.environ.get("MOVIEREC_LLM_NUM_GPU", "0")
        self.num_gpu: int | None = None if gpu.lower() == "auto" else int(gpu)
        # ngữ cảnh nhỏ thì mô hình chiếm ít VRAM hơn; lời nhắc của trợ lý ngắn nên 2048 token là đủ
        self.num_ctx = int(os.environ.get("MOVIEREC_LLM_NUM_CTX", "2048"))
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

    def _post_retry(self, payload: dict) -> dict:
        """Gửi lại một lần nếu kết nối lỗi. Khi Ollama vừa khởi động hoặc đang nạp mô hình, yêu cầu đầu tiên có thể
        thất bại; nếu không thử lại, trợ lý sẽ lặng lẽ rơi về chế độ dự phòng (mất bộ lọc thể loại, năm)."""
        try:
            return self._post(payload)
        except urllib.error.HTTPError:
            raise
        except OSError as err:
            if "CUDA" in str(err) or isinstance(err, TimeoutError):
                raise
            time.sleep(self.retry_wait)
            return self._post(payload)

    def available(self) -> bool:
        """Ollama đang chạy và đã tải mô hình cần dùng."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                names = [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
        except (OSError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    def __call__(self, messages: list[dict], schema: dict | None = None, temperature: float | None = None) -> str:
        if temperature is None:
            temperature = 0.0 if schema else 0.4
        payload = {"model": self.model, "messages": messages, "stream": False, "keep_alive": "30m",
                   "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                               "num_predict": 400}}  # chặn độ dài để giới hạn độ trễ trên CPU
        if schema:
            payload["format"] = schema  # structured output: Ollama ép câu trả lời theo JSON schema
        if self._think is not None:
            payload["think"] = self._think
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        try:
            data = self._post_retry(payload)
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


# Hiểu câu bằng luật khi không có LLM: từ khóa tiếng Việt -> thể loại MovieLens
GENRE_WORDS = {
    "Comedy": ["hài", "vui nhộn", "comedy"], "Horror": ["kinh dị", "ma ám", "horror"],
    "Action": ["hành động", "action"], "Adventure": ["phiêu lưu", "adventure"],
    "Animation": ["hoạt hình", "animation"], "Children's": ["thiếu nhi", "trẻ em", "gia đình"],
    "Crime": ["hình sự", "tội phạm", "crime"], "Documentary": ["tài liệu", "documentary"],
    "Drama": ["chính kịch", "tâm lý", "drama"], "Fantasy": ["giả tưởng", "thần thoại", "phép thuật", "fantasy"],
    "Film-Noir": ["noir"], "Musical": ["nhạc kịch", "ca nhạc", "musical"],
    "Mystery": ["bí ẩn", "trinh thám", "mystery"], "Romance": ["lãng mạn", "tình cảm", "tình yêu", "romance"],
    "Sci-Fi": ["khoa học viễn tưởng", "viễn tưởng", "vũ trụ", "sci-fi"], "Thriller": ["giật gân", "ly kỳ", "thriller"],
    "War": ["chiến tranh", "war"], "Western": ["miền tây", "cao bồi", "western"],
}
_GREETING = re.compile(r"^(xin chào|chào|hello|hi|alo|cảm ơn|cám ơn|thanks?|ok|oke)\b", re.I)
_MORE = re.compile(r"(còn phim|phim khác|phim nào khác|thêm phim|gợi ý thêm|thêm vài|khác không|nữa không)", re.I)
_SIMILAR = re.compile(r"(?:giống|tương tự|như)\s+(?:phim\s+)?(.+)", re.I)
_TITLE_BEFORE = re.compile(r"(?:thích|xem|mê)\s+(?:phim\s+)?(.+?)\s+(?:có phim|phim nào|có gì|tương tự)", re.I)
# từ chung chung của một câu xin gợi ý; bỏ hết các từ này (và từ thể loại, năm) mà còn lại rất ít thì câu không có mô tả nội dung
_FILLER = set("""gợi ý cho mình tôi em anh chị vài một ít mấy phim bộ muốn xem coi có nào hay không nhé nha đi ạ với
    hả gì đó thể loại kiểu loại về được tìm giúp nhưng mà và hoặc đừng trừ ngoại bỏ qua phải sau trước từ năm những
    thập niên kỷ tối nay bây giờ hôm cuối tuần cần một vài""".split())
_NEGATE = re.compile(r"(đừng|không có|không muốn|không phải|trừ|ngoại trừ|bỏ qua)", re.I)


def rule_intent(message: str, genres: list[str]) -> dict:
    """Ý định của câu chat bằng luật đơn giản, dùng khi không có (hoặc lỗi) LLM: lời chào, "còn phim nào khác",
    "phim giống X", thể loại muốn và không muốn (sau "đừng", "không có"...), thập niên/năm."""
    text = " ".join(message.lower().split())
    if _GREETING.match(text) and len(text.split()) <= 5 and not any(
            w in text for ws in GENRE_WORDS.values() for w in ws):
        return clean_intent({"action": "chat"}, genres)
    raw: dict = {"action": "describe", "description": message, "more": bool(_MORE.search(text))}
    if raw["more"] and len(text.split()) <= 8:
        raw["description"] = ""
    flat = " ".join(message.split())  # giữ nguyên chữ hoa của tên phim
    sim = _SIMILAR.search(flat)
    if sim:
        name = re.sub(r"[?.!]+$|\s+(không|nhé|nha|đi|ạ|với)$", "", sim.group(1).strip(), flags=re.I).strip()
        if name.lower() in _FILLER or not name:  # "thích X, có phim nào tương tự không": tên phim nằm trước
            before = _TITLE_BEFORE.search(flat)
            name = before.group(1).strip(" ,") if before else ""
        if name:
            raw.update(action="similar", similar_to=name, description="")
    if raw.get("similar_to"):  # tên phim mẫu không phải là thể loại ("Star Wars" không có nghĩa là phim chiến tranh)
        text = text.replace(raw["similar_to"].lower(), " ")
    neg = _NEGATE.search(text)
    want, avoid = [], []
    for genre, words in GENRE_WORDS.items():
        for w in words:
            hit = re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text)  # trọn từ: "war" không khớp "wars"
            pos = hit.start() if hit else -1
            if pos >= 0:
                (avoid if neg and pos > neg.start() else want).append(genre)
                break
    raw.update(genres=want, exclude_genres=avoid)
    if raw["action"] == "describe" and not raw["more"]:
        # câu chỉ nêu thể loại/năm (vd. "gợi ý phim hài, đừng có lãng mạn"): gợi ý theo sở thích + lọc, không tìm theo nghĩa
        rest = text
        for w in sorted({w for ws in GENRE_WORDS.values() for w in ws}, key=len, reverse=True):
            rest = rest.replace(w, " ")
        rest = re.sub(r"\b\d+\b|[^\w\s]", " ", rest)
        if not [t for t in rest.split() if t not in _FILLER]:
            raw.update(action="recommend", description="")
    decade = re.search(r"(?:thập niên|những năm|thập kỷ)\s*(?:19)?(\d0)\b", text)
    if decade:
        start = 1900 + int(decade.group(1))
        raw.update(year_from=start, year_to=start + 9)
    else:
        year = re.search(r"\b(19\d\d|20\d\d)\b", text)
        if year:
            y = int(year.group(1))
            if re.search(r"(sau|từ)\s+(năm\s+)?" + str(y), text):
                raw["year_from"] = y
            elif re.search(r"trước\s+(năm\s+)?" + str(y), text):
                raw["year_to"] = y
            elif not raw.get("similar_to"):
                raw.update(year_from=y, year_to=y)
    return clean_intent(raw, genres)


class Assistant:
    def __init__(self, service, semantic=None, llm: LLM | None = None, k: int = 5,
                 blurbs: dict[str, str] | None = None, live_blurbs: bool | None = None):
        self.svc, self.semantic, self.llm, self.k = service, semantic, llm, k
        years = service.movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
        self.years = years.fillna(0).astype(int).to_numpy()
        self.genre_sets = [set(g.split("|")) for g in service.movies["genres"]]
        # câu giới thiệu đã viết, theo tên phim; nạp sẵn từ tệp viết trước thì lúc chat không phải gọi LLM cho bước này
        self._blurbs: dict[str, str] = dict(blurbs or {})
        # Có bộ câu viết sẵn thì không viết thêm lúc chat: phim còn thiếu là phim mô hình đã viết hỏng ba lần lúc
        # chạy build_blurbs.py, gọi lại chỉ làm người dùng chờ thêm tới ~45 giây mà nhiều khả năng vẫn hỏng.
        self.live_blurbs = not self._blurbs if live_blurbs is None else live_blurbs
        self.rejected: list[tuple[str, str]] = []  # (tên phim, câu bị loại) để kiểm tra chất lượng

    # ---------- bước 1: hiểu câu chat ----------
    def parse(self, message: str, history: list[dict] | None = None) -> dict:
        """Ý định của câu chat. Không có LLM (hoặc LLM lỗi): hiểu câu bằng luật (rule_intent)."""
        fallback = rule_intent(message, self.svc.genres)
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
                cands = [dict(m, reason=f"Tương tự {display_title(base['title'], year=False)}") for m in self.svc.similar(base["item"], k=pool)]
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
        if movie["title"] in self._blurbs:
            return self._blurbs[movie["title"]]
        overview = str(movie.get("overview") or "").strip()
        if self.llm is None or not overview or not self.live_blurbs:
            return ""
        info = f"Phim: {movie['title']}\nThể loại: {movie['genres'].replace('|', ', ')}\nMô tả: {overview[:400]}"
        # Mô hình nhỏ đôi khi chép lại câu hướng dẫn, chen chữ Hán, hoặc lặp từ tới khi bị cắt giữa chừng (JSON hỏng):
        # loại các câu đó và thử lại với temperature cao hơn. self.rejected giữ câu bị loại để xem lại.
        messages = [{"role": "system", "content": REPLY_PROMPT}, *REPLY_EXAMPLE, {"role": "user", "content": info}]
        for n, temp in enumerate(BLURB_TEMPERATURES):
            try:
                out = (self.llm(messages, schema=REASON_SCHEMA) if n == 0
                       else self.llm(messages, schema=REASON_SCHEMA, temperature=temp))
            except OSError:
                return ""
            try:
                text = str(json.loads(out).get("reason") or "").strip()
            except (ValueError, AttributeError):
                text = ""
            if valid_blurb(text):
                self._blurbs[movie["title"]] = text
                return text
            self.rejected.append((movie["title"], (text or str(out))[:300]))
        return ""

    def intro(self, n: int, source: str, intent: dict | None = None, personal: bool = True) -> str:
        """Câu mở đầu nói ĐÚNG cách phim được chọn, để không khẳng định "hợp yêu cầu" khi chỉ là gợi ý chung.
        personal=False (chưa chấm phim nào, vd. khách): không nói "theo sở thích của bạn"."""
        intent = intent or {}
        conds = []
        if intent.get("genres"):
            conds.append(f"thuộc thể loại {genre_vi(intent['genres'])}")
        if intent.get("exclude_genres"):
            conds.append(f"không có {genre_vi(intent['exclude_genres'])}")
        y0, y1 = intent.get("year_from"), intent.get("year_to")
        if y0 and y1:
            conds.append(f"ra mắt năm {y0}" if y0 == y1 else f"ra mắt {y0}–{y1}")
        elif y0 or y1:
            conds.append(f"ra mắt từ {y0}" if y0 else f"ra mắt trước {y1}")
        cond = (" " + ", ".join(conds)) if conds else ""
        if source == "semantic":
            return f"Mình tìm được {n} phim có nội dung gần với mô tả của bạn{cond}:"
        if source == "similar":
            return f"Đây là {n} phim giống “{intent.get('similar_to')}” nhất{cond}:" if intent.get("similar_to") \
                else f"Đây là {n} phim giống phim bạn nhắc tới nhất{cond}:"
        basis = "gợi ý theo sở thích của bạn" if personal else "được yêu thích nhất trên MovieLens"
        if intent.get("action") == "similar" and intent.get("similar_to"):
            return f"Mình chưa tìm thấy phim “{intent['similar_to']}” trong kho, nên đây là {n} phim {basis}{cond}:"
        if intent.get("action") == "describe" or intent.get("description"):
            why = "Tính năng so nội dung mô tả đang tắt" if self.semantic is None else "Mình chưa so được nội dung mô tả"
            return f"{why}, nên đây là {n} phim {basis}{cond} (chưa chắc khớp mô tả):"
        return f"Đây là {n} phim {basis}{cond}:"

    def write(self, message: str, movies: list[dict], source: str = "semantic", intent: dict | None = None,
              personal: bool = True) -> str:
        """Lời đáp: câu mở đầu (theo cách chọn phim) + mỗi phim một dòng. Tên phim lấy từ dữ liệu,
        LLM chỉ viết câu giới thiệu."""
        if not movies:
            return self._fallback_text(movies)
        lines = []
        for m in movies:
            reason = self.explain(m)
            lines.append(f"- **{display_title(m['title'])}**" + (f": {reason}" if reason else ""))
        return "\n".join([self.intro(len(movies), source, intent, personal), ""] + lines)

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
            text = self.write(message, movies, source, intent, personal=bool(ratings))
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
