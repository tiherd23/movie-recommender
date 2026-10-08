"""Ghép phim MovieLens với TMDB để lấy poster, mô tả, từ khóa, diễn viên, đạo diễn.

MovieLens 1M không có sẵn mã TMDB, nên phải tìm theo tên + năm:
    "Matrix, The (1999)"  ->  tìm "The Matrix", năm 1999
Kết quả tốt nhất được chọn theo độ giống tên và độ lệch năm.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

API = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
_ARTICLES = r"The|A|An|Le|La|Les|L'|Il|El|Los|Las|Das|Der|Die|Un|Une|Uno|Una|I|Gli|O"


def _fix_article(title: str) -> str:
    """'Matrix, The' -> 'The Matrix'."""
    m = re.match(rf"^(.*),\s*({_ARTICLES})$", title.strip(), flags=re.IGNORECASE)
    if not m:
        return title.strip()
    article = m.group(2)
    return f"{article}{m.group(1)}" if article.endswith("'") else f"{article} {m.group(1)}"


def parse_title(raw: str) -> tuple[list[str], int | None]:
    """Tách tên MovieLens thành danh sách tên để thử tìm và năm phát hành.

    'City of Lost Children, The (Cité des enfants perdus, La) (1995)'
        -> (['The City of Lost Children', 'La Cité des enfants perdus'], 1995)
    """
    year = None
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", raw.strip())
    title = raw.strip()
    if m:
        title, year = m.group(1), int(m.group(2))
    alternates = re.findall(r"\(([^()]*)\)", title)
    main = re.sub(r"\s*\([^()]*\)", "", title).strip()
    names = [_fix_article(main)]
    for alt in alternates:
        alt = re.sub(r"^a\.k\.a\.\s*", "", alt.strip(), flags=re.IGNORECASE)
        if alt:
            names.append(_fix_article(alt))
    return [n for n in dict.fromkeys(names) if n], year


def short_names(names: list[str]) -> list[str]:
    """Tên rút gọn: phần trước dấu ':' hoặc ' - '.
    'Star Wars: Episode IV - A New Hope' -> 'Star Wars' (TMDB chỉ ghi 'Star Wars')."""
    out = []
    for n in names:
        head = re.split(r":| - ", n, maxsplit=1)[0].strip()
        if head and head != n and len(head) >= 4 and head not in names:
            out.append(head)
    return list(dict.fromkeys(out))


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def _similar(a: str, b: str) -> float:
    """Độ giống tên 0..1. Nếu tên này nằm trọn trong tên kia (theo từ) thì tối thiểu 0.72:
    'Vacation' khớp được "National Lampoon's Vacation"."""
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    na, nb = f" {_norm(a)} ", f" {_norm(b)} "
    if min(len(na), len(nb)) >= 7 and (na in nb or nb in na):
        ratio = max(ratio, 0.72)
    return ratio


def pick_best(results: list[dict], names: list[str], year: int | None,
              weak_names: list[str] | None = None) -> tuple[dict | None, float]:
    """Chọn kết quả khớp nhất.

    Điểm = độ giống tên (0..1) - phạt lệch năm + thưởng nhỏ cho phim nhiều lượt bình chọn
    (để phân định khi nhiều phim trùng tên). Tên rút gọn (weak_names) bị nhân 0.9.
    """
    weak_names = weak_names or []
    best, best_score = None, -1.0
    for r in results:
        cands = [c for c in (r.get("title") or "", r.get("original_title") or "") if c]
        if not cands:
            continue
        score = max([_similar(n, c) for n in names for c in cands]
                    + [0.9 * _similar(n, c) for n in weak_names for c in cands])
        release = (r.get("release_date") or "")[:4]
        if year and release.isdigit():
            score -= min(abs(int(release) - year), 5) * 0.15
        elif year:
            score -= 0.3
        score += min(math.log10(1 + (r.get("vote_count") or 0)), 4.0) * 0.0125  # tối đa +0.05
        if score > best_score:
            best, best_score = r, score
    return best, best_score


class TMDBClient:
    def __init__(self, api_key: str, retries: int = 4):
        self.api_key, self.retries = api_key, retries

    def get(self, path: str, **params) -> dict:
        query = urllib.parse.urlencode({"api_key": self.api_key, **params})
        url = f"{API}{path}?{query}"
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                if err.code == 429:  # quá giới hạn tốc độ: chờ theo yêu cầu của máy chủ
                    time.sleep(float(err.headers.get("Retry-After", 2)))
                elif err.code == 401:
                    raise SystemExit("Khóa TMDB không hợp lệ. Kiểm tra lại file .env") from None
                elif err.code == 404:
                    return {}
                else:
                    time.sleep(1 + attempt)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1 + attempt)
        return {}

    def match(self, raw_title: str) -> dict:
        """Tìm phim theo tên MovieLens, trả về một dòng dữ liệu (tmdb_id=None nếu không thấy)."""
        names, year = parse_title(raw_title)
        weak = short_names(names)
        best, score = None, -1.0
        for name in names + weak:  # thử tên đầy đủ trước, tên rút gọn sau
            for params in ([{"year": year}] if year else []) + [{}]:
                results = self.get("/search/movie", query=name, include_adult="false", **params).get("results", [])
                cand, s = pick_best(results[:10], names, year, weak)
                if cand is not None and s > score:
                    best, score = cand, s
                if score >= 0.85:
                    break
            if score >= 0.85:
                break
        row = {"tmdb_id": None, "match_score": round(min(max(score, 0.0), 1.0), 3)}
        if best is None or score < 0.45:  # dưới ngưỡng: coi như không tìm thấy, tránh ghép nhầm phim
            return row
        detail = self.get(f"/movie/{best['id']}", append_to_response="credits,keywords") or best
        crew = detail.get("credits", {}).get("crew", [])
        cast = detail.get("credits", {}).get("cast", [])
        row.update({
            "tmdb_id": best["id"],
            "tmdb_title": detail.get("title") or best.get("title"),
            "release_date": detail.get("release_date") or best.get("release_date"),
            "overview": (detail.get("overview") or best.get("overview") or "").replace("\n", " ").strip(),
            "poster_path": detail.get("poster_path") or best.get("poster_path"),
            "runtime": detail.get("runtime"),
            "vote_average": detail.get("vote_average"),
            "vote_count": detail.get("vote_count"),
            "original_language": detail.get("original_language"),
            "tmdb_genres": "|".join(g["name"] for g in detail.get("genres", [])),
            "keywords": "|".join(k["name"] for k in detail.get("keywords", {}).get("keywords", [])[:15]),
            "cast": "|".join(c["name"] for c in cast[:5]),
            "director": "|".join(c["name"] for c in crew if c.get("job") == "Director"),
        })
        return row
