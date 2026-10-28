"""Lớp phục vụ gợi ý cho web: gói các mô hình đã huấn luyện và gợi ý cho người dùng MỚI.

Vấn đề: người đăng ký trên web không có trong MovieLens, nên mô hình chưa có vector cho họ.
Giải pháp (fold-in): giữ nguyên vector phim Y đã học, giải nghiệm đóng của ALS để tính
vector cho người mới từ các phim họ vừa chấm. Mất vài mili-giây, không phải huấn luyện lại,
nên gợi ý thay đổi ngay sau mỗi lần chấm điểm.
"""
from __future__ import annotations

import json
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .data import ROOT, Dataset
from .models import ContentBased, ImplicitALS, ItemKNN, Popularity
from .tmdb import IMAGE_BASE

DEFAULT_ARTIFACT = ROOT / "artifacts" / "service.pkl"
LIKE = 4  # chấm từ 4 sao trở lên được coi là "thích"
MIN_LIKED_FOR_RANKER = 5  # đủ số phim thích mới dùng bộ xếp hạng hai tầng; ít hơn thì dùng công thức trộn
GENRE_SHARE = 0.5  # đã chọn thể loại thì ít nhất nửa danh sách (kiến trúc hai tầng) thuộc các thể loại đó

# Tên tiếng Việt của 18 thể loại MovieLens (chỉ để hiển thị; dữ liệu và mô hình vẫn dùng tên gốc)
GENRE_VI = {
    "Action": "Hành động", "Adventure": "Phiêu lưu", "Animation": "Hoạt hình", "Children's": "Thiếu nhi",
    "Comedy": "Hài", "Crime": "Hình sự", "Documentary": "Tài liệu", "Drama": "Chính kịch", "Fantasy": "Giả tưởng",
    "Film-Noir": "Phim đen (noir)", "Horror": "Kinh dị", "Musical": "Nhạc kịch", "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn", "Sci-Fi": "Khoa học viễn tưởng", "Thriller": "Giật gân", "War": "Chiến tranh",
    "Western": "Miền Tây",
}


def genre_vi(genres: str | list[str], sep: str = ", ") -> str:
    """"Comedy|Drama" hoặc ["Comedy", "Drama"] -> "Hài, Chính kịch"."""
    names = genres.split("|") if isinstance(genres, str) else genres
    return sep.join(GENRE_VI.get(g, g) for g in names if g)


_ARTICLE = re.compile(r"^(?P<base>.*?), (?P<art>The|A|An)(?P<alt>\s*\(.*\))?\s*(?P<year>\(\d{4}\))$")


def display_title(title: str, year: bool = True) -> str:
    """Tên MovieLens -> tên đọc tự nhiên: "Matrix, The (1999)" -> "The Matrix (1999)"."""
    m = _ARTICLE.match(title)
    if m:
        title = f"{m['art']} {m['base']}{m['alt'] or ''} {m['year']}"
    return title if year else re.sub(r"\s*\(\d{4}\)$", "", title)


def _norm_text(text: str) -> list[str]:
    """Chữ thường, bỏ dấu và dấu câu, tách từ (để "the matrix", "Matrix 1999" đều khớp "Matrix, The (1999)")."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def _z(v: np.ndarray) -> np.ndarray:
    """Chuẩn hóa z-score để điểm của các mô hình cộng được với nhau."""
    return (v - v.mean()) / (v.std() + 1e-8)


class RecommenderService:
    def __init__(self, movies: pd.DataFrame, item_factors: np.ndarray, sim: sparse.csr_matrix,
                 content: ContentBased, popularity: np.ndarray,
                 reg: float, alpha: float, weights: list[float], ranker: dict | None = None):
        self.movies, self.Y, self.sim = movies, item_factors, sim
        self.content, self.pop = content, popularity
        self.ranker = ranker  # None: chưa huấn luyện bộ xếp hạng, chỉ dùng công thức trộn
        self.item_vecs = content.item_vecs_
        self.reg, self.alpha, self.weights = reg, alpha, weights
        self._gram = self.Y.T @ self.Y + reg * np.eye(self.Y.shape[1], dtype=np.float32)
        self._title_lower = movies["title"].str.lower()
        self._title_tokens = [_norm_text(display_title(t)) for t in movies["title"]]
        self.genres = sorted({g for gs in movies["genres"] for g in gs.split("|")})
        self._init_stats()

    def _init_stats(self) -> None:
        """Số liệu MovieLens của từng phim để hiển thị: số lượt chấm, điểm TB, tỉ lệ thích (>= 4 sao), hạng.

        Lấy lại chính xác từ đặc trưng phim của bộ xếp hạng (không cần huấn luyện lại):
        cột 0 = log1p(n), cột 1 = (tổng điểm + 10*mu) / (n + 10), cột 2 = tỉ lệ thích.
        Chưa có bộ xếp hạng thì chỉ có số lượt thích (self.pop).
        """
        n_items = len(self.pop)
        self.like_rank = np.empty(n_items, dtype=np.int64)  # 1 = phim được thích nhiều nhất
        self.like_rank[np.argsort(-self.pop, kind="stable")] = np.arange(1, n_items + 1)
        self.n_ratings = self.mean_rating = self.like_frac = None
        if self.ranker is not None:
            f = self.ranker["item_feats"].astype(np.float64)
            n = np.rint(np.expm1(f[:, 0]))
            mu = (f[:, 1] * (n + 10)).sum() / (n.sum() + 10 * n_items)
            self.n_ratings = n.astype(np.int64)
            self.mean_rating = np.where(n > 0, (f[:, 1] * (n + 10) - 10 * mu) / np.maximum(n, 1), np.nan)
            self.like_frac = f[:, 2]

    def stats(self, j: int) -> dict:
        """Số liệu của phim j: likes, rank; thêm n, mean, like_frac khi có bộ xếp hạng."""
        out = {"likes": int(self.pop[j]), "rank": int(self.like_rank[j])}
        if self.n_ratings is not None:
            out.update(n=int(self.n_ratings[j]), mean=float(self.mean_rating[j]), like_frac=float(self.like_frac[j]))
        return out

    # ---------- huấn luyện / lưu / nạp ----------
    @classmethod
    def build(cls, ds: Dataset, params: dict | None = None, with_ranker: bool = True) -> "RecommenderService":
        """Huấn luyện trên TOÀN BỘ dữ liệu (train + val + test) để phục vụ thật.

        with_ranker=True: huấn luyện thêm bộ xếp hạng hai tầng. Bộ xếp hạng học trên train -> nhãn val
        (để không rò rỉ), còn các mô hình tầng 1 phục vụ web thì học trên toàn bộ dữ liệu.
        Bản cho web không dùng giới tính, tuổi, nghề nghiệp vì người đăng ký không khai các thông tin đó.
        """
        params = params or {}
        full = pd.concat([ds.train, ds.val, ds.test], ignore_index=True)
        als_p = params.get("ImplicitALS", {"factors": 32, "alpha": 1.0, "min_rating": 4.0})
        knn_p = params.get("ItemKNN", {"k": 100, "shrink": 0.0})
        weights = params.get("Hybrid", {}).get("weights", [1.0, 1.0, 0.25, 0.25])
        als = ImplicitALS(**als_p).fit(ds, full)
        knn = ItemKNN(**knn_p).fit(ds, full)
        content = ContentBased().fit(ds, full)
        pop = Popularity().fit(ds, full)
        content.profiles_ = None  # hồ sơ người dùng MovieLens không cần cho web, bỏ để file gọn
        ranker = None
        if with_ranker:
            from .ranker import TwoStageRanker

            two = TwoStageRanker(als_p, knn_p, use_demographics=False)
            two.fit_ranker(ds, ds.train, ds.val)
            two._fit_base(ds, full)  # thống kê phim (độ phổ biến, điểm trung bình...) trên toàn bộ dữ liệu
            ranker = {"booster": two.booster_, "n_candidates": two.n_candidates,
                      "item_feats": two.item_feats_, "item_genre": two.item_genre_}
        return cls(ds.movies, als.Y_, sparse.csr_matrix(knn.sim_), content, pop.scores_,
                   als.reg, als.alpha, weights, ranker)

    def save(self, path: Path | str = DEFAULT_ARTIFACT) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items()
                 if not k.startswith("_") and k not in ("genres", "item_vecs", "like_rank", "n_ratings", "mean_rating", "like_frac")}
        Path(path).write_bytes(pickle.dumps(state))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_ARTIFACT) -> "RecommenderService":
        s = pickle.loads(Path(path).read_bytes())
        return cls(s["movies"], s["Y"], s["sim"], s["content"], s["pop"],
                   s["reg"], s["alpha"], s["weights"], s.get("ranker"))

    # ---------- gợi ý ----------
    def _fold_in(self, liked: list[int]) -> np.ndarray:
        """Vector ẩn của người dùng mới: nghiệm đóng của ALS với Y cố định."""
        Yl = self.Y[liked]
        A = self._gram + self.alpha * (Yl.T @ Yl)
        b = (1.0 + self.alpha) * Yl.sum(axis=0)
        return np.linalg.solve(A, b)

    def popular_reason(self, j: int) -> str:
        """Lý do 'phổ biến' bằng số liệu cụ thể thay vì câu chung chung."""
        st = self.stats(j)
        reason = f"Hạng {st['rank']:,}/{len(self.pop):,} về lượt thích · {st['likes']:,} người chấm ≥ 4★"
        return reason.replace(",", ".")

    def _movie(self, j: int, **extra) -> dict:
        """Thông tin hiển thị của một phim (poster và mô tả có khi đã lấy dữ liệu TMDB)."""
        row = self.movies.iloc[j]
        poster = row.get("poster_path", "")
        return {"item": int(j), "title": row["title"], "genres": row["genres"],
                "poster_url": f"{IMAGE_BASE}{poster}" if poster else "",
                "overview": row.get("overview", ""), "stats": self.stats(j), **extra}

    def _rank_two_stage(self, ratings: dict[int, int], liked: list[int],
                        als_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Điểm từ bộ xếp hạng hai tầng cho MỘT người dùng web (phim ngoài ứng viên = -inf), kèm chỉ số phim
        ứng viên và ma trận đặc trưng của chúng (dùng để giải thích bằng SHAP).
        Đặc trưng được dựng bằng đúng hàm đã dùng lúc huấn luyện (candidate_features)."""
        from .ranker import candidate_features

        rated = list(ratings)
        values = np.fromiter(ratings.values(), dtype=np.float32)
        profile = self.item_vecs[liked].sum(axis=0)
        scores = {  # cùng thứ tự và cùng cách tính với các mô hình tầng 1 lúc huấn luyện
            "als": als_score[None, :],
            "knn": np.asarray(self.sim[rated].sum(axis=0)).reshape(1, -1),  # ItemKNN dùng mọi phim đã xem
            "pop": self.pop[None, :],
            "content": (self.item_vecs @ (profile / (np.linalg.norm(profile) + 1e-8)))[None, :],
        }
        seen = np.zeros((1, len(self.movies)), dtype=bool)
        seen[0, rated] = True
        user_feats = np.array([[np.log1p(len(rated)), values.mean(), (values >= LIKE).mean()]], dtype=np.float32)
        # item_genre đã chia theo số thể loại của phim; nhân lại để ra cờ 0/1 rồi lập hồ sơ thể loại
        genre_flags = (self.ranker["item_genre"][liked] > 0).astype(np.float32)
        user_genre = genre_flags.sum(axis=0, keepdims=True)
        user_genre /= max(user_genre.sum(), 1.0)
        _, items, X = candidate_features(scores, seen, self.ranker["n_candidates"], user_feats, user_genre,
                                         self.ranker["item_feats"], self.ranker["item_genre"])
        out = np.full(len(self.movies), -np.inf, dtype=np.float32)
        booster = self.ranker["booster"]
        out[items] = booster.predict(X, num_iteration=booster.best_iteration)
        return out, items, X

    def _explain_two_stage(self, top, items: np.ndarray, X: np.ndarray, n: int = 4) -> list[list[tuple[str, float]]]:
        """Với mỗi phim trong top: tối đa n nhóm đặc trưng ảnh hưởng mạnh nhất đến điểm (giá trị SHAP, + là đẩy lên).
        Bỏ nhóm đặc điểm của người dùng vì nhóm này gần như như nhau cho mọi phim của cùng một người."""
        from .ranker import FEATURE_GROUPS, grouped_contributions

        pos = {int(j): r for r, j in enumerate(items)}
        contrib = grouped_contributions(self.ranker["booster"], X[[pos[int(j)] for j in top]])
        contrib = contrib.drop(columns=[c for c in ("base", "user") if c in contrib.columns])
        out = []
        for _, row in contrib.iterrows():
            row = row[row.abs() >= 0.01].sort_values(key=np.abs, ascending=False)[:n]
            out.append([(FEATURE_GROUPS[g], float(v)) for g, v in row.items()])
        return out

    def recommend(self, ratings: dict[int, int], genres: list[str] | None = None, k: int = 12,
                  diversity: float = 0.0) -> list[dict]:
        """ratings: {chỉ số phim: số sao}. Trả về k phim chưa chấm, kèm lý do gợi ý.

        diversity (0..1): mức đa dạng thể loại. > 0 thì xếp hạng lại 50 phim điểm cao nhất bằng MMR với
        lam = 1 - diversity (xem movierec.beyond); 0 giữ nguyên thứ tự của mô hình.

        - Đủ MIN_LIKED_FOR_RANKER phim thích: dùng kiến trúc hai tầng (mô hình tốt nhất khi đánh giá).
        - Ít hơn: dùng công thức trộn, nơi thể loại đã chọn và độ phổ biến gánh phần thiếu dữ liệu.
        """
        genres = genres or []
        liked = [i for i, r in ratings.items() if r >= LIKE]
        knn_part = self.sim[liked].toarray() if liked else None  # (số phim thích, tổng số phim)
        als_score = self.Y @ self._fold_in(liked) if liked else None

        if self.ranker is not None and len(liked) >= MIN_LIKED_FOR_RANKER:
            (score, cand_items, cand_X), method = self._rank_two_stage(ratings, liked, als_score), "two_stage"
        else:
            method = "blend"
            score = self._blend_score(liked, genres, als_score, knn_part)
            if genres and not liked:
                # cold-start: chưa thích phim nào thì chỉ gợi ý trong các thể loại đã chọn
                score[~self._genre_mask(genres)] = -np.inf
        if ratings:
            score[list(ratings)] = -np.inf  # không gợi ý lại phim đã chấm

        top = self._top(score, k, diversity)
        added: set[int] = set()
        if method == "two_stage" and genres:
            top, added = self._genre_quota(top, score, genres, ratings, liked, als_score, knn_part, k)
        why = [None] * len(top)
        if method == "two_stage" and top:
            pos = set(int(j) for j in cand_items)
            in_cand = [j for j in top if j in pos]
            factors = dict(zip(in_cand, self._explain_two_stage(in_cand, cand_items, cand_X))) if in_cand else {}
            why = [factors.get(j) for j in top]
        out = []
        for j, factors in zip(top, why):
            row = self.movies.iloc[j]
            match = [g for g in genres if g in row["genres"].split("|")]
            if j in added:  # đưa vào vì thể loại đã chọn
                reason = f"Hợp thể loại bạn chọn: {genre_vi(match)}"
            elif knn_part is not None and knn_part[:, j].max() > 0:
                src = liked[int(knn_part[:, j].argmax())]
                reason = f"Vì bạn thích {display_title(self.movies.iloc[src]['title'], year=False)}"
            elif match:
                reason = f"Hợp thể loại bạn chọn: {genre_vi(match)}"
            else:
                reason = self.popular_reason(int(j))
            extra = {"why": factors} if factors else {}
            value = float(score[j]) if np.isfinite(score[j]) else None
            out.append(self._movie(int(j), score=value, reason=reason, method=method, **extra))
        return out

    def _blend_score(self, liked: list[int], genres: list[str], als_score, knn_part) -> np.ndarray:
        """Công thức trộn: ALS + ItemKNN + nội dung (gồm thể loại đã chọn) + độ phổ biến, mỗi phần chuẩn hóa z-score."""
        w_als, w_knn, w_cb, w_pop = self.weights
        score = np.zeros(len(self.movies), dtype=np.float32)
        if liked:
            score += w_als * _z(als_score) + w_knn * _z(knn_part.sum(axis=0))
        profile = self.item_vecs[liked].sum(axis=0) + len(liked or [0]) * self.content.genre_vector(genres)
        if np.linalg.norm(profile) > 0:
            # chưa có phim thích nào: thể loại đã chọn là tín hiệu chính
            score += (w_cb if liked else 1.0) * _z(self.item_vecs @ profile)
        # log1p: số lượt thích lệch rất mạnh về vài phim, lấy log để phim nổi tiếng không lấn át
        score += (w_pop if liked else 1.0) * _z(np.log1p(self.pop))
        return score

    def _genre_mask(self, genres: list[str]) -> np.ndarray:
        chosen = set(genres)
        return self.movies["genres"].map(lambda g: bool(chosen & set(g.split("|")))).to_numpy()

    def _top(self, score: np.ndarray, k: int, diversity: float) -> list[int]:
        if diversity > 0:
            from .beyond import genre_vectors, mmr_rerank

            if getattr(self, "_gvec", None) is None:  # tính một lần; bản dịch vụ đã lưu từ trước không có thuộc tính này
                self._gvec = genre_vectors(self.movies["genres"].reset_index(drop=True))
            top = mmr_rerank(score, self._gvec, 1.0 - diversity, k) if np.isfinite(score).any() else []
        else:
            top = np.argsort(-score)[:k]
        return [int(j) for j in top if np.isfinite(score[j])]

    def _genre_quota(self, top: list[int], score: np.ndarray, genres: list[str], ratings: dict[int, int],
                     liked: list[int], als_score, knn_part, k: int) -> tuple[list[int], set[int]]:
        """Kiến trúc hai tầng không dùng thể loại đã chọn, nên bảo đảm ít nhất GENRE_SHARE danh sách thuộc các thể
        loại đó: lấy thêm phim hợp thể loại theo điểm hai tầng (trong nhóm ứng viên), thiếu nữa thì theo công thức
        trộn (đã gồm sở thích của người dùng), rồi xen kẽ với các phim còn lại. Trả về (danh sách mới, phim thêm vào)."""
        match = self._genre_mask(genres)
        need = int(np.ceil(k * GENRE_SHARE))
        have = [j for j in top if match[j]]
        if len(have) >= need:
            return top, set()
        taken = set(top) | set(ratings)
        extras: list[int] = []
        for source in (score, None):
            if len(have) + len(extras) >= need:
                break
            s = (self._blend_score(liked, genres, als_score, knn_part) if source is None else source).astype(np.float64)
            s[~match] = -np.inf
            s[list(taken | set(extras))] = -np.inf
            extras += [int(j) for j in np.argsort(-s)[: need - len(have) - len(extras)] if np.isfinite(s[j])]
        picked, others = have + extras, [j for j in top if not match[j]]
        others = others[: k - len(picked)]
        merged: list[int] = []
        for n in range(max(len(picked), len(others))):  # xen kẽ, phim điểm cao nhất của mô hình vẫn đứng đầu
            merged += [x[n] for x in (others, picked) if n < len(x)]
        return merged[:k], set(extras)

    def similar(self, item: int, k: int = 8) -> list[dict]:
        """Phim tương tự = giống về người xem (ItemKNN) VÀ giống về nội dung.
        Chỉ dùng ItemKNN thì phim quá nổi tiếng hay lọt vào vì ai cũng xem."""
        score = _z(self.sim[item].toarray().ravel()) + _z(self.item_vecs @ self.item_vecs[item])
        score[item] = -np.inf
        return [self._movie(int(j)) for j in np.argsort(-score)[:k]]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Tìm theo tên, không phân biệt hoa thường, dấu câu, mạo từ đặt cuối ("Matrix, The"): mỗi từ trong câu tìm
        phải là đầu của một từ trong tên (gõ dở "termin" vẫn ra "Terminator"). Tên chứa nguyên văn câu tìm lên trước, rồi tên
        bắt đầu bằng câu tìm, sau đó theo độ phổ biến."""
        hits, _ = self.search_all(query)
        return [self._movie(int(j)) for j in hits[:limit]]

    def search_all(self, query: str) -> tuple[np.ndarray, int]:
        """(chỉ số mọi phim khớp đã xếp thứ tự, tổng số phim khớp)."""
        words = _norm_text(query)
        if not words:
            return np.array([], dtype=np.int64), 0
        hits = [j for j, toks in enumerate(self._title_tokens)
                if all(any(t.startswith(w) for t in toks) for w in words)]
        hits = np.array(hits, dtype=np.int64)
        if len(hits):
            starts = np.array([self._title_tokens[j][:len(words)] == words for j in hits])
            exact = self._title_lower.iloc[hits].str.contains(query.strip().lower(), regex=False).to_numpy()
            hits = hits[np.lexsort((-self.pop[hits], ~starts, ~exact))]  # khóa cuối là khóa chính
        return hits, len(hits)

    def popular(self, k: int = 24) -> list[dict]:
        top = np.argsort(-self.pop)[:k]
        return [self._movie(int(j)) for j in top]


def load_or_build(path: Path | str = DEFAULT_ARTIFACT) -> RecommenderService:
    """Nạp mô hình đã lưu; nếu chưa có thì huấn luyện (khoảng 10-20 giây) rồi lưu lại."""
    path = Path(path)
    if path.exists():
        return RecommenderService.load(path)
    from .data import build_dataset

    best = ROOT / "results" / "best_params.json"
    params = json.loads(best.read_text()) if best.exists() else None
    try:
        import lightgbm  # noqa: F401
        with_ranker = True
    except ImportError:
        with_ranker = False
    svc = RecommenderService.build(build_dataset(), params, with_ranker=with_ranker)
    svc.save(path)
    return svc
