"""Lớp phục vụ gợi ý cho web: gói các mô hình đã huấn luyện và gợi ý cho người dùng MỚI.

Vấn đề: người đăng ký trên web không có trong MovieLens, nên mô hình chưa có vector cho họ.
Giải pháp (fold-in): giữ nguyên vector phim Y đã học, giải nghiệm đóng của ALS để tính
vector cho người mới từ các phim họ vừa chấm. Mất vài mili-giây, không phải huấn luyện lại,
nên gợi ý thay đổi ngay sau mỗi lần chấm điểm.
"""
from __future__ import annotations

import json
import pickle
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
        self.genres = sorted({g for gs in movies["genres"] for g in gs.split("|")})

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
                 if not k.startswith("_") and k not in ("genres", "item_vecs")}
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

    def _movie(self, j: int, **extra) -> dict:
        """Thông tin hiển thị của một phim (poster và mô tả có khi đã lấy dữ liệu TMDB)."""
        row = self.movies.iloc[j]
        poster = row.get("poster_path", "")
        return {"item": int(j), "title": row["title"], "genres": row["genres"],
                "poster_url": f"{IMAGE_BASE}{poster}" if poster else "",
                "overview": row.get("overview", ""), **extra}

    def _rank_two_stage(self, ratings: dict[int, int], liked: list[int], als_score: np.ndarray) -> np.ndarray:
        """Điểm từ bộ xếp hạng hai tầng cho MỘT người dùng web (phim ngoài ứng viên = -inf).
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
        return out

    def recommend(self, ratings: dict[int, int], genres: list[str] | None = None, k: int = 12) -> list[dict]:
        """ratings: {chỉ số phim: số sao}. Trả về k phim chưa chấm, kèm lý do gợi ý.

        - Đủ MIN_LIKED_FOR_RANKER phim thích: dùng kiến trúc hai tầng (mô hình tốt nhất khi đánh giá).
        - Ít hơn: dùng công thức trộn, nơi thể loại đã chọn và độ phổ biến gánh phần thiếu dữ liệu.
        """
        genres = genres or []
        liked = [i for i, r in ratings.items() if r >= LIKE]
        w_als, w_knn, w_cb, w_pop = self.weights
        knn_part = self.sim[liked].toarray() if liked else None  # (số phim thích, tổng số phim)
        als_score = self.Y @ self._fold_in(liked) if liked else None

        if self.ranker is not None and len(liked) >= MIN_LIKED_FOR_RANKER:
            score, method = self._rank_two_stage(ratings, liked, als_score), "two_stage"
        else:
            method = "blend"
            score = np.zeros(len(self.movies), dtype=np.float32)
            if liked:
                score += w_als * _z(als_score) + w_knn * _z(knn_part.sum(axis=0))
            profile = self.item_vecs[liked].sum(axis=0) + len(liked or [0]) * self.content.genre_vector(genres)
            if np.linalg.norm(profile) > 0:
                # chưa có phim thích nào: thể loại đã chọn là tín hiệu chính
                score += (w_cb if liked else 1.0) * _z(self.item_vecs @ profile)
            # log1p: số lượt thích lệch rất mạnh về vài phim, lấy log để phim nổi tiếng không lấn át
            score += (w_pop if liked else 1.0) * _z(np.log1p(self.pop))
            if genres and not liked:
                # cold-start: chưa thích phim nào thì chỉ gợi ý trong các thể loại đã chọn
                chosen = set(genres)
                match = self.movies["genres"].map(lambda g: bool(chosen & set(g.split("|")))).to_numpy()
                score[~match] = -np.inf
        if ratings:
            score[list(ratings)] = -np.inf  # không gợi ý lại phim đã chấm

        out = []
        for j in np.argsort(-score)[:k]:
            if not np.isfinite(score[j]):
                break
            row = self.movies.iloc[j]
            reason = "Đang được nhiều người thích"
            if knn_part is not None and knn_part[:, j].max() > 0:
                src = liked[int(knn_part[:, j].argmax())]
                reason = f"Vì bạn thích {self.movies.iloc[src]['title']}"
            elif set(genres) & set(row["genres"].split("|")):
                reason = "Hợp thể loại bạn chọn"
            out.append(self._movie(int(j), score=float(score[j]), reason=reason, method=method))
        return out

    def similar(self, item: int, k: int = 8) -> list[dict]:
        """Phim tương tự = giống về người xem (ItemKNN) VÀ giống về nội dung.
        Chỉ dùng ItemKNN thì phim quá nổi tiếng hay lọt vào vì ai cũng xem."""
        score = _z(self.sim[item].toarray().ravel()) + _z(self.item_vecs @ self.item_vecs[item])
        score[item] = -np.inf
        return [self._movie(int(j)) for j in np.argsort(-score)[:k]]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        hits = np.flatnonzero(self._title_lower.str.contains(q, regex=False).to_numpy())
        hits = hits[np.argsort(-self.pop[hits])][:limit]  # phim phổ biến hơn lên trước
        return [self._movie(int(j)) for j in hits]

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
