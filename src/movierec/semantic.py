"""Tìm phim theo ngữ nghĩa: người dùng gõ một câu tiếng Việt, hệ thống trả về phim có nội dung phù hợp.

Ý tưởng: một mô hình nhúng văn bản đa ngôn ngữ biến cả câu hỏi tiếng Việt lẫn mô tả phim tiếng Anh
thành vector trong CÙNG một không gian. Hai câu cùng nghĩa, dù khác ngôn ngữ, cho hai vector gần nhau.
Tìm kiếm khi đó chỉ là lấy các phim có vector gần vector câu hỏi nhất (độ tương đồng cosine).

    Ngoại tuyến: nhúng mô tả của toàn bộ phim một lần, lưu thành ma trận (artifacts/semantic.npz)
    Trực tuyến : nhúng câu hỏi (vài chục mili-giây) rồi nhân ma trận để lấy top-k

Mô hình họ E5 yêu cầu tiền tố "query: " cho câu hỏi và "passage: " cho văn bản được tìm; các mô hình khác
(ví dụ BAAI/bge-m3) không cần. Mô hình và trọng số độ phổ biến được chọn bằng scripts/eval_semantic.py.

Trộn độ phổ biến: độ tương đồng ngữ nghĩa giữa các phim thường sít nhau, nên phim ít người biết dễ chen lên
trước phim nổi tiếng cùng chủ đề. Điểm cuối = z(độ tương đồng) + pop_weight * z(log độ phổ biến).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import ROOT

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def prefixes(model_name: str) -> tuple[str, str]:
    """(tiền tố câu hỏi, tiền tố văn bản) theo quy ước của từng họ mô hình."""
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-8)
DEFAULT_INDEX = ROOT / "artifacts" / "semantic.npz"
Encoder = Callable[[list[str]], np.ndarray]


def movie_passage(row: pd.Series) -> str:
    """Ghép thông tin của một phim thành đoạn văn để nhúng."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(row["title"]))
    m = re.match(r"^(.*),\s*(The|A|An)$", title)
    if m:
        title = f"{m.group(2)} {m.group(1)}"
    parts = [f"{title}.", "Genres: " + str(row["genres"]).replace("|", ", ") + "."]
    if row.get("overview"):
        parts.append(str(row["overview"]))
    if row.get("keywords"):
        parts.append("Keywords: " + str(row["keywords"]).replace("|", ", ") + ".")
    return " ".join(parts)


def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 32, half: bool = False) -> Encoder:
    """Nạp mô hình nhúng bằng thư viện sentence-transformers (cần cài riêng).

    device: "cuda", "cpu" hoặc None (tự chọn). half=True: dùng số thực 16 bit trên GPU để giảm một nửa bộ nhớ.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256  # mô tả phim ngắn; giới hạn để tiết kiệm bộ nhớ
    if half:
        model.half()

    def encode(texts: list[str]) -> np.ndarray:
        out = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 200)
        return np.asarray(out, dtype=np.float32)

    return encode


class SemanticIndex:
    def __init__(self, embeddings: np.ndarray, model_name: str = DEFAULT_MODEL, pop_weight: float = 0.0):
        self.pop_weight = float(pop_weight)
        emb = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)  # chuẩn hóa để tích vô hướng = cosine
        self.model_name = model_name
        self._encoder: Encoder | None = None

    # ---------- dựng / lưu / nạp ----------
    @classmethod
    def build(cls, movies: pd.DataFrame, encoder: Encoder, model_name: str = DEFAULT_MODEL) -> "SemanticIndex":
        passages = [prefixes(model_name)[1] + movie_passage(row) for _, row in movies.iterrows()]
        emb = np.asarray(encoder(passages), dtype=np.float32)
        if not np.isfinite(emb).all():  # thường do tính bằng số 16 bit bị tràn
            raise ValueError(f"Mô hình trả về vector không hợp lệ (NaN/inf) cho {int((~np.isfinite(emb)).any(axis=1).sum())} phim")
        index = cls(emb, model_name)
        index._encoder = encoder
        return index

    def save(self, path: Path | str = DEFAULT_INDEX) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embeddings=self.embeddings, model_name=np.array(self.model_name),
                            pop_weight=np.array(self.pop_weight))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX) -> "SemanticIndex":
        data = np.load(path, allow_pickle=False)
        weight = float(data["pop_weight"]) if "pop_weight" in data.files else 0.0
        if not np.isfinite(data["embeddings"]).all():
            raise ValueError(f"Chỉ mục {path} chứa giá trị không hợp lệ (NaN/inf); cần dựng lại")
        return cls(data["embeddings"], str(data["model_name"]), weight)

    def set_encoder(self, encoder: Encoder) -> "SemanticIndex":
        self._encoder = encoder
        return self

    # ---------- tìm kiếm ----------
    def similarities(self, query: str) -> np.ndarray:
        """Độ tương đồng cosine giữa câu hỏi và mọi phim."""
        if self._encoder is None:
            self._encoder = load_encoder(self.model_name)
        q = np.asarray(self._encoder([prefixes(self.model_name)[0] + query.strip()]), dtype=np.float32)[0]
        if not np.isfinite(q).all():
            raise ValueError("Mô hình trả về vector không hợp lệ (NaN/inf) cho câu hỏi")
        return self.embeddings @ (q / (np.linalg.norm(q) + 1e-8))

    def search(self, query: str, k: int = 12, exclude: set[int] | None = None,
               popularity: np.ndarray | None = None, pop_weight: float | None = None) -> list[tuple[int, float]]:
        """Trả về [(chỉ số phim, độ tương đồng cosine)], sắp theo điểm cuối giảm dần.

        popularity: số lượt thích của từng phim; nếu có, điểm xếp hạng được trộn thêm độ phổ biến
        với trọng số pop_weight (mặc định lấy giá trị đã lưu trong chỉ mục). exclude: các phim cần bỏ qua.
        """
        if not query.strip():
            return []
        sims = self.similarities(query)
        weight = self.pop_weight if pop_weight is None else pop_weight
        rank = _z(sims)
        if popularity is not None and weight:
            rank = rank + weight * _z(np.log1p(popularity))
        if exclude:
            rank[list(exclude)] = -np.inf
        top = np.argsort(-rank)[:k]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(rank[i])]
