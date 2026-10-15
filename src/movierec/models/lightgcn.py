"""LightGCN (He et al., 2020): lọc cộng tác bằng mạng nơ-ron đồ thị, viết bằng PyTorch.

Người dùng và phim là các đỉnh của một đồ thị hai phía; mỗi lượt thích (chấm >= min_rating) là một cạnh.
Mỗi đỉnh bắt đầu với một vector ẩn e^(0). Một lớp lan truyền lấy trung bình có chuẩn hóa vector của các láng giềng:

    e_u^(k+1) = Σ_{i ∈ N(u)} e_i^(k) / sqrt(|N(u)| · |N(i)|)      (tương tự cho phim)

Sau K lớp, vector cuối là trung bình của e^(0), ..., e^(K). Nhờ vậy sở thích lan qua đường đi
người dùng -> phim -> người dùng khác -> phim khác, điều mà phân rã ma trận chỉ học gián tiếp.
LightGCN bỏ hàm kích hoạt và ma trận biến đổi của GCN thông thường: chỉ còn vector ẩn là tham số.

Hàm mất mát BPR: với mỗi cặp (người dùng, phim đã thích) lấy ngẫu nhiên một phim khác làm mẫu âm,
và học sao cho điểm phim đã thích cao hơn điểm phim mẫu âm.

File này cần PyTorch nên KHÔNG được import trong movierec.models.__init__;
dùng: from movierec.models.lightgcn import LightGCN
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import Dataset
from .base import Recommender


class LightGCN(Recommender):
    name = "LightGCN"

    def __init__(self, dim: int = 64, layers: int = 3, lr: float = 1e-3, reg: float = 1e-4, epochs: int = 300,
                 batch: int = 4096, min_rating: float = 4.0, patience: int = 6, eval_every: int = 5,
                 seed: int = 42, device: str | None = None):
        self.dim, self.layers, self.lr, self.reg = dim, layers, lr, reg
        self.epochs, self.batch, self.min_rating = epochs, batch, min_rating
        self.patience, self.eval_every, self.seed = patience, eval_every, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- đồ thị ----------
    def _graph(self, users: np.ndarray, items: np.ndarray) -> torch.Tensor:
        """Ma trận kề chuẩn hóa D^-1/2 A D^-1/2 của đồ thị hai phía, kích thước (U + I) x (U + I), dạng thưa."""
        n = self.n_users_ + self.n_items_
        rows = np.concatenate([users, items + self.n_users_])
        cols = np.concatenate([items + self.n_users_, users])
        deg = np.bincount(rows, minlength=n).astype(np.float32)
        inv = np.zeros(n, dtype=np.float32)
        inv[deg > 0] = deg[deg > 0] ** -0.5  # đỉnh không có cạnh giữ hệ số 0, tránh chia cho 0
        vals = inv[rows] * inv[cols]
        idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.as_tensor(vals), (n, n), check_invariants=True).coalesce().to(self.device)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Vector cuối của người dùng và phim: trung bình các lớp 0..K."""
        e = self.emb_.weight
        total = e
        for _ in range(self.layers):
            e = torch.sparse.mm(self.graph_, e)
            total = total + e
        final = total / (self.layers + 1)
        return final[: self.n_users_], final[self.n_users_:]

    # ---------- huấn luyện ----------
    def fit(self, ds: Dataset, df: pd.DataFrame | None = None, eval_fn=None, verbose: bool = False) -> "LightGCN":
        """eval_fn(model) -> số càng cao càng tốt, gọi mỗi eval_every vòng; dừng sớm và giữ trọng số tốt nhất."""
        df = ds.train if df is None else df
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_users_, self.n_items_ = ds.n_users, ds.n_items
        pos = df[df["rating"] >= self.min_rating]
        users, items = pos["user"].to_numpy(), pos["item"].to_numpy()
        self.graph_ = self._graph(users, items)
        self.emb_ = nn.Embedding(self.n_users_ + self.n_items_, self.dim).to(self.device)
        nn.init.normal_(self.emb_.weight, std=0.1)
        opt = torch.optim.Adam(self.emb_.parameters(), lr=self.lr)
        u_all = torch.tensor(users, dtype=torch.long, device=self.device)  # tensor(): tạo bản sao
        i_all = torch.tensor(items, dtype=torch.long, device=self.device)  # tensor(): tạo bản sao

        best, best_state, bad = -np.inf, None, 0
        self.history_: list[dict] = []
        self.best_epoch_ = self.epochs
        for epoch in range(1, self.epochs + 1):
            order = torch.as_tensor(rng.permutation(len(u_all)), device=self.device)
            total, steps = 0.0, 0
            for start in range(0, len(order), self.batch):
                idx = order[start:start + self.batch]
                u, i = u_all[idx], i_all[idx]
                j = torch.randint(0, self.n_items_, (len(idx),), device=self.device)  # mẫu âm ngẫu nhiên
                eu, ei = self._propagate()
                pos_s = (eu[u] * ei[i]).sum(-1)
                neg_s = (eu[u] * ei[j]).sum(-1)
                loss = nn.functional.softplus(neg_s - pos_s).mean()  # BPR: -log sigmoid(pos - neg)
                ego = self.emb_.weight
                l2 = (ego[u].pow(2).sum() + ego[self.n_users_ + i].pow(2).sum()
                      + ego[self.n_users_ + j].pow(2).sum()) / (2 * len(idx))
                loss = loss + self.reg * l2
                if not torch.isfinite(loss):
                    raise FloatingPointError("Mất mát không hữu hạn (NaN/inf); kiểm tra lại GPU hoặc tốc độ học")
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
            record = {"epoch": epoch, "loss": total / max(steps, 1)}
            if eval_fn is not None and (epoch % self.eval_every == 0 or epoch == self.epochs):
                score = float(eval_fn(self))
                record["val"] = score
                if score > best + 1e-5:
                    best, bad, self.best_epoch_ = score, 0, epoch
                    best_state = copy.deepcopy(self.emb_.state_dict())
                else:
                    bad += 1
            self.history_.append(record)
            if verbose:
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in record.items()), flush=True)
            if eval_fn is not None and bad >= self.patience:
                break
        if best_state is not None:
            self.emb_.load_state_dict(best_state)
        return self

    # ---------- gợi ý ----------
    @torch.no_grad()
    def score_users(self, users: np.ndarray) -> np.ndarray:
        eu, ei = self._propagate()
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        scores = eu[u] @ ei.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Điểm không hữu hạn (NaN/inf); kết quả GPU có thể bị sai")
        return scores.cpu().numpy()
