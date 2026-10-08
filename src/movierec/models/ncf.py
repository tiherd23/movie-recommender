"""Neural Collaborative Filtering - mô hình NeuMF (He et al., 2017), viết bằng PyTorch.

NeuMF ghép hai nhánh cùng nhận (người dùng, phim):
    GMF: nhân từng phần tử hai vector ẩn  -> giống phân rã ma trận, nhưng có trọng số học được
    MLP: nối hai vector ẩn rồi qua các tầng ẩn -> học tương tác phi tuyến
Đầu ra: một số thực (logit), càng lớn càng có khả năng người dùng thích phim.

Huấn luyện trên phản hồi ngầm: phim được chấm >= min_rating là mẫu dương (nhãn 1);
với mỗi mẫu dương lấy ngẫu nhiên n_neg phim làm mẫu âm (nhãn 0). Hàm mất mát: binary cross-entropy.

File này cần PyTorch nên KHÔNG được import trong movierec.models.__init__;
dùng: from movierec.models.ncf import NeuMF
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import Dataset
from .base import Recommender


class _NeuMFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, layers: tuple[int, ...]):
        super().__init__()
        half = layers[0] // 2
        self.user_gmf, self.item_gmf = nn.Embedding(n_users, factors), nn.Embedding(n_items, factors)
        self.user_mlp, self.item_mlp = nn.Embedding(n_users, half), nn.Embedding(n_items, half)
        blocks, width = [], half * 2
        for hidden in layers[1:]:
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(factors + width, 1)
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class NeuMF(Recommender):
    name = "NeuMF"

    def __init__(self, factors: int = 32, layers: tuple[int, ...] = (64, 32, 16), lr: float = 1e-3,
                 weight_decay: float = 1e-6, epochs: int = 40, batch: int = 2048, n_neg: int = 4,
                 min_rating: float = 4.0, patience: int = 4, seed: int = 42, device: str | None = None):
        self.factors, self.layers, self.lr, self.weight_decay = factors, tuple(layers), lr, weight_decay
        self.epochs, self.batch, self.n_neg, self.min_rating = epochs, batch, n_neg, min_rating
        self.patience, self.seed = patience, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self, ds: Dataset, df: pd.DataFrame | None = None, eval_fn=None, verbose: bool = False) -> "NeuMF":
        """eval_fn(model) -> số càng cao càng tốt (ví dụ NDCG@10 trên val). Có eval_fn thì dừng sớm
        và giữ lại trọng số của vòng tốt nhất; số vòng đó lưu ở best_epoch_."""
        df = ds.train if df is None else df
        torch.manual_seed(self.seed)
        pos = df[df["rating"] >= self.min_rating]
        users = torch.tensor(pos["user"].to_numpy(), dtype=torch.long, device=self.device)  # tensor(): tạo bản sao
        items = torch.tensor(pos["item"].to_numpy(), dtype=torch.long, device=self.device)
        self.n_items_ = ds.n_items
        self.net_ = _NeuMFNet(ds.n_users, ds.n_items, self.factors, self.layers).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best, best_state, bad = -np.inf, None, 0
        self.history_: list[dict] = []
        self.best_epoch_ = self.epochs
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            perm = torch.randperm(len(users), device=self.device)
            total, steps = 0.0, 0
            for start in range(0, len(perm), self.batch):
                idx = perm[start:start + self.batch]
                u_pos, i_pos = users[idx], items[idx]
                # mẫu âm: phim ngẫu nhiên (hiếm khi trùng phim đã thích, chấp nhận nhiễu nhỏ này)
                u_neg = u_pos.repeat_interleave(self.n_neg)
                i_neg = torch.randint(0, self.n_items_, (len(u_neg),), device=self.device)
                u_all, i_all = torch.cat([u_pos, u_neg]), torch.cat([i_pos, i_neg])
                labels = torch.cat([torch.ones(len(u_pos), device=self.device),
                                    torch.zeros(len(u_neg), device=self.device)])
                loss = loss_fn(self.net_(u_all, i_all), labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
            record = {"epoch": epoch, "loss": total / max(steps, 1)}
            if eval_fn is not None:
                score = float(eval_fn(self))
                record["val"] = score
                if score > best + 1e-5:
                    best, bad, self.best_epoch_ = score, 0, epoch
                    best_state = copy.deepcopy(self.net_.state_dict())
                else:
                    bad += 1
            self.history_.append(record)
            if verbose:
                print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in record.items()), flush=True)
            if eval_fn is not None and bad >= self.patience:
                break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score_users(self, users: np.ndarray) -> np.ndarray:
        """Chấm điểm mọi phim cho từng người dùng trong lô."""
        self.net_.eval()
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        all_items = torch.arange(self.n_items_, device=self.device)
        out = torch.empty((len(u), self.n_items_), device=self.device)
        for start in range(0, len(u), 256):  # 256 người x 3.700 phim mỗi lượt để vừa VRAM
            part = u[start:start + 256]
            pair_u = part.repeat_interleave(self.n_items_)
            pair_i = all_items.repeat(len(part))
            out[start:start + 256] = self.net_(pair_u, pair_i).view(len(part), self.n_items_)
        return out.cpu().numpy()
