"""SASRec (Kang & McAuley, 2018): gợi ý theo chuỗi bằng cơ chế tự chú ý (self-attention), viết bằng PyTorch.

Các mô hình khác trong đồ án coi lịch sử của người dùng là một TẬP phim. SASRec đọc lịch sử như một CHUỖI theo thời
gian: tại mỗi vị trí t, mô hình chỉ được nhìn các phim từ 1..t (mặt nạ nhân quả) và học dự đoán phim thứ t+1.
Khi gợi ý, trạng thái ở vị trí cuối cùng (sau phim xem gần nhất) được nhân với vector của mọi phim để ra điểm.

    chuỗi phim đã xem -> vector phim + vector vị trí -> các lớp Transformer (mặt nạ nhân quả) -> h_t
    điểm của phim i tại bước t = h_t · e_i ; hàm mất mát: entropy chéo trên TOÀN BỘ kho phim

Dùng entropy chéo trên toàn kho thay cho lấy mẫu âm như bài gốc: kho chỉ khoảng 3.700 phim nên tính được trọn vẹn,
và cách này thường cho kết quả tốt hơn.

File này cần PyTorch nên KHÔNG được import trong movierec.models.__init__;
dùng: from movierec.models.sasrec import SASRec
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import Dataset
from .base import Recommender


class _SASRecNet(nn.Module):
    def __init__(self, n_items: int, dim: int, max_len: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)  # chỉ số 0 là ô trống (đệm)
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) chỉ số phim + 1, 0 = ô trống ở bên trái. Trả về trạng thái ẩn (B, L, dim)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device)
        x = self.drop(self.item(seq) * (self.item.embedding_dim ** 0.5) + self.pos(positions))
        # mặt nạ: không nhìn tương lai và không nhìn ô trống; luôn cho nhìn chính nó để không có hàng bị che hết (NaN)
        future = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        pad = (seq == 0).unsqueeze(1).expand(B, L, L)
        mask = (future.unsqueeze(0) | pad) & ~torch.eye(L, dtype=torch.bool, device=seq.device).unsqueeze(0)
        mask = mask.repeat_interleave(self.heads, dim=0)  # (B * số đầu chú ý, L, L)
        return self.norm(self.encoder(x, mask=mask))

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Điểm của mọi phim (bỏ cột ô trống): (..., dim) -> (..., n_items)."""
        return hidden @ self.item.weight[1:].T


class SASRec(Recommender):
    name = "SASRec"

    def __init__(self, dim: int = 64, max_len: int = 200, layers: int = 2, heads: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, weight_decay: float = 0.0, epochs: int = 200, batch: int = 128,
                 patience: int = 10, eval_every: int = 2, seed: int = 42, shuffle_order: bool = False,
                 device: str | None = None):
        """shuffle_order=True: xáo trộn thứ tự phim trong cửa sổ lịch sử của mỗi người (cả lúc học lẫn lúc gợi ý),
        để đo SASRec làm được bao nhiêu khi chỉ biết TẬP phim gần đây chứ không biết thứ tự."""
        self.dim, self.max_len, self.layers, self.heads, self.dropout = dim, max_len, layers, heads, dropout
        self.lr, self.weight_decay, self.epochs, self.batch = lr, weight_decay, epochs, batch
        self.patience, self.eval_every, self.seed, self.shuffle_order = patience, eval_every, seed, shuffle_order
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- dữ liệu ----------
    def _sequences(self, ds: Dataset, df: pd.DataFrame) -> np.ndarray:
        """Ma trận (n_users, max_len + 1): mỗi hàng là các phim gần nhất của người dùng theo thời gian, đệm 0 bên trái.
        Mọi lượt chấm đều được tính là một lần xem, kể cả điểm thấp: thứ tự xem là tín hiệu mô hình cần học."""
        df = df.sort_values(["user", "timestamp"], kind="stable")
        out = np.zeros((ds.n_users, self.max_len + 1), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        users, items = df["user"].to_numpy(), df["item"].to_numpy() + 1
        starts = np.r_[0, np.flatnonzero(np.diff(users)) + 1]
        ends = np.r_[starts[1:], len(users)]
        for s, e in zip(starts, ends):
            seq = items[max(s, e - self.max_len - 1):e]
            if self.shuffle_order:
                seq = rng.permutation(seq)
            out[users[s], -len(seq):] = seq
        return out

    # ---------- huấn luyện ----------
    def fit(self, ds: Dataset, df: pd.DataFrame | None = None, eval_fn=None, verbose: bool = False) -> "SASRec":
        """eval_fn(model) -> số càng cao càng tốt (ví dụ NDCG@10 trên val), được gọi mỗi eval_every vòng.
        Có eval_fn thì dừng sớm sau `patience` lần đánh giá không cải thiện và giữ trọng số tốt nhất."""
        df = ds.train if df is None else df
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items_ = ds.n_items
        full = self._sequences(ds, df)
        self.history_seq_ = torch.as_tensor(full[:, 1:], device=self.device)  # max_len phim gần nhất, dùng khi gợi ý
        train_users = np.flatnonzero((full > 0).sum(axis=1) >= 2)  # cần ít nhất 2 phim mới có cặp (đầu vào, đích)
        data = torch.as_tensor(full[train_users], device=self.device)
        self.net_ = _SASRecNet(ds.n_items, self.dim, self.max_len, self.layers, self.heads, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        best, best_state, bad = -np.inf, None, 0
        self.history_: list[dict] = []
        self.best_epoch_ = self.epochs
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            total, steps = 0.0, 0
            order = torch.as_tensor(rng.permutation(len(data)), device=self.device)
            for start in range(0, len(data), self.batch):
                batch = data[order[start:start + self.batch]]
                seq, target = batch[:, :-1], batch[:, 1:] - 1  # đích = phim kế tiếp; ô trống thành -1 và bị bỏ qua
                target = torch.where(seq == 0, torch.full_like(target, -1), target)
                logits = self.net_.logits(self.net_(seq))
                loss = loss_fn(logits.reshape(-1, self.n_items_), target.reshape(-1))
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

    # ---------- lưu / nạp ----------
    def save(self, path) -> None:
        cfg = {k: getattr(self, k) for k in ("dim", "max_len", "layers", "heads", "dropout", "lr", "weight_decay",
                                             "epochs", "batch", "patience", "eval_every", "seed", "shuffle_order")}
        torch.save({"config": cfg, "state": self.net_.state_dict(), "n_items": self.n_items_,
                    "history": self.history_seq_.cpu(), "best_epoch": self.best_epoch_}, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "SASRec":
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**data["config"], device=device)
        model.n_items_, model.best_epoch_ = int(data["n_items"]), int(data["best_epoch"])
        model.net_ = _SASRecNet(model.n_items_, model.dim, model.max_len, model.layers, model.heads,
                                model.dropout).to(model.device)
        model.net_.load_state_dict(data["state"])
        model.history_seq_ = data["history"].to(model.device)
        return model

    def shuffled_copy(self, seed: int = 0) -> "SASRec":
        """Bản sao dùng chung trọng số nhưng xáo trộn thứ tự phim trong lịch sử của mỗi người (giữ nguyên TẬP phim).
        Dùng để đo mô hình dựa vào THỨ TỰ xem đến mức nào."""
        rng = np.random.default_rng(seed)
        hist = self.history_seq_.cpu().numpy().copy()
        for row in hist:
            real = np.flatnonzero(row)
            row[real] = rng.permutation(row[real])
        other = copy.copy(self)
        other.history_seq_ = torch.as_tensor(hist, device=self.device)
        return other

    # ---------- gợi ý ----------
    @torch.no_grad()
    def score_users(self, users: np.ndarray) -> np.ndarray:
        """Điểm của mọi phim cho từng người dùng, từ trạng thái sau phim xem gần nhất."""
        self.net_.eval()
        u = torch.as_tensor(np.asarray(users), dtype=torch.long, device=self.device)
        out = np.empty((len(u), self.n_items_), dtype=np.float32)
        for start in range(0, len(u), 256):
            part = self.history_seq_[u[start:start + 256]]
            scores = self.net_.logits(self.net_(part)[:, -1])
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Điểm không hữu hạn (NaN/inf); kết quả GPU có thể bị sai")
            out[start:start + len(part)] = scores.cpu().numpy()
        return out
