import numpy as np
import pytest

torch = pytest.importorskip("torch")  # bỏ qua nếu máy chưa cài PyTorch

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import Popularity  # noqa: E402
from movierec.models.sasrec import SASRec  # noqa: E402


def test_sequences_are_chronological_and_left_padded(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    model = SASRec(max_len=8, device="cpu")
    seqs = model._sequences(ds, ds.train)
    assert seqs.shape == (ds.n_users, 9)
    u = 3
    hist = ds.train[ds.train["user"] == u].sort_values("timestamp")["item"].to_numpy() + 1
    assert list(seqs[u, -len(hist[-9:]):]) == list(hist[-9:])  # đúng thứ tự thời gian, lấy phần gần nhất
    short = SASRec(max_len=60, device="cpu")._sequences(ds, ds.train)
    assert (short[u, :60 - len(hist) + 1] == 0).all()  # đệm 0 ở bên trái


def test_mask_blocks_future_items():
    from movierec.models.sasrec import _SASRecNet

    net = _SASRecNet(n_items=10, dim=8, max_len=5, layers=1, heads=2, dropout=0.0).eval()
    a = torch.tensor([[0, 1, 2, 3, 4]])
    b = torch.tensor([[0, 1, 2, 3, 9]])  # chỉ khác phim cuối
    ha, hb = net(a), net(b)
    assert torch.isfinite(ha).all()
    assert torch.allclose(ha[:, :4], hb[:, :4], atol=1e-6)  # các vị trí trước không thấy phim cuối
    assert not torch.allclose(ha[:, 4], hb[:, 4])


def test_sasrec_learns_group_structure(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    model = SASRec(dim=16, max_len=30, layers=1, heads=1, dropout=0.1, epochs=40, batch=64, lr=5e-3,
                   patience=40, eval_every=5, device="cpu")
    model.fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"])
    scores = model.score_users(np.arange(5))
    assert scores.shape == (5, ds.n_items) and np.isfinite(scores).all()
    assert model.history_[-1]["loss"] < model.history_[0]["loss"]
    assert 1 <= model.best_epoch_ <= 40
    assert evaluate(model, ds)["NDCG@10"] > evaluate(Popularity().fit(ds), ds)["NDCG@10"]


def test_save_load_and_shuffled_copy(fake_ml1m, tmp_path):
    ds = build_dataset(fake_ml1m)
    model = SASRec(dim=8, max_len=20, layers=1, heads=1, epochs=2, batch=64, device="cpu").fit(ds)
    path = tmp_path / "sasrec.pt"
    model.save(path)
    loaded = SASRec.load(path, device="cpu")
    assert np.allclose(loaded.score_users(np.arange(4)), model.score_users(np.arange(4)), atol=1e-5)
    shuf = model.shuffled_copy(0)
    a, b = model.history_seq_.numpy(), shuf.history_seq_.numpy()
    assert all(sorted(x) == sorted(y) for x, y in zip(a, b))  # cùng tập phim
    assert (a != b).any() and (model.history_seq_.numpy() == a).all()  # thứ tự đổi, bản gốc không bị sửa


def test_shuffle_order_keeps_same_items(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    plain = SASRec(max_len=8, device="cpu")._sequences(ds, ds.train)
    shuf = SASRec(max_len=8, device="cpu", shuffle_order=True)._sequences(ds, ds.train)
    assert all(sorted(a) == sorted(b) for a, b in zip(plain, shuf))
    assert (plain != shuf).any()
