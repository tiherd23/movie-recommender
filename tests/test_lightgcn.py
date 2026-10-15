import numpy as np
import pytest

torch = pytest.importorskip("torch")  # bỏ qua nếu máy chưa cài PyTorch

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import Popularity  # noqa: E402
from movierec.models.lightgcn import LightGCN  # noqa: E402


def test_graph_is_symmetric_and_normalized():
    m = LightGCN(device="cpu")
    m.n_users_, m.n_items_ = 2, 3
    g = m._graph(np.array([0, 0, 1]), np.array([0, 1, 1])).to_dense()
    assert torch.allclose(g, g.T)
    # cạnh (người 0, phim 0): người 0 có 2 cạnh, phim 0 có 1 cạnh -> 1/sqrt(2*1)
    assert torch.isclose(g[0, 2], torch.tensor(2 ** -0.5))
    assert g[:, 4].sum() == 0  # phim 2 không có cạnh nào


def test_lightgcn_learns_group_structure(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    model = LightGCN(dim=16, layers=2, lr=1e-2, epochs=40, batch=512, patience=40, eval_every=5, device="cpu")
    model.fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"])
    scores = model.score_users(np.arange(5))
    assert scores.shape == (5, ds.n_items) and np.isfinite(scores).all()
    assert model.history_[-1]["loss"] < model.history_[0]["loss"]
    assert evaluate(model, ds)["NDCG@10"] > evaluate(Popularity().fit(ds), ds)["NDCG@10"]
