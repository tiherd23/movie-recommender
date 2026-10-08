import numpy as np
import pytest

torch = pytest.importorskip("torch")  # bỏ qua nếu máy chưa cài PyTorch

from movierec.data import build_dataset  # noqa: E402
from movierec.evaluate import evaluate  # noqa: E402
from movierec.models import Popularity  # noqa: E402
from movierec.models.ncf import NeuMF  # noqa: E402


def test_neumf_learns_group_structure(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    model = NeuMF(factors=8, layers=(16, 8), epochs=30, batch=256, lr=5e-3, patience=30, device="cpu")
    model.fit(ds, ds.train, eval_fn=lambda m: evaluate(m, ds, "val")["NDCG@10"])
    scores = model.score_users(np.arange(5))
    assert scores.shape == (5, ds.n_items) and np.isfinite(scores).all()
    assert 1 <= model.best_epoch_ <= 30 and len(model.history_) >= 1
    assert model.history_[-1]["loss"] < model.history_[0]["loss"]  # mất mát phải giảm
    assert evaluate(model, ds)["NDCG@10"] > evaluate(Popularity().fit(ds), ds)["NDCG@10"]
