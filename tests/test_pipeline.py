import numpy as np

from movierec.data import build_dataset
from movierec.evaluate import evaluate
from movierec.metrics import ranking_metrics, rmse
from movierec.models import BiasBaseline, ItemKNN, Popularity


def test_split_is_chronological_and_complete(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    assert len(ds.train) + len(ds.val) + len(ds.test) == 200 * 25
    assert len(ds.movies) == ds.n_items and len(ds.users) == ds.n_users
    assert ds.movies.loc[0, "title"].startswith("Movie")  # tên phim có ':' vẫn đọc đúng
    last_train = ds.train.groupby("user")["timestamp"].max()
    first_val = ds.val.groupby("user")["timestamp"].min()
    first_test = ds.test.groupby("user")["timestamp"].min()
    assert (last_train < first_val).all() and (ds.val.groupby("user")["timestamp"].max() < first_test).all()


def test_metrics_on_known_case():
    topk = np.array([[0, 1, 2], [3, 4, 5]])
    m = ranking_metrics(topk, [{0, 9}, set()], n_items=10)  # người dùng thứ 2 bị bỏ qua
    assert m["Precision@3"] == 1 / 3 and m["Recall@3"] == 0.5 and m["HitRate@3"] == 1.0
    assert abs(m["NDCG@3"] - 1 / (1 + 1 / np.log2(3))) < 1e-9
    assert m["Coverage"] == 0.6
    assert rmse([1, 2], [1, 4]) == 2 ** 0.5


def test_models_run_and_knn_beats_popularity(fake_ml1m):
    ds = build_dataset(fake_ml1m)
    res = {m.name: evaluate(m.fit(ds), ds, k=10) for m in [Popularity(), BiasBaseline(), ItemKNN(k=20)]}
    assert res["ItemKNN"]["NDCG@10"] > res["Popularity"]["NDCG@10"]
    assert 0 < res["BiasBaseline"]["RMSE"] < 2
    assert np.isnan(res["Popularity"]["RMSE"])


def test_new_models_run(fake_ml1m):
    from movierec.models import SVD, ContentBased, Hybrid, ImplicitALS

    ds = build_dataset(fake_ml1m)
    pop = evaluate(Popularity().fit(ds), ds)["NDCG@10"]
    als = ImplicitALS(factors=8, iters=5, min_rating=4.0).fit(ds)
    assert evaluate(als, ds)["NDCG@10"] > pop  # ALS phải học được cấu trúc nhóm

    svd = SVD(factors=8, epochs=20).fit(ds, val=ds.val)
    # dừng sớm phải giữ lại đúng vòng có RMSE thấp nhất trên val
    assert abs(min(svd.history_) - svd.history_[svd.best_epoch_ - 1]) < 1e-3
    assert evaluate(svd, ds)["RMSE"] < 2

    cb = ContentBased().fit(ds)
    assert cb.score_users(np.arange(3)).shape == (3, ds.n_items)
    hyb = Hybrid([als, cb], [1.0, 0.2], prefit=True)
    assert np.isfinite(hyb.score_users(np.arange(5))).all()


def test_two_stage_ranker(fake_ml1m):
    from movierec.ranker import TwoStageRanker

    ds = build_dataset(fake_ml1m)
    ranker = TwoStageRanker({"factors": 8, "alpha": 1.0, "min_rating": 4.0, "iters": 5}, {"k": 20, "shrink": 0.0},
                            n_candidates={"als": 30, "knn": 30, "pop": 20, "content": 10},
                            lgb_params={"min_data_in_leaf": 5, "num_leaves": 7})
    info = ranker.fit_ranker(ds, ds.train, ds.val, num_rounds=30)
    assert 0 < info["candidate_recall"] <= 1 and info["rows"] > 0
    assert abs(ranker.importance_.sum() - 1) < 1e-6
    scores = ranker.fit(ds).score_users(np.arange(4))
    assert scores.shape == (4, ds.n_items) and (scores > -1e8).any()
    res = evaluate(ranker, ds)
    assert res["NDCG@10"] > evaluate(Popularity().fit(ds), ds)["NDCG@10"]
