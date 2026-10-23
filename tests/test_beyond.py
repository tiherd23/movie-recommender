import numpy as np
import pandas as pd

from movierec.beyond import beyond_metrics, genre_vectors, gini, head_items, mmr_rerank


def test_genre_vectors_and_metrics():
    g = genre_vectors(pd.Series(["Action", "Action|Comedy", "Comedy", "Drama"]))
    assert g.shape == (4, 3) and np.allclose(np.linalg.norm(g, axis=1), 1.0)
    pop = np.array([100, 50, 10, 1])
    same = beyond_metrics(np.array([[0, 0]]), pop, 100, g)        # hai phim giống hệt: ILD = 0
    mixed = beyond_metrics(np.array([[0, 3]]), pop, 100, g)       # Action và Drama: ILD = 1
    assert same["ILD"] == 0.0 and np.isclose(mixed["ILD"], 1.0) and mixed["Genres"] == 2
    assert mixed["Novelty"] > beyond_metrics(np.array([[0, 1]]), pop, 100, g)["Novelty"]  # phim ít người xem mới hơn
    assert head_items(pop).tolist() == [True, True, False, False]  # 2 phim đầu chiếm >= 80% lượt chấm
    assert gini([5, 5, 5]) == 0.0 and gini([0, 0, 9]) > 0.6


def test_mmr_trades_relevance_for_diversity():
    g = genre_vectors(pd.Series(["Action"] * 5 + ["Comedy", "Drama"]))
    scores = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.5, 0.4, -np.inf])[:7]
    plain = mmr_rerank(scores, g, lam=1.0, k=3)
    diverse = mmr_rerank(scores, g, lam=0.5, k=3)
    assert plain.tolist() == [0, 1, 2]                    # lam = 1: giữ thứ tự gốc
    assert diverse[0] == 0 and set(diverse[1:]) == {5, 6}  # phim đầu giữ nguyên, sau đó chọn thể loại khác
    scores[3] = -np.inf
    assert 3 not in mmr_rerank(scores, g, lam=0.5, k=5)   # không chọn phim bị loại (đã xem)



def test_popularity_penalty_pushes_niche_items_up():
    g = genre_vectors(pd.Series(["Action"] * 6))
    scores = np.array([1.0, 0.98, 0.96, 0.94, 0.5, 0.45])
    pop = np.array([5000, 4000, 3000, 2000, 3, 2])
    assert mmr_rerank(scores, g, 1.0, k=3, pop=pop, beta=0.0).tolist() == [0, 1, 2]
    assert set(mmr_rerank(scores, g, 1.0, k=3, pop=pop, beta=1.0)) & {4, 5}  # phim ít người xem được đẩy lên
