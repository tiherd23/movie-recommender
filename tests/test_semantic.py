import numpy as np
import pandas as pd

from movierec.semantic import SemanticIndex, movie_passage

VOCAB = ["space", "love", "robot", "war", "cartoon", "không_gian", "tình_yêu"]
SYN = {"không_gian": "space", "tình_yêu": "love"}  # giả lập mô hình đa ngôn ngữ: từ Việt -> cùng hướng với từ Anh


def fake_encoder(texts):
    """Bộ nhúng giả: đếm từ khóa, không cần tải mô hình thật."""
    out = np.zeros((len(texts), 5), dtype=np.float32)
    for n, t in enumerate(texts):
        for w in t.lower().replace(",", " ").replace(".", " ").split():
            w = SYN.get(w, w)
            if w in VOCAB[:5]:
                out[n, VOCAB.index(w)] += 1
    return out + 1e-3


MOVIES = pd.DataFrame({
    "title": ["Star Trip, The (1990)", "Heart Song (1995)", "Metal Man (1999)"],
    "genres": ["Sci-Fi|Adventure", "Romance", "Action|Sci-Fi"],
    "overview": ["A crew lost in space.", "A story about love.", "A robot fights a war."],
    "keywords": ["space|ship", "love|music", "robot|war"],
})


def test_movie_passage_formats_title_and_fields():
    text = movie_passage(MOVIES.iloc[0])
    assert text.startswith("The Star Trip.") and "Genres: Sci-Fi, Adventure." in text
    assert "A crew lost in space." in text and "Keywords: space, ship." in text
    assert movie_passage(pd.Series({"title": "Solo (1996)", "genres": "Action"})) == "Solo. Genres: Action."


def test_search_ranks_cross_language_and_roundtrips(tmp_path):
    index = SemanticIndex.build(MOVIES, fake_encoder, "fake")
    assert index.embeddings.shape == (3, 5)
    assert index.search("phim về không_gian", k=1)[0][0] == 0  # câu tiếng Việt khớp phim mô tả bằng tiếng Anh
    assert index.search("phim tình_yêu", k=1)[0][0] == 1
    assert [i for i, _ in index.search("robot war", k=3)][0] == 2
    assert all(i != 2 for i, _ in index.search("robot war", k=3, exclude={2}))
    assert index.search("   ") == []
    path = tmp_path / "semantic.npz"
    index.save(path)
    loaded = SemanticIndex.load(path).set_encoder(fake_encoder)
    assert loaded.model_name == "fake" and np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.search("robot", k=1)[0][0] == 2


def test_popularity_blend_breaks_near_ties(tmp_path):
    emb = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]], dtype=np.float32)  # hai phim đầu gần như giống nhau
    index = SemanticIndex(emb, "fake", pop_weight=1.0).set_encoder(lambda texts: np.array([[1.0, 0.0]], dtype=np.float32))
    pop = np.array([5.0, 5000.0, 50.0])
    assert index.search("x", k=1)[0][0] == 0                      # không có độ phổ biến: theo ngữ nghĩa thuần
    assert index.search("x", k=1, popularity=pop)[0][0] == 1      # trộn độ phổ biến: phim nổi tiếng hơn lên trước
    assert index.search("x", k=1, popularity=pop, pop_weight=0.0)[0][0] == 0
    assert index.search("x", k=3, popularity=pop)[-1][0] == 2     # phim lạc đề vẫn đứng cuối
    path = tmp_path / "s.npz"; index.save(path)
    assert SemanticIndex.load(path).pop_weight == 1.0


def test_prefixes_by_model_family():
    from movierec.semantic import prefixes
    assert prefixes("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert prefixes("BAAI/bge-m3") == ("", "")


def test_nan_embeddings_are_rejected(tmp_path):
    import pytest

    bad = lambda texts: np.full((len(texts), 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        SemanticIndex.build(MOVIES, bad, "fake")
    index = SemanticIndex(np.eye(3, 4, dtype=np.float32), "fake").set_encoder(bad)
    with pytest.raises(ValueError):
        index.search("x")
    np.savez_compressed(tmp_path / "bad.npz", embeddings=np.full((2, 3), np.nan, dtype=np.float32), model_name=np.array("fake"))
    with pytest.raises(ValueError):
        SemanticIndex.load(tmp_path / "bad.npz")
