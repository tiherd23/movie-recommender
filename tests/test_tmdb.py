from movierec.tmdb import TMDBClient, parse_title, pick_best, short_names


def test_parse_title_handles_articles_and_alternates():
    assert parse_title("Matrix, The (1999)") == (["The Matrix"], 1999)
    assert parse_title("Toy Story (1995)") == (["Toy Story"], 1995)
    names, year = parse_title("City of Lost Children, The (Cité des enfants perdus, La) (1995)")
    assert names == ["The City of Lost Children", "La Cité des enfants perdus"] and year == 1995
    assert parse_title("Bug's Life, A (1998)")[0] == ["A Bug's Life"]
    assert parse_title("Enfer, L' (1994)")[0] == ["L'Enfer"]
    assert parse_title("Seven (Se7en) (1995)") == (["Seven", "Se7en"], 1995)
    assert parse_title("No Year Here") == (["No Year Here"], None)


def test_pick_best_prefers_matching_year():
    results = [{"id": 1, "title": "Hamlet", "release_date": "1996-12-25"},
               {"id": 2, "title": "Hamlet", "release_date": "1990-12-19"},
               {"id": 3, "title": "Hamlet 2", "release_date": "1990-01-01"}]
    assert pick_best(results, ["Hamlet"], 1990)[0]["id"] == 2
    assert pick_best([], ["Hamlet"], 1990) == (None, -1.0)


def test_short_names_and_tie_breaks():
    names, year = parse_title("Star Wars: Episode IV - A New Hope (1977)")
    assert short_names(names) == ["Star Wars"]
    results = [{"id": 11, "title": "Star Wars", "release_date": "1977-05-25", "vote_count": 20000},
               {"id": 99, "title": "Star Wars: Deleted Magic", "release_date": "2005-01-01", "vote_count": 50}]
    best, score = pick_best(results, names, year, short_names(names))
    assert best["id"] == 11 and score > 0.85
    # trùng mức khớp tên: chọn phim nhiều lượt bình chọn hơn
    results = [{"id": 1, "title": "Flintstones: Viva Vacation", "release_date": "1983-05-10", "vote_count": 5},
               {"id": 2, "title": "National Lampoon's Vacation", "release_date": "1983-07-29", "vote_count": 2500}]
    assert pick_best(results, ["Vacation"], 1983)[0]["id"] == 2


def test_match_builds_row_without_network(monkeypatch):
    def fake_get(self, path, **params):
        if path == "/search/movie":
            return {"results": [{"id": 603, "title": "The Matrix", "release_date": "1999-03-31"}]}
        return {"title": "The Matrix", "release_date": "1999-03-31", "overview": "A hacker\nlearns the truth.",
                "poster_path": "/p.jpg", "runtime": 136, "genres": [{"name": "Action"}, {"name": "Science Fiction"}],
                "keywords": {"keywords": [{"name": "dystopia"}]},
                "credits": {"cast": [{"name": "Keanu Reeves"}],
                            "crew": [{"name": "Lana Wachowski", "job": "Director"}, {"name": "X", "job": "Editor"}]}}

    monkeypatch.setattr(TMDBClient, "get", fake_get)
    row = TMDBClient("k").match("Matrix, The (1999)")
    assert row["tmdb_id"] == 603 and row["match_score"] == 1.0
    assert row["overview"] == "A hacker learns the truth." and row["poster_path"] == "/p.jpg"
    assert row["tmdb_genres"] == "Action|Science Fiction" and row["director"] == "Lana Wachowski"

    monkeypatch.setattr(TMDBClient, "get", lambda self, path, **p: {"results": []})
    assert TMDBClient("k").match("Unknown Film (1950)")["tmdb_id"] is None
