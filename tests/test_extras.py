import pandas as pd

from movierec.extras import load_movie_extras, pick_trailer


def test_pick_trailer_prefers_official_youtube_trailer():
    videos = [
        {"site": "YouTube", "key": "clip1", "type": "Clip", "official": True, "iso_639_1": "en", "size": 1080},
        {"site": "Vimeo", "key": "vim", "type": "Trailer", "official": True, "iso_639_1": "en", "size": 1080},
        {"site": "YouTube", "key": "fan", "type": "Trailer", "official": False, "iso_639_1": "en", "size": 1080},
        {"site": "YouTube", "key": "off", "type": "Trailer", "official": True, "iso_639_1": "en", "size": 720},
        {"site": "YouTube", "key": "fr", "type": "Trailer", "official": True, "iso_639_1": "fr", "size": 1080},
    ]
    assert pick_trailer(videos)["key"] == "off"
    assert pick_trailer(videos[:1])["key"] == "clip1"     # không có trailer thì lấy clip
    assert pick_trailer(videos[1:2]) is None and pick_trailer([]) is None  # chỉ nhận YouTube


def test_load_movie_extras_joins_and_tolerates_missing_files(tmp_path):
    pd.DataFrame({"movie_id": [1, 2], "tmdb_id": [10, 20], "runtime": [81, 104], "vote_average": [8.0, 7.2],
                  "vote_count": [100, 50], "release_date": ["1995-11-22", ""], "director": ["A", "B"],
                  "cast": ["X|Y", "Z"]}).to_csv(tmp_path / "tmdb.csv", index=False)
    pd.DataFrame({"movie_id": [1], "tmdb_id": [10], "youtube_key": ["abc"], "trailer_name": ["Official"],
                  "trailer_type": ["Trailer"]}).to_csv(tmp_path / "trailers.csv", index=False)
    ex = load_movie_extras(tmp_path / "tmdb.csv", tmp_path / "trailers.csv")
    assert ex.loc[1, "youtube_key"] == "abc" and pd.isna(ex.loc[2, "youtube_key"]) and ex.loc[2, "runtime"] == 104
    empty = load_movie_extras(tmp_path / "khong.csv", tmp_path / "khong2.csv")
    assert empty.empty and "youtube_key" in empty.columns
