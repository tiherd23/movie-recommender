import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def fake_ml1m(tmp_path_factory) -> Path:
    """Tạo dữ liệu giả đúng định dạng ML-1M: 4 nhóm người dùng, mỗi nhóm thích 1 nhóm phim."""
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("ml-1m")
    n_users, n_items, n_groups = 200, 120, 4
    lines = []
    for u in range(1, n_users + 1):
        g = u % n_groups
        own = [i for i in range(1, n_items + 1) if i % n_groups == g]
        other = [i for i in range(1, n_items + 1) if i % n_groups != g]
        picks = list(rng.choice(own, 20, replace=False)) + list(rng.choice(other, 5, replace=False))
        rng.shuffle(picks)
        for t, i in enumerate(picks):
            rating = int(rng.integers(4, 6)) if i % n_groups == g else int(rng.integers(1, 4))
            lines.append(f"{u}::{i}::{rating}::{1000 + t}")
    (d / "ratings.dat").write_text("\n".join(lines) + "\n")
    (d / "movies.dat").write_text(
        "".join(f"{i}::Movie {i}: Part {i} (2000)::Drama|Comedy\n" for i in range(1, n_items + 1)),
        encoding="latin-1",
    )
    (d / "users.dat").write_text("".join(f"{u}::M::25::4::12345\n" for u in range(1, n_users + 1)))
    return d
