"""Đánh giá bước hiểu câu chat của trợ lý trên 24 câu tiếng Việt có nhãn (data/eval/assistant_intents.csv).

So sánh các mô hình ngôn ngữ chạy qua Ollama theo:
    action_acc : tỉ lệ đoán đúng hành động (chat / recommend / similar / describe)
    genre_f1   : F1 trung bình của tập thể loại muốn xem (câu không có thể loại: đúng khi mô hình cũng để trống)
    exclude_acc: tỉ lệ khớp chính xác tập thể loại cần tránh
    year_acc   : tỉ lệ khớp cả hai mốc năm
    title_acc  : với câu "phim giống X", tên phim rút ra có chứa X
    latency    : thời gian trung bình mỗi câu (giây), không tính câu đầu tiên vì phải nạp mô hình

Chạy:  python scripts/eval_assistant.py --models qwen3:4b qwen2.5:3b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.assistant import Assistant, OllamaClient  # noqa: E402

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


class _Stub:  # parse() chỉ cần danh sách thể loại và bảng phim để khởi tạo
    genres = GENRES
    movies = pd.DataFrame({"title": ["x (1990)"], "genres": ["Drama"]})


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    hit = len(pred & gold)
    return 0.0 if not hit else 2 * hit / (len(pred) + len(gold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["qwen3:4b", "qwen2.5:3b"])
    ap.add_argument("--data", nargs="+", default=["assistant_intents.csv"],
                    help="các tệp câu có nhãn trong data/eval (bộ mới: assistant_intents_moi.csv assistant_intents_moi2.csv)")
    ap.add_argument("--tag", default="", help="hậu tố tên tệp kết quả, ví dụ _moi")
    ap.add_argument("--rules", action="store_true", help="chấm thêm bộ hiểu câu bằng luật (không cần Ollama)")
    args = ap.parse_args()
    data = pd.concat([pd.read_csv(ROOT / "data" / "eval" / f, keep_default_na=False) for f in args.data],
                     ignore_index=True)
    rows, details = [], []
    for model in (["luat"] if args.rules else []) + args.models:
        if model == "luat":  # bộ hiểu câu bằng luật, dùng khi không có LLM
            client = None
            bot = Assistant(_Stub(), None, None)
            print("== bộ hiểu câu bằng luật ==")
        else:
            client = OllamaClient(model)
            if not client.available():
                print(f"[bỏ qua] {model}: Ollama chưa chạy hoặc chưa tải mô hình (ollama pull {model})")
                continue
            bot = Assistant(_Stub(), None, client)
            client([{"role": "user", "content": "xin chào"}])  # nạp mô hình vào bộ nhớ trước khi bấm giờ
            print(f"== {model} | chạy trên: {'CPU' if client.num_gpu == 0 else 'GPU/tự chọn'} ==")
        stats = {"action": [], "genre": [], "exclude": [], "year": [], "title": [], "time": []}
        for _, r in data.iterrows():
            t0 = time.perf_counter()
            got = bot.parse(r["message"])
            stats["time"].append(time.perf_counter() - t0)
            gold_g = set(filter(None, r["genres"].split("|")))
            gold_x = set(filter(None, r["exclude_genres"].split("|")))
            stats["action"].append(got["action"] == r["action"])
            stats["genre"].append(f1(set(got["genres"]), gold_g))
            stats["exclude"].append(set(got["exclude_genres"]) == gold_x)
            stats["year"].append((got["year_from"], got["year_to"]) == (int(r["year_from"]), int(r["year_to"])))
            if r["similar_to"]:
                stats["title"].append(r["similar_to"].lower() in got["similar_to"].lower())
            details.append({"model": model, "message": r["message"], "gold_action": r["action"], **got})
            mark = "ok " if stats["action"][-1] else "SAI"
            print(f"  [{mark}] {r['message'][:50]:50s} -> {got['action']:9s} {got['genres']} "
                  f"{got['year_from'] or ''}-{got['year_to'] or ''} {got['similar_to']}")
        mean = lambda v: sum(v) / max(len(v), 1)
        if client is None:
            device = "luật"
        else:
            device = "CPU (GPU lỗi hoặc bị tắt)" if client.num_gpu == 0 else "GPU/tự chọn"
            print(f"  -> kết thúc trên: {device} | num_ctx {client.num_ctx}")
        rows.append({"model": model, "n": len(data), "action_acc": mean(stats["action"]), "genre_f1": mean(stats["genre"]),
                     "exclude_acc": mean(stats["exclude"]), "year_acc": mean(stats["year"]),
                     "title_acc": mean(stats["title"]), "latency_s": mean(stats["time"]), "device": device})
    if not rows:
        return
    out = pd.DataFrame(rows).round(3)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / f"assistant_eval{args.tag}.csv", index=False)
    pd.DataFrame(details).to_csv(ROOT / "results" / f"assistant_eval_details{args.tag}.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nĐã ghi results/assistant_eval{args.tag}.csv và results/assistant_eval_details{args.tag}.csv")


if __name__ == "__main__":
    main()
