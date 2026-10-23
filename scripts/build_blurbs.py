"""Viết sẵn câu giới thiệu tiếng Việt cho mọi phim bằng LLM cục bộ (Ollama), lưu vào artifacts/blurbs_vi.json.

Lúc chat, trợ lý chỉ còn phải gọi LLM để hiểu câu hỏi (~8 giây trên CPU) thay vì viết thêm 5 câu giới thiệu
(~40 giây). Các trang khác của web cũng dùng các câu này để hiện mô tả tiếng Việt.

Chạy:  python scripts/build_blurbs.py            # chạy lâu (vài giờ trên CPU), dừng bằng Ctrl+C rồi chạy lại sẽ làm tiếp
       python scripts/build_blurbs.py --limit 50 # thử trước với 50 phim phổ biến nhất
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movierec.assistant import Assistant, OllamaClient, load_blurbs, save_blurbs  # noqa: E402
from movierec.service import DEFAULT_ARTIFACT, load_or_build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "blurbs_vi.json"))
    ap.add_argument("--limit", type=int, default=0, help="chỉ viết cho N phim phổ biến nhất chưa có câu (0 = tất cả)")
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()

    client = OllamaClient(args.model)
    if not client.available():
        sys.exit(f"Chưa kết nối được Ollama với mô hình {args.model}. Mở Ollama (hoặc `ollama pull {args.model}`) rồi chạy lại.")
    svc = load_or_build(str(DEFAULT_ARTIFACT))
    blurbs = load_blurbs(args.out)
    bot = Assistant(svc, None, client, blurbs=blurbs)

    # phim phổ biến trước: dừng giữa chừng thì các phim hay được gợi ý nhất đã có câu
    order = np.argsort(-svc.pop)
    todo = [int(j) for j in order
            if str(svc.movies.iloc[j].get("overview") or "").strip() and svc.movies.iloc[j]["title"] not in bot._blurbs]
    if args.limit:
        todo = todo[:args.limit]
    no_overview = sum(1 for j in range(len(svc.movies)) if not str(svc.movies.iloc[j].get("overview") or "").strip())
    print(f"Đã có {len(bot._blurbs)} câu; cần viết {len(todo)} phim; {no_overview} phim không có mô tả TMDB nên bỏ qua.")
    print(f"Mô hình {args.model}, num_gpu={client.num_gpu}. Dừng bằng Ctrl+C, chạy lại sẽ làm tiếp.", flush=True)

    start, done, failed, streak = time.time(), 0, [], 0
    try:
        for n, j in enumerate(todo, 1):
            movie = svc._movie(j)
            text = bot.explain(movie)
            if text:
                done, streak = done + 1, 0
            else:
                failed.append(movie["title"])
                streak += 1
                if streak >= 10:  # mười lần hỏng liên tiếp: nhiều khả năng Ollama đã tắt
                    print("Mười phim liên tiếp không viết được, có thể Ollama đã dừng. Lưu lại và thoát.")
                    break
            if n % args.save_every == 0:
                save_blurbs(args.out, bot._blurbs)
            if n % 10 == 0 or n == len(todo):
                per = (time.time() - start) / n
                eta = per * (len(todo) - n) / 60
                print(f"[{n}/{len(todo)}] {per:.1f} s/phim, còn khoảng {eta:.0f} phút | "
                      f"{movie['title']}: {text or '(bỏ qua)'}", flush=True)
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu.")
    save_blurbs(args.out, bot._blurbs)
    print(f"Đã lưu {len(bot._blurbs)} câu vào {args.out} (lần này viết thêm {done}, hỏng {len(failed)}).")
    if failed:
        print("Phim chưa viết được (chạy lại để thử lần nữa):", "; ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))
    if bot.rejected:  # ghi các câu bị loại để xem vì sao (chép lời nhắc, chữ Hán, JSON hỏng...)
        log = ROOT / "logs" / "blurbs_rejected.txt"
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.writelines(f"{t}\t{x!r}\n" for t, x in bot.rejected)
        print(f"{len(bot.rejected)} câu bị loại, xem trong {log}")
    print("BLURBS_DONE")


if __name__ == "__main__":
    main()
