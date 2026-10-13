"""Kiểm tra GPU có tính toán đúng và ổn định không (tìm nguyên nhân lỗi "CUDA error: an illegal instruction").

Chạy lần lượt các phép tính nặng dần trên GPU, so kết quả với CPU và ghi lại bước nào hỏng.
Mỗi bước in ra ngay nên nếu chương trình sập, dòng cuối cùng cho biết nó sập ở đâu.

Chạy:  python scripts/gpu_check.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # báo lỗi đúng tại phép tính gây lỗi

import torch  # noqa: E402


def step(name: str, fn) -> bool:
    print(f"[..] {name}", flush=True)
    t0 = time.perf_counter()
    try:
        info = fn()
        torch.cuda.synchronize()
    except Exception as err:  # noqa: BLE001
        print(f"[LOI] {name}: {type(err).__name__}: {str(err)[:200]}", flush=True)
        return False
    used = torch.cuda.max_memory_allocated() / 2**20
    print(f"[ok] {name} | {time.perf_counter() - t0:.1f}s | VRAM đỉnh {used:.0f} MB | {info or ''}", flush=True)
    return True


def matmul(n: int, loops: int):
    def run():
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(n, n, generator=g), torch.randn(n, n, generator=g)
        want = a @ b
        ga, gb = a.cuda(), b.cuda()
        worst = 0.0
        for _ in range(loops):
            worst = max(worst, (ga @ gb).cpu().sub(want).abs().max().item())
        if not worst < 1e-1:
            raise ValueError(f"kết quả GPU lệch CPU tới {worst}")
        return f"lệch lớn nhất so với CPU {worst:.2e}"
    return run


def attention(batch: int, seq: int, dim: int, loops: int):
    def run():
        layer = torch.nn.TransformerEncoderLayer(dim, 8, dim * 4, batch_first=True).cuda().eval()
        x = torch.randn(batch, seq, dim, device="cuda")
        with torch.no_grad():
            for _ in range(loops):
                y = layer(x)
        if not torch.isfinite(y).all():
            raise ValueError("đầu ra có NaN/inf")
        return f"đầu ra hữu hạn, chuẩn {y.norm().item():.1f}"
    return run


def fill_vram(gb: float):
    def run():
        x = torch.empty(int(gb * 2**30 / 4), device="cuda").normal_()
        s = x.std().item()
        if not 0.9 < s < 1.1:
            raise ValueError(f"độ lệch chuẩn {s}, lẽ ra khoảng 1")
        return f"độ lệch chuẩn {s:.3f}"
    return run


def main() -> None:
    print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
    if not torch.cuda.is_available():
        print("Không thấy GPU CUDA.")
        return
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"| VRAM {p.total_memory / 2**30:.1f} GB | compute capability {p.major}.{p.minor}")
    tests = [("nhân ma trận 1000x1000 x50", matmul(1000, 50)),
             ("nhân ma trận 4000x4000 x30", matmul(4000, 30)),
             ("transformer nhỏ (dim 384, giống e5-small) x100", attention(32, 256, 384, 100)),
             ("transformer vừa (dim 768, giống e5-base) x100", attention(16, 256, 768, 100)),
             ("transformer lớn (dim 1024, giống bge-m3) x100", attention(8, 256, 1024, 100)),
             ("chiếm 2.0 GB VRAM", fill_vram(2.0)),
             ("chiếm 3.0 GB VRAM", fill_vram(3.0)),
             ("nhân ma trận 4000x4000 x300 (chạy lâu, GPU nóng lên)", matmul(4000, 300))]
    failed = [name for name, fn in tests if not step(name, fn)]
    print("\nKẾT LUẬN:", "mọi bước đều đạt" if not failed else f"hỏng ở: {failed}")


if __name__ == "__main__":
    main()
