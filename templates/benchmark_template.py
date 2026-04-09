"""Benchmark template — copy and adapt for each kernel.

Usage: python benchmark_{op}.py
"""
import torch
import time


def benchmark(fn, x, warmup=10, repeats=100):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


if __name__ == "__main__":
    shape = (32, 4096)
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device="cuda")

    # --- Baseline ---
    ref_fn = lambda x: x  # Replace with torch reference

    # --- Custom kernel ---
    kernel_fn = lambda x: x  # Replace with custom kernel

    ref_ms = benchmark(ref_fn, x)
    kernel_ms = benchmark(kernel_fn, x)

    print(f"Reference: {ref_ms:.4f} ms")
    print(f"Kernel:    {kernel_ms:.4f} ms")
    print(f"Speedup:   {ref_ms / kernel_ms:.3f}x")
