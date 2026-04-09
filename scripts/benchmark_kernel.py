#!/usr/bin/env python3
"""Kernel benchmark with warmup, median timing, and structured output.

Usage:
    python benchmark_kernel.py --kernel path/to/kernel.py --op rmsnorm --baseline torch_compile
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


def load_kernel_module(kernel_path: str):
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_fn(fn, input_tensor, warmup=10, repeats=100):
    """Benchmark a function: warmup + repeats, return median ms."""
    for _ in range(warmup):
        fn(input_tensor)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(input_tensor)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    return median, times


def main():
    parser = argparse.ArgumentParser(description="Kernel benchmark")
    parser.add_argument("--kernel", required=True, help="Path to kernel module")
    parser.add_argument("--op", required=True, help="Operator name")
    parser.add_argument("--baseline", default="torch_compile", help="Baseline to compare")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--shape", default="32,4096", help="Input shape as comma-separated ints")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    x = torch.randn(shape, dtype=dtype, device="cuda")

    module = load_kernel_module(args.kernel)
    kernel_fn = getattr(module, f"kernel_{args.op}", None) or getattr(module, args.op, None)
    ref_fn = getattr(module, f"ref_{args.op}", None)

    if kernel_fn is None:
        print(f"[ERROR] No kernel function for op={args.op}")
        sys.exit(1)

    kernel_ms, _ = benchmark_fn(kernel_fn, x, args.warmup, args.repeats)
    baseline_ms = None
    if ref_fn:
        baseline_ms, _ = benchmark_fn(ref_fn, x, args.warmup, args.repeats)

    speedup = baseline_ms / kernel_ms if baseline_ms else None

    # Estimate bandwidth utilization (rough)
    elem_bytes = x.element_size()
    total_bytes = x.numel() * elem_bytes * 2  # read + write
    bandwidth_gbps = (total_bytes / (kernel_ms / 1000)) / 1e9

    result = {
        "op": args.op,
        "shape": list(shape),
        "dtype": args.dtype,
        "kernel_ms": round(kernel_ms, 4),
        "baseline_ms": round(baseline_ms, 4) if baseline_ms else None,
        "speedup": round(speedup, 3) if speedup else None,
        "bandwidth_gbps": round(bandwidth_gbps, 1),
        "warmup": args.warmup,
        "repeats": args.repeats,
    }

    print(f"=== Benchmark: {args.op} shape={shape} dtype={args.dtype} ===")
    print(f"Kernel:   {kernel_ms:.4f} ms")
    if baseline_ms:
        print(f"Baseline: {baseline_ms:.4f} ms")
        print(f"Speedup:  {speedup:.3f}x")
    print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"=== Result: {'FASTER' if speedup and speedup > 1 else 'SLOWER'} ===")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
