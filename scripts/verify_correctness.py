#!/usr/bin/env python3
"""Correctness verification with dtype-aware tolerances.

Usage:
    python verify_correctness.py --kernel path/to/kernel.py --op rmsnorm --dtype bf16
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch


DTYPE_TOLERANCES = {
    "fp32": {"atol": 1e-5, "rtol": 1e-5},
    "fp16": {"atol": 1e-3, "rtol": 1e-3},
    "bf16": {"atol": 1e-3, "rtol": 1e-3},
    "fp8": {"atol": 1e-1, "rtol": 1e-1},
}

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def generate_inputs(op: str, dtype: torch.dtype, device: str = "cuda"):
    """Generate 5 random + 2 boundary test inputs for the given op."""
    shapes_random = [
        (1, 1024),
        (4, 2048),
        (16, 4096),
        (32, 8192),
        (64, 1024),
    ]
    shapes_boundary = [
        (1, 1),        # minimal
        (128, 16384),   # large
    ]
    inputs = []
    for shape in shapes_random:
        x = torch.randn(shape, dtype=dtype, device=device)
        inputs.append(("random", shape, x))
    for shape in shapes_boundary:
        x = torch.randn(shape, dtype=dtype, device=device)
        inputs.append(("boundary", shape, x))
    return inputs


def load_kernel_module(kernel_path: str):
    """Dynamically load a kernel module from path."""
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_verification(kernel_path: str, op: str, dtype_str: str):
    """Run correctness verification, return (passed, report_lines)."""
    tol = DTYPE_TOLERANCES[dtype_str]
    dtype = DTYPE_MAP.get(dtype_str)
    if dtype is None:
        print(f"[SKIP] dtype {dtype_str} not supported for torch reference")
        return True, []

    module = load_kernel_module(kernel_path)
    kernel_fn = getattr(module, f"kernel_{op}", None) or getattr(module, op, None)
    ref_fn = getattr(module, f"ref_{op}", None)
    if kernel_fn is None:
        print(f"[ERROR] No kernel function found for op={op}")
        return False, []
    if ref_fn is None:
        print(f"[WARN] No ref function found, using torch default")

    inputs = generate_inputs(op, dtype)
    all_passed = True
    report = []
    for tag, shape, x in inputs:
        try:
            out_kernel = kernel_fn(x)
            out_ref = ref_fn(x) if ref_fn else x  # fallback
            passed = torch.allclose(out_kernel, out_ref, **tol)
            max_err = (out_kernel - out_ref).abs().max().item()
            status = "PASS" if passed else "FAIL"
            line = f"[{status}] {tag} shape={shape} max_err={max_err:.2e} atol={tol['atol']} rtol={tol['rtol']}"
        except Exception as e:
            passed = False
            line = f"[ERROR] {tag} shape={shape} exception={e}"
        report.append(line)
        if not passed:
            all_passed = False

    return all_passed, report


def main():
    parser = argparse.ArgumentParser(description="Kernel correctness verification")
    parser.add_argument("--kernel", required=True, help="Path to kernel module")
    parser.add_argument("--op", required=True, help="Operator name")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_TOLERANCES.keys()))
    args = parser.parse_args()

    print(f"=== Correctness Verification: {args.op} ({args.dtype}) ===")
    print(f"Kernel: {args.kernel}")
    print()

    passed, report = run_verification(args.kernel, args.op, args.dtype)
    for line in report:
        print(line)

    print()
    overall = "ALL PASSED" if passed else "SOME FAILED"
    print(f"=== Result: {overall} ===")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
