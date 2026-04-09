# AMD Kernel Agent Skill Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a layered Agent Skill Pack that enables existing LLMs to optimize AMD GPU kernels across Triton/HIP/CK paths with automated verification.

**Architecture:** Main SKILL.md routes by programming path (Triton/HIP/CK) x target hardware (MI300X/MI355X). Each sub-skill follows a 6-step flow (hardware detection -> analysis -> implementation -> verification -> iteration -> knowledge accumulation). A 5-layer reference knowledge base provides AMD-specific domain knowledge. Three-gate verification (compile -> correctness -> performance) ensures quality.

**Tech Stack:** Python (Triton, PyTorch, benchmarking scripts), C++ (HIP kernels), Markdown (skills, references), ROCm toolchain (hipcc, rocprof, omniperf)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `amd-kernel-skill/` directory tree

- [ ] **Step 1: Create all directories**

```bash
mkdir -p amd-kernel-skill/{skills/{triton-kernel,hip-kernel,ck-kernel},references/{hardware,isa},templates,scripts,agent_output}
```

- [ ] **Step 2: Verify structure**

```bash
find amd-kernel-skill -type d | sort
```

Expected:
```
amd-kernel-skill
amd-kernel-skill/agent_output
amd-kernel-skill/references
amd-kernel-skill/references/hardware
amd-kernel-skill/references/isa
amd-kernel-skill/scripts
amd-kernel-skill/skills
amd-kernel-skill/skills/ck-kernel
amd-kernel-skill/skills/hip-kernel
amd-kernel-skill/skills/triton-kernel
amd-kernel-skill/templates
```

- [ ] **Step 3: Commit**

```bash
cd amd-kernel-skill && git init && cd ..
git add amd-kernel-skill/
git commit -m "chore: scaffold amd-kernel-skill directory structure"
```

---

### Task 2: Main SKILL.md (Router)

**Files:**
- Create: `amd-kernel-skill/SKILL.md`

- [ ] **Step 1: Write the main router skill**

```markdown
# AMD Kernel Optimization Skill

You are an AMD GPU kernel optimization expert. Follow this routing logic to handle kernel optimization tasks.

## Step 0: Hardware Detection

1. If user specifies target hardware, use it
2. Otherwise run: `rocminfo | grep gfx` to detect GPU
3. Load the matching reference:
   - `gfx942` → read `references/hardware/mi300x.md`
   - `gfx950` → read `references/hardware/mi355x.md`
4. Read `references/hardware/hardware-comparison.md` if targeting multiple GPUs

## Step 1: Route by Programming Path

Detect the programming path and read the corresponding sub-skill:

| Signal | Sub-Skill |
|--------|-----------|
| `.py` file, `triton`, `tl.` | Read `skills/triton-kernel/SKILL.md` |
| `.cpp` file, `hip`, `__global__` | Read `skills/hip-kernel/SKILL.md` |
| CK template, `ck::`, `TilePartitioner` | Read `skills/ck-kernel/SKILL.md` |
| User specifies path | Read corresponding sub-skill |
| Unclear | Ask user which path to use |

## Constraints (Always Apply)

1. **Correctness first**: Never sacrifice correctness for performance
2. **Version control**: `git commit` after every successful optimization round
3. **No fallback**: Never revert to `torch.nn.functional` — optimize the custom kernel
4. **Iteration records**: Every round outputs to `agent_output/<OP>/<backend>/round-N/`
5. **Persistence on failure**: At least 3 retry attempts before abandoning a direction
6. **Knowledge accumulation**: Success → design doc; Failure → postmortem; both backfill `references/`

## Available References

When stuck or optimizing, `grep`/`glob` these knowledge base files:
- `references/hardware/` — GPU architecture specs and recommended configs
- `references/isa/` — ISA-level instruction references for deep optimization
- `references/optimization-patterns.md` — Common optimization patterns
- `references/advanced-optimization.md` — Plateau-breaking techniques
- `references/common-mistakes.md` — Known AMD-specific pitfalls
- `references/amd-vs-nvidia-cheatsheet.md` — CUDA→HIP migration reference
```

- [ ] **Step 2: Verify token count is under 400**

```bash
wc -w amd-kernel-skill/SKILL.md
```

Expected: ~250-350 words (well under 400 token target)

- [ ] **Step 3: Commit**

```bash
git add amd-kernel-skill/SKILL.md
git commit -m "feat: add main SKILL.md router for AMD kernel optimization"
```

---

### Task 3: Triton Sub-Skill

**Files:**
- Create: `amd-kernel-skill/skills/triton-kernel/SKILL.md`

- [ ] **Step 1: Write the Triton sub-skill**

```markdown
# Triton Kernel Optimization (AMD ROCm)

## Step 1: Analyze Baseline

```bash
# Run baseline benchmark
python scripts/benchmark_kernel.py --op {OP} --backend torch

# Profile with rocprof
rocprof --stats python run_kernel.py
```

Identify bottleneck type: memory-bound / compute-bound / launch-overhead.

## Step 2: Implement Kernel

Start from `templates/triton_kernel_template.py`. Key rules:

- `BLOCK_SIZE`: use multiples of 64 (wavefront alignment)
- Use `tl.constexpr` for compile-time constants
- `triton.autotune` config space: include `num_warps` in [4, 8, 16] and `num_stages` in [1, 2, 3]
- ROCm backend limitations: no `tl.inline_asm_elementwise`, check `references/triton-rocm-quirks.md`

## Step 3: Verify

```bash
python scripts/verify_correctness.py --kernel {kernel_path} --op {OP} --dtype bf16
python scripts/benchmark_kernel.py --kernel {kernel_path} --op {OP} --baseline torch_compile
```

Three gates: compile → correctness (atol/rtol by dtype) → performance (10 warmup + 100 runs, median).

## Step 4: Iterate

- Each round: save kernel/logs/summary to `agent_output/<OP>/triton/round-N/`
- Update `agent_output/<OP>/triton/performance_trend.md`
- On failure: retry at least 3 times, consult `references/`, only abandon after 3+ rounds with no progress
- On plateau: read `references/advanced-optimization.md`, pick technique by bottleneck type
- Target: exceed `torch.compile` by 5%+

## Step 5: Knowledge Accumulation

- Success → write `final_report.md` (design doc), backfill patterns to `references/`
- Failure → write `final_report.md` (postmortem), backfill lessons to `references/common-mistakes.md`
- This step is mandatory, never skip
```

- [ ] **Step 2: Commit**

```bash
git add amd-kernel-skill/skills/triton-kernel/SKILL.md
git commit -m "feat: add Triton kernel sub-skill"
```

---

### Task 4: HIP C++ Sub-Skill

**Files:**
- Create: `amd-kernel-skill/skills/hip-kernel/SKILL.md`

- [ ] **Step 1: Write the HIP sub-skill**

```markdown
# HIP C++ Kernel Optimization (AMD)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
omniperf analyze -p workload/ --gui  # or --cli for text output
```

Use omniperf to identify micro-architecture bottleneck (memory, compute, occupancy, LDS).

## Step 2: Implement Kernel

Start from `templates/hip_kernel_template.cpp`. Key rules:

- Compile: `hipcc -O3 --offload-arch={arch}` (MI300X: `gfx942`, MI355X: `gfx950`)
- Multi-target: `--offload-arch=gfx942 --offload-arch=gfx950`
- Wavefront = 64 (not 32): affects unroll factors, bank conflict analysis, reduction patterns
- LDS: 64KB/CU — check `references/hardware/` for exact limit per target
- Intrinsics: `__builtin_amdgcn_readfirstlane`, `__builtin_amdgcn_ds_swizzle`, see `references/hip-intrinsics.md`
- Multi-hardware: use `#if __gfx942__` / `#if __gfx950__` for hardware-specific paths
- PyTorch binding: use `torch::utils::hip_extension`

## Step 3: Verify

```bash
python scripts/verify_correctness.py --kernel {kernel_path} --op {OP} --dtype bf16
python scripts/benchmark_kernel.py --kernel {kernel_path} --op {OP} --baseline torch_compile
```

Three gates: compile → correctness → performance.

## Step 4: Iterate

- Each round: save to `agent_output/<OP>/hip/round-N/`
- Update `performance_trend.md`
- On failure: retry 3+ times, consult `references/` (especially `hipcc-compilation.md`, `common-mistakes.md`)
- On plateau: read `references/advanced-optimization.md`
- For ISA-level optimization: check `references/isa/` docs, use `hipcc -save-temps` to inspect generated ISA
- Target: exceed both Triton implementation and `torch.compile`

## Step 5: Knowledge Accumulation

- Success → design doc + backfill to `references/`
- Failure → postmortem + backfill to `references/common-mistakes.md`
- Mandatory step, never skip
```

- [ ] **Step 2: Commit**

```bash
git add amd-kernel-skill/skills/hip-kernel/SKILL.md
git commit -m "feat: add HIP C++ kernel sub-skill"
```

---

### Task 5: CK Sub-Skill

**Files:**
- Create: `amd-kernel-skill/skills/ck-kernel/SKILL.md`

- [ ] **Step 1: Write the CK sub-skill**

```markdown
# Composable Kernel (CK) Optimization (AMD)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
rocprof --stats ./ck_benchmark --op {OP}
```

Understand current CK template configuration and identify tuning opportunities.

## Step 2: Implement / Configure Kernel

Refer to `references/ck-programming-model.md` for the three-level abstraction:
- **TilePartitioner**: How the problem is split across thread blocks
- **TileScheduler**: How tiles are assigned to CUs (static, persistent, dynamic)
- **TilePipeline**: How data flows through shared memory stages

Key tuning parameters: `block_m`, `block_n`, `block_k` — consult `references/ck-tile-tuning.md` for recommended values per hardware.

For new fused kernels: compose from existing CK primitives rather than writing from scratch.

## Step 3: Verify

```bash
python scripts/verify_correctness.py --kernel {kernel_path} --op {OP} --dtype bf16
python scripts/benchmark_kernel.py --kernel {kernel_path} --op {OP} --baseline torch_compile
```

Three gates: compile → correctness → performance.

## Step 4: Iterate

- Each round: save to `agent_output/<OP>/ck/round-N/`
- Update `performance_trend.md`
- Tile search strategy: start with recommended values from `ck-tile-tuning.md`, then grid search around best config
- On failure: retry 3+ times
- On plateau: try different TileScheduler (persistent vs static), adjust pipeline depth
- Target: find optimal tile config, assemble new fused kernels

## Step 5: Knowledge Accumulation

- Success → design doc + backfill optimal configs to `references/ck-tile-tuning.md`
- Failure → postmortem + backfill to `references/common-mistakes.md`
- Mandatory step, never skip
```

- [ ] **Step 2: Commit**

```bash
git add amd-kernel-skill/skills/ck-kernel/SKILL.md
git commit -m "feat: add Composable Kernel sub-skill"
```

---

### Task 6: Verification Scripts

**Files:**
- Create: `amd-kernel-skill/scripts/verify_correctness.py`
- Create: `amd-kernel-skill/scripts/benchmark_kernel.py`

- [ ] **Step 1: Write verify_correctness.py**

```python
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
```

- [ ] **Step 2: Write benchmark_kernel.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add amd-kernel-skill/scripts/
git commit -m "feat: add verification and benchmark scripts"
```

---

### Task 7: Templates

**Files:**
- Create: `amd-kernel-skill/templates/triton_kernel_template.py`
- Create: `amd-kernel-skill/templates/hip_kernel_template.cpp`
- Create: `amd-kernel-skill/templates/benchmark_template.py`

- [ ] **Step 1: Write triton_kernel_template.py**

```python
"""Triton kernel template for AMD ROCm.

Replace {OP} with the operator name. Adjust BLOCK_SIZE configs for target hardware.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def kernel_op_fwd(
    X_ptr,
    Y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)

    # === Kernel logic here ===
    y = x

    tl.store(Y_ptr + offsets, y, mask=mask)


def kernel_op(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for the Triton kernel."""
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    kernel_op_fwd[grid](x, y, N)
    return y


def ref_op(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return x  # Replace with actual reference
```

- [ ] **Step 2: Write hip_kernel_template.cpp**

```cpp
/*
 * HIP C++ kernel template for AMD GPUs.
 * Compile: hipcc -O3 --offload-arch=gfx942 -shared -fPIC -o kernel.so kernel.cpp \
 *          $(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch/TorchConfig.cmake
 *
 * For multi-target: --offload-arch=gfx942 --offload-arch=gfx950
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>

// === Kernel ===

template <typename T, int BLOCK_SIZE = 256>
__global__ void kernel_op_fwd(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int N
) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= N) return;

    // Wavefront = 64 on AMD (not 32)
    // const int lane_id = threadIdx.x % 64;
    // const int wf_id = threadIdx.x / 64;

    T val = input[idx];

    // === Kernel logic here ===

    output[idx] = val;
}

// === PyTorch binding ===

torch::Tensor kernel_op(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int N = input.numel();
    constexpr int BLOCK_SIZE = 256;
    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "kernel_op", [&] {
            kernel_op_fwd<scalar_t, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N
            );
        }
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_op", &kernel_op, "Custom HIP kernel");
}
```

- [ ] **Step 3: Write benchmark_template.py**

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add amd-kernel-skill/templates/
git commit -m "feat: add Triton, HIP, and benchmark templates"
```

---

### Task 8: Hardware Reference Documents

**Files:**
- Create: `amd-kernel-skill/references/hardware/mi300x.md`
- Create: `amd-kernel-skill/references/hardware/mi355x.md`
- Create: `amd-kernel-skill/references/hardware/hardware-comparison.md`

- [ ] **Step 1: Write mi300x.md**

```markdown
# MI300X (GFX942 / CDNA3) Optimization Guide

## Core Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | CDNA3 |
| offload-arch | `gfx942` |
| Compute Units | 192 |
| HBM | 192 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| Wavefront Size | 64 |
| LDS per CU | 64 KB |
| VGPR per CU | 65536 (512 per wavefront at full occupancy) |
| SGPR per CU | 3200 |
| Matrix Core | MFMA (FP16/BF16/FP8/INT8 → FP32) |
| L2 Cache | 256 MB |
| Max Threads/CU | 2048 (32 wavefronts) |

## Compilation

```bash
# HIP
hipcc -O3 --offload-arch=gfx942 kernel.cpp

# Triton: auto-detected, no flag needed

# CK: use CMake with AMDGPU_TARGETS=gfx942
```

## Recommended Configurations

| Path | Parameter | Recommended |
|------|-----------|-------------|
| Triton | BLOCK_SIZE | 128 or 256 |
| Triton | num_warps | 4-8 |
| HIP | Thread block | 256 threads (4 wavefronts) |
| CK (BF16 GEMM) | block_m, block_n, block_k | 256, 128, 64 |

## Performance Thresholds

| Metric | Good | Needs Work |
|--------|------|------------|
| Occupancy | > 50% | < 25% |
| HBM Bandwidth Utilization | > 60% (mem-bound) | < 30% |
| MFMA Utilization | > 70% (compute-bound) | < 40% |
| LDS Bank Conflicts | < 5% | > 20% |
| Register Spilling | 0 scratch waves | Any scratch |

## Hardware-Specific Tips

- 192 CUs = high parallelism. Small kernels may underutilize. Consider persistent kernels.
- 64-wide wavefront: reduction requires 6 shuffle steps (not 5 like NVIDIA warp=32)
- 256 MB L2: larger than NVIDIA counterparts. Tile traversal order matters for L2 reuse.
- HBM3 5.3 TB/s: memory-bound kernels can achieve high throughput if access is coalesced.

## Known Issues

- `hipcc` may silently fall back to slower instruction patterns if `-O3` is omitted
- `tl.inline_asm_elementwise` not available in Triton ROCm backend
- Different grid launch limits vs NVIDIA — check `hipDeviceGetAttribute`
```

- [ ] **Step 2: Write mi355x.md**

```markdown
# MI355X (GFX950 / CDNA4) Optimization Guide

## Core Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | CDNA4 |
| offload-arch | `gfx950` |
| Compute Units | TBD (check `rocminfo`) |
| HBM | 288 GB HBM3E |
| Memory Bandwidth | TBD |
| Wavefront Size | 64 |
| LDS per CU | TBD (check `rocminfo`) |
| Matrix Core | MFMA (CDNA4 — new instruction variants) |

## Compilation

```bash
hipcc -O3 --offload-arch=gfx950 kernel.cpp
```

## Differences from MI300X (GFX942)

| Aspect | MI300X (gfx942) | MI355X (gfx950) |
|--------|----------------|-----------------|
| HBM | 192 GB HBM3 | 288 GB HBM3E |
| New MFMA variants | — | Check ISA docs for CDNA4 additions |
| offload-arch | gfx942 | gfx950 |

## Porting from MI300X

1. Change `--offload-arch=gfx942` to `--offload-arch=gfx950`
2. Re-run tile size / block size tuning (different CU count, bandwidth)
3. Check for new MFMA instructions that may improve compute throughput
4. Re-validate performance thresholds (absolute numbers will differ)
5. Update `performance_trend.md` to note hardware change

## Known Issues

- Document issues as they are discovered during iteration
- Backfill this section from `agent_output/` postmortems
```

- [ ] **Step 3: Write hardware-comparison.md**

```markdown
# AMD GPU Hardware Comparison

## Quick Reference

| Parameter | MI300X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) |
|-----------|----------------------|----------------------|
| Compute Units | 192 | TBD |
| HBM | 192 GB HBM3 | 288 GB HBM3E |
| Memory BW | 5.3 TB/s | TBD |
| Wavefront | 64 | 64 |
| LDS/CU | 64 KB | TBD |
| Matrix Core | MFMA (CDNA3) | MFMA (CDNA4) |
| offload-arch | gfx942 | gfx950 |

## Cross-Hardware Porting Checklist

1. **Compilation**: Update `--offload-arch` flag
2. **Tile/Block size**: Re-tune. Optimal values depend on CU count, BW, LDS size
3. **New instructions**: Check if CDNA4 adds MFMA variants or memory instructions
4. **Performance baselines**: Re-benchmark. Absolute numbers change across hardware
5. **Conditional compilation**: Use `#if __gfx942__` / `#if __gfx950__` for hardware-specific paths
6. **Profiling thresholds**: Adjust "good" occupancy / bandwidth utilization targets

## When to Multi-Target

If code must run on both MI300X and MI355X:
```bash
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp
```
The compiler generates code for both targets. Use runtime dispatch or conditional compilation for hardware-specific optimizations.
```

- [ ] **Step 4: Commit**

```bash
git add amd-kernel-skill/references/hardware/
git commit -m "feat: add hardware reference documents (MI300X, MI355X, comparison)"
```

---

### Task 9: ISA Reference Documents

**Files:**
- Create: `amd-kernel-skill/references/isa/isa-overview.md`
- Create: `amd-kernel-skill/references/isa/mfma-instructions.md`
- Create: `amd-kernel-skill/references/isa/memory-instructions.md`
- Create: `amd-kernel-skill/references/isa/valu-salu-instructions.md`
- Create: `amd-kernel-skill/references/isa/register-allocation.md`
- Create: `amd-kernel-skill/references/isa/scheduling-pipeline.md`
- Create: `amd-kernel-skill/references/isa/inline-asm-patterns.md`

- [ ] **Step 1: Write isa-overview.md**

```markdown
# AMDGPU ISA Quick Reference

Entry point for ISA-level optimization. Read this first, then dive into specific docs as needed.

## Register Files

| Type | Count/CU | Per Wavefront (full occ) | Usage |
|------|----------|--------------------------|-------|
| VGPR | 65536 | 512 (at 8 wf/SIMD) | Per-lane data, MFMA operands |
| SGPR | 3200 | 102 | Uniform values, addresses, control |
| AGPR | 65536 | 512 | MFMA accumulators (CDNA) |

## Instruction Categories

| Category | Unit | Examples | Latency |
|----------|------|----------|---------|
| VALU | Vector ALU | v_add_f32, v_fma_f32 | 4-8 cycles |
| SALU | Scalar ALU | s_add_u32, s_cmp_eq | 2-4 cycles |
| MFMA | Matrix Core | v_mfma_f32_16x16x16_f16 | 8-64 cycles |
| VMEM | Vector Memory | global_load_dwordx4 | 300-500 cycles |
| LDS | Local Data Share | ds_read_b128, ds_write_b64 | 20-40 cycles |
| SMEM | Scalar Memory | s_load_dwordx4 | 200+ cycles |

## Pipeline Model (Simplified)

- 4 SIMD units per CU, each executes one wavefront per cycle
- VALU: 1 instruction/cycle/SIMD (4 cycles for 64 lanes)
- MFMA: issued to matrix core, long latency but high throughput when pipelined
- Memory: issued to memory unit, returns asynchronously → use s_waitcnt to synchronize

## When to Use ISA Knowledge

| Optimization Phase | ISA Depth | Documents |
|-------------------|-----------|-----------|
| Early (fusion, tiling) | Low | This file only |
| Mid (occupancy, memory) | Medium | memory-instructions.md, register-allocation.md |
| Late (scheduling, bubbles) | High | scheduling-pipeline.md, inline-asm-patterns.md |
| MFMA-heavy (GEMM, Attention) | High | mfma-instructions.md |
```

- [ ] **Step 2: Write mfma-instructions.md**

```markdown
# MFMA (Matrix Fused Multiply-Add) Instructions

## Instruction Format

`v_mfma_{out_type}_{M}x{N}x{K}_{in_type}` — Multiplies M×K by K×N matrices, accumulates into M×N.

## Available Instructions (CDNA3 / GFX942)

| Instruction | In Type | Out Type | M×N×K | VGPR In | VGPR Out | Cycles |
|-------------|---------|----------|-------|---------|----------|--------|
| v_mfma_f32_16x16x16_f16 | FP16 | FP32 | 16×16×16 | 4+4 | 4 | 64 |
| v_mfma_f32_16x16x16_bf16 | BF16 | FP32 | 16×16×16 | 4+4 | 4 | 64 |
| v_mfma_f32_32x32x8_f16 | FP16 | FP32 | 32×32×8 | 4+4 | 16 | 64 |
| v_mfma_f32_32x32x8_bf16 | BF16 | FP32 | 32×32×8 | 4+4 | 16 | 64 |
| v_mfma_f32_16x16x32_fp8 | FP8 | FP32 | 16×16×32 | 4+4 | 4 | 64 |
| v_mfma_i32_16x16x32_i8 | INT8 | INT32 | 16×16×32 | 4+4 | 4 | 64 |

## Register Layout

MFMA output is distributed across VGPR lanes in a specific pattern:
- For 16x16: each lane holds one element, 4 VGPRs hold the 4×16×16/64 result
- For 32x32: 16 VGPRs hold the result, distributed across 64 lanes

Key: the lane-to-matrix-element mapping differs from NVIDIA's MMA. Don't assume CUDA layouts.

## vs NVIDIA Tensor Core

| Aspect | AMD MFMA | NVIDIA MMA |
|--------|----------|------------|
| Execution unit | Per-wavefront (64 lanes) | Per-warp (32 lanes) |
| Accumulator | AGPR (dedicated) or VGPR | Shared with VGPR |
| Largest tile | 32×32 | 16×16 (Ampere), 64×64 (Hopper) |
| Scheduling | Must manually interleave with VMEM | TMA handles async |

## Optimization Tips

1. **Pipeline MFMA with loads**: while one MFMA executes (64 cycles), issue global_load for next tile
2. **Use AGPR for accumulators**: frees VGPR for data, reduces register pressure
3. **Prefer larger tiles**: 32×32×8 has better compute-to-register ratio than 16×16×16
4. **Check CDNA4 additions**: MI355X may have new MFMA variants — see `hardware/mi355x.md`
```

- [ ] **Step 3: Write memory-instructions.md**

```markdown
# Memory Instructions Reference

## Global Memory

| Instruction | Width | Notes |
|-------------|-------|-------|
| global_load_dword | 4B | Single lane |
| global_load_dwordx2 | 8B | Vectorized |
| global_load_dwordx4 | 16B | Best for coalesced access |
| global_store_dword[x2/x4] | 4-16B | Same widths |

**Coalescing rule**: 64 consecutive lanes accessing 64 consecutive elements = one coalesced transaction. Stride > 1 degrades bandwidth.

## LDS (Local Data Share)

| Instruction | Width | Notes |
|-------------|-------|-------|
| ds_read_b32 | 4B | Single bank |
| ds_read_b64 | 8B | Two banks |
| ds_read_b128 | 16B | Four banks |
| ds_write_b32/b64/b128 | 4-16B | Same widths |
| ds_swizzle_b32 | 4B | Lane permutation without LDS r/w |

**Bank conflict rule**: 32 banks × 4B each. Two lanes accessing same bank (different address) = conflict = serialization. Padding or swizzle to resolve.

## Buffer vs Flat Instructions

| Type | When to Use |
|------|------------|
| `buffer_load_*` | Known base + offset, enables range checking, slightly faster |
| `global_load_*` (flat) | Arbitrary 64-bit address, simpler codegen |

Compiler usually chooses. For inline asm, prefer buffer ops when base address is uniform (SGPR).

## Memory Fence & Synchronization

### s_waitcnt — The Critical Instruction

```
s_waitcnt vmcnt(N) lgkmcnt(M) expcnt(K)
```

| Counter | Tracks | Wait for |
|---------|--------|----------|
| vmcnt | Global load/store | N outstanding VMEM ops remaining |
| lgkmcnt | LDS + SMEM ops | M outstanding LDS/SMEM ops remaining |
| expcnt | Export (GDS, LDS→VGPR) | K outstanding exports |

**Strategy**: Don't use `s_waitcnt 0` everywhere (kills ILP). Count outstanding ops and wait for exactly what you need.

```
global_load_dwordx4 v[0:3], ...    // vmcnt = 1
global_load_dwordx4 v[4:7], ...    // vmcnt = 2
// ... do other work ...
s_waitcnt vmcnt(1)                  // wait for first load only
v_add_f32 v8, v0, v1               // use first load results
s_waitcnt vmcnt(0)                  // now wait for second load
v_add_f32 v9, v4, v5               // use second load results
```
```

- [ ] **Step 4: Write valu-salu-instructions.md**

```markdown
# Vector ALU (VALU) & Scalar ALU (SALU) Reference

## VALU — Per-Lane Computation

Operates on all 64 lanes of a wavefront in 4 cycles (16 lanes/cycle).

| Category | Examples | Throughput |
|----------|----------|------------|
| FP32 Arithmetic | v_add_f32, v_mul_f32, v_fma_f32 | 1/cycle/SIMD |
| FP16 Packed | v_pk_add_f16, v_pk_mul_f16 | 1/cycle (2 FP16/lane) |
| Type Convert | v_cvt_f32_f16, v_cvt_f16_f32 | 1/cycle |
| Compare | v_cmp_gt_f32 → writes to VCC | 1/cycle |
| Bit Ops | v_bfe_u32, v_bfi_b32 | 1/cycle |
| Transcendental | v_rcp_f32, v_rsq_f32, v_exp_f32 | 1/4 cycle (shared unit) |

**Key**: Transcendentals are 4x slower. Avoid in inner loops. Use polynomial approximations when possible.

## SALU — Uniform Computation

Operates on a single scalar value, shared across the wavefront.

| Category | Examples | Throughput |
|----------|----------|------------|
| Arithmetic | s_add_u32, s_mul_i32 | 1/cycle |
| Logic | s_and_b64, s_or_b64 | 1/cycle |
| Compare | s_cmp_eq_u32 | 1/cycle |
| Branch | s_cbranch_scc1 | Variable |
| Constant load | s_load_dwordx4 | ~200 cycles |

**Key**: Move uniform computations (loop counters, addresses) to SALU to free VALU for data work.

## Dual Issue

CDNA3 can dual-issue VALU + SALU in the same cycle if:
- No data dependency between the two
- Different register files (VGPR vs SGPR)

Optimization: interleave SALU address computation with VALU data computation.
```

- [ ] **Step 5: Write register-allocation.md**

```markdown
# Register Allocation & Occupancy Guide

## VGPR Budget vs Occupancy (MI300X, per SIMD unit)

| Max VGPR/Wavefront | Max Wavefronts/SIMD | Occupancy |
|---------------------|---------------------|-----------|
| 128 | 8 | 100% |
| 256 | 4 | 50% |
| 512 | 2 | 25% |
| 1024 | 1 | 12.5% |

Formula: `wavefronts_per_simd = floor(16384 / vgpr_per_wavefront)`, capped at 8.

## SGPR Budget

| Max SGPR/Wavefront | Max Wavefronts/SIMD |
|---------------------|---------------------|
| ≤ 102 | 8 |
| > 102 | Spills to memory |

SGPRs rarely limit occupancy. Focus on VGPR management.

## AGPR (Accumulation GPR)

- Dedicated to MFMA accumulator results
- Same file size as VGPR (65536/CU)
- Using AGPR for accumulators frees VGPRs for data → higher occupancy
- Move between VGPR and AGPR: `v_accvgpr_read` / `v_accvgpr_write`

## Detecting Register Pressure

```bash
# Check VGPR/SGPR usage
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# Look for .s file, search for .vgpr_count and .sgpr_count

# In omniperf output
# ScratchWaveslifetimeVGPR > 0 means register spilling!
```

## Reducing Register Pressure

| Technique | How | Impact |
|-----------|-----|--------|
| Reduce live ranges | Compute and consume values close together | Moderate |
| Use AGPR for accumulators | `v_accvgpr_write` | Frees VGPRs |
| `__launch_bounds__(threads, minBlocks)` | Hints compiler on register budget | Direct |
| Manual register reuse | Rewrite loops to reuse registers | High effort |
| Accept lower occupancy | If ILP hides latency, fewer waves is OK | Trade-off |

## The Occupancy vs ILP Trade-off

Low occupancy is NOT always bad. If your kernel has enough ILP (independent instructions between memory ops), fewer wavefronts with more registers can outperform many wavefronts with spilling.

Decision flow:
1. Check for spilling (ScratchWaves > 0)
2. If spilling: reduce VGPR usage or accept lower occupancy
3. If not spilling but low perf: try INCREASING occupancy via fewer VGPRs
4. Profile both configurations. The faster one wins.
```

- [ ] **Step 6: Write scheduling-pipeline.md**

```markdown
# Instruction Scheduling & Pipeline Guide

## CDNA3 Pipeline Model

Each CU has 4 SIMD units. Each SIMD:
- Executes one wavefront instruction per cycle
- Round-robins across ready wavefronts (TLP hides latency)

Latency hiding: if wavefront A stalls on memory, SIMD executes wavefront B, C, D, etc.

## Instruction-Level Parallelism (ILP)

When occupancy is low (few wavefronts), ILP within a single wavefront becomes critical.

**Goal**: keep the pipeline fed by interleaving independent instructions.

```
// BAD: dependent chain → pipeline stalls
global_load v0, ...
s_waitcnt vmcnt(0)    // stall here
v_add_f32 v1, v0, v2  // must wait for load

// GOOD: interleave independent work
global_load v0, ...    // issue load
v_mul_f32 v3, v4, v5   // independent VALU work
s_add_u32 s0, s0, 1    // independent SALU work (dual issue!)
s_waitcnt vmcnt(0)     // load likely done by now
v_add_f32 v1, v0, v2   // use loaded data
```

## s_waitcnt Strategy

**Principle**: wait as late as possible, for as few ops as possible.

| Pattern | Code | Why |
|---------|------|-----|
| Immediate wait | `s_waitcnt vmcnt(0)` after load | Bad: kills ILP |
| Deferred wait | Load, do other work, then wait | Good: hides latency |
| Partial wait | `vmcnt(N)` where N = remaining ops ok to be pending | Best: minimal stall |

## Double Issue Rules (CDNA3)

VALU + SALU can issue in same cycle if:
1. No register dependency between them
2. VALU uses VGPRs, SALU uses SGPRs
3. Both are ready (no pending waits)

**Optimization**: pair address calculations (SALU) with data operations (VALU).

## MFMA Scheduling

MFMA instructions have high latency (64 cycles) but can overlap with other work:

```
// Pipeline: while MFMA N executes, load data for N+1
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 64 cycles
global_load_dwordx4 v[0:3], ...  // issue during MFMA latency
global_load_dwordx4 v[4:7], ...  // issue during MFMA latency
s_waitcnt vmcnt(0)               // loads should be done by now
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // next MFMA
```
```

- [ ] **Step 7: Write inline-asm-patterns.md**

```markdown
# HIP Inline Assembly Patterns

## When to Use Inline ASM

1. Compiler fails to generate optimal instruction (verify with `-save-temps`)
2. Need specific instruction not exposed via intrinsics
3. Last resort for critical inner loops — prefer `__builtin_amdgcn_*` first

## Syntax

```cpp
asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
```

Constraint codes:
- `v` = VGPR, `s` = SGPR, `a` = AGPR
- `=` = output, no prefix = input

## Builtin → ISA Mapping

| Builtin | ISA Instruction | Purpose |
|---------|----------------|---------|
| `__builtin_amdgcn_readfirstlane(v)` | v_readfirstlane_b32 | Broadcast first lane to SGPR |
| `__builtin_amdgcn_ds_swizzle(v, pat)` | ds_swizzle_b32 | Lane permutation (no LDS traffic) |
| `__builtin_amdgcn_mov_dpp(v, ctrl, ...)` | v_mov_b32 dpp | Data parallel primitive |
| `__shfl_sync` equivalent | ds_swizzle / dpp | Cross-lane communication |

## Common Optimization Patterns

### 1. Vectorized Global Load

```cpp
// Force 128-bit load (4 floats at once)
float4 data;
asm volatile(
    "global_load_dwordx4 %0, %1, off"
    : "=v"(data) : "v"(addr)
);
```

### 2. LDS Swizzle for Bank Conflict Elimination

```cpp
// Swizzle pattern to avoid bank conflicts during matrix transpose
int swizzled = __builtin_amdgcn_ds_swizzle(val, 0x041f);
```

### 3. Manual MFMA in Inner Loop

```cpp
// When compiler doesn't schedule MFMA optimally
asm volatile(
    "v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
    : "+a"(acc)  // AGPR accumulator (read-write)
    : "v"(a_frag), "v"(b_frag)
);
```

### 4. Precise s_waitcnt

```cpp
// Wait for exactly 1 outstanding VMEM op
asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
```

## Decision Guide

| Situation | Approach |
|-----------|----------|
| Need cross-lane ops | Try `__builtin_amdgcn_*` first |
| Compiler emits suboptimal loads | Check if `-O3` fixes it, then try asm |
| MFMA scheduling is wrong | Profile first, then manual asm if needed |
| Need precise wait counts | Inline `s_waitcnt` with exact counts |
```

- [ ] **Step 8: Commit**

```bash
git add amd-kernel-skill/references/isa/
git commit -m "feat: add ISA reference documents (7 files)"
```

---

### Task 10: Toolchain Reference Documents

**Files:**
- Create: `amd-kernel-skill/references/rocprof-guide.md`
- Create: `amd-kernel-skill/references/omniperf-guide.md`
- Create: `amd-kernel-skill/references/hipcc-compilation.md`
- Create: `amd-kernel-skill/references/triton-rocm-quirks.md`

- [ ] **Step 1: Write rocprof-guide.md**

```markdown
# rocprof Usage Guide

## Basic Profiling

```bash
# Kernel execution stats (time, calls, occupancy)
rocprof --stats python run_kernel.py

# Hardware counters
rocprof -i counters.txt python run_kernel.py

# Timestamp trace
rocprof --timestamp on python run_kernel.py
```

## Key Counters

Create `counters.txt`:
```
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM SQ_INSTS_LDS
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum
pmc: SQ_INSTS_MFMA TA_FLAT_READ_WAVEFRONTS_sum
```

## Output Interpretation

| Counter | Meaning | Action |
|---------|---------|--------|
| SQ_WAVES | Total wavefronts launched | Sanity check grid size |
| SQ_INSTS_VALU | Vector ALU instructions | High = compute-bound |
| SQ_INSTS_LDS | LDS operations | High = check for bank conflicts |
| SQ_INSTS_MFMA | Matrix core ops | Should be high for GEMM kernels |
| TCC_HIT_sum / TCC_MISS_sum | L2 cache hit/miss | Low hit rate = poor data reuse |

## Common Workflow

1. `rocprof --stats` → identify slowest kernel
2. `rocprof -i counters.txt` → categorize bottleneck (compute/memory/LDS)
3. Optimize based on bottleneck
4. Re-profile to verify improvement
```

- [ ] **Step 2: Write omniperf-guide.md**

```markdown
# omniperf Profiling Guide

## Workflow

```bash
# 1. Collect data
omniperf profile -n workload_name -- python run_kernel.py

# 2. Analyze (CLI)
omniperf analyze -p workload_name/ --cli

# 3. Analyze (GUI — if available)
omniperf analyze -p workload_name/ --gui
```

## Key Metrics

| Panel | Metric | Healthy | Problem |
|-------|--------|---------|---------|
| Speed-of-Light | GPU Utilization | > 60% | < 30% |
| Speed-of-Light | Memory BW Utilization | > 60% (mem-bound) | < 30% |
| Compute | VALU Utilization | Depends | < 20% (wasted compute) |
| Compute | MFMA Utilization | > 70% (GEMM) | < 40% |
| Memory | L2 Hit Rate | > 80% | < 50% |
| Memory | LDS Bank Conflicts | < 5% | > 20% |
| Occupancy | Achieved Occupancy | > 50% | < 25% |
| Occupancy | ScratchWaveslifetimeVGPR | 0 | > 0 = register spilling! |

## Bottleneck Decision Tree

```
Is GPU Utilization low?
├── Yes → Kernel launch overhead or insufficient parallelism
│         Try: larger problem sizes, persistent kernels
└── No → Check Memory BW Utilization
    ├── High (>60%) → Memory-bound
    │   Try: coalescing, vectorized loads, L2 optimization
    └── Low → Check MFMA/VALU Utilization
        ├── High → Compute-bound
        │   Try: mixed precision, algorithmic improvements
        └── Low → Likely stalled on dependencies
            Try: ILP, software pipelining, s_waitcnt tuning
```
```

- [ ] **Step 3: Write hipcc-compilation.md**

```markdown
# hipcc Compilation Guide

## Basic Commands

```bash
# Single target
hipcc -O3 --offload-arch=gfx942 -o kernel.so -shared -fPIC kernel.cpp

# Multi-target
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 -o kernel.so -shared -fPIC kernel.cpp

# With PyTorch
hipcc -O3 --offload-arch=gfx942 \
  $(python3 -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")  \
  -shared -fPIC kernel.cpp -o kernel.so

# Save intermediate files (for ISA inspection)
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# Look for *.s files containing ISA
```

## Common Flags

| Flag | Purpose |
|------|---------|
| `-O3` | Full optimization (always use) |
| `--offload-arch=gfxNNN` | Target GPU architecture (required) |
| `-shared -fPIC` | Build shared library for Python loading |
| `-save-temps` | Keep intermediate .s (ISA) files |
| `-Rpass=inline` | Show inlining decisions |
| `-ffast-math` | Aggressive FP optimizations (may affect precision) |
| `-munsafe-fp-atomics` | Faster atomic FP ops (may lose precision in rare cases) |

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `error: unknown target CPU 'gfx942'` | Old ROCm version | Update ROCm or check `rocminfo` for correct arch |
| `undefined reference to __hip_*` | Missing HIP runtime link | Add `-lhip_hcc` or use `hipcc` instead of `g++` |
| `error: use of undeclared identifier '__shfl_sync'` | CUDA API not available in HIP | Use `__builtin_amdgcn_ds_swizzle` or `__shfl` |
| `error: too few register available` | Too many VGPRs | Add `__launch_bounds__`, reduce live variables |
| Kernel runs but wrong results | `--offload-arch` mismatch | Verify arch matches `rocminfo` output |
| Slow performance, no `-O3` | Debug build | Always compile with `-O3` |

## PyTorch Extension Build

```python
from torch.utils.hip_extension import load

module = load(
    name="custom_kernel",
    sources=["kernel.cpp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--offload-arch=gfx942"],
)
```
```

- [ ] **Step 4: Write triton-rocm-quirks.md**

```markdown
# Triton ROCm Backend — Known Differences & Quirks

## Unsupported Features

| Feature | Status | Workaround |
|---------|--------|------------|
| `tl.inline_asm_elementwise` | Not supported | Use pure Triton ops |
| Some `tl.extra.cuda` functions | Not available | Check availability before using |
| `tl.tensor` with certain layouts | May differ | Test explicitly |

## Autotune Differences

- `num_stages`: ROCm backend may ignore or handle differently than CUDA
- Config space should include `num_stages=1` as fallback
- `num_warps`: on AMD, warps = wavefronts (64 threads each, not 32)

## Block Size Guidelines

- Use multiples of 64 (wavefront size alignment)
- Typical good values: 64, 128, 256, 512, 1024
- 128 and 256 are usually the sweet spot for MI300X

## Performance Considerations

- Triton ROCm may generate different code quality than CUDA backend
- Always profile with `rocprof --stats` to verify actual performance
- If Triton performance plateaus, consider switching to HIP C++ for more control

## Debugging

```bash
# Set debug flags
TRITON_DEBUG=1 python kernel.py

# Check generated code
TRITON_CACHE_DIR=/tmp/triton_cache python kernel.py
# Inspect /tmp/triton_cache/ for generated IR and ASM
```

## Common Pitfalls

1. Assuming CUDA Triton performance translates to ROCm — always re-benchmark
2. Using `tl.constexpr` incorrectly — must be used for compile-time constants only
3. Not testing with `num_stages=1` — sometimes fewer stages is faster on AMD
4. Grid size too small for 192 CUs — need enough blocks to saturate MI300X
```

- [ ] **Step 5: Commit**

```bash
git add amd-kernel-skill/references/rocprof-guide.md
git add amd-kernel-skill/references/omniperf-guide.md
git add amd-kernel-skill/references/hipcc-compilation.md
git add amd-kernel-skill/references/triton-rocm-quirks.md
git commit -m "feat: add toolchain reference documents (rocprof, omniperf, hipcc, triton-rocm)"
```

---

### Task 11: Library & API Reference Documents

**Files:**
- Create: `amd-kernel-skill/references/ck-programming-model.md`
- Create: `amd-kernel-skill/references/ck-tile-tuning.md`
- Create: `amd-kernel-skill/references/aiter-ops-reference.md`
- Create: `amd-kernel-skill/references/hip-intrinsics.md`

- [ ] **Step 1: Write ck-programming-model.md**

```markdown
# Composable Kernel (CK) Programming Model

## Three-Level Abstraction

### Level 1: TilePartitioner
Splits the problem across thread blocks (workgroups).

- Divides M, N dimensions into block tiles
- Each thread block owns one (block_m × block_n) output tile
- Grid size = (M/block_m) × (N/block_n)

### Level 2: TileScheduler
Assigns tiles to CUs and manages execution order.

| Scheduler | Behavior | Best For |
|-----------|----------|----------|
| Static | Fixed tile-to-CU mapping | Simple kernels |
| Persistent | Single launch, tiles via atomicAdd counter | Small tiles, many iterations |
| Dynamic | Work-stealing between CUs | Irregular workloads (causal mask) |

### Level 3: TilePipeline
Manages data flow within a thread block: global → LDS → register → compute → writeback.

- Controls prefetch depth (single buffer, double buffer, triple buffer)
- Manages LDS allocation for A and B fragments
- Schedules MFMA instructions vs memory operations

## Using CK Templates

```cpp
// 1. Define the problem
using Problem = ck::tensor_operation::device::DeviceGemmXdl<...>;

// 2. Configure tile sizes
using Config = ck::TileConfig<
    BlockM, BlockN, BlockK,
    WarpM, WarpN, WarpK,
    PipelineDepth
>;

// 3. Instantiate and run
auto gemm = Problem{};
gemm.Run(args, stream);
```

## Extending CK

For new fused kernels:
1. Identify existing CK primitives that match sub-operations
2. Compose them in a new pipeline
3. Handle fusion points (where outputs feed inputs without going to global memory)
4. Tune tile sizes for the fused kernel's specific data flow
```

- [ ] **Step 2: Write ck-tile-tuning.md**

```markdown
# CK Tile Size Tuning Guide

## Parameters

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| block_m | M-dimension tile per block | 64-256 |
| block_n | N-dimension tile per block | 64-256 |
| block_k | K-dimension tile (reduction) | 32-128 |
| warp_m | M-dimension per warp | 16-64 |
| warp_n | N-dimension per warp | 16-64 |
| pipeline_depth | Prefetch stages | 1-3 |

## MI300X (GFX942) Recommended Configs

### BF16 GEMM

| M Range | N Range | block_m | block_n | block_k | Notes |
|---------|---------|---------|---------|---------|-------|
| ≥ 2048 | ≥ 2048 | 256 | 128 | 64 | Standard large GEMM |
| ≥ 2048 | ≥ 2048 | 128 | 256 | 64 | Alternative (try both) |
| < 1024 | ≥ 2048 | 64 | 256 | 64 | Tall-skinny |
| ≥ 2048 | < 1024 | 256 | 64 | 64 | Wide-short |
| < 512 | < 512 | 64 | 64 | 32 | Small GEMM |

### FP8 GEMM

| Config | block_m | block_n | block_k |
|--------|---------|---------|---------|
| Default | 256 | 128 | 128 |
| Alternative | 128 | 256 | 128 |

### Attention (FMHA)

| Config | block_m | block_n | head_dim |
|--------|---------|---------|----------|
| Default | 128 | 128 | 64-128 |
| Long seq | 64 | 256 | 64-128 |

## MI355X (GFX950) Recommended Configs

TBD — fill in after initial benchmarking on MI355X hardware.
Expect different optimal tile sizes due to different CU count and bandwidth.

## Tuning Strategy

1. Start with recommended config from table above
2. Run benchmark, record performance
3. Grid search: vary block_m and block_n by ±64, block_k by ±32
4. For each config, check:
   - LDS usage ≤ 64KB (block_m × block_k + block_k × block_n) × elem_size × pipeline_depth
   - Enough blocks to fill all CUs (grid_size ≥ 192 for MI300X)
   - VGPR usage doesn't cause spilling
5. Record best config in this file for future reference
```

- [ ] **Step 3: Write aiter-ops-reference.md**

```markdown
# AITER (AMD Inference Toolkit for Efficient Runtime) Reference

## Overview

AITER provides pre-optimized AMD GPU kernels for common inference operations.
Before writing a custom kernel, check if AITER already provides a high-performance version.

## Available Operations

| Category | Operations | Notes |
|----------|-----------|-------|
| Attention | Flash Attention (fwd/bwd), PagedAttention | CK-based, highly tuned |
| GEMM | BF16/FP16/FP8/INT8 GEMM variants | Multiple tile configs |
| Normalization | RMSNorm, LayerNorm | Fused versions |
| Activation | SiLU, GeLU, SwiGLU | Fused with linear |
| Quantization | FP8 quantize/dequantize | Per-tensor and per-token |
| MoE | MoE GEMM, expert routing | Optimized for MI300X |

## API Usage

```python
import aiter

# Example: Flash Attention
output = aiter.flash_attention(q, k, v, causal=True)

# Example: Fused RMSNorm
output = aiter.rms_norm(input, weight, eps=1e-6)

# Example: GEMM
output = aiter.gemm(a, b, dtype=torch.bfloat16)
```

## When to Use AITER vs Custom Kernel

| Scenario | Use |
|----------|-----|
| Standard op, standard shapes | AITER (already optimized) |
| Fused op not in AITER | Custom kernel |
| AITER perf insufficient | Custom kernel (use AITER as baseline) |
| Non-standard precision mix | Custom kernel |

## Benchmarking Against AITER

```bash
python scripts/benchmark_kernel.py --kernel my_kernel.py --op attention --baseline aiter
```
```

- [ ] **Step 4: Write hip-intrinsics.md**

```markdown
# AMD GPU HIP Intrinsics Reference

## Cross-Lane Operations

| Intrinsic | Purpose | AMD ISA |
|-----------|---------|---------|
| `__builtin_amdgcn_readfirstlane(val)` | Broadcast lane 0 to all lanes (returns scalar) | v_readfirstlane_b32 |
| `__builtin_amdgcn_readlane(val, lane)` | Read specific lane value | v_readlane_b32 |
| `__builtin_amdgcn_ds_swizzle(val, pattern)` | Lane permutation (no LDS traffic) | ds_swizzle_b32 |
| `__builtin_amdgcn_mov_dpp(val, ctrl, row_mask, bank_mask, bound_ctrl)` | Data-parallel primitive | v_mov_b32 dpp |
| `__shfl(val, lane)` | Cross-lane shuffle (HIP compat) | Implementation varies |
| `__shfl_xor(val, mask)` | XOR-based shuffle | Implementation varies |

## Wavefront Reduction

```cpp
// Sum reduction across 64-lane wavefront
float wf_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Note: 6 steps needed (64 lanes), not 5 (32 lanes like NVIDIA)
```

## Memory Intrinsics

| Intrinsic | Purpose |
|-----------|---------|
| `__builtin_nontemporal_load(ptr)` | Load bypassing cache |
| `__builtin_nontemporal_store(val, ptr)` | Store bypassing cache |
| `__builtin_amdgcn_s_waitcnt(val)` | Fine-grained memory fence |

## Math Intrinsics

| Intrinsic | Purpose | Precision |
|-----------|---------|-----------|
| `__builtin_amdgcn_rcpf(x)` | Fast reciprocal | ~1 ULP |
| `__builtin_amdgcn_rsqf(x)` | Fast rsqrt | ~1 ULP |
| `__builtin_amdgcn_exp2f(x)` | Fast exp2 | ~1 ULP |
| `__builtin_amdgcn_log2f(x)` | Fast log2 | ~1 ULP |

## AMD vs NVIDIA Intrinsics

| NVIDIA | AMD Equivalent |
|--------|---------------|
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` (no mask needed, full wavefront) |
| `__syncwarp()` | Not needed (wavefront is lock-step) |
| `__ballot_sync(mask, pred)` | `__ballot(pred)` returns 64-bit |
| `atomicAdd(ptr, val)` for FP16 | `__builtin_amdgcn_flat_atomic_fadd_f16` |
```

- [ ] **Step 5: Commit**

```bash
git add amd-kernel-skill/references/ck-programming-model.md
git add amd-kernel-skill/references/ck-tile-tuning.md
git add amd-kernel-skill/references/aiter-ops-reference.md
git add amd-kernel-skill/references/hip-intrinsics.md
git commit -m "feat: add library and API reference documents (CK, AITER, HIP intrinsics)"
```

---

### Task 12: Optimization Pattern Documents + AMD vs NVIDIA Cheatsheet

**Files:**
- Create: `amd-kernel-skill/references/optimization-patterns.md`
- Create: `amd-kernel-skill/references/advanced-optimization.md`
- Create: `amd-kernel-skill/references/common-mistakes.md`
- Create: `amd-kernel-skill/references/kernel-recipes.md`
- Create: `amd-kernel-skill/references/amd-vs-nvidia-cheatsheet.md`

- [ ] **Step 1: Write optimization-patterns.md**

```markdown
# Common Optimization Patterns (Early & Mid Stage)

## Memory Optimization

### 1. Coalesced Global Memory Access
- Ensure consecutive threads access consecutive addresses
- Wavefront (64 threads) should access a contiguous 256B region
- Stride-1 access pattern is ideal

### 2. Vectorized Loads
- Use `float4` / `dwordx4` for 128-bit loads (4x bandwidth efficiency)
- Align data to 16-byte boundaries
- Triton: handled automatically with proper BLOCK_SIZE

### 3. LDS Usage
- Use LDS for data reuse within a thread block
- Budget: 64KB per CU (shared among all blocks on the CU)
- More LDS per block → fewer concurrent blocks → lower occupancy

## Compute Optimization

### 4. Kernel Fusion
- Fuse element-wise ops with preceding/following GEMM/reduction
- Saves global memory round-trips
- Common fusions: Linear+Activation, Norm+Scale, Attention+Softmax

### 5. Tiling
- Break large problem into tiles that fit in LDS
- Tile size = balance between data reuse (larger) and occupancy (smaller)
- See `ck-tile-tuning.md` for CK-specific tile recommendations

### 6. Loop Unrolling
- `#pragma unroll N` for HIP, automatic in Triton
- Sweet spot: enough to fill pipeline, not enough to spill registers
- Check VGPR count after unrolling

## Launch Optimization

### 7. Grid Sizing
- MI300X: 192 CUs. Need at least 192 blocks for full utilization
- More blocks (2-4x CU count) helps hide per-block variations
- Very small kernels: consider batching or persistent kernel

### 8. Thread Block Size
- Default: 256 (4 wavefronts per block) — good general choice
- Minimum: 64 (1 wavefront) — for register-heavy kernels
- Maximum: 1024 (16 wavefronts) — for high-occupancy needs

## Anti-Patterns (Hardware-Tagged)

| Anti-Pattern | Impact | Hardware Note |
|-------------|--------|---------------|
| Stride-N access | Bandwidth waste | All AMD GPUs |
| LDS > 32KB/block | Limits to 2 blocks/CU on MI300X | MI300X: 64KB/CU |
| `__syncthreads` in divergent code | Deadlock risk | All AMD GPUs |
| Assuming warp=32 | Wrong reduction, wrong shuffle | AMD wavefront=64 |
```

- [ ] **Step 2: Write advanced-optimization.md**

This file is the longest reference document. Write the full content from the design spec Section 4.3 "极限优化技术文档" — all 8 techniques plus the selection guide table. The content is already specified in the design spec lines 318-449. Copy and implement that structure exactly.

```markdown
# Advanced Optimization: Breaking Through Performance Plateaus

## When to Use

When the kernel has passed correctness, exceeds torch.compile baseline, but has remaining gap to theoretical peak (bandwidth utilization <70% or MFMA utilization <80%). Try techniques in order below.

## 1. Software Pipelining & Multi-Stage Buffering

**Principle**: Overlap current iteration's compute with next iteration's data load.

**Techniques**:
- Double buffering: LDS split in half, alternate load/compute
- Triple pipeline: load N+1, compute N, store N-1
- Async copy: DMA engine for data movement concurrent with MFMA

**AMD Implementation**:
- Requires precise `s_waitcnt lgkmcnt/vmcnt` (see `isa/scheduling-pipeline.md`)
- LDS doubles: check 64KB/CU limit
- Example: prefetch next tile while current MFMA executes

**Expected gain**: 10-30% for memory-bound kernels

## 2. Wavefront Specialization

**Principle**: Different wavefronts in same block do different jobs, coordinated via barriers.

**Roles**: compute wavefront (MFMA), data mover (global→LDS), reducer (softmax/reduction)

**AMD Implementation**:
- wavefront=64 → each wavefront is heavier, specialization benefit is larger
- Role assignment: `threadIdx.x / 64` gives wavefront ID
- Coordinate via `__syncthreads()` or LDS fence

**Expected gain**: 5-15% for multi-stage kernels (Attention)

## 3. Data Layout & Swizzle

**Principle**: Change data arrangement to eliminate bank conflicts and improve coalescing.

**Techniques**:
- LDS padding: add padding per row to avoid 32-bank conflicts (AMD: 32 banks × 4B)
- `ds_swizzle`: hardware lane permutation without LDS read/write
- SOA layout for vectorized loads (`buffer_load_dwordx4`)

**AMD Implementation**:
- AMD LDS bank rules differ from NVIDIA — recalculate padding
- Use `omniperf` LDS bank conflict metric to verify

**Expected gain**: 5-20% when bank conflicts are present

## 4. Occupancy vs ILP Trade-off

**Principle**: Sometimes fewer wavefronts with more registers outperform many wavefronts with spilling.

**Techniques**:
- `__launch_bounds__(threads, minBlocks)` to control register allocation
- Detect spilling: `omniperf` → ScratchWaveslifetimeVGPR > 0
- Register rebalancing across wavefront groups (ref: AVO v33)

**AMD Implementation**:
- MI300X: 65536 VGPR/CU. See `isa/register-allocation.md` for occupancy table
- Profile both high-occupancy and low-occupancy configs — faster one wins

**Expected gain**: 3-10% when eliminating spilling

## 5. Persistent Kernels & Tile Scheduling

**Principle**: Launch once, self-assign tiles via atomic counter. Eliminates re-launch overhead + enables L2-friendly traversal.

**Techniques**:
- `atomicAdd` global tile counter
- Swizzled traversal (L-shape, Z-shape, Hilbert) for L2 reuse
- Cross-tile load balancing for irregular shapes (causal mask)

**AMD Implementation**:
- MI300X large L2 (256MB) — tile order has significant impact
- CK's TileScheduler as reference implementation

**Expected gain**: 5-15% for multi-launch scenarios; 3-8% from L2 optimization

## 6. Mixed Precision Strategy

**Principle**: Use different precision at different computation stages.

**Techniques**:
- Input FP8/BF16 → MFMA compute → FP32 accumulate → BF16 output
- Keep critical intermediates (softmax max/sum) in FP32
- Exploit MFMA mixed-precision (FP16 input, FP32 output)

**AMD Implementation**:
- Check `isa/mfma-instructions.md` for available precision combos
- MI355X/CDNA4 may add new formats — check hardware docs

**Expected gain**: 10-30% TFLOPS improvement for compute-bound kernels

## 7. Compiler Guidance & Override

**Principle**: Manually steer compiler where auto-optimization falls short.

**Techniques**:
- `#pragma unroll N`: precise unroll control
- `__launch_bounds__`: register budget hints
- `volatile` / `__builtin_nontemporal_*`: cache bypass
- Inline ASM: last resort (see `isa/inline-asm-patterns.md`)

**AMD Implementation**:
- `hipcc -save-temps` to inspect generated ISA
- Compare compiler output with expected instruction sequence

**Expected gain**: 2-10%

## 8. L2 Cache Global Optimization

**Principle**: Maximize L2 hit rate across tiles and fused kernels.

**Techniques**:
- Tile traversal order (K-dimension sensitive for GEMM)
- Cross-kernel fusion to keep data in L2
- Prefetch hints (if hardware supports)

**AMD Implementation**:
- MI300X L2 = 256MB, relatively large
- Use `rocprof` TCC_HIT/TCC_MISS counters to measure

**Expected gain**: 3-8% for memory-bound kernels

## Selection Guide

| Bottleneck | Diagnostic Metric | Priority Techniques |
|------------|-------------------|---------------------|
| HBM bandwidth limited | BW utilization >80% | Software pipelining, L2 optimization, data layout |
| LDS limited | Bank conflict rate high | Swizzle, padding, data layout |
| Compute limited | MFMA utilization <70% | Wavefront specialization, mixed precision, ILP |
| Register spilling | ScratchWaves > 0 | Occupancy tuning, register rebalancing |
| Launch overhead | Many small kernel launches | Persistent kernel |
| Compiler issues | ISA review shows redundant instructions | Compiler guidance, inline ASM |
```

- [ ] **Step 3: Write common-mistakes.md**

```markdown
# Common AMD-Specific Mistakes

## Compilation

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Missing `--offload-arch` | Compiles for wrong target | Always specify `--offload-arch=gfx942` (or `gfx950`) |
| Missing `-O3` | 5-10x slower than expected | Always use `-O3` |
| Using CUDA APIs directly | Compile errors | Replace with HIP equivalents (see `amd-vs-nvidia-cheatsheet.md`) |

## Architecture

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Assuming warp = 32 | Wrong reduction results, perf regression | AMD wavefront = 64. Use 6 shuffle steps, not 5 |
| Assuming shared mem = 48KB | LDS overflow | AMD LDS = 64KB/CU (check per-hardware) |
| Using `__syncwarp()` | Unnecessary sync | AMD wavefronts are lock-step, no partial sync needed |
| Wrong bank conflict math | Unexpected LDS contention | AMD: 32 banks × 4B (not 32 banks × 4B as NVIDIA but different conflict patterns) |

## Performance

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Too few thread blocks | Low GPU utilization | Need ≥192 blocks for MI300X (one per CU minimum) |
| Ignoring register spilling | Mysterious slowdown | Check `omniperf` ScratchWaves, use `__launch_bounds__` |
| Copy-pasting CUDA tuning parameters | Suboptimal perf | Re-tune block sizes, unroll factors for AMD |
| Not using AGPR for MFMA | Higher VGPR pressure | Use AGPR accumulators to free VGPRs |

## Triton Specific

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using `tl.inline_asm_elementwise` | Error on ROCm | Use pure Triton ops |
| BLOCK_SIZE not multiple of 64 | Wasted lanes | Use 64, 128, 256, 512, 1024 |
| Assuming CUDA Triton perf transfers | Disappointment | Always benchmark on AMD |

## Knowledge Base

This file is a living document. Backfill new mistakes discovered during iteration.
```

- [ ] **Step 4: Write kernel-recipes.md**

```markdown
# Kernel Recipe Skeletons

Reference implementations for common operators. Use as starting points, not final solutions.

## RMSNorm (Triton)

```python
@triton.jit
def rms_norm_kernel(X, W, Y, stride, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + EPS)
    x_hat = x * rrms
    w = tl.load(W + cols, mask=mask)
    y = x_hat * w
    tl.store(Y + row * stride + cols, y, mask=mask)
```

## Fused SwiGLU (Triton)

```python
@triton.jit
def swiglu_kernel(X, GATE, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    g = tl.load(GATE + offs, mask=mask)
    silu_g = g * tl.sigmoid(g)
    y = x * silu_g
    tl.store(Y + offs, y, mask=mask)
```

## Vector Add (HIP C++ — Minimal Example)

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

## Reduction (HIP C++ — Wavefront-Aware)

```cpp
__device__ float warp_reduce_sum(float val) {
    // AMD wavefront = 64, need 6 steps
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void block_reduce(const float* input, float* output, int N) {
    __shared__ float shared[16];  // max 16 wavefronts per block (1024 threads)
    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    float val = (blockIdx.x * blockDim.x + tid < N)
                ? input[blockIdx.x * blockDim.x + tid] : 0.0f;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wf_id] = val;
    __syncthreads();

    if (wf_id == 0 && lane < (blockDim.x / 64)) {
        val = shared[lane];
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}
```

## Notes

- These are starting points. Always profile and optimize.
- Adapt block sizes and data types to target hardware.
- See `ck-tile-tuning.md` for CK GEMM configurations.
- Backfill new recipes as they are developed during iteration.
```

- [ ] **Step 5: Write amd-vs-nvidia-cheatsheet.md**

```markdown
# AMD vs NVIDIA — Quick Difference Reference

## Terminology

| NVIDIA | AMD | Notes |
|--------|-----|-------|
| Warp (32 threads) | Wavefront (64 threads) | 2x wider — affects reductions, shuffles |
| SM | CU (Compute Unit) | Similar concept |
| Shared Memory | LDS (Local Data Share) | 64KB/CU on MI300X |
| CUDA Core | Stream Processor | — |
| Tensor Core | Matrix Core (MFMA) | Different ISA, different register layout |
| PTX | AMDGPU ISA | Intermediate vs final ISA |
| nvcc | hipcc | — |
| ncu / nsys | rocprof / omniperf | — |
| cuDNN | MIOpen | — |
| CUTLASS | CK (Composable Kernel) | — |

## API Mapping

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__syncwarp()` | Not needed (wavefront is lock-step) |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` |
| `__ballot_sync(mask, pred)` | `__ballot(pred)` (returns 64-bit) |
| `__shared__` | `__shared__` (same) |
| `blockDim.x` | `blockDim.x` (same) |

## Key Behavioral Differences

| Aspect | NVIDIA | AMD |
|--------|--------|-----|
| Warp/wavefront size | 32 | 64 |
| Shuffle steps for reduction | 5 | 6 |
| `__syncwarp()` needed? | Yes (independent scheduling) | No (lock-step) |
| Shared memory banks | 32 × 4B | 32 × 4B (but conflict patterns differ) |
| L2 cache size | 40-50 MB (A100/H100) | 256 MB (MI300X) |
| Tensor core input | HMMA (warp-level) | MFMA (wavefront-level) |
| Occupancy calculator | CUDA occ calculator | `rocminfo` + manual calc |
| Inline ASM | PTX asm | AMDGPU ISA asm |
| Compilation target flag | `-arch=sm_80` | `--offload-arch=gfx942` |

## Migration Checklist

1. Replace CUDA APIs with HIP equivalents (`cuda` → `hip`)
2. Change warp size assumptions: 32 → 64
3. Update reduction loops: 5 steps → 6 steps
4. Remove `__syncwarp()` calls
5. Remove `mask` parameter from shuffle/ballot
6. Change `__ballot_sync` return from 32-bit to 64-bit
7. Update compilation: `nvcc` → `hipcc`, `-arch=sm_XX` → `--offload-arch=gfxYYY`
8. Re-tune block sizes and unroll factors
9. Re-benchmark everything — don't assume CUDA perf transfers
```

- [ ] **Step 6: Commit**

```bash
git add amd-kernel-skill/references/optimization-patterns.md
git add amd-kernel-skill/references/advanced-optimization.md
git add amd-kernel-skill/references/common-mistakes.md
git add amd-kernel-skill/references/kernel-recipes.md
git add amd-kernel-skill/references/amd-vs-nvidia-cheatsheet.md
git commit -m "feat: add optimization patterns, advanced optimization, common mistakes, recipes, and AMD vs NVIDIA cheatsheet"
```

---

### Task 13: Integration Validation

**Files:**
- Verify: All files in `amd-kernel-skill/` exist and are well-formed

- [ ] **Step 1: Verify file count and structure**

```bash
find amd-kernel-skill -type f | sort
```

Expected: 31 files total:
- `SKILL.md` (1)
- `skills/*/SKILL.md` (3)
- `scripts/*.py` (2)
- `templates/*` (3)
- `references/hardware/*.md` (3)
- `references/isa/*.md` (7)
- `references/*.md` (9: rocprof, omniperf, hipcc, triton-rocm-quirks, ck-programming-model, ck-tile-tuning, aiter-ops, hip-intrinsics, amd-vs-nvidia-cheatsheet)
- `references/optimization-patterns.md` (1)
- `references/advanced-optimization.md` (1)
- `references/common-mistakes.md` (1)
- `references/kernel-recipes.md` (1)

- [ ] **Step 2: Verify main SKILL.md references exist**

Check that every file referenced in the main SKILL.md and sub-skills actually exists:

```bash
# Extract referenced paths from SKILL.md files
grep -rh "references/" amd-kernel-skill/SKILL.md amd-kernel-skill/skills/*/SKILL.md | \
  grep -oP 'references/[a-z0-9/_-]+\.md' | sort -u | \
  while read f; do
    [ -f "amd-kernel-skill/$f" ] && echo "[OK] $f" || echo "[MISSING] $f"
  done
```

Expected: All OK, no MISSING entries.

- [ ] **Step 3: Verify scripts are syntactically valid**

```bash
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/scripts/verify_correctness.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/scripts/benchmark_kernel.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/templates/triton_kernel_template.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/templates/benchmark_template.py', doraise=True)"
```

Expected: No errors.

- [ ] **Step 4: Final commit with tag**

```bash
git add -A amd-kernel-skill/
git commit -m "feat: complete AMD Kernel Agent Skill Pack v0.1

Includes:
- Main SKILL.md router (dual-dimension: path x hardware)
- 3 sub-skills: Triton, HIP C++, Composable Kernel
- 5-layer reference knowledge base (23 documents)
- Verification scripts (correctness + benchmark)
- 3 templates (Triton, HIP, benchmark)
- Structured iteration recording in agent_output/"
```

- [ ] **Step 5: Copy to Claude Code skills directory (for testing)**

```bash
# Claude Code
cp -r amd-kernel-skill/ /path/to/project/.claude/skills/amd-kernel-skill/

# Cursor
cp -r amd-kernel-skill/ /path/to/project/.cursor/skills/amd-kernel-skill/
```

Adjust `/path/to/project/` to the actual Primus-Turbo project path.
