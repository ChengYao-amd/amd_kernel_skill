# AMD Kernel Agent Skill Pack 实施计划

> **面向 Agent 工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 构建一个分层的 Agent Skill Pack，使现有 LLM 能够通过 Triton/HIP/CK 路径对 AMD GPU kernel 进行优化，并支持自动化验证。

**架构：** 主 SKILL.md 按编程路径（Triton/HIP/CK）× 目标硬件（MI300X/MI355X）进行路由。每个子技能遵循 6 步流程（硬件检测 -> 分析 -> 实现 -> 验证 -> 迭代 -> 知识积累）。5 层参考知识库提供 AMD 特定的领域知识。三门验证（编译 -> 正确性 -> 性能）确保质量。

**技术栈：** Python（Triton、PyTorch、benchmarking 脚本）、C++（HIP kernel）、Markdown（skills、references）、ROCm 工具链（hipcc、rocprof、omniperf）

---

### 任务 1：项目脚手架搭建

**文件：**
- 创建：`amd-kernel-skill/` 目录树

- [ ] **步骤 1：创建所有目录**

```bash
mkdir -p amd-kernel-skill/{skills/{triton-kernel,hip-kernel,ck-kernel},references/{hardware,isa},templates,scripts,agent_output}
```

- [ ] **步骤 2：验证结构**

```bash
find amd-kernel-skill -type d | sort
```

预期输出：
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

- [ ] **步骤 3：Commit**

```bash
cd amd-kernel-skill && git init && cd ..
git add amd-kernel-skill/
git commit -m "chore: scaffold amd-kernel-skill directory structure"
```

---

### 任务 2：主 SKILL.md（路由器）

**文件：**
- 创建：`amd-kernel-skill/SKILL.md`

- [ ] **步骤 1：编写主路由 skill**

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

- [ ] **步骤 2：验证 token 数量低于 400**

```bash
wc -w amd-kernel-skill/SKILL.md
```

预期：约 250-350 词（远低于 400 token 目标）

- [ ] **步骤 3：Commit**

```bash
git add amd-kernel-skill/SKILL.md
git commit -m "feat: add main SKILL.md router for AMD kernel optimization"
```

---

### 任务 3：Triton 子技能

**文件：**
- 创建：`amd-kernel-skill/skills/triton-kernel/SKILL.md`

- [ ] **步骤 1：编写 Triton 子技能**

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

- [ ] **步骤 2：Commit**

```bash
git add amd-kernel-skill/skills/triton-kernel/SKILL.md
git commit -m "feat: add Triton kernel sub-skill"
```

---

### 任务 4：HIP C++ 子技能

**文件：**
- 创建：`amd-kernel-skill/skills/hip-kernel/SKILL.md`

- [ ] **步骤 1：编写 HIP 子技能**

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

- [ ] **步骤 2：Commit**

```bash
git add amd-kernel-skill/skills/hip-kernel/SKILL.md
git commit -m "feat: add HIP C++ kernel sub-skill"
```

---

### 任务 5：CK 子技能

**文件：**
- 创建：`amd-kernel-skill/skills/ck-kernel/SKILL.md`

- [ ] **步骤 1：编写 CK 子技能**

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

- [ ] **步骤 2：Commit**

```bash
git add amd-kernel-skill/skills/ck-kernel/SKILL.md
git commit -m "feat: add Composable Kernel sub-skill"
```

---

### 任务 6：验证脚本

**文件：**
- 创建：`amd-kernel-skill/scripts/verify_correctness.py`
- 创建：`amd-kernel-skill/scripts/benchmark_kernel.py`

- [ ] **步骤 1：编写 verify_correctness.py**

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

- [ ] **步骤 2：编写 benchmark_kernel.py**

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

- [ ] **步骤 3：Commit**

```bash
git add amd-kernel-skill/scripts/
git commit -m "feat: add verification and benchmark scripts"
```

---

### 任务 7: 模板文件

**文件:**
- 创建: `amd-kernel-skill/templates/triton_kernel_template.py`
- 创建: `amd-kernel-skill/templates/hip_kernel_template.cpp`
- 创建: `amd-kernel-skill/templates/benchmark_template.py`

- [ ] **步骤 1: 编写 triton_kernel_template.py**

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

- [ ] **步骤 2: 编写 hip_kernel_template.cpp**

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

- [ ] **步骤 3: 编写 benchmark_template.py**

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

- [ ] **步骤 4: 提交**

```bash
git add amd-kernel-skill/templates/
git commit -m "feat: add Triton, HIP, and benchmark templates"
```

---

### 任务 8: 硬件参考文档

**文件:**
- 创建: `amd-kernel-skill/references/hardware/mi300x.md`
- 创建: `amd-kernel-skill/references/hardware/mi355x.md`
- 创建: `amd-kernel-skill/references/hardware/hardware-comparison.md`

- [ ] **步骤 1: 编写 mi300x.md**

```markdown
# MI300X (GFX942 / CDNA3) 优化指南

## 核心参数

| 参数 | 值 |
|------|-----|
| 架构 | CDNA3 |
| offload-arch | `gfx942` |
| 计算单元 (CU) | 192 |
| HBM | 192 GB HBM3 |
| 内存带宽 | 5.3 TB/s |
| Wavefront 大小 | 64 |
| LDS / CU | 64 KB |
| VGPR / CU | 65536（满 occupancy 时每个 wavefront 512） |
| SGPR / CU | 3200 |
| Matrix Core | MFMA (FP16/BF16/FP8/INT8 → FP32) |
| L2 Cache | 256 MB |
| 最大线程数 / CU | 2048（32 个 wavefront） |

## 编译

```bash
# HIP
hipcc -O3 --offload-arch=gfx942 kernel.cpp

# Triton: 自动检测，无需指定标志

# CK: 使用 CMake，设置 AMDGPU_TARGETS=gfx942
```

## 推荐配置

| 路径 | 参数 | 推荐值 |
|------|------|--------|
| Triton | BLOCK_SIZE | 128 或 256 |
| Triton | num_warps | 4-8 |
| HIP | Thread block | 256 线程（4 个 wavefront） |
| CK (BF16 GEMM) | block_m, block_n, block_k | 256, 128, 64 |

## 性能阈值参考

| 指标 | 良好 | 需改进 |
|------|------|--------|
| Occupancy | > 50% | < 25% |
| HBM 带宽利用率 | > 60%（内存受限） | < 30% |
| MFMA 利用率 | > 70%（计算受限） | < 40% |
| LDS Bank Conflict | < 5% | > 20% |
| 寄存器溢出 | 0 scratch waves | 任何 scratch |

## 硬件特有技巧

- 192 CU = 高并行度。小 kernel 可能利用率不足，考虑 persistent kernel。
- 64 宽 wavefront：reduction 需要 6 步 shuffle（而非 NVIDIA warp=32 的 5 步）
- 256 MB L2：大于 NVIDIA 对应产品。Tile 遍历顺序对 L2 复用影响显著。
- HBM3 5.3 TB/s：内存受限 kernel 在合并访问时可达到高吞吐。

## 已知问题

- `hipcc` 省略 `-O3` 时可能静默回退到较慢的指令模式
- Triton ROCm 后端不支持 `tl.inline_asm_elementwise`
- 与 NVIDIA 不同的 grid launch 限制 — 使用 `hipDeviceGetAttribute` 检查
```

- [ ] **步骤 2: 编写 mi355x.md**

```markdown
# MI355X (GFX950 / CDNA4) 优化指南

## 核心参数

| 参数 | 值 |
|------|-----|
| 架构 | CDNA4 |
| offload-arch | `gfx950` |
| 计算单元 (CU) | 待确认（使用 `rocminfo` 查询） |
| HBM | 288 GB HBM3E |
| 内存带宽 | 待确认 |
| Wavefront 大小 | 64 |
| LDS / CU | 待确认（使用 `rocminfo` 查询） |
| Matrix Core | MFMA（CDNA4 — 新指令变体） |

## 编译

```bash
hipcc -O3 --offload-arch=gfx950 kernel.cpp
```

## 与 MI300X (GFX942) 的差异

| 方面 | MI300X (gfx942) | MI355X (gfx950) |
|------|----------------|-----------------|
| HBM | 192 GB HBM3 | 288 GB HBM3E |
| 新 MFMA 变体 | — | 查阅 ISA 文档了解 CDNA4 新增指令 |
| offload-arch | gfx942 | gfx950 |

## 从 MI300X 迁移

1. 将 `--offload-arch=gfx942` 改为 `--offload-arch=gfx950`
2. 重新调优 tile size / block size（不同的 CU 数量和带宽）
3. 检查是否有新的 MFMA 指令可提升计算吞吐
4. 重新验证性能阈值（绝对数值会不同）
5. 在 `performance_trend.md` 中标注硬件变更

## 已知问题

- 在迭代过程中发现问题时记录到此处
- 从 `agent_output/` 复盘文档中回填本节
```

- [ ] **步骤 3: 编写 hardware-comparison.md**

```markdown
# AMD GPU 硬件对比

## 速查表

| 参数 | MI300X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) |
|------|----------------------|----------------------|
| 计算单元 | 192 | 待确认 |
| HBM | 192 GB HBM3 | 288 GB HBM3E |
| 内存带宽 | 5.3 TB/s | 待确认 |
| Wavefront | 64 | 64 |
| LDS/CU | 64 KB | 待确认 |
| Matrix Core | MFMA (CDNA3) | MFMA (CDNA4) |
| offload-arch | gfx942 | gfx950 |

## 跨硬件移植清单

1. **编译**：更新 `--offload-arch` 标志
2. **Tile/Block size**：重新调优。最优值取决于 CU 数量、带宽、LDS 大小
3. **新指令**：检查 CDNA4 是否新增 MFMA 变体或内存指令
4. **性能基线**：重新 benchmark。绝对数值因硬件不同而变化
5. **条件编译**：使用 `#if __gfx942__` / `#if __gfx950__` 实现硬件特化路径
6. **Profiling 阈值**：调整 occupancy / 带宽利用率的"良好"标准

## 何时使用多目标编译

如果代码需要同时在 MI300X 和 MI355X 上运行：
```bash
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp
```
编译器会为两个目标分别生成代码。通过运行时分发或条件编译实现硬件特化优化。
```

- [ ] **步骤 4: 提交**

```bash
git add amd-kernel-skill/references/hardware/
git commit -m "feat: add hardware reference documents (MI300X, MI355X, comparison)"
```

---

### 任务 9: ISA 参考文档

**文件:**
- 创建: `amd-kernel-skill/references/isa/isa-overview.md`
- 创建: `amd-kernel-skill/references/isa/mfma-instructions.md`
- 创建: `amd-kernel-skill/references/isa/memory-instructions.md`
- 创建: `amd-kernel-skill/references/isa/valu-salu-instructions.md`
- 创建: `amd-kernel-skill/references/isa/register-allocation.md`
- 创建: `amd-kernel-skill/references/isa/scheduling-pipeline.md`
- 创建: `amd-kernel-skill/references/isa/inline-asm-patterns.md`

- [ ] **步骤 1: 编写 isa-overview.md**

```markdown
# AMDGPU ISA 速查参考

ISA 级优化的入口文档。先读此文件，再根据需要深入具体文档。

## 寄存器文件

| 类型 | 每 CU 数量 | 满 occupancy 时每 wavefront | 用途 |
|------|-----------|---------------------------|------|
| VGPR | 65536 | 512（每 SIMD 8 个 wf 时） | 每 lane 数据、MFMA 操作数 |
| SGPR | 3200 | 102 | 统一值、地址、控制 |
| AGPR | 65536 | 512 | MFMA 累加器（CDNA） |

## 指令分类

| 类别 | 执行单元 | 示例 | 延迟 |
|------|---------|------|------|
| VALU | 向量 ALU | v_add_f32, v_fma_f32 | 4-8 周期 |
| SALU | 标量 ALU | s_add_u32, s_cmp_eq | 2-4 周期 |
| MFMA | Matrix Core | v_mfma_f32_16x16x16_f16 | 8-64 周期 |
| VMEM | 向量内存 | global_load_dwordx4 | 300-500 周期 |
| LDS | 本地数据共享 | ds_read_b128, ds_write_b64 | 20-40 周期 |
| SMEM | 标量内存 | s_load_dwordx4 | 200+ 周期 |

## 流水线模型（简化）

- 每个 CU 有 4 个 SIMD 单元，每个 SIMD 每周期执行一条 wavefront 指令
- VALU：每 SIMD 每周期 1 条指令（64 个 lane 需要 4 个周期）
- MFMA：发射到 matrix core，高延迟但流水线化后吞吐高
- 内存：发射到内存单元，异步返回 → 使用 s_waitcnt 同步

## ISA 知识使用时机

| 优化阶段 | ISA 深度 | 对应文档 |
|---------|---------|---------|
| 早期（融合、tiling） | 低 | 仅本文件 |
| 中期（occupancy、内存访问） | 中 | memory-instructions.md, register-allocation.md |
| 后期（指令调度、消除 bubble） | 高 | scheduling-pipeline.md, inline-asm-patterns.md |
| MFMA 密集型（GEMM、Attention） | 高 | mfma-instructions.md |
```

- [ ] **步骤 2: 编写 mfma-instructions.md**

```markdown
# MFMA（Matrix Fused Multiply-Add）指令参考

## 指令格式

`v_mfma_{out_type}_{M}x{N}x{K}_{in_type}` — 将 M×K 矩阵与 K×N 矩阵相乘，累加到 M×N 结果。

## 可用指令（CDNA3 / GFX942）

| 指令 | 输入类型 | 输出类型 | M×N×K | VGPR 输入 | VGPR 输出 | 周期 |
|------|---------|---------|-------|----------|----------|------|
| v_mfma_f32_16x16x16_f16 | FP16 | FP32 | 16×16×16 | 4+4 | 4 | 64 |
| v_mfma_f32_16x16x16_bf16 | BF16 | FP32 | 16×16×16 | 4+4 | 4 | 64 |
| v_mfma_f32_32x32x8_f16 | FP16 | FP32 | 32×32×8 | 4+4 | 16 | 64 |
| v_mfma_f32_32x32x8_bf16 | BF16 | FP32 | 32×32×8 | 4+4 | 16 | 64 |
| v_mfma_f32_16x16x32_fp8 | FP8 | FP32 | 16×16×32 | 4+4 | 4 | 64 |
| v_mfma_i32_16x16x32_i8 | INT8 | INT32 | 16×16×32 | 4+4 | 4 | 64 |

## 寄存器布局

MFMA 输出以特定模式分布在 VGPR lane 中：
- 16x16：每个 lane 持有一个元素，4 个 VGPR 保存 4×16×16/64 的结果
- 32x32：16 个 VGPR 保存结果，分布在 64 个 lane 上

关键：lane 到矩阵元素的映射与 NVIDIA 的 MMA 不同。不要假设 CUDA 布局。

## 与 NVIDIA Tensor Core 对比

| 方面 | AMD MFMA | NVIDIA MMA |
|------|----------|------------|
| 执行单元 | 每 wavefront（64 lane） | 每 warp（32 lane） |
| 累加器 | AGPR（专用）或 VGPR | 与 VGPR 共享 |
| 最大 tile | 32×32 | 16×16 (Ampere)、64×64 (Hopper) |
| 调度 | 必须手动与 VMEM 交错 | TMA 处理异步 |

## 优化建议

1. **MFMA 与 load 流水线化**：MFMA 执行期间（64 周期），发射下一个 tile 的 global_load
2. **使用 AGPR 做累加器**：释放 VGPR 给数据，降低寄存器压力
3. **优先使用大 tile**：32×32×8 比 16×16×16 有更好的计算/寄存器比
4. **检查 CDNA4 新增指令**：MI355X 可能有新的 MFMA 变体 — 查阅 `hardware/mi355x.md`
```

- [ ] **步骤 3: 编写 memory-instructions.md**

```markdown
# 内存指令参考

## Global Memory

| 指令 | 宽度 | 说明 |
|------|------|------|
| global_load_dword | 4B | 单 lane |
| global_load_dwordx2 | 8B | 向量化 |
| global_load_dwordx4 | 16B | 合并访问最佳 |
| global_store_dword[x2/x4] | 4-16B | 同样宽度 |

**合并规则**：64 个连续 lane 访问 64 个连续元素 = 一次合并事务。步长 > 1 会降低带宽。

## LDS（Local Data Share）

| 指令 | 宽度 | 说明 |
|------|------|------|
| ds_read_b32 | 4B | 单 bank |
| ds_read_b64 | 8B | 两个 bank |
| ds_read_b128 | 16B | 四个 bank |
| ds_write_b32/b64/b128 | 4-16B | 同样宽度 |
| ds_swizzle_b32 | 4B | 无 LDS 读写的 lane 排列 |

**Bank conflict 规则**：32 bank × 4B 每 bank。两个 lane 访问同一 bank（不同地址）= conflict = 串行化。通过 padding 或 swizzle 解决。

## Buffer vs Flat 指令

| 类型 | 使用场景 |
|------|---------|
| `buffer_load_*` | 已知 base + offset，支持范围检查，略快 |
| `global_load_*` (flat) | 任意 64 位地址，codegen 更简单 |

编译器通常自动选择。内联汇编时，base 地址为统一值（SGPR）时优先使用 buffer 指令。

## 内存屏障与同步

### s_waitcnt — 关键指令

```
s_waitcnt vmcnt(N) lgkmcnt(M) expcnt(K)
```

| 计数器 | 追踪内容 | 等待条件 |
|--------|---------|---------|
| vmcnt | Global load/store | 剩余 N 个未完成 VMEM 操作 |
| lgkmcnt | LDS + SMEM 操作 | 剩余 M 个未完成 LDS/SMEM 操作 |
| expcnt | Export (GDS, LDS→VGPR) | 剩余 K 个未完成 export |

**策略**：不要到处使用 `s_waitcnt 0`（会杀死 ILP）。计算未完成的操作数，只等待真正需要的。

```
global_load_dwordx4 v[0:3], ...    // vmcnt = 1
global_load_dwordx4 v[4:7], ...    // vmcnt = 2
// ... 做其他工作 ...
s_waitcnt vmcnt(1)                  // 只等第一个 load
v_add_f32 v8, v0, v1               // 使用第一个 load 的结果
s_waitcnt vmcnt(0)                  // 现在等第二个 load
v_add_f32 v9, v4, v5               // 使用第二个 load 的结果
```
```

- [ ] **步骤 4: 编写 valu-salu-instructions.md**

```markdown
# 向量 ALU (VALU) 与标量 ALU (SALU) 参考

## VALU — 每 Lane 计算

对 wavefront 的全部 64 个 lane 操作，4 个周期完成（每周期 16 lane）。

| 类别 | 示例 | 吞吐量 |
|------|------|--------|
| FP32 算术 | v_add_f32, v_mul_f32, v_fma_f32 | 每 SIMD 每周期 1 条 |
| FP16 Packed | v_pk_add_f16, v_pk_mul_f16 | 每周期 1 条（每 lane 2 个 FP16） |
| 类型转换 | v_cvt_f32_f16, v_cvt_f16_f32 | 每周期 1 条 |
| 比较 | v_cmp_gt_f32 → 写入 VCC | 每周期 1 条 |
| 位操作 | v_bfe_u32, v_bfi_b32 | 每周期 1 条 |
| 超越函数 | v_rcp_f32, v_rsq_f32, v_exp_f32 | 每 4 周期 1 条（共享单元） |

**关键**：超越函数慢 4 倍。避免在内层循环使用。可能时用多项式近似替代。

## SALU — 统一值计算

对单个标量值操作，wavefront 内所有 lane 共享。

| 类别 | 示例 | 吞吐量 |
|------|------|--------|
| 算术 | s_add_u32, s_mul_i32 | 每周期 1 条 |
| 逻辑 | s_and_b64, s_or_b64 | 每周期 1 条 |
| 比较 | s_cmp_eq_u32 | 每周期 1 条 |
| 分支 | s_cbranch_scc1 | 可变 |
| 常量加载 | s_load_dwordx4 | ~200 周期 |

**关键**：将统一计算（循环计数器、地址）移到 SALU，释放 VALU 给数据计算。

## 双发射

CDNA3 可在同一周期双发射 VALU + SALU，条件：
- 两者之间无数据依赖
- 使用不同寄存器文件（VGPR vs SGPR）

优化：将 SALU 地址计算与 VALU 数据计算交错排列。
```

- [ ] **步骤 5: 编写 register-allocation.md**

```markdown
# 寄存器分配与 Occupancy 指南

## VGPR 预算与 Occupancy（MI300X，每个 SIMD 单元）

| 最大 VGPR/Wavefront | 最大 Wavefront/SIMD | Occupancy |
|---------------------|---------------------|-----------|
| 128 | 8 | 100% |
| 256 | 4 | 50% |
| 512 | 2 | 25% |
| 1024 | 1 | 12.5% |

公式：`wavefronts_per_simd = floor(16384 / vgpr_per_wavefront)`，上限 8。

## SGPR 预算

| 最大 SGPR/Wavefront | 最大 Wavefront/SIMD |
|---------------------|---------------------|
| ≤ 102 | 8 |
| > 102 | 溢出到内存 |

SGPR 很少限制 occupancy。重点管理 VGPR。

## AGPR（Accumulation GPR）

- 专用于 MFMA 累加器结果
- 与 VGPR 相同的文件大小（65536/CU）
- 使用 AGPR 做累加器可释放 VGPR 给数据 → 更高 occupancy
- VGPR 与 AGPR 之间移动：`v_accvgpr_read` / `v_accvgpr_write`

## 检测寄存器压力

```bash
# 检查 VGPR/SGPR 使用量
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# 查找 .s 文件，搜索 .vgpr_count 和 .sgpr_count

# 在 omniperf 输出中
# ScratchWaveslifetimeVGPR > 0 表示寄存器溢出！
```

## 降低寄存器压力

| 技术 | 方法 | 影响 |
|------|------|------|
| 缩短活跃范围 | 就近计算和消费值 | 中等 |
| 使用 AGPR 做累加器 | `v_accvgpr_write` | 释放 VGPR |
| `__launch_bounds__(threads, minBlocks)` | 提示编译器寄存器预算 | 直接 |
| 手动寄存器复用 | 重写循环以复用寄存器 | 高工作量 |
| 接受较低 occupancy | 如果 ILP 能隐藏延迟，更少 wave 也可以 | 权衡 |

## Occupancy vs ILP 权衡

低 occupancy 不一定是坏事。如果 kernel 有足够的 ILP（内存操作之间的独立指令），更少的 wavefront 配合更多寄存器可能优于大量 wavefront 配合寄存器溢出。

决策流程：
1. 检查是否溢出（ScratchWaves > 0）
2. 如果溢出：减少 VGPR 使用或接受较低 occupancy
3. 如果未溢出但性能低：尝试通过减少 VGPR 来提高 occupancy
4. 两种配置都 profile。更快的那个获胜。
```

- [ ] **步骤 6: 编写 scheduling-pipeline.md**

```markdown
# 指令调度与流水线指南

## CDNA3 流水线模型

每个 CU 有 4 个 SIMD 单元。每个 SIMD：
- 每周期执行一条 wavefront 指令
- 在就绪的 wavefront 之间轮询（TLP 隐藏延迟）

延迟隐藏：如果 wavefront A 在内存上停顿，SIMD 执行 wavefront B、C、D 等。

## 指令级并行（ILP）

当 occupancy 较低（wavefront 少）时，单个 wavefront 内的 ILP 变得至关重要。

**目标**：通过交错独立指令保持流水线饱满。

```
// 差：依赖链 → 流水线停顿
global_load v0, ...
s_waitcnt vmcnt(0)    // 在此停顿
v_add_f32 v1, v0, v2  // 必须等待 load

// 好：交错独立工作
global_load v0, ...    // 发射 load
v_mul_f32 v3, v4, v5   // 独立的 VALU 工作
s_add_u32 s0, s0, 1    // 独立的 SALU 工作（可双发射！）
s_waitcnt vmcnt(0)     // 此时 load 大概率已完成
v_add_f32 v1, v0, v2   // 使用已加载的数据
```

## s_waitcnt 策略

**原则**：尽可能晚等待，尽可能少等待。

| 模式 | 代码 | 原因 |
|------|------|------|
| 立即等待 | load 后立即 `s_waitcnt vmcnt(0)` | 差：杀死 ILP |
| 延迟等待 | load，做其他工作，然后等待 | 好：隐藏延迟 |
| 部分等待 | `vmcnt(N)` 其中 N = 允许挂起的剩余操作数 | 最佳：最小停顿 |

## 双发射规则（CDNA3）

VALU + SALU 可在同一周期发射，条件：
1. 两者之间无寄存器依赖
2. VALU 使用 VGPR，SALU 使用 SGPR
3. 两者都就绪（无挂起的等待）

**优化**：将地址计算（SALU）与数据操作（VALU）配对。

## MFMA 调度

MFMA 指令延迟高（64 周期）但可与其他工作重叠：

```
// 流水线：MFMA N 执行时，加载 N+1 的数据
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 64 周期
global_load_dwordx4 v[0:3], ...  // 在 MFMA 延迟期间发射
global_load_dwordx4 v[4:7], ...  // 在 MFMA 延迟期间发射
s_waitcnt vmcnt(0)               // 此时 load 应已完成
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 下一个 MFMA
```
```

- [ ] **步骤 7: 编写 inline-asm-patterns.md**

```markdown
# HIP 内联汇编常用 Pattern

## 何时使用内联汇编

1. 编译器未能生成最优指令（用 `-save-temps` 验证）
2. 需要 intrinsics 未暴露的特定指令
3. 关键内层循环的最后手段 — 优先使用 `__builtin_amdgcn_*`

## 语法

```cpp
asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
```

约束代码：
- `v` = VGPR, `s` = SGPR, `a` = AGPR
- `=` = 输出, 无前缀 = 输入

## Builtin → ISA 映射

| Builtin | ISA 指令 | 用途 |
|---------|---------|------|
| `__builtin_amdgcn_readfirstlane(v)` | v_readfirstlane_b32 | 将第一个 lane 广播到 SGPR |
| `__builtin_amdgcn_ds_swizzle(v, pat)` | ds_swizzle_b32 | Lane 排列（无 LDS 流量） |
| `__builtin_amdgcn_mov_dpp(v, ctrl, ...)` | v_mov_b32 dpp | 数据并行原语 |
| `__shfl_sync` 等价物 | ds_swizzle / dpp | 跨 lane 通信 |

## 常见优化 Pattern

### 1. 向量化 Global Load

```cpp
// 强制 128 位 load（一次 4 个 float）
float4 data;
asm volatile(
    "global_load_dwordx4 %0, %1, off"
    : "=v"(data) : "v"(addr)
);
```

### 2. LDS Swizzle 消除 Bank Conflict

```cpp
// 矩阵转置时避免 bank conflict 的 swizzle pattern
int swizzled = __builtin_amdgcn_ds_swizzle(val, 0x041f);
```

### 3. 内层循环手动 MFMA

```cpp
// 当编译器 MFMA 调度不理想时
asm volatile(
    "v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
    : "+a"(acc)  // AGPR 累加器（读写）
    : "v"(a_frag), "v"(b_frag)
);
```

### 4. 精确 s_waitcnt

```cpp
// 等待恰好 1 个未完成的 VMEM 操作
asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
```

## 决策指南

| 场景 | 方法 |
|------|------|
| 需要跨 lane 操作 | 先尝试 `__builtin_amdgcn_*` |
| 编译器生成了次优 load | 检查 `-O3` 是否修复，否则尝试 asm |
| MFMA 调度不正确 | 先 profile，确认后再用手动 asm |
| 需要精确的等待计数 | 内联 `s_waitcnt` 并指定确切计数 |
```

- [ ] **步骤 8: 提交**

```bash
git add amd-kernel-skill/references/isa/
git commit -m "feat: add ISA reference documents (7 files)"
```

---

### 任务 10: 工具链参考文档

**文件:**
- 创建: `amd-kernel-skill/references/rocprof-guide.md`
- 创建: `amd-kernel-skill/references/omniperf-guide.md`
- 创建: `amd-kernel-skill/references/hipcc-compilation.md`
- 创建: `amd-kernel-skill/references/triton-rocm-quirks.md`

- [ ] **步骤 1: 编写 rocprof-guide.md**

```markdown
# rocprof 使用指南

## 基本 Profiling

```bash
# Kernel 执行统计（时间、调用次数、occupancy）
rocprof --stats python run_kernel.py

# 硬件计数器
rocprof -i counters.txt python run_kernel.py

# 时间戳 trace
rocprof --timestamp on python run_kernel.py
```

## 关键计数器

创建 `counters.txt`：
```
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM SQ_INSTS_LDS
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum
pmc: SQ_INSTS_MFMA TA_FLAT_READ_WAVEFRONTS_sum
```

## 输出解读

| 计数器 | 含义 | 行动 |
|--------|------|------|
| SQ_WAVES | 发射的 wavefront 总数 | 检查 grid 大小是否合理 |
| SQ_INSTS_VALU | 向量 ALU 指令数 | 高 = 计算受限 |
| SQ_INSTS_LDS | LDS 操作数 | 高 = 检查 bank conflict |
| SQ_INSTS_MFMA | Matrix core 操作数 | GEMM kernel 应该很高 |
| TCC_HIT_sum / TCC_MISS_sum | L2 cache 命中/未命中 | 命中率低 = 数据复用差 |

## 常用工作流

1. `rocprof --stats` → 找到最慢的 kernel
2. `rocprof -i counters.txt` → 分类瓶颈（计算/内存/LDS）
3. 根据瓶颈优化
4. 重新 profile 验证改进
```

- [ ] **步骤 2: 编写 omniperf-guide.md**

```markdown
# omniperf Profiling 指南

## 工作流

```bash
# 1. 采集数据
omniperf profile -n workload_name -- python run_kernel.py

# 2. 分析（命令行）
omniperf analyze -p workload_name/ --cli

# 3. 分析（图形界面 — 如可用）
omniperf analyze -p workload_name/ --gui
```

## 关键指标

| 面板 | 指标 | 健康值 | 问题值 |
|------|------|--------|--------|
| Speed-of-Light | GPU 利用率 | > 60% | < 30% |
| Speed-of-Light | 内存带宽利用率 | > 60%（内存受限） | < 30% |
| 计算 | VALU 利用率 | 视情况 | < 20%（计算浪费） |
| 计算 | MFMA 利用率 | > 70% (GEMM) | < 40% |
| 内存 | L2 命中率 | > 80% | < 50% |
| 内存 | LDS Bank Conflict | < 5% | > 20% |
| Occupancy | 实际 Occupancy | > 50% | < 25% |
| Occupancy | ScratchWaveslifetimeVGPR | 0 | > 0 = 寄存器溢出！ |

## 瓶颈决策树

```
GPU 利用率低吗？
├── 是 → Kernel launch 开销或并行度不足
│         尝试：更大问题规模、persistent kernel
└── 否 → 检查内存带宽利用率
    ├── 高（>60%）→ 内存受限
    │   尝试：合并访问、向量化 load、L2 优化
    └── 低 → 检查 MFMA/VALU 利用率
        ├── 高 → 计算受限
        │   尝试：混合精度、算法改进
        └── 低 → 可能在依赖上停顿
            尝试：ILP、软件流水线、s_waitcnt 调优
```
```

- [ ] **步骤 3: 编写 hipcc-compilation.md**

```markdown
# hipcc 编译指南

## 基本命令

```bash
# 单目标
hipcc -O3 --offload-arch=gfx942 -o kernel.so -shared -fPIC kernel.cpp

# 多目标
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 -o kernel.so -shared -fPIC kernel.cpp

# 配合 PyTorch
hipcc -O3 --offload-arch=gfx942 \
  $(python3 -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")  \
  -shared -fPIC kernel.cpp -o kernel.so

# 保存中间文件（用于 ISA 检查）
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# 查找包含 ISA 的 *.s 文件
```

## 常用标志

| 标志 | 用途 |
|------|------|
| `-O3` | 完整优化（必须使用） |
| `--offload-arch=gfxNNN` | 目标 GPU 架构（必需） |
| `-shared -fPIC` | 构建共享库供 Python 加载 |
| `-save-temps` | 保留中间 .s（ISA）文件 |
| `-Rpass=inline` | 显示内联决策 |
| `-ffast-math` | 激进 FP 优化（可能影响精度） |
| `-munsafe-fp-atomics` | 更快的原子 FP 操作（极少情况损失精度） |

## 常见错误与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `error: unknown target CPU 'gfx942'` | ROCm 版本过旧 | 更新 ROCm 或用 `rocminfo` 检查正确 arch |
| `undefined reference to __hip_*` | 缺少 HIP runtime 链接 | 添加 `-lhip_hcc` 或用 `hipcc` 代替 `g++` |
| `error: use of undeclared identifier '__shfl_sync'` | CUDA API 在 HIP 中不可用 | 使用 `__builtin_amdgcn_ds_swizzle` 或 `__shfl` |
| `error: too few register available` | VGPR 过多 | 添加 `__launch_bounds__`，减少活跃变量 |
| Kernel 运行但结果错误 | `--offload-arch` 不匹配 | 验证 arch 与 `rocminfo` 输出一致 |
| 性能差，未使用 `-O3` | Debug 构建 | 始终使用 `-O3` 编译 |

## PyTorch Extension 构建

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

- [ ] **步骤 4: 编写 triton-rocm-quirks.md**

```markdown
# Triton ROCm 后端 — 已知差异与注意事项

## 不支持的特性

| 特性 | 状态 | 替代方案 |
|------|------|---------|
| `tl.inline_asm_elementwise` | 不支持 | 使用纯 Triton 操作 |
| 部分 `tl.extra.cuda` 函数 | 不可用 | 使用前检查可用性 |
| `tl.tensor` 特定布局 | 可能有差异 | 显式测试 |

## Autotune 差异

- `num_stages`：ROCm 后端可能忽略或处理方式不同于 CUDA
- 配置空间应包含 `num_stages=1` 作为回退
- `num_warps`：在 AMD 上，warps = wavefronts（每个 64 线程，不是 32）

## Block Size 指南

- 使用 64 的倍数（wavefront 大小对齐）
- 典型良好值：64, 128, 256, 512, 1024
- MI300X 上 128 和 256 通常是最佳选择

## 性能注意事项

- Triton ROCm 可能生成与 CUDA 后端不同质量的代码
- 始终使用 `rocprof --stats` profile 验证实际性能
- 如果 Triton 性能达到平台期，考虑切换到 HIP C++ 获得更多控制

## 调试

```bash
# 设置调试标志
TRITON_DEBUG=1 python kernel.py

# 检查生成的代码
TRITON_CACHE_DIR=/tmp/triton_cache python kernel.py
# 检查 /tmp/triton_cache/ 中生成的 IR 和 ASM
```

## 常见陷阱

1. 假设 CUDA Triton 性能可迁移到 ROCm — 必须重新 benchmark
2. 不正确使用 `tl.constexpr` — 必须用于编译时常量
3. 未测试 `num_stages=1` — AMD 上有时更少 stage 更快
4. Grid size 对于 192 CU 太小 — 需要足够的 block 来填满 MI300X
```

- [ ] **步骤 5: 提交**

```bash
git add amd-kernel-skill/references/rocprof-guide.md
git add amd-kernel-skill/references/omniperf-guide.md
git add amd-kernel-skill/references/hipcc-compilation.md
git add amd-kernel-skill/references/triton-rocm-quirks.md
git commit -m "feat: add toolchain reference documents (rocprof, omniperf, hipcc, triton-rocm)"
```

---

### 任务 11: 库与 API 参考文档

**文件:**
- 创建: `amd-kernel-skill/references/ck-programming-model.md`
- 创建: `amd-kernel-skill/references/ck-tile-tuning.md`
- 创建: `amd-kernel-skill/references/aiter-ops-reference.md`
- 创建: `amd-kernel-skill/references/hip-intrinsics.md`

- [ ] **步骤 1: 编写 ck-programming-model.md**

```markdown
# Composable Kernel (CK) 编程模型

## 三级抽象

### 第一级：TilePartitioner
将问题分配到各个 thread block（workgroup）。

- 将 M、N 维度划分为 block tile
- 每个 thread block 拥有一个 (block_m × block_n) 输出 tile
- Grid size = (M/block_m) × (N/block_n)

### 第二级：TileScheduler
将 tile 分配到 CU 并管理执行顺序。

| 调度器 | 行为 | 适用场景 |
|--------|------|---------|
| Static | 固定 tile 到 CU 的映射 | 简单 kernel |
| Persistent | 单次 launch，通过 atomicAdd 计数器分配 tile | 小 tile、多次迭代 |
| Dynamic | CU 之间工作窃取 | 不规则负载（causal mask） |

### 第三级：TilePipeline
管理 thread block 内的数据流：global → LDS → 寄存器 → 计算 → 写回。

- 控制预取深度（单缓冲、双缓冲、三缓冲）
- 管理 A 和 B fragment 的 LDS 分配
- 调度 MFMA 指令与内存操作

## 使用 CK 模板

```cpp
// 1. 定义问题
using Problem = ck::tensor_operation::device::DeviceGemmXdl<...>;

// 2. 配置 tile 大小
using Config = ck::TileConfig<
    BlockM, BlockN, BlockK,
    WarpM, WarpN, WarpK,
    PipelineDepth
>;

// 3. 实例化并运行
auto gemm = Problem{};
gemm.Run(args, stream);
```

## 扩展 CK

构建新的融合 kernel：
1. 识别匹配子操作的已有 CK 原语
2. 在新的 pipeline 中组合它们
3. 处理融合点（输出直接作为输入，不经过 global memory）
4. 为融合 kernel 的特定数据流调优 tile 大小
```

- [ ] **步骤 2: 编写 ck-tile-tuning.md**

```markdown
# CK Tile Size 调优指南

## 参数说明

| 参数 | 含义 | 典型范围 |
|------|------|---------|
| block_m | 每 block 的 M 维度 tile | 64-256 |
| block_n | 每 block 的 N 维度 tile | 64-256 |
| block_k | K 维度 tile（归约） | 32-128 |
| warp_m | 每 warp 的 M 维度 | 16-64 |
| warp_n | 每 warp 的 N 维度 | 16-64 |
| pipeline_depth | 预取阶段数 | 1-3 |

## MI300X (GFX942) 推荐配置

### BF16 GEMM

| M 范围 | N 范围 | block_m | block_n | block_k | 说明 |
|--------|--------|---------|---------|---------|------|
| ≥ 2048 | ≥ 2048 | 256 | 128 | 64 | 标准大 GEMM |
| ≥ 2048 | ≥ 2048 | 128 | 256 | 64 | 备选（两者都试） |
| < 1024 | ≥ 2048 | 64 | 256 | 64 | 窄高型 |
| ≥ 2048 | < 1024 | 256 | 64 | 64 | 宽矮型 |
| < 512 | < 512 | 64 | 64 | 32 | 小 GEMM |

### FP8 GEMM

| 配置 | block_m | block_n | block_k |
|------|---------|---------|---------|
| 默认 | 256 | 128 | 128 |
| 备选 | 128 | 256 | 128 |

### Attention (FMHA)

| 配置 | block_m | block_n | head_dim |
|------|---------|---------|----------|
| 默认 | 128 | 128 | 64-128 |
| 长序列 | 64 | 256 | 64-128 |

## MI355X (GFX950) 推荐配置

待定 — 在 MI355X 硬件上初始 benchmark 后填写。
由于 CU 数量和带宽不同，预期最优 tile 大小会有差异。

## 调优策略

1. 从上表推荐配置开始
2. 运行 benchmark，记录性能
3. 网格搜索：block_m 和 block_n 各 ±64，block_k ±32
4. 每个配置检查：
   - LDS 使用量 ≤ 64KB：(block_m × block_k + block_k × block_n) × elem_size × pipeline_depth
   - 足够的 block 填满所有 CU（MI300X 需 grid_size ≥ 192）
   - VGPR 使用量不导致溢出
5. 将最佳配置记录到本文件供未来参考
```

- [ ] **步骤 3: 编写 aiter-ops-reference.md**

```markdown
# AITER（AMD Inference Toolkit for Efficient Runtime）参考

## 概述

AITER 提供预优化的 AMD GPU kernel，覆盖常见推理操作。
编写自定义 kernel 前，先检查 AITER 是否已有高性能版本。

## 可用操作

| 类别 | 操作 | 说明 |
|------|------|------|
| Attention | Flash Attention (fwd/bwd), PagedAttention | 基于 CK，高度调优 |
| GEMM | BF16/FP16/FP8/INT8 GEMM 变体 | 多种 tile 配置 |
| 归一化 | RMSNorm, LayerNorm | 融合版本 |
| 激活函数 | SiLU, GeLU, SwiGLU | 与 linear 融合 |
| 量化 | FP8 量化/反量化 | per-tensor 和 per-token |
| MoE | MoE GEMM, expert routing | 为 MI300X 优化 |

## API 使用

```python
import aiter

# 示例：Flash Attention
output = aiter.flash_attention(q, k, v, causal=True)

# 示例：融合 RMSNorm
output = aiter.rms_norm(input, weight, eps=1e-6)

# 示例：GEMM
output = aiter.gemm(a, b, dtype=torch.bfloat16)
```

## 何时使用 AITER vs 自定义 Kernel

| 场景 | 使用 |
|------|------|
| 标准算子、标准 shape | AITER（已优化） |
| AITER 中没有的融合算子 | 自定义 kernel |
| AITER 性能不够 | 自定义 kernel（以 AITER 为 baseline） |
| 非标准精度组合 | 自定义 kernel |

## 对比 AITER Benchmark

```bash
python scripts/benchmark_kernel.py --kernel my_kernel.py --op attention --baseline aiter
```
```

- [ ] **步骤 4: 编写 hip-intrinsics.md**

```markdown
# AMD GPU HIP Intrinsics 参考

## 跨 Lane 操作

| Intrinsic | 用途 | AMD ISA |
|-----------|------|---------|
| `__builtin_amdgcn_readfirstlane(val)` | 将 lane 0 广播到所有 lane（返回标量） | v_readfirstlane_b32 |
| `__builtin_amdgcn_readlane(val, lane)` | 读取指定 lane 的值 | v_readlane_b32 |
| `__builtin_amdgcn_ds_swizzle(val, pattern)` | Lane 排列（无 LDS 流量） | ds_swizzle_b32 |
| `__builtin_amdgcn_mov_dpp(val, ctrl, row_mask, bank_mask, bound_ctrl)` | 数据并行原语 | v_mov_b32 dpp |
| `__shfl(val, lane)` | 跨 lane shuffle（HIP 兼容） | 实现因情况而异 |
| `__shfl_xor(val, mask)` | 基于 XOR 的 shuffle | 实现因情况而异 |

## Wavefront Reduction

```cpp
// 64 lane wavefront 求和归约
float wf_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// 注意：需要 6 步（64 lane），不是 5 步（NVIDIA 32 lane）
```

## 内存 Intrinsics

| Intrinsic | 用途 |
|-----------|------|
| `__builtin_nontemporal_load(ptr)` | 绕过 cache 的 load |
| `__builtin_nontemporal_store(val, ptr)` | 绕过 cache 的 store |
| `__builtin_amdgcn_s_waitcnt(val)` | 细粒度内存屏障 |

## 数学 Intrinsics

| Intrinsic | 用途 | 精度 |
|-----------|------|------|
| `__builtin_amdgcn_rcpf(x)` | 快速倒数 | ~1 ULP |
| `__builtin_amdgcn_rsqf(x)` | 快速 rsqrt | ~1 ULP |
| `__builtin_amdgcn_exp2f(x)` | 快速 exp2 | ~1 ULP |
| `__builtin_amdgcn_log2f(x)` | 快速 log2 | ~1 ULP |

## AMD vs NVIDIA Intrinsics

| NVIDIA | AMD 等价物 |
|--------|-----------|
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)`（无需 mask，全 wavefront） |
| `__syncwarp()` | 不需要（wavefront 是锁步的） |
| `__ballot_sync(mask, pred)` | `__ballot(pred)` 返回 64 位 |
| `atomicAdd(ptr, val)` for FP16 | `__builtin_amdgcn_flat_atomic_fadd_f16` |
```

- [ ] **步骤 5: 提交**

```bash
git add amd-kernel-skill/references/ck-programming-model.md
git add amd-kernel-skill/references/ck-tile-tuning.md
git add amd-kernel-skill/references/aiter-ops-reference.md
git add amd-kernel-skill/references/hip-intrinsics.md
git commit -m "feat: add library and API reference documents (CK, AITER, HIP intrinsics)"
```

---

### 任务 12: 优化模式文档 + AMD vs NVIDIA 速查表

**文件:**
- 创建: `amd-kernel-skill/references/optimization-patterns.md`
- 创建: `amd-kernel-skill/references/advanced-optimization.md`
- 创建: `amd-kernel-skill/references/common-mistakes.md`
- 创建: `amd-kernel-skill/references/kernel-recipes.md`
- 创建: `amd-kernel-skill/references/amd-vs-nvidia-cheatsheet.md`

- [ ] **步骤 1: 编写 optimization-patterns.md**

```markdown
# 通用优化模式（早期与中期阶段）

## 内存优化

### 1. 合并 Global Memory 访问
- 确保连续线程访问连续地址
- Wavefront（64 线程）应访问连续的 256B 区域
- Stride-1 访问模式是理想的

### 2. 向量化 Load
- 使用 `float4` / `dwordx4` 进行 128 位 load（4 倍带宽效率）
- 数据对齐到 16 字节边界
- Triton：正确设置 BLOCK_SIZE 后自动处理

### 3. LDS 使用
- 用 LDS 实现 thread block 内的数据复用
- 预算：每 CU 64KB（CU 上所有 block 共享）
- 每 block 使用更多 LDS → 更少并发 block → 更低 occupancy

## 计算优化

### 4. Kernel 融合
- 将逐元素操作与前后的 GEMM/reduction 融合
- 节省 global memory 往返
- 常见融合：Linear+Activation、Norm+Scale、Attention+Softmax

### 5. Tiling
- 将大问题分解为适合 LDS 的 tile
- Tile 大小 = 数据复用（越大）与 occupancy（越小）的平衡
- CK 特定 tile 推荐见 `ck-tile-tuning.md`

### 6. 循环展开
- HIP 用 `#pragma unroll N`，Triton 自动处理
- 甜蜜点：足够填满流水线，但不至于寄存器溢出
- 展开后检查 VGPR 数量

## Launch 优化

### 7. Grid 大小设定
- MI300X：192 CU。至少需要 192 个 block 才能充分利用
- 更多 block（2-4 倍 CU 数）有助于隐藏每 block 的变异
- 非常小的 kernel：考虑批处理或 persistent kernel

### 8. Thread Block 大小
- 默认：256（每 block 4 个 wavefront）— 良好通用选择
- 最小：64（1 个 wavefront）— 寄存器重度 kernel
- 最大：1024（16 个 wavefront）— 高 occupancy 需求

## 反模式（标注硬件相关性）

| 反模式 | 影响 | 硬件说明 |
|--------|------|---------|
| Stride-N 访问 | 带宽浪费 | 所有 AMD GPU |
| LDS > 32KB/block | MI300X 上限制为 2 block/CU | MI300X: 64KB/CU |
| 发散代码中使用 `__syncthreads` | 死锁风险 | 所有 AMD GPU |
| 假设 warp=32 | 错误的 reduction、错误的 shuffle | AMD wavefront=64 |
```

- [ ] **步骤 2: 编写 advanced-optimization.md**

此文件是最长的参考文档。完整编写设计规格书第 4.3 节"极限优化技术文档"中的全部 8 种技术和选择指南表。内容已在设计规格书第 318-449 行中详细定义，按该结构完整实现。

```markdown
# 极限优化：突破性能平台期

## 使用时机

当 kernel 已通过正确性验证、已超过 torch.compile baseline，但与理论峰值仍有差距（带宽利用率 <70% 或 MFMA 利用率 <80%）时。按以下顺序尝试。

## 1. 软件流水线与多级缓冲

**原理**：当前迭代的计算与下一迭代的数据加载重叠执行。

**技术**：
- 双缓冲：LDS 分两半，交替加载和计算
- 三级流水线：加载 N+1、计算 N、写回 N-1
- 异步拷贝：DMA 引擎搬运数据与 MFMA 完全并发

**AMD 实现要点**：
- 需要精确的 `s_waitcnt lgkmcnt/vmcnt`（参见 `isa/scheduling-pipeline.md`）
- LDS 翻倍：检查 64KB/CU 上限
- 示例：当前 MFMA 执行时预取下一个 tile

**预期收益**：内存受限 kernel 通常 10-30%

## 2. Wavefront 特化

**原理**：同一 block 内不同 wavefront 承担不同角色，通过 barrier 协调。

**角色**：计算 wavefront（MFMA）、数据搬运（global→LDS）、归约（softmax/reduction）

**AMD 实现要点**：
- wavefront=64 → 每个 wavefront 更重，特化收益更大
- 角色分配：`threadIdx.x / 64` 给出 wavefront ID
- 通过 `__syncthreads()` 或 LDS fence 协调

**预期收益**：多阶段 kernel（Attention）可获 5-15%

## 3. 数据布局与 Swizzle

**原理**：改变数据排列以消除 bank conflict 和提升合并访问率。

**技术**：
- LDS padding：每行末尾加 padding 避免 32-bank conflict（AMD：32 bank × 4B）
- `ds_swizzle`：硬件 lane 排列，无需 LDS 读写
- SOA 布局用于向量化 load（`buffer_load_dwordx4`）

**AMD 实现要点**：
- AMD LDS bank 规则与 NVIDIA 不同 — 重新计算 padding
- 用 `omniperf` LDS bank conflict 指标验证

**预期收益**：存在 bank conflict 时 5-20%

## 4. Occupancy vs ILP 权衡

**原理**：有时更少 wavefront 配合更多寄存器优于大量 wavefront 配合溢出。

**技术**：
- `__launch_bounds__(threads, minBlocks)` 控制寄存器分配
- 检测溢出：`omniperf` → ScratchWaveslifetimeVGPR > 0
- 跨 wavefront group 的寄存器重平衡（参考 AVO v33）

**AMD 实现要点**：
- MI300X：65536 VGPR/CU。参见 `isa/register-allocation.md` 中的 occupancy 表
- 高 occupancy 和低 occupancy 配置都 profile — 更快的获胜

**预期收益**：消除溢出时 3-10%

## 5. Persistent Kernel 与 Tile 调度

**原理**：只 launch 一次，通过原子计数器自行分配 tile。消除重复 launch 开销 + L2 友好的遍历顺序。

**技术**：
- `atomicAdd` 全局 tile 计数器
- Swizzled 遍历（L 形、Z 形、Hilbert 曲线）提升 L2 复用
- 跨 tile 负载均衡用于不规则形状（causal mask）

**AMD 实现要点**：
- MI300X L2 较大（256MB）— tile 遍历顺序影响显著
- CK 的 TileScheduler 可作为参考实现

**预期收益**：多次 launch 场景 5-15%；L2 优化通常 3-8%

## 6. 混合精度策略

**原理**：超越"全用 BF16"，在不同计算阶段使用不同精度。

**技术**：
- 输入 FP8/BF16 → MFMA 计算 → FP32 累积 → BF16 输出
- 关键中间结果（如 softmax max/sum）保持 FP32
- 利用 MFMA 的混合精度能力（FP16 输入、FP32 输出）

**AMD 实现要点**：
- 查阅 `isa/mfma-instructions.md` 了解可用精度组合
- MI355X/CDNA4 可能新增精度格式 — 查阅对应硬件文档

**预期收益**：计算受限 kernel 可获 10-30% TFLOPS 提升

## 7. 编译器对抗与引导

**原理**：在编译器自动优化不足或过度时手动干预。

**技术**：
- `#pragma unroll N`：精确控制展开
- `__launch_bounds__`：引导寄存器分配
- `volatile` / `__builtin_nontemporal_*`：绕过 cache / 阻止重排序
- 内联汇编：最后手段（参见 `isa/inline-asm-patterns.md`）

**AMD 实现要点**：
- `hipcc -save-temps` 可查看生成的 ISA，验证编译器行为

**预期收益**：因情况而异，通常 2-10%

## 8. L2 Cache 全局优化

**原理**：全局层面的数据复用策略，跨 tile / 跨 kernel 最大化 L2 命中率。

**技术**：
- Tile 遍历顺序（GEMM 的 K 维度尤其敏感）
- 跨 kernel fusion 保持数据在 L2 中
- L2 cache residency 控制（如硬件支持 prefetch hint）

**AMD 实现要点**：
- MI300X L2 = 256MB，相对较大
- 用 `rocprof` TCC_HIT/TCC_MISS 计数器测量

**预期收益**：内存受限 kernel 通常 3-8%

## 选择指南

| 瓶颈类型 | 诊断指标 | 优先技术 |
|----------|---------|---------|
| HBM 带宽受限 | 带宽利用率 >80% | 软件流水线、L2 优化、数据布局 |
| LDS 受限 | Bank conflict 率高 | Swizzle、padding、数据布局 |
| 计算受限 | MFMA 利用率 <70% | Wavefront 特化、混合精度、ILP |
| 寄存器溢出 | ScratchWaves > 0 | Occupancy 调优、寄存器重平衡 |
| Launch 开销 | 多次小 kernel launch | Persistent kernel |
| 编译器问题 | ISA 审查发现冗余指令 | 编译器引导、内联汇编 |
```

- [ ] **步骤 3: 编写 common-mistakes.md**

```markdown
# AMD 特有常见错误

## 编译

| 错误 | 症状 | 修复 |
|------|------|------|
| 缺少 `--offload-arch` | 编译目标错误 | 始终指定 `--offload-arch=gfx942`（或 `gfx950`） |
| 缺少 `-O3` | 比预期慢 5-10 倍 | 始终使用 `-O3` |
| 直接使用 CUDA API | 编译错误 | 替换为 HIP 等价物（参见 `amd-vs-nvidia-cheatsheet.md`） |

## 架构

| 错误 | 症状 | 修复 |
|------|------|------|
| 假设 warp = 32 | 错误的 reduction 结果、性能退化 | AMD wavefront = 64。使用 6 步 shuffle，不是 5 步 |
| 假设 shared mem = 48KB | LDS 溢出 | AMD LDS = 64KB/CU（按硬件查证） |
| 使用 `__syncwarp()` | 不必要的同步 | AMD wavefront 是锁步的，无需部分同步 |
| 错误的 bank conflict 计算 | 意外的 LDS 争用 | AMD：32 bank × 4B（conflict pattern 与 NVIDIA 不同） |

## 性能

| 错误 | 症状 | 修复 |
|------|------|------|
| Thread block 太少 | GPU 利用率低 | MI300X 至少需要 192 个 block（每 CU 至少一个） |
| 忽略寄存器溢出 | 不明原因的性能下降 | 检查 `omniperf` ScratchWaves，使用 `__launch_bounds__` |
| 直接复制 CUDA 调优参数 | 次优性能 | 为 AMD 重新调优 block size、展开因子 |
| 未使用 AGPR 做 MFMA 累加器 | 更高 VGPR 压力 | 使用 AGPR 累加器释放 VGPR |

## Triton 特有

| 错误 | 症状 | 修复 |
|------|------|------|
| 使用 `tl.inline_asm_elementwise` | ROCm 上报错 | 使用纯 Triton 操作 |
| BLOCK_SIZE 不是 64 的倍数 | 浪费 lane | 使用 64, 128, 256, 512, 1024 |
| 假设 CUDA Triton 性能可迁移 | 失望 | 始终在 AMD 上 benchmark |

## 知识库

本文件是活文档。在迭代过程中发现新错误时回填。
```

- [ ] **步骤 4: 编写 kernel-recipes.md**

```markdown
# Kernel 参考实现骨架

常见算子的参考实现。作为起点使用，不是最终方案。

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

## 融合 SwiGLU (Triton)

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

## Vector Add（HIP C++ — 最小示例）

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

## Reduction（HIP C++ — Wavefront 感知）

```cpp
__device__ float warp_reduce_sum(float val) {
    // AMD wavefront = 64，需要 6 步
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void block_reduce(const float* input, float* output, int N) {
    __shared__ float shared[16];  // 最多 16 个 wavefront/block（1024 线程）
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

## 说明

- 这些是起点。始终 profile 和优化。
- 根据目标硬件调整 block size 和数据类型。
- CK GEMM 配置参见 `ck-tile-tuning.md`。
- 在迭代中开发出新实现时回填本文件。
```

- [ ] **步骤 5: 编写 amd-vs-nvidia-cheatsheet.md**

```markdown
# AMD vs NVIDIA — 关键差异速查表

## 术语对照

| NVIDIA | AMD | 说明 |
|--------|-----|------|
| Warp（32 线程） | Wavefront（64 线程） | 2 倍宽 — 影响 reduction、shuffle |
| SM | CU（Compute Unit） | 类似概念 |
| Shared Memory | LDS（Local Data Share） | MI300X 上 64KB/CU |
| CUDA Core | Stream Processor | — |
| Tensor Core | Matrix Core (MFMA) | 不同 ISA，不同寄存器布局 |
| PTX | AMDGPU ISA | 中间 vs 最终 ISA |
| nvcc | hipcc | — |
| ncu / nsys | rocprof / omniperf | — |
| cuDNN | MIOpen | — |
| CUTLASS | CK (Composable Kernel) | — |

## API 映射

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__syncwarp()` | 不需要（wavefront 是锁步的） |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` |
| `__ballot_sync(mask, pred)` | `__ballot(pred)`（返回 64 位） |
| `__shared__` | `__shared__`（相同） |
| `blockDim.x` | `blockDim.x`（相同） |

## 关键行为差异

| 方面 | NVIDIA | AMD |
|------|--------|-----|
| Warp/Wavefront 大小 | 32 | 64 |
| Reduction 的 shuffle 步数 | 5 | 6 |
| 需要 `__syncwarp()`？ | 是（独立调度） | 否（锁步） |
| Shared memory bank | 32 × 4B | 32 × 4B（但 conflict pattern 不同） |
| L2 cache 大小 | 40-50 MB (A100/H100) | 256 MB (MI300X) |
| Tensor core 输入 | HMMA（warp 级） | MFMA（wavefront 级） |
| Occupancy 计算器 | CUDA occ calculator | `rocminfo` + 手动计算 |
| 内联汇编 | PTX asm | AMDGPU ISA asm |
| 编译目标标志 | `-arch=sm_80` | `--offload-arch=gfx942` |

## 迁移清单

1. 将 CUDA API 替换为 HIP 等价物（`cuda` → `hip`）
2. 修改 warp size 假设：32 → 64
3. 更新 reduction 循环：5 步 → 6 步
4. 删除 `__syncwarp()` 调用
5. 从 shuffle/ballot 中移除 `mask` 参数
6. 将 `__ballot_sync` 返回值从 32 位改为 64 位
7. 更新编译：`nvcc` → `hipcc`，`-arch=sm_XX` → `--offload-arch=gfxYYY`
8. 重新调优 block size 和展开因子
9. 重新 benchmark 所有内容 — 不要假设 CUDA 性能可迁移
```

- [ ] **步骤 6: 提交**

```bash
git add amd-kernel-skill/references/optimization-patterns.md
git add amd-kernel-skill/references/advanced-optimization.md
git add amd-kernel-skill/references/common-mistakes.md
git add amd-kernel-skill/references/kernel-recipes.md
git add amd-kernel-skill/references/amd-vs-nvidia-cheatsheet.md
git commit -m "feat: add optimization patterns, advanced optimization, common mistakes, recipes, and AMD vs NVIDIA cheatsheet"
```

---

### 任务 13: 集成验证

**文件:**
- 验证: `amd-kernel-skill/` 中所有文件存在且格式正确

- [ ] **步骤 1: 验证文件数量和结构**

```bash
find amd-kernel-skill -type f | sort
```

预期：共 31 个文件：
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

- [ ] **步骤 2: 验证主 SKILL.md 引用的文件都存在**

检查主 SKILL.md 和子 Skill 中引用的每个文件是否实际存在：

```bash
# 从 SKILL.md 文件中提取引用路径
grep -rh "references/" amd-kernel-skill/SKILL.md amd-kernel-skill/skills/*/SKILL.md | \
  grep -oP 'references/[a-z0-9/_-]+\.md' | sort -u | \
  while read f; do
    [ -f "amd-kernel-skill/$f" ] && echo "[OK] $f" || echo "[MISSING] $f"
  done
```

预期：全部 OK，无 MISSING 条目。

- [ ] **步骤 3: 验证脚本语法正确**

```bash
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/scripts/verify_correctness.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/scripts/benchmark_kernel.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/templates/triton_kernel_template.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('amd-kernel-skill/templates/benchmark_template.py', doraise=True)"
```

预期：无错误。

- [ ] **步骤 4: 最终提交并打标签**

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

- [ ] **步骤 5: 复制到 Claude Code skills 目录（用于测试）**

```bash
# Claude Code
cp -r amd-kernel-skill/ /path/to/project/.claude/skills/amd-kernel-skill/

# Cursor
cp -r amd-kernel-skill/ /path/to/project/.cursor/skills/amd-kernel-skill/
```

将 `/path/to/project/` 替换为实际的 Primus-Turbo 项目路径。
