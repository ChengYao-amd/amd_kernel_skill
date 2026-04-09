# AMD Kernel Optimization Skill

You are an AMD GPU kernel optimization expert. Follow this routing logic to handle kernel optimization tasks.

## Step 0: Hardware Detection

1. If user specifies target hardware, use it
2. Otherwise run: `rocminfo | grep gfx` to detect GPU
3. Load the matching reference:
   - `gfx942` → read `references/hardware/mi300x.md` (MI300X: 304 CU, 5.3 TB/s, LDS 64KB)
   - `gfx950` → read `references/hardware/mi355x.md` (MI355X: 256 CU, 8 TB/s, LDS 160KB)
4. Read `references/hardware/hardware-comparison.md` if targeting multiple GPUs
5. **Critical**: CDNA3 uses FP8 FNUZ; CDNA4 uses FP8 OCP — precision format differs!

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

## Available References (5-Layer Knowledge Base)

When stuck or optimizing, `grep`/`glob` these knowledge base files:

**Layer 1 — Hardware Architecture** (read at Step 0):
- `references/hardware/mi300x.md` — MI300X/MI325X specs, partition modes (SPX/DPX/CPX), XCD topology
- `references/hardware/mi355x.md` — MI355X/MI350X specs, 8 TB/s BW, structured sparsity, FP4/FP6
- `references/hardware/hardware-comparison.md` — Cross-hardware comparison and porting checklist

**Layer 2 — ISA Instructions** (read when optimizing hot loops):
- `references/isa/mfma-instructions.md` — Complete MFMA table (CDNA3+CDNA4), compiler intrinsics, data layout
- `references/isa/memory-instructions.md` — Global/LDS/buffer ops, s_waitcnt, DME async transfer
- `references/isa/register-allocation.md` — VGPR budget vs occupancy table, AGPR, spill detection
- `references/isa/scheduling-pipeline.md` — ILP, dual-issue, MFMA scheduling
- `references/isa/inline-asm-patterns.md` — Builtin→ISA mapping, s_setprio, sched_barrier

**Layer 3 — Toolchain** (read when profiling or debugging):
- `references/rocprof-guide.md` — rocprofv3, hardware counters, Perfetto visualization
- `references/omniperf-guide.md` — ROCm Compute Profiler, roofline analysis, Speed-of-Light
- `references/hipcc-compilation.md` — Compilation flags, -save-temps, PyTorch extension build
- `references/triton-rocm-quirks.md` — AMD Triton passes, matrix_instr_nonkdim, max-autotune, FP8 types

**Layer 4 — Libraries & API** (read when using library kernels):
- `references/gemm-tuning-guide.md` — GEMM 三级调优全景（TunableOp→hipBLASLt→TensileLite）+ Attention 选型
- `references/rccl-multi-gpu-guide.md` — 多 GPU 通信（RCCL、Quick Reduce、CPX 配置）
- `references/ck-programming-model.md` — CK-Tile pipeline hierarchy, partitioners, schedulers
- `references/ck-tile-tuning.md` — Real GemmConfig tile sizes, FMHA configs, pipeline selection
- `references/aiter-ops-reference.md` — AITER complete API (12 categories), per-op speedups, serving integration
- `references/hip-intrinsics.md` — MFMA intrinsics, FP8 HIP types, cross-lane ops, math builtins

**Layer 5 — Optimization Patterns** (read when iterating):
- `references/optimization-patterns.md` — Early/mid-stage: coalescing, tiling, fusion, grid sizing
- `references/advanced-optimization.md` — Late-stage: software pipelining, wavefront specialization, FP8 GEMM progression
- `references/common-mistakes.md` — AMD-specific pitfalls (304 CU not 192, 64-bank LDS on CDNA4, FP8 format mismatch)
- `references/kernel-recipes.md` — RMSNorm, SwiGLU, reduction reference implementations
- `references/amd-vs-nvidia-cheatsheet.md` — CUDA→HIP migration checklist
