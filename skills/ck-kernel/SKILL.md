# Composable Kernel (CK) Optimization (AMD)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
rocprofv3 --stats -- ./ck_benchmark --op {OP}
```

Understand current CK template configuration and identify tuning opportunities.

## Step 2: Implement / Configure Kernel

Refer to `references/libraries/ck-programming-model.md` for the CK-Tile pipeline hierarchy:

```
TileGemmShape → TilePartitioner → TileGemmTraits → GemmPipelineProblem → Pipeline → GemmKernel
```

### Key Abstractions (from actual CK codebase):
- **TilePartitioner**: 2D / 1D / SpatiallyLocal (with RemapXCD for multi-die MI300X cache optimization)
- **GemmPipelineScheduler**: Default / Intrawave (compute-bound) / Interwave (memory-bound)
- **Pipeline**: MEMORY / COMPUTE_V3-V6 / COMPUTE_ASYNC / PRESHUFFLE_V2

### Pipeline Selection Guide:
| Workload | Pipeline | Scheduler | Why |
|----------|----------|-----------|-----|
| Memory-bound GEMM | MEMORY | Interwave | Deep prefetch (32KB in-flight), 2-8 stages |
| Compute-bound GEMM | COMPUTE_V3 | Intrawave | 2-stage pipeline, tight loops |
| Large GEMM + LDS ping-pong | COMPUTE_V4 | Intrawave | Double SMEM buffer |
| FP4 inference | COMPUTE_ASYNC | Default | Large K_Warp_Tile=128, async |
| Weight-preshuffle inference | PRESHUFFLE_V2 | Default | Pre-shuffled weights for decode/prefill |

Key tuning parameters: `M_Tile`, `N_Tile`, `K_Tile`, warp layout — consult `references/libraries/ck-tile-tuning.md` for real configurations from the CK codebase.

For new fused kernels: compose from existing CK primitives rather than writing from scratch.
For FMHA: hdim 64 and 128 are best tuned, with CppConstraint for CU-aware dispatch.

## Step 3: Verify

```bash
python scripts/verify_correctness.py --kernel {kernel_path} --op {OP} --dtype bf16
python scripts/benchmark_kernel.py --kernel {kernel_path} --op {OP} --baseline torch_compile
```

Three gates: compile → correctness → performance.

## Step 4: Iterate

- Each round: save to `agent_output/<OP>/ck/round-N/`
- Update `performance_trend.md`
- Tile search strategy: start with recommended configs from `references/libraries/ck-tile-tuning.md`, then grid search ±64 on M/N, ±32 on K
- Pipeline switching: if COMPUTE_V3 plateaus, try MEMORY pipeline or COMPUTE_V4 (LDS ping-pong)
- On failure: retry 3+ times
- On plateau: try different partitioner (SpatiallyLocal with RemapXCD for MI300X L2 optimization)
- For FMHA: check `references/libraries/ck-tile-tuning.md` for hdim-specific block sizes from codegen tuning tables
- Target: find optimal tile config, assemble new fused kernels

## Step 5: Knowledge Accumulation

- Success → design doc + backfill optimal configs to `references/libraries/ck-tile-tuning.md`
- Failure → postmortem + backfill to `references/optimization/common-mistakes.md`
- Mandatory step, never skip

## Quick Reference: Key Documents

| When | Read |
|------|------|
| Every profiling round | `references/toolchain/profiling-decision-tree.md` — mechanical bottleneck classification |
| Understanding CK architecture | `references/libraries/ck-programming-model.md` — pipeline hierarchy + API |
| Choosing tile sizes | `references/libraries/ck-tile-tuning.md` — real GemmConfig tables from CK codebase |
| MFMA instruction details | `references/isa/mfma-instructions.md` — WarpGemm maps to these |
| Hardware constraints | `references/hardware/mi300x.md` or `mi355x.md` — CU count, LDS, L2 |
| SOTA code patterns | `references/optimization/kernel-recipes.md` — buffer_load_lds, swizzle, ping-pong |
| Alternative: AITER | `references/libraries/aiter-ops-reference.md` — production-tuned CK-based ops |
| Advanced techniques | `references/optimization/advanced-optimization.md` — software pipelining, swizzle |
| GEMM library tuning | `references/libraries/gemm-tuning-guide.md` — maybe library tuning is better |
