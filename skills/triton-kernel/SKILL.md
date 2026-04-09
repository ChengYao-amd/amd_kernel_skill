# Triton Kernel Optimization (AMD ROCm)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
rocprofv3 --stats -- python run_kernel.py
```

Identify bottleneck type: memory-bound / compute-bound / launch-overhead.
For deeper analysis: `rocprof-compute profile -n {OP}_baseline -- python run_kernel.py`
Follow `references/toolchain/profiling-decision-tree.md` to classify bottleneck mechanically.

## Step 2: Implement Kernel

Start from `templates/triton_kernel_template.py`. Key rules:

- `BLOCK_SIZE`: use multiples of 64 (wavefront alignment). MI300X sweet spot: 128 or 256
- Use `tl.constexpr` for compile-time constants
- `triton.autotune` config space: include `num_warps` in [4, 8, 16], `num_stages` in [1, 2, 3], and `matrix_instr_nonkdim` in [16, 32] for MFMA tile selection
- ROCm backend limitations: check `references/toolchain/triton-rocm-quirks.md`
  - No `tl.inline_asm_elementwise`
  - FP8 types: gfx942 uses FNUZ, gfx950 uses OCP — see `references/libraries/hip-intrinsics.md`
- **AMD Triton compiler passes** (auto-applied, but useful to know for debugging):
  - AccelerateMatmul: selects optimal MFMA instruction
  - BlockPingpong: LDS double-buffer scheduling
  - ConvertToBufferOps: converts flat loads to buffer ops (faster)
  - Debug: `MLIR_ENABLE_DUMP=1` or `TORCH_COMPILE_DEBUG=1`

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
- On plateau: read `references/optimization/advanced-optimization.md`, pick technique by bottleneck type
- For GEMM: consider `TORCHINDUCTOR_MAX_AUTOTUNE=1` with `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK`
- For FP8: check `references/libraries/hip-intrinsics.md` for correct FP8 type per architecture
- For ISA inspection: `TRITON_CACHE_DIR=/tmp/triton_cache` then check generated `.amdgcn` files
- Target: exceed `torch.compile` by 5%+

## Step 5: Knowledge Accumulation

- Success → write `final_report.md` (design doc), backfill patterns to `references/`
- Failure → write `final_report.md` (postmortem), backfill lessons to `references/optimization/common-mistakes.md`
- This step is mandatory, never skip

## Quick Reference: Key Documents

| When | Read |
|------|------|
| Every profiling round | `references/toolchain/profiling-decision-tree.md` — mechanical bottleneck classification |
| Starting optimization | `references/optimization/optimization-patterns.md` — coalescing, tiling, fusion |
| Writing MFMA kernel | `references/isa/mfma-instructions.md` — instruction table, intrinsics, data layout |
| FP8 kernel | `references/libraries/hip-intrinsics.md` — FP8 HIP types by architecture |
| Debugging AMD Triton | `references/toolchain/triton-rocm-quirks.md` — compiler passes, debug flags, autotune |
| Performance plateau | `references/optimization/advanced-optimization.md` — FP8 GEMM 2600 TFLOPS progression |
| Need SOTA code pattern | `references/optimization/kernel-recipes.md` — RMSNorm, MoE, GEMM, swizzle, ping-pong |
| Check before custom kernel | `references/libraries/aiter-ops-reference.md` — AITER already-optimized ops |
| Before submitting | `references/optimization/common-mistakes.md` — AMD pitfall checklist |
