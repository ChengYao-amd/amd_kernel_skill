# Triton Kernel Optimization (AMD ROCm)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
rocprofv3 --stats -- python run_kernel.py
```

Identify bottleneck type: memory-bound / compute-bound / launch-overhead.
For deeper analysis: `rocprof-compute profile -n {OP}_baseline -- python run_kernel.py`

## Step 2: Implement Kernel

Start from `templates/triton_kernel_template.py`. Key rules:

- `BLOCK_SIZE`: use multiples of 64 (wavefront alignment). MI300X sweet spot: 128 or 256
- Use `tl.constexpr` for compile-time constants
- `triton.autotune` config space: include `num_warps` in [4, 8, 16], `num_stages` in [1, 2, 3], and `matrix_instr_nonkdim` in [16, 32] for MFMA tile selection
- ROCm backend limitations: check `references/triton-rocm-quirks.md`
  - No `tl.inline_asm_elementwise`
  - FP8 types: gfx942 uses FNUZ, gfx950 uses OCP — see `references/hip-intrinsics.md`
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
- On plateau: read `references/advanced-optimization.md`, pick technique by bottleneck type
- For GEMM: consider `TORCHINDUCTOR_MAX_AUTOTUNE=1` with `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK`
- For FP8: check `references/hip-intrinsics.md` for correct FP8 type per architecture
- For ISA inspection: `TRITON_CACHE_DIR=/tmp/triton_cache` then check generated `.amdgcn` files
- Target: exceed `torch.compile` by 5%+

## Step 5: Knowledge Accumulation

- Success → write `final_report.md` (design doc), backfill patterns to `references/`
- Failure → write `final_report.md` (postmortem), backfill lessons to `references/common-mistakes.md`
- This step is mandatory, never skip

## Quick Reference: Key Documents

| When | Read |
|------|------|
| Starting out | `references/optimization-patterns.md` |
| MFMA-related | `references/isa/mfma-instructions.md` |
| FP8 kernel | `references/hip-intrinsics.md` (FP8 types by arch) |
| Debugging AMD Triton | `references/triton-rocm-quirks.md` (compiler passes, debug flags) |
| Performance plateau | `references/advanced-optimization.md` (FP8 GEMM 2600 TFLOPS progression) |
| Check before AITER | `references/aiter-ops-reference.md` (already-optimized ops) |
