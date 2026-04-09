# HIP C++ Kernel Optimization (AMD)

## Step 1: Analyze Baseline

```bash
python scripts/benchmark_kernel.py --op {OP} --backend torch
rocprof-compute profile -n {OP}_baseline -- python run_kernel.py
rocprof-compute analyze -p {OP}_baseline/ --cli
```

Use ROCm Compute Profiler to identify micro-architecture bottleneck (memory, compute, occupancy, LDS).
Follow `references/toolchain/profiling-decision-tree.md` to classify bottleneck mechanically.

## Step 2: Implement Kernel

Start from `templates/hip_kernel_template.cpp`. Key rules:

- Compile: `hipcc -O3 --offload-arch={arch}` (MI300X: `gfx942`, MI355X: `gfx950`)
- Multi-target: `--offload-arch=gfx942 --offload-arch=gfx950`
- Wavefront = 64 (not 32): affects unroll factors, bank conflict analysis, reduction (6 steps not 5)
- **LDS per CU**: MI300X = 64 KB (32 banks); MI355X = **160 KB** (64 banks) — check `references/hardware/`
- Intrinsics: see `references/libraries/hip-intrinsics.md` for complete list including:
  - `__builtin_amdgcn_readfirstlane` — broadcast lane 0 to SGPR (reduces VGPR pressure, CK uses extensively)
  - `__builtin_amdgcn_mfma_*` — Matrix Core intrinsics (see `references/isa/mfma-instructions.md`)
  - `__builtin_amdgcn_s_setprio()` — wave scheduling priority control
  - `__builtin_amdgcn_sched_barrier()` — instruction scheduling barrier
- FP8 types: `__hip_fp8_storage_t` (gfx942 = FNUZ, gfx950 = OCP)
- Multi-hardware: `#if __gfx942__` / `#if __gfx950__` for hardware-specific paths
- PyTorch binding: `torch.utils.cpp_extension.load()` with `extra_cuda_cflags=["--offload-arch=gfx942"]`

## Step 3: Verify

```bash
python scripts/verify_correctness.py --kernel {kernel_path} --op {OP} --dtype bf16
python scripts/benchmark_kernel.py --kernel {kernel_path} --op {OP} --baseline torch_compile
```

Three gates: compile → correctness → performance.

## Step 4: Iterate

- Each round: save to `agent_output/<OP>/hip/round-N/`
- Update `performance_trend.md`
- On failure: retry 3+ times, consult `references/` (especially `references/toolchain/hipcc-compilation.md`, `references/optimization/common-mistakes.md`)
- On plateau: read `references/optimization/advanced-optimization.md` — includes FP8 GEMM 1→2600 TFLOPS progression
- For ISA-level optimization: `hipcc -save-temps` to inspect generated `.s` files, check `references/isa/`
- For register pressure: check `.vgpr_count`/`.sgpr_count` in `.s` file, consult `references/isa/register-allocation.md`
- For MFMA scheduling: pipeline MFMA with loads, use double-buffer LDS — see `references/isa/scheduling-pipeline.md`
- Target: exceed both Triton implementation and `torch.compile`

## Step 5: Knowledge Accumulation

- Success → design doc + backfill to `references/`
- Failure → postmortem + backfill to `references/optimization/common-mistakes.md`
- Mandatory step, never skip

## Quick Reference: Key Documents

| When | Read |
|------|------|
| Every profiling round | `references/toolchain/profiling-decision-tree.md` — mechanical bottleneck classification |
| Writing MFMA kernel | `references/isa/mfma-instructions.md` — full instruction table + intrinsic format |
| Register pressure | `references/isa/register-allocation.md` — VGPR budget, occupancy table |
| Memory optimization | `references/isa/memory-instructions.md` — DME, buffer vs flat, s_waitcnt, LDS bank rules |
| MFMA scheduling | `references/isa/scheduling-pipeline.md` — **NOP dependency table** (ISA Table 38) |
| Extreme optimization | `references/isa/inline-asm-patterns.md` — s_setprio, sched_barrier, buffer_load_lds |
| SOTA code patterns | `references/optimization/kernel-recipes.md` — FP8 GEMM, swizzle, ping-pong, RMSNorm, MoE |
| Compilation errors | `references/toolchain/hipcc-compilation.md` + `references/optimization/common-mistakes.md` |
| Check before writing | `references/libraries/aiter-ops-reference.md` — AITER may already have optimized version |
| GEMM tuning | `references/libraries/gemm-tuning-guide.md` — TunableOp → hipBLASLt → TensileLite |
