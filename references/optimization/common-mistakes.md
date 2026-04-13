# AMD-Specific Common Mistakes

## Compilation

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Missing `--offload-arch` | Wrong compilation target or kernel doesn't execute at runtime | Always specify `--offload-arch=gfx942` (MI300X) or `gfx950` (MI355X) |
| Missing `-O3` | 5-10x slower than expected | Always use `-O3` |
| Using CUDA APIs directly | Compilation errors | Replace with HIP equivalents (see `amd-vs-nvidia-cheatsheet.md`) |
| `--offload-arch` doesn't match actual GPU | Kernel runs but produces wrong results or crashes | Verify actual arch with `rocminfo \| grep gfx` |
| Using CDNA4-exclusive instructions but targeting gfx942 | Compilation errors | FP6/FP4 MFMA is gfx950+ only, check target hardware |

## Architecture

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Assuming warp = 32 | Wrong reduction results, performance degradation | AMD wavefront = 64. Use 6-step shuffle, not 5-step |
| **Assuming MI300X has 192 CUs** | Insufficient grid size, low GPU utilization | MI300X/MI325X actually has **304 CUs** (38/XCD x 8 XCDs) |
| Assuming shared mem = 48KB | LDS overflow or incorrect occupancy calculation | CDNA4 **160 KB/CU** (64 banks x 640 entries x 4B, ISA spec); CDNA3 **64 KB/CU** (32 banks x 512 entries x 4B, ISA spec); read bandwidth **128 B/clock** (CDNA3, inferred from CDNA4 WP "doubles to 256") vs **256 B/clock** (CDNA4, whitepaper p.9); see `isa/memory-instructions.md`; do not use the NVIDIA 48KB default |
| Using `__syncwarp()` | Unnecessary synchronization | AMD wavefront is lockstep, no partial synchronization needed |
| Wrong bank conflict calculation | Unexpected LDS contention | **CDNA3**: 32 bank x 4B; **CDNA4**: **64 bank** (different from CDNA3, padding must be recalculated); conflict pattern differs from NVIDIA |
| Overusing **flat load** in HIP, ignoring **buffer ops** | High VMEM pressure, sub-optimal bandwidth | Prefer **`buffer_load_*` / buffer ops** where applicable (compiler or handwritten ISA); do not assume flat and buffer are equally optimal |
| Still using **32-bank LDS** padding on CDNA4 | Hidden bank conflicts, performance below expectations | CDNA4 is **64-bank LDS**, update swizzle / padding per `isa/memory-instructions.md` |
| Having **structured sparsity** capability but running fully dense | Not utilizing full effective throughput | When input has **zero ratio >= 50%** within **groups of 4 elements**, hardware can **double** effective throughput; evaluate enabling when operator/library support is available and verify precision (see `advanced-optimization.md`) |
| Not using **`__builtin_amdgcn_s_setprio`** and other wave scheduling mechanisms | Insufficient ping-pong / multi-wave overlap | In advanced GEMM (see `advanced-optimization.md`), use with **`sched_barrier`** and `buffer_load_lds` for wave-level orchestration |
| Mixing up FP8 precision formats | Incorrect numerical results | CDNA3 uses E4M3**FNUZ**/E5M2**FNUZ**; CDNA4 uses E4M3**FN**(OCP)/E5M2(OCP), different exponent bias |
| Ignoring XCD (multi-die) topology | Abnormally high L2 cache miss rate | MI300X has 8 XCDs, each XCD has its own L2 (4MB); cross-XCD access goes through L3 |
| **Assuming CDNA4 still has TF32 hardware matrix path** | Performance/numerical expectations confused with NVIDIA | CDNA4 has **no TF32 Matrix hardware**; must use **BF16** or other paths with **software emulation**, and re-evaluate precision and throughput (see `isa/mfma-instructions.md`) |
| **Applying CDNA3 partition modes directly to CDNA4** | Topology/NPS behavior doesn't match expectations | CDNA4 introduces **QPX** and **only supports NPS1 / NPS2**, **no NPS4**; re-select partition based on platform documentation before migration |
| **Ignoring CDNA4 Matrix FP64 slowdown** | HPC double-precision GEMM or implicit FP64 far below CDNA3 experience | CDNA4 **Matrix FP64** is **128 FLOPS/clock/CU** (relative to CDNA3's **256**, **halved**); hotspots need re-evaluation of algorithm, precision, or whether to use non-matrix FP64 path |

## Performance

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Too few thread blocks | Low GPU utilization | MI300X needs at least **304 blocks** (at least one per CU); MI355X needs at least 256 |
| Ignoring register spilling | Unexplained performance degradation | Check ScratchWaves with ROCm Compute Profiler, or inspect `.s` files via `hipcc -save-temps` |
| Directly copying CUDA tuning parameters | Sub-optimal performance | Re-tune block size, unroll factors, and tile sizes for AMD |
| Not using AGPR for MFMA accumulators | Higher VGPR pressure | Use AGPR accumulators to free up VGPRs (see `isa/register-allocation.md`) |
| Ignoring `__builtin_amdgcn_readfirstlane` | High SGPR pressure, poor branch efficiency | Broadcast lane 0 value to SGPR (this pattern is heavily used in CK code) |
| Not enabling TunableOp | Unstable GEMM performance | Set `PYTORCH_TUNABLEOP_ENABLED=1` to let rocBLAS/hipBLASLt auto-tune |

## Triton-Specific

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using `tl.inline_asm_elementwise` | Errors on ROCm | Use pure Triton operations |
| BLOCK_SIZE not a multiple of 64 | Wasted lanes | Use 64, 128, 256, 512, 1024 |
| Assuming CUDA Triton performance translates | Disappointment | Always benchmark on AMD |
| Not knowing about `matrix_instr_nonkdim` | Not selecting the optimal MFMA instruction | Add this parameter to autotune config (16 or 32) to control MFMA tile size |
| Not using `max-autotune` | Unoptimized GEMM performance | `TORCHINDUCTOR_MAX_AUTOTUNE=1`, or `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK` |
| FP8 kernel fails on different architectures | Precision type mismatch | gfx950 uses OCP FP8, other archs use FNUZ; AITER handles this automatically, custom kernels must check |

## Cross-Hardware Migration

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Using MI300X tile sizes directly on MI355X | Performance below target | CU count (304 vs 256), LDS size (64 vs 160 KB), and bandwidth differ; must re-tune |
| Not considering CDNA4 new instructions | Missed performance gains | MI355X's FP16 MFMA has larger K dimension (16x16x32, 32x32x16), plus FP6/FP4 support |
| Cross-arch compilation without testing | Different runtime behavior | After compiling with `--offload-arch=gfx942 --offload-arch=gfx950`, verify and benchmark on both hardware targets separately |

## Knowledge Base

This file is a living document. Backfill when new mistakes are discovered during iterations.
