# Extreme Optimization: Breaking Through Performance Plateaus

## When to Use

When a kernel has already passed correctness verification, already surpassed the torch.compile baseline, but still has a gap from the theoretical peak (bandwidth utilization <70% or MFMA utilization <80%). Try the following in order.

## 1. Software Pipelining and Multi-Level Buffering

**Principle**: Overlap the current iteration's computation with the next iteration's data loading.

**Techniques**:
- Double buffering: Split LDS in half, alternating between loading and computing
- Three-stage pipeline: Load N+1, compute N, write back N-1
- Async copy: DMA engine transfers data fully concurrently with MFMA

**AMD Implementation Notes**:
- Requires precise `s_waitcnt lgkmcnt/vmcnt` (see `isa/scheduling-pipeline.md`)
- LDS buffer: verify per-CU **LDS capacity and read bandwidth** (CDNA4 whitepaper: **160 KiB/CU**, **256 B/clock** read bandwidth; relative to CDNA3 capacity and read bandwidth approximately **2x**; see `isa/memory-instructions.md`)
- **CDNA4**: There is a **direct load from L1 data cache to LDS** path, which can keep hot data on the **L1->LDS** link, reducing contention with the global path; profile together with double buffering and async transfers.
- Example: Prefetch the next tile while the current MFMA is executing

**Expected Gains**: Memory-bound kernels typically 10-30%

## 2. Wavefront Specialization

**Principle**: Different wavefronts within the same block take on different roles, coordinated via barriers.

**Roles**: Compute wavefront (MFMA), data transfer (global->LDS), reduction (softmax/reduction)

**AMD Implementation Notes**:
- wavefront=64 -> each wavefront is heavier, specialization gains are larger
- Role assignment: `threadIdx.x / 64` gives wavefront ID
- Coordinate via `__syncthreads()` or LDS fence

**Expected Gains**: Multi-stage kernels (Attention) can achieve 5-15%

## 3. Data Layout and Swizzle

**Principle**: Rearrange data layout to eliminate bank conflicts and improve coalesced access rates.

**Techniques**:
- LDS padding: Pad according to target architecture **bank count** (CDNA3: **32 bank** x 4B; CDNA4: **64 bank** -- see `isa/memory-instructions.md`)
- `ds_swizzle`: Hardware lane permutation without LDS reads/writes
- SOA layout for vectorized loads (`buffer_load_dwordx4`)

**AMD Implementation Notes**:
- AMD LDS bank rules differ from NVIDIA -- recalculate padding
- Verify with `omniperf` LDS bank conflict metrics

**Expected Gains**: 5-20% when bank conflicts exist

## 4. Occupancy vs ILP Tradeoff

**Principle**: Sometimes fewer wavefronts with more registers outperforms many wavefronts with spilling.

**Techniques**:
- `__launch_bounds__(threads, minBlocks)` to control register allocation
- Detect spilling: `omniperf` -> ScratchWaveslifetimeVGPR > 0
- Register rebalancing across wavefront groups (refer to AVO v33)

**AMD Implementation Notes**:
- MI300X: 65536 VGPR/CU. See the occupancy table in `isa/register-allocation.md`
- Profile both high occupancy and low occupancy configurations -- the faster one wins

**Expected Gains**: 3-10% when spilling is eliminated

## 5. Persistent Kernel and Tile Scheduling

**Principle**: Launch only once, self-assign tiles via atomic counters. Eliminates repeated launch overhead + L2-friendly traversal order.

**Techniques**:
- `atomicAdd` global tile counter
- Swizzled traversal (L-shaped, Z-shaped, Hilbert curve) to improve L2 reuse
- Cross-tile load balancing for irregular shapes (causal mask)

**AMD Implementation Notes**:
- MI300X L2 is relatively large (256MB) -- tile traversal order has significant impact
- CK's TileScheduler can serve as a reference implementation

**Expected Gains**: 5-15% for multi-launch scenarios; L2 optimization typically 3-8%

## 6. Mixed Precision Strategy

**Principle**: Go beyond "use BF16 for everything" -- use different precisions for different computation stages.

**Techniques**:
- Input FP8/BF16 -> MFMA computation -> FP32 accumulation -> BF16 output
- Keep critical intermediate results (e.g., softmax max/sum) in FP32
- Leverage MFMA's mixed-precision capability (FP16 input, FP32 output)

**AMD Implementation Notes**:
- Consult `isa/mfma-instructions.md` for available precision combinations and **per-CU FLOPS/clock** (improvements on CDNA4 for FP16/FP8/MX etc. relative to CDNA3, **Matrix FP64 halved**)
- MI355X/CDNA4 adds new **MXFP6/MXFP4** formats -- refer to hardware documentation and ROCm release notes for details
- **Transcendental throughput**: On CDNA4, related instructions have an effective rate approximately **2x** relative to CDNA3; segments containing **exp/log** such as **softmax** and activations are more likely to become optimizable hotspots within an operator
- **Structured sparsity**: When input has **zero element ratio >= 50%** within **groups of 4 elements**, the hardware path can **double** effective throughput (requires operator/library support and precision verification)

**Expected Gains**: Compute-bound kernels can achieve 10-30% TFLOPS improvement

## 7. Compiler Counter-Measures and Guidance

**Principle**: Manually intervene when compiler auto-optimization is insufficient or excessive.

**Techniques**:
- `#pragma unroll N`: Precise control over unrolling
- `__launch_bounds__`: Guide register allocation
- `volatile` / `__builtin_nontemporal_*`: Bypass cache / prevent reordering
- Inline assembly: Last resort (see `isa/inline-asm-patterns.md`)

**AMD Implementation Notes**:
- `hipcc -save-temps` lets you inspect the generated ISA to verify compiler behavior

**Expected Gains**: Varies by situation, typically 2-10%

## 8. L2 Cache Global Optimization

**Principle**: Global-level data reuse strategy, maximizing L2 hit rate across tiles / across kernels.

**Techniques**:
- Tile traversal order (GEMM's K dimension is especially sensitive)
- Cross-kernel fusion to keep data in L2
- L2 cache residency control (e.g., hardware-supported prefetch hints)

**AMD Implementation Notes**:
- MI300X L2 = 256MB, relatively large
- Measure with `rocprof` TCC_HIT/TCC_MISS counters

**Expected Gains**: Memory-bound kernels typically 3-8%

## 9. FP8 GEMM Optimization Progression (CDNA4 / ROCm Practice)

ROCm blog posts and other materials describe a typical progression from naive implementation to near-peak on **CDNA4** for **FP8 GEMM** (numbers are examples from the articles; actual results should be based on local profiling):

| Stage | Approach | Representative Result (Example) |
|-------|----------|---------------------------------|
| Naive | Insufficient tiling / not aligned with hardware MFMA | ~**1.15 TFLOPS** |
| LDS tiling + MFMA | Software tile aligned with matrix core | Significant improvement |
| Vectorized load + `buffer_load_lds` | Async DMA, relieving VMEM | Further improvement |
| LDS swizzle + double buffering | Reducing bank conflicts, overlapping transfer and compute | Notable gains |
| 8-wave ping-pong scheduling | `__builtin_amdgcn_s_setprio()`, `sched_barrier`, coordinated with `buffer_load_lds` | ~**2597 TFLOPS** (near peak region) |

**Key Technical Summary**:

- **`buffer_load_lds`**: Asynchronously sends data into LDS, facilitating overlap with MFMA pipeline.
- **`__builtin_amdgcn_s_setprio()`** and **`sched_barrier`**: Control wave priority and scheduling barriers, serving **ping-pong** multi-wave orchestration.
- Cross-reference with the "Software Pipelining" and "Data Layout and Swizzle" sections above: On CDNA4 **LDS has 64 banks**, padding must be recalculated for **64-bank** (see `isa/memory-instructions.md`).

## Selection Guide

| Bottleneck Type | Diagnostic Metric | Priority Techniques |
|-----------------|-------------------|---------------------|
| HBM bandwidth bound | Bandwidth utilization >80% | Software pipelining, L2 optimization, data layout |
| LDS bound | High bank conflict rate | Swizzle, padding, data layout |
| Compute bound | MFMA utilization <70% | Wavefront specialization, mixed precision, ILP |
| Register spilling | ScratchWaves > 0 | Occupancy tuning, register rebalancing |
| Launch overhead | Many small kernel launches | Persistent kernel |
| Compiler issues | ISA review reveals redundant instructions | Compiler guidance, inline assembly |
