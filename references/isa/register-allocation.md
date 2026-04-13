# Register Allocation and Occupancy Guide

## Register File Capacity (from ROCm Official Hardware Specs)

| Register Type | MI300X (CDNA3) | MI355X (CDNA4) | Description |
|--------------|---------------|---------------|-------------|
| VGPR File | 512 KiB / CU | 512 KiB / CU | Per-lane vector data, MFMA operands |
| SGPR File | 12.5 KiB / CU | 12.5 KiB / CU | Uniform values, addresses, control flow |
| AGPR File | Same size as VGPR | Same size as VGPR | Dedicated to MFMA accumulators (CDNA) |

> **Note**: Older documentation often writes "65536 VGPR/CU", which refers to the **register count** (each 32-bit), equivalent to `65536 x 4B = 256 KB`. However, the ROCm official GPU spec sheet states **512 KiB/CU** (which includes the physical file shared between VGPR + AGPR). When writing kernels, using **register count** is more intuitive; when evaluating hardware capacity, refer to **KiB**.

## VGPR Budget and Occupancy (MI300X / CDNA3, per SIMD unit)

Each CU has 4 SIMD units, and each SIMD has 16384 VGPRs (32-bit).

| Max VGPR / Wavefront | Max Wavefronts / SIMD | Occupancy |
|-----------------------|-----------------------|-----------|
| 128 | 8 | 100% |
| 192 | 5 | 62.5% |
| 256 | 4 | 50% |
| 384 | 2 | 25% |
| 512 | 2 | 25% |
| 1024 | 1 | 12.5% |

Formula: `wavefronts_per_simd = floor(16384 / vgpr_per_wavefront)`, capped at 8.

## SGPR Budget

| Max SGPR / Wavefront | Max Wavefronts / SIMD |
|-----------------------|-----------------------|
| <= 102 | 8 |
| > 102 | Spills to memory |

SGPR rarely limits occupancy. Focus on managing VGPR.

## AGPR (Accumulation GPR)

- Dedicated to MFMA accumulator results
- Shares the same physical register file as VGPR (512 KiB / CU)
- Using AGPR for accumulators frees VGPR for data -> higher occupancy
- VGPR <-> AGPR movement: `v_accvgpr_read` / `v_accvgpr_write` (has latency; avoid frequent moves in hot loops)

## MI355X (CDNA4) Differences

- VGPR File is also 512 KiB / CU, but the number of CUs differs (256 vs 304)
- LDS increased to **160 KiB / CU**: more LDS can reduce VGPR spill demand
- New FP6/FP4 MFMA instructions: larger K dimensions mean a single instruction consumes more VGPRs; watch the register budget

## Detecting Register Pressure

```bash
# Method 1: Check at compile time (recommended first choice)
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# In the generated .s file, search for:
#   .vgpr_count   -> number of VGPRs used by this kernel
#   .sgpr_count   -> number of SGPRs used by this kernel
#   .agpr_count   -> number of AGPRs used by this kernel

# Method 2: ROCm Compute Profiler (formerly omniperf)
rocprof-compute profile -n check_spill -- python run_kernel.py
rocprof-compute analyze -p check_spill/ --cli
# Watch for ScratchWaveslifetimeVGPR > 0 -> indicates register spilling!

# Method 3: rocprofv3 counters
rocprofv3 -i counters.txt python run_kernel.py
# Check SQ_WAVES, SQ_INSTS_VALU, etc.
```

## Reducing Register Pressure

| Technique | Method | Impact |
|-----------|--------|--------|
| Shorten live ranges | Compute and consume values close together, reducing simultaneously live variables | Medium |
| Use AGPR for accumulators | `v_accvgpr_write` to store accumulation results in AGPR | Frees VGPR |
| `__launch_bounds__(threads, minBlocks)` | Hint to compiler about register budget ceiling | Direct control |
| `#pragma unroll N` | Precisely control unroll factor (excessive unrolling = more VGPRs) | Balance ILP vs registers |
| `__builtin_amdgcn_readfirstlane` | Broadcast lane 0 value to SGPR, reducing VGPR usage (common CK pattern) | Low overhead |
| Manual register reuse | Rewrite loops to reuse registers | High effort |
| Accept lower occupancy | If ILP can hide latency, fewer waves may be acceptable | Trade-off |

## Occupancy vs ILP Trade-off

Low occupancy is not necessarily bad. If a kernel has sufficient ILP (independent instructions between memory operations), fewer wavefronts with more registers may outperform many wavefronts with register spilling.

**Decision flow**:
1. Check for spilling (ScratchWaves > 0 or spill count > 0 in the `.s` file)
2. If spilling: reduce VGPR usage (shorten live ranges, use AGPR) or accept lower occupancy
3. If not spilling but performance is low: try increasing occupancy by reducing VGPRs
4. If increasing occupancy actually decreases performance: ILP matters more, revert to low-occupancy configuration
5. **Profile both configurations; the faster one wins**

## Occupancy Theory and Calculation

### Definition

```
Occupancy = allocated_wavefronts / max_available_slots
```

- CDNA: each SIMD has up to **8** wavefront slots (wave64), max **8** waves/SIMD
- Occupancy is limited by the most constrained resource

### Occupancy Limiting Factors

| Resource | Description |
|----------|-------------|
| **VGPR** | High VGPR usage per wave reduces co-resident wave count |
| **SGPR** | Usually not the bottleneck (spills to memory above 102) |
| **LDS** | Workgroup LDS usage limits groups per CU |
| **Thread Group Size** | Workgroup must fit on a single CU |
| **Barriers** | Limited barrier slots per SIMD pair |

### Theoretical Occupancy Formula

```
max_waves = min(
    floor(total_VGPRs_per_SIMD / VGPRs_per_wave),
    floor(total_LDS_per_CU / LDS_per_workgroup) * waves_per_workgroup,
    max_wave_slots_per_SIMD
)
occupancy = max_waves / max_wave_slots_per_SIMD
```

### Performance vs Occupancy

- High occupancy does **not** guarantee good performance (ALU-bound workloads may not benefit)
- Low occupancy is **not** necessarily bad (more registers per wave reduces spilling, may be faster)
- Excessive occupancy with memory-bound workloads can cause **cache thrashing**
- The value of occupancy is primarily in **latency hiding** via thread-level parallelism

## MI300X Practical Reference Values

| Scenario | Recommended Occupancy | Rationale |
|----------|----------------------|-----------|
| Memory-bound kernel (element-wise, normalization) | >= 50% (<= 256 VGPR) | Need TLP to hide memory latency |
| Compute-bound GEMM (MFMA-intensive) | 25-50% (256-512 VGPR) | MFMA latency hidden by ILP |
| Extreme optimization (manual pipelining) | Can go as low as 12.5% (<= 1024 VGPR) | Rely on ILP + software pipelining |
