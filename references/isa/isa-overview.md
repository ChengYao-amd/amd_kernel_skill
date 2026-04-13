# AMDGPU ISA Quick Reference

Entry document for ISA-level optimization. Read this file first, then dive into specific documents as needed (e.g., `mfma-instructions.md`, `memory-instructions.md`, `register-allocation.md`).

## Register Files (capacities per ROCm / hardware whitepapers)

**Important**: Register file sizes should be expressed as **per-CU storage capacity (KiB)**. Previously misusing numbers like "65536" as "number of VGPRs" caused confusion -- **512 KiB VGPR / CU** refers to the **physical vector register file capacity**; the actual **number of VGPRs** available to each wavefront is still constrained by **architectural limits and occupancy** (see `register-allocation.md`).

| Type | Per-CU Capacity (Typical CDNA) | Purpose |
|------|-------------------------------|---------|
| **VGPR** | **512 KiB** | Per-lane data, MFMA operands, and general vector operations |
| **SGPR** | **12.5 KiB** | Uniform scalars, addresses, control flow, and some memory instructions |
| **AGPR** | **512 KiB / CU (same size as the VGPR file, CDNA)** | Dedicated to matrix accumulation such as MFMA (paths like `v_accvgpr_read` / `v_accvgpr_write` exist between AGPR and VGPR) |

Supplementary notes:

- The **per-wavefront VGPR/SGPR count limits** determine the **number of waves that can co-reside**, which in turn affects **occupancy**; this is a different dimension from "total KiB capacity of the entire CU" -- both must be considered during tuning.
- For specific values of **max VGPR count/wave**, **max waves/SIMD**, etc., always refer to the **target GPU and omniperf/compiler reports**.

## Instruction Categories

| Category | Execution Unit | Examples | Latency (order of magnitude) |
|----------|---------------|---------|------------------------------|
| VALU | Vector ALU | `v_add_f32`, `v_fma_f32` | ~4-8 cycles |
| SALU | Scalar ALU | `s_add_u32`, `s_cmp_eq` | ~2-4 cycles |
| MFMA | Matrix Core | `v_mfma_f32_16x16x16_f16`, etc. | **~16-64 cycles depending on variant** (see `mfma-instructions.md`) |
| VMEM | Vector Memory | `global_load_dwordx4` | Hundreds of cycles (highly dependent on cache and bandwidth) |
| LDS | Local Data Share | `ds_read_b128`, `ds_write_b64` | ~Tens of cycles |
| SMEM | Scalar Memory | `s_load_dwordx4` | On the order of hundreds of cycles |

## Pipeline Model: CDNA3 (GFX942) Brief Overview

- **CU Structure**: Each **CU** contains multiple **SIMD** execution units; each **SIMD** schedules at **wavefront (64 lanes)** granularity and can advance **one** wave instruction **per cycle** (understand **IPC** in conjunction with details like **VALU 4-cycle throughput**).
- **MFMA**: Instructions enter the **Matrix Core** pipeline with **high latency but pipelineable**; when co-issued with **VMEM/LDS**, use **`s_waitcnt`** to manage **vmem/vs/sc/lds** counters.
- **Memory**: **Global/texture** accesses complete asynchronously, relying on **waitcnt** and **barriers** for semantic correctness.

For more detailed scheduling and bubble elimination, see `scheduling-pipeline.md` and `inline-asm-patterns.md`.

## CDNA4 (GFX950) Pipeline and ISA Incremental Highlights

While inheriting the **CDNA3** overall execution model (wavefront, SIMD, VMEM/LDS async model), **CDNA4** significantly extends the **MFMA** family on the **Matrix Core**:

- **Larger-K FP16/BF16 tiles** (e.g., **16x16x32**, **32x32x16**): higher **K-dimension throughput** for the same **MxN**, beneficial for **reducing instruction count** and **increasing FMA density** (must be evaluated together with **register and LDS occupancy**).
- **FP8 / FP6 / FP4** paths: MFMA variants where **A and B types can be independently configured**, with implementations in the **16-64 cycle** range (varying by shape and type combination).
- **MXFP8 / MXFP6 / MXFP4**: **Block-scaled** low-precision formats, used with **scale MFMA** intrinsics (e.g., `__builtin_amdgcn_mfma_scale_f32_*`) to compress **weight/activation** storage and bandwidth **while accumulating in FP32 precision**.

**Migration tip**: When targeting CDNA4, use **`gfx950`** as the compilation target (or the offload arch specified in product documentation); **FP8 encoding** on CDNA4 aligns with **OCP (E4M3FN / E5M2)**, which differs from CDNA3's **FNUZ** variant in **exponent bias and semantics** -- cross-generation kernels require **explicit conversion and numerical verification**.

## CU Internal Structure (CDNA)

### Compute Unit (CU) Components

| Component | Count / Capacity | Description |
|-----------|-----------------|-------------|
| SIMD Processors | 4 per CU | Each SIMD has 16 ALUs = 64 ALUs total per CU |
| VGPR per CU | 256-512 KiB | Split across 4 SIMDs |
| SGPR per CU | 12.5 KiB | Shared across waves |
| Vector L1 Cache | 16 KiB per CU (CDNA1-2), 32 KiB (CDNA3-4) | Write-through |
| LDS | 64 KiB (CDNA3), 160 KiB (CDNA4) | 32 banks (CDNA3), 64 banks (CDNA4) |
| Matrix Cores | 4 per CU | MFMA execution units |
| SFU | 1 per CU | Hardware-accelerated exp, log, sin, cos, rcp, rsqrt |

### Instruction Sequencer

- **4 warp pools** x **10 slots** = **40 warps max** per CU (theoretical)
- Fetch bandwidth: **32 bytes/cycle**
- Issue model: **round-robin** among ready wavefronts
- Up to **5 instructions/cycle per CU** across different functional units (VALU + VMEM + SALU/SMEM + LDS + Branch)
- **Zero-overhead context switching** between warps (all contexts are resident on CU)

### Special Function Unit (SFU)

Hardware-accelerated transcendental functions tuned for deep learning in CDNA3+:
- `exp`, `log`, `sin`, `cos`, `rcp` (reciprocal), `rsqrt` (reciprocal square root)
- Throughput: **1 per 4 cycles** (shared unit) -- avoid in inner loops

### Data Movement Engine (DME, CDNA3/4)

Hardware unit for asynchronous bulk memory transfers between HBM and LDS:
- Performs **affine address calculations** in hardware (stride x index + offset)
- Decouples data movement from compute pipelines
- Reduces VMEM pressure by offloading some data paths
- Access via `__builtin_amdgcn_async_work_group_copy` intrinsic
- Enables effective **double buffering** and software pipelining overlaps

## When to Use ISA Knowledge

| Optimization Phase | ISA Depth | Corresponding Documents |
|-------------------|-----------|------------------------|
| Early (fusion, tiling) | Low | This file only |
| Mid (occupancy, memory access) | Medium | `memory-instructions.md`, `register-allocation.md` |
| Late (instruction scheduling, bubble elimination) | High | `scheduling-pipeline.md`, `inline-asm-patterns.md` |
| MFMA-intensive (GEMM, Attention) | High | `mfma-instructions.md` |
