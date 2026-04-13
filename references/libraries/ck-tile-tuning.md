# CK Tile & Pipeline Tuning Reference (ck_tile GEMM / FMHA)

This document summarizes the **GEMM pipeline variants, scheduling strategies, partitioners, and representative tile configurations** readable from the **Composable Kernel source code** (`include/ck_tile/`), with supplementary **FMHA** constraints; intended to narrow the tuning search space on AMD GPUs such as the MI300 series. Values follow the **preset configuration names** in the repository; actual LDS/VGPR occupancy should be verified via compile-time checks and profiler.

## 1. Pipeline & Implementation Classes (`ops/gemm/pipeline/`)

| Pipeline Tag | Implementation Class (Example Name) | Design Orientation |
|--------------|--------------------------------------|-------------------|
| `MEMORY` | `GemmPipelineAgBgCrMem` | **Memory-bound**: `MinMemInFlyBytes` commonly on the order of **32768**; `PrefetchStages` **2-8**; **Intrawave / Interwave** selectable. |
| `COMPUTE_V3` | `GemmPipelineAgBgCrCompV3` | `PrefetchStages=2`; **Intrawave only**. |
| `COMPUTE_V4` | `GemmPipelineAgBgCrCompV4` | **Double-buffered LDS (ping-pong)**, improving memory access latency hiding. |
| `COMPUTE_V5` | `GemmPipelineAgBgCrCompV5` | `NumWaveGroups=2`. |
| `COMPUTE_V6` | `GemmPipelineAgBgCrCompV6` | **K tile fixed at 32**. |
| `COMPUTE_ASYNC` | `GemmPipelineAgBgCrCompAsync` | **Large `K_Warp_Tile` (e.g., 128)**, dual SMEM, asynchronous path for **FP4**, etc. |
| `PRESHUFFLE_V2` | `WeightPreshufflePipelineAGmemBGmemCRegV2` | **Inference weight preshuffle** (AG mem / BG mem / C reg path). |

**Scheduler pairing rules of thumb**:

- **Memory pipeline**: Focus on **Interwave** with **PrefetchStages** and **MinMemInFlyBytes**; use larger "in-flight" byte counts to hide DRAM / L2 latency.
- **Compute pipeline**: **Intrawave** is often the default; consider alongside the wave and fixed-K constraints of **COMPUTE_V3/V4/V5/V6**.

## 2. Tile Partitioner Selection

| Partitioner | Applicable Scenarios |
|-------------|---------------------|
| `GemmTile2DPartitioner` | General-purpose **2D** `(M_blocks, N_blocks)`, easy to align with intuitive batch / head partitioning. |
| `GemmTile1DPartitioner` | **Linearized** block index; suitable for persistent tile or simplified grid logic. |
| `GemmSpatiallyLocalTilePartitioner` | **Spatially local** + **RemapXCD** (**gfx94x multi-die**): reduces cross-XCD / cross-cache coherence traffic. |

When profiling shows **L2 / cross-die** anomalous hotspots, prefer trying the **SpatiallyLocal** and **MEMORY / Interwave** combination first.

## 3. Representative `GemmConfig*` from Source Code (Actual Tile Shapes)

The table below shows **block tile (M x N x K)** and **warp grid (M x N x K)** directly readable from configuration names; in the K column, **(128/elem)** means K tile counted in **elements**, **(256/storage)** means K counted in **storage blocks** (related to FP8/packed layouts; consult `TileGemmTraits`).

| Config Name | Pipeline | Block (M x N x K) | Warps (M x N x K) | Typical Use Case |
|-------------|----------|--------------------|--------------------|------------------|
| `GemmConfigMemoryInterwave` | `MEMORY` | 128x32x(128/elem) | 4x1x1 | **Memory-bound**, bandwidth-oriented scenarios |
| `GemmConfigComputeV3` | `COMPUTE_V3` | 16x64x(256/storage) | 1x4x1 | **Small M**, high parallelism subdivision |
| `GemmConfigComputeV3_1` | `COMPUTE_V3` | 256x256x(128/elem) | 2x2x1 | **Large GEMM** |
| `GemmConfigComputeV4` | `COMPUTE_V4` | 256x256x(64/elem) | 2x2x1 | **LDS ping-pong** |
| `GemmConfigComputeV5` | `COMPUTE_V5` | 128x128x(64/elem) | 1x1x2 | **Dual wave group** |
| `GemmConfigComputeAsync` | `COMPUTE_ASYNC` | 64x64x256 | 1x4x1 | **FP4** and other asynchronous paths |
| `GemmConfigPreshuffleDecode` | `PRESHUFFLE_V2` | 16x64x(256/storage) | 1x4x1 | **Inference decode** |
| `GemmConfigPreshufflePrefill` | `PRESHUFFLE_V2` | 128x128x(128/storage) | 1x4x1 | **Inference prefill** |

When tuning, use the table above as **discrete starting points**, then search within +/-1 step in warp / block dimensions; the **PRESHUFFLE** and **COMPUTE_ASYNC** paths require satisfying **weight layout and asynchronous pipeline** prerequisites.

## 4. FMHA (Flash / Tile Attention) Related Constraints

Observable from CK-side tuning and instance configurations:

- **head_dim (hdim) 64 and 128** typically have the **most complete** tuning coverage; other hdim values may require extrapolation from neighboring configurations or additional kernels.
- **Representative block shapes** include: **64x64x32**, **128x64x32**, **32x128x32** (numbers correspond to M/N/sequence or KV dimension segments of the tile; exact semantics per the corresponding `Fmha*` / `Block*` templates).
- **`CppConstraint`**: Used for **dispatch constraints per CU / architecture** (if not satisfied, falls back or selects another instance).
- **Score blocks**: Commonly **`kM0=64`** as the **score tiling** M-direction baseline, beneficial for MFMA and mask layout when aligned with Q block rows.

FMHA and GEMM share the **ck_tile** "Shape -> Traits -> Pipeline -> Kernel" philosophy, but are additionally constrained by **causal mask, GQA/MQA, varlen** and other logic; tuning should proceed with performance search only after **correctness tests** (numerical and boundary) pass.

## 5. General Parameter Descriptions (Cross-Referenced with Table Above)

| Parameter Domain | Meaning | Notes |
|-----------------|---------|-------|
| Block M/N/K | Thread block tile on output and reduction dimensions | K may be counted per element or per packed storage. |
| Warp M/N/K | Sub-tile each warp is responsible for | Together with **wave** and **wave group** (e.g., V5), determines occupancy. |
| `PrefetchStages` | Number of prefetch stages | Can reach **2-8** on **MEMORY**; commonly **2** for **COMPUTE_V3**. |
| Dual SMEM / ping-pong | Two sets of LDS buffers alternating | **COMPUTE_V4**, **COMPUTE_ASYNC**, etc. |
| `MinMemInFlyBytes` | Target minimum "in-flight" bytes | Related to bandwidth hiding in **MEMORY** pipeline (on the order of **32768**). |

## 6. Resource & Occupancy Checklist

When tuning each candidate configuration, it is recommended to verify the following items:

1. **LDS**: Roughly proportional to **(A tile + B tile) x stage count x data width**; ping-pong and **PrefetchStages** multiply this. Must satisfy the **per-CU LDS limit** for the architecture (e.g., on the order of 64KB; refer to target ISA documentation).
2. **Grid**: Block count should be sufficient to **fill the GPU**; small batch / narrow M easily leads to **insufficient launches**.
3. **VGPR / SGPR**: Large warp tiles or complex epilogues may cause **register spilling** or **occupancy degradation**.
4. **Multi-die**: On **gfx94x**, compare **`GemmTile2DPartitioner`** vs **`GemmSpatiallyLocalTilePartitioner`**.

## 7. MI300X (GFX942) & MI355X (GFX950) Practical Tips

- **GFX942**: The **GEMM / FMHA** configurations and **RemapXCD** strategies in the table above are specifically targeted; for large GEMM, consider starting with **`GemmConfigComputeV3_1`** / **`GemmConfigComputeV4`**.
- **GFX950 (MI355X)**: After changes in CU count, frequency, and cache hierarchy, **identically named configurations** may need re-tuning; it is recommended to start from **CK branch instances for that arch**, then sweep **block_m / block_n** and **pipeline tags**.

## 8. Tuning Workflow (Recommended)

1. Select the **pipeline tag** based on the operator bottleneck (memory vs compute vs preshuffle vs async).
2. Select the **partitioner** (general-purpose 2D -> switch to **SpatiallyLocal** when multi-die locality is poor).
3. Choose the nearest **block/warp** shape from the **named configurations in Section 3**.
4. Perform a **small-range grid search** over **block_m, block_n** (and allowable K tile), while fixing **pipeline depth** and **wave configuration** before comparing.
5. Use **rocprof / RGP** etc. to confirm whether **LDS, VGPR, bandwidth, and MFMA** match expectations.

## Related Documentation

- Layered model and type chain: [[ck-programming-model.md|ck-programming-model]]
- Inference-side packaging: [[aiter-ops-reference.md|aiter-ops-reference]]
