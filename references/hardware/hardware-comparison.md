# AMD GPU Hardware Comparison

## Quick Reference (Four GPUs)

| Parameter | MI300X (GFX942/CDNA3) | MI325X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) | MI350X (GFX950/CDNA4) |
|-----------|------------------------|-------------------------|------------------------|------------------------|
| XCD / CU per die | **40** per XCD (**38** active) | (same as left) | **36** per XCD (**32** active) | (same as left) |
| Active Compute Units (CU) | **304** (8x38) | **304** (same as left) | **256** (8x32) | **256** (same as left) |
| IOD Count | **4** | **4** | **2** | **2** |
| VRAM | 192 GiB HBM3 | 256 GiB HBM3E | 288 GiB HBM3E | 288 GiB HBM3E (same tier as MI355X) |
| HBM Peak Bandwidth (white paper) | Per SKU specifications | **6.0 TB/s** | **8 TB/s** | **8 TB/s** |
| L2 Aggregate Read Bandwidth | **34.4 TB/s** | **34.4 TB/s** (same architecture) | Per CDNA4 documentation | (same as left) |
| Infinity Cache Bandwidth | **17.2 TB/s** | **17.2 TB/s** (same architecture) | Per CDNA4 documentation | (same as left) |
| Wavefront | 64 | 64 | 64 | 64 |
| LDS / CU | 64 KiB | 64 KiB | **160 KiB** | **160 KiB** |
| L3 Cache | 256 MiB | 256 MiB | 256 MiB | 256 MiB |
| L2 Cache | 32 MiB (4 MiB/XCD) | 32 MiB (same as left) | 32 MiB (4 MiB/XCD) | 32 MiB (same as left) |
| L1 Vector / CU | 32 KiB | 32 KiB | 32 KiB | 32 KiB |
| VGPR File / CU | 512 KiB | 512 KiB | 512 KiB | 512 KiB |
| SGPR File / CU | 12.5 KiB | 12.5 KiB | 12.5 KiB | 12.5 KiB |
| Matrix Core Count | 1216 (4x304) | 1216 (same as left) | **1024** (4x256) | (same as left) |
| Matrix Peak (official rated) | FP16 1307.4 TF; BF16 1307.4 TF; FP8 2614.9 TF; FP64 163.4 TF | Same architecture, per SKU documentation | FP16 2.5 PF; FP8 5 PF; FP6 10 PF; FP4 10 PF | FP16 2.3 PF; FP8 4.6 PF; MXFP 9.2 PF (Table 2) |
| Max engine clock | 2100 MHz | See product specifications | **2400 MHz** | **2200 MHz** |
| TBP | See product specifications | See product specifications | **1400 W** (DLC) | **1000 W** (air-cooled) |
| FP8 Format Focus | CDNA3 commonly uses **FNUZ** context | (same as left) | **OCP** (E4M3FN + E5M2) | **OCP** (same as left) |
| New Features | — | — | FP6, FP4, MXFP | Same as MI355X |
| offload-arch | `gfx942` | `gfx942` | `gfx950` | `gfx950` |

Note: MI350X and MI355X are both **GFX950 / CDNA4**, with identical CU/XCD, cache, and interconnect architecture; **clock speed, power consumption, cooling, and Matrix PF** vary by SKU (see MI355X documentation **Table 2**).

### Infinity Fabric and Partition (Cross-Generation Differences Summary)

| Item | CDNA3 (MI300X/MI325X) | CDNA4 (MI350X/MI355X) |
|------|------------------------|-------------------------|
| Fabric Link Speed | **32 Gbps**/link | **38.4 Gbps**/link (approximately **+20%**) |
| Per-Link Bidirectional Bandwidth | **64 GB/s** | **76.8 GB/s** |
| P2P / Aggregate (CDNA4) | — | ring **1075.2 GB/s**; aggregate **1203.2 GB/s** (white paper) |
| Partition: QPX | **None** (commonly SPX / DPX / CPX) | **QPX**: **2** XCD/partition, **4** partitions, approximately **72 GB**/partition |
| NPS Options | **NPS1 / NPS2 / NPS4**, etc. (platform-dependent) | **NPS1** or **NPS2** (**per-IOD**); commonly recommended **DPX+NPS2** |

### FLOPS/clock/CU (Cross-Generation Compute Density)

The table below facilitates **per-CU** intensity comparison; on **MI355X**, **Matrix FP64** is **halved** relative to MI300X, while **Matrix FP16/BF16/FP8/INT8 (sparsity)** and **MXFP** paths are **improved** (see `mi300x.md` / `mi355x.md` for complete tables).

| Computation | MI300X | MI355X |
|-------------|--------|--------|
| Vector FP64 / FP32 / FP16 | 128 / 256 / 256 | (same as left) |
| Matrix FP64 | 256 | **128** |
| Matrix FP32 | 256 | 256 |
| Matrix FP16 / Sparsity | 2048 | **4096** |
| Matrix BF16 / Sparsity | 2048 | **4096** |
| Matrix FP8 / Sparsity | 4096 | **8192** |
| Matrix INT8 / Sparsity | 4096 | **8192** |
| Matrix MXFP6 / MXFP4 | N/A | **16384** |

**TF32**: CDNA4 has **no** dedicated TF32 hardware unit; requirements are met via **BF16** and other software-level paths (per white paper).

## MI300X vs NVIDIA H100 SXM Comparison

| Metric | MI300X | H100 SXM | MI300X Advantage |
|--------|--------|----------|-----------------|
| HBM Capacity | 192 GB | 80 GB | **2.4x** |
| HBM Bandwidth | 5.325 TB/s | 3.35 TB/s | **1.59x** |
| TDP | 750 W | 700 W | 1.07x |

### Inference Performance Implications

- **Decode phase**: MI300X excels in decode-heavy workloads (low-medium batch) due to higher memory bandwidth
- **Large models**: Llama-3.1-405B fits in a single MI300X node; avoid TP>1 for models <=30B
- **TP1 for small models**: Run 8 instances of TP1 for models <=70B FP8 to maximize throughput
- **KV cache advantage**: Larger HBM avoids cache eviction that degrades H100 at high batch sizes
- **Decode-heavy workloads**: MI300X shows **1.31x geomean throughput advantage** over H100 for Llama 3.1 405B FP8 (TP8); up to **1.62x** at long output lengths (128 input / 4096 output tokens)

## Atomic Operations Support

| Operation | CDNA1 | CDNA2+ | Notes |
|-----------|-------|--------|-------|
| int32/int64 atomics | Yes | Yes | Widely supported |
| float32 atomics | Yes | Yes | Global memory |
| float64 atomics | No | Yes | CDNA2+ only |
| 2x float16 atomics | Yes | Yes | Packed half-precision |
| 2x bfloat16 atomics | Yes | Yes | Packed BF16 |

## Cross-Hardware Porting Checklist

1. **Compilation**: Update `--offload-arch` (`gfx942` <-> `gfx950`); see multi-target example below.
2. **Tile/Block size**: Re-tune based on CU count, LDS/CU (especially important when crossing generations from 64 KiB to 160 KiB), and memory hierarchy.
3. **Data types**: When migrating CDNA3 -> CDNA4, verify **FP8 (FNUZ vs OCP)** and **FP6/FP4/MXFP** availability and semantics.
4. **Performance baseline**: Re-benchmark; peak TFLOPS/PF and cache capacities have changed across generations.
5. **Conditional compilation**: Use `#if defined(__gfx942__)` / `#if defined(__gfx950__)` (verify actual compiler predefined macros) for hardware specialization.
6. **Profiling thresholds**: Qualitative ranges for occupancy, bandwidth, and MFMA utilization can be retained; absolute values must be recalibrated per device.

## When to Use Multi-Target Compilation

If code needs to run on both **gfx942** and **gfx950**:

```bash
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp
```

The compiler generates code for each target; runtime dispatch or conditional compilation then layers in generation-specific specialization paths for LDS/FP8 differences.
