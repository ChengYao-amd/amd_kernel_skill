# GPU Performance Counters Reference

> Complete reference for AMD GPU hardware performance counters on MI300/MI200 (CDNA3) and MI350 (CDNA4) series. For profiling workflow and decision-tree usage, see `profiling-decision-tree.md`. For tool usage, see `rocprof-guide.md` and `omniperf-guide.md`.

---

## 1. Counter Categories Overview

| Category | Prefix | Primary Use |
|----------|--------|-------------|
| Command Processor Fetcher | `CPF_` | Pipeline stall analysis |
| Command Processor Compute | `CPC_` | Packet decode, TLB analysis |
| Graphics Register Bus Manager | `GRBM_` | Global GPU busy/idle monitoring |
| Shader Processor Input | `SPI_` | Wave dispatch, resource bottleneck detection |
| Compute Unit -- Instruction Mix | `SQ_INSTS_*` | Instruction type classification |
| Compute Unit -- MFMA Ops | `SQ_INSTS_VALU_MFMA_MOPS_*` | Matrix core FLOP counting (unit = 512 FLOPs) |
| Compute Unit -- Wavefront | `SQ_WAVES*` | Wave dispatch and occupancy |
| Compute Unit -- LDS | `SQ_LDS_*` | LDS bank conflict and stall analysis |
| Instruction Cache (L1i) / Scalar L1d | `SQC_*` | Instruction and constant cache hit/miss |
| Vector L1d (TCP) | `TCP_*` | Vector L1D access, hit, miss, latency |
| Texture Addressing | `TA_*` | Buffer/flat wavefront processing |
| Texture Data | `TD_*` | Data path stall analysis |
| L2 Cache | `TCC_*` | L2 access, hit, miss, bandwidth |

---

## 2. Command Processor Counters

| Counter | Description |
|---------|-------------|
| `CPF_CPF_STAT_BUSY` | CPF busy cycles |
| `CPF_CPF_STAT_IDLE` | CPF idle cycles |
| `CPF_CPF_STAT_STALL` | CPF stall cycles |
| `CPC_CPC_STAT_BUSY` | CPC busy cycles |
| `CPC_CPC_STAT_IDLE` | CPC idle cycles |
| `CPC_CPC_STAT_STALL` | CPC stall cycles |

**CDNA4 additions:**

| Counter | Description |
|---------|-------------|
| `CPC_ADC_DISPATCH_ALLOC_DONE` | Multi-XCD coordination: dispatch allocation complete |
| `CPC_STALLED_BY_SE*_SPI` | CPC stalled waiting for Shader Engine SPI |

---

## 3. GRBM (Global Activity) Counters

| Counter | Description |
|---------|-------------|
| `GRBM_GUI_ACTIVE` | GPU active cycles (any engine doing work) |
| `GRBM_COUNT` | Free-running clock cycle counter (denominator for utilization ratios) |

**GPU utilization:**
```
GPU_Utilization = GRBM_GUI_ACTIVE / GRBM_COUNT
```

---

## 4. SPI (Shader Processor Input) -- Occupancy Stall Counters

These counters identify WHY waves cannot be dispatched:

| Counter | Stall Reason |
|---------|-------------|
| `SPI_CSN_WAVE` | Total wavefronts dispatched by SPI |
| `SPI_RA_WAVE_SIMD_FULL_CSN` | SIMD wave slots exhausted |
| `SPI_RA_VGPR_SIMD_FULL_CSN` | VGPR shortage preventing wave allocation |
| `SPI_RA_SGPR_SIMD_FULL_CSN` | SGPR shortage preventing wave allocation |
| `SPI_RA_LDS_CU_FULL_CSN` | LDS capacity exhausted on CU |
| `SPI_RA_BAR_CU_FULL_CSN` | Barrier resource exhausted on CU |
| `SPI_RA_WVLIM_STALL_CSN` | WAVE_LIMIT configuration preventing dispatch |

**Interpretation guide:**

- High `SPI_RA_VGPR_SIMD_FULL_CSN` --> reduce VGPR usage via `__launch_bounds__` or algorithmic changes.
- High `SPI_RA_LDS_CU_FULL_CSN` --> reduce per-block LDS allocation.
- High `SPI_RA_BAR_CU_FULL_CSN` --> reduce barrier usage or block size.
- High `SPI_RA_WVLIM_STALL_CSN` --> a user-set wave limit is constraining occupancy.

---

## 5. Compute Unit -- Instruction Mix Counters

### 5.1 General Instruction Counters

| Counter | Description |
|---------|-------------|
| `SQ_INSTS` | Total instructions issued |
| `SQ_INSTS_VALU` | Vector ALU instructions (may or may not include MFMA depending on doc version) |
| `SQ_INSTS_MFMA` | Matrix FMA instruction count |
| `SQ_INSTS_VMEM` | Vector memory instructions (global/flat loads and stores) |
| `SQ_INSTS_VMEM_RD` | Vector memory read instructions |
| `SQ_INSTS_VMEM_WR` | Vector memory write instructions |
| `SQ_INSTS_LDS` | LDS instructions (aggregate on MI300; separated on MI350) |
| `SQ_INSTS_SALU` | Scalar ALU instructions |
| `SQ_INSTS_SMEM` | Scalar memory instructions |
| `SQ_INSTS_SMEM_NORM` | Normalized scalar memory instructions (for latency calculation) |
| `SQ_IFETCH` | Instruction fetch count |
| `SQ_BUSY_CYCLES` | CU busy cycles |

### 5.2 MFMA Instruction Counters (by precision)

| Counter | Data Type | Unit |
|---------|-----------|------|
| `SQ_INSTS_VALU_MFMA_F16` | FP16 MFMA | Instructions |
| `SQ_INSTS_VALU_MFMA_BF16` | BF16 MFMA | Instructions |
| `SQ_INSTS_VALU_MFMA_F32` | FP32 MFMA | Instructions |
| `SQ_INSTS_VALU_MFMA_F64` | FP64 MFMA | Instructions |
| `SQ_INSTS_VALU_MFMA_I8` | INT8 MFMA | Instructions |

### 5.3 MFMA Throughput Counters (MOPS, unit = 512 FLOPs)

| Counter | Data Type |
|---------|-----------|
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | FP16 matrix operations |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | BF16 matrix operations |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` | FP32 matrix operations |
| `SQ_INSTS_VALU_MFMA_MOPS_F64` | FP64 matrix operations |
| `SQ_INSTS_VALU_MFMA_MOPS_I8` | INT8 matrix operations |

### 5.4 MFMA Pipeline Counter

| Counter | Description |
|---------|-------------|
| `SQ_VALU_MFMA_BUSY_CYCLES` | Cycles where MFMA ALU is busy (key for compute utilization) |

---

## 6. Wavefront Counters

| Counter | Description |
|---------|-------------|
| `SQ_WAVES` | Total wavefronts dispatched to sequencer (includes restores) |
| `SQ_WAVES_EQ_64` | Waves with all 64 lanes active (fully utilized) |
| `SQ_WAVES_LT_64` | Waves with fewer than 64 active lanes |
| `SQ_WAVES_LT_48` | Waves with fewer than 48 active lanes |
| `SQ_WAVES_LT_32` | Waves with fewer than 32 active lanes (significant SIMD waste) |
| `SQ_WAVES_LT_16` | Waves with fewer than 16 active lanes |
| `SQ_LEVEL_WAVES` | In-flight waves at sample time (occupancy snapshot) |

**Partial wave detection:** If `SQ_WAVES_LT_32` is significant relative to `SQ_WAVES`, the kernel has excessive lane waste, often from divergent control flow or tail effects.

---

## 7. LDS Counters

### 7.1 MI300/MI200 (CDNA2/3)

| Counter | Description |
|---------|-------------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict stall cycles |
| `SQ_LDS_ADDR_CONFLICT` | LDS address conflict stall cycles |
| `SQ_LDS_UNALIGNED_STALL` | Flat unaligned operation stall cycles |
| `SQ_LDS_MEM_VIOLATIONS` | LDS memory violation thread count |

### 7.2 MI350 (CDNA4) -- Granular LDS Counters

| Counter | Description |
|---------|-------------|
| `SQ_INSTS_LDS_LOAD` | LDS load instructions |
| `SQ_INSTS_LDS_STORE` | LDS store instructions |
| `SQ_INSTS_LDS_ATOMIC` | LDS atomic instructions |
| `SQ_INSTS_LDS_LOAD_BANDWIDTH` | Actual LDS load bandwidth (64-byte units) |
| `SQ_LDS_DATA_FIFO_FULL` | LDS data FIFO full stall cycles |
| `SQ_LDS_CMD_FIFO_FULL` | LDS command FIFO full stall cycles |

---

## 8. Vector L1 Cache (TCP) Counters

| Counter | Description |
|---------|-------------|
| `TCP_TOTAL_CACHE_ACCESSES` | Total vL1D cache accesses |
| `TCP_TCC_READ_REQ` | Read requests forwarded to L2 (L1 misses) |
| `TCP_READ_TAGCONFLICT_STALL_CYCLES` | Tag conflict stall cycles |

**CDNA4 additions:**

| Counter | Description |
|---------|-------------|
| `TCP_TCP_LATENCY` | Total wave latency through TCP |
| `TCP_TCC_READ_REQ_LATENCY` | Read request latency to L2 |
| `TCP_TCC_WRITE_REQ_LATENCY` | Write request latency to L2 |

---

## 9. L2 Cache (TCC) Counters

| Counter | Description |
|---------|-------------|
| `TCC_HIT` | L2 cache hit count |
| `TCC_MISS` | L2 cache miss count |
| `TCC_EA_RDREQ_DRAM_32B` | 32-byte read requests to HBM (DRAM) |
| `TCC_EA_WRREQ_64B` | 64-byte write requests to external memory |

**CDNA4 additions:**

| Counter | Description |
|---------|-------------|
| `TCC_READ_SECTORS` | L2 read sector count |
| `TCC_WRITE_SECTORS` | L2 write sector count |
| `TCC_EA0_RDREQ_DRAM_32B` | Per-channel 32-byte DRAM read requests |

**Multi-instance note:** On MI300 series, TCC counters may have per-instance `[n]` suffixes or `_sum` aggregate forms depending on the tool version.

---

## 10. Texture Addressing / Data (TA/TD) Counters

| Counter | Description |
|---------|-------------|
| `TA_FLAT_READ_WAVEFRONTS` | Flat read wavefronts processed by TA |
| `TA_BUFFER_READ_WAVEFRONTS` | Buffer read wavefronts processed by TA |
| `TA_FLAT_WRITE_WAVEFRONTS` | Flat write wavefronts processed by TA |
| `TA_BUFFER_WRITE_WAVEFRONTS` | Buffer write wavefronts processed by TA |

---

## 11. CDNA4 (MI350) -- New Counter Categories

### 11.1 New Precision Counters

| Counter | Description |
|---------|-------------|
| `SQ_INSTS_VALU_MFMA_F6F4` | FP6/FP4 MFMA instruction count |
| `SQ_INSTS_VALU_MFMA_MOPS_F6F4` | FP6/FP4 MFMA operations (unit = 512 FLOPs) |

### 11.2 Per-Instruction FLOP Counters (new in CDNA4)

| Counter | Description |
|---------|-------------|
| `SQ_INSTS_VALU_FLOPS_FP16` | FP16 FLOPs per instruction |
| `SQ_INSTS_VALU_FLOPS_FP32` | FP32 FLOPs per instruction |
| `SQ_INSTS_VALU_FLOPS_FP64` | FP64 FLOPs per instruction |
| `SQ_INSTS_VALU_IOPS` | Integer operations per instruction |

### 11.3 Dual-Issue Detection

| Counter | Description |
|---------|-------------|
| `SQ_ACTIVE_INST_VALU2` | Cycles with two VALU instructions issued simultaneously |

> CDNA4 supports dual VALU issue. When this counter is high, do not assume high VALU throughput is waste -- compare against the CDNA4 peak table.

### 11.4 Resource Stall Counters

| Counter | Description |
|---------|-------------|
| `SQ_VMEM_TA_ADDR_FIFO_FULL` | VMEM address FIFO full stall |
| `SQ_VMEM_TA_CMD_FIFO_FULL` | VMEM command FIFO full stall |

---

## 12. Key Derived Formulas

### 12.1 Latency Calculations

| Metric | Formula |
|--------|---------|
| Vector memory latency (cycles/instruction) | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_VMEM` |
| LDS latency (cycles/instruction) | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_LDS` |
| Scalar memory latency (cycles/instruction) | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_SMEM_NORM` |
| Instruction fetch latency (cycles/fetch) | `SQ_ACCUM_PREV_HIRES / SQ_IFETCH` |
| Wave latency (cycles/wave) | `SQ_ACCUM_PREV_HIRES / SQ_WAVES` |

### 12.2 Utilization Calculations

| Metric | Formula |
|--------|---------|
| MFMA pipe busy ratio | `SQ_VALU_MFMA_BUSY_CYCLES / SQ_BUSY_CYCLES` |
| GPU utilization | `GRBM_GUI_ACTIVE / GRBM_COUNT` |

### 12.3 Cache Hit Rates

| Metric | Formula |
|--------|---------|
| L2 hit rate | `TCC_HIT / (TCC_HIT + TCC_MISS)` |
| L1 hit rate (from TCP) | `(TCP_TOTAL_CACHE_ACCESSES - TCP_TCC_READ_REQ) / TCP_TOTAL_CACHE_ACCESSES` |

### 12.4 HBM Bandwidth

```
HBM_BW_GB_per_s = (TCC_EA_RDREQ_DRAM_32B + TCC_EA_WRREQ_32B) * 32 / time_seconds / 1e9
HBM_BW_util = HBM_BW_GB_per_s / peak_HBM_BW_GB_per_s
```

---

## 13. ROCm Compute Profiler Metric Sets

When using `rocprof-compute profile`, the `--set` flag selects single-pass metric groups:

| Set Name | Counters Included |
|----------|-------------------|
| `compute_thruput_util` | SALU utilization, VALU utilization, VMEM utilization, Branch utilization |
| `launch_stats` | Grid size, workgroup size, total wavefronts, VGPRs, AGPRs, SGPRs, LDS allocation, scratch allocation |

### Roofline Profiling

```bash
rocprof-compute profile --roof-only -- <command>
rocprof-compute profile --mem-level HBM,L2,L1,LDS -- <command>
rocprof-compute profile --roofline-data-type FP8 -- <command>
```

### MI200 Empirical Roofline Reference Values

| Metric | Measured Value |
|--------|---------------|
| HBM bandwidth | 1388.0 GB/s |
| L2 bandwidth | 5020.8 GB/s |
| L1 bandwidth | 9229.5 GB/s |
| LDS bandwidth | 17645.6 GB/s |
| Peak FP32 VALU | 20986.9 GFLOPS |
| Peak FP64 VALU | 20408.0 GFLOPS |
| Peak MFMA BF16 | 170280.0 GFLOPS |
| Peak MFMA F16 | 164733.6 GFLOPS |
| Peak MFMA F32 | 41399.6 GFLOPS |
| Peak MFMA F64 | 41379.2 GFLOPS |
| Peak MFMA I8 | 166281.9 GOPS |

---

## 14. Counter Collection Notes

### Multi-GCD / Multi-XCD (MI300 Series)

- MI300X has 8 XCDs. Some counters are per-XCD and must be summed for aggregate values.
- TCC counters may appear with `[n]` instance suffixes or `_sum` aggregate forms.
- When profiling with `rocprof-compute`, the tool handles aggregation automatically.

### Counter Multiplexing

- Not all counters can be collected simultaneously (limited PMC slots per pass).
- `rocprof-compute` automatically handles multi-pass collection via its perfmon input files.
- When using `rocprofv3` directly, group related counters to minimize passes.

### Naming Variations

- Counter names may vary slightly between `rocprof` (v1), `rocprofv3`, and `rocprof-compute`.
- Always verify available counters with `rocprofv3 --list-counters` on the target system.
- MI300 and MI200 share most counter names; MI350 adds new counters (see Section 11).
