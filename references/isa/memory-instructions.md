# Memory Instruction Reference

## Global Memory

| Instruction | Width | Description |
|-------------|-------|-------------|
| global_load_dword | 4B | Single lane |
| global_load_dwordx2 | 8B | Vectorized |
| global_load_dwordx4 | 16B | Best for coalesced access |
| global_store_dword[x2/x4] | 4-16B | Same widths |

**Coalescing rule**: 64 consecutive lanes accessing 64 consecutive elements = one coalesced transaction. Stride > 1 reduces bandwidth.

## LDS (Local Data Share)

| Instruction | Width | Description |
|-------------|-------|-------------|
| ds_read_b32 | 4B | Single bank |
| ds_read_b64 | 8B | Two banks |
| ds_read_b128 | 16B | Four banks |
| ds_write_b32/b64/b128 | 4-16B | Same widths |
| ds_swizzle_b32 | 4B | Lane permutation without LDS read/write |

**Bank count, capacity, and read bandwidth (generational differences, CDNA3/4 whitepapers)**:

| Item | CDNA3 | CDNA4 |
|------|-------|-------|
| LDS capacity/CU | Approximately **half** of CDNA4 (relative to CDNA4's **160 KB/CU**) | **160 KB/CU** |
| Number of banks | **32** | **64** (**doubled** relative to CDNA3) |
| LDS read bandwidth | **128 B/clock** (approximately half of CDNA4) | **256 B/clock** (both capacity and read bandwidth are approximately **2x** relative to CDNA3) |

- **CDNA3 (e.g., MI300X / gfx942)**: **32-bank LDS**, each bank **512 entries** of **4B** = **64 KB** (ISA spec: "configured with 32 banks, each with 512 entries of 4 bytes").
- **CDNA4 (e.g., MI355X / gfx950)**: **64-bank LDS**, each bank **640 entries** of **4B** = **160 KB** (ISA spec: "configured with 64 banks, each with 640 entries of 4 bytes"). Both generations have **32 integer atomic units**. **Bank conflict patterns and padding calculations differ from CDNA3** -- do not reuse fixed padding based on the 32-bank assumption; when migrating to CDNA4, re-derive swizzle / padding based on **64 banks**.
- **Direct LDS load (CDNA4 new path)**: On CDNA4, LDS supports **loading data directly from the L1 data cache** (whitepaper: "reducing vector register usage and latency"), which is distinct from the `buffer_load_lds` / `GLOBAL_LOAD_LDS_*` global-to-LDS paths. During tuning, compare these three LDS fill paths via profiling.

**L1 / L2 cache (common aspects between CDNA3 and CDNA4, and bandwidth figures)**:

- **L1 (data)**: **32 KB**, **128 B** cache line, **64-way** set-associative (same for CDNA3 / CDNA4).
- **L2 (per XCD)**: **4 MB**, **16-way**, **16 channels**; each channel reads **128 B** and writes **64 B** per cycle; **per-XCD L2 read bandwidth is ~2 KB/clock**.
- **L2 aggregate read bandwidth (8-XCD scale)**: CDNA3 ~**34.4 TB/s**; CDNA4 ~**32 TB/s** (cross-generation figures are per whitepaper specifications; actual values vary with product and configuration).

**Bank conflict rule**: When multiple lanes hit different words in the same bank within the same cycle -> conflict -> serialization. Mitigate through **padding**, **swizzle**, or access pattern rearrangement.

**LDS allocation granularity** (CDNA4 ISA):
- LDS is allocated per work-group (or per-wavefront when work-groups are not in use) in contiguous blocks of **1280 bytes** on **1280-byte alignment**.
- LDS allocations do **not** wrap around the LDS storage.
- Clamping is controlled by two size registers: the SPI-allocated LDS size and M0; operations use the **smaller** of the two.

**LDS indexed access timing** (CDNA4 ISA):
- Operations can complete in as little as **2 cycles** or take up to **64 cycles** depending on bank conflicts.
- Reads across a wavefront are dispatched over **4 cycles** in waterfall fashion.
- All 64 banks (CDNA4) or 32 banks (CDNA3) can execute a store or load simultaneously when conflict-free.

### MFMA Transpose Load from LDS (CDNA4 Only)

CDNA4 introduces dedicated instructions that perform **matrix transpose while transferring data from LDS to VGPRs**, eliminating the need for explicit transpose kernels or complex swizzle patterns:

| Instruction | Element Size | Description |
|------------|-------------|-------------|
| `DS_READ_B64_TR_B16` | 16-bit | Column-major A or row-major B load; two instructions load a complete matrix; each lane holds 4 consecutive M/N values |
| `DS_READ_B64_TR_B8` | 8-bit | Two instructions load a complete matrix for FP8/INT8 tile data |
| `DS_READ_B64_TR_B4` | 4-bit | Two instructions load a complete matrix for FP4 tile data |
| `DS_READ_B96_TR_B6` | 6-bit | Three VGPRs per instruction for FP6 tile data; does **not** require even-VGPR alignment |

**Requirements**: EXEC mask must be set to all 1s; LDS address must be aligned to the data size; any DS op reading/writing 64-bit or larger data must use an even-aligned VGPR (except `DS_READ_B96_TR_B6`).

### Global/Buffer Load Direct to LDS (CDNA4 ISA)

CDNA4 ISA provides **GLOBAL_LOAD_LDS_*** instructions for direct global-to-LDS data movement **without passing through VGPRs**:

| Instruction | Description |
|------------|-------------|
| `GLOBAL_LOAD_LDS_UBYTE` / `_SBYTE` | Load byte from global memory directly to LDS |
| `GLOBAL_LOAD_LDS_USHORT` / `_SSHORT` | Load 16-bit from global memory directly to LDS |
| `GLOBAL_LOAD_LDS_DWORD` | Load 32-bit from global memory directly to LDS |
| `GLOBAL_LOAD_LDS_DWORDX3` | Load 96-bit from global memory directly to LDS |
| `GLOBAL_LOAD_LDS_DWORDX4` | Load 128-bit from global memory directly to LDS |

LDS addressing for these instructions: `LDS_ADDR = LDSbase(hw alloc) + LDSoffset(M0[17:2] * 4) + INST.OFFSET + ThreadID * {4,16}` (DWORDX1 uses `*4`, DWORDX4 uses `*16`).

These instructions only increment **VM_CNT** (not LGKM_CNT), since they go through the global memory path. Combined with `buffer_load_*_lds` (buffer path), these provide two complementary VGPR-free data paths to LDS.

## Data Movement Engine (DME)

**CDNA3 / CDNA4** are equipped with a **Data Movement Engine (DME)** for asynchronous **HBM -> LDS** data transfer paths (complementing explicit global loads on vector cores).

- **Reduces VMEM pressure**: Some data paths can use DME, reducing contention for `global_load` / flat load.
- **Overlap of compute and data movement**: Combined with MFMA, LDS pipelining, and mechanisms like `buffer_load_lds`, this facilitates **double buffering** and software pipelining.

During tuning, if the bottleneck is VMEM, consult the target ISA / ROCm documentation for **async DMA / buffer_load_lds** and DME-related constraints, and compare VMEM vs LDS timelines in the profiler.

## Buffer vs Flat Instructions

| Type | Use Case |
|------|----------|
| `buffer_load_*` | Known base + offset, supports bounds checking, slightly faster |
| `global_load_*` (flat) | Arbitrary 64-bit address, simpler codegen |

The compiler typically selects automatically. When using inline assembly, prefer buffer instructions when the base address is a uniform value (SGPR).

**Flat instruction caveats** (CDNA4 ISA):
- Flat instructions **cannot** return data directly to LDS (only to VGPRs); use buffer or global instructions for direct-to-LDS paths.
- Internally, FLAT instructions are executed as **both** an LDS and a Buffer instruction, incrementing **both** `VM_CNT` and `LGKM_CNT`. The only safe `s_waitcnt` value after FLAT instructions is **zero**.
- Flat instructions can complete **out of order** with each other; if two flat loads return data to the same VGPR, results are undefined.

## Memory Barriers and Synchronization

### s_waitcnt -- Critical Instruction

```
s_waitcnt vmcnt(N) lgkmcnt(M) expcnt(K)
```

| Counter | Tracks | Wait Condition |
|---------|--------|----------------|
| vmcnt | Global load/store | N outstanding VMEM operations remaining |
| lgkmcnt | LDS + SMEM operations | M outstanding LDS/SMEM operations remaining |
| expcnt | Export (GDS, LDS->VGPR) | K outstanding exports remaining |

**Strategy**: Do not use `s_waitcnt 0` everywhere (it kills ILP). Count the outstanding operations and only wait for what is truly needed.

```
global_load_dwordx4 v[0:3], ...    // vmcnt = 1
global_load_dwordx4 v[4:7], ...    // vmcnt = 2
// ... do other work ...
s_waitcnt vmcnt(1)                  // only wait for the first load
v_add_f32 v8, v0, v1               // use the first load's result
s_waitcnt vmcnt(0)                  // now wait for the second load
v_add_f32 v9, v4, v5               // use the second load's result
```
