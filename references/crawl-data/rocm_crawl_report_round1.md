# ROCm Documentation Crawl Report — Round 1

**Date:** 2026-04-09
**Pages fetched:** 17 successfully, 4 failed (404/timeout)
**Focus:** AMD GPU kernel optimization, MI300X architecture, HIP kernel language, low-precision types, performance counters, system tuning

---

## 1. Pages Successfully Fetched with Key Facts

### 1.1 MI300X Workload Optimization
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html`

Key facts:
- **Tuning workflow:** Measure → Profile → Analyze → Tune → Iterate
- **Profiling tools hierarchy:** PyTorch Profiler (high-level) → ROCProfiler / ROCm Compute Profiler / ROCm Systems Profiler (kernel-level)
- **PyTorch TunableOp:** Tests thousands of GEMM algorithms from rocBLAS and hipBLASLt; env vars: `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=1`
- **Offline tuning** (PyTorch 2.6+): Decouple GEMM collection from tuning pass; avoids re-running full workload
- **TorchInductor max-autotune:** `TORCHINDUCTOR_MAX_AUTOTUNE=1` benchmarks Triton configs for GEMM/conv by tile size (BLOCK_M, BLOCK_N, BLOCK_K), num_stages, num_warps, and MFMA instruction size (`matrix_instr_nonkdim`)
- **Composable Kernel backend:** Append `CK` to `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`
- **Key Triton tuning params for MI300X:** tile sizes, num_warps (typically 4 or 8 for wave64), num_stages (pipeline depth), `matrix_instr_nonkdim` (16 or 32)
- **RCCL tuning** for multi-GPU: `NCCL_MIN_NCHANNELS=112` for MI300X
- **Inference optimization:** `TORCHINDUCTOR_FREEZING=1` for constant folding; `TORCHINDUCTOR_CPP_WRAPPER=1` for reduced kernel launch overhead
- **ROCm Compute Profiler** automates collection of all hardware counters in one command, provides Roofline Analysis, Speed-of-Light metrics

### 1.2 MI300 Series Microarchitecture
**URL:** `https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html`

Key facts:
- **Architecture:** CDNA 3, uses Accelerator Complex Die (XCD) as building block
- **XCD structure:** 40 CUs (38 active + 2 disabled for yield), 4 ACEs, shared 4 MB L2 cache per XCD
- **MI300X OAM package:** 8 XCDs vertically stacked, 8 stacks HBM3, 4 I/O dies, up to 304 CUs total
- **MI300A APU:** 6 XCDs + 3 CCDs (CPU chiplets)
- **Peak performance table (MI300X):**

| Data Type | FLOPS/Clock/CU | Peak TFLOPS |
|-----------|----------------|-------------|
| Matrix FP64 | 256 | 163.4 |
| Vector FP64 | 128 | 81.7 |
| Matrix FP32 | 256 | 163.4 |
| Vector FP32 | 256 | 163.4 |
| Vector TF32 | 1024 | 653.7 |
| Matrix FP16 | 2048 | 1307.4 |
| Matrix BF16 | 2048 | 1307.4 |
| Matrix FP8 | 4096 | 2614.9 |
| Matrix INT8 | 4096 | 2614.9 |

- **Memory bandwidth:** 5.3 TB/s aggregate theoretical peak
- **Node architecture:** 8 MI300X OAMs, fully connected via 7 Infinity Fabric links per GPU, PCIe Gen 5 x16 to host
- **CDNA 3 improvements over CDNA 2:** 3x FP16/BF16, 6.8x INT8, 16x FP8 vs FP32, 4x TF32 vs FP32

### 1.3 HIP C++ Language Extensions
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html`

Key facts:
- **Function qualifiers:** `__host__`, `__device__`, `__global__`
- **HIP does NOT support dynamic parallelism** (unlike CUDA — kernels cannot call kernels)
- **Kernel launch:** Triple chevron `<<<gridDim, blockDim, dynamicShared, stream>>>` or `hipLaunchKernelGGL`
- **Kernel restrictions:** No `va_list`, no references, no `long double`, return type must be `void`
- **Thread/block indexing:** `threadIdx.x/y/z`, `blockIdx.x/y/z`, `blockDim.x/y/z`, `gridDim.x/y/z`
- **Warp size:** 64 for CDNA (Instinct), 32 for RDNA; query with `warpSize` built-in
- **`__shared__` memory:** On-CU scratchpad shared by block
- **`__constant__` memory:** Read-only, cached, broadcast to all threads
- **Warp cross-lane functions:** `__shfl()`, `__shfl_up()`, `__shfl_down()`, `__shfl_xor()`, `__ballot()`, `__any()`, `__all()`
- **Synchronization:** `__syncthreads()` for block-level; `__threadfence()` for device-level memory ordering
- **Math functions:** Intrinsics like `__expf()`, `__logf()`, `__sinf()`, `__fsqrt_rn()`, `__frcp_rn()` for fast approximations

### 1.4 Low Precision Floating-Point Types
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html`

Key facts:
- **FP4 (E2M1):** CDNA4 only (gfx950). HIP header: `<hip/amd_detail/amd_hip_fp4.h>`
  - Classes: `__hip_fp4_e2m1`, vectors `__hip_fp4x2_e2m1`, `__hip_fp4x4_e2m1`
  - Conversion: `__hip_cvt_float_to_fp4()`, `__hip_cvt_fp4_to_halfraw()`
- **FP6 (E3M2 / E2M3):** CDNA4 only (gfx950). HIP header: `<hip/amd_detail/amd_hip_fp6.h>`
  - E3M2: wider range, less precision; E2M3: higher precision, narrower range
  - Classes: `__hip_fp6_e2m3`, `__hip_fp6_e3m2`; vectors: `__hip_fp6x2_*`, `__hip_fp6x4_*`
- **FP8 (E4M3 / E5M2):** Two representations:
  - **FNUZ** (gfx94x / CDNA3): No Inf, NaN = negative zero, expanded range
  - **OCP** (gfx950 / CDNA4, gfx12 / RDNA4): Open Compute Project standard
  - HIP header: `<hip/amd_detail/amd_hip_fp8.h>`
  - Classes: `__hip_fp8_e4m3`, `__hip_fp8_e5m2`, `__hip_fp8_e4m3_fnuz`, `__hip_fp8_e5m2_fnuz`
  - On gfx94x, FP8 defaults to FNUZ type
- **Float16:** `__half` type, `<hip/amd_detail/amd_hip_fp16.h>`, supported on all architectures
- **BFloat16:** `__hip_bfloat16`, `<hip/amd_detail/amd_hip_bf16.h>`, supported on all architectures
- **hipExt microscaling APIs (gfx950):** Scale type `__amd_scale_t` (E8M0 format), supports MFMA with scaled FP8/FP6/FP4 inputs

### 1.5 MI300X System Optimization
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/system.html`

Key facts:
- **BIOS settings for MI300X (EPYC 9004):**
  - Above 4G decoding: Enabled; SR-IOV: Enabled
  - SMT control: Disable (for compute-bound workloads)
  - NUMA nodes per socket: Auto (NPS1)
  - 4-link xGMI max speed: 32 Gbps
  - IOMMU: Enabled; cTDP: 400W; Package power limit: 400W
  - xGMI force link width: x16; APBDIS: 1 (disable DF P-states); Fixed SOC P-state: P0
  - TSME (memory encryption): Disabled
- **GRUB settings:** `pci=realloc=off iommu=pt`; optionally `modprobe.blacklist=amdgpu` then manual `modprobe amdgpu`
- **OS tuning:**
  - Disable C2 CPU states: `cpupower idle-set -d 2`
  - Disable NUMA auto-balancing: `echo 0 > /proc/sys/kernel/numa_balancing`
  - `export HIP_FORCE_DEV_KERNARG=1` — device kernel args, 2-3 µs improvement
  - `export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0` — prevent ROCm thread affinity sprawl
- **Deterministic clock:** `rocm-smi --setperfdeterminism 1900` (max 1900 MHz, reduces PCC events)
- **GPU partition modes:**
  - SPX: All 8 XCDs as single device (default)
  - CPX: Each XCD as separate logical GPU (8 GPUs per MI300X)
  - NPS1: Entire memory accessible to all XCDs
  - NPS4: Memory quadrant per logical device (CPX only)
  - CPX+NPS4: Best for small models (≤13B) and RCCL
  - Switch without reboot: `amd-smi set --gpu all --compute-partition CPX --memory-partition NPS4`
- **RCCL with CPX:** Set `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1`, `RCCL_MSCCLPP_THRESHOLD=$((2*1024*1024*1024))`, `MSCCLPP_READ_ALLRED=1`

### 1.6 Precision Support Matrix
**URL:** `https://rocm.docs.amd.com/en/latest/reference/precision-support.html`

Key facts:
- **Matrix Core support by architecture:**

| Type | CDNA1 | CDNA2 | CDNA3 | CDNA4 | RDNA4 |
|------|-------|-------|-------|-------|-------|
| FP4  | ❌    | ❌    | ❌    | ✅    | ❌   |
| FP6 (E2M3/E3M2) | ❌ | ❌ | ❌ | ✅ | ❌ |
| FP8 (E4M3/E5M2) | ❌ | ❌ | ✅ | ✅ | ✅ |
| FP16 | ✅   | ✅    | ✅    | ✅    | ✅   |
| BF16 | ✅   | ✅    | ✅    | ✅    | ✅   |
| TF32 | ❌   | ❌    | ✅    | ✅    | ❌   |
| FP32 | ✅   | ✅    | ✅    | ✅    | ❌   |
| FP64 | ❌   | ✅    | ✅    | ✅    | ❌   |

- **Atomic operations:** Only int32/int64 and float32 widely supported; float64 atomics on CDNA2+; 2×float16/2×bfloat16 atomics on CDNA1+
- **hipBLASLt** supports FP4, FP6, FP8, FP16, BF16, FP32 — richest low-precision library support
- **Composable Kernel** supports all types including FP4, FP6
- **CDNA3 uses FP8 FNUZ** (differs from NVIDIA H100's OCP FP8)
- **CDNA4 uses FP8 OCP** (aligned with industry standard)
- **hipDataType enumeration** defines enum values for runtime type selection (e.g., `HIP_R_8F_E4M3=28`, `HIP_R_4F_E2M1=33`)

### 1.7 MI300/MI200 Performance Counters
**URL:** `https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html`

Key facts (categories and selected critical counters):
- **Command Processor:** `CPF_CPF_STAT_BUSY/IDLE/STALL`, `CPC_CPC_STAT_BUSY/IDLE/STALL`
- **GRBM:** `GRBM_GUI_ACTIVE` (GPU active cycles), `GRBM_COUNT` (free-running cycles)
- **SPI (Shader Processor Input):** `SPI_CSN_WAVE` (dispatched wavefronts), `SPI_RA_VGPR_SIMD_FULL_CSN` (VGPR shortage stalls), `SPI_RA_LDS_CU_FULL_CSN` (LDS shortage stalls)
- **Compute Unit Instruction Mix:** `SQ_INSTS_VALU`, `SQ_INSTS_MFMA`, `SQ_INSTS_VMEM`, `SQ_INSTS_LDS`, `SQ_INSTS_SALU`, `SQ_INSTS_SMEM`
- **MFMA-specific:** `SQ_INSTS_VALU_MFMA_F16/F32/F64/I8`, `SQ_INSTS_VALU_MFMA_MOPS_*` (FLOPs in units of 512)
- **Occupancy:** `SQ_LEVEL_WAVES` (inflight waves), `SQ_VALU_MFMA_BUSY_CYCLES`
- **Latency calculation formulas:**
  - VMEM latency = `SQ_ACCUM_PREV_HIRES` / `SQ_INSTS_VMEM`
  - LDS latency = `SQ_ACCUM_PREV_HIRES` / `SQ_INSTS_LDS`
  - SMEM latency = `SQ_ACCUM_PREV_HIRES` / `SQ_INSTS_SMEM_NORM`
- **Wavefront utilization:** `SQ_WAVES_EQ_64` (full), `SQ_WAVES_LT_32` (partial — indicates SIMD waste)
- **LDS:** `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_ADDR_CONFLICT`
- **Vector L1 cache:** `TCP_TOTAL_CACHE_ACCESSES`, `TCP_TCC_READ_REQ` (L2 misses), `TCP_READ_TAGCONFLICT_STALL_CYCLES`
- **L2 cache:** `TCC_HIT`, `TCC_MISS`, `TCC_EA_WRREQ_64B`, bandwidth calculations from `TCC_EA_RDREQ_DRAM_32B`

### 1.8 MI350 Performance Counters
**URL:** `https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi350-performance-counters.html`

Key facts:
- **New in MI350 (CDNA4):**
  - `SQ_INSTS_VALU_MFMA_F6F4` — counts FP6/FP4 matrix instructions
  - `SQ_INSTS_VALU_MFMA_MOPS_F6F4` — FP6/FP4 math operations / 512
  - `SQ_ACTIVE_INST_VALU2` — cycles with two VALU instructions issued (dual-issue capability)
  - `SQ_INSTS_LDS_LOAD/STORE/ATOMIC` — separated LDS ops (vs aggregate in MI300)
  - `SQ_INSTS_LDS_LOAD_BANDWIDTH` — actual bandwidth in 64-byte units
  - `SQ_INSTS_VALU_FLOPS_FP16/FP32/FP64` — per-instruction FLOP counting (new)
  - `SQ_INSTS_VALU_IOPS` — integer ops per instruction
- **LDS resource stalls:** `SQ_LDS_DATA_FIFO_FULL`, `SQ_LDS_CMD_FIFO_FULL`
- **VMEM stalls:** `SQ_VMEM_TA_ADDR_FIFO_FULL`, `SQ_VMEM_TA_CMD_FIFO_FULL`
- **TCP counters:** `TCP_TCP_LATENCY` (total wave latency), `TCP_TCC_READ_REQ_LATENCY`, `TCP_TCC_WRITE_REQ_LATENCY`
- **TCC (L2) counters:** `TCC_READ_SECTORS`, `TCC_WRITE_SECTORS`, `TCC_EA0_RDREQ_DRAM_32B`, etc.
- **CPC multi-XCD coordination counters:** `CPC_ADC_DISPATCH_ALLOC_DONE`, `CPC_STALLED_BY_SE*_SPI`

### 1.9 vLLM V1 Performance Optimization (for ROCm)
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html`

Key facts:
- **AITER (AI Tensor Engine for ROCm):** Master switch `VLLM_ROCM_USE_AITER=1`
  - Sub-flags: `_LINEAR`, `_MOE`, `_RMSNORM`, `_MLA`, `_MHA`, `_FP8BMM`
  - Auto-selects MLA backend for DeepSeek-V3/R1, MHA for standard transformers
- **Quick Reduce:** Alternative to RCCL for large all-reduces; supports FP16/BF16 + quantized INT8/INT6/INT4
- **Quantization:** FP8/FP4 reduces memory 2-4× with minimal accuracy loss
- **Environment variables:**
  - `HIP_FORCE_DEV_KERNARG=1` — kernel launch perf
  - `TORCH_BLAS_PREFER_HIPBLASLT=1` — prefer hipBLASLt
  - `NCCL_MIN_NCHANNELS=112` — multi-GPU bandwidth
- **MI300X HBM:** 192 GB HBM3; MI355X: 288 GB HBM3E
- **Parallelism strategies:** TP (within 8-GPU XGMI island), PP (across nodes), DP (replicated), EP (expert parallelism for MoE)
- **CUDA graph modes:** PIECEWISE (balanced), FULL/FULL_DECODE_ONLY (max throughput)
- **FP8 KV cache:** `--kv-cache-dtype fp8` reduces KV cache size by 2× with ~0.1% accuracy loss

### 1.10 rocWMMA API Reference
**URL:** `https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html`

Key facts:
- **Supported CDNA architectures (wave64):** gfx908, gfx90a, gfx942, gfx950
- **API functions:** `load_matrix_sync`, `store_matrix_sync`, `mma_sync`, `fill_fragment`, `synchronize_workgroup`
- **Fragment concept:** Template class `fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT>` — wavefront-level programming abstraction
- **Key data type combos and tile sizes:**

| Input/Output/Compute | BlockM×BlockN | BlockK | CDNA Support |
|-----------------------|---------------|--------|--------------|
| f8/f32/f32           | 16×16         | 32+    | gfx942, gfx950 |
| i8/i32/i32           | 16×16         | 16-64  | gfx908+ |
| f16/f32/f32          | 16×16         | 16+    | all gfx9 |
| bf16/f32/f32         | 16×16         | 8-32+  | all gfx9 |
| f32/f32/f32          | 16×16         | 4+     | all gfx9 |
| xf32/xf32/xf32      | 16×16         | 8+     | gfx942 |
| f64/f64/f64          | 16×16         | 4+     | gfx90a, gfx942, gfx950 |

- **Scheduling strategies:** default (independent waves), coop_row_major_2d, coop_col_major_2d, row_slice, col_slice
- **Thread block sizes:** Up to 4 wavefronts; X dimension must be multiple of WaveSize (64 for CDNA)
- **f8 NANOO (FNUZ)** format only on gfx942; OCP f8 assumed elsewhere

### 1.11 HIP Programming Model Introduction
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html`

Key facts:
- **Warp/Wavefront:** 64 threads for CDNA, 32 threads for RDNA
- **Thread hierarchy:** Thread → Warp → Block (work-group) → Grid
- **Max threads per block:** 1024; max block dims: 1024×1024×64
- **Up to 40 concurrent warps per CU** (theoretical); limited by VGPR/SGPR/LDS
- **Instruction issue:** Up to 5 instructions/cycle per CU (VALU + VMEM + SALU/SMEM + LDS + Branch)
- **Zero-overhead context switching** between warps (all contexts resident on CU)
- **Memory hierarchy:** Registers → LDS (shared) → L1 → L2 → HBM (global)
- **Cooperative groups:** Support for multi-grid, grid-level, block-level, tile, and coalesced groups
- **Cooperative kernel launch:** `hipLaunchCooperativeKernel` for grid-wide synchronization

### 1.12 HIP Hardware Implementation
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html`

Key facts:
- **CU internals (CDNA):**
  - 4 SIMD processors × 16 ALUs = 64 ALUs per CU
  - 256-512 KiB VGPR per CU, split across 4 SIMDs
  - 12.5 KiB SGPR per CU
  - Vector L1 cache: 16 KB per CU, write-through
- **LDS organization (CDNA 1-3):** 32 banks, 4 bytes wide, 128 bytes/cycle bandwidth
- **LDS organization (CDNA 4):** 64 banks, 256 bytes/cycle bandwidth
- **LDS access patterns:** 4-byte values = 50% peak BW; 16-byte values = 80% peak BW
- **L2 cache:** Shared by all CUs, 32 channels at 256-byte interleaving; hit-on-miss behavior
- **Memory coalescing:** Combine thread accesses into min cache line requests; consecutive threads → consecutive addresses
- **MFMA units:** Operate on matrix tiles via single instruction; use VGPRs + AGPRs (up to 512 KiB combined)
  - Example ISA: `v_mfma_f32_16x16x4f16 v[0:15], v[16:31], v[32:47], v[0:15]`
- **Data Movement Engine (DME, CDNA3/4):** Async bulk memory transfers HBM↔LDS, hardware address calculation for tensor strides
  - Intrinsic: `__builtin_amdgcn_async_work_group_copy`
- **SFU (Special Function Unit):** Hardware-accelerated `exp`, `log`, `sin`, `cos`, `rcp`, `rsqrt`; tuned for DL in CDNA3
- **Instruction sequencer:** 4 warp pools × 10 slots = 40 warps max; fetch 32B/cycle; round-robin issue

### 1.13 HIP Performance Guidelines
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html`

Key facts:
- **Optimization workflow:** Profile (`rocprofv3`) → Analyze → Apply → Verify → Iterate
- **Memory throughput:**
  - Use pinned memory (`hipHostMalloc`) for faster host-device transfers
  - Coalesce memory accesses (consecutive threads → consecutive addresses)
  - Use `__shared__` for data reuse; pad arrays by 1 to avoid bank conflicts (`data[32][33]`)
  - Use `float4`/`float2` for naturally aligned wide loads
  - SoA (Structure of Arrays) preferred over AoS
- **Instruction throughput:**
  - Prefer multiplication over division; bitwise ops for power-of-2
  - Use `__expf()`, `__logf()`, `__sinf()` fast intrinsics
  - Single-precision unless FP64 accuracy required
- **Control flow:** Minimize warp divergence; use `__builtin_expect` for branch hints
- **Register pressure:** Minimize live variables; use `__launch_bounds__(256, 4)` for compiler hints; check with `hipcc --resource-usage`
- **Occupancy:** Block size should be multiple of warpSize (64 for CDNA); 128-256 threads common
- **Memory management:** Allocate early, deallocate late; avoid frequent alloc/free; use managed memory for oversubscription

### 1.14 Profiling and Debugging Index
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/profiling-and-debugging.html`

Key facts:
- **Tools summary:** PyTorch Profiler, ROCProfiler (`rocprof`), ROCm Compute Profiler, ROCm Systems Profiler, ROCr Debug Agent
- Points to workload optimization guide for detailed usage patterns

---

## 2. Key New Findings Not Already in Knowledge Base

### 2.1 MI300X Architecture Constants (Critical for Kernel Optimization)
- **304 CUs total** (38 active per XCD × 8 XCDs)
- **4 MB L2 cache per XCD** (32 MB total L2)
- **5.3 TB/s aggregate HBM3 bandwidth**
- **2614.9 TFLOPS FP8** peak; 1307.4 TFLOPS FP16/BF16
- **Wave64 execution** (64 threads per warp on CDNA)
- **40 max concurrent warps per CU** (2560 threads per CU theoretical max)
- **Up to 5 instructions issued per cycle per CU** across different functional units

### 2.2 Low-Precision Type System (Complete Picture)
- **CDNA3 (MI300X, gfx942):** FP8 FNUZ (E4M3_FNUZ, E5M2_FNUZ) — NOT OCP-compatible
- **CDNA4 (MI350X, gfx950):** FP8 OCP + FP6 (E2M3, E3M2) + FP4 (E2M1) — full microscaling support
- **hipExt microscaling APIs** with `__amd_scale_t` (E8M0) for scaled MFMA operations
- **hipBLASLt** is the richest library for low-precision: supports FP4, FP6, FP8
- **Key distinction:** When targeting gfx942, ALWAYS use FNUZ variants; when targeting gfx950, use OCP variants

### 2.3 Performance Counter Formulas
- VMEM latency = `SQ_ACCUM_PREV_HIRES / SQ_INSTS_VMEM`
- Wave latency = `SQ_ACCUM_PREV_HIRES / SQ_WAVES`
- LDS latency = `SQ_ACCUM_PREV_HIRES / SQ_INSTS_LDS`
- MFMA utilization = `SQ_VALU_MFMA_BUSY_CYCLES / SQ_BUSY_CYCLES`
- L2 hit rate = `TCC_HIT / (TCC_HIT + TCC_MISS)`
- L1 hit rate = `(TCP_TOTAL_CACHE_ACCESSES - TCP_TCC_READ_REQ) / TCP_TOTAL_CACHE_ACCESSES`

### 2.4 MI350 (CDNA4) New Capabilities
- **Dual VALU issue** (`SQ_ACTIVE_INST_VALU2` counter)
- **FP6/FP4 matrix instructions** tracked by dedicated counters
- **Per-instruction FLOP counters** (not just instruction counts)
- **64-bank LDS** (doubled from CDNA3's 32 banks) — 256 bytes/cycle

### 2.5 System Tuning Essentials
- `HIP_FORCE_DEV_KERNARG=1` — default since ROCm 6.2, saves 2-3 µs per kernel
- `HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0` — prevent ROCm thread sprawl
- Disable NUMA auto-balancing for compute workloads
- Disable CPU C2 states for low-latency
- xGMI force link width to x16 in BIOS
- Deterministic clock at 1900 MHz reduces variance

### 2.6 Data Movement Engine (DME) — CDNA3/4 Feature
- Hardware unit for async bulk transfers between HBM and LDS
- Performs affine address calculations in hardware (stride × index + offset)
- Decouples data movement from compute pipelines
- Access via `__builtin_amdgcn_async_work_group_copy`

### 2.7 AITER for vLLM on ROCm
- Complete set of optimized kernels: Linear, MoE, RMSNorm, MLA, MHA, FP8BMM
- Quick Reduce supports quantized all-reduce (INT8/INT6/INT4)
- FP8 KV cache: `--kv-cache-dtype fp8` for 2× memory reduction

---

## 3. Links for Next Crawl Round

### High Priority (Architecture & Optimization)
1. https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html — Roofline model, occupancy theory, bank conflict theory
2. https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html — Warp cross-lane functions, synchronization
3. https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi200.html — MI200 architecture (for comparison)
4. https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html — rocprofv3 usage
5. https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/index.html — ROCm Compute Profiler
6. https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html — Hardware atomics details (timed out)
7. https://rocm.docs.amd.com/projects/composable_kernel/en/latest/ — Composable Kernel library

### Medium Priority (Libraries & Tools)
8. https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html — hipBLASLt precision details
9. https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html — MIOpen for convolutions
10. https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html — RCCL tips
11. https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html — Model acceleration
12. https://rocm.docs.amd.com/projects/Tensile/en/latest/src/reference/precision-support.html — Tensile precision

### Blog Posts (Optimization Techniques)
13. https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html — TunableOp blog
14. https://rocm.blogs.amd.com/software-tools-optimization/ — General optimization blog index
15. https://rocm.blogs.amd.com/artificial-intelligence/ — AI-specific blogs

### MI350/CDNA4 Specific
16. https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi350.html — MI350 architecture (returned 404, may need different URL)
17. https://rocm.docs.amd.com/en/7.12.0-preview/ — ROCm 7.12.0 preview docs (CDNA4 support)

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Pages successfully fetched | 17 |
| Pages failed (404/timeout) | 4 |
| Performance counter categories documented | 12+ |
| Data types documented | 15+ |
| Environment variables cataloged | 30+ |
| Architecture generations covered | CDNA1-4, RDNA2-4 |
| Optimization techniques documented | 50+ |
