# ROCm Documentation Crawl Report — Round 2

**Date:** 2026-04-09
**Pages fetched:** 15 successfully (6 via corrected URLs), 2 failed (404)
**Focus:** HIP performance guidelines, programming model, env variables, CU masking, profiler modes, MI300X system tuning, FP8 GEMM on CDNA4, FlyDSL, hipBLASLt tuning, GPU partitioning, speculative decoding

---

## 1. HIP Performance Guidelines
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html`

### Optimization Workflow
1. Profile baseline with `rocprofv3 --stats -- <app>`
2. Analyze metrics (compute-bound vs memory-bound)
3. Apply targeted optimizations
4. Verify improvements via re-profiling
5. Iterate

### Profiling Tools
- **rocprofv3:** kernel execution time, memory bandwidth, warp occupancy, CU utilization, instruction-level counters; exports JSON/CSV; Perfetto trace integration
- **ROCprof Compute Viewer:** GUI kernel analysis, counter correlation, hierarchical breakdown
- **amd-smi:** real-time GPU utilization, power, thermals; `amd-smi static`, `amd-smi metric --usage`

### Memory Throughput Optimization
- **Minimize host-device transfers:** batch small transfers into single large `hipMemcpy`
- **Pinned memory:** `hipHostMalloc(&ptr, size)` for DMA-based transfers
- **Mapped memory on APUs:** `hipHostMalloc(&ptr, size, hipHostMallocMapped)` for zero-copy
- **Coalesced access:** consecutive threads → consecutive addresses; pad 2D arrays to warp-size multiples
- **Shared memory (LDS) tiling:** load once, reuse many times; classic for matrix multiply
- **Bank conflict avoidance:** pad shared arrays (`float data[32][33]` instead of `[32][32]`)
- **Texture memory:** hardware-accelerated 2D spatial access with filtering

### Instruction Throughput
- Prefer `* 0.5f` over `/ 2.0f`; bitwise shifts for power-of-2
- Single-precision (`sinf`, `expf`) much faster than double-precision
- Fast intrinsics: `__expf`, `__logf`, `__fsqrt_rn`, `__frcp_rn`

### Control Flow
- Minimize warp divergence; structure conditions on `threadIdx` ranges
- Branch hints: `__builtin_expect(cond, 0)`, C++20 `[[likely]]`

### Register Pressure & Occupancy
- Minimize live variables; chain computations instead of storing intermediates
- `__launch_bounds__(256, 4)` to hint threads/block and min blocks/CU
- `hipcc --resource-usage` to check register counts
- **Block sizes:** multiples of 64 (Instinct/CDNA) or 32 (RDNA); 128-256 common
- `rocprofv3 --occupancy ./app` to profile occupancy

### Memory Management
- Allocate early, deallocate late; avoid repeated alloc/free in loops
- Leave ~10% memory margin (`free * 0.9`)
- `hipMallocManaged` for oversubscription; `hipMemPrefetchAsync` for prefetch hints

---

## 2. HIP Programming Model
**URL:** `https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html`

### Key Architectural Concepts
- **Warp/Wavefront sizes:** 64 threads (CDNA), 32 threads (RDNA)
- **Thread hierarchy:** Thread → Warp → Block (Work-group) → Grid
- **Block limits:** max 1024 threads/block, max dimensions 1024×1024×64
- **SIMT execution:** all lanes in warp execute same instruction; divergent branches mask lanes
- **Context switching:** zero-cost on GPU (all warps have registers on CU)

### Memory Model
- **Register (VGPR/SGPR):** fastest, per-thread
- **LDS (shared memory):** per-CU, fast intra-block communication
- **L1/L2 cache:** hardware-managed
- **Global memory (HBM):** highest capacity, highest latency

### Cooperative Groups
- Grid-level synchronization via `cooperative_groups::this_grid().sync()`
- Multi-grid synchronization across devices
- Tiled partitions for sub-warp operations

### API Patterns
```cpp
// Kernel launch
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
// Alternative
hipLaunchKernelGGL(kernel, gridDim, blockDim, sharedMem, stream, args...);
```

---

## 3. Setting the Number of Compute Units
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/setting-cus.html`

### Environment Variables
- **`HSA_CU_MASK`:** low-level driver queue CU mask; also applies to profiled queues
- **`ROC_GLOBAL_CU_MASK`:** HIP/OpenCL runtime queue CU mask

### Syntax
```
HSA_CU_MASK = GPU_list : CU_list [; GPU_list : CU_list]*
CU_list = 0x[hex_mask] | ID_list
```
Examples:
- `HSA_CU_MASK=0,2-4,7:0-15,32-47` — enable CUs 0-15 and 32-47 on GPUs 0,2,3,4,7
- `ROC_GLOBAL_CU_MASK=0xf` — enable only 4 CUs

### Constraints
- GPU indices are post-`ROCR_VISIBLE_DEVICES` reordering
- Cannot disable single CU in WGP pair (for WGP-mode kernels)
- Mask with 0 usable CUs → syntax error

---

## 4. ROCm Environment Variables (Comprehensive)
**URL:** `https://rocm.docs.amd.com/en/latest/reference/env-variables.html`

### GPU Isolation
| Variable | Purpose |
|---|---|
| `ROCR_VISIBLE_DEVICES` | Device indices/UUIDs exposed (recommended on Linux) |
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` | HIP-level device filtering |
| `GPU_DEVICE_ORDINAL` | OpenCL + HIP device filtering |

### Performance-Critical Variables
| Variable | Default | Effect |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG` | 1 | Store kernel args in device memory; saves 2-3 µs |
| `GPU_MAX_HW_QUEUES` | 4 | Max hardware queues per device per process |
| `HIP_LAUNCH_BLOCKING` | 0 | Serialize kernel execution for debugging |
| `AMD_SERIALIZE_KERNEL` | 0 | Wait before/after enqueue (1/2/3) |

### Memory Variables
| Variable | Default | Effect |
|---|---|---|
| `HIP_INITIAL_DM_SIZE` | 8388608 (8MB) | Initial device malloc heap |
| `HIP_MEM_POOL_SUPPORT` | 0 | Enable memory pool |
| `GPU_SINGLE_ALLOC_PERCENT` | 100 | Max single allocation as % of GPU memory |
| `GPU_MAX_HEAP_SIZE` | 100 | Max GPU heap as % of board memory |
| `HIP_HOST_COHERENT` | 0 | Host-GPU memory coherence |

### Debug/Logging
| Variable | Default | Effect |
|---|---|---|
| `AMD_LOG_LEVEL` | 0 | 0=off, 1=error, 2=warn, 3=info, 4=debug |
| `AMD_LOG_MASK` | 0x7FFFFFFF | Bitmask for log categories |
| `HIPCC_VERBOSE` | - | 1=clang cmd, 2=env, 4=args, 7=all |

### ROCR-Runtime Variables
| Variable | Default | Effect |
|---|---|---|
| `HSA_ENABLE_SDMA` | 1 | Enable DMA engines for copies |
| `HSA_NO_SCRATCH_RECLAIM` | 0 | Permanent scratch assignment to queues |
| `HSA_SCRATCH_SINGLE_LIMIT` | ~140MB | Scratch threshold per XCC |
| `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` | 3GB | Async scratch threshold per XCC |
| `HSA_XNACK` | - | Enable XNACK (set to 1) |
| `HSA_DISABLE_CACHE` | 0 | Disable L2 cache (set MTYPE=UC) |

### Library-Specific Env Vars
Key libraries with their own env vars: hipBLASLt, rocBLAS, MIOpen, MIGraphX, RCCL, Tensile

---

## 5. ROCm Compute Profiler — Profile Mode
**URL:** `https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html`

### Basic Usage
```bash
rocprof-compute profile --name <workload> -- <command>
```

### Profiling Stages
1. Collect all hardware counters (multiple passes)
2. Collect roofline data (empirical bandwidth/FLOPS benchmarks)

### Filtering Options
- `-b / --block <N>`: profile specific hardware report blocks (e.g., `-b 10 7`)
- `-k / --kernel <name>`: kernel name substring filter
- `-d / --dispatch <N>`: dispatch ID filter (zero-based)
- `--set <name>`: single-pass metric sets (e.g., `compute_thruput_util`, `launch_stats`)

### Metric Set Examples
- `compute_thruput_util`: SALU/VALU/VMEM/Branch Utilization
- `launch_stats`: Grid Size, Workgroup Size, Total Wavefronts, VGPRs, AGPRs, SGPRs, LDS Allocation, Scratch Allocation

### Roofline Analysis
- `--roof-only`: roofline-only profiling
- `--mem-level <levels>`: specify cache levels (HBM, L2, L1, LDS)
- `--roofline-data-type <type>`: FP32, FP64, FP16, BF16, INT8, FP8 etc.
- Output: empirical bandwidth + peak FLOPS measurements, PDF plots

### MI200 Measured Roofline Numbers (example)
| Metric | Value |
|---|---|
| HBM BW | 1388.0 GB/s |
| L2 BW | 5020.8 GB/s |
| L1 BW | 9229.5 GB/s |
| LDS BW | 17645.6 GB/s |
| Peak FP32 VALU | 20986.9 GFLOPS |
| Peak FP64 VALU | 20408.0 GFLOPS |
| Peak MFMA BF16 | 170280.0 GFLOPS |
| Peak MFMA F16 | 164733.6 GFLOPS |
| Peak MFMA F32 | 41399.6 GFLOPS |
| Peak MFMA F64 | 41379.2 GFLOPS |
| Peak MFMA I8 | 166281.9 GOPS |

---

## 6. ROCm Compute Profiler — Performance Model
**URL:** `https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/performance-model.html`

### Architecture Support Matrix
| Feature | CDNA | CDNA 2 | CDNA 3 | CDNA 4 |
|---|---|---|---|---|
| Chip packaging | Single Die | Two GCDs | Multi-chiplet + partition modes | Multi-Die + 2 IODs |
| Supported | MI100 | MI200 series | MI300A/X/MI325X | MI350X/MI355X |
| Compute partition | No | No | CPX/SPX | CPX/SPX |

### Data Type Support (MFMA)
- **CDNA:** FP32, FP64, FP16, INT8 DOT, INT4 DOT, FP32 GEMM
- **CDNA2:** adds FP64/FP16/BF16/INT8 GEMM, Packed FP32
- **CDNA3:** adds TF32 GEMM, FP8/BF8
- **CDNA4:** adds FP8/BF8 (no TF32 GEMM), same GEMM types as CDNA3

### Hardware Blocks Profiled
- Compute Unit (CU), L2 Cache (TCC), Shader Engine (SE), Command Processor (CP), System Speed-of-Light

---

## 7. MI300X System Optimization (Tuning Guide)
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/system.html`

### Critical BIOS Settings (AMD EPYC 9004)
| Setting | Value | Purpose |
|---|---|---|
| Above 4G decoding | Enabled | GPU large BAR |
| SR-IOV | Enabled | Single root IO virtualization |
| SMT control | Disable | Better for compute-bound workloads |
| NUMA nodes/socket | Auto (NPS1) | Unified memory domain |
| xGMI link width | Force x16 | Max inter-chip bandwidth |
| cTDP / Package power limit | 400W | Max CPU power |
| APBDIS | 1 | Disable DF P-states |
| Fixed SOC P-state | P0 | Max SOC frequency |
| TSME | Disabled | Disable memory encryption |

### GRUB Settings
```
GRUB_CMDLINE_LINUX="pci=realloc=off iommu=pt"
```
- `pci=realloc=off`: unambiguous GPU detection
- `iommu=pt`: passthrough mode, no DMA translation overhead

### OS Tuning
- **Disable C2 CPU state:** `cpupower idle-set -d 2`
- **Disable NUMA auto-balancing:** `echo 0 > /proc/sys/kernel/numa_balancing`
- **Set env vars:**
  ```bash
  export HIP_FORCE_DEV_KERNARG=1
  export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
  ```
- **Deterministic clock:** `rocm-smi --setperfdeterminism 1900` (avoid PCC events)

### GPU Partition Modes (MI300X)
| Compute Mode | Description | Use Case |
|---|---|---|
| **SPX** | All 8 XCDs = 1 logical GPU | Large models, full utilization |
| **DPX** | 4 XCDs per partition = 2 logical GPUs | Dual model serving |
| **CPX** | 1 XCD per partition = 8 logical GPUs | Small models, multi-tenant |

| Memory Mode | NUMA Domains | Compatible With |
|---|---|---|
| **NPS1** | 1 (all HBM unified) | SPX, CPX |
| **NPS2** | 2 | DPX |
| **NPS4** | 4 | CPX |

Commands:
```bash
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4
```

### RCCL with CPX mode
```bash
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=$((2*1024*1024*1024))
export MSCCLPP_READ_ALLRED=1
```

---

## 8. FP8 GEMM Optimization on CDNA4 (MI355X)
**URL:** `https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html`

### CDNA4 vs CDNA3 Architecture Differences
| Feature | CDNA4 | CDNA3 |
|---|---|---|
| LDS capacity | **160 KB** per CU | 64 KB |
| LDS banks | **64** | 32 |
| LDS read bandwidth | **256 B/clk** | 128 B/clk |
| GLOBAL_LOAD_LDS per-lane | **128 bits** | 32 bits |
| FP4/FP6 MFMA | **Supported** | No |
| Block-scaled MFMA | **V_MFMA_SCALE_F32_{16x16x128,32x32x64}_F8F6F4** | No |
| FP16/BF16 MFMA | Adds 16x16x32, 32x32x16 | Up to 16x16x16, 32x32x8 |

### Performance Progression (M=N=K=4096, FP8 → BF16, FP32 accum)
| Stage | TFLOPS/s | Speedup |
|---|---|---|
| Naive (1 thread/element) | 1.15 | baseline |
| LDS tiling | 4.80 | 4.2× |
| MFMA matrix-core | 30.05 | 26× |
| + Vectorized loads (fp8x16) | 336.88 | 293× |
| + Direct global-to-LDS load | 506.70 | 441× |
| + LDS swizzle | 497.43 | 432× |
| + Double-buffer pipeline | 1166.41 | 1014× |
| + 256×256 tile, 8 waves | **2288.16** | 1990× |

**hipBLASLt reference:** ~2750 TFLOPS/s (4096), ~3130 TFLOPS/s (8192)

### Key Optimization Techniques
1. **MFMA 16×16×128:** 65536 FLOPs/instruction (vs 128 for FMA, 512× larger)
2. **Vectorized FP8 loads:** `fp8x16_t` = 16 FP8 values per load (128 bits)
3. **Direct global-to-LDS:** `llvm_amdgcn_raw_buffer_load_lds()` bypasses registers
4. **LDS swizzling:** XOR-based row remap eliminates bank conflicts with 64-bank LDS
5. **Double-buffering:** ping-pong LDS slots overlap load(t+1) with compute(t)
6. **8-wave ping-pong scheduling:** 2 waves/SIMD alternate memory/MMA ops; uses:
   - `__builtin_amdgcn_s_barrier()` for wave stalling
   - `__builtin_amdgcn_s_setprio(x)` for priority control (0-3)
   - `__builtin_amdgcn_sched_barrier(x)` for instruction scheduling control

### Optimal Configuration
- **256×256 output tile, 512 threads (8 waves):** highest measured performance
- 1024-thread variants show diminishing returns from sync overhead

---

## 9. FlyDSL — Python-first GPU Kernel DSL
**URL:** `https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html`

### What is FlyDSL
- Python-first, MLIR-native DSL for expert GPU kernel development on AMD
- Compilation: Python DSL → AST transforms → Fly dialect (MLIR) → ROCDL → HSACO
- Built on CuTe Layout Algebra for formal layout representation
- Install: `pip install flydsl`

### Key Capabilities
- Thread-level and IR-level control (complements Triton's block-level)
- Explicit lane control, register usage, custom layouts, ISA-level hints
- JIT compilation for fast iteration
- Layout-agnostic design for portability

### Supported Operators
- Softmax, LayerNorm/RMSNorm, Quantization, GEMM, MoE kernels
- Production adoption at hyperscale on MI GPU clusters

### Use Cases vs Triton
- **Triton:** block-level programming, mainstream developers
- **FlyDSL:** thread-level/IR-level control, expert developers targeting roofline perf

---

## 10. LLM Inference Optimization via GPU Partitioning
**URL:** `https://rocm.blogs.amd.com/software-tools-optimization/multi-inf-engine-gpu-partition/README.html`

### MI300X Partitioning for Inference
- **SPX + NPS1:** single large model, full GPU resources
- **DPX + NPS2:** 2 logical GPUs per physical GPU; 2× vLLM instances
- **CPX + NPS4:** 8 logical GPUs per physical GPU; 8× vLLM instances

### Performance Results (Mistral-Nemo FP8, single MI300X)
| Mode | Max Throughput | Throughput (tokens/sec) at max QPS |
|---|---|---|
| SPX (1 instance) | ~4001 tokens/sec | 25.98 QPS |
| DPX (2 instances) | Scales with partitions | Lower per-instance, higher aggregate |

### Key Configuration
```bash
# Set DPX partition
amd-smi set --gpu all --compute-partition DPX
amd-smi set --gpu all --memory-partition NPS2
sudo amd-smi reset -r  # reload driver

# vLLM in container
export HIP_VISIBLE_DEVICES="$GPU_ID"
vllm serve $MODEL --port 8000
```

---

## 11. hipBLASLt Online GEMM Tuning
**URL:** `https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt_online_tuning/README.html`

### Concept
- Runtime GEMM algorithm selection without offline tuning
- Evaluates candidate kernels on first encounter of new matrix shape
- Caches optimal solution index for reuse

### Enable
```bash
export HIP_ONLINE_TUNING=1
# For vLLM:
export VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1
```

### Performance
- Online tuning: **105.98%** of baseline, **100.79%** of offline tuning (on average)
- One-time overhead: ~31 seconds for benchmarking unseen GEMM shapes
- Integrated into AITER framework

### Workflow
1. Check cache file for matching GEMM config
2. If miss: benchmark candidates, select best
3. Save solution index to cache

---

## 12. hipBLASLt TensileLite GEMM Tuning (Advanced)
**URL:** `https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html`

### Three-Level Tuning Hierarchy
1. **Offline Tuning:** select best from existing kernel pool
2. **Online Tuning:** runtime selection from existing pool
3. **TensileLite Tuning:** **generate entirely new kernels** for specific (M,N,K)

### Key Tensile Parameters
| Category | Parameters |
|---|---|
| Thread/Workgroup | WorkGroup [dim0,dim1,LocalSplitU], ThreadTile, MacroTile, WorkGroupMapping |
| Loop/Unrolling | LoopUnroll, DepthU, LoopDoWhile |
| Split-K | LocalSplitU (intra-WG), GlobalSplitU (inter-WG, needs atomic reduction) |
| Memory | PrefetchGlobalRead, PrefetchLocalRead, VectorWidth |
| Instruction | MatrixInstruction (MFMA shape), wave tiling |

### Workflow
```bash
# Step 1: Capture GEMM shape
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE=./hipblaslt.log
hipblaslt-bench -m 1280 -n 32 -k 5120 --a_type f16_r ...

# Step 2: Generate config + tune
python tensile_config_generator.py --hipblaslt_log ./hipblaslt.log \
  --tensile_config ./tuning_template.yaml --gpus 4 --iters 100
./TensileLite/build_tmp/Tensile.sh tuning_template.yaml tuning_result

# Step 3: Merge + rebuild
python3 merge.py <existing_logic_dir> <tuning_result/3_LibraryLogic/> <output_dir>
./install.sh -idc -a gfx942
```

### Performance Results (MI300X, f16→f32)
- **Average speedup vs offline tuning:** 119%
- **Average speedup vs baseline:** 225%
- Best cases: **150-320%** speedup vs baseline for small-M shapes (LLM decode)
- Example: shape (165, 14400, 120) → 42.3µs vs 135.9µs baseline = **3.2× speedup**

---

## 13. Speculative Decoding on MI300X
**URL:** `https://rocm.blogs.amd.com/artificial-intelligence/spec_decode_mi300x/README.html`

### Key Results
| Framework | Mode | Speedup Range |
|---|---|---|
| gpt-fast | Eager | 1.26×–1.71× |
| gpt-fast | torch.compile | **1.75×–2.99×** |
| vLLM | Eager | 1.32×–2.0× |
| vLLM | Graph | **1.5×–2.9×** |

### MI300X Specs Summary
| Resource | Value |
|---|---|
| HBM | 192 GiB |
| CUs | 304 (38/XCD × 8 XCDs) |
| LDS | 64 KiB/CU |
| L2 Cache | 32 MiB (4 MiB/XCD) |
| L3 Cache | 256 MiB |
| VGPR File | 512 KiB/CU |
| SGPR File | 12.5 KiB/CU |

### Insights
- **Batch size matters:** SpD speedup decreases with batch size; slowdown after BS=8 (eager) or BS=32 (graph)
- `torch.compile` / HIP Graphs reduce kernel launch overhead → larger draft-vs-target latency gap → better SpD
- SpD is memory-bandwidth-bound optimization; loses efficacy as workload becomes compute-bound
- gpt-fast compile mode is 12-21% faster than vLLM graph mode due to kernel re-compilation

---

## 14. Profiling & Debugging Index for AI Inference
**URL:** `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/profiling-and-debugging.html`

### Tool Hierarchy
1. **PyTorch Profiler:** high-level operator timing
2. **ROCProfiler (rocprofv3):** kernel-level hardware counters
3. **ROCm Compute Profiler:** automated counter collection + analysis
4. **ROCm Systems Profiler:** system-wide tracing
5. **ROCr Debug Agent:** GPU fault debugging

---

## Summary of Key Findings Across All Pages

### Top Optimization Techniques
1. **MFMA matrix cores** are essential — 512× more FLOPs/inst than FMA
2. **LDS tiling + double buffering** hides global memory latency
3. **LDS swizzling** (XOR remap) eliminates bank conflicts in 64-bank CDNA4
4. **Direct global-to-LDS loads** (`llvm_amdgcn_raw_buffer_load_lds`) bypass register file
5. **8-wave ping-pong scheduling** maximizes SIMD utilization (2 waves/SIMD alternating memory/MMA)
6. **hipBLASLt 3-tier tuning**: online (easy) → offline (stable) → TensileLite (custom kernels, 2-3× gains)
7. **GPU partitioning** (CPX/DPX) for multi-tenant inference
8. **Speculative decoding** provides up to 3× speedup at low batch sizes

### Critical Environment Variables
| Variable | Value | Effect |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG` | 1 | 2-3µs kernel arg latency reduction |
| `HIP_ONLINE_TUNING` | 1 | Runtime GEMM algorithm selection |
| `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK` | hex mask | CU isolation for profiling/multi-tenancy |
| `ROCR_VISIBLE_DEVICES` | device list | GPU isolation |
| `GPU_MAX_HW_QUEUES` | 4 | Hardware queue limit per device |
| `HSA_ENABLE_SDMA` | 1 | DMA engine for copies |
| `HSA_OVERRIDE_CPU_AFFINITY_DEBUG` | 0 | Inherit parent CPU affinity |

### Performance Numbers Reference
| Metric | MI200 | MI300X | MI355X (CDNA4) |
|---|---|---|---|
| HBM BW | 1388 GB/s | ~5.3 TB/s (8 stacks) | — |
| Peak MFMA BF16 | 170 TFLOPS | — | — |
| Peak MFMA F32 | 41.4 TFLOPS | — | — |
| FP8 GEMM (4096³) | — | ~2750 TFLOPS (hipBLASLt) | ~2288 TFLOPS (hand-tuned) |
| FP8 GEMM (8192³) | — | — | ~3130 TFLOPS (hipBLASLt) |
| TensileLite speedup | — | 2.25× avg vs baseline | — |

### New Tools Discovered
- **FlyDSL:** Python-first MLIR DSL for expert kernel dev (`pip install flydsl`)
- **AITER:** AMD AI inference engine with online tuning integration
- **inference-benchmarker:** HuggingFace tool for LLM serving benchmarks
- **rocm-bandwidth-test:** Inter-device bandwidth measurement
