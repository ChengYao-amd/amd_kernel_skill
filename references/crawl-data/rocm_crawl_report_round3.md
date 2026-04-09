# ROCm Documentation Crawl - Round 3: Deep Technical Content

## Crawl Metadata
- **Date**: 2026-04-09
- **Focus**: Deep technical content for AMD GPU kernel optimization
- **Sources fetched**: 14 pages successfully (8 blog 404s redirected to actual URLs)

---

## 1. Triton Compiler Architecture on AMD GPUs

**Source**: [Triton Kernel Optimizations Blog](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html) (April 2025)

### Compilation Pipeline

```
Python @triton.jit → AST → Triton-IR → Triton-GPU IR (TTGIR) → LLVM-IR → AMDGCN → .hsaco
                     Frontend    Optimizer (layout + passes)      LLVM Backend   AMD JIT
```

### AMD-Specific Triton-GPU IR Optimization Passes

| Pass | Description |
|------|-------------|
| **AMD GPU Accelerate Matmul** | Optimize dot instruction I/O layout for AMD matrix cores |
| **AMD GPU Optimize Epilogue** | Store accumulators directly without going through SMEM |
| **AMD GPU Stream Pipeline** | Pipeline global loads through registers to shared memory while computing previous tile |
| **AMD GPU Insert Instruction Sched Hints** | Insert scheduling hints after dot ops in main loop |
| **AMD GPU Reorder Instructions** | Decrease register pressure; promote LLVM-friendly instruction order |
| **AMD GPU Block Ping-pong** | Interleave instructions from two warps on same SIMD unit for better occupancy |
| **AMD GPU Canonicalize Pointers** | Rewrite pointers as (scalar_base, vector_offset) pairs for efficient addressing |
| **AMD GPU Convert To Buffer Ops** | Convert tt.load/tt.store to amdgpu buffer operations |

### AMD-Specific LLVM-IR Passes

| Pass | Description |
|------|-------------|
| **Decompose Unsupported Conversions** | Handle conversions not natively supported by AMD GPU |
| **Optimize LDS Usage** | Find peak LDS consumers, transform operations to fit in LDS |
| **Lower Instruction Sched Hints** | Convert scheduling hints to LLVM intrinsics |
| **Convert Builtin Func to LLVM** | AMD-specific builtin function lowering |

### Triton Layouts on AMD

- **blocked**: Each warp owns contiguous portion of tensor
- **slice**: Restructures tensor along a dimension
- **dot_op**: Optimized for block matrix product
- **shared**: GPU shared memory (LDS)
- **amd_mfma**: AMD MFMA matrix core layout
- **amd_wmma**: AMD WMMA matrix core layout
- **linear**: Unified layout within and across backends

### Key LLVM-IR Attributes for AMD
- `amdgpu-flat-work-group-size`
- `amdgpu-waves-per-eu`
- `denormal-fp-math-f32`

---

## 2. GEMM Optimization Toolkit for AMD GPUs

**Source**: [GEMM Kernel Optimization Blog](https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html) (Feb 2025)

### Three-Level GEMM Tuning Strategy

#### Level 1: Pre-Tuned Docker (Fastest start)
```bash
docker pull rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6
```

#### Level 2: PyTorch TunableOp (Framework-level)
```bash
# Step 1: Tuning pass (slow)
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 your_script.sh
# Step 2: Use results (fast)
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 your_script.sh
```
**Result**: >20% GEMM improvement typical

#### Level 3: Library-Level Tuning

**rocBLAS (Tensile-based)**:
```bash
ROCBLAS_LAYER=4 ROCBLAS_LOG_PATH=./gemms.yaml ./app
/opt/rocm/bin/rocblas-gemm-tune --yaml gemms.yaml
export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=result.csv
```

**hipBLASLt (hipblaslt-bench)**:
```bash
HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=log.log ./app
/opt/rocm/bin/hipblaslt-bench --api_method c -m M -n N -k K ...
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
```

---

## 3. hipBLASLt TensileLite Advanced GEMM Tuning

**Source**: [TensileLite Blog](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html) (April 2026)

### Three Tuning Approaches Compared

| Approach | Mechanism | Speed | Performance Ceiling |
|----------|-----------|-------|-------------------|
| **Offline Tuning** | Select best from existing pool | Fast | Limited by pool |
| **Online Tuning** | Runtime benchmarking + caching | Medium (warm-up) | Limited by pool |
| **TensileLite Tuning** | Generate new kernels | Slow | Near-optimal |

### TensileLite Tuning Parameters

**Thread & Workgroup Organization**:
- `WorkGroup [dim0, dim1, LocalSplitU]` — workgroup size for occupancy
- `ThreadTile [dim0, dim1]` — per-thread tile size
- `MacroTile = WorkGroup × ThreadTile`
- `WorkGroupMapping` — work-group ordering for L2 cache locality

**Loop & Unrolling**:
- `LoopUnroll` — inner loop unroll factor (higher = more registers, better coalescing)
- `DepthU = LoopUnroll × SplitU`

**Split-K Parallelism**:
- `LocalSplitU` — split summation within workgroup
- `GlobalSplitU` — split across workgroups (needs atomic reduction kernel)

**Memory Access**:
- `PrefetchGlobalRead` — prefetch global data one iteration ahead
- `PrefetchLocalRead` — prefetch LDS data one iteration ahead
- `VectorWidth` — vector load size (e.g., float4)

**Instruction**:
- `MatrixInstruction` — MFMA instruction + wave tiling parameters

### Benchmark Results (MI300X)
- TensileLite vs Baseline: **avg 2.25x speedup**
- TensileLite vs Offline Tuning: **avg 1.19x additional speedup**
- Skinny matrices (M=3): up to **3.2x speedup** over baseline

---

## 4. AMD Matrix Cores (MFMA Instructions)

**Source**: [Matrix Cores Blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html) (Nov 2022, updated Dec 2025)

### Matrix Core Performance (Flops/Clock/CU)

| Data Format | MI100 | MI250X |
|-------------|-------|--------|
| FP64 | N/A | 256 |
| FP32 | 256 | 256 |
| FP16 | 1024 | 1024 |
| BF16 | 512 | 1024 |
| INT8 | 1024 | 1024 |

### MFMA Compiler Intrinsic Syntax

```c
d = __builtin_amdgcn_mfma_<CDFmt>_<M>x<N>x<K><ABFmt>(a, b, c, cbsz, abid, blgp);
```

Parameters:
- `cbsz`: Control Broadcast Size (broadcast input block to 2^cbsz neighbors)
- `abid`: A-matrix Broadcast Identifier (which block to broadcast)
- `blgp`: B-matrix Lane Group Pattern (swizzling operations, 0-7)

### Supported MFMA Instructions (CDNA2)

| A/B Format | C/D Format | M×N×K | Blocks | Flops/cycle/CU |
|------------|------------|-------|--------|----------------|
| FP32 | FP32 | 32×32×2, 16×16×4, 4×4×1 | 1-16 | 256 |
| FP16 | FP32 | 32×32×8, 16×16×16, 4×4×4 | 1-16 | 1024 |
| BF16 | FP32 | 32×32×8, 16×16×16, 4×4×4 | 1-16 | 512-1024 |
| INT8 | INT32 | 32×32×8, 16×16×16, 4×4×4 | 1-16 | 1024 |
| FP64 | FP64 | 16×16×4, 4×4×4 | 1-4 | 128-256 |

### Using Matrix Cores — Approaches

1. **rocBLAS/rocWMMA** — highest level, library-managed
2. **Compiler intrinsics** — `__builtin_amdgcn_mfma_*` (recommended for custom kernels)
3. **Inline assembly** — not recommended (compiler can't track data hazards)
4. **AMD Matrix Instruction Calculator** — [tool](https://github.com/ROCm/amd_matrix_instruction_calculator) for register layout analysis

---

## 5. Composable Kernel (CK) TensorDescriptor System

**Source**: [CK Programming Guide Blog](https://rocm.blogs.amd.com/artificial-intelligence/amd_gpu_programming_guide/README.html) (March 2026)

### Core Abstraction

TensorDescriptor maps logical multi-dimensional coordinates to physical memory:
```
Physical offset = Σ(coordinate_i × stride_i)
```

### Transform Types

| Transform | Description |
|-----------|-------------|
| **Embed** | Maps multi-dimensional coordinates to linear memory (implicit in naive descriptors) |
| **Unmerge** | Splits one dimension into multiple (e.g., M → M1, M2) |
| **Merge** | Combines multiple dimensions into one |
| **PassThrough** | Identity transform (preserves dimension unchanged) |

### Key Code Patterns

```cpp
// Create 2D tensor descriptor (row-major)
auto desc = make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, 1));

// Transform: split first dimension
auto transformed = transform_tensor_descriptor(
    desc,
    make_tuple(unmerge, passthrough),
    make_tuple(Sequence<0>{}, Sequence<1>{}),     // lower dim ids
    make_tuple(Sequence<0, 1>{}, Sequence<2>{})   // upper dim ids
);

// Calculate physical offset
auto offset = transformed.CalculateOffset(make_multi_index(1, 3, 2));
```

### Matrix Transpose Performance
- CK implementation: 5.82 μs
- PyTorch: 8.4 μs
- **44.3% throughput improvement** with register-level transpose

---

## 6. CK-Tile GEMM Kernel Architecture

**Source**: [CK-Tile GEMM Blog](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html) (April 2025)

### CUDA → ROCm Terminology

| CUDA | ROCm |
|------|------|
| Thread | Work-item |
| Warp (32 threads) | Wavefront (32/64 work-items) |
| Block | Work Group |
| Shared Memory | LDS (Local Data Share) |
| SM | CU (Compute Unit) |
| Tensor Core | Matrix Core |
| SIMT | SIMD |
| Register | VGPR, SGPR |

### GEMM Pipeline Hierarchy

```
TilePartitioner → GEMMKernel → GEMMPipeline → BlockGEMM → WarpGEMM → MFMA instruction
                                    ↓
                            GEMMPipelinePolicy (DRAM + LDS access patterns)
```

### Pipeline Execution (Double-buffered)

1. **copy_lds**: Receives data from DRAM for next iteration
2. **gemm_lds**: Used for current iteration's BlockGemm
3. Overlap: DRAM→LDS copy and computation happen simultaneously

### WarpGEMM MFMA Example (F16 32×32×8)
```cpp
struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8 {
    kM = 32; kN = 32; kK = 8;
    kAMLane = 32; kBNLane = 32;   // lane layout for A,B
    kABKLane = 2; kABKPerLane = 4; // K-dim: 2 lanes × 4 elements
    kCMLane = 2; kCNLane = 32;     // output lane layout
    kCM0PerLane = 4;               // 4x repeat along M
    kCM1PerLane = 4;               // 4 consecutive elements per lane
    // D = A × B + C
    operator()(c, a, b) {
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    }
};
```

---

## 7. hipBLASLt Stream-K Scheduling

**Source**: [Stream-K Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-streamk.html)

### Enabling Stream-K
```bash
export TENSILE_SOLUTION_SELECTION_METHOD=2  # Enable Origami with Stream-K
```

**Note**: On MI350 series, Stream-K is the ONLY strategy (env var has no effect).

### Stream-K Configuration

| Env Variable | Description |
|-------------|-------------|
| `TENSILE_STREAMK_DYNAMIC_GRID=6` | Auto-pick optimal workgroup count (default) |
| `TENSILE_STREAMK_FIXED_GRID=N` | Fix to N workgroups (useful for concurrency) |
| `TENSILE_STREAMK_MAX_CUS=N` | Limit compute units for Stream-K |

### When to Use Stream-K
- Wide range of GEMM sizes
- Non-uniform dimensions (one dim much larger)
- Need consistent peak performance

---

## 8. hipBLASLt Data Type Support

**Source**: [Data Type Reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html)

### FP8 Types by Platform

| Type | gfx942 (MI300) | gfx950 (MI350) | gfx12 |
|------|----------------|-----------------|-------|
| `HIP_R_8F_E4M3_FNUZ` (f8_fnuz) | ✅ | ❌ | ❌ |
| `HIP_R_8F_E5M2_FNUZ` (bf8_fnuz) | ✅ | ❌ | ❌ |
| `HIP_R_8F_E4M3` (f8 OCP) | ❌ | ✅ | ✅ |
| `HIP_R_8F_E5M2` (bf8 OCP) | ❌ | ✅ | ✅ |

### New Sub-byte Types (hipBLASLt 1.2.2)
- `HIP_R_4F_E2M1` — 4-bit float
- `HIP_R_6F_E2M3` — 6-bit float
- `HIP_R_6F_E3M2` — 6-bit bfloat

### Compute Modes
- `HIPBLAS_COMPUTE_32F_FAST_16F` — auto down-convert to FP16 for Tensor Cores
- `HIPBLAS_COMPUTE_32F_FAST_16BF` — auto down-convert to BF16
- `HIPBLAS_COMPUTE_32F_FAST_TF32` — TF32 on gfx942/gfx950

---

## 9. AITER: AI Tensor Engine for ROCm

**Source**: [AITER-SGLang Integration Blog](https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html) (May 2025)

### AITER Optimized Operators for DeepSeek-R1

| Component | Backend | Description |
|-----------|---------|-------------|
| **MoE Top-K Routing** | HIP kernel | Fused biased grouped top-k |
| **MoE Sorting** | CK | MoE alignment and sort |
| **MoE FP8 Blockscale** | Assembly | Fused FP8 blockscale group GEMM (best perf on AMD) |
| **FP8 GEMM** | CK (pre-shuffle) | Block-scale with 1×128 activation, 128×128 weight scales |
| **MLA Decode** | Assembly | Latent attention (head dim 576/512) with weight absorption |
| **MHA Prefill** | CK | Multi-head attention (head dim 192/128) |
| **MLA Prefill** | Assembly | Latent attention (limited to q_extend < 160) |
| **Custom AllReduce** | HIP | Optimized for MI300X IPC |

### Performance Impact on DeepSeek-R1 (batch=64, input=512, output=32, TP=8)
- Prefill latency: **↓ 52%** (3.13s → 1.51s)
- Decode latency: **↓ 47%** (53ms → 28ms median)
- Total throughput: **↑ 100%** (7332 → 14636 tok/s)

### Key API Pattern
```python
from aiter import biased_grouped_topk
from aiter.fused_moe_bf16_asm import asm_moe
from aiter import gemm_a8w8_blockscale_wpreshuffle_CK
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
```

### Weight Pre-shuffle for GEMM
```python
from aiter.ops.shuffle import shuffle_weight
layer.weight.data = shuffle_weight(layer.weight.contiguous(), (16, 16))
```

---

## 10. FlashInfer on ROCm

**Source**: [FlashInfer Release Blog](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html) (April 2026)

### Porting NVIDIA wmma → AMD MFMA

Key changes:
- 32-thread warps → 64-thread wavefronts
- Modified shared memory access (different bank conflict rules)
- Updated indexing for MFMA's 16×16 matrix tile geometry

### Feature Support

| Kernel Type | FP16/BF16 | FP8 |
|-------------|-----------|-----|
| Decode Attention | ✅ | ✅ |
| Prefill Attention | ✅ | WIP |

Supports: MHA, GQA, MQA, paged KV-cache, ragged tensors

### Architecture Support
- CDNA3 (gfx942): MI300X, MI325X
- CDNA4 (gfx950): MI355X

---

## 11. FlyDSL: Rapid GPU Kernel Development

**Source**: [Kimi-K2.5 Optimization Blog](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html) (March 2026)

### What is FlyDSL
- Python DSL backed by custom MLIR stack
- FLIR (Flexible Layout IR) — layout algebra inspired by CuTe
- Composable `(Shape, Stride)` abstractions for tiling, swizzling, vectorization
- Compiles through MLIR passes → optimized binaries for gfx942/gfx950

### Fused MoE Performance (vs Triton/CK)

**Large shape** (tokens=16384, model_dim=7168, E=384, topk=8):

| dtype | Triton | CK | FlyDSL |
|-------|--------|-----|--------|
| BF16 | 12.09ms | gpu_fault | **8.68ms** |
| W4A16 | 31.43ms | unsupported | **9.77ms** |

### Kimi-K2.5 End-to-End Results (MI300X, concurrency=40)
- TTFT: **↓ 47%** mean
- TPOT: **↓ 69%** mean
- Throughput: **↑ 162%** (135 → 355 tok/s)
- Accuracy: identical on GSM8K

---

## 12. MI300X Inference Optimization Best Practices

**Source**: [LLM Inference Blog](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html) (Jan 2025)

### MI300X vs H100 Specs

| Metric | MI300X | H100 SXM | MI300X Advantage |
|--------|--------|----------|-----------------|
| HBM Capacity | 192 GB | 80 GB | **2.4x** |
| HBM Bandwidth | 5.325 TB/s | 3.35 TB/s | **1.59x** |
| TDP | 750W | 700W | 1.07x |

### Key MI300X Inference Guidelines

1. **Memory-bound advantage**: MI300X wins in decode phase (low-medium batch) due to higher bandwidth
2. **Large model hosting**: Llama-3.1-405B fits in single node; avoid TP>1 for ≤30B models
3. **TP1 for small models**: Run 8 instances of TP1 for ≤70B FP8 models to maximize throughput
4. **KV cache**: Larger HBM avoids cache eviction that degrades H100 at high batch sizes
5. **Decode-heavy workloads**: MI300X excels with short input / long output (e.g., 128 ISL, 2048 OSL)

### Llama 3.1 405B FP8 Performance (TP8)
- MI300X: **1.31x geomean throughput advantage** over H100
- Particularly strong at long output (128 in / 4096 out): **1.62x**

### vLLM KV Cache Eviction Warning Signs
```
WARNING: Sequence group ... is preempted by PreemptionMode.RECOMPUTE
GPU KV cache usage: 99.4%
```
Solution: Increase `gpu_memory_utilization` (but not >0.95) or use more TP

---

## 13. MI300X Workload Profiling Toolkit

**Source**: [Workload Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)

### Profiling Tool Hierarchy

| Tool | Level | Use Case |
|------|-------|----------|
| **PyTorch Profiler** | High-level | CPU+GPU timeline, identify hotspot ops |
| **ROCProfiler** (`rocprof`) | Low-level | Hardware counter collection (CSV) |
| **ROCm Compute Profiler** | Analysis | Auto-collect all counters, roofline analysis, GUI |
| **ROCm Systems Profiler** | System-wide | CPU/GPU/memory/page faults, process-level metrics |

### PyTorch Profiler Quick Start
```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)
prof.export_chrome_trace("trace.json")
# Visualize at https://ui.perfetto.dev/
```

### Auto-tuning Stack

| Level | Tool | Env Vars |
|-------|------|----------|
| PyTorch ops | TunableOp | `PYTORCH_TUNABLEOP_ENABLED=1` |
| Convolutions | MIOpen | `MIOPEN_FIND_MODE`, `MIOPEN_ENABLE_LOGGING` |
| Triton | max-autotune | `TORCHINDUCTOR_MAX_AUTOTUNE=1` |
| CK backend | CK in inductor | Append `CK` to gemm backends |

### TorchInductor Optimization Flags
```bash
TORCHINDUCTOR_MAX_AUTOTUNE=1         # Enable auto-tuning for GEMM/conv
TORCHINDUCTOR_FREEZING=1             # Inline weights as constants
TORCHINDUCTOR_CPP_WRAPPER=1          # C++ wrapper for lower overhead
TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1  # Force channels_last for conv
PYTORCH_MIOPEN_SUGGEST_NHWC=1        # MIOpen channels_last hint
TORCH_COMPILE_DEBUG=1                # Dump generated Triton kernels
```

---

## 14. HIP Porting Guide — Key Technical Notes

**Source**: [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)

### CUDA→HIP API Mapping

- `cuEventCreate` → `hipEventCreate`
- `cudaMalloc` → `hipMalloc`
- `__CUDA_ARCH__` → `__HIP_DEVICE_COMPILE__` + feature macros

### HIPIFY Tools
- **hipify-clang**: Clang-based AST translation (more powerful, needs CUDA headers)
- **hipify-perl**: Regex-based (easier setup, works on incorrect code)

### Key Differences from CUDA

1. **Unified address space**: CPU and all devices share single address pool
2. **No explicit context management**: `hipCtx` deprecated; use `hipSetDevice`
3. **Code object format**: `.hsaco` (not PTX/CUBIN)
4. **Fat binary**: `.hip_fatbin` section in ELF
5. **Feature macros**: Use `__HIP_ARCH_HAS_*__` instead of `__CUDA_ARCH__ >= N`

### Library Mapping
- Use `roc`-prefixed libraries for AMD-optimized path (rocBLAS > hipBLAS)
- `--roc` flag in hipify tools to auto-select AMD-optimized libraries

---

## 15. HIP Math API — Precision Reference

**Source**: [HIP Math API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html)

### Key Precision Facts

- All arithmetic functions (abs, fma, fmod): **0 ULP** error
- Trig functions (sin, cos, tan): **1 ULP** (single), **1 ULP** (double)
- Exponential/log: **1-2 ULP**
- Special functions (erf, tgamma): **2-6 ULP**

### Intrinsic (Fast) Math Functions
- `__cosf`: 4 ULP (vs 1 ULP for `cosf`)
- `__expf`: 6 ULP
- `__exp10f`: 18 ULP
- Only **nearest-even rounding** by default; `_rz/_ru/_rd` need `OCML_BASIC_ROUNDED_OPERATIONS` macro

### Unsupported Functions
- `cyl_bessel_i0f/i1f` — not available in HIP

---

## Cross-Cutting Insights for Kernel Development

### Optimization Priority Stack (from most to least impactful)

1. **GEMM tuning** — Pre-tuned docker → TunableOp → hipblaslt-bench → TensileLite
2. **Attention kernel selection** — AITER assembly > CK FMHA > Triton FA
3. **MoE kernel optimization** — AITER asm_moe / FlyDSL fused MoE
4. **Memory layout** — Prefer channels_last; use CK TensorDescriptor transforms
5. **Quantization** — FP8 blockscale (1×128 act, 128×128 weight); FP4 emerging
6. **Communication** — Custom AllReduce for MI300X; disable for small TP

### Tool Selection Guide

| Task | Best Tool |
|------|-----------|
| Quick GEMM optimization | PyTorch TunableOp |
| Production GEMM tuning | hipblaslt-bench + TensileLite |
| Custom attention kernel | CK-Tile or FlyDSL |
| Custom MoE kernel | FlyDSL (fastest dev) or AITER assembly (best perf) |
| Triton kernel on AMD | Use AMD-specific passes; dump TTGIR to verify |
| Profile bottlenecks | ROCm Compute Profiler (GUI) or rocprof (raw) |
| Port CUDA code | hipify-clang (preferred) or hipify-perl |
| Matrix instruction analysis | AMD Matrix Instruction Calculator |

### Critical Environment Variables Reference

```bash
# GEMM Tuning
PYTORCH_TUNABLEOP_ENABLED=1
ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=tuned.csv
HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
TENSILE_SOLUTION_SELECTION_METHOD=2  # Stream-K

# Triton Debugging
TRITON_INTERPRET=1                   # Interpreter mode
TORCH_COMPILE_DEBUG=1                # Dump generated kernels

# vLLM/Inference
SGLANG_USE_AITER=1                   # Enable AITER operators
AITER_MOE=1                          # Enable AITER MoE path

# Profiling
ROCBLAS_LAYER=4                      # rocBLAS logging
HIPBLASLT_LOG_MASK=32                # hipBLASLt GEMM logging
```
