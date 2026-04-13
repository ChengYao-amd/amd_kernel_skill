# Optimization Case Studies on AMD GPUs

> Real-world optimization case studies from production deployments, research, and vendor benchmarks. Each case includes performance numbers, configuration, profiling insights, and lessons learned.

---

## Table of Contents

1. [DeepSeek-R1/V3 Inference on MI300X](#1-deepseek-r1v3-inference-on-mi300x)
2. [Kimi-K2/K2.5 MoE Inference on MI355X](#2-kimi-k2k25-moe-inference-on-mi355x)
3. [MI355X Training Performance](#3-mi355x-training-performance)
4. [MoE Training Best Practices (DeepSeek-Scale)](#4-moe-training-best-practices-deepseek-scale)
5. [FP8 GEMM Progressive Optimization on CDNA4](#5-fp8-gemm-progressive-optimization-on-cdna4)
6. [GEAK: AI Agent for HIP Kernel Optimization](#6-geak-ai-agent-for-hip-kernel-optimization)
7. [Cross-Cutting Lessons](#7-cross-cutting-lessons)

---

## 1. DeepSeek-R1/V3 Inference on MI300X

**Source**: ROCm Blog -- DeepSeek-R1 Part 1 & Part 2 (Mar 2025), DeepSeekV3 Kernel Analysis (May 2025)

### Performance: MI300X vs H200

| Metric | MI300X (8 GPU) | vs H200 |
|---|---|---|
| Throughput at same latency | -- | **2x-5x higher** |
| Throughput at same concurrency | -- | **Up to 75% higher** |
| Latency at same concurrency | -- | **Up to 60% lower** |
| Max concurrent requests under 50ms ITL | **128** | H200: 16 |

### AITER Kernel-Level Speedups

| Kernel | Speedup over Baseline |
|---|---|
| Block-scale GEMM | **2x** |
| Block-scale fused MoE | **3x** |
| MLA decode | **17x** |
| MHA prefill | **14x** |

### Profiling Insights (DeepSeekV3 on SGLang)

**Prefill phase**: Attention + MoE consume the majority of compute time; GEMM and miscellaneous operations are secondary.

**Decode phase**: MoE is dominant; attention takes a smaller share; the rest is spread across GEMM, all_reduce, and miscellaneous operations.

**Key bottleneck identified**: Inefficient device-to-host memory transfers (data copied GPU to CPU unnecessarily).

**Decode kernel latency comparison**: MLA is 2x less and MoE is 3x less vs NVIDIA H200.

### Configuration

```bash
python3 -m sglang.launch_server \
  --model /model --tp 8 --trust-remote-code \
  --chunked-prefill-size 131072 \
  --enable-torch-compile --torch-compile-max-bs 256

export HSA_NO_SCRATCH_RECLAIM=1
```

### Lessons Learned

1. **`chunked_prefill_size=131072`** accelerates prefill but costs more VRAM; tune per use case
2. **MI300X 192GB HBM3** enables higher concurrency than H200's 141GB -- memory capacity is a competitive advantage
3. **`--enable-torch-compile`** reduces CPU-side kernel launch overhead, especially during decode
4. **Profiling tools** (RPD + TorchProfiler) are essential for kernel-level bottleneck identification
5. **AITER integration** via `VLLM_ROCM_USE_AITER=1` provides the largest single optimization (especially MLA at 17x)

---

## 2. Kimi-K2/K2.5 MoE Inference on MI355X

**Source**: Kimi-K2 on MI355 (Oct 2025), Kimi-K2.5 with FlyDSL (Mar 2026)

### Kimi-K2 on MI355X vs B200

| Metric | MI355X Advantage |
|---|---|
| Mean TTFT (high concurrency) | **>3x better** than B200 |
| Mean E2E latency (high concurrency) | Significantly lower |
| Total token throughput | Better as concurrency increases |
| Memory capacity | 288GB vs 180GB (B200) |

### Kimi-K2.5: Profile-Driven Optimization

**Step 1 -- Identify bottleneck**: Profiling revealed `fused_moe` consumed:
- 87.8% of GPU time at concurrency=2
- 89.7% of GPU time at concurrency=40

**Step 2 -- Apply FlyDSL optimization**: Custom kernel using Python-native MLIR DSL.

### FlyDSL Kernel Benchmarks (tokens=16384, dim=7168)

| Dtype | Triton (ms) | FlyDSL (ms) | Speedup |
|---|---|---|---|
| BF16 (A16W16) | 12.09 | 8.68 | **1.39x** |
| W4A16 | 31.43 | 9.77 | **3.22x** |

### End-to-End Results (Concurrency=40)

| Metric | Before | After | Improvement |
|---|---|---|---|
| TTFT (mean) | 33.5s | 17.7s | **-47.0%** |
| TPOT (mean) | 230ms | 71ms | **-69.2%** |
| Output throughput | 135 tok/s | 355 tok/s | **+162.4%** |
| GSM8K accuracy | 0.96 | 0.96 | Identical |

### Configuration

```bash
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1
export FLYDSL_W4A16_HYBRID=w2_bf16  # Stage1=W4A16, Stage2=BF16
export SGLANG_USE_AITER=1

# Additional flags
--disable-radix-cache --enable-torch-compile --disable-custom-all-reduce
```

### Lessons Learned

1. **Profile first**: `fused_moe` dominated 88-90% of GPU time -- targeted optimization has outsized impact
2. **FlyDSL** enables Python-native kernel authoring with instruction-level control (MFMA, LDS, register scheduling)
3. **Mixed precision across MoE stages** (W4A16 gate/up + BF16 down projection) balances throughput and accuracy
4. **`--disable-radix-cache`** helps when benchmarks use random inputs with no shared prefixes
5. **MI355X 288GB** provides decisive high-concurrency advantage over B200's 180GB

---

## 3. MI355X Training Performance

**Source**: MI355X Training Blog (Dec 2025)

### MI355X vs B200 -- Single Node (PyTorch)

| Model | Precision | MI355X Advantage |
|---|---|---|
| Llama3 70B | FP8 | 1.00x (parity) |
| Llama3 70B | BF16 | **1.16x** |
| Llama3 8B | FP8 | 1.08x |
| Llama3 8B | BF16 | 1.02x |
| Mixtral 8x7B | FP16 | **1.15x** |

### MI355X vs B200 -- JAX MaxText

| Model | Precision | MI355X Advantage |
|---|---|---|
| Llama3.1 70B | FP8 | **1.11x** |
| Llama3.1 8B | FP8 | 1.07x |
| Mixtral 8x7B | FP16 | 1.00x (parity) |

### Multi-Node Scaling

| Model | Config | MI355X Advantage |
|---|---|---|
| Mixtral 8x22B BF16 | 4-node | **1.14x** |
| Llama3 70B FP8 | 4-node | 1.01x |
| Llama3.1 405B FP8 | 8-node | 0.96x |

### MI355X Hardware Specifications

| Feature | Value |
|---|---|
| Architecture | CDNA4 |
| HBM | 288GB HBM3E |
| Memory bandwidth | 8 TB/s |
| CUs | 256 |
| LDS per CU | 160 KB |
| LDS banks | 64 |
| FP8 peak | Higher than CDNA3 (new MFMA shapes) |
| New capabilities | FP6/FP4 support, block-scaled MFMA |

### Training Stack

- **Primus**: Unified LLM training framework (TorchTitan + Megatron-LM)
- **Primus-Turbo**: Transformer model accelerator for MI355X

### Lessons Learned

1. MI355X achieves **parity or advantage** vs B200 on most workloads, with strongest gains in BF16 training
2. Multi-node scaling is competitive up to 4 nodes; 8-node 405B FP8 shows slight disadvantage (communication overhead)
3. JAX MaxText and PyTorch show similar relative performance, validating cross-framework support
4. Mixtral MoE workloads scale well on MI355X (1.14x at 4-node)

---

## 4. MoE Training Best Practices (DeepSeek-Scale)

**Source**: ROCm Blog -- MoE Training Best Practices (Primus MoE Package)

### Representative Model Scales

| Model | Total Parameters | Active Parameters |
|---|---|---|
| DeepSeek-V2-Lite | 16B | 2.4B |
| DeepSeek-V2 | 236B | 21B |
| MoE-1T | 1T | 44B |
| MoE-2T | 2T | 80B |

### Performance Bottleneck Analysis

1. **Grouped GEMM overhead**: Multi-stream approaches still have scheduling gaps
2. **All-to-all communication**: EP >= 8 causes significant cross-node overhead
3. **CPU synchronization latency**: D2H sync blocks kernel launch queue
4. **Too many small kernels**: Fine-grained MoE operators create CPU launch pressure
5. **Pipeline load imbalance**: Uneven work distribution throttles overall throughput

### Key Optimization Techniques

#### Turbo Grouped GEMM

Uses CK to fuse grouped GEMM into a single kernel for all experts. Auto-tuning selects the fastest backend for forward and backward passes.

#### DeepEP Acceleration

GPU-side index computation replaces CPU coordination. Eliminates CPU-GPU synchronization for fully sync-free pipeline.

#### Sync-Free MoE (4 Levels)

| Level | Description | Trade-off |
|---|---|---|
| 0 | Default (disabled) | Baseline |
| 1 | Remove Router + Permutation sync | Small memory increase |
| 2 | Remove Router + DeepEP + GroupMLP sync | Moderate memory increase |
| 3 | Fully sync-free | High memory consumption |

#### 1F1B A2A Overlap

Interleaves communication and computation: micro-batch N's all-to-all communication overlaps with micro-batch N-1's backward computation.

#### CPU Launch Optimization

```bash
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912  # 12MB kernel arg pool
```

### Lessons Learned

1. **Sync-free MoE** is the highest-impact optimization for large-scale MoE training
2. **DeepEP** eliminates the #3 bottleneck (CPU sync) entirely
3. **NUMA binding** + large kernel arg pool reduce CPU-side overhead
4. **Selective recomputation** (`--recompute_layer_ids`) balances memory and compute
5. **Pipeline visualization** (pp_vis) is essential for identifying load imbalance

---

## 5. FP8 GEMM Progressive Optimization on CDNA4

**Source**: ROCm Blog -- FP8 GEMM Optimization on AMD CDNA4 (Mar 2026)

This case study demonstrates the full optimization journey from naive implementation to near-peak performance, covering every major GPU optimization technique.

### Performance Progression (M=N=K=4096)

| Stage | TFLOP/s | Speedup vs Naive | Key Technique |
|---|---|---|---|
| Naive (1 thread/element) | 1.15 | 1x | Baseline |
| LDS tiling | 4.80 | 4.2x | Shared memory blocking |
| MFMA matrix core | 30.05 | 26x | Matrix instructions |
| + Vectorized FP8x16 loads | 336.88 | 293x | 128-bit vector load |
| + Direct global-to-LDS | 506.70 | 441x | `buffer_load_lds` bypass |
| + LDS swizzle | 497.43 | 432x | XOR bank conflict elimination |
| + Double-buffer pipeline | 1,166.41 | 1,014x | Software pipelining |
| + 256x256 tile, 8-wave | 2,288.16 | 1,990x | Ping-pong scheduling |
| hipBLASLt reference (4096) | ~2,750 | -- | Production library |
| hipBLASLt reference (8192) | ~3,130 | -- | Production library |

### Why Each Stage Matters

| Technique | Bottleneck Addressed |
|---|---|
| MFMA 16x16x128 | 65,536 FLOPs/instruction vs 128 for FMA (512x per-instruction increase) |
| Vectorized FP8x16 load | Reduces load instruction count by 16x |
| Global-to-LDS direct | Eliminates VGPR pressure from data staging; 128 bits/lane on CDNA4 |
| LDS swizzle | Eliminates bank conflicts in 4-phase `ds_read_b128` on 64-bank LDS |
| Double buffering | Hides global memory latency by overlapping load(t+1) with compute(t) |
| 8-wave ping-pong | Maximizes SIMD utilization: 2 waves/SIMD alternate memory and MMA |

### CDNA4 vs CDNA3 Hardware Enablers

| Feature | CDNA4 (MI355X) | CDNA3 (MI300X) |
|---|---|---|
| LDS capacity | **160 KB/CU** | 64 KB |
| LDS banks | **64** | 32 |
| LDS read bandwidth | **256 B/clk** | 128 B/clk |
| GLOBAL_LOAD_LDS per-lane | **128 bits** | 32 bits |
| FP4/FP6 MFMA | Supported | No |
| Block-scaled MFMA | Supported | No |
| FP16 MFMA shapes | 16x16x32, 32x32x16 | 16x16x16, 32x32x8 |

### Lessons Learned

1. The jump from naive to MFMA (26x) is the single largest improvement -- always use matrix cores
2. LDS swizzle alone can show slight regression (497 vs 507) if double buffering is not applied simultaneously -- apply swizzle + double buffer together
3. The 256x256 tile with 8 waves outperforms 128x128; larger tiles better amortize memory access overhead
4. 1024-thread variants show diminishing returns from synchronization overhead
5. Achieving >80% of hipBLASLt requires instruction-level scheduling control (s_setprio, sched_barrier)

---

## 6. GEAK: AI Agent for HIP Kernel Optimization

**Source**: ROCm Blog -- GEAK HIP (Dec 2025)

### Framework

Generator -> Evaluator -> Reflector loop:
- **Generator**: Takes instructions + baseline HIP code, outputs optimized code
- **Evaluator**: In-place replacement, compilation, execution, performance extraction
- **Reflector**: On compilation/execution failure, feeds errors back for regeneration

### Results

| Case | Agent Speedup | Manual Engineer |
|---|---|---|
| ROCm examples (6 kernels) | 1.08x avg, 1.20x max | -- |
| MMCV examples (10 kernels) | 1.20x avg, 2.15x max | -- |
| **Voxelization** | **2.07x** | 1.84x |
| **SwiGLU** | **1.68x** | 1.30x |
| GEMM heuristic (Qwen3-32B) | 1.28x vs default | 0.8x vs fully tuned |

### Agent-Discovered Patterns

**Voxelization optimizations**:
- Shared memory caching of predecessor coordinates
- Coalesced parallel loads
- Block-sized tiling
- Unrolled loop ILP
- `launch_bounds` occupancy hints
- Early exits

**SwiGLU optimizations**:
- `bf16x2` pairs + `uint4` 128-bit vectorization
- 16B alignment detection with fallback
- `__expf`/`__fdividef` fast math intrinsics
- Instruction interleaving across elements

**GEMM heuristic**:
- Auto-generated size-dependent kernel selection rules
- Based on M/N/K geometry (skinny, square, tall, tiny)
- 1.28x over defaults without exhaustive tuning

### Lessons Learned

1. Agent-optimized code **surpassed** human kernel engineers on specific kernels (2.07x vs 1.84x)
2. Multiple offspring (parallel optimization variants) improve max speedup but slow generation
3. The generator-evaluator-reflector loop is applicable to both Triton and HIP
4. AI agents can generate useful GEMM heuristics without exhaustive tuning

---

## 7. Cross-Cutting Lessons

### Profile-Driven Optimization

Every successful case study started with profiling to identify the dominant bottleneck before optimizing. The `fused_moe` kernel consuming 88-90% of GPU time in Kimi-K2.5 is the canonical example. Tools: RPD, TorchProfiler, rocprof-compute.

### Memory Capacity as Competitive Advantage

MI300X (192GB) and MI355X (288GB) consistently outperform H200 (141GB) and B200 (180GB) at high concurrency. Memory capacity enables larger batch sizes, more KV cache, and more concurrent users under latency SLOs.

| GPU | Memory | High-Concurrency Advantage |
|---|---|---|
| MI355X | 288GB HBM3E | Decisive advantage over B200 (180GB) |
| MI300X | 192GB HBM3 | Strong advantage over H200 (141GB) |

### Kernel Fusion and Mixed Precision

Three complementary approaches:
- **FlyDSL**: Python-native MLIR DSL for thread-level control
- **Composable Kernel (CK)**: Tile-based HIP C++ templates
- **AITER**: Centralized operator library with pre-optimized kernels

Mixed precision consistently delivers throughput gains without accuracy loss:
- W4A16 + BF16 across MoE stages
- FP8 block-scale GEMM
- MXFP4/MXFP6 with RNE rounding

### Software-Hardware Co-Design

CDNA4 hardware advances require kernel rewrites to exploit:
- Wider GLOBAL_LOAD_LDS (128-bit vs 32-bit)
- Larger LDS (160KB vs 64KB)
- New MFMA shapes (FP4/FP6/block-scaled)
- 64 LDS banks (vs 32) requiring new swizzle patterns

The 8-wave ping-pong scheduling pattern demonstrates the level of instruction-level control needed for peak performance on AMD GPUs.

### Practical Optimization Priority

Based on impact observed across all case studies:

| Priority | Technique | Typical Impact |
|---|---|---|
| 1 | Use AITER pre-optimized kernels | 2x-17x per kernel |
| 2 | Profile and target dominant kernel | Depends on bottleneck |
| 3 | Enable torch.compile / HIP graphs | 10-30% end-to-end |
| 4 | GEMM tuning (online -> offline -> TensileLite) | 6-225% per GEMM shape |
| 5 | Custom kernel (FlyDSL/CK) for hot path | 1.4x-3.2x per kernel |
| 6 | Prefill-decode disaggregation | 2-7x goodput |
| 7 | GPU partitioning for multi-tenancy | Higher aggregate throughput |
