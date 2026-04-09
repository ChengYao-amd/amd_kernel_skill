# P3 Deep Dive: ROCm Optimization Techniques & Case Studies

> Crawled 2026-04-09 from AMD ROCm blogs and docs. Organized by topic with performance numbers, configuration patterns, and lessons learned.

---

## 1. DeepSeek-R1/V3 Inference Optimization on MI300X

**Sources:** DeepSeek-R1 Part2 (Mar 2025), DeepSeekV3 Kernel Analysis (May 2025)

### Performance Numbers (MI300X vs H200)
| Metric | MI300X | vs H200 |
|---|---|---|
| Throughput at same latency | — | 2X–5X higher |
| Throughput at same concurrency | — | Up to 75% higher |
| Latency at same concurrency | — | Up to 60% lower |
| Max concurrent requests < 50ms ITL | 128 | H200: 16 |

### AITER Kernel-Level Speedups
| Kernel | Speedup over Baseline |
|---|---|
| Block-scale GEMM | 2X |
| Block-scale fused MoE | 3X |
| MLA decode | 17X |
| MHA prefill | 14X |

### Profiling Insights (DeepSeekV3 on SGLang)
- **Prefill phase**: Attention + MoE consume majority of compute time; GEMM and misc are secondary
- **Decode phase**: MoE is dominant; attention takes smaller share; rest spread across GEMM, all_reduce, misc
- **Key bottleneck**: Inefficient device-to-host memory transfers (data copied GPU→CPU unnecessarily)
- **Decode kernel latency**: MLA is 2x less and MoE is 3x less vs Nvidia H200

### Configuration Pattern
```bash
# SGLang server for DeepSeek-R1 on MI300X
python3 -m sglang.launch_server \
  --model /model --tp 8 --trust-remote-code \
  --chunked-prefill-size 131072 \
  --enable-torch-compile --torch-compile-max-bs 256
```

### Lessons Learned
- `chunked_prefill_size=131072` accelerates prefill but costs more VRAM; tune per use-case
- MI300X 192GB HBM3 enables higher concurrency than H200 141GB — memory capacity is a competitive advantage
- `--enable-torch-compile` reduces CPU-side kernel launch overhead, especially during decode
- Profiling tools (RPD + TorchProfiler) essential for kernel-level bottleneck identification

---

## 2. vLLM V1 Performance Optimization on ROCm

**Source:** ROCm docs — vLLM V1 performance optimization guide

### Critical Environment Variables
```bash
export HIP_FORCE_DEV_KERNARG=1          # Kernel launch perf (default in Docker)
export TORCH_BLAS_PREFER_HIPBLASLT=1    # Prefer hipBLASLt for GEMM
export NCCL_MIN_NCHANNELS=112           # Multi-GPU RCCL channels (MI300X)
export VLLM_ROCM_USE_AITER=1            # Master switch for all AITER optimizations
```

### AITER Feature Flags (all default True when master switch on)
| Flag | Purpose |
|---|---|
| `VLLM_ROCM_USE_AITER_LINEAR` | FP8 quantized GEMM for linear layers |
| `VLLM_ROCM_USE_AITER_MOE` | Fused MoE routing + computation |
| `VLLM_ROCM_USE_AITER_RMSNORM` | Accelerated RMSNorm |
| `VLLM_ROCM_USE_AITER_MLA` | Multi-head Latent Attention (DeepSeek) |
| `VLLM_ROCM_USE_AITER_MHA` | Multi-Head Attention (Llama, Mistral) |
| `VLLM_ROCM_USE_AITER_FP8BMM` | FP8 batched matmul for MLA |
| `VLLM_ROCM_USE_SKINNY_GEMM` | Skinny-GEMM for small batch sizes |

### Attention Backend Selection
| Model Type | Recommended Backend |
|---|---|
| Standard transformers (Llama, Mistral, Qwen) | AITER MHA (`VLLM_ROCM_USE_AITER=1`) |
| MLA models (DeepSeek-V3/R1/V2) | AITER MLA (auto-selected, `--block-size 1` required) |
| gpt-oss models | AITER Unified Attention |
| Debugging/fallback | vLLM Triton Unified (default) |

### Parallelism Strategy Guide
| Strategy | When to Use |
|---|---|
| Tensor Parallelism (TP) | Model doesn't fit on 1 GPU; stay within XGMI island (≤8 GPUs) |
| Pipeline Parallelism (PP) | Multi-node; TP per node, PP across nodes |
| Data Parallelism (DP) | Model fits on 1 GPU or TP group; need higher throughput |
| Expert Parallelism (EP) | MoE models cross-node with fast interconnect |

### Quick Reduce (Multi-GPU All-Reduce)
- Supports FP16/BF16 + symmetric INT8/INT6/INT4 quantized all-reduce
- Helps throughput at TP 4–8 with many concurrent requests
- Quantization affects accuracy — validate before deploying

### Key Tuning Knobs
| Knob | Latency-Sensitive | Max Throughput |
|---|---|---|
| `--max-num-batched-tokens` | 8k–16k | ≥32k |
| `cudagraph_mode` | PIECEWISE | FULL |
| `--gpu-memory-utilization` | 0.90 (default) | 0.95 |

### Lessons Learned
- For 95% of users: simply set `VLLM_ROCM_USE_AITER=1` and let vLLM auto-select the backend
- DeepSeek MLA **requires** `--block-size 1` — omitting it causes an error
- Total throughput from N single-GPU instances usually exceeds one instance stretched across N GPUs with `-tp N`
- MI300X = 192GB HBM3; MI355X = 288GB HBM3E — plan instance packing accordingly

---

## 3. Kimi-K2/K2.5 MoE Inference Optimization

**Sources:** Kimi-K2 on MI355 (Oct 2025), Kimi-K2.5 with FlyDSL (Mar 2026)

### Kimi-K2 on MI355X vs B200
| Metric | MI355X Advantage |
|---|---|
| Mean TTFT (high concurrency) | >3X better than B200 |
| Mean E2E latency (high concurrency) | Significantly lower |
| Total token throughput | Better as concurrency increases |
| Memory capacity | 288GB vs 180GB (B200) |

### Kimi-K2.5 Fused MoE Optimization (FlyDSL)

**Profiling identified** `fused_moe` as the bottleneck:
- 87.8% of GPU time at concurrency=2
- 89.7% of GPU time at concurrency=40

**FlyDSL kernel benchmarks** (most critical shape: tokens=16384, dim=7168):
| dtype | Triton (ms) | FlyDSL (ms) | Speedup |
|---|---|---|---|
| BF16 (A16W16) | 12.09 | 8.68 | 1.39x |
| W4A16 | 31.43 | 9.77 | 3.22x |

**End-to-End Results (Concurrency=40)**:
| Metric | Improvement |
|---|---|
| TTFT (mean) | -47.0% (33.5s → 17.7s) |
| TPOT (mean) | **-69.2%** (230ms → 71ms) |
| Output throughput | **+162.4%** (135 → 355 tok/s) |
| GSM8K accuracy | Identical (0.96) |

### Key Configuration
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
- Profile first: `fused_moe` dominated 88-90% of GPU time — targeted optimization has outsized impact
- FlyDSL enables Python-native kernel authoring with instruction-level control (MFMA, LDS, register scheduling)
- Mixed-precision across MoE stages (W4A16 gate/up + BF16 down projection) balances throughput and accuracy
- `--disable-radix-cache` helps when benchmarks use random inputs with no shared prefixes
- MI355X 288GB memory gives decisive high-concurrency advantage over B200's 180GB

---

## 4. Adaptive Top-K Selection (AITER Library)

**Source:** Adaptive Top-K blog (Feb 2026)

### Problem
Standard Radix Sort suffers performance cliffs at small K values due to fixed histogram overhead (LDS atomics scan all 2048 buckets regardless of K).

### Solution: Two Complementary Strategies
| Strategy | Best For | Mechanism |
|---|---|---|
| **BlockTopkSort** | Small input per warp | Bitonic sort on register-resident data via DPP instructions |
| **BlockTopkFilter** | Larger inputs | Ballot-based filtering prunes candidates before sorting |
| **AdaptiveTopK** | All K values | Auto-selects between bitonic (small K) and radix (large K) |

### Hardware-Specific Optimizations
| Technique | Impact |
|---|---|
| DPP (Data Parallel Primitives) | Ultra-low-latency register exchange (faster than shuffle) |
| `med3` instruction | Branch-free compare-and-swap via median-of-3 hardware |
| Buffer `load_dwordx4` | 4-8x fewer load instructions (16 bytes per instruction) |
| Double buffering | Hides memory latency; compute on chunk N while loading N+1 |

### Performance Highlights
- DPP + med3 optimizations: **up to 32% improvement** over shuffle/conditional baseline
- Buffer instructions: **up to 55% improvement** on long sequences (131K length)
- AdaptiveTopK consistently beats PyTorch `torch.topk`, Triton, and radix_11bits across all K values

### Adaptive Threshold Formula (MI300X-tuned)
```
n + K·log²(K) ≥ 3 × Factor(n) × n
Factor(n) = 1/3 + 1.6/(log₂(n) - 9.5)
```
- n=8192 → threshold K ≈ 195 (use bitonic below, radix above)
- n=65536 → threshold K ≈ 576
- n=131072 → threshold K ≈ 878

---

## 5. FP8 GEMM Kernel Optimization on CDNA4 (MI355X)

**Source:** FP8 GEMM blog (Mar 2026)

### Progressive Optimization Results (M=N=K=4096)
| Stage | TFLOPS/s | Speedup vs Naive |
|---|---|---|
| Naive (1 thread/element) | 1.15 | 1x |
| LDS tiling | 4.80 | 4.2x |
| MFMA matrix-core | 30.05 | 26x |
| + Vectorized loads | 336.88 | 293x |
| + Direct global-to-LDS load | 506.70 | 441x |
| + LDS swizzling + double buffer | 1,166.41 | 1,014x |
| + Multi-wave (256x256_t512) | 2,288.16 | 1,990x |
| hipBLASLt reference (4096) | ~2,750 | — |
| hipBLASLt reference (8192) | ~3,130 | — |

### CDNA4 vs CDNA3 Key Differences
| Feature | CDNA4 (MI355X) | CDNA3 (MI300X) |
|---|---|---|
| LDS capacity | 160 KB/CU | 64 KB |
| LDS banks | 64 | 32 |
| LDS read bandwidth | 256 B/clk | 128 B/clk |
| GLOBAL_LOAD_LDS per-lane | 128 bits | 32 bits |
| FP4/FP6 MFMA | Supported | No |
| Block-scaled MFMA | Supported | No |

### Key Techniques
1. **MFMA instructions**: Single `16x16x128` FP8 MFMA = 65,536 FLOPs (vs 128 for FMA) → 512x larger per instruction
2. **LDS swizzling**: XOR-based row remap eliminates bank conflicts in `ds_read_b128` four-phase access
3. **Double buffering**: Ping-pong LDS slots overlap load of tile t+1 with compute of tile t
4. **8-wave ping-pong scheduling**: Alternates memory and MMA instructions across 2 waves per SIMD using `s_barrier`, `s_setprio`, `sched_barrier` intrinsics
5. **`GLOBAL_LOAD_LDS`**: Direct global-to-LDS bypasses registers — 4x wider per lane on CDNA4

---

## 6. GEAK: AI Agent for HIP Kernel Optimization

**Source:** GEAK HIP blog (Dec 2025)

### Framework: Generator → Evaluator → Reflector loop
- **Generator**: Takes instructions + baseline HIP code → outputs optimized code
- **Evaluator**: In-place replacement, compilation, execution, performance extraction
- **Reflector**: Activates on compilation/execution failure; feeds errors back

### Case Study Results
| Case | Speedup (Agent) | vs Manual Engineer |
|---|---|---|
| ROCm examples (6 kernels) | 1.08x avg, 1.20x max | — |
| MMCV examples (10 kernels) | 1.20x avg, 2.15x max | — |
| **Voxelization** | **2.07x** | Engineer: 1.84x |
| **SwiGLU** | **1.68x** | Engineer: 1.30x |
| GEMM heuristic (Qwen3-32B) | 1.28x vs default | 0.8x vs fully tuned |

### Agent-Discovered Optimization Patterns
**Voxelization**: Shared memory caching of predecessor coordinates, coalesced parallel loads, block-sized tiling, unrolled loop ILP, `launch_bounds` occupancy hints, early exits

**SwiGLU**: `bf16x2` pairs + `uint4` 128-bit vectorization, 16B alignment detection with fallback, `__expf`/`__fdividef` fast math intrinsics, instruction interleaving across elements

**GEMM Heuristic**: Auto-generated size-dependent kernel selection rules based on M/N/K geometry (skinny, square, tall, tiny)

### Lessons Learned
- Agent-optimized code **surpassed** human kernel engineers on Voxelization (2.07x vs 1.84x) and SwiGLU (1.68x vs 1.30x)
- Multiple offspring (parallel optimization variants) improve max speedup but slow generation
- AI agents can generate useful GEMM heuristics (1.28x over defaults) without exhaustive tuning

---

## 7. ROCm as First-Class vLLM Platform

**Source:** vLLM-omni blog (Jan 2026)

### Milestones
- **CI pass rate**: 37% (Nov 2025) → 93% (Jan 2026), targeting 100%
- **vLLM v0.14.0**: Official ROCm Docker images + wheel pipeline with sccache
- **Installation**: `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`
- **vLLM-Omni**: Day-0 ROCm support (Nov 2025), ROCm CI pipeline (Dec 2025), Docker image (Jan 2026)

### vLLM v0.12.0–v0.14.0 ROCm Highlights
- **Quantization**: Native AITER FP8, fused LayerNorm/SiLU FP8 block quant, MXFP4 W4A4 MoE, FP8 MLA decode
- **Performance**: Optimized KV cache + assembly Paged Attention, AITER sampling ops, fastsafetensors loading
- **New capabilities**: DeepSeek v3.2 + SparseMLA, Whisper v1, sliding window attention, multi-token prediction for MLA
- **Multi-modal**: Qwen2.5-Omni, Qwen3-Omni-MoE (text/image/audio/video → text/audio), vLLM-Omni ROCm Docker

---

## 8. MI355X Training Performance

**Source:** MI355X training blog (Dec 2025)

### MI355X vs B200 (Single-Node PyTorch)
| Model | Precision | MI355X Ratio |
|---|---|---|
| Llama3 70B | FP8 | 1.00x |
| Llama3 70B | BF16 | **1.16x** |
| Llama3 8B | FP8 | 1.08x |
| Llama3 8B | BF16 | 1.02x |
| Mixtral 8x7B | FP16 | **1.15x** |

### MI355X vs B200 (JAX MaxText)
| Model | Precision | MI355X Ratio |
|---|---|---|
| Llama3.1 70B | FP8 | **1.11x** |
| Llama3.1 8B | FP8 | 1.07x |
| Mixtral 8x7B | FP16 | 1.00x |

### Multi-Node Scaling
| Model | Config | MI355X Ratio |
|---|---|---|
| Mixtral 8x22B BF16 | 4-node | **1.14x** |
| Llama3 70B FP8 | 4-node | 1.01x |
| Llama3.1 405B FP8 | 8-node | 0.96x |

### Training Stack
- **Primus**: Unified LLM training framework supporting TorchTitan + Megatron-LM backends
- **Primus-Turbo**: Transformer model accelerator for MI355X
- MI355X: CDNA4, 288GB HBM3E, 8TB/s bandwidth, FP6/FP4 support

---

## 9. Profiling Toolchain

**Sources:** ROCm profiling blog (May 2024), Profiling guide Part 1 (Jun 2025), DeepSeekV3 kernel analysis (May 2025)

### Tool Selection Guide
| Tool | Use Case |
|---|---|
| `rocprofv3` | CLI tracing + raw counter collection (replaces legacy rocprof) |
| `rocprof-sys` | Holistic host + device + MPI tracing in one unified trace |
| `rocprof-compute` | Kernel roofline analysis, baseline comparisons, speed-of-light |
| RPD (RocmProfileData) | Timeline tracing of API calls, GPU ops, dependencies |
| TorchProfiler | PyTorch call stack + GPU trace (deep model-level insight) |
| Perfetto UI | Visualization of `.pftrace` / `.json` traces |

### Profiling Workflow
1. **End-to-end benchmarking** → establish baseline
2. **Roofline analysis** → assess hardware utilization (peak FLOPs)
3. **Kernel breakdown** (RPD/TorchProfiler) → identify hotspots
4. **Gap identification** → targeted optimization → iterate

### Key Profiling Commands
```bash
# RPD tracing in SGLang
export RPDT_AUTOFLUSH=1
runTracer.sh python3 -m sglang.launch_server ...

# Convert to viewable JSON (last 2% of trace)
python3 rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json --start 98% --end 100%

# TorchProfiler in SGLang
export SGLANG_TORCH_PROFILER_DIR=/profile/
curl http://localhost:30000/start_profile
# ... run benchmark ...
curl http://localhost:30000/stop_profile
```

### Profiling Best Practices (from vLLM docs)
1. Fix prompt distribution (ISL/OSL) and vary one knob at a time
2. Measure TTFT, ITL, and TPS together — don't optimize one in isolation
3. Compare graph modes: PIECEWISE (balanced) vs FULL (max throughput)
4. Sweep `--max-num-batched-tokens` around 8k–64k for latency/throughput balance

---

## 10. Composable Kernel (CK) Library

**Source:** CK docs (v1.2.0)

### Architecture: 4-Layer Design
1. **Templated tile operator layer** — tile-based programming model
2. **Templated kernel + invoker layer** — kernel composition
3. **Instantiated kernel + invoker layer** — concrete kernels
4. **Client API layer** — user-facing interface

### CK Tile System Concepts
- **Tile Distribution**: Maps work from block → warp → thread → MFMA instruction
- **Coordinate Systems**: 5 coordinate spaces (thread ID → logical work → physical tensor → memory)
- **Space-Filling Curves**: Optimal memory traversal patterns
- **LoadStoreTraits**: Vectorization selection, memory access pattern optimization
- **LDS Index Swapping**: Bank conflict elimination via swizzled layouts
- **Static Distributed Tensor**: Thread-local register storage with coordination

### Key Insight
CK uses **tensor coordinate transformation** as a complexity reduction technique — complex ML operators are decomposed into composable tile operations with explicit memory hierarchy control (global → LDS → register → MFMA).

---

## 11. ROCm Offline Installer Creator

**Source:** ROCm offline installer blog

### Features
- **Multi-distro**: Ubuntu 20.04/22.04/24.04, RHEL 8/9, SUSE 15 SP5/SP6
- **Customization**: ROCm version selection, component picking, AMDGPU driver integration
- **Air-gapped deployment**: Auto-resolves and packages all dependencies
- Menu-driven UI; produces self-contained `.run` file

### Workflow
1. Download from `https://repo.radeon.com/rocm/installer/rocm-linux-install-offline/`
2. Run on internet-connected machine → select ROCm version + components
3. Copy `rocm-offline-install.run` to target → run offline

---

## Cross-Cutting Themes & Lessons

### 1. Profile-Driven Optimization
Every successful case study (DeepSeek, Kimi-K2.5, FP8 GEMM) started with profiling to identify the dominant bottleneck before optimizing. The `fused_moe` kernel consuming 88-90% of GPU time in Kimi-K2.5 is the canonical example.

### 2. Memory Capacity as Competitive Advantage
MI300X (192GB) and MI355X (288GB) consistently win at high concurrency against H200 (141GB) and B200 (180GB). This enables larger batch sizes, more KV cache, and more concurrent users under latency SLOs.

### 3. Kernel Fusion & Mixed Precision
FlyDSL (Python-native MLIR DSL), Composable Kernel (tile-based HIP C++), and AITER (centralized operator library) are three complementary approaches. Mixed precision (W4A16 + BF16, FP8 block-scale) consistently delivers throughput gains without accuracy loss.

### 4. AI Agents for Kernel Optimization
GEAK demonstrates that LLM-driven code optimization can **surpass human engineers** on specific kernels (2.07x vs 1.84x on voxelization). The generator→evaluator→reflector loop pattern is applicable to both Triton and HIP.

### 5. Software-Hardware Co-Design
CDNA4's wider GLOBAL_LOAD_LDS (128-bit vs 32-bit), larger LDS (160KB vs 64KB), and new MFMA shapes (FP4/FP6/block-scaled) require kernel rewrites to exploit. The 8-wave ping-pong scheduling pattern shows the level of instruction-level control needed for peak performance.
