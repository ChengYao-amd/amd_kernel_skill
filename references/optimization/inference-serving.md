# Inference Serving on AMD GPUs

> Covers vLLM, SGLang, speculative decoding, prefill-decode disaggregation, GPU partitioning for inference, and the Primus training framework on ROCm.

---

## Table of Contents

1. [vLLM on ROCm](#1-vllm-on-rocm)
2. [SGLang on ROCm](#2-sglang-on-rocm)
3. [Prefill-Decode Disaggregation](#3-prefill-decode-disaggregation)
4. [Speculative Decoding on MI300X](#4-speculative-decoding-on-mi300x)
5. [GPU Partitioning for Multi-Instance Inference](#5-gpu-partitioning-for-multi-instance-inference)
6. [hipBLASLt GEMM Tuning for Inference](#6-hipblaslt-gemm-tuning-for-inference)
7. [FlyDSL -- Python-First Kernel DSL](#7-flydsl----python-first-kernel-dsl)
8. [Primus Training Framework](#8-primus-training-framework)
9. [Environment Variable Quick Reference](#9-environment-variable-quick-reference)

---

## 1. vLLM on ROCm

### ROCm Platform Status

- vLLM v0.14.0+: Official ROCm Docker images and wheel pipeline
- CI pass rate: 37% (Nov 2025) to 93% (Jan 2026), targeting 100%
- Installation: `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`
- Day-0 ROCm support via vLLM-Omni (Nov 2025)

### Critical Environment Variables

```bash
export HIP_FORCE_DEV_KERNARG=1          # Kernel launch latency reduction (default in Docker)
export TORCH_BLAS_PREFER_HIPBLASLT=1    # Prefer hipBLASLt for GEMM
export NCCL_MIN_NCHANNELS=112           # Multi-GPU RCCL channels (MI300X)
export VLLM_ROCM_USE_AITER=1            # Master switch for all AITER optimizations
```

### AITER Feature Flags

When the master switch `VLLM_ROCM_USE_AITER=1` is enabled, all sub-flags default to True.

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

**Important**: DeepSeek MLA requires `--block-size 1`. Omitting it causes an error.

### Parallelism Strategy Guide

| Strategy | When to Use |
|---|---|
| Tensor Parallelism (TP) | Model does not fit on 1 GPU; stay within XGMI island (up to 8 GPUs) |
| Pipeline Parallelism (PP) | Multi-node; TP per node, PP across nodes |
| Data Parallelism (DP) | Model fits on 1 GPU or TP group; need higher throughput |
| Expert Parallelism (EP) | MoE models cross-node with fast interconnect |

**Practical tip**: Total throughput from N single-GPU instances usually exceeds one instance stretched across N GPUs with `-tp N`.

### Quick Reduce (Multi-GPU All-Reduce)

- Alternative to RCCL for large all-reduces
- Supports FP16/BF16 + symmetric INT8/INT6/INT4 quantized all-reduce
- Helps throughput at TP 4-8 with many concurrent requests
- Quantization affects accuracy -- validate before deploying

### Key Tuning Knobs

| Knob | Latency-Sensitive | Max Throughput |
|---|---|---|
| `--max-num-batched-tokens` | 8k-16k | 32k or higher |
| `cudagraph_mode` | PIECEWISE | FULL |
| `--gpu-memory-utilization` | 0.90 (default) | 0.95 |

### Quantization Features (v0.12.0-v0.14.0)

- Native AITER FP8
- Fused LayerNorm/SiLU FP8 block quantization
- MXFP4 W4A4 MoE
- FP8 MLA decode
- FP8 KV cache: `--kv-cache-dtype fp8` for 2x memory reduction (~0.1% accuracy loss)

### Flash Attention Backend Selection

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"  # CK backend (default, generally faster)
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"   # Triton backend
```

### PyTorch TunableOp

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
```

Automatically selects the best GEMM kernel from rocBLAS/hipBLASLt. Generates a GEMM lookup table for reuse across runs.

---

## 2. SGLang on ROCm

### Core Features

- **RadixAttention**: Radix tree-based automatic KV cache reuse
- **Jump-Forward Constrained Decoding**: Skips unnecessary computation
- **Continuous Batching**: Dynamic batch size adjustment
- **Paged Attention**: Partitioned attention matrices for long sequences

### Multi-GPU Deployment

```bash
# Tensor Parallelism (2 GPUs)
python -m sglang.launch_server --model-path <model> --tp 2

# Data Parallelism (2 DP x 2 TP = 4 GPUs)
python -m sglang.launch_server --model-path <model> --dp 2 --tp 2
```

### Quantization Options

```bash
# FP8 weight quantization
--quantization fp8

# FP8 KV cache quantization
--kv-cache-dtype fp8_e5m2

# AMD Quark FP8 model
--model-path amd/Meta-Llama-3.1-405B-Instruct-FP8-KV --tp 8 --quant fp8
```

### DeepSeek-R1 Configuration

```bash
python3 -m sglang.launch_server \
  --model /model \
  --tp 8 \
  --trust-remote-code \
  --chunked-prefill-size 131072 \
  --enable-torch-compile \
  --torch-compile-max-bs 256

export HSA_NO_SCRATCH_RECLAIM=1
```

**Notes**:
- `chunked_prefill_size=131072` accelerates prefill but costs more VRAM
- `--enable-torch-compile` reduces CPU-side kernel launch overhead (especially during decode)

### Docker Image

```bash
docker pull rocm/sgl-dev:upstream_20250312_v1
docker run -it --ipc=host --network=host --privileged --shm-size 32G \
  --device=/dev/kfd --device=/dev/dri \
  -v $MODEL_DIR:/model \
  rocm/sgl-dev:upstream_20250312_v1
```

### Profiling SGLang

```bash
# RPD tracing
export RPDT_AUTOFLUSH=1
runTracer.sh python3 -m sglang.launch_server ...

# Convert to Perfetto-viewable JSON (last 2% of trace)
python3 rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json --start 98% --end 100%

# TorchProfiler integration
export SGLANG_TORCH_PROFILER_DIR=/profile/
curl http://localhost:30000/start_profile
# ... run benchmark ...
curl http://localhost:30000/stop_profile
```

---

## 3. Prefill-Decode Disaggregation

### Concept

Separates prefill (compute-intensive) and decode (memory-bandwidth-intensive) phases onto different GPU groups, enabling independent scaling and optimization.

### SGLang Server Parameters

| Parameter | Description | Default |
|---|---|---|
| `--disaggregation-mode` | `prefill` or `decode` | null |
| `--disaggregation-transfer-backend` | KV transfer backend | mooncake |
| `--disaggregation-ib-device` | InfiniBand device | auto-detect |
| `--disaggregation-bootstrap-port` | Bootstrap port | 8998 |

### Fine-Grained Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | KV transfer threads per TP rank | `int(0.75 * cpu_count()) // 8)` (4-12) |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Parallel transfer queues | 4 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | KV index receive timeout (seconds) | 300 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | Health check interval (seconds) | 5.0 |

### Performance Results (Llama 3.3 70B, MI300X, 128 concurrent)

| Scenario | P95 ITL SLO | P99 ITL SLO | Goodput Improvement |
|---|---|---|---|
| Chatbot (3200in/800out) | **7.1x stricter** | **13.2x stricter** | **6.9x** (TTFT<=1s, TPOT<=25ms) |
| Heavy decode (1024in/2048out) | **1.3x stricter** | **6.1x stricter** | **2.23x** (TTFT<=500ms, TPOT<=25ms) |

**Key insight**: Disaggregation enables much tighter tail latency SLOs. The improvement is most dramatic in chatbot scenarios with long prefills and short outputs.

---

## 4. Speculative Decoding on MI300X

### Performance Results

| Framework | Mode | Speedup Range |
|---|---|---|
| gpt-fast | Eager | 1.26x - 1.71x |
| gpt-fast | torch.compile | **1.75x - 2.99x** |
| vLLM | Eager | 1.32x - 2.0x |
| vLLM | Graph | **1.5x - 2.9x** |

### Key Insights

- **Batch size matters**: Speedup decreases with batch size. Slowdown begins after BS=8 (eager) or BS=32 (graph mode)
- **torch.compile / HIP Graphs** reduce kernel launch overhead, creating a larger latency gap between draft and target models -- better speculative decoding benefit
- **Bandwidth-bound optimization**: Speculative decoding loses effectiveness as workload becomes compute-bound (high batch sizes)
- gpt-fast compile mode is **12-21% faster** than vLLM graph mode due to kernel recompilation

### When to Use Speculative Decoding

| Condition | SpD Effectiveness |
|---|---|
| Low batch size (1-8) | High |
| Medium batch size (8-32) | Moderate (graph mode helps) |
| High batch size (32+) | Low or negative |
| Memory-bandwidth-bound | High |
| Compute-bound | Low |

---

## 5. GPU Partitioning for Multi-Instance Inference

### MI300X Partition Modes

| Mode | Logical GPUs | Memory per Instance | Use Case |
|---|---|---|---|
| **SPX + NPS1** | 1 | 192GB | Single large model, full resources |
| **DPX + NPS2** | 2 | ~96GB each | Dual model serving |
| **CPX + NPS4** | 8 | ~24GB each | Small models, multi-tenant |

### Configuration Commands

```bash
# Set DPX partition
amd-smi set --gpu all --compute-partition DPX
amd-smi set --gpu all --memory-partition NPS2
sudo amd-smi reset -r  # reload driver

# Set CPX partition
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4
```

### Performance (Mistral-Nemo FP8, Single MI300X)

| Mode | Max Throughput | Scenario |
|---|---|---|
| SPX (1 instance) | ~4001 tokens/sec | 25.98 QPS |
| DPX (2 instances) | Higher aggregate | Lower per-instance |
| CPX (8 instances) | Highest aggregate for small models | Lowest per-instance |

### vLLM in Container with Partition

```bash
export HIP_VISIBLE_DEVICES="$GPU_ID"
vllm serve $MODEL --port 8000
```

### RCCL with CPX Mode

```bash
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=$((2*1024*1024*1024))
export MSCCLPP_READ_ALLRED=1
```

### MI300X CPX+NPS4 Performance (Intra-OAM)

| Configuration | Bus Bandwidth |
|---|---|
| Default SPX allreduce (PyTorch, ROCm 6.2.4) | ~170 GB/s |
| **Optimized CPX allreduce** (single OAM) | **~315 GB/s (PyTorch) / ~340 GB/s (rccl-tests)** |

---

## 6. hipBLASLt GEMM Tuning for Inference

### Three-Level Tuning Hierarchy

| Level | Method | Effort | Typical Speedup |
|---|---|---|---|
| **Online Tuning** | Runtime kernel selection | Low | ~106% of baseline |
| **Offline Tuning** | Pre-benchmarked kernel pool | Medium | ~100% of optimal |
| **TensileLite** | Generate new custom kernels | High | **225% of baseline** avg |

### Online Tuning (Easiest)

```bash
export HIP_ONLINE_TUNING=1
# For vLLM:
export VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1
```

- One-time overhead: ~31 seconds for benchmarking unseen GEMM shapes
- Performance: 100.79% of offline tuning on average
- Integrated into AITER framework

### TensileLite Tuning (Most Powerful)

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

### Key TensileLite Parameters

| Category | Parameters |
|---|---|
| Thread/Workgroup | WorkGroup [dim0, dim1, LocalSplitU], ThreadTile, MacroTile |
| Loop | LoopUnroll, DepthU, LoopDoWhile |
| Split-K | LocalSplitU (intra-WG), GlobalSplitU (inter-WG, needs atomic) |
| Memory | PrefetchGlobalRead, PrefetchLocalRead, VectorWidth |
| Instruction | MatrixInstruction (MFMA shape), wave tiling |

### Best Results

- Average speedup vs baseline: **225%**
- Best cases for small-M shapes (LLM decode): **150-320%**
- Example: shape (165, 14400, 120) achieved **3.2x speedup** (42.3us vs 135.9us)

---

## 7. FlyDSL -- Python-First Kernel DSL

### Overview

- Python-first, MLIR-native DSL for expert GPU kernel development on AMD
- Compilation: Python DSL -> AST transforms -> Fly dialect (MLIR) -> ROCDL -> HSACO
- Built on CuTe Layout Algebra
- Install: `pip install flydsl`

### Positioning vs Triton

| Aspect | Triton | FlyDSL |
|---|---|---|
| Abstraction level | Block-level | Thread-level and IR-level |
| Target users | Mainstream developers | Expert developers targeting roofline |
| Control granularity | Compiler-managed | Explicit lane control, register, custom layouts |

### Kimi-K2.5 FlyDSL Results

FlyDSL was used to optimize the `fused_moe` kernel (which consumed 88-90% of GPU time):

| Metric | Improvement |
|---|---|
| `fused_moe` kernel (tokens=16384, BF16) | 1.39x vs Triton |
| `fused_moe` kernel (tokens=16384, W4A16) | 3.22x vs Triton |
| TPOT (mean, concurrency=40) | **-69.2%** (230ms to 71ms) |
| Output throughput | **+162.4%** (135 to 355 tok/s) |

### Configuration for FlyDSL

```bash
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1
export FLYDSL_W4A16_HYBRID=w2_bf16  # Stage1=W4A16, Stage2=BF16
export SGLANG_USE_AITER=1
```

---

## 8. Primus Training Framework

### Overview

AMD's unified, modular training framework supporting Megatron-LM and TorchTitan backends.

### Core Features

- **YAML-driven configuration**: Declarative experiment definitions
- **Multi-backend**: Megatron-LM (TP/PP/EP), TorchTitan (FP8 + fused attention)
- **Preflight validation**: Cluster connectivity, GPU diagnostics, RDMA/RCCL, bandwidth
- **Structured logging**: By experiment/module/rank/severity

### Quick Start

```bash
docker pull docker.io/rocm/megatron-lm:v25.7_py310

export EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml
bash examples/run_pretrain.sh
```

### Configuration Example

```yaml
work_group: AMD
exp_name: llama3.1_8b-pretrain
modules:
  pre_trainer:
    framework: megatron
    model: llama3.1_8B.yaml
    overrides:
      train_iters: 50
      micro_batch_size: 2
      global_batch_size: 128
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
```

### MoE Training Optimizations

#### Turbo Grouped GEMM

Uses CK (Composable Kernel) to fuse grouped GEMM into a single kernel launch for all experts, with auto-tuning for best forward/backward backends.

#### DeepEP Acceleration

- GPU-side index computation replaces CPU coordination
- Eliminates CPU-GPU synchronization for fully sync-free pipeline
- Key flags: `use_cuda_num_token_per_expert`, `num_worst_token`

#### Sync-Free MoE (4 Levels)

| Level | Description |
|---|---|
| 0 | Default (disabled) |
| 1 | Remove Router + Permutation synchronization |
| 2 | Remove Router + DeepEP + GroupMLP synchronization |
| 3 | Fully sync-free (high memory consumption) |

```bash
--turbo_sync_free_moe_stage 3
```

#### Other Techniques

- **1F1B A2A Overlap**: Communication of micro-batch N overlaps with backward compute of micro-batch N-1
- **Arbitrary Pipeline Partitioning**: Custom pipeline layout for memory/compute balance
- **Selective Recomputation**: `--recompute_layer_ids 0,1,2,3`
- **CPU Launch Optimization**:

```bash
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912  # 12MB kernel arg pool
```

### Profiling Tools for Training

| Tool | Purpose |
|---|---|
| Torch Profiler + Perfetto UI | Timeline visualization |
| TraceLens | Layer-level performance decomposition, roofline, multi-GPU diagnostics |
| Memory Projection | VRAM usage analysis |
| pp_vis | Pipeline parallelism visualization |

---

## 9. Environment Variable Quick Reference

### Inference Serving

```bash
# vLLM/SGLang core
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export VLLM_ROCM_USE_AITER=1
export HSA_NO_SCRATCH_RECLAIM=1

# Multi-GPU
export NCCL_MIN_NCHANNELS=112
export NCCL_IGNORE_CPU_AFFINITY=1

# GEMM tuning
export HIP_ONLINE_TUNING=1
export PYTORCH_TUNABLEOP_ENABLED=1

# Flash Attention
export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"

# FlyDSL
export AITER_USE_FLYDSL_MOE=1
export SGLANG_USE_AITER=1
```

### Multi-GPU Communication (RCCL)

```bash
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_FORCE_DEV_KERNARG=1

# CPX mode
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export MSCCLPP_READ_ALLRED=1
```

### Training

```bash
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_TUNABLEOP_ENABLED=1
```

### Disaggregation

```bash
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
SGLANG_DISAGGREGATION_QUEUE_SIZE=4
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300
```

### Memory Capacity Planning

| GPU | HBM | Typical Models |
|---|---|---|
| MI300X | 192GB HBM3 | DeepSeek-R1 671B (8-GPU), Llama 405B (8-GPU) |
| MI355X | 288GB HBM3E | Larger batch sizes, higher concurrency |
| H200 (reference) | 141GB | Same models at lower concurrency |
| B200 (reference) | 180GB | Competitive single-GPU, lower aggregate memory |
