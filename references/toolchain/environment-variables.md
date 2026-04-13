# ROCm / HIP / Library Environment Variables Reference

> Comprehensive reference for all environment variables relevant to AMD GPU kernel development, inference optimization, and system tuning. Organized by category. For system-level tuning context, see `system-tuning.md`.

---

## 1. GPU Device Isolation

| Variable | Values | Description |
|----------|--------|-------------|
| `ROCR_VISIBLE_DEVICES` | Device indices or UUIDs | Primary device filter on Linux (recommended). Indices are physical GPU order. |
| `HIP_VISIBLE_DEVICES` | Device indices | HIP-level device filter. Indices are post-`ROCR_VISIBLE_DEVICES` reordering. |
| `CUDA_VISIBLE_DEVICES` | Device indices | Alias for `HIP_VISIBLE_DEVICES` (CUDA compatibility). |
| `GPU_DEVICE_ORDINAL` | Device indices | OpenCL + HIP device filter (legacy). |

**Priority:** `ROCR_VISIBLE_DEVICES` filters at the driver level (lowest); `HIP_VISIBLE_DEVICES` filters at the runtime level on top of that.

---

## 2. HIP Runtime -- Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_FORCE_DEV_KERNARG` | 1 (ROCm 6.2+) | Store kernel arguments in device memory. Saves 2-3 us per kernel launch. |
| `GPU_MAX_HW_QUEUES` | 4 | Maximum hardware dispatch queues per device per process. Increase for concurrent kernel workloads. |
| `HIP_LAUNCH_BLOCKING` | 0 | Set to 1 to serialize all kernel launches (debugging only). |
| `AMD_SERIALIZE_KERNEL` | 0 | Wait before/after kernel enqueue: 1=before, 2=after, 3=both. Debugging only. |

---

## 3. HIP Runtime -- Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_INITIAL_DM_SIZE` | 8388608 (8 MB) | Initial device malloc heap size. |
| `HIP_MEM_POOL_SUPPORT` | 0 | Enable HIP memory pool for faster alloc/free cycles. |
| `GPU_SINGLE_ALLOC_PERCENT` | 100 | Maximum single allocation as percentage of GPU memory. |
| `GPU_MAX_HEAP_SIZE` | 100 | Maximum GPU heap as percentage of board memory. |
| `HIP_HOST_COHERENT` | 0 | Enable host-GPU memory coherence. |

---

## 4. HSA Runtime (ROCr)

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_ENABLE_SDMA` | 1 | Enable System DMA engines for host-device copies. Set to 0 to force shader-based copies. |
| `HSA_NO_SCRATCH_RECLAIM` | 0 | Set to 1 for permanent scratch memory assignment to queues. Avoids re-allocation overhead. |
| `HSA_SCRATCH_SINGLE_LIMIT` | ~140 MB | Maximum scratch memory per XCC. Increase for high-register kernels. |
| `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` | 3 GB | Async scratch memory threshold per XCC. |
| `HSA_XNACK` | (unset) | Set to 1 to enable XNACK (page fault retry). Required for UVM and some FBGEMM tests. |
| `HSA_DISABLE_CACHE` | 0 | Set to 1 to disable L2 cache (sets MTYPE=UC). Debugging/testing only. |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 0 | Set to 1 for fine-grained PCIe P2P transfers between GPUs. |
| `HSA_OVERRIDE_CPU_AFFINITY_DEBUG` | (unset) | Set to 0 to inherit parent process CPU affinity instead of ROCm runtime's own placement. |
| `HSA_KERNARG_POOL_SIZE` | (default) | Kernel argument pool size in bytes. Set to 12582912 (12 MB) for MoE workloads with many concurrent kernels. |
| `HSA_CU_MASK` | (all CUs) | Low-level CU mask for driver queues. Format: `GPU_list:CU_list`. |

### HSA_CU_MASK Syntax

```bash
# Enable CUs 0-15 and 32-47 on GPUs 0,2,3,4,7
HSA_CU_MASK=0,2-4,7:0-15,32-47

# Hex mask: enable only first 4 CUs
HSA_CU_MASK=0:0xf
```

Constraints:
- GPU indices are post-`ROCR_VISIBLE_DEVICES` reordering.
- Cannot disable a single CU in a WGP pair (for WGP-mode kernels).
- Mask with 0 usable CUs results in a syntax error.

---

## 5. CU Masking (HIP Runtime Level)

| Variable | Values | Description |
|----------|--------|-------------|
| `ROC_GLOBAL_CU_MASK` | Hex mask | HIP/OpenCL runtime-level CU mask for all queues. E.g., `0xf` enables only 4 CUs. |

Use cases: performance isolation for multi-tenant, controlled profiling.

---

## 6. Debug and Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `AMD_LOG_LEVEL` | 0 | 0=off, 1=error, 2=warning, 3=info, 4=debug |
| `AMD_LOG_MASK` | 0x7FFFFFFF | Bitmask for log categories |
| `HIPCC_VERBOSE` | (unset) | 1=show clang commands, 2=show env, 4=show args, 7=all |
| `HIP_TRACE_API` | 0 | Trace HIP API calls |

---

## 7. Tensile / hipBLASLt GEMM Tuning

### 7.1 Stream-K Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSILE_SOLUTION_SELECTION_METHOD` | 0 | 0=standard tuned (default), 2=Stream-K (Origami). On MI350 series, Stream-K is the ONLY strategy (env var has no effect). |
| `TENSILE_STREAMK_DYNAMIC_GRID` | 6 | Grid size selection: 6=auto (default), 0=use all CUs. |
| `TENSILE_STREAMK_FIXED_GRID` | (unset) | Fix to exactly N workgroups. Use 64 to prevent GEMM from monopolizing GPU during concurrent kernels. |
| `TENSILE_STREAMK_MAX_CUS` | (unset) | Cap maximum CUs available for Stream-K kernels. |

### 7.2 hipBLASLt Tuning and Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `HIPBLASLT_LOG_MASK` | 0 | Set to 32 to log GEMM shapes for offline tuning. |
| `HIPBLASLT_LOG_FILE` | (stderr) | File path for hipBLASLt log output. |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | (unset) | Path to offline-tuned GEMM solution file. |
| `HIP_ONLINE_TUNING` | 0 | Set to 1 for runtime GEMM algorithm benchmarking and caching. Performance: ~106% of baseline, ~101% of offline tuning. One-time overhead: ~31s for new shapes. |

### 7.3 rocBLAS Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ROCBLAS_LAYER` | 0 | Set to 4 to log GEMM calls in YAML format. |
| `ROCBLAS_LOG_PATH` | (unset) | Output file for rocBLAS logging. |
| `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` | (unset) | Path to tuned GEMM solution CSV. |

### 7.4 GEMM Tuning Workflow

```bash
# Level 1: PyTorch TunableOp
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 python train.py   # Tuning pass
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 python train.py   # Use results

# Level 2: rocBLAS offline tuning
ROCBLAS_LAYER=4 ROCBLAS_LOG_PATH=./gemms.yaml ./app
/opt/rocm/bin/rocblas-gemm-tune --yaml gemms.yaml
export ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=result.csv

# Level 3: hipBLASLt offline tuning
HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=log.log ./app
/opt/rocm/bin/hipblaslt-bench --api_method c -m M -n N -k K ...
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
```

---

## 8. PyTorch / TorchInductor

### 8.1 TunableOp

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | Enable TunableOp (tests GEMM algorithms from rocBLAS and hipBLASLt). |
| `PYTORCH_TUNABLEOP_TUNING` | 1 | Set to 0 to use previously tuned results without re-tuning. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | 0 | Set to 1 to prefer hipBLASLt over rocBLAS for BLAS operations. |

### 8.2 TorchInductor

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCHINDUCTOR_MAX_AUTOTUNE` | 0 | Set to 1 to enable extensive autotuning for GEMM/conv (benchmarks Triton configs by tile size, stages, warps, MFMA instruction size). |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | (default) | Comma-separated backends. Append `CK` for Composable Kernel. E.g., `TRITON,ATEN,CK`. |
| `TORCHINDUCTOR_FREEZING` | 0 | Set to 1 for constant folding (inline weights as constants). Good for inference. |
| `TORCHINDUCTOR_CPP_WRAPPER` | 0 | Set to 1 for C++ wrapper to reduce kernel launch overhead. |
| `TORCHINDUCTOR_LAYOUT_OPTIMIZATION` | 0 | Set to 1 to force channels_last layout for convolutions. |
| `PYTORCH_MIOPEN_SUGGEST_NHWC` | 0 | Set to 1 for MIOpen channels_last hint. |
| `TORCH_COMPILE_DEBUG` | 0 | Set to 1 to dump generated Triton kernels to `torch_compile_debug/` directory. |
| `PYTORCH_ROCM_ARCH` | (auto) | Override target architecture for build. E.g., `gfx942` for MI300 series. |

---

## 9. MIOpen (Convolution)

| Variable | Default | Description |
|----------|---------|-------------|
| `MIOPEN_FIND_MODE` | `DYNAMIC_HYBRID` (5) | Convolution solver search strategy. |
| `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` | ON | Use neural network heuristic for solver prediction (~90% accuracy). |
| `MIOPEN_ENABLE_LOGGING` | 0 | Enable MIOpen logging. |

### MIOPEN_FIND_MODE Values

| Value | Name | Behavior |
|-------|------|----------|
| 1 | `NORMAL` | Full benchmark of all solvers |
| 2 | `FAST` | Use FindDb cache, fallback to immediate mode on miss |
| 3 | `HYBRID` | Use FindDb cache, fallback to full find on miss |
| 5 | `DYNAMIC_HYBRID` | **Default.** Use FindDb cache, skip non-dynamic kernels on miss |
| 6 | `TRUST_VERIFY` | Auto-tuning with tolerance verification |
| 7 | `TRUST_VERIFY_FULL` | Full auto-tuning without time limits |

---

## 10. RCCL (Multi-GPU Communication)

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_MIN_NCHANNELS` | (auto) | Minimum communication channels. Set to 32 for <8 GPUs, 112 for MI300X. |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | Set to 1 for multi-node to prevent CPU affinity conflicts. |
| `RCCL_MSCCLPP_ENABLE` | 0 | Enable MSCCL++ high-performance communication. |
| `RCCL_MSCCLPP_THRESHOLD` | 1048576 (1 MB) | Max message size for MSCCL++. |
| `RCCL_MSCCL_FORCE_ENABLE` | 0 | Force MSCCL on non-MI300X platforms. |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS` | 0 | Allow MSCCL in single-process multi-thread config. |
| `RCCL_ENABLE_CONTEXT_TRACKING` | 0 | Enable context tracking for specific performance scenarios. |
| `HIP_FORCE_DEV_KERNARG` | 1 | Also improves RCCL kernel launch latency. |
| `MSCCLPP_READ_ALLRED` | 0 | Set to 1 for read-based allreduce optimization in CPX mode. |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK` | 0 | Set to 1 for PyTorch tensor register allocator hook (CPX mode). |

---

## 11. Triton Compiler

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_PRINT_AUTOTUNING` | 0 | Set to 1 to dump autotune results. |
| `TRITON_DEBUG` | 0 | Set to 1 for debug output. |
| `TRITON_CACHE_DIR` | (default) | Fix cache directory for inspecting generated LLVM/ISA artifacts. |
| `TRITON_INTERPRET` | 0 | Set to 1 for interpreter mode (slow, for debugging). |
| `MLIR_ENABLE_DUMP` | 0 | Set to 1 to dump MLIR intermediate representations at each stage. |

### Triton LLVM-IR Attributes (set via kernel metadata, not env vars)

| Attribute | Description |
|-----------|-------------|
| `amdgpu-flat-work-group-size` | Workgroup size constraints for the compiler |
| `amdgpu-waves-per-eu` | Target occupancy (waves per execution unit) |
| `denormal-fp-math-f32` | Denormal handling mode for FP32 |

---

## 12. Flash Attention Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `FALSE` | Set to `TRUE` for Triton backend, `FALSE` for CK backend (default). |

---

## 13. AITER / vLLM / SGLang (Inference Serving)

### 13.1 AITER Master Switches

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ROCM_USE_AITER` | 0 | Master switch for AITER in vLLM. |
| `VLLM_ROCM_USE_AITER_LINEAR` | (auto) | AITER linear (GEMM) kernels. |
| `VLLM_ROCM_USE_AITER_MOE` | (auto) | AITER fused MoE kernels. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | (auto) | AITER RMSNorm kernel. |
| `VLLM_ROCM_USE_AITER_MLA` | (auto) | AITER Multi-Latent Attention decode. |
| `VLLM_ROCM_USE_AITER_MHA` | (auto) | AITER Multi-Head Attention prefill. |
| `VLLM_ROCM_USE_AITER_FP8BMM` | (auto) | AITER FP8 block matrix multiply. |
| `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING` | 0 | Enable hipBLASLt online tuning via AITER. |

### 13.2 vLLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_AITER_MOE` | 0 | Enable AITER fused MoE (alternative flag). |
| `VLLM_USE_AITER_BLOCK_GEMM` | 0 | Enable AITER block-scale GEMM. |
| `VLLM_USE_AITER_MLA` | 0 | Enable AITER MLA decode (alternative flag). |
| `VLLM_FP8_PADDING` | 0 | Enable FP8 padding for alignment. |
| `VLLM_USE_TRITON_FLASH_ATTN` | 1 | Set to 0 to disable Triton Flash Attention (use CK). |
| `VLLM_USE_ROCM_FP8_FLASH_ATTN` | 1 | Set to 0 to disable ROCm FP8 Flash Attention. |

### 13.3 SGLang Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CK_BLOCK_GEMM` | 0 | Enable CK block GEMM in SGLang. |
| `SGLANG_ROCM_AITER_BLOCK_MOE` | 0 | Enable AITER MoE in SGLang. |
| `SGLANG_USE_AITER` | 0 | Enable AITER operators in SGLang. |
| `AITER_MOE` | 0 | Enable AITER MoE path. |

### 13.4 SGLang Disaggregation (Prefill-Decode Separation)

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | auto | KV transfer threads per TP rank. Default: `int(0.75 * cpu_count()) // 8)` (4-12). |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | 4 | Parallel transfer queue count. |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 300 | KV index receive timeout (seconds). |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | 5.0 | Heartbeat check interval (seconds). |

### 13.5 Example Launch Commands

```bash
# vLLM with AITER for DeepSeek
VLLM_USE_AITER_MOE=1 VLLM_USE_AITER_BLOCK_GEMM=1 \
VLLM_FP8_PADDING=1 VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve "deepseek-ai/DeepSeek-V3" \
  --tensor-parallel-size 8 --trust-remote-code

# SGLang with AITER for DeepSeek
CK_BLOCK_GEMM=1 SGLANG_ROCM_AITER_BLOCK_MOE=1 \
python3 -m sglang.launch_server --model "deepseek-ai/DeepSeek-V3" --tp 8

# SGLang with advanced options
HSA_NO_SCRATCH_RECLAIM=1 \
python3 -m sglang.launch_server \
  --model /model \
  --tp 8 \
  --trust-remote-code \
  --chunked-prefill-size 131072 \
  --enable-torch-compile \
  --torch-compile-max-bs 256
```

---

## 14. FBGEMM_GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_XNACK` | (unset) | Set to 1 for UVM support (required for FBGEMM UVM tests). |
| `PYTORCH_ROCM_ARCH` | (auto) | Target architecture for FBGEMM build. E.g., `gfx942`. |

---

## 15. Profiling Tools

| Variable | Default | Description |
|----------|---------|-------------|
| `ROCBLAS_LAYER` | 0 | Set to 4 to log rocBLAS GEMM calls in YAML format. |
| `HIPBLASLT_LOG_MASK` | 0 | Set to 32 to log hipBLASLt GEMM shapes. |
| `HIPBLASLT_LOG_FILE` | (stderr) | Output file for hipBLASLt logs. |

---

## 16. MoE Training Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_NUMA_BINDING` | 0 | Set to 1 for NUMA-aware thread binding during training. |
| `HSA_KERNARG_POOL_SIZE` | (default) | Kernel argument pool size. Set to 12582912 (12 MB) for MoE workloads. |

---

## 17. Quick-Copy Recipes

### Inference Serving (MI300X, vLLM + AITER)

```bash
export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NCCL_MIN_NCHANNELS=112
export VLLM_ROCM_USE_AITER=1
export VLLM_FP8_PADDING=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export HSA_NO_SCRATCH_RECLAIM=1
```

### Training (MI300X, Multi-GPU)

```bash
export HIP_FORCE_DEV_KERNARG=1
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_TUNABLEOP_ENABLED=1
export NCCL_MIN_NCHANNELS=112
export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
```

### Profiling Session

```bash
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_TUNABLEOP_ENABLED=0
# Set deterministic clock for reproducible measurements
rocm-smi --setperfdeterminism 1900
```

### CPX Mode (Small Model Multi-Tenant)

```bash
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export MSCCLPP_READ_ALLRED=1
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
