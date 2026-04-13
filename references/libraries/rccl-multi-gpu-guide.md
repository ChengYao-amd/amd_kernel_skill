# Multi-GPU Communication Guide (RCCL)

This document covers **RCCL**, **Quick Reduce**, and topology configuration for multi-GPU training and inference on **ROCm**.

---

## RCCL Overview & NCCL Compatibility

**RCCL** (**ROCm Communication Collectives Library**) is the multi-**GPU** / multi-node **collective communication** library for **AMD GPUs**, analogous to **NVIDIA NCCL**. It supports **PCIe** and **xGMI** high-speed interconnects.

Most frameworks using **NCCL** **API** and environment variable naming (such as **PyTorch Distributed**) are backed by the **RCCL** backend on **ROCm**; therefore documentation and troubleshooting often reference **`NCCL_*`** variable names, which are actually interpreted by **RCCL**.

---

## Key Environment Variables

### Communication (Excerpted from P2 Crawl Table)

| Variable | Function | Default / Notes |
|----------|----------|-----------------|
| `RCCL_MSCCL_FORCE_ENABLE=1` | Force enable **MSCCL** on non-**MI300X** GPUs | Not needed when enabled by default on **MI300X** |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1` | Allow **MSCCL** in single-process/multi-thread configurations | Default **Off** |
| `RCCL_MSCCLPP_ENABLE=1` | Enable **MSCCL++** communication kernels | Default **Off** |
| `RCCL_MSCCLPP_THRESHOLD=<bytes>` | Maximum message size for **MSCCL++** activation | Crawl report example **1MB**; can be increased during tuning (e.g., on the order of **1GB**) |
| `NCCL_MIN_NCHANNELS=32` | Increase channel count for improved bandwidth when fewer than **8 GPUs** | Auto |
| `NCCL_IGNORE_CPU_AFFINITY=1` | Ignore **CPU** affinity in multi-node setups | Default **Off** |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | Related to **P2P** transfer for **PCIe**-connected **GPUs** | Default **Off** |
| `RCCL_ENABLE_CONTEXT_TRACKING=1` | Context tracking (beneficial in some scenarios) | Default **Off** |
| `HIP_FORCE_DEV_KERNARG=1` | Optimize **allreduce** and similar paths in **CPX** mode | Often used with partition tuning |
| `MSCCLPP_READ_ALLRED=1` | Optimize **read-based allreduce** under **CPX** | Used with **MSCCL++** |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1` | **PyTorch** **tensor** register allocator hook | Appears alongside **RCCL** in official tuning lists |

### MSCCL++ Usage Limitations (Crawl Summary)

- Message size must be a non-zero multiple of **32** bytes.
- Does not support **`hipMallocManaged`** buffers.
- **Allreduce** types and supported data type sets are subject to the current version documentation (crawl table lists **float16/int32/uint32/float32/bfloat16** and constraints like **sum**).

---

## Quick Reduce & Quantized All-Reduce (INT8 / INT6 / INT4)

**Quick Reduce** is an optimized path for large-tensor **all-reduce** in the ecosystem (commonly mentioned alongside the **vLLM** / **AITER** stack): beneficial for throughput at **TP 4-8** with many concurrent requests.

- Supports **FP16** / **BF16** as well as symmetric quantized **INT8** / **INT6** / **INT4** **all-reduce**.
- Quantization changes numerical characteristics; accuracy and **SLO** verification are required before production deployment.

---

## RCCL Configuration Under CPX Mode

**CPX** (**Core Partitioned X-celerator**): Each **XCD** operates as an independent logical **GPU** (on **MI300X**, commonly **8** logical cards per **OAM** view). **NPS4**: Memory is visible by quadrant to compute units, often combined with **CPX** for communication and locality optimization.

Example (partition + environment variables, from crawl reports):

```bash
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export MSCCLPP_READ_ALLRED=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### Bandwidth Comparison (Crawl Data)

| Configuration | Bus Bandwidth Order of Magnitude |
|---------------|----------------------------------|
| Default **SPX** **allreduce** (**PyTorch**, **ROCm 6.2.4**) | ~**170 GB/s** |
| Optimized **CPX** **allreduce** (single **OAM** context) | ~**315 GB/s** (**PyTorch**) / ~**340 GB/s** (**rccl-tests**) |

---

## Communication & Compute Overlap Strategies

On the **training** side (from crawl report **MoE** best practices), common approaches include:

- **1F1B A2A Overlap**: Interleave **all-to-all** communication between **micro-batches** with forward/backward computation, hiding latency.
- **Turbo Grouped GEMM**, **Sync-Free MoE**, and other multi-level options: Reduce **CPU**/**GPU** synchronization and launch gaps, making communication-computation overlap easier.
- **NUMA** and **kernel arg pool**: e.g., `ENABLE_NUMA_BINDING=1`, `HSA_KERNARG_POOL_SIZE`, etc., reducing **launch** and **CPU**-side jitter, indirectly benefiting pipeline overlap.

On the inference side, this can be combined with **Prefill/Decode separation**, **multi-stream**, and framework-level **continuous batching** to prevent **prefill** batches from blocking **decode** (see the **SGLang** **disaggregation** documentation for details).

---

## MI300X 8-GPU Fully Connected Topology (Conceptual)

Common description for single-node **MI300X**:

- **8** **XCDs**; under **CPX** these can be mapped to **8** logical **GPUs**.
- Inter-card connectivity relies on **xGMI** high-speed links; system-level tuning often emphasizes **xGMI** **link width**, **BIOS**, and **OS** settings to approach nominal bandwidth.
- **Tensor Parallelism** is generally recommended to stay within the same **xGMI** **island** (e.g., **<=8 GPUs**) to reduce cross-node or cross-slow-link **allreduce** costs.

More detailed **NIC**, **intra-chassis wiring**, and **OAM** topology vary by machine model; before deployment, it is recommended to sweep bandwidth and latency using **rccl-tests** and workload-specific **micro-benchmarks**.

---

## References

- **RCCL** official documentation and **usage tips** (ROCm Docs).
- Related docs: `optimization/inference-serving.md` (vLLM/SGLang), `optimization/case-studies.md` (DeepSeek multi-GPU), `toolchain/environment-variables.md` (RCCL env vars).
