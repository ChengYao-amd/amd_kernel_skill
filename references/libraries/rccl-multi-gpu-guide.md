# 多 GPU 通信指南（RCCL）

本文以 `references/crawl-data/p2_system_optimization.md` 为主，并补充同目录下其它爬取材料中与 **RCCL**、**Quick Reduce**、拓扑相关的条目，便于在 **ROCm** 上配置多卡训练与推理。

---

## RCCL 概述与 NCCL 兼容性

**RCCL**（**ROCm Communication Collectives Library**）是 **AMD GPU** 上的多 **GPU** / 多节点**集合通信**库，角色接近 **NVIDIA NCCL**。支持 **PCIe** 与 **xGMI** 等高速互联。

多数基于 **NCCL** **API** 与环境变量命名的框架（如 **PyTorch Distributed**）在 **ROCm** 上由 **RCCL** 后端承载；因此文档与排障常出现 **`NCCL_*`** 变量名，实际由 **RCCL** 解释。

---

## 关键环境变量

### 通信用（摘自 P2 爬取表）

| 变量 | 作用 | 默认 / 备注 |
|------|------|-------------|
| `RCCL_MSCCL_FORCE_ENABLE=1` | 在非 **MI300X** 上强制 **MSCCL** | **MI300X** 上默认开启时可不设 |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1` | 允许 **MSCCL** 在单进程/多线程配置下使用 | 默认 **Off** |
| `RCCL_MSCCLPP_ENABLE=1` | 启用 **MSCCL++** 通信内核 | 默认 **Off** |
| `RCCL_MSCCLPP_THRESHOLD=<bytes>` | **MSCCL++** 生效的消息大小上限 | 爬取报告中示例 **1MB**；调优时可增大（如 **1GB** 量级） |
| `NCCL_MIN_NCHANNELS=32` | 少于 **8 GPU** 时增加通道数以提升带宽 | 自动 |
| `NCCL_IGNORE_CPU_AFFINITY=1` | 多节点时忽略 **CPU** 亲和 | 默认 **Off** |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | **PCIe** 连接 **GPU** 的 **P2P** 传输相关 | 默认 **Off** |
| `RCCL_ENABLE_CONTEXT_TRACKING=1` | 上下文跟踪（部分场景有益） | 默认 **Off** |
| `HIP_FORCE_DEV_KERNARG=1` | **CPX** 模式下优化 **allreduce** 等路径 | 常与分区调优联用 |
| `MSCCLPP_READ_ALLRED=1` | **CPX** 下优化 **read-based allreduce** | 与 **MSCCL++** 联用 |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1` | **PyTorch** 中 **tensor** 注册分配器钩子 | 官方调优列表中与 **RCCL** 同现 |

### MSCCL++ 使用限制（爬取摘要）

- 消息大小须为 **32** 字节的非零倍数。
- 不支持 **`hipMallocManaged`** 缓冲区。
- **Allreduce** 类型与支持的数据类型集合以当前版本文档为准（爬取表中列举 **float16/int32/uint32/float32/bfloat16** 及 **sum** 等约束）。

---

## Quick Reduce 与量化 all-reduce（INT8 / INT6 / INT4）

**Quick Reduce** 是生态中面向大张量 **all-reduce** 的优化路径（常与 **vLLM** / **AITER** 栈并提）：在 **TP 4–8** 且并发请求较多时有助于吞吐。

- 支持 **FP16** / **BF16** 以及对称量化 **INT8** / **INT6** / **INT4** **all-reduce**。
- 量化会改变数值特性，上线前需做精度与 **SLO** 验证。

---

## CPX 模式下的 RCCL 配置

**CPX**（**Core Partitioned X-celerator**）：每个 **XCD** 作为独立逻辑 **GPU**（**MI300X** 上常见 **8** 逻辑卡 / **OAM** 视图）。**NPS4**：内存按象限对计算单元可见，常与 **CPX** 组合做通信与局部性优化。

示例（分区 + 环境变量，摘自爬取报告）：

```bash
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export MSCCLPP_READ_ALLRED=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 带宽对比（爬取数据）

| 配置 | 总线带宽量级 |
|------|----------------|
| 默认 **SPX** **allreduce**（**PyTorch**，**ROCm 6.2.4**） | ~**170 GB/s** |
| 优化 **CPX** **allreduce**（单 **OAM** 语境） | ~**315 GB/s**（**PyTorch**）/ ~**340 GB/s**（**rccl-tests**） |

---

## 通信与计算重叠策略

在**训练**侧（爬取报告中的 **MoE** 最佳实践），常见思路包括：

- **1F1B A2A Overlap**：将 **micro-batch** 间 **all-to-all** 通信与前后向计算交错，隐藏延迟。
- **Turbo Grouped GEMM**、**Sync-Free MoE** 等多级选项：减少 **CPU**/**GPU** 同步与启动间隙，使通信与计算更易重叠。
- **NUMA** 与 **kernel arg pool**：如 `ENABLE_NUMA_BINDING=1`、`HSA_KERNARG_POOL_SIZE` 等，降低 **launch** 与 **CPU** 侧抖动，间接利于流水线重叠。

推理侧可配合 **Prefill/Decode 分离**、**多流** 与框架级 **continuous batching**，避免 **prefill** 批量打断 **decode**（详见 **SGLang** **disaggregation** 文档）。

---

## MI300X 8-GPU 全连接拓扑（概念）

单节点 **MI300X** 常见描述：

- **8** 个 **XCD**；在 **CPX** 下可映射为 **8** 张逻辑 **GPU**。
- 卡间依赖 **xGMI** 高速链路；系统级调优中常强调 **xGMI** **link width**、**BIOS** 与 **OS** 设置以逼近标称带宽。
- **Tensor Parallelism** 通常建议落在同一 **xGMI** **island** 内（例如 **≤8 GPU**），以降低跨机或跨慢链路的 **allreduce** 代价。

更细的 **NIC**、**机内布线** 与 **OAM** 拓扑因机型而异，部署前建议用 **rccl-tests** 与业务 **micro-benchmark** 扫带宽与延迟。

---

## 参考

- **RCCL** 官方文档与 **usage tips**（ROCm Docs）。
- 爬取报告：`references/crawl-data/p2_system_optimization.md`；**Quick Reduce** 与 **vLLM** 条目另见 `references/crawl-data/p3_deep_dive_cases.md`、`references/crawl-data/rocm_crawl_report_round1.md`。
