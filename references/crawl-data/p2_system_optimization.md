# P2 ROCm System Optimization Crawl Report

> Crawled 2026-04-09 — Multi-GPU communication, quantization, training frameworks, inference optimization

---

## 1. RCCL — Multi-GPU Collective Communication

**Source:** [RCCL docs](https://rocm.docs.amd.com/projects/rccl/en/latest/) + [Usage tips](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html)

### 核心定位

RCCL (ROCm Communication Collectives Library) 是 AMD GPU 上的多 GPU / 多节点集合通信库，类似 NVIDIA NCCL。支持 PCIe 和 xGMI 高速互联。

### 关键环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `RCCL_MSCCL_FORCE_ENABLE=1` | 在非 MI300X 平台强制启用 MSCCL | Off (MI300X 默认开启) |
| `RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1` | 允许 MSCCL 在多线程/单线程配置下使用 | Off |
| `RCCL_MSCCLPP_ENABLE=1` | 启用 MSCCL++ 高效通信内核 | Off |
| `RCCL_MSCCLPP_THRESHOLD=<bytes>` | MSCCL++ 生效的消息大小上限 | 1MB |
| `NCCL_MIN_NCHANNELS=32` | 少于 8 GPU 时增加通道数提升带宽 | 自动 |
| `NCCL_IGNORE_CPU_AFFINITY=1` | 多节点时忽略 CPU 亲和性 | Off |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | PCIe 连接 GPU 的 P2P 传输 | Off |
| `RCCL_ENABLE_CONTEXT_TRACKING=1` | 启用上下文跟踪（特定场景提升性能） | Off |
| `HIP_FORCE_DEV_KERNARG=1` | CPX 模式下优化 allreduce | — |
| `MSCCLPP_READ_ALLRED=1` | 优化 CPX 模式下 read-based allreduce | — |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1` | PyTorch 中使用 tensor 注册分配器 | — |

### MSCCL++ 限制

- 消息大小必须是 32 字节的非零倍数
- 不支持 `hipMallocManaged` 缓冲区
- Allreduce 仅支持 `float16, int32, uint32, float32, bfloat16`，仅支持 sum 操作

### MI300X 性能优化 — CPX + NPS4 模式

**CPX** (Core Partitioned X-celerator): 每个 XCD 作为独立逻辑 GPU，8 XCD = 8 逻辑 GPU/OAM
**NPS4**: 每个内存象限对计算单元直接可见

```bash
# 设置分区模式
amd-smi set --gpu all --compute-partition CPX
amd-smi set --gpu all --memory-partition NPS4

# 优化环境变量组合
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export MSCCLPP_READ_ALLRED=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 性能基准

| 配置 | Bus Bandwidth |
|---|---|
| 默认 SPX allreduce (PyTorch, ROCm 6.2.4) | ~170 GB/s |
| **优化 CPX allreduce** (单 OAM) | **~315 GB/s (PyTorch) / ~340 GB/s (rccl-tests)** |

---

## 2. MIOpen — 深度学习原语库

**Source:** [MIOpen docs](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) + [Find API & Immediate Mode](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html)

### 核心功能

- AMD 开源深度学习原语库，实现算子融合以优化内存带宽和 GPU 启动开销
- 首个公开支持 `bfloat16` 卷积的库
- 自动调优基础设施覆盖卷积的大型设计空间

### Find API vs Immediate Mode

| 模式 | 特点 | 适用场景 |
|---|---|---|
| **Find API** (`miopenFindConvolution*`) | 编译+基准测试所有 solver，结果缓存到磁盘 | 首次运行，需最佳性能 |
| **Immediate Mode** (`miopenConvolution*Immediate`) | 查询 FindDb，无需 find 调用，降低运行时开销 | 生产部署，快速启动 |

### Find Mode 环境变量 (`MIOPEN_FIND_MODE`)

| 值 | 名称 | 行为 |
|---|---|---|
| `1` / `NORMAL` | 完全 find | 基准测试所有 solver |
| `2` / `FAST` | 快速 find | FindDb 命中则用，miss 用 immediate fallback |
| `3` / `HYBRID` | 混合 find | FindDb 命中则用，miss 用完整 find |
| **`5`** / `DYNAMIC_HYBRID` | **动态混合 (默认)** | FindDb 命中则用，miss 跳过非动态内核 |
| `6` / `TRUST_VERIFY` | 信任验证 | 带容差检查的自动调优 |
| `7` / `TRUST_VERIFY_FULL` | 完全信任验证 | 无时间限制的调优 |

### Immediate Mode Fallback

- **AI 启发式 fallback** (默认 `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=ON`): 用神经网络预测最优解，~90% 准确率
- **加权吞吐量索引 fallback**: 基于卷积参数估算最优解

### API 使用模式 — Immediate Mode

```c
// 1. 查询解数量
miopenConvolutionForwardGetSolutionCount(handle, ...);
// 2. 获取按性能排序的解列表
miopenConvolutionForwardGetSolution(handle, ...);
// 3. (可选) 预编译选择的解
miopenConvolutionForwardCompileSolution(handle, ...);
// 4. 执行
miopenConvolutionForwardImmediate(handle, ...);
```

---

## 3. MXFP4/MXFP6 量化 — AMD Quark

**Source:** [MXFP4/MXFP6 blog](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)

### MXFP 格式规范 (OCP Microscaling)

| 格式 | 元素类型 | 位数 | 范围 | Block Size | Scale |
|---|---|---|---|---|---|
| MXFP4 | FP4 (E2M1) | 4 | [-6, 6] | 32 | E8M0 |
| MXFP6 | FP6 (E2M3) | 6 | [-7.5, 7.5] | 32 | E8M0 |
| MXFP6 | FP6 (E3M2) | 6 | [-28.0, 28.0] | 32 | E8M0 |

### MI355X GPU 原生支持

- FP4 和 FP6 相比 FP16 可达 **4× 峰值吞吐**
- MXFP6 与 MXFP4 在 AMD MI GPU 上有**相同的计算 FLOPs**

### 量化流程

1. **Scaling**: 每个 block (32 values) 共享 E8M0 scale factor
2. **Clipping**: 裁剪到可表示范围
3. **Rounding**: 使用 **RNE (Round-to-Nearest-Even)** — 关键细节，省略 RNE 会显著降低模型精度

### AMD Quark 工具链

- 支持算法: GPTQ, SmoothQuant, Quarot, **AutoSmoothQuant** (自适应逐层平滑)
- 输出兼容: vLLM, SGLang
- 混合精度: MXFP4-MXFP6 联合使用

### 精度结果 (关键模型)

| 模型 | MXFP4 精度保持 | 最佳配置 |
|---|---|---|
| DeepSeek-R1-0528 | >99.5% (AIME24, GPQA, MATH-500) | MXFP6 > mixed > MXFP4 |
| Llama-3.1-405B | 强性能 | MXFP6/mixed 更优 |
| Llama-3.3-70B | 有可见退化 | MXFP6/mixed 显著缓解 |

### 已发布量化模型

- [amd/DeepSeek-R1-0528-MXFP4-ASQ](https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4-ASQ)
- [amd/Llama-3.1-405B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.1-405B-Instruct-MXFP4-Preview)
- [amd/Llama-3.3-70B-Instruct-MXFP4-Preview](https://huggingface.co/amd/Llama-3.3-70B-Instruct-MXFP4-Preview)

---

## 4. FP8 GEMM 优化 — CDNA4 架构

**Source:** [FP8 GEMM blog](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)

### CDNA4 vs CDNA3 关键差异

| 特性 | CDNA4 | CDNA3 |
|---|---|---|
| LDS 容量 | **160 KB** | 64 KB |
| LDS bank 数 | **64** | 32 |
| LDS 读带宽 | **256 B/clock** | 128 B/clock |
| GLOBAL_LOAD_LDS 每线程传输 | **128 bits** | 32 bits |
| FP4/FP6 MFMA | **支持** | 不支持 |
| Block-scaled MFMA | **支持** | 不支持 |

### 优化阶段与性能 (M=N=K=4096)

| 阶段 | TFLOPS/s | 加速比 |
|---|---|---|
| Naive 实现 | 1.15 | 1× |
| LDS tiling | 4.80 | 4.2× |
| Matrix-core (MFMA) | 30.05 | 26× |
| + 向量化加载 | 336.88 | 293× |
| + Direct global-to-LDS | 506.70 | 441× |
| + LDS swizzle + 双缓冲 | 1,166.41 | 1,014× |
| + 256×256 multi-wave | 2,288.16 | 1,990× |
| + **8-wave ping-pong** | **~2,750** | ~2,391× |
| hipBLASLt 参考 (4096) | ~2,750 | — |
| hipBLASLt 参考 (8192) | ~3,130 | — |

### 关键优化技术

1. **MFMA 指令** — 16x16x128 FP8→FP32，单条指令 65,536 FLOPs (vs FMA 128 FLOPs)
2. **LDS Swizzling** — XOR remap 消除 bank conflicts
3. **Double Buffering** — Ping-pong LDS slots 隐藏 load 延迟
4. **8-wave Ping-Pong** — LLVM intrinsics (`s_barrier`, `s_setprio`, `sched_barrier`) 精细控制调度

---

## 5. DeepSeek-R1 推理优化 — SGLang + AITER

**Source:** [Part1](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html), [Part2](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

### AITER (AI Tensor Engine for ROCm) 内核加速

| 内核 | 加速倍数 |
|---|---|
| Block-scale GEMM | **2×** |
| Block-scale fused MoE | **3×** |
| MLA decode | **17×** |
| MHA prefill | **14×** |

### MI300X vs H200 性能 (DeepSeek-R1 671B FP8)

| 指标 | MI300X (8 GPU) | vs H200 |
|---|---|---|
| 吞吐量 (相同延迟) | — | **2×–5× 更高** |
| 吞吐量 (相同并发) | — | **75% 更高** |
| 延迟 (相同并发) | — | **60% 更低** |
| 50ms ITL 下最大并发 | **128 请求** | H200: 16 请求 |

### 关键 SGLang 参数

```bash
python3 -m sglang.launch_server \
  --model /model \
  --tp 8 \
  --trust-remote-code \
  --chunked-prefill-size 131072 \   # 大 chunk 加速 prefill
  --enable-torch-compile \           # 编译优化
  --torch-compile-max-bs 256

# 环境变量
export HSA_NO_SCRATCH_RECLAIM=1
```

### Docker 复现

```bash
docker pull rocm/sgl-dev:upstream_20250312_v1
docker run -it --ipc=host --network=host --privileged --shm-size 32G \
  --device=/dev/kfd --device=/dev/dri \
  -v $MODEL_DIR:/model \
  rocm/sgl-dev:upstream_20250312_v1
```

---

## 6. SGLang 推理框架

**Source:** [SGLang overview](https://rocm.blogs.amd.com/artificial-intelligence/sglang/README.html)

### 核心特性

- **RadixAttention**: 基于 radix tree 自动复用 KV cache
- **Jump-Forward Constrained Decoding**: 跳过不必要计算
- **Continuous Batching**: 动态调整 batch size
- **Paged Attention**: 分区注意力矩阵，支持长序列

### Multi-GPU 部署

```bash
# Tensor Parallelism (2 GPU)
python -m sglang.launch_server --model-path <model> --tp 2

# Data Parallelism (2 DP × 2 TP = 4 GPU)
python -m sglang.launch_server --model-path <model> --dp 2 --tp 2
```

### 量化选项

```bash
# FP8 weight 量化
--quantization fp8

# FP8 KV cache 量化
--kv-cache-dtype fp8_e5m2

# AMD Quark FP8 模型
--model-path amd/Meta-Llama-3.1-405B-Instruct-FP8-KV --tp 8 --quant fp8
```

---

## 7. Prefill-Decode Disaggregation

**Source:** [Disaggregation blog](https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html)

### 核心思想

将 LLM 推理的 prefill (计算密集) 和 decode (内存密集) 阶段分离到不同 GPU 上。

### SGLang 服务器参数

| 参数 | 描述 | 默认 |
|---|---|---|
| `--disaggregation-mode` | `prefill` 或 `decode` | null |
| `--disaggregation-transfer-backend` | KV 传输后端 | mooncake |
| `--disaggregation-ib-device` | InfiniBand 设备 | 自动检测 |
| `--disaggregation-bootstrap-port` | Bootstrap 端口 | 8998 |

### 环境变量 (细粒度控制)

| 变量 | 描述 | 默认 |
|---|---|---|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | KV 传输线程数/TP rank | `int(0.75 * cpu_count()) // 8)` (4–12) |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | 并行传输队列数 | 4 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | KV 索引接收超时 (秒) | 300 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | 心跳检查间隔 (秒) | 5.0 |

### 性能收益 (Llama 3.3 70B, MI300X, 128 并发)

| 场景 | P95 ITL SLO | P99 ITL SLO | Goodput 提升 |
|---|---|---|---|
| Chatbot (3200in/800out) | **7.1× 更严格** | **13.2× 更严格** | **6.9×** (TTFT≤1s, TPOT≤25ms) |
| Heavy decode (1024in/2048out) | **1.3× 更严格** | **6.1× 更严格** | **2.23×** (TTFT≤500ms, TPOT≤25ms) |

---

## 8. Primus — 统一训练框架

**Source:** [Primus blog](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)

### 定位

Primus 是 AMD 的统一、模块化训练框架，支持 Megatron-LM 和 TorchTitan 后端。

### 核心特性

- **YAML 驱动配置**: 声明式实验定义，消除重复性脚本
- **多后端支持**: Megatron-LM (TP/PP/EP)，未来 TorchTitan (FP8 + fused attention)
- **Preflight 验证**: 集群连通性、GPU 诊断、RDMA/RCCL 通信、网络带宽基准
- **结构化日志**: 按 experiment/module/rank/severity 层级组织

### 快速启动

```bash
# Docker
docker pull docker.io/rocm/megatron-lm:v25.7_py310

# YAML 配置
export EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml
bash examples/run_pretrain.sh
```

### 配置示例

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

---

## 9. MoE 训练最佳实践

**Source:** [MoE Training blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)

### 代表性模型规模

| 模型 | 总参数 | 激活参数 |
|---|---|---|
| DeepSeek-V2-Lite | 16B | 2.4B |
| DeepSeek-V2 | 236B | 21B |
| MoE-1T | 1T | 44B |
| MoE-2T | 2T | 80B |

### 性能瓶颈分析

1. **Grouped GEMM 开销** — 多流方案仍有调度间隙
2. **All-to-all 通信** — EP ≥ 8 时跨节点开销大
3. **CPU 同步延迟** — D2H 同步阻塞内核启动队列
4. **小内核过多** — MoE 层细粒度算子多，CPU launch 压力大
5. **Pipeline 负载不均** — 不均匀工作分配拖慢整体吞吐

### 关键优化技术

#### (1) Turbo Grouped GEMM
- 使用 CK (Composable Kernel) 融合 grouped GEMM，单次内核启动处理所有 experts
- 自动调优选择前向/后向最快后端

#### (2) DeepEP 加速
- GPU 端索引计算替代 CPU 协调
- 消除 CPU-GPU 同步，实现完全 sync-free pipeline
- 关键环境变量: `use_cuda_num_token_per_expert`, `num_worst_token`

#### (3) Sync-Free MoE (4 级)

| 级别 | 描述 |
|---|---|
| 0 | 默认 (禁用) |
| 1 | 移除 Router + Permutation 同步 |
| 2 | 移除 Router + DeepEP + GroupMLP 同步 |
| 3 | 完全 sync-free (⚠ 内存消耗大) |

```bash
--turbo_sync_free_moe_stage 3
```

#### (4) 1F1B A2A Overlap
交错通信和计算：micro-batch N 的通信与 micro-batch N-1 的后向计算重叠

#### (5) 任意 Pipeline 分区
自定义 pipeline 布局，优化内存和计算平衡

#### (6) 选择性重计算
```bash
--recompute_layer_ids 0,1,2,3
```

#### (7) CPU Launch 优化

```bash
export ENABLE_NUMA_BINDING=1        # NUMA 绑定
export HSA_KERNARG_POOL_SIZE=12582912  # 12MB kernel arg pool
```

### 分析工具链

1. **Torch Profiler** → [Perfetto UI](https://ui.perfetto.dev/) 可视化
2. **TraceLens** — 层级性能分解、Roofline 分析、多 GPU 通信诊断
3. **Memory Projection** — VRAM 使用分析
4. **pp_vis** — Pipeline 并行可视化

---

## 10. 模型加速库 — Flash Attention, TunableOp, FBGEMM

**Source:** [Model Acceleration Libraries](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html)

### Flash Attention 2

两个后端切换:
```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"  # CK 后端 (默认)
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"   # Triton 后端
```

使用示例:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).cuda()
```

### PyTorch TunableOp

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
```
- 自动从 rocBLAS / hipBLASLt 选择最佳 GEMM 内核
- 生成 GEMM Table 在后续运行中复用

### FBGEMM_GPU

- 高性能 embedding 操作、数据布局变换、量化支持
- UVM 测试需要 `HSA_XNACK=1`
- 构建目标架构: `PYTORCH_ROCM_ARCH=gfx942` (MI300 系列)

### xFormers (CK 后端)

```bash
PYTORCH_ROCM_ARCH=gfx942 python setup.py install
```

---

## 11. ROCm 7.12.0 Preview — 关键更新

**Source:** [Release Notes](https://rocm.docs.amd.com/en/7.12.0-preview/about/release-notes.html)

### 硬件支持扩展

| 系列 | LLVM Target | 架构 |
|---|---|---|
| MI355X / MI350X | gfx950 | CDNA 4 |
| MI325X / MI300X / MI300A | gfx942 | CDNA 3 |
| MI250X / MI250 / MI210 | gfx90a | CDNA 2 |
| MI100 (新增) | gfx908 | CDNA |
| Radeon RX 9070 系列 | gfx1201 | RDNA 4 |

### GPU 分区支持

| 设备 | 计算分区 | NPS 模式 |
|---|---|---|
| MI355X / MI350X | CPX | NPS 2 |
| MI300X | CPX | NPS 4 |

### AI 生态系统

- **PyTorch 2.10.0** — Linux + Windows
- **JAX 0.8.2** — 通过 TheRock 构建发布
- **vLLM 0.16.0** — gfx950, gfx942, gfx1200, gfx1201, gfx1151

### 性能分析工具增强

- **ROCm Optiq (Beta)**: Compute Profiler 数据可视化，roofline 分析
- **Iteration Multiplexing**: 单次运行收集完整硬件计数器集
- **Torch 算子级分析**: 实验性 PyTorch 算子级 counter 收集
- **进程附加 Profiling**: `rocprof-sys-attach` 附加到运行中的进程
- **Pensando AI NIC 网络指标**: CNP 和带宽利用率

---

## 环境变量速查表

### Multi-GPU / Communication

```bash
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCLPP_THRESHOLD=1073741824
export NCCL_MIN_NCHANNELS=32
export NCCL_IGNORE_CPU_AFFINITY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_FORCE_DEV_KERNARG=1
```

### Training Optimization

```bash
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_TUNABLEOP_ENABLED=1
export HSA_NO_SCRATCH_RECLAIM=1
```

### Inference / Serving

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"
export HSA_XNACK=1  # UVM support
```

### MIOpen

```bash
export MIOPEN_FIND_MODE=DYNAMIC_HYBRID  # 默认
export MIOPEN_FIND_MODE=NORMAL          # 完整搜索
```

---

## 数据来源

| # | URL | 状态 |
|---|---|---|
| 1 | RCCL index | ✅ |
| 2 | RCCL usage tips | ✅ |
| 3 | MIOpen index | ✅ |
| 4 | MIOpen find-and-immediate | ✅ |
| 5 | FP8 GEMM on CDNA4 (替代 URL) | ✅ |
| 6 | MXFP4/MXFP6 quantization (替代 URL) | ✅ |
| 7 | DeepSeek-R1 Part 1 + Part 2 (替代 URL) | ✅ |
| 8 | SGLang overview + disaggregation (替代 URL) | ✅ |
| 9 | ROCm 7.12.0 release notes | ✅ |
| 10 | Model acceleration libraries | ✅ |
| 11 | MoE training best practices (替代 URL) | ✅ |
| 12 | Primus framework (替代 URL) | ✅ |
