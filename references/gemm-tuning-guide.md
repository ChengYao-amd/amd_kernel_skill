# GEMM 调优全景指南

本文综合 `crawl-data/p1_gemm_attention_tuning.md` 与 `crawl-data/p2_system_optimization.md` 中的官方与博文信息，给出 **AMD Instinct** 上 **GEMM** 的分层调优路径、库级 **API**、**Stream-K**、**Online Tuning**、**Tensile** 参数与 **Attention** 实现选型及公开性能数据。

---

## 三级 GEMM 调优策略（自顶向下）

| 层级 | 手段 | 说明 |
|------|------|------|
| **1. Pre-tuned Docker / 预置配置表** | 官方推理镜像（如 `rocm/sgl-dev:*`）、AITER `gradlib` 产出的 `*_tuned_gemm.csv`、仓库内按模型预置的 **CSV/JSON** | 复现环境与已知最优 **shape→kernel** 映射，避免冷启动全面搜索。 |
| **2. PyTorch TunableOp → hipBLASLt / rocBLAS** | `PYTORCH_TUNABLEOP_ENABLED=1` 驱动运行时从 **rocBLAS** / **hipBLASLt** 候选中选优并落盘 **GEMM Table** | 框架内自动适配当前硬件与精度，适合训练/推理中长期复用。 |
| **3. TensileLite / Tensile 自定义 kernel** | **Tensile** 生成内核，经 **YAML** 描述 **GEMM** 变体，由 **rocBLAS** / **hipBLASLt** 加载 | 需要覆盖库未优化的 **shape**、特殊 **epilogue** 或新 **dtype** 时使用；迭代成本高、上限也最高。 |

**关系简述**：底层 **GEMM** 多由 **Tensile** 生成；**rocBLAS** 暴露 `rocblas_*gemm*`，**hipBLASLt** 提供更丰富的融合与 **algorithm** 选择；**PyTorch** 通过 **TunableOp** 在二者（及表驱动结果）之间择优。

---

## hipBLASLt API 工作流

核心运算：\(D = \text{Activation}(\alpha \cdot op(A) \cdot op(B) + \beta \cdot op(C) + \text{bias})\)。

推荐调用顺序：

1. `hipblasLtCreate()` — 创建 **library handle**
2. `hipblasLtMatrixLayoutCreate()` — **A/B/C/D** 的 **layout**（类型、行列、**ld**）
3. `hipblasLtMatmulDescCreate()` — **matmul descriptor**（`computeType`、`scaleType` 等）
4. `hipblasLtMatmulDescSetAttribute()` — **transpose**、**epilogue**、**bias** 等（`hipblasLtMatmulDescAttributes_t`）
5. `hipblasLtMatmulPreferenceCreate()` — **workspace** 上限等偏好
6. `hipblasLtMatmulAlgoGetHeuristic()` — 按估计耗时排序的 **algorithm** 列表（`requestedAlgoCount`）
7. `hipblasLtMatmul()` — 执行 **GEMM**（`alpha`、`beta`、`algo`、`workspace`、`stream`）
8. `hipblasLtDestroy()` — 释放资源

常用函数速查：`hipblasLtMatmulDescCreate`、`hipblasLtMatmulAlgoGetHeuristic`、`hipblasLtMatmul`。

---

## rocBLAS GEMM 入口与调优关系

**rocBLAS** 为 **ROCm** 基础 **BLAS**，**Level-3 GEMM** 内部委托 **Tensile**。命名习惯：`rocblas_<type>gemm()`。

| 入口示例 | 用途 |
|----------|------|
| `rocblas_sgemm` | **FP32** |
| `rocblas_dgemm` | **FP64** |
| `rocblas_hgemm` | **FP16** |
| `rocblas_gemm_ex` | 混精（**FP8**、**BF16**、**INT8** 等） |
| `rocblas_*gemm_batched` / `*strided_batched` | **Batched** 场景 |

调优层面：除直接换 **API** 外，通常配合 **`TENSILE_*` 环境变量**、**TunableOp** 或 **Tensile** 逻辑文件（见下文）改变实际选中内核。

---

## Stream-K 调度（`TENSILE_SOLUTION_SELECTION_METHOD=2`）

**Stream-K** 将 **K** 维内层迭代跨 **CU** 划分，利于非方阵与负载不均形状，在 **MI350** 系列上为**唯一**策略。

| 环境变量 | 取值与含义 |
|----------|------------|
| `TENSILE_SOLUTION_SELECTION_METHOD` | `0` = 标准 **tuned**（默认）；`2` = **Stream-K** |
| `TENSILE_STREAMK_DYNAMIC_GRID` | `6` = 自动（默认）；`0` = 使用全部 **CU** |
| `TENSILE_STREAMK_FIXED_GRID` | 整数：固定 **workgroup** 数量 |
| `TENSILE_STREAMK_MAX_CUS` | 上限 **CU** 数，用于限制 **Stream-K** 占用 |

**并发提示**：例如设 `TENSILE_STREAMK_FIXED_GRID=64` 可避免单次 **GEMM** 占满 **GPU**，便于与其它内核共栖。

---

## hipBLASLt Online Tuning（`HIP_ONLINE_TUNING=1`）

启用 **hipBLASLt** 的 **GEMM online tuning**（**gradlib** / 工具链侧常用）：

```bash
export HIP_ONLINE_TUNING=1
```

首次会有数分钟级一次性开销；结果可写入 **`hip_online_tuning_res.csv`** 供后续加载。适合 **shape** 分布变化大、希望在线收敛到当前机器最优 **solution** 的场景。

---

## Tensile / TensileLite 侧关键参数（概念）

在 **Tensile** 逻辑与 **YAML** 中，下列项共同决定 **macro** 与 **instruction** 选择（名称随版本略有差异，以所用 **Tensile** 分支为准）：

| 参数类 | 作用 |
|--------|------|
| **WorkGroup** | **Workgroup** 网格与每 **block** 负责的 **tile** 范围 |
| **ThreadTile** | **Thread** 级 **M×N** 子块，影响寄存器与 **lane** 映射 |
| **MacroTile** | **Block** 级大 **tile**（常对应 **M/N/K** 方向的主分块） |
| **LoopUnroll** | **K** 环展开深度，平衡 **ILP** 与寄存器压力 |
| **SplitU / GlobalSplitU** | **K** 维拆分与多 **wave** 归约策略 |
| **MatrixInstruction** | 选用的 **MFMA**（或其它 **Matrix Core**）指令形状与类型 |

**YAML** 示例（摘自爬取报告）：`SGEMM{M: 5504, N: 5504, K: 5504, transposeA: false, transposeB: true, dataType: S}`；混精可写 **`GEMM_EX (HHS){...}`** 等。

---

## 关键环境变量总表

### Tensile / hipBLASLt / Stream-K

| 变量 | 说明 |
|------|------|
| `TENSILE_SOLUTION_SELECTION_METHOD` | `0` 标准 / `2` **Stream-K** |
| `TENSILE_STREAMK_DYNAMIC_GRID` | **Stream-K** **grid** 策略 |
| `TENSILE_STREAMK_FIXED_GRID` | 固定 **workgroup** 数 |
| `TENSILE_STREAMK_MAX_CUS` | **Stream-K** 最大 **CU** |
| `HIP_ONLINE_TUNING` | `1` 启用 **hipBLASLt online tuning** |

### PyTorch / 模型加速（摘自 P2）

| 变量 | 说明 |
|------|------|
| `PYTORCH_TUNABLEOP_ENABLED=1` | 启用 **TunableOp**，从 **rocBLAS**/**hipBLASLt** 择优并缓存 |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | 优先走 **hipBLASLt**（与框架版本相关） |

### Flash Attention 后端（AMD）

| 变量 | 说明 |
|------|------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE` | **CK** 后端（常见默认） |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` | **Triton** 后端 |

---

## Attention kernel 选择指南

综合延迟、内核成熟度与 **ROCm** 支持，典型优先级（同场景下需实测）：

**AITER asm（汇编等极致路径） > CK FMHA（Composable Kernel / CK-Tile） > Triton FA > FlashInfer**

| 依据类型 | 说明 |
|----------|------|
| **AITER** | **inline asm**、**CK**、**Triton** 多路径；**DeepSeek** 等场景博文给出相对基线的大幅 **speedup**（见下表）。 |
| **CK-Tile** | **FA-v2** 可在约百行量级表达，**MFMA** 与 **LDS** 管线清晰。 |
| **Triton** | 开发迭代快；**MI300X** 上 **TileLang** 博文同配置下常弱于最优 **CK/手写**。 |
| **FlashInfer** | 服务向 **paged KV**、**GQA/MQA**、**torch.compile**；**ROCm** 多版本支持，可作通用后端。 |

---

## 各实现性能对比数据（公开博文 / 爬取汇总）

### GEMM 与 MoE / 端到端（AITER，MI300X）

| 项目 | 相对基线 | 备注 |
|------|-----------|------|
| **Block-scale GEMM** | **2×** | **FP8** 块缩放等 |
| **Block-scale fused MoE** | **3×** | 融合 **MoE** |
| **MLA decode** | **17×** | **DeepSeek V3/R1** |
| **MHA prefill** | **14×** | 同上系列 |
| **End-to-end SGLang（DeepSeek）** | **2.1×**（6484 → 13704 tok/s） | **8×MI300X** |

### Attention 微基准（其它）

| 实现 | 配置要点 | 结果 |
|------|----------|------|
| **Flash Attention（PyTorch SDPA）** | **AMD** 上相对 **eager** | 独立 **attention** 约 **2–8×**（随 **seq** 增大）；若干 **7B** 级 **prefill** 约 **1.4–1.6×** |
| **TileLang FA** | **MI300X**，batch=1，heads=8，seq=4096，dim=128 | **0.36 ms**，相对 **PyTorch** 参考 **2.69×** |
| **Triton FA** | 同上 | **0.55 ms**，**1.76×** |
| **hipBLASLt Stream-K** | 相对标准 **tuned** | 多变 **shape** 下峰值更稳（定性，**MI300** 系） |

---

## 参考与延伸阅读

- **hipBLASLt**、**rocBLAS**、**Tensile** 官方文档（ROCm Docs）。
- 爬取报告：`references/crawl-data/p1_gemm_attention_tuning.md`、`references/crawl-data/p2_system_optimization.md`。
