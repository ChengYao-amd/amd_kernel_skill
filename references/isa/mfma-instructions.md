# MFMA（Matrix Fused Multiply-Add）指令参考

本文汇总 **CDNA3（GFX942）** 与 **CDNA4（GFX950）** 上 Matrix Core 的 MFMA 语义、官方披露的 **tile 形状与周期**、**编译器 intrinsic**、**数据布局** 与 **FP8 编码差异**。表格与公式主要依据 AMD ROCm 官方「Matrix Core Programming」系列博文及 LLVM/HIP 公开接口；汇编层助记符形如 `v_mfma_*`，具体编码请以对应 `gfx*` 的 ISA 手册为准。

## 语义与记号

- MFMA 在一条指令内完成 **D ← A×B + C**（矩阵乘加 fused），其中 tile 形状记为 **M×N×K**：A 为 **M×K**，B 为 **K×N**，C/D 为 **M×N**。
- 下文 **(C,D) ← (A,B)** 表示累加器/结果元素类型与 A、B 元素类型；同一指令内 A、B 类型可相同或按变体混合（例如部分 FP8 混合 intrinsic）。

## CDNA3（GFX942）MFMA：完整指令表（官方博文）

下列为 ROCm 文档中针对 **GFX942** 给出的 MFMA 类型、**M×N×K** 与 **Cycles**（单条指令在 Matrix Core 上的周期数，用于粗略估算吞吐，**非**内存访问延迟）。

| Type (C,D) ← (A,B) | M×N×K | Cycles |
|---------------------|-------|--------|
| FP64 ← FP64 | 16×16×4 | 64 |
| FP32 ← FP32 | 32×32×2 | 64 |
| FP32 ← FP32 | 16×16×4 | 32 |
| FP32 ← FP16/BF16 | 32×32×8 | 32 |
| FP32 ← FP16/BF16 | 16×16×16 | 16 |
| FP32 ← FP8 (E4M3FNUZ / E5M2FNUZ) | 16×16×32 | 16 |
| FP32 ← FP8 (E4M3FNUZ / E5M2FNUZ) | 32×32×16 | 32 |

说明：

- **FP16/BF16**：同一类 MFMA 家族下，BF16 与 FP16 通常共享 **M×N×K** 与 **Cycles**，差异在数据编码与指令助记符中的类型后缀（如 `_f16` / `_bf16`）。
- **FP64 / FP32 原生**：用于双精度/单精度 GEMM 的「全精度路径」，K 维较小时周期数相对较大，需与访存与指令级并行一起建模。

## CDNA4（GFX950）：在 CDNA3 基础上的扩展

**CDNA4** 保留上表中 **全部 CDNA3 MFMA**，并增加下列条目（更大 **K** 的 FP16 类 tile，以及 **FP8/FP6/FP4** 与 **MXFP*** 块缩放路径）：

| Type (C,D) ← (A,B) | M×N×K | Cycles | Note |
|---------------------|-------|--------|------|
| FP32 ← FP16/BF16 | 16×16×32 | 16 | **NEW**：相对 CDNA3，**K 加倍** |
| FP32 ← FP16/BF16 | 32×32×16 | 32 | **NEW**：**K 加倍** |
| FP32 ← FP8/FP6/FP4 | 16×16×128 | 16–32 | **NEW**：A、B 类型可独立配置 |
| FP32 ← FP8/FP6/FP4 | 32×32×64 | 32–64 | **NEW** |
| FP32 ← MXFP8/MXFP6/MXFP4 | 16×16×128 | 16–32 | **NEW**：**block-scaled** |
| FP32 ← MXFP8/MXFP6/MXFP4 | 32×32×64 | 32–64 | **NEW**：**block-scaled** |

其中 **Cycles** 写成区间（如 16–32）表示与具体操作数类型组合、内部实现或时钟域配置相关的可变延迟；做性能建模时应以 **实际内核与 profiler** 为准。

## 编译器 Intrinsic：通用形式与修饰参数

LLVM/Clang 暴露的 MFMA 内建函数常见形式为：

```c
d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

- **a_reg / b_reg / c_reg / d_reg**：由编译器与寄存器分配器管理的向量寄存器（通常为 VGPR；若使用 AGPR 累加路径则由 ABI 与后端约定）。
- **cbsz, abid, blgp**：与 **稀疏/块压缩/操作数位选择** 等相关的控制字段（依具体 intrinsic 与架构而定）；编写汇编或内联实验代码时需对照 **LLVM 内置函数文档** 与对应 `gfx*` 的 **ISA 手册**，避免与 CUDA MMA 的「静态索引」假设混用。

### 命名示例（FP16 / FP8）

| 语义（示意） | Intrinsic 示例 |
|--------------|------------------|
| FP16，16×16×16 | `__builtin_amdgcn_mfma_f32_16x16x16f16` |
| FP16，32×32×8 | `__builtin_amdgcn_mfma_f32_32x32x8f16` |
| FP8，32×32×16，A 与 B 均为 FP8 | `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` |
| FP8，32×32×16，A 为 FP8、B 为 BF8（混合） | `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8` |

### CDNA4：带缩放的 MFMA（scale）

CDNA4 引入与 **FP8/FP6/FP4** 及 **scale** 相关的内建函数，形式示例：

```c
__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a, b, c,
    Atype, Btype,
    OPSEL_A, scale_a,
    OPSEL_B, scale_b
);
```

- **Atype / Btype**：标明 A、B 的低比特格式（如 FP8/FP6/FP4 组合）。
- **OPSEL_*** 与 **scale_***：用于操作数位选择与 per-block 或 per-element scale 路径（与 **MXFP*** 块缩放语义配合）；具体枚举与合法组合以 **ROCm/LLVM 发布说明** 为准。

## FP8 与低精度格式：CDNA3 vs CDNA4

| 项目 | CDNA3（GFX942） | CDNA4（GFX950） |
|------|-----------------|-----------------|
| FP8 风格 | **E4M3FNUZ**、**E5M2FNUZ**（非标准 OCP，指数偏置常为 **8 / 16**） | **E4M3FN**、**E5M2**（**OCP** 风格，指数偏置 **7 / 15**） |
| 迁移注意 | 既有内核若硬编码 FNUZ 语义，在 CDNA4 上需核对 **reinterpret / 转换** | 与 **OCP FP8** 生态（训练框架、量化工具）对齐更容易 |

HIP 侧常见存储类型（名称随 ROCm 版本演进，以头文件为准）：

- `__hip_fp8_storage_t`
- `__amd_fp8_storage_t`

在调用 intrinsic 前，通常需要将 **FP8 操作数** 按 ABI 要求 **扩展到合适寄存器宽度**；官方示例中常将 FP8 操作数 **cast 为 `long`** 再传入内建，以避免隐式转换与未定义布局。

## 峰值算力估算公式（Matrix Core）

官方博文给出的 **峰值 TFLOP/s** 估算形式为：

```text
Peak_TFLOPS = 2 * M * N * K * num_matrix_cores * (max_engine_clock_Hz / cycle_count) / 1e6
```

- **2×M×N×K**：一次 MFMA 在 **FMA** 意义下的浮点 op 计数（乘与加各计）。
- **num_matrix_cores**：GPU 上 Matrix Core 总个数（因产品而异，例如 MI300 系列文档中为 **1216**）。
- **max_engine_clock**：峰值引擎频率（Hz）。
- **cycle_count**：该 MFMA 变体对应的 **Cycles**（上表）。

**MI325X（GFX942）** 示例（FP16 **32×32×8**，**Cycles = 32**，**1216** 个 Matrix Core，**2100 MHz**）：

```text
2 * 32 * 32 * 8 * 1216 * (2100e6 / 32) / 1e6 ≈ 1307.4 TFLOP/s
```

该数值与 `hardware/mi300x.md` 中官方标称 **FP16/BF16 峰值** 一致；换用其他 **M×N×K** 或 **cycle_count** 时，应同步替换公式中的两项。

## 数据布局与 Wavefront 映射要点

以下归纳自官方示例与社区实践，用于手写内核或核对编译器输出：

- **Wavefront 宽度**：**64 threads**；MFMA 的矩阵元素在 **64 条 lane** 间 **分布存放**，与 NVIDIA warp（32 lane）的 MMA 布局 **不同**，**不可**按 CUDA 经验直接照搬索引。
- **每线程持有的元素数（概念上）**：
  - A：**M×K / 64**
  - B：**K×N / 64**
  - C/D：**M×N / 64**
- **寄存器类型**：结果常落在 **VGPR** 的向量类型中（如 `fp32x4_t`、`fp32x16_t` 等，依 tile 与后端而定）。
- **FP8 调用约定**：操作数在传入 intrinsic 前常 **cast 为 `long`**，以保证寄存器宽度与调用约定一致。
- **与 NVIDIA MMA 对比**：lane 到 **(row,col)** 的映射规则不同；做 **layout 转换** 或 **与 CUTLASS/cuBLAS 结果 bitwise 对比** 时，必须以 **AMD 文档或 llvm-mca / 反汇编** 为准。

## 汇编层与 INT8 等其它变体

- 汇编助记符习惯写作 **`v_mfma_{out}_{M}x{N}x{K}_{in}`**（具体后缀因类型与编码而异）。
- **INT8 → INT32** 等整数 MFMA 在 ISA 中同样存在（旧版笔记曾列 `v_mfma_i32_16x16x32_i8`）；**M×N×K、周期与寄存器 packing** 请以目标 `gfx` 的 **ISA 手册** 与 **LLVM `IntrinsicsAMDGPU.td`** 为准，本文以浮点与 FP8 主线为主。

## 与 NVIDIA Tensor Core 的简要对比

| 方面 | AMD MFMA | NVIDIA Tensor Core (MMA) |
|------|----------|---------------------------|
| 执行粒度 | **64 lane** wavefront | **32 lane** warp |
| 累加器 | **AGPR**（CDNA）或 VGPR，与 CUDA 路径不同 | 架构相关，常与寄存器文件紧耦合 |
| 布局 | **lane–矩阵** 映射与 CUDA **不同** | 以 NVIDIA 文档为准 |
| 软件栈 | ROCm、HIP、LLVM intrinsics | CUDA、PTX、mma.sync |

## 优化建议（与访存协同）

1. **指令级并行**：在 MFMA **占满周期** 的同时，发射 **global_load / DS 读写**，用 `s_waitcnt` 精确控制，隐藏 **VMEM/LDS** 延迟。
2. **累加器与寄存器压力**：优先用 **AGPR** 承载 MFMA 累加链，**释放 VGPR**，提高 wave **occupancy**（详见 `register-allocation.md`）。
3. **Tile 选择**：在 **周期与寄存器占用** 间权衡；例如 **32×32×8** 相对 **16×16×16** 往往在不同瓶颈下表现不同，需结合 **roofline** 与 **occupancy** 实测。
4. **跨代兼容**：面向 **CDNA4** 时，关注 **更大 K 的 FP16** 与 **FP6/FP4/MX** 新指令族；编译选项与 `offload-arch` 需设为 **`gfx950`**（或产品对应目标）。
5. **数值路径**：FP8 训练/推理若在 **FNUZ（CDNA3）** 与 **OCP（CDNA4）** 间迁移，务必做 **数值回归** 与 **精度对齐**。

## 参考

- AMD ROCm Blog: *Matrix Core Programming*（含 **MFMA 表**、intrinsic 示例与峰值公式）。
- LLVM：`IntrinsicsAMDGPU.td`、Clang `__builtin_amdgcn_mfma_*` 定义。
- 各代 *RDNA/CDNA ISA Reference*（指令编码与延迟细节）。
