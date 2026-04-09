# AMDGPU ISA 速查参考

ISA 级优化的入口文档。先读此文件，再根据需要深入具体文档（如 `mfma-instructions.md`、`memory-instructions.md`、`register-allocation.md`）。

## 寄存器文件（容量以 ROCm / 硬件白皮书为准）

**重要**：寄存器文件大小应使用 **每 CU 存储容量（KiB）** 表述。此前误用「65536」等数字当作「VGPR 个数」易造成误解——**512 KiB VGPR / CU** 表示的是 **物理向量寄存器堆容量**，实际每个 wavefront 能用的 **VGPR 个数** 仍受 **架构上限与 occupancy** 约束（见 `register-allocation.md`）。

| 类型 | 每 CU 容量（典型 CDNA） | 用途 |
|------|-------------------------|------|
| **VGPR** | **512 KiB** | 每 lane 数据、MFMA 操作数与一般向量运算 |
| **SGPR** | **12.5 KiB** | 统一标量、地址、控制流与部分内存指令 |
| **AGPR** | **512 KiB / CU（与 VGPR 文件同大小，CDNA）** | MFMA 等矩阵累加专用（与 VGPR 间有 `v_accvgpr_read` / `v_accvgpr_write` 等路径） |

补充：

- **每 wavefront** 的 **VGPR/SGPR 个数上限** 决定 **可同时驻留的 wave 数**，进而影响 **occupancy**；这与「整颗 CU 的 KiB 容量」是不同层面的两个量，调优时两者都要看。
- 具体 **最大 VGPR 数/wave**、**最大 wave/SIMD** 等表格以 **目标 GPU 与 omniperf/编译器报告** 为准。

## 指令分类

| 类别 | 执行单元 | 示例 | 延迟（量级） |
|------|---------|------|----------------|
| VALU | 向量 ALU | `v_add_f32`, `v_fma_f32` | 约 4–8 周期 |
| SALU | 标量 ALU | `s_add_u32`, `s_cmp_eq` | 约 2–4 周期 |
| MFMA | Matrix Core | `v_mfma_f32_16x16x16_f16` 等 | **依变体约 16–64 周期**（见 `mfma-instructions.md`） |
| VMEM | 向量内存 | `global_load_dwordx4` | 数百周期（高度依赖缓存与带宽） |
| LDS | 本地数据共享 | `ds_read_b128`, `ds_write_b64` | 约数十周期 |
| SMEM | 标量内存 | `s_load_dwordx4` | 数百周期量级 |

## 流水线模型：CDNA3（GFX942）简述

- **CU 结构**：每个 **CU** 内含多个 **SIMD** 执行单元；每个 **SIMD** 以 **wavefront（64 lane）** 为调度粒度，**每周期** 可对 **一条** wave 指令推进（与 **VALU 4-cycle 吞吐** 等细节结合理解 **IPC**）。
- **MFMA**：指令进入 **Matrix Core** 流水线，**延迟高、可流水线化**；与 **VMEM/LDS** 并行发射时需用 **`s_waitcnt`** 管理 **vmem/vs/sc/lds** 等计数器。
- **内存**：**全局/纹理** 访问异步完成，依赖 **waitcnt** 与 **屏障** 保证语义。

更细的调度与气泡消除见 `scheduling-pipeline.md`、`inline-asm-patterns.md`。

## CDNA4（GFX950）流水线与 ISA 增量要点

在继承 **CDNA3** 整体执行模型（wavefront、SIMD、VMEM/LDS 异步模型）的前提下，**CDNA4** 在 **Matrix Core** 上显著扩展了 **MFMA** 族：

- **更大 K 的 FP16/BF16 tile**（例如 **16×16×32**、**32×32×16**），在相同 **M×N** 下提高 **K 维吞吐**，有利于 **减少指令条数** 与 **提高 FMA 密度**（需配合 **寄存器与 LDS 占用** 评估）。
- **FP8 / FP6 / FP4** 路径：**A、B 类型可独立配置** 的 MFMA 变体，以及 **16–64 周期** 量级的实现（依形状与类型组合变化）。
- **MXFP8 / MXFP6 / MXFP4**：**块缩放（block-scaled）** 低精度格式，与 **scale MFMA** intrinsic（如 `__builtin_amdgcn_mfma_scale_f32_*`）配合，用于 **在 FP32 累加精度下** 压缩 **权重/激活** 存储与带宽。

**迁移提示**：面向 CDNA4 时，编译目标使用 **`gfx950`**（或产品文档指定的 offload arch）；**FP8 编码** 在 CDNA4 上对齐 **OCP（E4M3FN / E5M2）**，与 CDNA3 的 **FNUZ** 变体在 **指数偏置与语义** 上不同，跨代内核需做 **显式转换与数值验证**。

## ISA 知识使用时机

| 优化阶段 | ISA 深度 | 对应文档 |
|---------|---------|---------|
| 早期（融合、tiling） | 低 | 仅本文件 |
| 中期（occupancy、内存访问） | 中 | `memory-instructions.md`, `register-allocation.md` |
| 后期（指令调度、消除 bubble） | 高 | `scheduling-pipeline.md`, `inline-asm-patterns.md` |
| MFMA 密集型（GEMM、Attention） | 高 | `mfma-instructions.md` |
