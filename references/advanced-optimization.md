# 极限优化：突破性能平台期

## 使用时机

当 kernel 已通过正确性验证、已超过 torch.compile baseline，但与理论峰值仍有差距（带宽利用率 <70% 或 MFMA 利用率 <80%）时。按以下顺序尝试。

## 1. 软件流水线与多级缓冲

**原理**：当前迭代的计算与下一迭代的数据加载重叠执行。

**技术**：
- 双缓冲：LDS 分两半，交替加载和计算
- 三级流水线：加载 N+1、计算 N、写回 N-1
- 异步拷贝：DMA 引擎搬运数据与 MFMA 完全并发

**AMD 实现要点**：
- 需要精确的 `s_waitcnt lgkmcnt/vmcnt`（参见 `isa/scheduling-pipeline.md`）
- LDS 缓冲：核对每 CU **LDS 容量与读带宽**（CDNA4 白皮书：**160 KiB/CU**、**256 B/clock** 读带宽；相对 CDNA3 容量与读带宽约 **2×**；详见 `isa/memory-instructions.md`）
- **CDNA4**：存在 **LDS 自 L1 data cache 的 direct load** 路径，可将热数据尽量留在 **L1→LDS** 链路上，减轻与全局路径的争抢；与 double buffering、异步搬运一起 profile。
- 示例：当前 MFMA 执行时预取下一个 tile

**预期收益**：内存受限 kernel 通常 10-30%

## 2. Wavefront 特化

**原理**：同一 block 内不同 wavefront 承担不同角色，通过 barrier 协调。

**角色**：计算 wavefront（MFMA）、数据搬运（global→LDS）、归约（softmax/reduction）

**AMD 实现要点**：
- wavefront=64 → 每个 wavefront 更重，特化收益更大
- 角色分配：`threadIdx.x / 64` 给出 wavefront ID
- 通过 `__syncthreads()` 或 LDS fence 协调

**预期收益**：多阶段 kernel（Attention）可获 5-15%

## 3. 数据布局与 Swizzle

**原理**：改变数据排列以消除 bank conflict 和提升合并访问率。

**技术**：
- LDS padding：按目标架构 **bank 数** padding（CDNA3：**32 bank** × 4B；CDNA4：**64 bank** — 见 `isa/memory-instructions.md`）
- `ds_swizzle`：硬件 lane 排列，无需 LDS 读写
- SOA 布局用于向量化 load（`buffer_load_dwordx4`）

**AMD 实现要点**：
- AMD LDS bank 规则与 NVIDIA 不同 — 重新计算 padding
- 用 `omniperf` LDS bank conflict 指标验证

**预期收益**：存在 bank conflict 时 5-20%

## 4. Occupancy vs ILP 权衡

**原理**：有时更少 wavefront 配合更多寄存器优于大量 wavefront 配合溢出。

**技术**：
- `__launch_bounds__(threads, minBlocks)` 控制寄存器分配
- 检测溢出：`omniperf` → ScratchWaveslifetimeVGPR > 0
- 跨 wavefront group 的寄存器重平衡（参考 AVO v33）

**AMD 实现要点**：
- MI300X：65536 VGPR/CU。参见 `isa/register-allocation.md` 中的 occupancy 表
- 高 occupancy 和低 occupancy 配置都 profile — 更快的获胜

**预期收益**：消除溢出时 3-10%

## 5. Persistent Kernel 与 Tile 调度

**原理**：只 launch 一次，通过原子计数器自行分配 tile。消除重复 launch 开销 + L2 友好的遍历顺序。

**技术**：
- `atomicAdd` 全局 tile 计数器
- Swizzled 遍历（L 形、Z 形、Hilbert 曲线）提升 L2 复用
- 跨 tile 负载均衡用于不规则形状（causal mask）

**AMD 实现要点**：
- MI300X L2 较大（256MB）— tile 遍历顺序影响显著
- CK 的 TileScheduler 可作为参考实现

**预期收益**：多次 launch 场景 5-15%；L2 优化通常 3-8%

## 6. 混合精度策略

**原理**：超越"全用 BF16"，在不同计算阶段使用不同精度。

**技术**：
- 输入 FP8/BF16 → MFMA 计算 → FP32 累积 → BF16 输出
- 关键中间结果（如 softmax max/sum）保持 FP32
- 利用 MFMA 的混合精度能力（FP16 输入、FP32 输出）

**AMD 实现要点**：
- 查阅 `isa/mfma-instructions.md` 了解可用精度组合与 **每 CU FLOPS/clock**（CDNA4 上 FP16/FP8/MX 等相对 CDNA3 提升，**Matrix FP64 减半**）
- MI355X/CDNA4 新增 **MXFP6/MXFP4** 等格式 — 以硬件文档与 ROCm 发布说明为准
- **超越函数（transcendental）吞吐**：CDNA4 上相关指令有效速率相对 CDNA3 约 **2×**，**softmax**、激活等含 **exp/log** 的片段更易成为算子内可优化热点
- **Structured sparsity**：当输入在 **每 4 个元素一组** 中 **零元素比例 ≥ 50%** 时，硬件路径上可 **翻倍** 有效吞吐（需算子/库支持与精度验证）

**预期收益**：计算受限 kernel 可获 10-30% TFLOPS 提升

## 7. 编译器对抗与引导

**原理**：在编译器自动优化不足或过度时手动干预。

**技术**：
- `#pragma unroll N`：精确控制展开
- `__launch_bounds__`：引导寄存器分配
- `volatile` / `__builtin_nontemporal_*`：绕过 cache / 阻止重排序
- 内联汇编：最后手段（参见 `isa/inline-asm-patterns.md`）

**AMD 实现要点**：
- `hipcc -save-temps` 可查看生成的 ISA，验证编译器行为

**预期收益**：因情况而异，通常 2-10%

## 8. L2 Cache 全局优化

**原理**：全局层面的数据复用策略，跨 tile / 跨 kernel 最大化 L2 命中率。

**技术**：
- Tile 遍历顺序（GEMM 的 K 维度尤其敏感）
- 跨 kernel fusion 保持数据在 L2 中
- L2 cache residency 控制（如硬件支持 prefetch hint）

**AMD 实现要点**：
- MI300X L2 = 256MB，相对较大
- 用 `rocprof` TCC_HIT/TCC_MISS 计数器测量

**预期收益**：内存受限 kernel 通常 3-8%

## 9. FP8 GEMM 优化递进（CDNA4 / ROCm 实践）

ROCm 博文等资料给出在 **CDNA4** 上 **FP8 GEMM** 从朴素实现到接近峰值的典型递进（数值为文中示例，实际以本机 profiling 为准）：

| 阶段 | 做法 | 代表性结果（示例） |
|------|------|---------------------|
| Naive | 未充分 tiling / 未对齐硬件 MFMA | ~**1.15 TFLOPS** |
| LDS tiling + MFMA | 软件 tile 与矩阵核对齐 | 大幅提升 |
| 向量化 load + `buffer_load_lds` | 异步 DMA、缓解 VMEM | 再提升 |
| LDS swizzle + double buffering | 降 bank conflict、重叠搬运与计算 | 显著增益 |
| 8-wave ping-pong 调度 | `__builtin_amdgcn_s_setprio()`、`sched_barrier`、与 `buffer_load_lds` 协同 | ~**2597 TFLOPS**（接近峰值区） |

**关键技术摘要**：

- **`buffer_load_lds`**：异步把数据送入 LDS，利于与 MFMA 流水重叠。
- **`__builtin_amdgcn_s_setprio()`** 与 **`sched_barrier`**：控制 wave 优先级与调度屏障，服务 **ping-pong** 多 wave 编排。
- 与上文「软件流水线」「数据布局与 Swizzle」两节对照：CDNA4 上 **LDS 为 64 bank**，padding 需按 **64-bank** 重算（见 `isa/memory-instructions.md`）。

## 选择指南

| 瓶颈类型 | 诊断指标 | 优先技术 |
|----------|---------|---------|
| HBM 带宽受限 | 带宽利用率 >80% | 软件流水线、L2 优化、数据布局 |
| LDS 受限 | Bank conflict 率高 | Swizzle、padding、数据布局 |
| 计算受限 | MFMA 利用率 <70% | Wavefront 特化、混合精度、ILP |
| 寄存器溢出 | ScratchWaves > 0 | Occupancy 调优、寄存器重平衡 |
| Launch 开销 | 多次小 kernel launch | Persistent kernel |
| 编译器问题 | ISA 审查发现冗余指令 | 编译器引导、内联汇编 |
