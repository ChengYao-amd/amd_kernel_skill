# 寄存器分配与 Occupancy 指南

## 寄存器文件容量（来自 ROCm 官方硬件规格）

| 寄存器类型 | MI300X (CDNA3) | MI355X (CDNA4) | 说明 |
|-----------|---------------|---------------|------|
| VGPR File | 512 KiB / CU | 512 KiB / CU | 每 lane 向量数据、MFMA 操作数 |
| SGPR File | 12.5 KiB / CU | 12.5 KiB / CU | 统一值、地址、控制流 |
| AGPR File | 与 VGPR 同大小 | 与 VGPR 同大小 | MFMA 累加器专用（CDNA） |

> **注意**：旧文档常写"65536 VGPR/CU"，这是指**寄存器数量**（每个 32-bit），等价于 `65536 × 4B = 256 KB`。但 ROCm 官方 GPU 规格表标注为 **512 KiB/CU**（包含 VGPR + AGPR 共享的物理文件）。编写 kernel 时用**寄存器个数**更直观；评估硬件容量时参考 **KiB**。

## VGPR 预算与 Occupancy（MI300X / CDNA3，每个 SIMD 单元）

每个 CU 有 4 个 SIMD 单元，每个 SIMD 有 16384 个 VGPR（32-bit）。

| 最大 VGPR / Wavefront | 最大 Wavefront / SIMD | Occupancy |
|-----------------------|-----------------------|-----------|
| 128 | 8 | 100% |
| 192 | 5 | 62.5% |
| 256 | 4 | 50% |
| 384 | 2 | 25% |
| 512 | 2 | 25% |
| 1024 | 1 | 12.5% |

公式：`wavefronts_per_simd = floor(16384 / vgpr_per_wavefront)`，上限 8。

## SGPR 预算

| 最大 SGPR / Wavefront | 最大 Wavefront / SIMD |
|-----------------------|-----------------------|
| ≤ 102 | 8 |
| > 102 | 溢出到内存 |

SGPR 很少限制 occupancy。重点管理 VGPR。

## AGPR（Accumulation GPR）

- 专用于 MFMA 累加器结果
- 与 VGPR 共享同一物理寄存器文件（512 KiB / CU）
- 使用 AGPR 做累加器可释放 VGPR 给数据 → 更高 occupancy
- VGPR ↔ AGPR 移动：`v_accvgpr_read` / `v_accvgpr_write`（有延迟，避免在热循环中频繁移动）

## MI355X (CDNA4) 差异

- VGPR File 同为 512 KiB / CU，但 CU 数量不同（256 vs 304）
- LDS 增大到 **160 KiB / CU**：更多 LDS 可减少对 VGPR 的 spill 需求
- 新增 FP6/FP4 MFMA 指令：更大的 K 维度意味着单条指令消耗更多 VGPR，需注意寄存器预算

## 检测寄存器压力

```bash
# 方法 1：编译时查看（推荐首选）
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# 在生成的 .s 文件中搜索:
#   .vgpr_count   → 该 kernel 使用的 VGPR 数量
#   .sgpr_count   → 该 kernel 使用的 SGPR 数量
#   .agpr_count   → 该 kernel 使用的 AGPR 数量

# 方法 2：ROCm Compute Profiler（原 omniperf）
rocprof-compute profile -n check_spill -- python run_kernel.py
rocprof-compute analyze -p check_spill/ --cli
# 关注 ScratchWaveslifetimeVGPR > 0 → 表示寄存器溢出！

# 方法 3：rocprofv3 计数器
rocprofv3 -i counters.txt python run_kernel.py
# 查看 SQ_WAVES, SQ_INSTS_VALU 等
```

## 降低寄存器压力

| 技术 | 方法 | 影响 |
|------|------|------|
| 缩短活跃范围 | 就近计算和消费值，减少同时活跃的变量 | 中等 |
| 使用 AGPR 做累加器 | `v_accvgpr_write` 把累加结果存入 AGPR | 释放 VGPR |
| `__launch_bounds__(threads, minBlocks)` | 提示编译器寄存器预算上限 | 直接控制 |
| `#pragma unroll N` | 精确控制展开系数（过多展开 = 更多 VGPR） | 平衡 ILP 与寄存器 |
| `__builtin_amdgcn_readfirstlane` | 将 lane 0 的值广播到 SGPR，减少 VGPR 占用（CK 常用模式） | 低开销 |
| 手动寄存器复用 | 重写循环以复用寄存器 | 高工作量 |
| 接受较低 occupancy | 如果 ILP 能隐藏延迟，更少 wave 也可以 | 权衡 |

## Occupancy vs ILP 权衡

低 occupancy 不一定是坏事。如果 kernel 有足够的 ILP（内存操作之间的独立指令），更少的 wavefront 配合更多寄存器可能优于大量 wavefront 配合寄存器溢出。

**决策流程**：
1. 检查是否溢出（ScratchWaves > 0 或 `.s` 文件中 spill 计数 > 0）
2. 如果溢出：减少 VGPR 使用（缩短活跃范围、用 AGPR）或接受较低 occupancy
3. 如果未溢出但性能低：尝试通过减少 VGPR 来提高 occupancy
4. 如果提高 occupancy 后性能反而下降：说明 ILP 更重要，回退到低 occupancy 配置
5. **两种配置都 profile，更快的那个获胜**

## MI300X 实用参考数值

| 场景 | 推荐 Occupancy | 理由 |
|------|---------------|------|
| 内存受限 kernel（逐元素、归一化） | ≥ 50%（≤ 256 VGPR） | 需要 TLP 隐藏内存延迟 |
| 计算受限 GEMM（MFMA 密集） | 25-50%（256-512 VGPR） | MFMA 延迟靠 ILP 隐藏 |
| 极限优化（手动流水线） | 可低至 12.5%（≤ 1024 VGPR） | 靠 ILP + software pipelining |
