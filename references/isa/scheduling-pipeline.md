# 指令调度与流水线指南

## CDNA3 流水线模型

每个 CU 有 4 个 SIMD 单元。每个 SIMD：

- 每周期执行一条 wavefront 指令
- 在就绪的 wavefront 之间轮询（TLP 隐藏延迟）

延迟隐藏：如果 wavefront A 在内存上停顿，SIMD 执行 wavefront B、C、D 等。

## 指令级并行（ILP）

当 occupancy 较低（wavefront 少）时，单个 wavefront 内的 ILP 变得至关重要。

**目标**：通过交错独立指令保持流水线饱满。

```
// 差：依赖链 → 流水线停顿
global_load v0, ...
s_waitcnt vmcnt(0)    // 在此停顿
v_add_f32 v1, v0, v2  // 必须等待 load

// 好：交错独立工作
global_load v0, ...    // 发射 load
v_mul_f32 v3, v4, v5   // 独立的 VALU 工作
s_add_u32 s0, s0, 1    // 独立的 SALU 工作（可双发射！）
s_waitcnt vmcnt(0)     // 此时 load 大概率已完成
v_add_f32 v1, v0, v2   // 使用已加载的数据
```

## s_waitcnt 策略

**原则**：尽可能晚等待，尽可能少等待。

| 模式 | 代码 | 原因 |
|------|------|------|
| 立即等待 | load 后立即 `s_waitcnt vmcnt(0)` | 差：杀死 ILP |
| 延迟等待 | load，做其他工作，然后等待 | 好：隐藏延迟 |
| 部分等待 | `vmcnt(N)` 其中 N = 允许挂起的剩余操作数 | 最佳：最小停顿 |

## 双发射规则（CDNA3）

VALU + SALU 可在同一周期发射，条件：

1. 两者之间无寄存器依赖
2. VALU 使用 VGPR，SALU 使用 SGPR
3. 两者都就绪（无挂起的等待）

**优化**：将地址计算（SALU）与数据操作（VALU）配对。

## MFMA 调度

MFMA 指令延迟高（64 周期）但可与其他工作重叠：

```
// 流水线：MFMA N 执行时，加载 N+1 的数据
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 64 周期
global_load_dwordx4 v[0:3], ...  // 在 MFMA 延迟期间发射
global_load_dwordx4 v[4:7], ...  // 在 MFMA 延迟期间发射
s_waitcnt vmcnt(0)               // 此时 load 应已完成
v_mfma_f32_32x32x8_bf16 a[0:15], v[0:3], v[4:7], a[0:15]  // 下一个 MFMA
```

---

## MFMA 依赖解析规则（CDNA4，ISA Table 38）

以下为 **CDNA4 ISA 文档依赖表（Table 38）** 中与本仓库 agent 调度相关的摘录：用于判断两条指令之间需插入多少 **NOP**（或等价独立工作），避免 RAW/WAR 冒险。数值与具体 **MFMA 变体**有关；无转发时需按最大间隔规划 ILP。

| 场景 | NOP 要求 |
|------|----------|
| 非 MFMA 的 VALU 写入某 VGPR → MFMA 读取**同一** VGPR | **2**（需 2 个 NOP） |
| MFMA 写入 → **同一** MFMA 作为 **SrcC**（累加，**完全相同** opcode） | **0**（支持 forwarding） |
| MFMA 写入 → **不同** MFMA 作为 SrcA/B 读 | **5 / 8 / 12 / 20**（依 MFMA 变体，**无** forwarding） |
| MFMA 写入 → VALU / VM / LDS / FLAT 读**重叠目的**寄存器 | **5 / 8 / 12 / 20** |
| SGEMM 写入 → SGEMM 读 SrcC（同一目的） | **0**（forwarding） |
| `V_CMPX` 写 **EXEC** → MFMA | **4**（无 exec mask forwarding） |
| XDL / SMFMAC 读 SrcC → VALU 写（WAR） | **1 / 3 / 7 / 15**（依变体） |

**对 agent 的含义**：

1. **背对背 MFMA 累加**（同 opcode、SrcC 链）：可利用 **0 NOP** 转发，但仍需满足其它 hazard（如 LDS/VMEM）。
2. **切换 SrcA/B 来源**或 **MFMA 与 VALU/LDS 混排**：必须按上表插入足够 **独立指令**（或 `s_nop`），不能假设与 SrcC 相同。
3. 与 **`V_CMPX` 修改 EXEC** 相邻的 MFMA：预留 **4 NOP** 级间隔。
4. 设计 **8-wave ping-pong**、**sched_group_barrier** 时，应用上述规则校验内层展开是否足够「填满」流水线。

> 完整矩阵以 AMD CDNA4 ISA PDF 为准；不同 `v_mfma_*` / `v_smfmac_*` 形状对应不同 NOP 列。
