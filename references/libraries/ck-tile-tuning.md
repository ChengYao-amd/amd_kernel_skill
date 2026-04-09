# CK Tile 与 Pipeline 调优参考（ck_tile GEMM / FMHA）

本文档汇总从 **Composable Kernel 源码**（`include/ck_tile/`）可读出的 **GEMM pipeline 变体、调度策略、partitioner 与代表性 tile 配置**，并补充 **FMHA** 侧的常见约束；用于在 MI300 系列等 AMD GPU 上缩小调优搜索空间。数值以仓库中 **预设配置名** 为准，实际 LDS/VGPR 占用需以编译期检查与 profiler 为准。

## 1. Pipeline 与实现类（`ops/gemm/pipeline/`）

| Pipeline 标签 | 实现类（示例名） | 设计取向 |
|---------------|------------------|----------|
| `MEMORY` | `GemmPipelineAgBgCrMem` | **Memory-bound**：`MinMemInFlyBytes` 常与 **32768** 相关；`PrefetchStages` **2–8**；**Intrawave / Interwave** 可选。 |
| `COMPUTE_V3` | `GemmPipelineAgBgCrCompV3` | `PrefetchStages=2`；**仅 Intrawave**。 |
| `COMPUTE_V4` | `GemmPipelineAgBgCrCompV4` | **双缓冲 LDS（ping-pong）**，提高访存隐藏能力。 |
| `COMPUTE_V5` | `GemmPipelineAgBgCrCompV5` | `NumWaveGroups=2`。 |
| `COMPUTE_V6` | `GemmPipelineAgBgCrCompV6` | **K tile 固定为 32**。 |
| `COMPUTE_ASYNC` | `GemmPipelineAgBgCrCompAsync` | **大 `K_Warp_Tile`（如 128）**、双 SMEM、**FP4** 等异步路径。 |
| `PRESHUFFLE_V2` | `WeightPreshufflePipelineAGmemBGmemCRegV2` | **推理权重 preshuffle**（AG mem / BG mem / C reg 路径）。 |

**调度器搭配经验法则**：

- **Memory pipeline**：关注 **Interwave** 与 **PrefetchStages**、**MinMemInFlyBytes**，用较大「在飞」字节数掩盖 DRAM / L2。
- **Compute pipeline**：**Intrawave** 常为默认；与 **COMPUTE_V3/V4/V5/V6** 的 wave、K 固定约束一并考虑。

## 2. Tile Partitioner 选择

| Partitioner | 适用场景 |
|-------------|----------|
| `GemmTile2DPartitioner` | 通用 **2D** `(M_blocks, N_blocks)`，易与直观 batch / head 划分对齐。 |
| `GemmTile1DPartitioner` | **线性化** block 索引；适合与 persistent tile、或简化 grid 逻辑。 |
| `GemmSpatiallyLocalTilePartitioner` | **空间局部** + **RemapXCD**（**gfx94x multi-die**）：减轻跨 XCD / 跨 cache 一致性流量。 |

当 profile 显示 **L2 / 跨 die** 异常热点时，优先尝试 **SpatiallyLocal** 与 **MEMORY / Interwave** 组合。

## 3. 源码中的代表性 `GemmConfig*`（真实 tile 形状）

下表为从配置命名可直接读出的 **block tile（M×N×K）** 与 **warp 网格（M×N×K）**；K 列中 **(128/elem)** 表示按 **元素** 计的 K tile，**(256/storage)** 表示按 **存储块** 计的 K（与 FP8/打包布局相关，需结合 `TileGemmTraits`）。

| 配置名 | Pipeline | Block (M×N×K) | Warps (M×N×K) | 典型用途 |
|--------|----------|---------------|---------------|----------|
| `GemmConfigMemoryInterwave` | `MEMORY` | 128×32×(128/elem) | 4×1×1 | **Memory-bound**、偏带宽场景 |
| `GemmConfigComputeV3` | `COMPUTE_V3` | 16×64×(256/storage) | 1×4×1 | **小 M**、高并行细分 |
| `GemmConfigComputeV3_1` | `COMPUTE_V3` | 256×256×(128/elem) | 2×2×1 | **大 GEMM** |
| `GemmConfigComputeV4` | `COMPUTE_V4` | 256×256×(64/elem) | 2×2×1 | **LDS ping-pong** |
| `GemmConfigComputeV5` | `COMPUTE_V5` | 128×128×(64/elem) | 1×1×2 | **双 wave group** |
| `GemmConfigComputeAsync` | `COMPUTE_ASYNC` | 64×64×256 | 1×4×1 | **FP4** 等异步路径 |
| `GemmConfigPreshuffleDecode` | `PRESHUFFLE_V2` | 16×64×(256/storage) | 1×4×1 | **推理 decode** |
| `GemmConfigPreshufflePrefill` | `PRESHUFFLE_V2` | 128×128×(128/storage) | 1×4×1 | **推理 prefill** |

调优时可将上表作为 **离散起点**，再在 ±1 档 warp / block 上搜索；**PRESHUFFLE** 与 **COMPUTE_ASYNC** 路径需同时满足 **权重布局与异步流水** 的前置条件。

## 4. FMHA（Flash / tile attention）相关约束

从 CK 侧调优与实例配置可观察到：

- **head_dim（hdim）64 与 128** 通常有 **最完整** 的 tuning 覆盖；其它 hdim 可能需要从邻近配置外推或补 kernel。
- **代表性 block 形状**包括：**64×64×32**、**128×64×32**、**32×128×32**（数字分别对应 tile 的 M/N/序列或 KV 维分段，具体语义以对应 `Fmha*` / `Block*` 模板为准）。
- **`CppConstraint`**：用于 **按 CU / 架构** 做 **dispatch 约束**（不满足则回退或选另一实例）。
- **score 块**：常见 **`kM0=64`** 作为 **score tiling** 的 M 向基准，与 Q 块行数对齐时利于 MFMA 与 mask 布局。

FMHA 与 GEMM 共用 **ck_tile** 的「Shape → Traits → Pipeline → Kernel」思想，但额外受 **causal mask、GQA/MQA、varlen** 等逻辑约束；调优需在 **正确性测试**（数值与边界）通过后再做性能搜索。

## 5. 通用参数说明（与上表对照）

| 参数域 | 含义 | 备注 |
|--------|------|------|
| Block M/N/K | Thread block 在输出与规约维上的 tile | K 可能按 element 或 packed storage 计。 |
| Warp M/N/K | 每个 warp 负责的子 tile | 与 **wave**、**wave group**（如 V5）共同决定占用。 |
| `PrefetchStages` | 预取段数 | **MEMORY** 上可达 **2–8**；**COMPUTE_V3** 常为 **2**。 |
| 双 SMEM / ping-pong | 两套 LDS 缓冲轮换 | **COMPUTE_V4**、**COMPUTE_ASYNC** 等。 |
| `MinMemInFlyBytes` | 目标「在飞」最小字节 | **MEMORY** pipeline 与带宽 hiding 相关（如 **32768** 量级）。 |

## 6. 资源与占用检查清单

调优每个候选配置时建议逐项确认：

1. **LDS**：大致与 **(A tile + B tile) × 阶段数 × 数据宽度** 相关；ping-pong 与 **PrefetchStages** 成倍放大。需满足架构 **每 CU LDS 上限**（如 64KB 量级，以目标 ISA 文档为准）。
2. **Grid**：block 数应足以 **填满 GPU**；小 batch / 窄 M 时易出现 **launch 不足**。
3. **VGPR / SGPR**：大 warp tile 或复杂 epilogue 可能导致 **寄存器溢出** 或 **occupancy 下降**。
4. **多 die**：**gfx94x** 上对比 **`GemmTile2DPartitioner`** 与 **`GemmSpatiallyLocalTilePartitioner`**。

## 7. MI300X（GFX942）与 MI355X（GFX950）实践提示

- **GFX942**：上表 **GEMM / FMHA** 配置与 **RemapXCD** 策略针对性强；大 GEMM 可优先参考 **`GemmConfigComputeV3_1`** / **`GemmConfigComputeV4`**。
- **GFX950（MI355X）**：CU 数、频率与缓存层次变化后，**同名配置**可能需重调；建议以 **CK 中该 arch 的分支实例** 为起点，再扫 **block_m / block_n** 与 **pipeline 标签**。

## 8. 调优流程（建议）

1. 依据算子瓶颈选 **pipeline 标签**（memory vs compute vs preshuffle vs async）。
2. 选 **partitioner**（通用 2D → 多 die 局部性不佳时再换 **SpatiallyLocal**）。
3. 从 **§3 命名配置** 中选最近邻 **block/warp** 形状。
4. 在 **block_m、block_n**（及允许的 K tile）上做 **小范围网格搜索**，并固定 **流水线深度** 与 **wave 配置** 再比较。
5. 用 **rocprof / RGP** 等确认 **LDS、VGPR、带宽与 MFMA** 是否匹配预期。

## 相关文档

- 分层模型与类型链：[[ck-programming-model.md|ck-programming-model]]
- 推理侧封装：[[aiter-ops-reference.md|aiter-ops-reference]]
