# 内存指令参考

## Global Memory

| 指令 | 宽度 | 说明 |
|------|------|------|
| global_load_dword | 4B | 单 lane |
| global_load_dwordx2 | 8B | 向量化 |
| global_load_dwordx4 | 16B | 合并访问最佳 |
| global_store_dword[x2/x4] | 4-16B | 同样宽度 |

**合并规则**：64 个连续 lane 访问 64 个连续元素 = 一次合并事务。步长 > 1 会降低带宽。

## LDS（Local Data Share）

| 指令 | 宽度 | 说明 |
|------|------|------|
| ds_read_b32 | 4B | 单 bank |
| ds_read_b64 | 8B | 两个 bank |
| ds_read_b128 | 16B | 四个 bank |
| ds_write_b32/b64/b128 | 4-16B | 同样宽度 |
| ds_swizzle_b32 | 4B | 无 LDS 读写的 lane 排列 |

**Bank 数量与容量、读带宽（代际差异，CDNA3/4 白皮书）**：

| 项目 | CDNA3 | CDNA4 |
|------|-------|-------|
| LDS 容量/CU | 约为 CDNA4 的 **一半**（相对 CDNA4 **160 KB/CU**） | **160 KB/CU** |
| Bank 数 | **32** | **64**（相对 CDNA3 **翻倍**） |
| LDS 读带宽 | **128 B/clock**（约为 CDNA4 一半） | **256 B/clock**（容量与读带宽相对 CDNA3 均为约 **2×**） |

- **CDNA3（如 MI300X / gfx942）**：**32-bank LDS**，每 bank **4B**。
- **CDNA4（如 MI355X / gfx950）**：**64-bank LDS**。**Bank conflict 模式与 padding 计算与 CDNA3 不同**，勿沿用 32-bank 假设下的固定 padding；迁移到 CDNA4 时需按 **64 bank** 重新推导 swizzle / pad。
- **Direct LDS load（CDNA4 新路径）**：CDNA4 上 LDS 可经 **direct load** 从 **L1 data cache** 直接装填，减少经向量路径迂回的访存开销；调优时可与 MFMA、软件流水及 `buffer_load_lds` / DME 路径对照 profiling。

**L1 / L2 缓存（CDNA3 与 CDNA4 一致处及带宽口径）**：

- **L1（data）**：**32 KB**，**128 B** cache line，**64-way** 组相联（CDNA3 / CDNA4 相同）。
- **L2（每 XCD）**：**4 MB**，**16-way**，**16 channels**；每 channel 每周期 **读 128 B、写 64 B**；**每 XCD L2 读带宽约 2 KB/clock**。
- **L2 聚合读带宽（8 XCD 规模）**：CDNA3 约 **34.4 TB/s**；CDNA4 约 **32 TB/s**（代际间以白皮书标称为准，实际随产品与配置变化）。

**Bank conflict 规则**：同一周期内多个 lane 命中同一 bank 的不同 word → conflict → 串行化。通过 **padding**、**swizzle** 或访问模式重排缓解。

## Data Movement Engine（DME）

**CDNA3 / CDNA4** 配备 **Data Movement Engine（DME）**，用于 **HBM → LDS** 的异步搬运路径（与向量核上的显式 global load 相配合）。

- **减轻 VMEM 压力**：部分数据路径可走 DME，减少对 `global_load` / flat load 的争抢。
- **计算与数据搬运重叠**：与 MFMA、LDS 流水线及 `buffer_load_lds` 等机制结合，便于做 **double buffering** 与软件流水。

调优时若瓶颈在 VMEM，可查阅目标 ISA / ROCm 文档中 **async DMA / buffer_load_lds** 与 DME 相关约束，并在 profiler 中对比 VMEM vs LDS 时间线。

## Buffer vs Flat 指令

| 类型 | 使用场景 |
|------|---------|
| `buffer_load_*` | 已知 base + offset，支持范围检查，略快 |
| `global_load_*` (flat) | 任意 64 位地址，codegen 更简单 |

编译器通常自动选择。内联汇编时，base 地址为统一值（SGPR）时优先使用 buffer 指令。

## 内存屏障与同步

### s_waitcnt — 关键指令

```
s_waitcnt vmcnt(N) lgkmcnt(M) expcnt(K)
```

| 计数器 | 追踪内容 | 等待条件 |
|--------|---------|---------|
| vmcnt | Global load/store | 剩余 N 个未完成 VMEM 操作 |
| lgkmcnt | LDS + SMEM 操作 | 剩余 M 个未完成 LDS/SMEM 操作 |
| expcnt | Export (GDS, LDS→VGPR) | 剩余 K 个未完成 export |

**策略**：不要到处使用 `s_waitcnt 0`（会杀死 ILP）。计算未完成的操作数，只等待真正需要的。

```
global_load_dwordx4 v[0:3], ...    // vmcnt = 1
global_load_dwordx4 v[4:7], ...    // vmcnt = 2
// ... 做其他工作 ...
s_waitcnt vmcnt(1)                  // 只等第一个 load
v_add_f32 v8, v0, v1               // 使用第一个 load 的结果
s_waitcnt vmcnt(0)                  // 现在等第二个 load
v_add_f32 v9, v4, v5               // 使用第二个 load 的结果
```
