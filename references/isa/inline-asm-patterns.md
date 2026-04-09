# HIP 内联汇编常用 Pattern

## 何时使用内联汇编

1. 编译器未能生成最优指令（用 `-save-temps` 验证）
2. 需要 intrinsics 未暴露的特定指令
3. 关键内层循环的最后手段 — 优先使用 `__builtin_amdgcn_*`

## 语法

```cpp
asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
```

约束代码：

- `v` = VGPR, `s` = SGPR, `a` = AGPR
- `=` = 输出, 无前缀 = 输入

## Builtin → ISA 映射

| Builtin | ISA 指令 | 用途 |
|---------|---------|------|
| `__builtin_amdgcn_readfirstlane(v)` | v_readfirstlane_b32 | 将第一个 lane 广播到 SGPR |
| `__builtin_amdgcn_ds_swizzle(v, pat)` | ds_swizzle_b32 | Lane 排列（无 LDS 流量） |
| `__builtin_amdgcn_mov_dpp(v, ctrl, ...)` | v_mov_b32 dpp | 数据并行原语 |
| `__shfl_sync` 等价物 | ds_swizzle / dpp | 跨 lane 通信 |

## 常见优化 Pattern

### 1. 向量化 Global Load

```cpp
// 强制 128 位 load（一次 4 个 float）
float4 data;
asm volatile(
    "global_load_dwordx4 %0, %1, off"
    : "=v"(data) : "v"(addr)
);
```

### 2. LDS Swizzle 消除 Bank Conflict

```cpp
// 矩阵转置时避免 bank conflict 的 swizzle pattern
int swizzled = __builtin_amdgcn_ds_swizzle(val, 0x041f);
```

### 3. 内层循环手动 MFMA

```cpp
// 当编译器 MFMA 调度不理想时
asm volatile(
    "v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
    : "+a"(acc)  // AGPR 累加器（读写）
    : "v"(a_frag), "v"(b_frag)
);
```

### 4. 精确 s_waitcnt

```cpp
// 等待恰好 1 个未完成的 VMEM 操作
asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
```

---

## 调度与优先级 Builtins（CK / 博文）

### `__builtin_amdgcn_s_setprio(N)` — Wave 优先级

- **语义**：设置当前 wave 的调度优先级，`N ∈ [0, 3]`。
- **惯例**：**0** = 低优先级，**3** = 高优先级；在 CU 资源争用时，高优先级 wave 更容易被选中。
- **用途**：8-wave ping-pong 中在 **memory phase** 短暂 `s_setprio(1)`（或更高），compute phase 回到 `0`，与 barrier 配合重叠访存与 MFMA。

```cpp
__builtin_amdgcn_s_setprio(1);
// issue async loads / buffer_load_lds ...
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
```

### `__builtin_amdgcn_sched_barrier(mask)` — 调度栅栏

- **语义**：限制编译器/硬件对某些指令类型的跨栅栏重排。
- **`mask == 0`**：**硬栅栏**，不允许任意指令越过该点重排（与 CK 中 `sched_barrier(0)` 用法一致）。
- **非零 mask**：仅允许**掩码对应类别**的指令越过（具体行为以编译器与架构文档为准；常用于与 MFMA/SALU 交错）。

**与 `sched_group_barrier` 的分工**：在 CK 与 LLVM 暴露的接口里，**按指令类计数发射**（例如「再发 1 组 MFMA」「再发 1 组 SALU」）通常写作 **`__builtin_amdgcn_sched_group_barrier(mask, count, ds_id)`**，其中掩码与 `LLVMSchedGroupMask` 一致：

- **`0x008`**：MFMA 类
- **`0x004`**：SALU 类

纯 **`sched_barrier(0)`** 则用于 **完全禁止重排** 的同步点；不要把「只放行 MFMA」与「硬栅栏」混用同一调用，需对照生成的 ISA。

### `__builtin_amdgcn_sched_group_barrier(mask, count, ds_id)` — 分组调度

- **语义**：按**指令组**发射：在发出 `count` 个属于 `mask` 类型的指令组之前，限制其它组的乱序跨越。
- **示例**（CK eight-wave hot loop）：连续三次 `sched_group_barrier(0x008, 1, 0)` 约束 MFMA 发射节奏，再 `sched_group_barrier(0x004, 1, 0)` 与 `s_waitcnt` 包装的 LDS 行为交错，最后 `sched_barrier(0)` 硬栅栏。

```cpp
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // issue 1 MFMA group
__builtin_amdgcn_sched_group_barrier(0x004, 1, 0);  // issue 1 SALU group
__builtin_amdgcn_sched_barrier(0);
```

`ds_id` 参数在 CK 中常取 **0**；多 DS 队列场景见 ISA/编译器说明。

---

## buffer_load_lds（Global → LDS）内联路径

### LLVM intrinsic（推荐在 HIP 中外链声明）

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr,
    int size, int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
```

### 手写 asm：`buffer_load_dwordx4 ... lds`（gfx950 等）

数据直接进入 LDS，output 操作数为 LDS 指针表示目标（由编译器约束为 `=r`/`smem`）：

```cpp
asm volatile("buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
             : "=r"(smem_ptr)
             : "v"(voffset), "s"(rsrc), "n"(ioffset)
             : "memory");
```

配合 **128-bit buffer resource**（`make_wave_buffer_resource` / `buffer_resource` 结构体 bitcast 为 `int32x4`）使用。详见 `references/kernel-recipes.md` 中 buffer_load_lds 节。

---

## 架构自适应 waitcnt（CK 思路）

gfx9/gfx11 使用 `s_waitcnt` 合并域；gfx12 等可能拆分 load/ds 计数。CK 用模板封装统一接口，避免手写错误编码：

```cpp
// 典型调用（伪代码）：在 global→LDS 后等待 VMEM，再 lgkm
// s_waitcnt<vmcnt, expcnt, lgkmcnt>();
// gfx12: 可能变为 s_wait_loadcnt_dscnt + barrier_signal / barrier_wait
```

**实践**：在移植 kernel 时，用 `__builtin_amdgcn_s_waitcnt` 与项目里的架构宏（`__gfx12__` 等）分支；不要在未测架构上硬编码单一 asm。

---

## 决策指南

| 场景 | 方法 |
|------|------|
| 需要跨 lane 操作 | 先尝试 `__builtin_amdgcn_*` |
| 编译器生成了次优 load | 检查 `-O3` 是否修复，否则尝试 asm |
| MFMA 调度不正确 | 先 profile，确认后再用手动 asm；并对照 `scheduling-pipeline.md` MFMA NOP 表 |
| 需要精确的等待计数 | 内联 `s_waitcnt` 或 CK 式模板；gfx12 查 loadcnt/dscnt |
| Global→LDS 直达 | `llvm.amdgcn.raw.buffer.load.lds` 或 `buffer_load_* ... lds` |
| Wave 间抢占 LDS/VM 带宽 | `s_setprio` + `sched_barrier` / `sched_group_barrier` |
