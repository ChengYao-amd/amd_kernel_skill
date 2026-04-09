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

## 决策指南

| 场景 | 方法 |
|------|------|
| 需要跨 lane 操作 | 先尝试 `__builtin_amdgcn_*` |
| 编译器生成了次优 load | 检查 `-O3` 是否修复，否则尝试 asm |
| MFMA 调度不正确 | 先 profile，确认后再用手动 asm |
| 需要精确的等待计数 | 内联 `s_waitcnt` 并指定确切计数 |
