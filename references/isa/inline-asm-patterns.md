# HIP Inline Assembly Common Patterns

## When to Use Inline Assembly

1. The compiler fails to generate optimal instructions (verify with `-save-temps`)
2. You need specific instructions not exposed by intrinsics
3. Last resort for critical inner loops -- prefer `__builtin_amdgcn_*` first

## Syntax

```cpp
asm volatile("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
```

Constraint codes:

- `v` = VGPR, `s` = SGPR, `a` = AGPR
- `=` = output, no prefix = input

## Builtin -> ISA Mapping

| Builtin | ISA Instruction | Purpose |
|---------|----------------|---------|
| `__builtin_amdgcn_readfirstlane(v)` | v_readfirstlane_b32 | Broadcast first lane to SGPR |
| `__builtin_amdgcn_ds_swizzle(v, pat)` | ds_swizzle_b32 | Lane permutation (no LDS traffic) |
| `__builtin_amdgcn_mov_dpp(v, ctrl, ...)` | v_mov_b32 dpp | Data parallel primitive |
| `__shfl_sync` equivalent | ds_swizzle / dpp | Cross-lane communication |

## Common Optimization Patterns

### 1. Vectorized Global Load

```cpp
// Force 128-bit load (4 floats at once)
float4 data;
asm volatile(
    "global_load_dwordx4 %0, %1, off"
    : "=v"(data) : "v"(addr)
);
```

### 2. LDS Swizzle to Eliminate Bank Conflicts

```cpp
// Swizzle pattern to avoid bank conflicts during matrix transpose
int swizzled = __builtin_amdgcn_ds_swizzle(val, 0x041f);
```

### 3. Manual MFMA in Inner Loop

```cpp
// When compiler MFMA scheduling is suboptimal
asm volatile(
    "v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
    : "+a"(acc)  // AGPR accumulator (read-write)
    : "v"(a_frag), "v"(b_frag)
);
```

### 4. Precise s_waitcnt

```cpp
// Wait for exactly 1 outstanding VMEM operation
asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
```

---

## Scheduling and Priority Builtins (CK / Blog Posts)

### `__builtin_amdgcn_s_setprio(N)` -- Wave Priority

- **Semantics**: Sets the scheduling priority of the current wave, `N in [0, 3]`.
- **Convention**: **0** = low priority, **3** = high priority; when CU resources are contended, higher-priority waves are more likely to be selected.
- **Use case**: In 8-wave ping-pong, briefly `s_setprio(1)` (or higher) during the **memory phase**, return to `0` during the compute phase, combined with barriers to overlap memory access and MFMA.

```cpp
__builtin_amdgcn_s_setprio(1);
// issue async loads / buffer_load_lds ...
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
```

### `__builtin_amdgcn_sched_barrier(mask)` -- Scheduling Fence

- **Semantics**: Restricts compiler/hardware reordering of certain instruction types across the fence.
- **`mask == 0`**: **Hard fence** -- no instructions of any type may be reordered past this point (consistent with CK's `sched_barrier(0)` usage).
- **Non-zero mask**: Only allows instructions **matching the mask categories** to cross (specific behavior depends on compiler and architecture documentation; commonly used for interleaving with MFMA/SALU).

**Division of labor with `sched_group_barrier`**: In the interfaces exposed by CK and LLVM, **counted dispatch by instruction category** (e.g., "issue 1 more MFMA group", "issue 1 more SALU group") is typically written as **`__builtin_amdgcn_sched_group_barrier(mask, count, ds_id)`**, where the mask aligns with `LLVMSchedGroupMask`:

- **`0x008`**: MFMA category
- **`0x004`**: SALU category

A plain **`sched_barrier(0)`** is used for **completely disabling reordering** at synchronization points; do not mix "only allow MFMA through" with "hard fence" in the same call -- verify against the generated ISA.

### `__builtin_amdgcn_sched_group_barrier(mask, count, ds_id)` -- Group Scheduling

- **Semantics**: Dispatch by **instruction group**: before issuing `count` instruction groups of the type specified by `mask`, restrict other groups from reordering across.
- **Example** (CK eight-wave hot loop): Three consecutive `sched_group_barrier(0x008, 1, 0)` constrain MFMA dispatch cadence, then `sched_group_barrier(0x004, 1, 0)` interleaves with `s_waitcnt`-wrapped LDS behavior, followed by a `sched_barrier(0)` hard fence.

```cpp
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // issue 1 MFMA group
__builtin_amdgcn_sched_group_barrier(0x004, 1, 0);  // issue 1 SALU group
__builtin_amdgcn_sched_barrier(0);
```

The `ds_id` parameter is commonly **0** in CK; for multi-DS-queue scenarios, see ISA/compiler documentation.

---

## buffer_load_lds (Global -> LDS) Inline Path

### LLVM intrinsic (recommended to declare as extern in HIP)

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr,
    int size, int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
```

### Hand-written asm: `buffer_load_dwordx4 ... lds` (gfx950, etc.)

Data goes directly into LDS; the output operand is an LDS pointer indicating the destination (constrained by the compiler as `=r`/`smem`):

```cpp
asm volatile("buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
             : "=r"(smem_ptr)
             : "v"(voffset), "s"(rsrc), "n"(ioffset)
             : "memory");
```

Used with **128-bit buffer resource** (`make_wave_buffer_resource` / `buffer_resource` struct bitcast to `int32x4`). See the buffer_load_lds section in `references/kernel-recipes.md` for details.

---

## Architecture-Adaptive waitcnt (CK Approach)

gfx9/gfx11 use `s_waitcnt` with merged domains; gfx12 and later may split load/ds counters. CK uses template wrappers for a unified interface to avoid hand-writing incorrect encodings:

```cpp
// Typical call (pseudocode): after global->LDS, wait for VMEM, then lgkm
// s_waitcnt<vmcnt, expcnt, lgkmcnt>();
// gfx12: may become s_wait_loadcnt_dscnt + barrier_signal / barrier_wait
```

**Practice**: When porting kernels, use `__builtin_amdgcn_s_waitcnt` with architecture macros in the project (`__gfx12__`, etc.) to branch; do not hard-code a single asm on untested architectures.

---

## Decision Guide

| Scenario | Approach |
|----------|----------|
| Need cross-lane operations | Try `__builtin_amdgcn_*` first |
| Compiler generates suboptimal loads | Check if `-O3` fixes it, otherwise try asm |
| MFMA scheduling is incorrect | Profile first, confirm, then use manual asm; cross-reference the MFMA NOP table in `scheduling-pipeline.md` |
| Need precise wait counts | Inline `s_waitcnt` or CK-style templates; for gfx12, check loadcnt/dscnt |
| Global->LDS direct path | `llvm.amdgcn.raw.buffer.load.lds` or `buffer_load_* ... lds` |
| Waves competing for LDS/VM bandwidth | `s_setprio` + `sched_barrier` / `sched_group_barrier` |
