# Kernel 参考实现（生产级模式摘录）

本文档从 CK、AITER、ROCm CDNA4 FP8 GEMM 博文等来源整理**可落地的关键内层循环与 API 用法**，供 agent 直接对照实现与调优。正文为中文，代码与标识符为 English。

> **说明**：下列片段为教学/移植用「关键路径」代码；完整可编译工程需补齐类型、`__launch_bounds__`、grid 划分与边界检查。性能数字来自博文/厂商基准（M=N=K=4096，CDNA4），实际以 profile 为准。

---

## 目录

1. [Vector Add（HIP — 最小参考）](#vector-addhip--最小参考)
2. [FP8 GEMM with MFMA（LDS 分块 + 向量化 load + double buffer）](#fp8-gemm-with-mfma)
3. [buffer_load_lds：Global→LDS 直达](#buffer_load_ldsgloballds-直达)
4. [LDS XOR Swizzle（消除 bank conflict）](#lds-xor-swizzle消除-bank-conflict)
5. [8-Wave Ping-Pong 调度](#8-wave-ping-pong-调度)
6. [RMSNorm（AITER Triton，persistent + blocked）](#rmsnormaiter-tritonpersistent--blocked)
7. [Fused MoE + SiLU（AITER Triton）](#fused-moe--siluaiter-triton)
8. [Wavefront-aware Reduction（含 AGPR 注记）](#wavefront-aware-reduction含-agpr-注记)

---

## Vector Add（HIP — 最小参考）

保留为最小 HIP kernel 示例，不涉及矩阵核心。

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

**优化关键词**：coalesced global access，occupancy。

---

## FP8 GEMM with MFMA

**目标**：在 CDNA4 上使用 `fp8` 输入、`f32` 累加，通过 LDS 分块、**16 字节/线程**向量化 load、双缓冲与 MFMA 饱和算力。

**所用指令**：`__builtin_amdgcn_mfma_f32_16x16x128_fp8_fp8`（每 wave 一次处理 K=128 的 fp8 块，与布局约定一致时使用）。

**关键技术**：LDS tiling；**vectorized fp8×16 load**（`uint4`/`vector_size(16)`）；**double-buffer LDS**（`cur`/`nxt` 乒乓）；可选 **XOR swizzle**、**buffer_load_lds**、**8-wave ping-pong**（见后文）。

### 向量化 FP8×16 load（Global→寄存器→LDS 或直写 LDS 前的寄存器侧）

```cpp
using fp8_t = _Float16;  // placeholder: use ROCm fp8e4m3 type in project
using fp8x16_t = __attribute__((vector_size(16))) char;

static inline __device__ fp8x16_t load_fp8x16_u4(const char* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    return *reinterpret_cast<const fp8x16_t*>(&v);
}

// Example: fill a LDS tile row with 16 fp8 per vector op
// const fp8x16_t a_vec = load_fp8x16_u4(A_base + row * lda + k0);
```

### LDS 双缓冲骨架（与 MFMA 内层交错）

```cpp
// Ping-pong: two slots for A and B tiles
// __shared__ fp8 A_lds[2][TILE_M][TILE_K];
// __shared__ fp8 B_lds[2][TILE_N][TILE_K];
int cur = 0, nxt = 1;

// Prologue: prefetch tile 0
// prefetch_tile_to_lds(A_lds[cur], B_lds[cur], 0);
// wait_for_global_loads();
// block_sync();

// for (int t = 0; t < num_k_tiles; ++t) {
//     if (t + 1 < num_k_tiles)
//         prefetch_tile_to_lds_async(A_lds[nxt], B_lds[nxt], t + 1);
//     // fragments from A_lds[cur], B_lds[cur] with swizzle-aware addressing
//     acc = mfma_fp8(acc, frag_a, frag_b);  // maps to __builtin_amdgcn_mfma_f32_16x16x128_fp8_fp8
//     if (t + 1 < num_k_tiles) {
//         wait_for_global_loads();
//         block_sync();
//         cur ^= 1;
//         nxt ^= 1;
//     }
// }
```

### 关键 MFMA 调用（编译器内置）

内层在布局满足 CK/博文 lane 映射时，累加器与 A/B 向量由编译器绑定到 MFMA。典型形式（参数依子块布局而定，见 LLVM/ROCm 文档）：

```cpp
using fp8x16 = char __attribute__((vector_size(16)));
using float4 = float __attribute__((vector_size(16)));

__device__ inline float4 mfma_fp8_16x16x128(
    fp8x16 a, fp8x16 b, float4 acc) {
    return __builtin_amdgcn_mfma_f32_16x16x128_fp8_fp8(
        a, b, acc, /*cbsz*/ 0, /*abid*/ 0, /*blgp*/ 0);
}
```

**性能标注（博文，M=N=K=4096）**：朴素 ~1.15 TFLOP/s → LDS tile ~4.8 → MFMA+向量化 load ~337 → +Global→LDS ~507 → +swizzle/double buffer ~1166 → 8-wave 最优 ~2288 TFLOP/s（与 tile 与占用强相关）。

**优化技术名称**：MFMA FP8 矩阵核心；LDS blocking；software pipelining（double buffer）；vectorized global load；bank conflict avoidance（XOR swizzle）；multi-wave ping-pong。

---

## buffer_load_lds：Global→LDS 直达

**要点**：使用 `llvm.amdgcn.raw.buffer.load.lds` 把数据从 global **直接写入 LDS**，减少 VGPR 压力（不经全向量寄存器文件搬运）。CDNA4（gfx950）上每 lane 可达 128-bit；旧架构每 lane 可能仅 32-bit，需查 ISA/CK 分支。

### LLVM intrinsic 声明

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc,
    as3_uint32_ptr lds_ptr,
    int size,      // bytes per lane: 4, 8, 12, 16
    int voffset,
    int soffset,
    int offset,
    int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
```

### `make_wave_buffer_resource()`（CK 风格）

```cpp
struct __attribute__((packed)) buffer_resource {
    const void* ptr;
    uint32_t range;
    uint32_t config;  // e.g. CK_TILE_BUFFER_RESOURCE_3RD_DWORD
};

__device__ inline i32x4 make_wave_buffer_resource(
    const void* ptr, uint32_t size = 0xffffffffu) {
    buffer_resource res{ptr, size, 0x00020000u};  // example config dword
    return *reinterpret_cast<i32x4*>(&res);
}
```

### Inline asm：`buffer_load_dwordx4 ... lds`（gfx950）

```cpp
template <unsigned num_dwords>
__device__ void async_buffer_load_dwordxn_lds(
    void* smem, i32x4 rsrc, int voffset, int ioffset) {
    static_assert(num_dwords == 1 || num_dwords == 4);
    if constexpr (num_dwords == 4) {
        asm volatile("buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    } else {
        asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    }
}
```

**高层调用（逻辑）**：

```cpp
// llvm_amdgcn_raw_buffer_load_lds(rsrc, (as3_uint32_ptr)smem, bytes, v_offset, soffset, 0, coherence);
```

**优化技术名称**：raw buffer load；global-to-LDS shortcut；降低 VGPR 占用。

---

## LDS XOR Swizzle（消除 bank conflict）

**要点**：CDNA4 上 `ds_read_b128` 等宽向量读在 64 bank 上分多 phase；对 **16×128** 一类 tile，对列索引做 **XOR remap**，使同 phase 内线程访问不同 bank。映射为**自逆**，读写用同一公式即可。

### XOR 公式（16B 列对齐场景，来自博文/CK 摘录）

```cpp
__device__ __host__ int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}
// LDS write:  addr = base + row * stride_row + swizzle_col(row, col);
// LDS read:   same formula on (row, logical_col)
```

### CK 中的 SwizzleA / SwizzleB 类型（概念）

CK 在 `WarpGemm` 层用模板区分是否在 K 维迭代上启用 **SwizzleA** 或 **SwizzleB**（及 `TransposedCDistribution`），例如：

- `WarpGemmMfmaF16F16F32M32N32K8SwizzleA` — A 侧 swizzle。
- `WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<4>` — B 侧 swizzle + factor=4（FP8 32×32×32）。

**优化技术名称**：XOR bank swizzle；phase-balanced LDS access；SwizzleA/SwizzleB policy。

---

## 8-Wave Ping-Pong 调度

**要点**：8 个 wave 分成两组（`wave_m = waveid / 4`），通过 **barrier** 与 **优先级** 让一组做 **memory**，另一组做 **compute**，重叠 MFMA 与异步搬运。

### `s_setprio` / `sched_barrier` / `sched_group_barrier`

```cpp
// Priority: 0 = low, 3 = high (wave priority under CU contention)
__builtin_amdgcn_s_setprio(1);
// ... memory phase loads ...
__builtin_amdgcn_sched_barrier(0);   // hard fence: no reorder across
__builtin_amdgcn_s_setprio(0);

// Fine-grained: issue groups — 0x008 = MFMA class, 0x004 = SALU
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // one MFMA group
__builtin_amdgcn_sched_group_barrier(0x004, 1, 0);  // one SALU group
```

### CK hot-loop scheduler 片段（概念）

```cpp
auto hot_loop_scheduler = [&]() {
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
    // s_waitcnt_lgkm<4>();  // keep LDS loads in flight — CK helper
    __builtin_amdgcn_sched_group_barrier(0x004, 1, 0);
    // ... more MFMA sched_group_barrier(0x008,1,0) ...
    __builtin_amdgcn_sched_barrier(0);
};
```

### Wave 分组与 barrier 交错（博文风格）

```cpp
int waveid = threadIdx.x / 64;
int wave_m = waveid / 4;   // 0 or 1: two groups
// int wave_n = waveid % 4;

if (wave_m == 1)
    __builtin_amdgcn_s_barrier();  // stagger: one group waits first
// ... block 1: group 0 memory while group 1 compute ...
__builtin_amdgcn_s_barrier();
// ... swap roles ...
```

**优化技术名称**：wave priority (`s_setprio`)；scheduling barriers；MFMA/SALU/LDS 交错；8-wave occupancy pattern。

---

## RMSNorm（AITER Triton：persistent + blocked）

**要点**：`tl.range(row_start, n_rows, NUM_PRGMS)` 做 **persistent** 调度；`USE_BLOCKED` 为大 `n_cols` 多 block 两遍扫描；小 `n_cols` 单 block 一遍 `rsqrt` + scale。

```python
@triton.jit
def _rms_norm_kernel(
    input_ptr, output_ptr, g_ptr, rsigma_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = tl.multiple_of(row_input_ptr + cols, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=0.0,
                        cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)
            norm_factor = tl.rsqrt(sum_squares / n_cols + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                x = tl.load(tl.multiple_of(row_input_ptr + cols, (16,))).to(tl.float32)
                g = tl.load(g_ptr + cols).to(tl.float32)
                tl.store(row_output_ptr + cols,
                         (x * norm_factor * g).to(output_ptr.type.element_ty))
            # remainder column block omitted for brevity
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets,
                          mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            norm_factor = tl.math.rsqrt(tl.sum(row * row, axis=-1) / n_cols + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)
            rms_norm = row * norm_factor * g
            tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
                     rms_norm.to(output_ptr.type.element_ty), mask=mask)
```

**性能**：依赖序列长度与宽度；优化来自 persistent grid、`tl.multiple_of(..., (16,))` 向量化与两路径分支。

**优化技术名称**：persistent scheduling；blocked reduction；two-pass normalize；cache modifier `.cg`。

---

## Fused MoE + SiLU（AITER Triton）

**要点**：`remap_xcd` 重映射 `program_id` 改善跨 chiplet 的 L2 局部性；`pid_grid` + `GROUP_SIZE_M` 控制 tile 遍历；SiLU 路径对 gate/up **列交错**（`offs_bn`）。下列为 **PID 映射 + 列寻址 + GEMM 累加 + SiLU + store** 核心；`a_ptrs` / `b_ptrs` / `c_ptrs` 及 INT4/INT8/FP8 解包见 `aiter/.../moe_op_silu_fused.py` 全文。

```python
@triton.jit
def _fused_moe_silu_kernel_gptq_awq(
    a_ptr, b_ptr, c_ptr, b_scale_ptr, b_zp_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, num_valid_tokens,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_SIZE_M, MUL_ROUTED_WEIGHT, top_k,
    NUM_XCDS: tl.constexpr,
    compute_type: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_n * num_pid_m

    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    offs_bn = (offs_half + (i % 2) * (N // 2)) % N

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Pointers for A/B/C tiles: expert table + offs_bn + quant metadata (full source)
    a_ptrs = a_ptr  # + expert/offset arithmetic in production
    b_ptrs = b_ptr
    c_ptrs = c_ptr

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = _silu_exp2(accumulator)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator *= moe_weight[:, None]

    tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None])
```

**所用指令层**：Triton `tl.dot` 降至后端 MFMA（类型依赖）。

**优化技术名称**：XCD remap；L2-friendly tile ordering；fused SiLU；quantized GEMM（INT4/INT8/FP8 路径在完整源码中）。

---

## Wavefront-aware Reduction（含 AGPR 注记）

AMD wavefront = 64 lanes；warp shuffle 常用 `__shfl_xor`。多 wave block 需 **shared memory** 二次归约。

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__global__ void block_reduce(const float* input, float* output, int N) {
    __shared__ float shared[16];
    int tid = threadIdx.x;
    int wf_id = tid / 64;
    int lane = tid % 64;

    float val = (blockIdx.x * blockDim.x + tid < N)
                ? input[blockIdx.x * blockDim.x + tid] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wf_id] = val;
    __syncthreads();

    if (wf_id == 0 && lane < (blockDim.x / 64)) {
        val = shared[lane];
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}
```

**AGPR 注记**：在 **MFMA 累加链**中，生产库常把累加器放在 **AGPR**（`asm` 中 `"a"` 约束或 CK `DISPATCH_MFMA_` 的 `+a`），与 VGPR 分离，便于双缓冲与 **8-wave** 方案中一组 wave 用 VGPR、另一组用 AGPR 交错。纯 reduction 如上通常仅用 VGPR；若与 MFMA 同 kernel 融合，需对照 ISA 的 **SrcC 转发与 NOP 间隔**（见 `isa/scheduling-pipeline.md`）。

**优化技术名称**：wave shuffle reduction；hierarchical block reduce；AGPR accumulator（与 MFMA 联用时）。

---

## 参考与延伸阅读

- CK：`composable_kernel` — `amd_buffer_addressing.hpp`，`gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp`。
- AITER：`rmsnorm.py`，`moe_op_silu_fused.py`。
- ROCm Blog：[FP8 GEMM Optimization on AMD CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
- 更细 MFMA 调度与 NOP：`references/isa/scheduling-pipeline.md`；内联封装：`references/isa/inline-asm-patterns.md`。
