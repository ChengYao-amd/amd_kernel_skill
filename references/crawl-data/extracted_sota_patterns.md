# AMD GPU Kernel Code Patterns — Extracted from Production Sources

> 从 CK (composable_kernel)、AITER、ROCm Blog 中提取的可运行级别代码模式。
> 每个代码段均标注原始文件路径和所用优化技术。

---

## 目录

1. [Task 1: CK Source Patterns](#task-1-ck-source-patterns)
   - 1.1 buffer_load_lds (Global→LDS Direct Load)
   - 1.2 LDS Swizzle / Bank Conflict Elimination
   - 1.3 s_setprio / sched_barrier Scheduling Control
   - 1.4 Double Buffer / Ping-Pong LDS
   - 1.5 MFMA Register Layout Helpers
2. [Task 2: AITER Patterns](#task-2-aiter-patterns)
   - 2.1 RMSNorm Triton Kernel
   - 2.2 Fused MoE SiLU Kernel
   - 2.3 MLA Decode with RoPE
3. [Task 3: FP8 GEMM Blog (ROCm CDNA4)](#task-3-fp8-gemm-blog)
   - 3.1 Naive Baseline
   - 3.2 LDS Tiling
   - 3.3 MFMA Matrix Core + Vectorized Load
   - 3.4 Global→LDS Direct Load (buffer_load_lds)
   - 3.5 LDS Swizzle for Bank Conflict Elimination
   - 3.6 Double-Buffer Software Pipelining
   - 3.7 8-Wave Ping-Pong Scheduling

---

## Task 1: CK Source Patterns

### 1.1 buffer_load_lds — Global→LDS Direct Load

**源文件**: `composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing.hpp`

**技术要点**: 绕过 VGPR，直接从 Global Memory 搬运到 LDS，节省寄存器压力。CDNA4 (gfx950) 支持 128-bit/lane，CDNA3 仅 32-bit/lane。

#### 1.1a 底层 LLVM intrinsic 声明

```cpp
// File: ck_tile/core/arch/amd_buffer_addressing.hpp:1367-1375
// Direct loads from global to LDS — the raw LLVM intrinsic.
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,        // 128-bit buffer resource descriptor (SGPR)
                                as3_uint32_ptr lds_ptr, // LDS target address (address_space(3))
                                index_t size,           // bytes per lane: 4 (dword), 8, 12, 16
                                index_t voffset,        // per-lane offset (VGPR)
                                index_t soffset,        // wave-uniform offset (SGPR)
                                index_t offset,         // immediate offset
                                index_t aux)            // coherence flags
    __asm("llvm.amdgcn.raw.buffer.load.lds");
```

#### 1.1b Buffer Resource 构造函数

```cpp
// File: ck_tile/core/arch/amd_buffer_addressing.hpp:101-120
// 128-bit SGPRs supply buffer resource for buffer instructions.
struct __attribute__((packed)) buffer_resource
{
    const void* ptr;    // base address (64-bit)
    uint32_t range;     // valid byte range
    uint32_t config;    // CK_TILE_BUFFER_RESOURCE_3RD_DWORD (typically 0x00020000)
};

template <typename ForceSGPR = std::false_type>
CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr,
                                                   uint32_t size = 0xffffffff,
                                                   ForceSGPR     = {})
{
    buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    if constexpr(std::is_same_v<ForceSGPR, std::true_type>)
    {
        r = amd_wave_read_first_lane(r);   // hoist to SGPR
    }
    return r;
}
```

#### 1.1c Async buffer_load_dwordxN with LDS flag

```cpp
// File: ck_tile/core/arch/amd_buffer_addressing.hpp:1377-1417
// Inline asm wrapper — appends "lds" suffix to route data directly to LDS.
// gfx950 supports dwordx3 and dwordx4 (12/16 bytes per lane).
template <unsigned num_dwords, bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dwordxn_v(void* smem,
                                                int32x4_t rsrc,
                                                index_t voffset,
                                                index_t /*soffset*/,
                                                index_t ioffset,
                                                index_t /*flag*/       = 0,
                                                bool_constant<pre_nop> = {})
{
    // Macro generates asm with optional s_nop before instruction for hazard avoidance
    if constexpr(num_dwords == 1)
    {
        // "buffer_load_dword %1, %2, 0 offen offset:%3 lds"
        asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    }
    else if constexpr(num_dwords == 4)   // gfx950 only: 16 bytes/lane
    {
        asm volatile("buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    }
}
```

#### 1.1d 高层封装调用 (amd_async_buffer_load_raw)

```cpp
// File: ck_tile/core/arch/amd_buffer_addressing.hpp:1965-1974
// High-level call used by pipeline code.
// 'smem' = LDS pointer, 'v_offset' = per-lane VGPR offset.
llvm_amdgcn_raw_buffer_load_lds(src_wave_buffer_resource,
                                (as3_uint32_ptr)(smem),
                                bytes,
                                v_offset,
                                src_wave_addr_offset,
                                /*src_immediate_addr_offset*/ 0,
                                static_cast<index_t>(coherence));
```

---

### 1.2 LDS Swizzle / Bank Conflict Elimination

**源文件**: `ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp`

**技术要点**: `ds_read_b128` 在 CDNA4 (64 banks) 上分 4 phase 执行。XOR-based swizzle 确保每个 phase 内线程访问不同 bank。

#### 1.2a Blog 中的 Swizzle 公式 (适用于 16×128 tile, CDNA4 64-bank)

```cpp
// From ROCm Blog: FP8 GEMM Optimization on CDNA4
// XOR remap on 16-byte columns. Self-inverse: apply same formula to swizzle/unswizzle.
int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}
```

#### 1.2b CK WarpGemm 中的 SwizzleA / SwizzleB 类型选择

```cpp
// File: ck_tile/ops/gemm/warp/warp_gemm.hpp:107-113
// WarpGemm with SwizzleA — used for fp16 32x32 tiles
using WarpGemmMfmaF16F16F32M32N32K8SwizzleA = WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    1>>;

using WarpGemmMfmaF16F16F32M32N32K16SwizzleA = WarpGemmImpl<WarpGemmAttributeMfmaIterateK_SwizzleA<
    WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
    2>>;
```

#### 1.2c CK FMHA Policy 中的 FP8 Swizzle 配置

```cpp
// File: ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp:276-278
// swizzle_factor=4 for fp8 warp gemm 32x32x32 — redistributes bank accesses across 4 phases
constexpr index_t swizzle_factor = 4;
return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<swizzle_factor>{};
```

---

### 1.3 s_setprio / sched_barrier Scheduling Control

**源文件**: `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp`

**技术要点**: 通过 `sched_group_barrier` 控制 MFMA、SALU、VMEM 指令交错，实现 ping-pong 调度。`sched_barrier(0)` 禁止任何指令跨越该 barrier 重排。

#### 1.3a CK 八波 Pipeline 中的 Hot Loop Scheduler

```cpp
// File: ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp:188-198
// MFMA_INST = MIterPerWarp * NIterPerWarp * KIterPerWarp
auto hot_loop_scheduler = [&]() {
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // issue 1 MFMA
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // issue 1 MFMA
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // issue 1 MFMA
    s_waitcnt_lgkm<4>();                                 // wait for LDS loads, keep 4 in flight
    __builtin_amdgcn_sched_group_barrier(0x004, 1, 0); // issue 1 SALU (lgkmcnt decrement)
    static_for<0, MFMA_INST - 3, 1>{}([&](auto) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // remaining MFMA instructions
    });
    __builtin_amdgcn_sched_barrier(0);                  // hard fence: no reordering past here
};
```

#### 1.3b sched_group_barrier Mask 值参考 (from arch.hpp)

```cpp
// File: ck_tile/core/arch/arch.hpp:1199-1214
enum LLVMSchedGroupMask : int32_t
{
    NONE       = 0,         // 0x000
    ALU        = 1 << 0,    // 0x001
    VALU       = 1 << 1,    // 0x002
    SALU       = 1 << 2,    // 0x004
    MFMA       = 1 << 3,    // 0x008
    VMEM       = 1 << 4,    // 0x010
    VMEM_READ  = 1 << 5,    // 0x020
    VMEM_WRITE = 1 << 6,    // 0x040
    DS         = 1 << 7,    // 0x080
    DS_READ    = 1 << 8,    // 0x100
    DS_WRITE   = 1 << 9,    // 0x200
    TRANS      = 1 << 10,   // 0x400
    ALL        = (TRANS << 1) - 1,
};
```

#### 1.3c Blog 中的 s_setprio / sched_barrier 用法

```cpp
// From ROCm Blog: 8-Wave Ping-Pong Scheduling
// __builtin_amdgcn_s_setprio(x): priority 0-3, higher wins on CU resource contention
// __builtin_amdgcn_sched_barrier(x): mask of instruction types allowed to cross
//   sched_barrier(0) = hard fence, no instructions may cross

// Barrier stagger for ping-pong wave separation:
int waveid = threadIdx.x / 64;
int wave_m = waveid / 4;  // 0 or 1 (two wave groups)
int wave_n = waveid % 4;  // 0..3 (four SIMDs)

// code block 0 ...
if (wave_m == 1) {
    __builtin_amdgcn_s_barrier();  // stalls waves 4-7
}
// code block 1 (waves 0-3 execute here while 4-7 stalled) ...
__builtin_amdgcn_s_barrier();      // releases 4-7, stalls 0-3
// code block 2 (waves 4-7 now running block 1, waves 0-3 start block 2) ...
```

---

### 1.4 Double Buffer / Ping-Pong LDS

**源文件**: `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_eight_waves_base.hpp`

**技术要点**: LDS 分配 `2×(A_tile + B_tile)` 两份空间。`cur ^= 1; nxt ^= 1` 乒乓切换。Prologue 预取第一个 tile，主循环中 load(t+1) 和 compute(t) 重叠执行。

#### 1.4a CK Eight-Wave Base — Ping-Pong LDS View 创建

```cpp
// File: ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_eight_waves_base.hpp:36
static constexpr index_t warp_groups = 2; // ping-pong: two buffer slots

// SmemSize = 2 * (smem_size_a + smem_size_b)  (line 126 in async variant)
```

#### 1.4b CK 中的 GlobalPrefetchAsync + Slot 切换

```cpp
// File: ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_eight_waves_base.hpp:52-59
// Async prefetch A/B tile into the given LDS slot
template <typename DataType, typename DstBlockWindow, typename SrcTileWindow>
CK_TILE_DEVICE void GlobalPrefetchAsync(DataType* smem,
                                        DstBlockWindow& dts_block_window,
                                        SrcTileWindow& dram_tile_window) const
{
    constexpr auto NEG1 = number<-1>{};
    dts_block_window.set_bottom_tensor_view_data_ptr(smem); // point to correct LDS slot
    async_load_tile(dts_block_window, dram_tile_window, NEG1, false_type{}, true_type{});
}
```

#### 1.4c Blog 中的 Double-Buffer Pseudo-code

```cpp
// From ROCm Blog: Double-Buffer Software Pipelining
LdsTile A_lds[2], B_lds[2];
int cur = 0, nxt = 1;

prefetch_tile_to_lds(A_lds[cur], B_lds[cur], /*tile=*/0);
wait_for_global_loads();
block_sync();

for (int t = 0; t < num_k_tiles; ++t) {
    if (t + 1 < num_k_tiles) {
        prefetch_tile_to_lds_async(A_lds[nxt], B_lds[nxt], /*tile=*/t + 1);
    }

    fragments_a = read_fragments_from_lds(A_lds[cur]);
    fragments_b = read_fragments_from_lds(B_lds[cur]);
    acc = mfma(acc, fragments_a, fragments_b);

    if (t + 1 < num_k_tiles) {
        wait_for_global_loads();
        block_sync();
        cur ^= 1;  // swap slots
        nxt ^= 1;
    }
}
```

---

### 1.5 MFMA Register Layout Helpers

**源文件**: `ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`

**技术要点**: 每个 MFMA instruction 有固定的 lane→matrix-element 映射。下面列出 FP8 16×16×128 和 FP32 16×16×4 / 32×32×2 的 lane 参数。

#### 1.5a F32 16×16×4 MFMA — Lane Layout Constants

```cpp
// File: ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:66-127
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF32F32F32M16N16K4
{
    using AVecType = ext_vector_t<float, 1>;
    using BVecType = ext_vector_t<float, 1>;
    using CVecType = ext_vector_t<float, 4>;    // each lane holds 4 output elements

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 4;

    // Lane mapping: 64 lanes across M and K
    static constexpr index_t kAMLane     = 16;  // A rows mapped across 16 lanes
    static constexpr index_t kBNLane     = 16;  // B cols mapped across 16 lanes
    static constexpr index_t kABKLane    = 4;   // K-dim split across 4 groups of lanes
    static constexpr index_t kABKPerLane = 1;   // each lane handles 1 K element

    // Output layout: 64 lanes → 16×16 output via (M-stripe, N-lane)
    static constexpr index_t kCMLane     = 4;   // 4 M-groups (lane_id >> 4)
    static constexpr index_t kCNLane     = 16;  // 16 N-columns (lane_id & 15)
    static constexpr index_t kCM0PerLane = 1;   // inner M per lane
    static constexpr index_t kCM1PerLane = 4;   // 4 output rows per lane (CVecType length)

    // Inline asm invocation with register class control (VGPR/AGPR)
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        // Ctrl dispatches to different register classes: vvv, vaa, vav, vva, avv
        // Default uses compiler builtin:
        c_vec = __builtin_amdgcn_mfma_f32_16x16x4f32(a_vec[0], b_vec[0], c_vec, 0, 0, 0);
    }
};
```

#### 1.5b F32 32×32×2 MFMA — Lane Layout Constants

```cpp
// File: ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:129-191
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF32F32F32M32N32K2
{
    using CVecType = ext_vector_t<float, 16>;   // each lane holds 16 output elements

    static constexpr index_t kM = 32;
    static constexpr index_t kN = 32;
    static constexpr index_t kK = 2;

    static constexpr index_t kAMLane     = 32;
    static constexpr index_t kBNLane     = 32;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 1;

    static constexpr index_t kCMLane     = 2;   // 2 M-groups
    static constexpr index_t kCNLane     = 32;  // 32 N-columns
    static constexpr index_t kCM0PerLane = 4;   // 4 blocks of 4
    static constexpr index_t kCM1PerLane = 4;   // 4 rows per block → 16 total per lane
};
```

#### 1.5c MFMA Inline Asm Dispatch Macro (VGPR/AGPR control)

```cpp
// File: ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:25-62
// 控制 C/A/B 操作数使用 VGPR(v) 还是 AGPR(a)
// 这对 8-wave ping-pong 至关重要: 一个 wave 组用 VGPR, 另一个用 AGPR
#define DISPATCH_MFMA_(mfma_, dmod_, amod_, bmod_, cmod_)       \
    asm volatile(mfma_ " %0, %1, %2, %3\n"                     \
                 : dmod_(c_vec)                                 \
                 : amod_(a_vec), bmod_(b_vec), cmod_(c_vec));

// Example: Raw_vav = C in VGPR, A in AGPR, B in VGPR
//   DISPATCH_MFMA_("v_mfma_f32_16x16x4f32", "+v", "a", "v", "v")
// Example: Raw_avv = C in AGPR, A in VGPR, B in VGPR
//   DISPATCH_MFMA_("v_mfma_f32_16x16x4f32", "+a", "v", "v", "a")
```

#### 1.5d WarpGemm Type Aliases (from warp_gemm.hpp)

```cpp
// File: ck_tile/ops/gemm/warp/warp_gemm.hpp (selected type aliases)
// FP8 is built by composing these with IterateK and SwizzleB:
using WarpGemmMfmaF16F16F32M32N32K16 = ...;  // 2× iterate of 32x32x8
using WarpGemmMfmaF16F16F32M16N16K32 = ...;  // 2× iterate of 16x16x16

// gfx950 native:
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16<>>>;
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K32<>>>;
```

#### 1.5e Blog 中的 FP8 16×16×128 Lane Mapping Code

```cpp
// From ROCm Blog: Wave-Lane Mapping for MFMA 16x16x128 FP8
const int lane = lane_in_wave;            // [0, 63]
const int row_in_tile = lane & 15;         // [0, 15]  — selects M/N row
const int row_group   = lane >> 4;         // [0, 3]   — selects 4-row output stripe

const int a_row = a_tile_row_start + row_in_tile;
const int b_row = b_tile_row_start + row_in_tile;
const int k_chunk0 = row_group * 16;       // K offsets: 0, 16, 32, 48
const int k_chunk1 = k_chunk0 + 64;        // K offsets: 64, 80, 96, 112

// Each lane reads 32 FP8 elements for A and B (two 16-element chunks)
// Each lane produces 4 FP32 accumulators:
const int output_col = output_tile_col + row_in_tile;
const int output_row_start = output_tile_row + row_group * 4;
for (int t = 0; t < 4; ++t) {
    C[output_row_start + t][output_col] = bf16(alpha * accum_fp32[t] + beta * c_old);
}
```

---

### 1.6 Waitcnt Helper (架构自适应)

**源文件**: `ck_tile/core/arch/arch.hpp`

```cpp
// File: ck_tile/core/arch/arch.hpp:1017-1063
// Portable s_waitcnt wrapper — adapts to gfx9/gfx11/gfx12 encoding differences
template <index_t vmcnt, index_t expcnt, index_t lgkmcnt>
CK_TILE_DEVICE void s_waitcnt()
{
#if defined(__gfx12__)
    constexpr index_t wait_mask = waitcnt_arg::from_vmcnt<vmcnt>() |
                                  waitcnt_arg::from_lgkmcnt<lgkmcnt>();
    asm volatile("s_wait_loadcnt_dscnt %0" : : "n"(wait_mask) : "memory");
#else
    __builtin_amdgcn_s_waitcnt(waitcnt_arg::from_vmcnt<vmcnt>() |
                               waitcnt_arg::from_expcnt<expcnt>() |
                               waitcnt_arg::from_lgkmcnt<lgkmcnt>());
#endif
}

// Barrier + waitcnt combined (gfx12 uses split barrier signals)
template <index_t vmcnt, index_t expcnt, index_t lgkmcnt>
CK_TILE_DEVICE void s_waitcnt_barrier()
{
#if defined(__gfx12__)
    asm volatile("s_wait_loadcnt_dscnt %0\n"
                 "s_barrier_signal -1\n"
                 "s_barrier_wait -1"
                 : : "n"(wait_mask) : "memory");
#else
    s_waitcnt<vmcnt, expcnt, lgkmcnt>();
    __builtin_amdgcn_s_barrier();
#endif
}

// Convenience: block_sync after direct global→LDS loads
template <index_t vmcnt = 0>
CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    s_waitcnt_barrier<vmcnt, /*expcnt_max*/, /*lgkmcnt_max*/>();
}
```

---

## Task 2: AITER Patterns

### 2.1 RMSNorm Triton Kernel (完整)

**源文件**: `aiter/aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`

**优化要点**:
- Persistent kernel (`tl.range(row_start, n_rows, NUM_PRGMS)`) — 每个 CU 处理多行
- 两种模式: `USE_BLOCKED` (大 n_cols, 多 block 累加) vs 单 block (n_cols ≤ BLOCK_SIZE)
- 两遍扫描: 第一遍累计 `sum(x²)`, 第二遍乘 `rsqrt * weight`
- `tl.multiple_of(ptr, (16,))` 对齐提示帮助编译器生成向量化 load

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

            # Pass 1: accumulate sum of squares across column blocks
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = tl.multiple_of(row_input_ptr + cols, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Remainder block
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            x = tl.load(row_input_ptr + cols, mask=mask, other=0.0,
                         cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Normalization factor
            norm_factor = tl.rsqrt(sum_squares / n_cols + epsilon)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Pass 2: normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                x = tl.load(tl.multiple_of(row_input_ptr + cols, (16,))).to(tl.float32)
                g = tl.load(g_ptr + cols).to(tl.float32)
                tl.store(row_output_ptr + cols,
                         (x * norm_factor * g).to(output_ptr.type.element_ty))
            # ... remainder handling ...

    else:
        # Small n_cols path: single block per row
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

### 2.2 Fused MoE SiLU Kernel — 核心 GEMM 循环

**源文件**: `aiter/aiter/ops/triton/_triton_kernels/moe/moe_op_silu_fused.py`

**优化要点**:
- `remap_xcd`: 跨 XCD (chiplet) 重映射 program_id，改善 L2 cache 局部性
- `pid_grid` + `GROUP_SIZE_M`: L2-friendly tile 遍历顺序
- SiLU 融合: gate/up split 交错在同一 block 内 (`offs_bn` interleave 两半 N)
- 支持 INT4/INT8/FP8 量化路径

```python
@triton.jit
def _fused_moe_silu_kernel_gptq_awq(
    a_ptr, b_ptr, c_ptr, b_scale_ptr, b_zp_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, num_valid_tokens, ...
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_SIZE_M, MUL_ROUTED_WEIGHT, top_k, ...
):
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    GRID_MN = num_pid_n * num_pid_m

    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)  # cross-XCD remapping
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # SiLU interleaved column addressing (gate + up in alternating columns)
    BLOCK_SIZE_HALF = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    i_floor = i // 2
    offs_half = (pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)
    offs_bn = (offs_half + (i % 2) * (N // 2)) % N  # interleave gate & up

    # Token routing via sorted_token_ids
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Main GEMM accumulation loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        # INT4 unpack: b = (b >> b_shifter) & 0xF
        # Dequantize: b = (b - zp) * scale
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Fused SiLU: reshape accumulator to [M, N/2, 2], apply silu to gate
    accumulator = _silu_exp2(accumulator)  # gate * sigmoid(gate) fused

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token)
        accumulator *= moe_weight[:, None]

    tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None])
```

### 2.3 MLA Decode with RoPE (Multi-Latent Attention)

**源文件**: `aiter/aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py`

**优化要点**:
- DeepSeek-V2 MLA 架构: Q=[Q_NOPE; Q_PE], K=[KV; K_PE], V=[KV]
- Online RoPE 在 decode stage1 kernel 内融合
- `remap_xcd` 跨 chiplet PID 重映射
- Split-KV 并行: 每个 batch 的 KV 长度切分为 `NUM_KV_SPLITS` 段

```python
@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,           # [batch, heads, kv_lora_rank + qk_rope_head_dim]
    K_Buffer,    # [total_tokens, kv_lora_rank + qk_rope_head_dim]
    V_buffer,    # [total_tokens, kv_lora_rank]
    cos_sin_cache, positions,
    sm_scale, kv_indptr, kv_indices,
    Att_Out,     # [batch, heads, NUM_KV_SPLITS, kv_lora_rank + 1]
    ...
    kv_lora_rank: tl.constexpr,    # e.g. 512
    qk_rope_head_dim: tl.constexpr, # e.g. 64
    BLOCK_C: tl.constexpr,         # tile over compressed KV dim
    BLOCK_R: tl.constexpr,         # tile over rope dim
    BLOCK_N: tl.constexpr,         # tile over sequence length
    BLOCK_H: tl.constexpr,         # tile over query heads
    NUM_KV_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_q_head_blk = tl.cdiv(q_head_num, BLOCK_H)
    pid_head_kv_split = remap_xcd(pid % (num_q_head_blk * NUM_KV_SPLITS),
                                   num_q_head_blk * NUM_KV_SPLITS)

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = pid_head_kv_split // num_q_head_blk
    cur_batch = pid // (num_q_head_blk * NUM_KV_SPLITS)

    # Load Q_NOPE and Q_PE
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_c[None, :], other=0.0)
    q_pe = tl.load(Q + off_q_pe, mask=mask_h[:, None] & mask_qk_r[None, :], other=0.0)

    # Online RoPE application to Q_PE
    cos = tl.load(cos_sin_cache + pos * stride + offs_rotary)
    sin = tl.load(cos_sin_cache + pos * stride + offs_rotary + rotary_dim // 2)
    q_pe_rot = tl.load(Q + off_q_pe_rot, ...)
    q_pe_rot = tl.where(mask_rotate, -q_pe_rot, q_pe_rot)
    q_pe = q_pe * cos + q_pe_rot * sin  # RoPE: x*cos + rotate(x)*sin

    # Split-KV attention loop
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # Main attention loop over KV blocks
    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        # Load K_PE, apply RoPE, compute Q_PE @ K_PE^T → rope scores
        # Load KV compressed, compute Q_NOPE @ KV^T → nope scores
        # Combined attention score = rope_scores + nope_scores
        # Online softmax update
        # Accumulate: att_out += softmax_weight @ V_compressed
```

---

## Task 3: FP8 GEMM Blog (ROCm CDNA4)

> Source: [FP8 GEMM Optimization on AMD CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
> 性能进阶: 1.15 → 4.80 → 30 → 337 → 507 → 1166 → 2288 TFLOPS/s (M=N=K=4096)

### 3.1 Naive Baseline (1.15 TFLOP/s)

```cpp
__global__ void baseline_fp8_gemm_kernel(const fp8e4m3* A, const fp8e4m3* B,
                                          bf16* C, int M, int N, int K,
                                          int lda, int ldb, int ldc,
                                          float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += float(A[row * lda + k]) * float(B[col * ldb + k]);  // C = A * B^T
    }
    const float c_prev = (beta == 0.0f) ? 0.0f : static_cast<float>(C[row * ldc + col]);
    C[row * ldc + col] = bf16(alpha * acc + beta * c_prev);
}
```

### 3.2 LDS Tiling (4.80 TFLOP/s)

```cpp
__global__ void lds_tiled_fp8_gemm_kernel(const fp8e4m3* A, const fp8e4m3* B,
                                           bf16* C, int M, int N, int K,
                                           int lda, int ldb, int ldc,
                                           float alpha, float beta) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        As[threadIdx.y][threadIdx.x] = float(A[row * lda + (k0 + threadIdx.x)]);
        Bs[threadIdx.y][threadIdx.x] = float(B[col * ldb + (k0 + threadIdx.y)]);
        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * ldc + col] = bf16(alpha * acc + beta * ...);
}
```

### 3.3 MFMA + Vectorized FP8×16 Load (336.88 TFLOP/s)

```cpp
using fp8x16_t = __attribute__((vector_size(16))) fp8_t;

static inline __device__ fp8x16_t load_fp8x16_u4(const fp8_t* p) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    return *reinterpret_cast<const fp8x16_t*>(&v);
}

// In the load loop:
const fp8x16_t a_vec = load_fp8x16_u4(A_storage + (base_m + r) * lda + (k0 + k));
reinterpret_cast<fp8x16_t&>(As[r][k]) = a_vec;

const fp8x16_t b_vec = load_fp8x16_u4(B_storage + (base_n + r) * ldb + (k0 + k));
reinterpret_cast<fp8x16_t&>(Bs[r][k]) = b_vec;
```

### 3.4 Global→LDS Direct Load (506.70 TFLOP/s)

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr,
    int size, int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = {reinterpret_cast<uint64_t>(ptr), range_bytes, 0x110000};
    return *reinterpret_cast<i32x4*>(&rsrc);
}

__global__ void lds_buffer_copy(const float* src, float* dst) {
    __shared__ float lds_mem[NUM_ELEM];

    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(reinterpret_cast<uintptr_t>(lds_mem));
    i32x4 srsrc = make_srsrc(src, NUM_ELEM * sizeof(float));

    // Global → LDS (16 bytes per lane)
    llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, 16, threadIdx.x * 4, 0, 0, 0);
    asm volatile("s_waitcnt vmcnt(0)");

    // LDS → register (128-bit read)
    u32x4 reg_b128;
    const uint32_t lds_load_addr = reinterpret_cast<uintptr_t>(lds_mem + threadIdx.x * 4) * 4;
    asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                 : "=v"(reg_b128) : "v"(lds_load_addr), "i"(0) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)");
}
```

### 3.5 LDS Swizzle (Bank Conflict Elimination)

```cpp
// XOR-based swizzle for 16x128 tile on CDNA4 (64 banks)
// Self-inverse: swizzle(swizzle(col)) = col
int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}

// Usage during LDS write (after global→LDS load):
//   lds_addr = base + row * row_stride + swizzle_col(row, col);
// Usage during LDS read (before MFMA):
//   read_col = swizzle_col(row, original_col);  // same formula, XOR is self-inverse
```

### 3.6 Double-Buffer + Swizzle (1166.41 TFLOP/s)

```cpp
// Ping-pong LDS: two slots for each of A and B
__shared__ fp8 A_lds[2][TILE_M][TILE_K];  // slot 0 and slot 1
__shared__ fp8 B_lds[2][TILE_N][TILE_K];
int cur = 0, nxt = 1;

// Prologue: fill slot 0
prefetch_tile_to_lds(A_lds[cur], B_lds[cur], /*tile=*/0);
wait_for_global_loads();
block_sync();

for (int t = 0; t < num_k_tiles; ++t) {
    if (t + 1 < num_k_tiles) {
        prefetch_tile_to_lds_async(A_lds[nxt], B_lds[nxt], /*tile=*/t + 1);
    }
    fragments_a = read_fragments_from_lds(A_lds[cur]);  // with swizzle
    fragments_b = read_fragments_from_lds(B_lds[cur]);  // with swizzle
    acc = mfma(acc, fragments_a, fragments_b);

    if (t + 1 < num_k_tiles) {
        wait_for_global_loads();
        block_sync();
        cur ^= 1;
        nxt ^= 1;
    }
}
```

### 3.7 8-Wave Ping-Pong Scheduling (2288 TFLOP/s)

```cpp
// LDS layout: 2 ping-pong buffers × 2 halves (for A and B)
__shared__ fp8 A_lds[2][2][128][128];
__shared__ fp8 B_lds[2][2][128][128];

// Wave identification
int waveid = threadIdx.x / 64;   // 0..7
int wave_m = waveid / 4;          // 0 or 1 (ping vs pong group)
int wave_n = waveid % 4;          // SIMD lane

// Prologue: load initial tiles, stagger wave groups
async_load(A_lds[0], B_lds[0], /*k_tile=*/0);
s_waitcnt(vmcnt(0));
__builtin_amdgcn_s_barrier();

// Stagger: waves 4-7 hit extra barrier → creates offset
if (wave_m == 1)
    __builtin_amdgcn_s_barrier();

// Main loop: each iteration handles 2 K-tiles (unroll 2 for register pressure)
#pragma unroll 2
for (int k = 0; k < num_k_tiles - 2; k += 2) {
    // Phase A: wave_m=0 does MEMORY, wave_m=1 does COMPUTE (or vice versa)
    __builtin_amdgcn_s_setprio(1);        // boost memory-phase priority

    // Memory phase: load next K-tile into alternate LDS slot
    async_load(A_lds[nxt][half], B_lds[nxt][half], k + 2);

    __builtin_amdgcn_sched_barrier(0);     // hard scheduling fence
    __builtin_amdgcn_s_setprio(0);        // drop priority for compute phase
    __builtin_amdgcn_s_barrier();          // swap roles

    // Compute phase: MFMA on current LDS slot
    fragments_a = ds_read_b128(A_lds[cur][half]);
    fragments_b = ds_read_b128(B_lds[cur][half]);
    acc = mfma(acc, fragments_a, fragments_b);

    __builtin_amdgcn_sched_barrier(0);     // fence before role swap
    __builtin_amdgcn_s_barrier();          // swap roles again

    // swap cur/nxt slots
    cur ^= 1; nxt ^= 1;
}

// Epilogue: drain remaining tiles (manually unrolled, not shown)
```

#### 3.7b 性能汇总表

| Kernel Stage | M=N=K=4096 | M=N=K=8192 |
|---|---|---|
| Naive FP8 GEMM | 1.15 TFLOP/s | — |
| LDS Tiling | 4.80 | — |
| MFMA Matrix Core | 30.05 | — |
| + Vectorized Load | 336.88 | — |
| + Global→LDS Direct | 506.70 | — |
| + Swizzle | 497.43 | — |
| + Double Buffer + Swizzle | 1166.41 | — |
| + 128×128 8-wave | 1828.74 | — |
| + 256×256 8-wave (best) | 2288.16 | ~3000+ |
| hipBLASLt (target) | ~2750 | ~3130 |
