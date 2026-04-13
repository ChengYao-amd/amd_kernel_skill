# SOTA Code Patterns -- Production-Level Excerpts (Deduplicated)

> Code patterns extracted from CK (composable_kernel), AITER, and ROCm blog posts that are **not already covered** in `kernel-recipes.md`. Each snippet includes source file path and optimization context.
>
> For buffer_load_lds intrinsics, LDS swizzle formulas, sched_barrier patterns, double-buffer pseudocode, RMSNorm Triton kernel, Fused MoE kernel, and FP8 GEMM progression stages, see `kernel-recipes.md`.

---

## Table of Contents

1. [MFMA Register Layout Constants (Detailed)](#1-mfma-register-layout-constants-detailed)
2. [MFMA Inline ASM Dispatch Macro (VGPR/AGPR Control)](#2-mfma-inline-asm-dispatch-macro-vgpragpr-control)
3. [FP8 16x16x128 Wave-Lane Mapping](#3-fp8-16x16x128-wave-lane-mapping)
4. [CK WarpGemm Type Aliases](#4-ck-warpgemm-type-aliases)
5. [CK FMHA FP8 Swizzle Configuration](#5-ck-fmha-fp8-swizzle-configuration)
6. [Waitcnt Helper -- Architecture-Adaptive](#6-waitcnt-helper--architecture-adaptive)
7. [MLA Decode with RoPE (Multi-Latent Attention)](#7-mla-decode-with-rope-multi-latent-attention)
8. [FP8 GEMM Naive Baseline and LDS Tiling Stages](#8-fp8-gemm-naive-baseline-and-lds-tiling-stages)
9. [FP8 GEMM MFMA + Vectorized Load Stage](#9-fp8-gemm-mfma--vectorized-load-stage)
10. [FP8 GEMM Global-to-LDS Direct Load Stage](#10-fp8-gemm-global-to-lds-direct-load-stage)
11. [Adaptive Top-K Selection Patterns](#11-adaptive-top-k-selection-patterns)

---

## 1. MFMA Register Layout Constants (Detailed)

**Source**: `ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`

Each MFMA instruction has a fixed lane-to-matrix-element mapping. Understanding these constants is essential for writing correct warp-level GEMM tiles.

### F32 16x16x4 MFMA Layout

```cpp
// File: warp_gemm_attribute_mfma_impl.hpp:66-127
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImplF32F32F32M16N16K4
{
    using AVecType = ext_vector_t<float, 1>;
    using BVecType = ext_vector_t<float, 1>;
    using CVecType = ext_vector_t<float, 4>;    // each lane holds 4 output elements

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 4;

    // Lane mapping: 64 lanes across M and K dimensions
    static constexpr index_t kAMLane     = 16;  // A rows mapped across 16 lanes
    static constexpr index_t kBNLane     = 16;  // B cols mapped across 16 lanes
    static constexpr index_t kABKLane    = 4;   // K-dim split across 4 groups
    static constexpr index_t kABKPerLane = 1;   // 1 K element per lane

    // Output layout: 64 lanes -> 16x16 via (M-stripe, N-lane)
    static constexpr index_t kCMLane     = 4;   // 4 M-groups (lane_id >> 4)
    static constexpr index_t kCNLane     = 16;  // 16 N-columns (lane_id & 15)
    static constexpr index_t kCM0PerLane = 1;   // inner M per lane
    static constexpr index_t kCM1PerLane = 4;   // 4 output rows per lane (CVecType length)

    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_16x16x4f32(a_vec[0], b_vec[0], c_vec, 0, 0, 0);
    }
};
```

### F32 32x32x2 MFMA Layout

```cpp
// File: warp_gemm_attribute_mfma_impl.hpp:129-191
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
    static constexpr index_t kCM1PerLane = 4;   // 4 rows per block -> 16 total per lane
};
```

### Quick Reference: Lane Mapping Rules

| MFMA Shape | CVec Length | kCMLane | kCNLane | kCM0PerLane | kCM1PerLane |
|---|---|---|---|---|---|
| 16x16x4 (F32) | 4 | 4 | 16 | 1 | 4 |
| 32x32x2 (F32) | 16 | 2 | 32 | 4 | 4 |
| 16x16x128 (FP8) | 4 | 4 | 16 | 1 | 4 |

---

## 2. MFMA Inline ASM Dispatch Macro (VGPR/AGPR Control)

**Source**: `ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp:25-62`

Controls whether C/A/B operands use VGPR (`v`) or AGPR (`a`). This is critical for 8-wave ping-pong: one wave group uses VGPR accumulators, the other uses AGPR.

```cpp
// Register class selection for MFMA operands
// dmod_ = destination (C), amod_ = A, bmod_ = B, cmod_ = source C
#define DISPATCH_MFMA_(mfma_, dmod_, amod_, bmod_, cmod_)       \
    asm volatile(mfma_ " %0, %1, %2, %3\n"                     \
                 : dmod_(c_vec)                                 \
                 : amod_(a_vec), bmod_(b_vec), cmod_(c_vec));

// Common register class combinations:
// Raw_vvv: C=VGPR, A=VGPR, B=VGPR  (standard)
// Raw_vav: C=VGPR, A=AGPR, B=VGPR  (one wave group)
// Raw_avv: C=AGPR, A=VGPR, B=VGPR  (other wave group)

// Example usage:
//   DISPATCH_MFMA_("v_mfma_f32_16x16x4f32", "+v", "a", "v", "v")
//   DISPATCH_MFMA_("v_mfma_f32_16x16x4f32", "+a", "v", "v", "a")
```

**Why this matters**: In the 8-wave ping-pong scheme, waves 0-3 accumulate in VGPR while waves 4-7 accumulate in AGPR (or vice versa). This doubles the effective accumulator space without register pressure conflict between the two groups.

---

## 3. FP8 16x16x128 Wave-Lane Mapping

**Source**: ROCm Blog -- FP8 GEMM Optimization on CDNA4

The largest FP8 MFMA instruction processes K=128 per wave invocation (65,536 FLOPs). This is the lane-to-element mapping for correct data placement.

```cpp
// Wave-Lane Mapping for v_mfma_f32_16x16x128_fp8_fp8
const int lane = lane_in_wave;            // [0, 63]
const int row_in_tile = lane & 15;         // [0, 15]  -- selects M/N row
const int row_group   = lane >> 4;         // [0, 3]   -- selects 4-row output stripe

// A input: each lane reads 32 FP8 elements (two K-chunks)
const int a_row = a_tile_row_start + row_in_tile;
const int k_chunk0 = row_group * 16;       // K offsets: 0, 16, 32, 48
const int k_chunk1 = k_chunk0 + 64;        // K offsets: 64, 80, 96, 112

// B input: same structure (transposed)
const int b_row = b_tile_row_start + row_in_tile;

// Output: each lane produces 4 FP32 accumulators
const int output_col = output_tile_col + row_in_tile;
const int output_row_start = output_tile_row + row_group * 4;
for (int t = 0; t < 4; ++t) {
    C[output_row_start + t][output_col] = bf16(alpha * accum_fp32[t] + beta * c_old);
}
```

**Key insight**: The K dimension is split into two 64-element halves, each further split into 4 groups of 16. Row groups (0-3) determine which 16-element K-chunk each lane processes.

---

## 4. CK WarpGemm Type Aliases

**Source**: `ck_tile/ops/gemm/warp/warp_gemm.hpp`

CK composes MFMA instructions via template aliases. These show how iterate-K and swizzle variants are built.

```cpp
// FP16 warp-level GEMM with IterateK and SwizzleA
using WarpGemmMfmaF16F16F32M32N32K8SwizzleA = WarpGemmImpl<
    WarpGemmAttributeMfmaIterateK_SwizzleA<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        1>>;

using WarpGemmMfmaF16F16F32M32N32K16SwizzleA = WarpGemmImpl<
    WarpGemmAttributeMfmaIterateK_SwizzleA<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8<WGAttrCtlEnum::Default_>,
        2>>;  // 2x iterate of base K=8

// gfx950 native (larger K per instruction):
using WarpGemmMfmaF16F16F32M32N32K16 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16<>>>;
using WarpGemmMfmaF16F16F32M16N16K32 = WarpGemmImpl<
    WarpGemmAttributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K32<>>>;
```

**Pattern**: `IterateK<Base, N>` runs the base MFMA instruction N times with K-offset stepping. `SwizzleA`/`SwizzleB` applies bank-conflict-free LDS access patterns.

---

## 5. CK FMHA FP8 Swizzle Configuration

**Source**: `ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp:276-278`

In Flash Attention with FP8, the swizzle factor controls how many LDS access phases are interleaved to eliminate bank conflicts.

```cpp
// FP8 warp gemm 32x32x32 with swizzle_factor=4
// Redistributes bank accesses across 4 phases of ds_read_b128
constexpr index_t swizzle_factor = 4;
return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<swizzle_factor>{};
```

**Context**: On CDNA4 with 64 LDS banks, `ds_read_b128` (128-bit read) executes in 4 phases. The swizzle factor must match the phase count to ensure each phase accesses different banks.

---

## 6. Waitcnt Helper -- Architecture-Adaptive

**Source**: `ck_tile/core/arch/arch.hpp:1017-1063`

Portable `s_waitcnt` wrapper that adapts encoding for gfx9 vs gfx12 architectures.

```cpp
// Architecture-adaptive s_waitcnt wrapper
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

// Combined barrier + waitcnt (gfx12 uses split barrier signals)
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

// Convenience: block_sync after direct global->LDS loads
template <index_t vmcnt = 0>
CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    s_waitcnt_barrier<vmcnt, /*expcnt_max*/, /*lgkmcnt_max*/>();
}
```

**Why this matters**: gfx12 (RDNA4) splits wait counters differently from gfx9 (CDNA). Portable code must handle both. The `block_sync_lds_direct_load` pattern is the canonical way to synchronize after `buffer_load_lds` operations.

---

## 7. MLA Decode with RoPE (Multi-Latent Attention)

**Source**: `aiter/aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py`

DeepSeek-V2/V3/R1 Multi-Latent Attention architecture. The decode kernel fuses RoPE application with split-KV attention computation.

**Architecture**: Q = [Q_NOPE; Q_PE], K = [KV_compressed; K_PE], V = [KV_compressed]

**Optimizations**:
- Online RoPE fused inside the decode kernel (no separate RoPE pass)
- `remap_xcd` for cross-chiplet PID distribution
- Split-KV parallelism: each batch's KV sequence is divided into `NUM_KV_SPLITS` segments

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

    # Online RoPE: x*cos + rotate(x)*sin
    cos = tl.load(cos_sin_cache + pos * stride + offs_rotary)
    sin = tl.load(cos_sin_cache + pos * stride + offs_rotary + rotary_dim // 2)
    q_pe_rot = tl.load(Q + off_q_pe_rot, ...)
    q_pe_rot = tl.where(mask_rotate, -q_pe_rot, q_pe_rot)
    q_pe = q_pe * cos + q_pe_rot * sin

    # Split-KV attention loop
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        # Load K_PE, apply RoPE, compute Q_PE @ K_PE^T -> rope scores
        # Load KV compressed, compute Q_NOPE @ KV^T -> nope scores
        # Combined score = rope_scores + nope_scores
        # Online softmax update
        # Accumulate: att_out += softmax_weight @ V_compressed
```

**AITER Speedup**: MLA decode achieves **17x** speedup over baseline on MI300X. This is the highest-impact kernel in DeepSeek model inference.

**Key Implementation Details**:
- `kv_lora_rank` (512) and `qk_rope_head_dim` (64) are DeepSeek-specific dimensions
- RoPE is applied to the PE portion only; the NOPE portion uses direct dot product
- Split-KV allows parallelism along the sequence dimension within a single head
- Stage 2 (not shown) performs the final softmax reduction across KV splits

---

## 8. FP8 GEMM Naive Baseline and LDS Tiling Stages

**Source**: ROCm Blog -- FP8 GEMM Optimization on CDNA4

These early stages are included for completeness. `kernel-recipes.md` covers the optimized stages; these show the progression starting point.

### Stage 1: Naive Baseline (1.15 TFLOP/s)

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

### Stage 2: LDS Tiling (4.80 TFLOP/s)

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

---

## 9. FP8 GEMM MFMA + Vectorized Load Stage

**Source**: ROCm Blog -- FP8 GEMM Optimization on CDNA4

This stage introduces MFMA matrix cores and 128-bit vectorized FP8 loads, jumping from ~30 to ~337 TFLOP/s.

```cpp
using fp8x16_t = __attribute__((vector_size(16))) fp8_t;

// 128-bit vectorized load: 16 FP8 values per load
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

**Performance leap explanation**: A single `v_mfma_f32_16x16x128_fp8_fp8` instruction performs 65,536 FLOPs, compared to 128 for a scalar FMA -- a 512x per-instruction throughput increase.

---

## 10. FP8 GEMM Global-to-LDS Direct Load Stage

**Source**: ROCm Blog -- FP8 GEMM Optimization on CDNA4

Standalone example showing the complete global-to-LDS direct load flow (506.70 TFLOP/s stage).

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

    // Global -> LDS (16 bytes per lane, bypasses register file)
    llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, 16, threadIdx.x * 4, 0, 0, 0);
    asm volatile("s_waitcnt vmcnt(0)");

    // LDS -> register (128-bit read)
    u32x4 reg_b128;
    const uint32_t lds_load_addr = reinterpret_cast<uintptr_t>(lds_mem + threadIdx.x * 4) * 4;
    asm volatile("ds_read_b128 %0, %1 offset:%2\n"
                 : "=v"(reg_b128) : "v"(lds_load_addr), "i"(0) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)");
}
```

**Key difference from `kernel-recipes.md`**: This shows the complete standalone flow including `make_srsrc` with the blog post's config value (0x110000 vs CK's 0x00020000), the `ds_read_b128` LDS-to-register read, and explicit waitcnt placement.

---

## 11. Adaptive Top-K Selection Patterns

**Source**: AITER Library -- Adaptive Top-K blog (Feb 2026)

### Problem

Standard Radix Sort has fixed histogram overhead (LDS atomics scan all 2048 buckets) regardless of K, causing performance cliffs at small K values.

### Two Complementary Strategies

| Strategy | Best For | Mechanism |
|---|---|---|
| **BlockTopkSort** | Small input per warp | Bitonic sort on register-resident data via DPP instructions |
| **BlockTopkFilter** | Larger inputs | Ballot-based filtering prunes candidates before sorting |
| **AdaptiveTopK** | All K values | Auto-selects between bitonic (small K) and radix (large K) |

### Hardware-Specific Optimizations

| Technique | Impact |
|---|---|
| DPP (Data Parallel Primitives) | Ultra-low-latency register exchange (faster than shuffle) |
| `med3` instruction | Branch-free compare-and-swap via median-of-3 hardware |
| Buffer `load_dwordx4` | 4-8x fewer load instructions (16 bytes per instruction) |
| Double buffering | Compute on chunk N while loading N+1 |

### Performance

- DPP + med3: up to **32% improvement** over shuffle/conditional baseline
- Buffer instructions: up to **55% improvement** on long sequences (131K length)
- AdaptiveTopK consistently beats PyTorch `torch.topk`, Triton, and radix across all K values

### Adaptive Threshold Formula (MI300X-Tuned)

```
n + K * log2(K) >= 3 * Factor(n) * n
Factor(n) = 1/3 + 1.6 / (log2(n) - 9.5)
```

Example thresholds:
- n=8192: K threshold ~195 (use bitonic below, radix above)
- n=65536: K threshold ~576
- n=131072: K threshold ~878

---

## Performance Summary Table

| Pattern | Source | Key Metric |
|---|---|---|
| MFMA 16x16x128 FP8 | ROCm Blog | 65,536 FLOPs/instruction |
| Vectorized FP8x16 load | ROCm Blog | 128-bit per load instruction |
| Global-to-LDS direct | CK/Blog | Bypasses VGPR, 128-bit/lane on CDNA4 |
| AGPR dispatch macro | CK | Doubles accumulator space for ping-pong |
| MLA decode + RoPE | AITER | 17x speedup over baseline |
| Adaptive Top-K | AITER | Up to 55% over standard implementations |
| Waitcnt helper | CK | Portable gfx9/gfx12 synchronization |
