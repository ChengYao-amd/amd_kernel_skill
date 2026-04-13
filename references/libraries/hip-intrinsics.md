# AMD GPU HIP & Compiler Intrinsics Reference

This document supplements **MFMA compiler built-ins**, **FP8 storage types**, **CDNA4 block-scaled MFMA**, and common patterns such as **`__builtin_amdgcn_readfirstlane`**; cross-lane and memory access sections retain comparisons with NVIDIA. Details are subject to the **target ROCm / LLVM version** and **`offload-arch`**.

## MFMA: Compiler Intrinsic General Form

The matrix multiply-add built-in functions exposed by LLVM/Clang are commonly written as:

```c
d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

- **`ODType_MxNxKInDType`**: Encodes the output element type, **M x N x K** tile, and input element type (e.g., `f32_32x32x8f16`).
- **`a_reg` / `b_reg` / `c_reg`**: Vector register operands; **`d_reg`** is the accumulated result (corresponding to the intrinsic return value).
- **`cbsz`, `abid`, `blgp`**: Immediate fields related to **sparsity/block compression/sub-block selection**; semantics depend on the **specific intrinsic and ISA**; experimental handwritten code should cross-reference **LLVM `IntrinsicsAMDGPU.td`** and the **CDNA ISA manual**.

## CDNA3 (GFX942) Common `__builtin_amdgcn_mfma_*` Names (Excerpt)

The following names are commonly seen in **CDNA3** documentation and LLVM; **whether they are enabled** depends on `-mcpu`/`-target` and architecture features. The complete set is subject to the **LLVM release**.

**FP64 / FP32 accumulation, double/single precision matrix:**

- `__builtin_amdgcn_mfma_f64_16x16x4f64`
- `__builtin_amdgcn_mfma_f32_32x32x2f32`
- `__builtin_amdgcn_mfma_f32_16x16x4f32`

**FP16 / BF16 -> FP32 accumulation:**

- `__builtin_amdgcn_mfma_f32_32x32x8f16`
- `__builtin_amdgcn_mfma_f32_16x16x16f16`
- `__builtin_amdgcn_mfma_f32_32x32x2bf16`
- `__builtin_amdgcn_mfma_f32_16x16x2bf16`

**INT8 -> INT32 accumulation (examples):**

- `__builtin_amdgcn_mfma_i32_32x32x4i8`
- `__builtin_amdgcn_mfma_i32_16x16x4i8`

**FP8 (FNUZ style, CDNA3) -> FP32 accumulation (examples):**

- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`
- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8`
- `__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8`
- `__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8`
- (As well as **16x16x32** and similar variants; see LLVM)

## CDNA4 (GFX950) Extensions: Larger-K FP16-Class MFMA

Building on the CDNA3 table, CDNA4 adds FP16/BF16-class tiles with **larger K dimension** (example names, subject to LLVM):

- `__builtin_amdgcn_mfma_f32_16x16x32f16`
- `__builtin_amdgcn_mfma_f32_32x32x16f16`
- (Parallel variants with BF16 suffix `bf16`)

## CDNA4: Block-Scaled MFMA -- `__builtin_amdgcn_mfma_scale_f32_*_f8f6f4`

For **FP8 / FP6 / FP4** and **block-scaled** paths, Clang provides built-ins of the following form (**specific tile names and parameter lists are subject to the current LLVM headers/documentation**):

```c
__builtin_amdgcn_mfma_scale_f32_MxNxK_f8f6f4(
    a, b, c,
    Atype, Btype,
    OPSEL_A, scale_a,
    OPSEL_B, scale_b
);
```

- **`M x N x K`**: e.g., **16x16x128**, **32x32x64**, and other CDNA4 disclosed shapes.
- **`Atype` / `Btype`**: Encodes the **A/B low-bit format**; common enumeration conventions (consistent with LLVM `MFMAScaleFormats` etc., **subject to header files**):

| Value | Meaning |
|-------|---------|
| 0 | E4M3 (fp8) |
| 1 | E5M2 (bf8) |
| 2 | E2M3 (fp6) |
| 3 | E3M2 (bf6) |
| 4 | E2M1 (fp4) |

- **`OPSEL_A` / `OPSEL_B`**: Controls related to operand **sub-word/vector lane selection** and packing.
- **`scale_a` / `scale_b`**: Register paths related to **per-block / MXFP** scale factors (used in conjunction with ISA-level scale loading).

> Tip: The bit interpretation of **OCP FP8** on CDNA4 differs from **FNUZ** on CDNA3; when used with Triton/PyTorch, **quantization and types** must be unified.

## FP8 HIP Types & Vector Wrappers

Storage and vector wrapper names evolve with ROCm; common ones include:

- **`__hip_fp8_storage_t`**, **`__amd_fp8_storage_t`**: 8-bit storage units.
- Wrapped into short vectors using **`vector_size`** for convenient passing and alignment in kernels:

```cpp
using fp8x8_t = __attribute__((vector_size(8))) __hip_fp8_storage_t;
```

The actual ABI and intrinsic input parameter widths must match the **LLVM built-in signatures**; official examples commonly use **explicit widening** (e.g., cast to **`long`**) for FP8 operands to satisfy register width requirements.

## `__builtin_amdgcn_readfirstlane` (Common Pattern in CK and Other Libraries)

**`__builtin_amdgcn_readfirstlane(val)`** corresponds to the ISA instruction **`v_readfirstlane_b32`**: reads the scalar value from **lane 0** and broadcasts it as a **uniform** result across the entire wavefront. Commonly used in Composable Kernel (CK) and other libraries for:

- Broadcasting addresses, constants, or table entries **computed only on lane 0** to the entire wavefront;
- Reducing **redundant scalars in VGPRs** and partially relieving **SGPR pressure** (depending on the specific register allocation).

Typical pattern: first compute **`val`** across all lanes, then **`scalar = __builtin_amdgcn_readfirstlane(val);`**, and subsequently use **`scalar`** for memory accesses or indexing that require a **uniform offset**. Note: if **`val`** is not active on lane 0 or contains divergent values, the semantics must be consistent with the **active mask** and **WF64** rules.

## Cross-Lane Operations (Summary Table)

| Intrinsic | Use Case | ISA Hint |
|-----------|----------|----------|
| `__builtin_amdgcn_readfirstlane(val)` | Lane 0 -> broadcast to entire wave (scalarization) | `v_readfirstlane_b32` |
| `__builtin_amdgcn_readlane(val, lane)` | Read from a specific lane | `v_readlane_b32` |
| `__builtin_amdgcn_ds_swizzle(val, pattern)` | Lane permutation (no LDS traffic) | `ds_swizzle_b32` |
| `__builtin_amdgcn_mov_dpp(val, ctrl, row_mask, bank_mask, bound_ctrl)` | DPP data movement | `v_mov_b32 dpp` |
| `__shfl` / `__shfl_xor` | Compatibility layer shuffle | Aligned with wavefront width 64 |

## Wavefront Reduction

AMD **wavefront width is 64** (not NVIDIA's 32). **`__shfl_xor`** reduction requires **6 steps** (`offset = 32,16,...,1`), not 5.

```cpp
float wf_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}
```

## Memory & Math Intrinsics (Excerpt)

| Category | Example | Description |
|----------|---------|-------------|
| Non-temporal access | `__builtin_nontemporal_load` / `__builtin_nontemporal_store` | Hints to bypass or weaken cache |
| Wait count | `__builtin_amdgcn_s_waitcnt` | Fine-grained memory/instruction wait |
| Fast math | `__builtin_amdgcn_rcpf`, `rsqf`, `exp2f`, `log2f` | Fast approximations; precision per documentation |

## AMD vs NVIDIA (Brief Comparison)

| NVIDIA | Common AMD Counterpart |
|--------|----------------------|
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` (full wavefront, no mask) |
| `__syncwarp()` | Wavefront lockstep; no equivalent needed in most scenarios |
| `__ballot_sync(mask, pred)` | `__ballot(pred)`, **64-bit** |

## Further Reading

- More complete MFMA table and peak throughput formulas in this repository: [`isa/mfma-instructions.md`](isa/mfma-instructions.md)
- AMD ROCm Blog: *Matrix Core Programming on AMD CDNA3 and CDNA4*
- LLVM: `IntrinsicsAMDGPU.td`
