# MFMA (Matrix Fused Multiply-Add) Instruction Reference

This document summarizes the MFMA semantics, officially disclosed **tile shapes and cycle counts**, **compiler intrinsics**, **data layouts**, and **FP8 encoding differences** for Matrix Cores on **CDNA3 (GFX942)** and **CDNA4 (GFX950)**. Tables and formulas are primarily based on AMD ROCm official "Matrix Core Programming" blog series and LLVM/HIP public interfaces; assembly mnemonics take the form `v_mfma_*`, and specific encodings should be verified against the ISA manual for the corresponding `gfx*` target.

## Semantics and Notation

- MFMA completes **D <- A x B + C** (fused matrix multiply-add) in a single instruction, where the tile shape is denoted as **MxNxK**: A is **MxK**, B is **KxN**, C/D is **MxN**.
- Below, **(C,D) <- (A,B)** denotes the accumulator/result element type and the A, B element types; within the same instruction, A and B types can be identical or mixed by variant (e.g., some FP8 mixed intrinsics).

## Peak FLOPS/clock per CU (Matrix Core, whitepaper figures)

The following are the **floating-point/integer matrix operation throughput** achievable on the Matrix Core **per CU, per clock cycle** (for cross-referencing with the instruction cycle table and roofline; specific instructions still depend on the **MxNxK** and **Cycles** combination).

| Precision / Format | CDNA3 (FLOPS/clock/CU) | CDNA4 (FLOPS/clock/CU) |
|---------------------|-------------------------|-------------------------|
| FP16 | **2048** | **4096** |
| FP8 | **4096** | **8192** |
| INT8 | **4096** | (Refer to CDNA4 ISA / whitepaper for this column) |
| MXFP6 | -- | **16384** |
| MXFP4 | -- | **16384** |
| Matrix FP64 | **256** | **128** (halved relative to CDNA3, an architectural trade-off targeting AI workloads) |
| Matrix MXFP8 | -- | **8192** |

**Whitepaper Table 1 note**: The CDNA4 whitepaper lists MXFP6 and MXFP4 as "16834" FLOPS/clock/CU. This is almost certainly a **typo** for **16384** (= 2^14), which is consistent with the doubling pattern and peak PF calculations. Reference docs use the computed value **16384**.

**Cross-generation note (CDNA4)**:

- **TF32**: CDNA4 **hardware no longer provides a TF32 Matrix path**; it must rely on **BF16** or other paths for **software emulation** (unlike NVIDIA Tensor Core's native TF32 assumption -- expectations and numerical verification procedures must be adjusted during migration).

## CDNA3 (GFX942) MFMA: Complete Instruction Table (Official Blog)

The following are the MFMA types, **MxNxK**, and **Cycles** given in ROCm documentation for **GFX942** (cycle count per instruction on the Matrix Core, for rough throughput estimation; this is **not** memory access latency).

| Type (C,D) <- (A,B) | MxNxK | Cycles |
|---------------------|-------|--------|
| FP64 <- FP64 | 16x16x4 | 64 |
| FP32 <- FP32 | 32x32x2 | 64 |
| FP32 <- FP32 | 16x16x4 | 32 |
| FP32 <- FP16/BF16 | 32x32x8 | 32 |
| FP32 <- FP16/BF16 | 16x16x16 | 16 |
| FP32 <- FP8 (E4M3FNUZ / E5M2FNUZ) | 16x16x32 | 16 |
| FP32 <- FP8 (E4M3FNUZ / E5M2FNUZ) | 32x32x16 | 32 |

Notes:

- **FP16/BF16**: Within the same MFMA family, BF16 and FP16 typically share the same **MxNxK** and **Cycles**; the difference lies in the data encoding and the type suffix in the instruction mnemonic (e.g., `_f16` / `_bf16`).
- **FP64 / FP32 native**: "Full precision paths" for double/single precision GEMM; cycle counts are relatively high when the K dimension is small, and must be modeled together with memory access and instruction-level parallelism.

## CDNA4 (GFX950): Extensions Beyond CDNA3

**CDNA4** increases **per-CU matrix peak throughput** in the **FP16/FP8/low-bit and MX** directions, but **Matrix FP64 throughput is halved** and there is **no native TF32 Matrix** (see the table above and "Cross-generation note"). At the instruction level, it retains most MFMA families compatible with CDNA3, and adds the following entries (larger **K** FP16-class tiles, plus **FP8/FP6/FP4** and **MXFP*** block-scaled paths):

| Type (C,D) <- (A,B) | MxNxK | Cycles | Note |
|---------------------|-------|--------|------|
| FP32 <- FP16/BF16 | 16x16x32 | 16 | **NEW**: **K doubled** relative to CDNA3 |
| FP32 <- FP16/BF16 | 32x32x16 | 32 | **NEW**: **K doubled** |
| FP32 <- FP8/FP6/FP4 | 16x16x128 | 16-32 | **NEW**: A and B types can be independently configured |
| FP32 <- FP8/FP6/FP4 | 32x32x64 | 32-64 | **NEW** |
| FP32 <- MXFP8/MXFP6/MXFP4 | 16x16x128 | 16-32 | **NEW**: **block-scaled** |
| FP32 <- MXFP8/MXFP6/MXFP4 | 32x32x64 | 32-64 | **NEW**: **block-scaled** |

Where **Cycles** is written as a range (e.g., 16-32), this indicates variable latency depending on the specific operand type combination, internal implementation, or clock domain configuration; for performance modeling, always use **actual kernels and profilers** as the ground truth.

## Compiler Intrinsics: General Form and Modifier Parameters

The MFMA built-in functions exposed by LLVM/Clang have the common form:

```c
d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

- **a_reg / b_reg / c_reg / d_reg**: Vector registers managed by the compiler and register allocator (typically VGPRs; if using the AGPR accumulation path, this is governed by ABI and backend conventions).
- **cbsz, abid, blgp**: Control fields related to **sparsity/block compression/operand bit selection** (depending on the specific intrinsic and architecture); when writing assembly or inline experimental code, cross-reference the **LLVM built-in function documentation** and the corresponding `gfx*` **ISA manual** to avoid conflating with CUDA MMA's "static index" assumptions.

### Naming Examples (FP16 / FP8)

| Semantics (illustrative) | Intrinsic Example |
|--------------------------|-------------------|
| FP16, 16x16x16 | `__builtin_amdgcn_mfma_f32_16x16x16f16` |
| FP16, 32x32x8 | `__builtin_amdgcn_mfma_f32_32x32x8f16` |
| FP8, 32x32x16, both A and B are FP8 | `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` |
| FP8, 32x32x16, A is FP8, B is BF8 (mixed) | `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8` |

### CDNA4: Scaled MFMA (scale)

CDNA4 introduces built-in functions related to **FP8/FP6/FP4** and **scale**, example form:

```c
__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a, b, c,
    Atype, Btype,
    OPSEL_A, scale_a,
    OPSEL_B, scale_b
);
```

- **Atype / Btype**: Specify the low-bit format of A and B (e.g., FP8/FP6/FP4 combinations).
- **OPSEL_*** and **scale_***: Used for operand bit selection and per-block or per-element scale paths (used in conjunction with **MXFP*** block-scaled semantics); specific enumerations and valid combinations should follow the **ROCm/LLVM release notes**.

## AMD Matrix Instruction Calculator (Tool)

**AMD Matrix Instruction Calculator** (repository **ROCm/amd_matrix_instruction_calculator**) is a command-line/Python tool for querying encoding, register mapping, and throughput information for **MFMA** / **WMMA** and other matrix instructions. `--architecture` can specify aliases such as **CDNA2/gfx90a**, **CDNA3/gfx942**, **RDNA4/gfx1201**, etc.

### Five Query Modes

| Option | Function |
|--------|----------|
| `--detail-instruction` (`-d`) | Print instruction encoding, register usage, compute throughput, whether it **co-executes** with **VALU** |
| `--get-register` (`-g`) | Given matrix coordinates -> output **vector register**, **lane**, **bit range** |
| `--matrix-entry` (`-m`) | Given **register** + **lane** -> reverse-lookup matrix coordinates |
| `--register-layout` (`-R`) | Print the **register/lane** mapping table for the entire matrix block |
| `--matrix-layout` (`-M`) | Print the matrix elements corresponding to all **register/lane** pairs |

### CBSZ / ABID / BLGP Modifier Semantics

The three integer parameters at the end of classic **MFMA** **intrinsics** correspond to **VOP3P-MAI** encoding fields:

| Field | Meaning |
|-------|---------|
| **CBSZ** (**Control Broadcast Size**) | Controls the **A** matrix **block** broadcast granularity; valid range is `0 .. log2(blocks)` |
| **ABID** (**A-matrix Broadcast Identifier**) | Under the broadcast scheme defined by **CBSZ**, selects which **block** of **A** participates in the computation; range is `0 .. 2^CBSZ-1` |
| **BLGP** (**B-matrix Lane Group Pattern**) | **B** matrix **swizzle** / **broadcast** pattern across **lanes** (**CDNA2** common values: `0` normal; `1`/`2` broadcast between half **waves**; `3` **lane** data shift; `4-7` **16-lane group** broadcast modes, etc.; specifics depend on target architecture **ISA**) |

Additionally: **OPSEL**, **NEG** / **NEG_HI** are used for half-word selection within **32-bit** registers for **16-bit** data and sign control; see the tool and **ISA** manual.

### Register Output Format `Vx{y}.z`

The **`Vx{y}.z`** notation is commonly seen in tools and documentation:

- **`x`**: **Register** offset or number.
- **`y`**: **Lane** index (**wave64**: `0-63`).
- **`.z`**: The **bit** range within the register at that **lane** (e.g., `[15:0]`, `[31:16]`, `[7:0]`), used for unpacked elements or sub-words.

## rocWMMA fragment and Its Relationship to MFMA

**rocWMMA** provides a **`fragment`** abstraction at the **wavefront** granularity: **load_matrix_sync** / **mma_sync** / **store_matrix_sync** bridge **DRAM/LDS** data with the register layouts used by **MFMA**. Template parameters include **FragM/N/K**, **DataT**, **DataLayout**, **Scheduler**, etc.

- **fragment** internals use **packed** register storage where element ordering may not be intuitive; they must be used in accordance with the documentation for **layout** and **scheduler** pairing (e.g., **default_schedule**, **coop_row_major_2d**, etc.).
- **mma_sync** on **CDNA** ultimately dispatches **MFMA** (or the matrix instructions permitted by the architecture); **WMMA** corresponds to the **RDNA** path. When hand-writing **intrinsics**, you directly control **MFMA** and **modifiers**; **rocWMMA** hides most **lane-to-matrix** mapping details via **fragments**, but still requires the entire **wave** to be active -- otherwise behavior is undefined.

## FP8 and Low-Precision Formats: CDNA3 vs CDNA4

| Item | CDNA3 (GFX942) | CDNA4 (GFX950) |
|------|-----------------|-----------------|
| FP8 flavor | **E4M3FNUZ**, **E5M2FNUZ** (non-standard OCP, exponent bias typically **8 / 16**) | **E4M3FN**, **E5M2** (**OCP** flavor, exponent bias **7 / 15**) |
| Migration note | Existing kernels that hard-code FNUZ semantics need to verify **reinterpret / conversion** on CDNA4 | Easier to align with the **OCP FP8** ecosystem (training frameworks, quantization tools) |

Common storage types on the HIP side (names evolve with ROCm versions; refer to header files):

- `__hip_fp8_storage_t`
- `__amd_fp8_storage_t`

Before calling intrinsics, **FP8 operands** typically need to be **extended to the appropriate register width** per ABI requirements; official examples commonly **cast FP8 operands to `long`** before passing them to built-ins, to avoid implicit conversion and undefined layout.

## Peak Compute Estimation Formula (Matrix Core)

The official blog provides the **peak TFLOP/s** estimation in the following form:

```text
Peak_TFLOPS = 2 * M * N * K * num_matrix_cores * (max_engine_clock_Hz / cycle_count) / 1e6
```

- **2 x M x N x K**: The floating-point op count of one MFMA in the **FMA** sense (multiply and add each counted).
- **num_matrix_cores**: Total number of Matrix Cores on the GPU (varies by product; e.g., the MI300 series documentation states **1216**).
- **max_engine_clock**: Peak engine frequency (Hz).
- **cycle_count**: The **Cycles** corresponding to that MFMA variant (from the table above).

**MI325X (GFX942)** example (FP16 **32x32x8**, **Cycles = 32**, **1216** Matrix Cores, **2100 MHz**):

```text
2 * 32 * 32 * 8 * 1216 * (2100e6 / 32) / 1e6 ~ 1307.4 TFLOP/s
```

This value is consistent with the official **FP16/BF16 peak** stated in `hardware/mi300x.md`; when using a different **MxNxK** or **cycle_count**, both terms in the formula must be updated accordingly.

## Data Layout and Wavefront Mapping Key Points

The following is summarized from official examples and community practice, for use when hand-writing kernels or verifying compiler output:

- **Wavefront width**: **64 threads**; MFMA matrix elements are **distributed across 64 lanes**, which is **different** from NVIDIA's warp (32 lanes) MMA layout -- you **cannot** directly copy indexing from CUDA experience.
- **Number of elements held per thread (conceptually)**:
  - A: **M x K / 64**
  - B: **K x N / 64**
  - C/D: **M x N / 64**
- **Register types**: Results typically reside in **VGPR** vector types (e.g., `fp32x4_t`, `fp32x16_t`, depending on tile and backend).
- **FP8 calling convention**: Operands are commonly **cast to `long`** before passing to intrinsics, to ensure register width and calling convention consistency.
- **Comparison with NVIDIA MMA**: The lane-to-**(row, col)** mapping rules are different; when performing **layout conversion** or **bitwise comparison with CUTLASS/cuBLAS results**, always use **AMD documentation or llvm-mca / disassembly** as the reference.

## CDNA4: Unified F8F6F4 Instructions

CDNA4 introduces two **unified MFMA instructions** that accept FP8, FP6, and FP4 formats for A and B **independently**:

| Instruction | Tile Shape | Notes |
|------------|-----------|-------|
| `V_MFMA_F32_16x16x128_F8F6F4` | 16x16x128 | A/B format selected via CBSZ field |
| `V_MFMA_F32_32x32x64_F8F6F4` | 32x32x64 | A/B format selected via CBSZ field |
| `V_MFMA_SCALE_F32_16X16X128_F8F6F4` | 16x16x128 | Scaled variant with per-block E8M0 exponents |
| `V_MFMA_SCALE_F32_32X32X64_F8F6F4` | 32x32x64 | Scaled variant (MXFP path); requires ABID[0]=1 |

For the `_SCALE_` variants, setting `ABID[0]=0` forces all scales to 1.0 (exponent = 0x7F biased), effectively running unscaled. These instructions are the hardware backing for MXFP8/MXFP6/MXFP4 operations.

## CDNA4: SMFMAC (Sparse MFMA) Instructions

CDNA4 extends the sparse matrix instruction set with larger K dimensions:

| Instruction Family | Tile Shape | Data Types |
|-------------------|-----------|------------|
| `V_SMFMAC_F32_16X16X64_{F16,BF16}` | 16x16x64 | FP16/BF16 sparse |
| `V_SMFMAC_F32_32X32X32_{F16,BF16}` | 32x32x32 | FP16/BF16 sparse |
| `V_SMFMAC_I32_16X16X128_I8` | 16x16x128 | INT8 sparse |
| `V_SMFMAC_I32_32X32X64_I8` | 32x32x64 | INT8 sparse |
| `V_SMFMAC_F32_16x16x128_{FP8,BF8}` | 16x16x128 | FP8/BF8 sparse (all 4 A/B combinations) |
| `V_SMFMAC_F32_32x32x64_{FP8,BF8}` | 32x32x64 | FP8/BF8 sparse (all 4 A/B combinations) |

**Sparse matrix format**: A is a sparse matrix where 2 out of every 4 consecutive K elements are zero (2:4 structured sparsity). The non-zero values are packed densely, with separate index VGPRs (`SRC2`) encoding which 2 of 4 positions contain data. This enables approximately **2x** effective throughput for qualifying data.

**Key restriction**: SMFMAC instructions interpret `ACC_CD` differently from standard MFMA, and `CBSZ[1:0]`/`ABID[1:0]` fields are ignored for sparsity index instructions.

## Assembly Layer and INT8 and Other Variants

- Assembly mnemonics are conventionally written as **`v_mfma_{out}_{M}x{N}x{K}_{in}`** (specific suffixes vary by type and encoding).
- **INT8 -> INT32** and other integer MFMA instructions also exist in the ISA (older notes listed `v_mfma_i32_16x16x32_i8`); for **MxNxK, cycles, and register packing**, refer to the target `gfx` **ISA manual** and **LLVM `IntrinsicsAMDGPU.td`** -- this document focuses on the floating-point and FP8 mainline.

## Brief Comparison with NVIDIA Tensor Core

| Aspect | AMD MFMA | NVIDIA Tensor Core (MMA) |
|--------|----------|---------------------------|
| Execution granularity | **64-lane** wavefront | **32-lane** warp |
| Accumulator | **AGPR** (CDNA) or VGPR, different from the CUDA path | Architecture-dependent, often tightly coupled with register file |
| Layout | **Lane-to-matrix** mapping is **different** from CUDA | Per NVIDIA documentation |
| Software stack | ROCm, HIP, LLVM intrinsics | CUDA, PTX, mma.sync |

## Optimization Recommendations (Coordinating with Memory Access)

1. **Instruction-level parallelism**: While MFMA **occupies its full cycle count**, issue **global_load / DS reads and writes** simultaneously, using `s_waitcnt` for precise control to hide **VMEM/LDS** latency.
2. **Accumulator and register pressure**: Prefer using **AGPR** to carry MFMA accumulation chains, **freeing VGPRs** and improving wave **occupancy** (see `register-allocation.md` for details).
3. **Tile selection**: Balance between **cycle count and register occupancy**; for example, **32x32x8** vs **16x16x16** often perform differently under different bottlenecks -- test with **roofline** and **occupancy** measurements.
4. **Cross-generation compatibility**: When targeting **CDNA4**, pay attention to **larger-K FP16** and the new **FP6/FP4/MX** instruction families; compilation options and `offload-arch` should be set to **`gfx950`** (or the product-specific target).
5. **Numerical paths**: If FP8 training/inference migrates between **FNUZ (CDNA3)** and **OCP (CDNA4)**, be sure to perform **numerical regression** and **precision alignment**.

## VOP3P-MAI Encoding Format (CDNA1-3)

MFMA instructions are encoded in the **VOP3P-MAI** format. The key encoding fields are:

| Field | Purpose |
|-------|---------|
| **Src0** | A matrix source register |
| **Src1** | B matrix source register |
| **Src2** | C matrix source (accumulator input) |
| **Vdst** | D matrix destination (accumulator output) |
| **CBSZ** (3-bit) | Control Broadcast Size |
| **ABID** (4-bit) | A-matrix Broadcast Identifier |
| **BLGP** (3-bit) | B-matrix Lane Group Pattern |

### BLGP Values and Their Effects on B-Matrix (CDNA2 Reference)

| BLGP Value | Effect on B-matrix |
|------------|-------------------|
| 0 | Normal layout (no transformation) |
| 1 | Lanes 0-31 broadcast to lanes 32-63 |
| 2 | Lanes 32-63 broadcast to lanes 0-31 |
| 3 | All lane data shifted down by 16 bits |
| 4-7 | 16-lane group broadcast modes |

These values are architecture-dependent; always verify against the target `gfx*` ISA manual.

## Low-Precision Floating-Point Type Details

### Complete Type Reference Table

| Width | Shorthand | Exp Bias | Range | Hardware Support |
|-------|-----------|----------|-------|-----------------|
| 16-bit | E5M10 (FP16) | 15 | +/-65504 | All architectures |
| 16-bit | E8M7 (BF16) | 127 | +/-3.39e38 | gfx90a+ |
| 8-bit | E4M3FN (OCP FP8) | 7 | +/-448 | CDNA4 (gfx950) default |
| 8-bit | E4M3FNUZ | 8 | +/-240 | CDNA3 (gfx942) default |
| 8-bit | E5M2 (OCP BF8) | 15 | +/-57344 | CDNA4 (gfx950) |
| 8-bit | E5M2FNUZ | 16 | (FNUZ range) | CDNA3 (gfx942) |
| 8-bit | E8M0 | 127 | 2^(+/-127) | Scale factor only (CDNA4) |
| 6-bit | E2M3 | 1 | +/-7.5 | CDNA4 only |
| 6-bit | E3M2 | 3 | +/-28 | CDNA4 only |
| 4-bit | E2M1 (FP4) | 1 | +/-6 | CDNA4 only |

### HIP Header Paths and Classes

| Type | Header | Classes |
|------|--------|---------|
| FP4 (E2M1) | `<hip/amd_detail/amd_hip_fp4.h>` | `__hip_fp4_e2m1`, `__hip_fp4x2_e2m1`, `__hip_fp4x4_e2m1` |
| FP6 (E2M3/E3M2) | `<hip/amd_detail/amd_hip_fp6.h>` | `__hip_fp6_e2m3`, `__hip_fp6_e3m2`, plus `x2`/`x4` vector variants |
| FP8 | `<hip/amd_detail/amd_hip_fp8.h>` | `__hip_fp8_e4m3`, `__hip_fp8_e5m2` (OCP); `__hip_fp8_e4m3_fnuz`, `__hip_fp8_e5m2_fnuz` (FNUZ) |
| FP16 | `<hip/amd_detail/amd_hip_fp16.h>` | `__half` |
| BF16 | `<hip/amd_detail/amd_hip_bf16.h>` | `__hip_bfloat16` |

### Microscaling APIs (gfx950 / CDNA4)

CDNA4 introduces **hipExt microscaling APIs** with `__amd_scale_t` (E8M0 format) for scaled MFMA operations. These enable block-scaled FP8/FP6/FP4 inputs with per-block scale factors, used for mixed-precision GEMM with minimal accuracy loss.

## FP8 MFMA Kernel Example (HIP)

Complete example of a 32x32x16 FP8 MFMA kernel:

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

using fp8_t = __hip_fp8_storage_t;
using fp8x8_t = __attribute__((vector_size(8 * sizeof(fp8_t)))) fp8_t;
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

__global__ void mfma_fp32_32x32x16_fp8(const fp8_t* A, const fp8_t* B, float* C) {
    fp8x8_t a_reg;
    fp8x8_t b_reg;
    fp32x16_t c_reg {};

    a_reg = *reinterpret_cast<const fp8x8_t*>(
        A + (threadIdx.x / 32) * 8 + (threadIdx.x % 32) * 16);

    for (int i = 0; i < 8; i++)
        b_reg[i] = *(B + i * 32 + threadIdx.x % 32 + (threadIdx.x / 32) * 8 * 32);

    // Note: intrinsic expects (long) casts for FP8 operands
    c_reg = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
        (long)a_reg, (long)b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i*4];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32*1 + i * 32 * 8]   = c_reg[i*4+1];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32*2 + i * 32 * 8]   = c_reg[i*4+2];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32*3 + i * 32 * 8]   = c_reg[i*4+3];
    }
}
```

Key points:
- FP8 operands stored as `fp8x8_t` (8 bytes per thread for A and B)
- Accumulator is `fp32x16_t` (16 floats per thread for C/D)
- FP8 operands must be cast to `long` before passing to the intrinsic
- The lane-to-matrix mapping follows CDNA conventions (64-lane wavefront)

## References

- AMD ROCm Blog: *Matrix Core Programming* (contains **MFMA tables**, intrinsic examples, and peak formulas).
- LLVM: `IntrinsicsAMDGPU.td`, Clang `__builtin_amdgcn_mfma_*` definitions.
- Per-generation *RDNA/CDNA ISA Reference* (instruction encoding and latency details).
