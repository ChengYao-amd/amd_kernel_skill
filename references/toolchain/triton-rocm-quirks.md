# Triton ROCm Backend -- Differences, Autotune, and Debugging

## Unsupported Features

| Feature | Status | Alternative |
|---------|--------|-------------|
| `tl.inline_asm_elementwise` | Not supported | Use pure Triton operator composition |
| Some `tl.extra.cuda` APIs | Unavailable or behave differently | Probe based on ROCm build and documentation |
| `tl.tensor` layout details | May not be fully consistent with CUDA | Test on target GPU |

## Autotune and `matrix_instr_nonkdim` (AMD)

On AMDGPU, **MFMA** instructions have multiple **MxNxK** tiles. **`matrix_instr_nonkdim`** constrains the **non-K dimension** hardware instruction shape (common values are **16 or 32**, etc., matching the MFMA specs exposed by the target architecture). **PyTorch Inductor** has included this parameter in the search space for AMDGPU GEMM autotune to automatically select between **32x32x8** and **16x16x16** and other MFMA shapes; without it, performance may be significantly below optimal (see community PRs and Inductor changelog in PyTorch / Triton release notes).

**Relationship with Triton kernel configuration**: Common fields in AMD backend autotune configuration include:

- **`BLOCK_M`, `BLOCK_N`, `BLOCK_K`**: Software tiles.
- **`num_stages`**: Software pipeline stages (related to double buffering and prefetch strategies on CDNA).
- **`num_warps`**: On AMD, corresponds to **wavefront organization** (each wavefront has **64** threads, not NVIDIA's 32).
- **`matrix_instr_nonkdim`**: Aligns with **hardware MFMA instruction shape**, for the compiler to select the appropriate MFMA.

Recommendation: Keep **`num_stages=1`** in the configuration space as a fallback; **`num_warps`** and **block size** must satisfy **64-thread wavefront** and register/LDS constraints.

## `torch.compile` + Max Autotune (Inductor)

When enabling Inductor for more aggressive autotune on GEMM and other operators, use with environment variables:

```bash
# Enable Inductor extensive autotune (including GEMM etc.)
export TORCHINDUCTOR_MAX_AUTOTUNE=1

# GEMM backend candidates: Triton, PyTorch ATen, Composable Kernel (CK, commonly used on ROCm)
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK
```

**Notes**:

- Must be paired with **`torch.compile(..., mode=...)`** or other paths in code that actually trigger Inductor; setting environment variables alone without going through the compilation path will have no effect.
- **`CK`** depends on whether the ROCm build and PyTorch's CK integration are enabled; if a backend is unavailable, Inductor will skip or fall back (check runtime logs).
- Combinations and capitalization should follow the current PyTorch version's **`torch._inductor.config`** / documentation.

## Block Size and MI300-class GPUs

- **Thread count per block** should be a **multiple of 64** (aligned with wavefront).
- Common candidates: **64, 128, 256, 512, 1024**; on **MI300X**, **128 / 256** are common starting points.
- When **grid is too small**, it is hard to fill the **large number of CUs**; verify with occupancy and profiling.

## Performance and Verification

- Code generation quality on ROCm may differ from the CUDA backend; use **`rocprof` / `rocprofv3`** or **ROCm Compute Profiler** to verify actual hotspots and MFMA usage.
- If Triton hits a plateau, evaluate **HIP C++ / CK** for finer control.

## AMD ROCm Triton: TTGIR and LLVM-specific Passes (concepts)

The AMD-maintained **ROCm Triton fork** includes several optimization passes at the **TTGIR** layer closely related to CDNA (names and versions per actual LLVM/ROCm), commonly including:

| Pass (TTGIR) | Summary |
|--------------|---------|
| **AccelerateMatmul** | Select the optimal **MFMA** instruction shape |
| **BlockPingpong** | Implement **LDS double-buffer** scheduling |
| **CanonicalizePointers** | Normalize and optimize pointers, benefiting subsequent memory access optimization |
| **ConvertToBufferOps** | Convert **flat loads** to **buffer loads** (generally more performance-friendly) |

There are also **LLVM-IR** layer passes. To dump intermediate representations, try:

```bash
export MLIR_ENABLE_DUMP=1
# Combined with TRITON_DEBUG / cache directory settings, inspect IR at each stage
```

Note: The pass set changes across **ROCm / Triton** versions; after upgrading, compare generated IR and `rocprof` results.

## Debugging

### Triton itself

```bash
export TRITON_DEBUG=1
python kernel.py
```

### Build artifacts and cache directory

```bash
# Fix cache directory for easy inspection of LLVM/ISA and other generated artifacts
export TRITON_CACHE_DIR=/tmp/triton_cache
python kernel.py
# Browse kernel caches organized by hash under TRITON_CACHE_DIR
```

### PyTorch Inductor: `TORCH_COMPILE_DEBUG`

```bash
export TORCH_COMPILE_DEBUG=1
python your_script.py
```

When enabled, this typically generates a **`torch_compile_debug/`** directory in the working directory, containing files such as **`output_code.py`**, which allows you to inspect the **Triton kernel source code generated by Inductor** and scheduling logic, making it easy to verify whether **`BLOCK_*`, `num_stages`, `num_warps`, `matrix_instr_nonkdim`** and other configurations are as expected.

## FP8 and Architecture (gfx950 vs others)

- **CDNA3 (e.g., gfx942)** FP8 training/inference ecosystem is commonly aligned with **FNUZ** style (e.g., **E4M3FNUZ / E5M2FNUZ**).
- **CDNA4 (gfx950)** emphasizes **OCP FP8** (e.g., **E4M3 / E5M2** standard encoding), which differs from **FNUZ** in exponent bias and bit interpretation.
- **Triton / PyTorch / LLVM** in newer versions differentiate FP8 paths for **gfx942 / gfx950**; when migrating across architectures, perform **numerical and bitwise regression**, and do not assume CUDA or older gfx FP8 behavior can be directly reused.

## Common Pitfalls

1. **Assuming CUDA Triton optimal configuration can be directly migrated to ROCm** -- must re-test.
2. **`tl.constexpr`** is only for true compile-time constants; misuse causes unexpected recompilation or incorrect specialization.
3. **Not including `num_stages=1` and smaller `matrix_instr_nonkdim` candidates** -- occasionally better on CDNA.
4. **Grid too small relative to CU count** -- MI300X and similar require sufficient parallelism to hide latency.

## References

- ROCm documentation: [Optimizing Triton kernels](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)
- PyTorch: `TORCHINDUCTOR_MAX_AUTOTUNE`, `TORCH_COMPILE_DEBUG` related documentation per official docs and `torch._inductor.config`.
