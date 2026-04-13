# hipcc Compilation Guide

## Basic Commands

```bash
# Single target
hipcc -O3 --offload-arch=gfx942 -o kernel.so -shared -fPIC kernel.cpp

# Multiple targets
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 -o kernel.so -shared -fPIC kernel.cpp

# With PyTorch
hipcc -O3 --offload-arch=gfx942 \
  $(python3 -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")  \
  -shared -fPIC kernel.cpp -o kernel.so

# Save intermediate files (for ISA inspection)
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# Look for *.s files containing ISA
```

## Common Flags

| Flag | Purpose |
|------|---------|
| `-O3` | Full optimization (must use) |
| `--offload-arch=gfxNNN` | Target GPU architecture (required) |
| `-shared -fPIC` | Build shared library for Python loading |
| `-save-temps` | Retain intermediate .s (ISA) files |
| `-Rpass=inline` | Show inlining decisions |
| `-ffast-math` | Aggressive FP optimization (may affect precision) |
| `-munsafe-fp-atomics` | Faster atomic FP operations (rarely loses precision) |

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `error: unknown target CPU 'gfx942'` | ROCm version too old | Update ROCm or use `rocminfo` to check correct arch |
| `undefined reference to __hip_*` | Missing HIP runtime linkage | Add `-lhip_hcc` or use `hipcc` instead of `g++` |
| `error: use of undeclared identifier '__shfl_sync'` | CUDA API not available in HIP | Use `__builtin_amdgcn_ds_swizzle` or `__shfl` |
| `error: too few register available` | Too many VGPRs | Add `__launch_bounds__`, reduce live variables |
| Kernel runs but produces wrong results | `--offload-arch` mismatch | Verify arch matches `rocminfo` output |
| Poor performance, `-O3` not used | Debug build | Always compile with `-O3` |

## PyTorch Extension Build

```python
from torch.utils.hip_extension import load

module = load(
    name="custom_kernel",
    sources=["kernel.cpp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--offload-arch=gfx942"],
)
```
