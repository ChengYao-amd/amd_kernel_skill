# AMD vs NVIDIA -- Key Differences Quick Reference

## Terminology Mapping

| NVIDIA | AMD | Notes |
|--------|-----|-------|
| Warp (32 threads) | Wavefront (64 threads) | 2x wider -- affects reduction, shuffle |
| SM | CU (Compute Unit) | Similar concept |
| Shared Memory | LDS (Local Data Share) | 64KB/CU on MI300X |
| CUDA Core | Stream Processor | -- |
| Tensor Core | Matrix Core (MFMA) | Different ISA, different register layout |
| PTX | AMDGPU ISA | Intermediate vs final ISA |
| nvcc | hipcc | -- |
| ncu / nsys | rocprof / omniperf | -- |
| cuDNN | MIOpen | -- |
| CUTLASS | CK (Composable Kernel) | -- |

## API Mapping

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__syncwarp()` | Not needed (wavefront is lockstep) |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` |
| `__ballot_sync(mask, pred)` | `__ballot(pred)` (returns 64-bit) |
| `__shared__` | `__shared__` (same) |
| `blockDim.x` | `blockDim.x` (same) |

## Key Behavioral Differences

| Aspect | NVIDIA | AMD |
|--------|--------|-----|
| Warp/Wavefront size | 32 | 64 |
| Shuffle steps for reduction | 5 | 6 |
| Need `__syncwarp()`? | Yes (independent scheduling) | No (lockstep) |
| Shared memory bank | 32 x 4B | 32 x 4B (but conflict pattern differs) |
| L2 cache size | 40-50 MB (A100/H100) | 256 MB (MI300X) |
| Tensor core input | HMMA (warp-level) | MFMA (wavefront-level) |
| Occupancy calculator | CUDA occ calculator | `rocminfo` + manual calculation |
| Inline assembly | PTX asm | AMDGPU ISA asm |
| Compilation target flag | `-arch=sm_80` | `--offload-arch=gfx942` |

## Migration Checklist

1. Replace CUDA APIs with HIP equivalents (`cuda` -> `hip`)
2. Modify warp size assumptions: 32 -> 64
3. Update reduction loops: 5 steps -> 6 steps
4. Remove `__syncwarp()` calls
5. Remove `mask` parameter from shuffle/ballot
6. Change `__ballot_sync` return value from 32-bit to 64-bit
7. Update compilation: `nvcc` -> `hipcc`, `-arch=sm_XX` -> `--offload-arch=gfxYYY`
8. Re-tune block size and unroll factors
9. Re-benchmark everything -- do not assume CUDA performance translates
