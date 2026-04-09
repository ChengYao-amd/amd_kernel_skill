# AMD vs NVIDIA — 关键差异速查表

## 术语对照

| NVIDIA | AMD | 说明 |
|--------|-----|------|
| Warp（32 线程） | Wavefront（64 线程） | 2 倍宽 — 影响 reduction、shuffle |
| SM | CU（Compute Unit） | 类似概念 |
| Shared Memory | LDS（Local Data Share） | MI300X 上 64KB/CU |
| CUDA Core | Stream Processor | — |
| Tensor Core | Matrix Core (MFMA) | 不同 ISA，不同寄存器布局 |
| PTX | AMDGPU ISA | 中间 vs 最终 ISA |
| nvcc | hipcc | — |
| ncu / nsys | rocprof / omniperf | — |
| cuDNN | MIOpen | — |
| CUTLASS | CK (Composable Kernel) | — |

## API 映射

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__syncwarp()` | 不需要（wavefront 是锁步的） |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` |
| `__ballot_sync(mask, pred)` | `__ballot(pred)`（返回 64 位） |
| `__shared__` | `__shared__`（相同） |
| `blockDim.x` | `blockDim.x`（相同） |

## 关键行为差异

| 方面 | NVIDIA | AMD |
|------|--------|-----|
| Warp/Wavefront 大小 | 32 | 64 |
| Reduction 的 shuffle 步数 | 5 | 6 |
| 需要 `__syncwarp()`？ | 是（独立调度） | 否（锁步） |
| Shared memory bank | 32 × 4B | 32 × 4B（但 conflict pattern 不同） |
| L2 cache 大小 | 40-50 MB (A100/H100) | 256 MB (MI300X) |
| Tensor core 输入 | HMMA（warp 级） | MFMA（wavefront 级） |
| Occupancy 计算器 | CUDA occ calculator | `rocminfo` + 手动计算 |
| 内联汇编 | PTX asm | AMDGPU ISA asm |
| 编译目标标志 | `-arch=sm_80` | `--offload-arch=gfx942` |

## 迁移清单

1. 将 CUDA API 替换为 HIP 等价物（`cuda` → `hip`）
2. 修改 warp size 假设：32 → 64
3. 更新 reduction 循环：5 步 → 6 步
4. 删除 `__syncwarp()` 调用
5. 从 shuffle/ballot 中移除 `mask` 参数
6. 将 `__ballot_sync` 返回值从 32 位改为 64 位
7. 更新编译：`nvcc` → `hipcc`，`-arch=sm_XX` → `--offload-arch=gfxYYY`
8. 重新调优 block size 和展开因子
9. 重新 benchmark 所有内容 — 不要假设 CUDA 性能可迁移
