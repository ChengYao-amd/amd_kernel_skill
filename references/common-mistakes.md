# AMD 特有常见错误

## 编译

| 错误 | 症状 | 修复 |
|------|------|------|
| 缺少 `--offload-arch` | 编译目标错误或运行时 kernel 不执行 | 始终指定 `--offload-arch=gfx942`（MI300X）或 `gfx950`（MI355X） |
| 缺少 `-O3` | 比预期慢 5-10 倍 | 始终使用 `-O3` |
| 直接使用 CUDA API | 编译错误 | 替换为 HIP 等价物（参见 `amd-vs-nvidia-cheatsheet.md`） |
| `--offload-arch` 不匹配实际 GPU | Kernel 运行但结果错误或崩溃 | 用 `rocminfo \| grep gfx` 验证实际 arch |
| 使用了 CDNA4 独有指令但目标为 gfx942 | 编译错误 | FP6/FP4 MFMA 仅 gfx950+，检查目标硬件 |

## 架构

| 错误 | 症状 | 修复 |
|------|------|------|
| 假设 warp = 32 | 错误的 reduction 结果、性能退化 | AMD wavefront = 64。使用 6 步 shuffle，不是 5 步 |
| **假设 MI300X 有 192 CU** | Grid 大小不足，GPU 利用率低 | MI300X/MI325X 实际有 **304 CU**（38/XCD × 8 XCDs）|
| 假设 shared mem = 48KB | LDS 溢出或 occupancy 错误计算 | MI300X LDS = **64 KB/CU**；MI355X = **160 KB/CU** |
| 使用 `__syncwarp()` | 不必要的同步 | AMD wavefront 是锁步的，无需部分同步 |
| 错误的 bank conflict 计算 | 意外的 LDS 争用 | **CDNA3**：32 bank × 4B；**CDNA4**：**64 bank**（与 CDNA3 不同，padding 需重算）；conflict pattern 与 NVIDIA 不同 |
| HIP 中滥用 **flat load**、忽略 **buffer ops** | VMEM 压力大、低于最优带宽 | 在可用场景优先 **`buffer_load_*` / buffer ops**（编译器或手写 ISA），勿假设 flat 与 buffer 等价最优 |
| CDNA4 上仍按 **32-bank LDS** 做 padding | 隐性 bank conflict、性能不达预期 | CDNA4 为 **64-bank LDS**，按 `isa/memory-instructions.md` 更新 swizzle / pad |
| 有 **structured sparsity** 能力却全程 dense | 未吃满有效吞吐 | CDNA4 官方标称含 **structured sparsity** 翻倍有效吞吐；算子/库支持时评估开启并做精度验证 |
| 未使用 **`__builtin_amdgcn_s_setprio`** 等 wave 调度手段 | ping-pong / 多 wave 重叠不足 | 高阶 GEMM（见 `advanced-optimization.md`）中配合 **`sched_barrier`**、`buffer_load_lds` 做 wave 级编排 |
| FP8 精度格式搞混 | 数值结果错误 | CDNA3 用 E4M3**FNUZ**/E5M2**FNUZ**；CDNA4 用 E4M3**FN**(OCP)/E5M2(OCP)，exponent bias 不同 |
| 忽略 XCD（多 die）拓扑 | L2 cache miss 率异常高 | MI300X 有 8 个 XCD，每个 XCD 有自己的 L2 (4MB)；跨 XCD 访问走 L3 |

## 性能

| 错误 | 症状 | 修复 |
|------|------|------|
| Thread block 太少 | GPU 利用率低 | MI300X 至少需要 **304 个 block**（每 CU 至少一个）；MI355X 至少 256 个 |
| 忽略寄存器溢出 | 不明原因的性能下降 | 用 ROCm Compute Profiler 检查 ScratchWaves，或 `hipcc -save-temps` 看 `.s` 文件 |
| 直接复制 CUDA 调优参数 | 次优性能 | 为 AMD 重新调优 block size、展开因子、tile 大小 |
| 未使用 AGPR 做 MFMA 累加器 | 更高 VGPR 压力 | 使用 AGPR 累加器释放 VGPR（参见 `isa/register-allocation.md`）|
| 忽略 `__builtin_amdgcn_readfirstlane` | SGPR 压力高，分支效率低 | 将 lane 0 值广播到 SGPR（CK 代码中大量使用此 pattern）|
| 未启用 TunableOp | GEMM 性能不稳定 | 设置 `PYTORCH_TUNABLEOP_ENABLED=1` 让 rocBLAS/hipBLASLt 自动调优 |

## Triton 特有

| 错误 | 症状 | 修复 |
|------|------|------|
| 使用 `tl.inline_asm_elementwise` | ROCm 上报错 | 使用纯 Triton 操作 |
| BLOCK_SIZE 不是 64 的倍数 | 浪费 lane | 使用 64, 128, 256, 512, 1024 |
| 假设 CUDA Triton 性能可迁移 | 失望 | 始终在 AMD 上 benchmark |
| 不知道 `matrix_instr_nonkdim` | 未选到最优 MFMA 指令 | autotune 配置中加入此参数（16 或 32），控制 MFMA tile 大小 |
| 不使用 `max-autotune` | GEMM 性能未优化 | `TORCHINDUCTOR_MAX_AUTOTUNE=1`，或 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK` |
| FP8 kernel 在不同架构上失败 | 精度类型不匹配 | gfx950 使用 OCP FP8，其他 arch 使用 FNUZ；AITER 自动处理，自定义 kernel 需检查 |

## 多硬件迁移

| 错误 | 症状 | 修复 |
|------|------|------|
| MI300X 的 tile size 直接用在 MI355X | 性能不达标 | CU 数量（304 vs 256）、LDS 大小（64 vs 160 KB）、带宽不同，必须重新调优 |
| 未考虑 CDNA4 新指令 | 错失性能提升 | MI355X 的 FP16 MFMA 有更大 K 维度（16×16×32, 32×32×16），以及 FP6/FP4 支持 |
| 跨 arch 编译但未测试 | 运行时行为不同 | `--offload-arch=gfx942 --offload-arch=gfx950` 编译后，在两种硬件上分别验证和 benchmark |

## 知识库

本文件是活文档。在迭代过程中发现新错误时回填。
