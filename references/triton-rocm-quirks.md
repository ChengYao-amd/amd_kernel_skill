# Triton ROCm 后端 — 差异、Autotune 与调试

## 不支持的特性

| 特性 | 状态 | 替代方案 |
|------|------|----------|
| `tl.inline_asm_elementwise` | 不支持 | 使用纯 Triton 算子组合 |
| 部分 `tl.extra.cuda` API | 不可用或行为不同 | 以 ROCm 构建与文档为准做探测 |
| `tl.tensor` 布局细节 | 可能与 CUDA 不完全一致 | 在目标 GPU 上实测 |

## Autotune 与 `matrix_instr_nonkdim`（AMD）

在 AMDGPU 上，**MFMA** 指令有多种 **M×N×K** tile。**`matrix_instr_nonkdim`** 用于约束 **非 K 维** 上的硬件指令形状（常见取值为 **16 或 32** 等，与目标架构暴露的 MFMA 规格一致）。**PyTorch Inductor** 在 AMDGPU GEMM autotune 中已将该参数纳入搜索空间，以便在 **32×32×8** 与 **16×16×16** 等 MFMA 形态间自动选型；未纳入时可能显著低于最优（社区 PR 与 Inductor 变更说明可参考 PyTorch / Triton 发行说明）。

**与 Triton kernel 配置的关系**：AMD 后端 autotune 配置中常见字段包括：

- **`BLOCK_M`, `BLOCK_N`, `BLOCK_K`**：软件 tile。
- **`num_stages`**：软件流水线级数（与 CDNA 上双缓冲、prefetch 策略相关）。
- **`num_warps`**：在 AMD 上对应 **wavefront 组织方式**（每 wavefront **64** 线程，而非 NVIDIA 的 32）。
- **`matrix_instr_nonkdim`**：与 **硬件 MFMA 指令形状** 对齐，供编译器选择合适 MFMA。

建议：配置空间内保留 **`num_stages=1`** 作为回退；**`num_warps`** 与 **block 尺寸** 需满足 **64 线程 wavefront** 与寄存器/LDS 约束。

## `torch.compile` + Max Autotune（Inductor）

启用 Inductor 对 GEMM 等算子做更激进 autotune 时，可配合环境变量：

```bash
# 打开 Inductor 广泛 autotune（含 GEMM 等）
export TORCHINDUCTOR_MAX_AUTOTUNE=1

# GEMM 后端候选：Triton、PyTorch ATen、Composable Kernel（CK，ROCm 上常用）
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON,ATEN,CK
```

**说明**：

- 需在代码中配合 **`torch.compile(..., mode=...)`** 等实际触发 Inductor 的路径；仅设环境变量而不走编译路径不会生效。
- **`CK`** 依赖 ROCm 构建与 PyTorch 的 CK 集成是否开启；若某后端不可用，Inductor 会跳过或回退（以运行时日志为准）。
- 组合与大小写以当前 PyTorch 版本 **`torch._inductor.config`** / 文档为准。

## Block 尺寸与 MI300 类 GPU

- **Block 内线程数** 宜为 **64 的倍数**（与 wavefront 对齐）。
- 常见候选：**64, 128, 256, 512, 1024**；**MI300X** 上 **128 / 256** 常作为起点。
- **Grid 过小** 时难以填满 **大量 CU**；需结合 occupancy 与 profile 验证。

## 性能与验证

- ROCm 上生成代码质量可能与 CUDA 后端不同；应用 **`rocprof` / `rocprofv3`** 或 **ROCm Compute Profiler** 验证真实热点与 MFMA 使用。
- 若 Triton 达到平台期，可评估 **HIP C++ / CK** 以获得更细控制。

## AMD ROCm Triton：TTGIR 与 LLVM 专属 Pass（概念）

AMD 维护的 **ROCm Triton fork** 在 **TTGIR** 层包含若干与 CDNA 密切相关的优化 pass（名称与版本以实际 LLVM/ROCm 为准），常见包括：

| Pass（TTGIR） | 作用摘要 |
|---------------|----------|
| **AccelerateMatmul** | 选择最优 **MFMA** 指令形态 |
| **BlockPingpong** | 实现 **LDS double-buffer** 调度 |
| **CanonicalizePointers** | 规范化与优化指针，利于后续访存优化 |
| **ConvertToBufferOps** | 将 **flat load** 转为 **buffer load**（通常更有利于性能） |

此外还有 **LLVM-IR** 层 pass。需要 dump 中间表示时，可尝试：

```bash
export MLIR_ENABLE_DUMP=1
# 再配合 TRITON_DEBUG / 缓存目录等，查看各阶段 IR
```

说明：pass 集合随 **ROCm / Triton** 版本变化，升级后建议对比生成 IR 与 `rocprof` 结果。

## 调试

### Triton 自身

```bash
export TRITON_DEBUG=1
python kernel.py
```

### 编译产物与缓存目录

```bash
# 固定缓存目录，便于查看 LLVM/ISA 等生成物
export TRITON_CACHE_DIR=/tmp/triton_cache
python kernel.py
# 在 TRITON_CACHE_DIR 下浏览按 hash 组织的内核缓存
```

### PyTorch Inductor：`TORCH_COMPILE_DEBUG`

```bash
export TORCH_COMPILE_DEBUG=1
python your_script.py
```

开启后通常会在工作目录生成 **`torch_compile_debug/`**，其中包含 **`output_code.py`** 等文件，可查看 **Inductor 生成的 Triton kernel 源码** 与调度逻辑，便于核对 **`BLOCK_*`、`num_stages`、`num_warps`、`matrix_instr_nonkdim`** 等配置是否如预期。

## FP8 与架构（gfx950 vs 其他）

- **CDNA3（如 gfx942）** 上 FP8 训练/推理生态常与 **FNUZ** 风格（如 **E4M3FNUZ / E5M2FNUZ**）对齐。
- **CDNA4（gfx950）** 上更强调 **OCP FP8**（如 **E4M3 / E5M2** 标准编码），与 **FNUZ** 在指数偏置与位解释上不同。
- **Triton / PyTorch / LLVM** 在较新版本中针对 **gfx942 / gfx950** 区分 FP8 路径；跨架构迁移时需做 **数值与 bitwise 回归**，勿假设 CUDA 或旧 gfx 的 FP8 行为可直接复用。

## 常见陷阱

1. **假设 CUDA Triton 最优配置可原样迁移到 ROCm** — 必须重测。
2. **`tl.constexpr`** 仅用于真编译时常量；滥用会导致意外重编译或错误特化。
3. **未覆盖 `num_stages=1` 与较小 `matrix_instr_nonkdim` 候选** — CDNA 上偶发更优。
4. **Grid 相对 CU 数过小** — MI300X 等需要足够并行度才能隐藏延迟。

## 参考

- ROCm 文档：[Optimizing Triton kernels](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)
- PyTorch：`TORCHINDUCTOR_MAX_AUTOTUNE`、`TORCH_COMPILE_DEBUG` 相关说明以官方文档与 `torch._inductor.config` 为准。
