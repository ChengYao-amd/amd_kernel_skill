# AMD ISA / MFMA / Low-Level Programming — P0 Crawl Report

> Crawl date: 2026-04-09 | Sources: 12 URLs requested, 9 successfully fetched (3 returned 404/timeout)

---

## Table of Contents

1. [AMD Matrix Instruction Calculator](#1-amd-matrix-instruction-calculator)
2. [MFMA ISA Encoding & Compiler Intrinsics](#2-mfma-isa-encoding--compiler-intrinsics)
3. [rocWMMA API Reference](#3-rocwmma-api-reference)
4. [rocWMMA Programming Guide](#4-rocwmma-programming-guide)
5. [HIP C++ Language Extensions](#5-hip-c-language-extensions)
6. [HIP Memory Management API](#6-hip-memory-management-api)
7. [HIP CUDA Porting Guide](#7-hip-cuda-porting-guide)
8. [HIP Deprecated APIs](#8-hip-deprecated-apis)
9. [MI300/MI200 Performance Counters](#9-mi300mi200-performance-counters)
10. [GPU Occupancy](#10-gpu-occupancy)

---

## 1. AMD Matrix Instruction Calculator

**Source:** [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)

### 概述

Python 命令行工具，用于生成 AMD Radeon/Instinct 加速器矩阵乘法指令的信息。支持以下指令族：

| 指令族 | 硬件目标 |
|--------|----------|
| **MFMA** (Matrix Fused-Multiply Add) | MI100 (CDNA1), MI200 (CDNA2), MI300 (CDNA3) |
| **SMFMAC** (Sparse Matrix FMA) | CDNA3 |
| **WMMA** (Wave Matrix Multiply Accumulate) | RDNA3 |
| **SWMMAC** (Sparse WMMA) | RDNA4 |

### 支持的架构标识

```
--architecture 选项可接受：
  CDNA/CDNA1/gfx908/MI100
  CDNA2/gfx90a/MI200/MI210/MI250/MI250X
  CDNA3/gfx940/gfx941/gfx942/MI300/MI300A/MI300X/MI325X
  RDNA3/gfx1100/gfx1101/gfx1102
  RDNA4/gfx1200/gfx1201
```

### 五种查询模式

| 选项 | 功能 |
|------|------|
| `--detail-instruction` (`-d`) | 打印指令编码、寄存器用量、计算吞吐量、co-execution 能力 |
| `--get-register` (`-g`) | 给定矩阵坐标 → 返回 vector register + lane + bit range |
| `--matrix-entry` (`-m`) | 给定 register + lane → 返回矩阵坐标 |
| `--register-layout` (`-R`) | 打印整个矩阵的 register/lane 映射表 |
| `--matrix-layout` (`-M`) | 打印所有 register/lane 对应的矩阵元素 |

### 指令修饰符 (Modifiers)

| Modifier | 说明 |
|----------|------|
| `--cbsz {0..log2(blocks)}` | Control Broadcast Size — A 矩阵 block 广播控制 |
| `--abid {0..2^CBSZ-1}` | A-matrix Broadcast Identifier — 选择广播的 block |
| `--blgp {0..7}` | B-matrix Lane Group Pattern — B 矩阵 lane 间 swizzle/broadcast |
| `--opsel {#}` | Operand Select — 16-bit 值在 32-bit 寄存器中的高/低半位选择 |
| `--neg {0..7}` | Negate — 3-bit field 对 A/B/C 矩阵取反 |
| `--neg_hi {0..7}` | Negate Hi — 类似 neg，用于高位取反或 C 矩阵绝对值 |

### 关键输出示例 (`--detail-instruction`)

```
Architecture: CDNA2
Instruction: V_MFMA_F32_4X4X1F32
    Encoding: VOP3P-MAI
    VOP3P Opcode: 0x42
    Matrix Dimensions: M=4, N=4, K=1, blocks=16
    Execution statistics:
        FLOPs: 512
        Execution cycles: 8
        FLOPs/CU/cycle: 256
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 4
    Register usage:
        GPRs for A: 1, B: 1, C: 4, D: 4
        Alignment: 8 bytes
    Register capabilities:
        A/B can use ArchVGPRs: True, AccVGPRs: True
        C/D can use ArchVGPRs: True, AccVGPRs: True
    Modifiers: CBSZ/ABID=True, BLGP=True, Sparse=False
```

### Register 输出格式

`Vx{y}.z` — `x`=register offset, `y`=lane, `.z`=bit range (如 `[15:0]`, `[31:16]`, `[7:0]` 等)

---

## 2. MFMA ISA Encoding & Compiler Intrinsics

**Sources:** ROCm Blog — Matrix Core Programming on CDNA3/CDNA4; AMD matrix cores blog

### 2.1 MFMA 指令表 (CDNA3 & CDNA4)

| Type (C,D) ← (A,B) | MxNxK (CDNA3) | MxNxK (CDNA4) | Cycles |
|----------------------|---------------|---------------|--------|
| FP64 ← FP64 | 16x16x4 | 16x16x4 | 64 |
| FP32 ← FP32 | 32x32x2, 16x16x4 | 同左 | 64/32 |
| FP32 ← FP16/BF16 | 32x32x8, 16x16x16 | +16x16x32, 32x32x16 | 32/16 |
| FP32 ← FP8 | 16x16x32, 32x32x16 | 同左 | 16/32 |
| FP32 ← FP8/FP6/FP4 | — | 16x16x128, 32x32x64 | 16-64 |
| FP32 ← MXFP8/6/4 (block-scaled) | — | 16x16x128, 32x32x64 | 16-64 |

### 2.2 理论峰值性能

**MI325X (CDNA3):**

| Type | Peak | vs FP32 |
|------|------|---------|
| Matrix FP64 | 163.4 TF | 1x |
| Matrix FP32 | 163.4 TF | 1x |
| Matrix FP16 | 1307.4 TF | ~8x |
| Matrix FP8 | 2614.9 TF | ~16x |

**MI355X (CDNA4):**

| Type | Peak | vs FP32 |
|------|------|---------|
| Matrix FP64 | 78.6 TF | ~0.5x |
| Matrix FP16 | 2.5 PF | ~16x |
| Matrix FP8 | 5 PF | ~32x |
| Matrix FP6/FP4 | 10 PF | ~64x |

**峰值计算公式:**
```
TFLOP/s = 2*M*N*K * num_matrix_cores * (max_engine_clock_MHz / cycle_count) / 10^6
```

### 2.3 Compiler Intrinsic 语法

**经典 MFMA:**
```cpp
d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

**Block-Scaled MFMA (CDNA4 only):**
```cpp
d_reg = __builtin_amdgcn_mfma_scale_f32_MxNxK_f8f6f4(
    a_reg, b_reg, c_reg,
    Atype,      // 0=E4M3, 1=E5M2, 2=E2M3, 3=E3M2, 4=E2M1
    Btype,      // same encoding
    OPSEL_A, scale_a,
    OPSEL_B, scale_b
);
```

### 2.4 低精度浮点类型总结

| Width | Shorthand | Exp Bias | Range | 用途 |
|-------|-----------|----------|-------|------|
| 16-bit | E5M10 (FP16) | 15 | ±65504 | 通用 |
| 16-bit | E8M7 (BF16) | 127 | ±3.39e38 | ML Training |
| 8-bit | E4M3FN (OCP) | 7 | ±448 | CDNA4 default |
| 8-bit | E4M3FNUZ | 8 | ±240 | CDNA3 default |
| 8-bit | E5M2 (OCP) | 15 | ±57344 | BF8 |
| 8-bit | E8M0 | 127 | 2^(±127) | Scale factor only |
| 6-bit | E2M3 / E3M2 | 1 / 3 | ±7.5 / ±28 | CDNA4 new |
| 4-bit | E2M1 (FP4) | 1 | ±6 | CDNA4 new |

### 2.5 MFMA 数据布局核心规则

- MFMA 是 **wavefront-level** 指令 (CDNA wave size = 64)
- 操作数分布在 wavefront 所有 lane 中，每个 thread 持有矩阵的一部分
- D = A * B + C，其中 A/B/C/D 的寄存器分布由指令决定
- 寄存器中元素的布局取决于 MxNxK 和数据类型

**数据布局示例 — `32x32x2 FP32`:**
- 每 thread 持有: A=1 elem, B=1 elem, C/D=16 elems
- A lane 映射: `lane = 4 * block + i` (对于 4x4 block 指令)

**数据布局示例 — `32x32x16 FP8`:**
- 每 thread 持有: A=8 elems, B=8 elems, C/D=16 elems

### 2.6 VOP3P-MAI Encoding (CDNA1-3)

| Field | 用途 |
|-------|------|
| Src0 | A matrix source |
| Src1 | B matrix source |
| Src2 | C matrix source |
| Vdst | D matrix destination |
| CBSZ (3-bit) | Control Broadcast Size |
| ABID (4-bit) | A-matrix Broadcast Identifier |
| BLGP (3-bit) | B-matrix Lane Group Pattern |

**BLGP 值对 B 矩阵的影响 (CDNA2):**

| blgp | 效果 |
|------|------|
| 0 | 正常布局 |
| 1 | lanes 0-31 广播到 lanes 32-63 |
| 2 | lanes 32-63 广播到 lanes 0-31 |
| 3 | 所有 lane 数据下移 16 位 |
| 4-7 | 16-lane group 广播模式 |

### 2.7 代码示例 — FP8 MFMA Kernel

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

---

## 3. rocWMMA API Reference

**Source:** [rocWMMA API Reference Guide](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html) (v2.2.0)

### 3.1 支持的 GPU 架构

| 架构族 | Wave Size | GPU IDs |
|--------|-----------|---------|
| CDNA (wave64) | 64 | gfx908, gfx90a, gfx942, gfx950 |
| RDNA (wave32) | 32 | gfx1100, gfx1101, gfx1102, gfx1200, gfx1201 |

### 3.2 核心 API 函数

```cpp
// 填充 fragment
template<typename FragT, typename DataT>
void rocwmma::fill_fragment(FragT &frag, DataT value);

// 从内存加载到 fragment (同步)
template<typename FragT, typename DataT>
void rocwmma::load_matrix_sync(FragT &frag, const DataT *data, uint32_t ldm);

// 带 layout 覆盖的加载
template<typename FragT, typename DataT>
void rocwmma::load_matrix_sync(FragT &frag, const DataT *data, uint32_t ldm, layout_t layout);

// 从 fragment 存储到内存 (同步)
template<typename FragT, typename DataT>
void rocwmma::store_matrix_sync(DataT *data, FragT const &frag, uint32_t ldm);

// 矩阵乘加: D = A * B + C
template<typename FragAccumOut, typename FragA, typename FragB, typename FragAccumIn>
void rocwmma::mma_sync(FragAccumOut &d, FragA const &a, FragB const &b, FragAccumIn &c);

// Workgroup 同步 (LDS fence)
void rocwmma::synchronize_workgroup();
```

### 3.3 Fragment 类模板

```cpp
template<typename MatrixT,      // matrix_a | matrix_b | accumulator
         uint32_t FragM,        // fragment M dimension
         uint32_t FragN,        // fragment N dimension
         uint32_t FragK,        // fragment K dimension
         typename DataT,        // 数据类型
         typename DataLayoutT,  // row_major | col_major
         typename Scheduler>    // 调度策略
class rocwmma::fragment;
```

**关键方法:**
- `fragment::height()` / `width()` — 几何尺寸
- `fragment::blockDim()` / `kDim()` — Block/K 维度
- `fragment::size()` — unpacked 元素数量
- `operator[]` — 元素访问
- `operator*()` — packed storage 访问

### 3.4 支持的数据类型组合 (精选)

| Input (A/B) | Output (C/D) | Compute | BlockM | BlockN | BlockK | CDNA |
|-------------|-------------|---------|--------|--------|--------|------|
| f16 | f32 | f32 | 16 | 16 | 16 | gfx9 全系 |
| f16 | f32 | f32 | 32 | 32 | 8 | gfx9 全系 |
| bf16 | f32 | f32 | 16 | 16 | 16 | gfx90a+ |
| i8 | i32 | i32 | 16 | 16 | 16-64 | gfx908+ |
| f8 | f32 | f32 | 16 | 16 | 32+ | gfx940+ |
| f64 | f64 | f64 | 16 | 16 | 4+ | gfx90a+ |

### 3.5 Fragment 调度策略

| Scheduler | 说明 |
|-----------|------|
| `default_schedule` | 每 wave 独立操作 |
| `coop_row_major_2d` | 2D thread block 中 wave 按 row-major 协作 |
| `coop_col_major_2d` | 2D thread block 中 wave 按 col-major 协作 |
| `coop_row_slice_2d` | 同一行的 wave 协作 |
| `coop_col_slice_2d` | 同一列的 wave 协作 |
| `single` | 仅指定 wave 参与 |

### 3.6 Thread Block 尺寸约束

| TBlock_X | TBlock_Y | 总 Waves |
|----------|----------|----------|
| WaveSize | 1 | 1 |
| WaveSize | 2 | 2 |
| WaveSize | 4 | 4 |
| WaveSize*2 | 1 | 2 |
| WaveSize*2 | 2 | 4 |
| WaveSize*4 | 1 | 4 |

最多支持 4 个 wavefront/thread block。

### 3.7 重要约束

- `load_matrix_sync` / `store_matrix_sync` 对 global memory 同步；对 LDS 需显式调用 `synchronize_workgroup()`
- Fragment 存储在 packed registers 中，**元素顺序和 locality 无保证**
- BlockM/N 值为最小推荐值，低于此值使用 padding（影响性能）
- 支持 partial fragment sizes，内部自动 pad 到最近支持的 BlockMNK

---

## 4. rocWMMA Programming Guide

**Source:** [rocWMMA Programming Guide](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html)

### 4.1 设计理念

- **Header-only C++17 库**，位于 `rocwmma` namespace
- 编程模型：**wavefront-centric** — load/store/MMA 函数假设整个 wavefront 参与
- 如果 wavefront 中有 inactive threads → **undefined behavior**
- 较大 fragment size 能更好利用内存带宽
- rocWMMA 2.0.0+ 的 `mma_sync` 自动处理 partial/large tiles

### 4.2 三层实现架构

| Layer | 功能 |
|-------|------|
| **Unit Backend** | 封装 `amdgcn_*` intrinsics，处理架构差异 |
| **Vector Operations** | 处理变长向量，展开为 unit backend 调用 |
| **Fragment Operations** | Wavefront 级别 API，将 fragment 操作转译为 vector 操作 |

### 4.3 GEMM Kernel 命名约定

| 前缀 | 含义 |
|------|------|
| PGR0/PGR1 | Global Read Prefetch: 0=无, 1=1阶段 |
| LB0/LB2 | LDS Buffer: 0=不用LDS, 2=双缓冲 |
| MP0/MP1 | MFMA Priority: 0=默认, 1=提升 |
| SB/MB | Single/Multiple output blocks per wave |
| NC/CP | Non-Cooperative/Cooperative load/store |
| BLK/WV/WG | 协作粒度: block tile / wave tile / macro tile |

### 4.4 hipRTC 兼容

rocWMMA 兼容 HIP Runtime Compilation (hipRTC) 环境，可用于运行时生成的 kernel。

---

## 5. HIP C++ Language Extensions

**Source:** [HIP C++ Language Extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html) (HIP 7.2)

### 5.1 函数限定符

| Qualifier | 执行位置 | 可组合 |
|-----------|----------|--------|
| `__host__` | Host (默认) | 不能与 `__global__` 组合 |
| `__device__` | Device | 可与 `__host__` 组合 |
| `__global__` | Device (kernel) | 返回类型必须为 `void` |

### 5.2 Kernel 参数限制

Kernel 不能使用:
- `std::initializer_list` 或 `va_list` 参数
- 可变参数
- 引用参数
- host/device 间 size 不同的类型 (如 `long double`)
- host/device 间 layout 不同的 struct

**重要:** HIP 不支持 dynamic parallelism（kernel 不能调用 kernel）

### 5.3 Kernel 启动方式

**Triple chevron syntax:**
```cpp
kernel_name<<<gridDim, blockDim, dynamicSharedMem, stream>>>(args...);
```

**hipLaunchKernelGGL macro:**
```cpp
hipLaunchKernelGGL(kernelName, gridDim, blockDim, dynamicShared, stream, args...);
```

### 5.4 关键编译宏

| Macro | 定义条件 |
|-------|----------|
| `__HIP_PLATFORM_AMD__` | amdclang++ 自动定义 |
| `__HIPCC__` | 编译 .hip 文件时 |
| `__HIP_DEVICE_COMPILE__` | Device 编译 pass (替代 `__CUDA_ARCH__`) |
| `__HIP__` | 编译 .hip 文件时 |

---

## 6. HIP Memory Management API

**Source:** [HIP Memory Management](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)

### 6.1 Device Memory 分配

| API | 说明 |
|-----|------|
| `hipMalloc(void **ptr, size_t size)` | 在 default accelerator 分配 |
| `hipExtMallocWithFlags(ptr, size, flags)` | 带 flag 分配 (Default/Finegrained/Uncached/Signal) |
| `hipMallocPitch(ptr, pitch, width, height)` | 2D pitch 分配，alignment=128 bytes |
| `hipFree(void *ptr)` | 释放，**隐式执行 hipDeviceSynchronize** |

### 6.2 Host Memory 分配

| API | 说明 |
|-----|------|
| `hipHostMalloc(ptr, size, flags)` | Page-locked (pinned) host memory |
| `hipHostAlloc(ptr, size, flags)` | 同上，flag 选项更丰富 |
| `hipHostRegister(hostPtr, size, flags)` | 注册已有 host memory 为 pinned |
| `hipHostUnregister(hostPtr)` | 取消注册 |

**hipHostMalloc flags:**

| Flag | 效果 |
|------|------|
| `hipHostAllocDefault` | 默认 pinned |
| `hipHostAllocPortable` | 所有 context 可见 |
| `hipHostAllocMapped` | 映射到 device 地址空间 |
| `hipHostAllocWriteCombined` | Write-combined |
| `hipHostAllocUncached` | Extended fine-grained access |

### 6.3 Memory Copy

| API | 方向 |
|-----|------|
| `hipMemcpy(dst, src, size, kind)` | 通用 (H2D/D2H/D2D/H2H) |
| `hipMemcpyHtoD(dst, src, size)` | Host → Device (可能更快) |
| `hipMemcpyDtoH(dst, src, size)` | Device → Host |
| `hipMemcpyDtoD(dst, src, size)` | Device → Device |
| `hipMemcpyAsync(...)` | 异步版本 |
| `hipMemcpy2D(...)` | 2D pitched copy |

### 6.4 性能关键约束

- `hipFree` 隐式触发 `hipDeviceSynchronize()`
- 跨 GPU copy 建议将 current device 设为数据物理所在的 GPU
- 未启用 peer access 时，copy 通过 host staging buffer 中转
- Host memory 注册建议 64-byte cache-line 对齐
- 同一 cache line 不同部分在不同 device 写入 → **未定义行为**

---

## 7. HIP CUDA Porting Guide

**Source:** [Porting CUDA to HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)

### 7.1 HIPIFY 工具

| 工具 | 机制 | 要求 |
|------|------|------|
| `hipify-clang` | Clang AST 解析 | 需要可编译的 CUDA + CUDA headers |
| `hipify-perl` | 正则 pattern matching | 不需要 CUDA 安装 |

**常见映射:** `cudaMalloc` → `hipMalloc`, `cudaMemcpy` → `hipMemcpy`, `cuda_runtime.h` → `hip/hip_runtime.h`

### 7.2 Module API (hipModule)

| CUDA Driver | HIP | 说明 |
|-------------|-----|------|
| `cuModuleLoad` | `hipModuleLoad` | 从文件加载 .hsaco |
| `cuModuleLoadData` | `hipModuleLoadData` | 从内存加载 |
| `cuModuleGetFunction` | `hipModuleGetFunction` | 获取 kernel function |
| `cuLaunchKernel` | `hipModuleLaunchKernel` | 启动 kernel |

**Code object 格式:** HIP-Clang 使用 `.hsaco` 格式，Fat binary 使用 `.hip_fatbin`

### 7.3 关键差异

| 项目 | CUDA | HIP |
|------|------|-----|
| Dynamic parallelism | 支持 | **不支持** |
| Context API | 显式管理 | 通过 `hipSetDevice` 简化 |
| Address space | Per-context | **Process-wide unified** |
| JIT 编译 | 支持 | `hipModuleLoadDataEx` 忽略 JIT 选项（code 已预编译） |

### 7.4 编译架构特性检测

**Device code:** 使用 `__HIP_ARCH_HAS_*__` 宏（替代 `__CUDA_ARCH__`）

**Host code:** 运行时查询 `hipGetDeviceProperties()` 或 `hipDeviceGetAttribute()`

| Feature Macro | Device Property | 说明 |
|---------------|-----------------|------|
| `__HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__` | `hasGlobalInt32Atomics` | 32-bit int atomics (global) |
| `__HIP_ARCH_HAS_DOUBLES__` | `hasDoubles` | FP64 支持 |
| `__HIP_ARCH_HAS_WARP_SHUFFLE__` | `hasWarpShuffle` | Warp shuffle |

---

## 8. HIP Deprecated APIs

**Source:** [HIP Deprecated API List](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/deprecated_api_list.html)

### 关键废弃 API 及替代

| 废弃 API | 废弃版本 | 替代方案 |
|----------|----------|----------|
| `hipMallocHost()` | ROCm 3.1 | `hipHostAlloc()` |
| `hipMemAllocHost()` | ROCm 3.1 | `hipHostAlloc()` |
| `hipBindTexture()` | ROCm 3.8 | Texture objects |
| `hipMemcpyToArray()` | ROCm 3.8 | `hipMemcpy2D()` |
| `hipProfilerStart/Stop()` | ROCm 3.0 | `roctracer` / `rocTX` |
| `hipCtx*()` 全系 | ROCm 1.9 | `hipSetDevice()` / stream API |
| `hipTexRef*()` 全系 | ROCm 4.3-6.1 | Texture objects API |

---

## 9. MI300/MI200 Performance Counters

**Source:** [MI300/MI200 Performance Counters](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html)

### 9.1 Counter 分类总览

| 类别 | 前缀 | 关键用途 |
|------|------|----------|
| Command Processor Fetcher | CPF_ | Pipeline stall 分析 |
| Command Processor Compute | CPC_ | Packet decode / TLB 分析 |
| Graphics Register Bus Manager | GRBM_ | 全局 busy/idle 监控 |
| Shader Processor Input | SPI_ | Wave dispatch / 资源瓶颈 |
| Compute Unit - Instruction Mix | SQ_INSTS_* | 指令类型统计 |
| Compute Unit - MFMA Ops | SQ_INSTS_VALU_MFMA_MOPS_* | MFMA FLOPs (单位=512) |
| Compute Unit - Wavefront | SQ_WAVES* | Wave dispatch/occupancy |
| Compute Unit - LDS | SQ_LDS_* | LDS bank conflict/stall |
| L1i / Scalar L1d | SQC_* | 指令/常量 cache hit/miss |
| Vector L1d (TCP) | TCP_* | vL1D access/hit/miss/latency |
| Texture Addressing (TA) | TA_* | Buffer/flat wavefront 处理 |
| Texture Data (TD) | TD_* | Data stall 分析 |
| L2 Cache | TCC_* | L2 access/hit/miss/bandwidth |

### 9.2 关键 MFMA 相关 Counters

| Counter | 单位 | 说明 |
|---------|------|------|
| `SQ_INSTS_MFMA` | Instr | MFMA 指令总数 |
| `SQ_INSTS_VALU_MFMA_F16` | Instr | FP16 MFMA 指令数 |
| `SQ_INSTS_VALU_MFMA_F32` | Instr | FP32 MFMA 指令数 |
| `SQ_INSTS_VALU_MFMA_F64` | Instr | FP64 MFMA 指令数 |
| `SQ_INSTS_VALU_MFMA_I8` | Instr | INT8 MFMA 指令数 |
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | FLOP (×512) | FP16 MFMA 操作数 |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | FLOP (×512) | BF16 MFMA 操作数 |
| `SQ_INSTS_VALU_MFMA_MOPS_F64` | FLOP (×512) | FP64 MFMA 操作数 |
| `SQ_VALU_MFMA_BUSY_CYCLES` | Cycles | MFMA ALU busy 周期 |

### 9.3 延迟计算公式

| 指标 | 公式 |
|------|------|
| Vector memory latency | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_VMEM` |
| LDS latency | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_LDS` |
| Scalar memory latency | `SQ_ACCUM_PREV_HIRES / SQ_INSTS_SMEM_NORM` |
| Instruction fetch latency | `SQ_ACCUM_PREV_HIRES / SQ_IFETCH` |
| Wave latency | `SQ_ACCUM_PREV_HIRES / SQ_WAVE` |

### 9.4 SPI Occupancy Stall Counters

| Counter | 说明 |
|---------|------|
| `SPI_RA_WAVE_SIMD_FULL_CSN` | Wave slot 不足导致的 SIMD stall |
| `SPI_RA_VGPR_SIMD_FULL_CSN` | VGPR 不足导致的 SIMD stall |
| `SPI_RA_SGPR_SIMD_FULL_CSN` | SGPR 不足导致的 SIMD stall |
| `SPI_RA_LDS_CU_FULL_CSN` | LDS 不足导致的 CU stall |
| `SPI_RA_BAR_CU_FULL_CSN` | Barrier 等待的 CU 数 |
| `SPI_RA_WVLIM_STALL_CSN` | WAVE_LIMIT 导致的 stall |

### 9.5 LDS Bank Conflict Counters

| Counter | 说明 |
|---------|------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict stall cycles |
| `SQ_LDS_ADDR_CONFLICT` | LDS address conflict stall cycles |
| `SQ_LDS_UNALIGNED_STALL` | Flat unaligned 操作 stall |
| `SQ_LDS_MEM_VIOLATIONS` | LDS memory violation threads |

---

## 10. GPU Occupancy

**Source:** [GPUOpen — Occupancy Explained](https://gpuopen.com/learn/occupancy-explained)

### 10.1 Occupancy 定义

```
Occupancy = 已分配 wavefronts / 最大可用 slots
```

- RDNA2+: 每 SIMD 16 slots → occupancy = assigned_waves / 16
- CDNA: 每 SIMD 10 slots (wave64)

### 10.2 Occupancy 限制因素

| 限制资源 | 说明 |
|----------|------|
| **VGPR** | Vector GPR 用量过高 → 可分配 wave 减少 |
| **SGPR** | Scalar GPR (RDNA 上固定分配，通常不限制) |
| **LDS** | Local Data Share 用量 |
| **Thread Group Size** | Compute shader 的 threadgroup 必须在同一 WGP |
| **Barriers** | 每 SIMD pair 最多 16 个 barrier |

### 10.3 理论 Occupancy 计算

```
max_waves = min(
    floor(total_VGPRs / VGPRs_per_wave),
    floor(total_LDS / LDS_per_threadgroup) * waves_per_threadgroup,
    max_wave_slots
)
occupancy = max_waves / max_wave_slots
```

**RDNA3 (RX 7900 XTX) 示例:**
- 1536 VGPRs per SIMD
- 120 VGPRs 的 shader → `1536/120 = 12.8` → 最多 12 waves → occupancy = 12/16 = 75%

### 10.4 性能影响

- Occupancy 高 → **不一定**性能好（ALU-bound workload 无收益）
- Occupancy 低 → **不一定**性能差（更多 register → 减少 spill → 可能更快）
- Occupancy 过高 + memory-bound → **cache thrashing** 可能降低性能
- 关键：occupancy 的价值在于 **latency hiding**

### 10.5 实际优化建议

1. **VGPR 限制:** 使用 RGA (Radeon GPU Analyzer) 分析 live VGPR，找到 spikes
2. **LDS 限制:** 减少 compute shader LDS 用量
3. **Thread Group Size 限制:** 减小 threadgroup 尺寸提高调度粒度
4. **Work 不足:** 重叠无依赖的 workload
5. **Launch rate 限制:** 让每个 wave 做更多工作
6. **Register spilling:** 编译器可能将 register spill 到 memory — 延迟极大增加

---

## Fetch Status Summary

| # | URL | Status | Key Content |
|---|-----|--------|-------------|
| 1 | GitHub amd_matrix_instruction_calculator | ✅ Full README | Tool usage, ISA encoding details |
| 2 | rocWMMA programmers-guide | ✅ (corrected URL) | Design concepts, GEMM kernels |
| 3 | rocWMMA API reference | ✅ | Full API signatures, data types |
| 4 | HIP cooperative groups | ❌ 404 | Endpoint moved/unavailable |
| 5 | HIP C++ language extensions | ✅ | Qualifiers, kernel launch, macros |
| 6 | HIP memory management | ✅ (corrected URL) | Full memory API |
| 7 | HIP porting guide | ✅ (corrected URL) | CUDA-to-HIP mapping |
| 8 | HIP deprecated API list | ✅ | Deprecated texture/context APIs |
| 9 | MI300/MI200 perf counters | ✅ | Full counter tables |
| 10 | GPU atomics | ❌ Timeout | — |
| 11 | Matrix instruction calculator blog | ✅ (alternate: matrix-cores-cdna) | CDNA3/4 MFMA, FP8 examples |
| 12 | Occupancy calculator blog | ✅ (alternate: GPUOpen) | Occupancy theory and tuning |
