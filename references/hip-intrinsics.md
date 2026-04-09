# AMD GPU HIP 与编译器 Intrinsics 参考

本文补充 **MFMA 编译器内建**、**FP8 存储类型**、**CDNA4 块缩放 MFMA** 及 **`__builtin_amdgcn_readfirstlane`** 等常用模式；跨 lane 与访存部分保留与 NVIDIA 对照。细节以 **目标 ROCm / LLVM 版本** 及 **`offload-arch`** 为准。

## MFMA：编译器 intrinsic 通用形式

LLVM/Clang 暴露的矩阵乘加内建函数常写成：

```c
d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

- **`ODType_MxNxKInDType`**：输出元素类型、**M×N×K** tile、输入元素类型编码（如 `f32_32x32x8f16`）。
- **`a_reg` / `b_reg` / `c_reg`**：向量寄存器操作数；**`d_reg`** 为累加结果（与 intrinsic 返回值对应）。
- **`cbsz`, `abid`, `blgp`**：与 **稀疏/块压缩/子块选择** 相关的立即数字段，语义依 **具体 intrinsic 与 ISA** 而定；手写实验代码需对照 **LLVM `IntrinsicsAMDGPU.td`** 与 **CDNA ISA 手册**。

## CDNA3（GFX942）常用 `__builtin_amdgcn_mfma_*` 名称（节选）

下列名称在 **CDNA3** 文档与 LLVM 中常见；**是否启用**取决于 `-mcpu`/`-target` 与架构特性。完整集合以 **LLVM 发行版** 为准。

**FP64 / FP32 累加，双精度/单精度矩阵：**

- `__builtin_amdgcn_mfma_f64_16x16x4f64`
- `__builtin_amdgcn_mfma_f32_32x32x2f32`
- `__builtin_amdgcn_mfma_f32_16x16x4f32`

**FP16 / BF16 → FP32 累加：**

- `__builtin_amdgcn_mfma_f32_32x32x8f16`
- `__builtin_amdgcn_mfma_f32_16x16x16f16`
- `__builtin_amdgcn_mfma_f32_32x32x2bf16`
- `__builtin_amdgcn_mfma_f32_16x16x2bf16`

**INT8 → INT32 累加（示例）：**

- `__builtin_amdgcn_mfma_i32_32x32x4i8`
- `__builtin_amdgcn_mfma_i32_16x16x4i8`

**FP8（FNUZ 风格，CDNA3）→ FP32 累加（示例）：**

- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`
- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8`
- `__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8`
- `__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8`
- （以及 **16×16×32** 等同类变体，见 LLVM）

## CDNA4（GFX950）扩展：更大 K 的 FP16 类 MFMA

在 CDNA3 表基础上，CDNA4 增加 **K 维更大** 的 FP16/BF16 类 tile（示例名，以 LLVM 为准）：

- `__builtin_amdgcn_mfma_f32_16x16x32f16`
- `__builtin_amdgcn_mfma_f32_32x32x16f16`
- （BF16 后缀 `bf16` 的平行变体）

## CDNA4：块缩放 MFMA — `__builtin_amdgcn_mfma_scale_f32_*_f8f6f4`

对 **FP8 / FP6 / FP4** 及 **块缩放（block-scaled）** 路径，Clang 提供形如下式的内建（**具体 tile 名与参数列表以当前 LLVM 头文件/文档为准**）：

```c
__builtin_amdgcn_mfma_scale_f32_MxNxK_f8f6f4(
    a, b, c,
    Atype, Btype,
    OPSEL_A, scale_a,
    OPSEL_B, scale_b
);
```

- **`M×N×K`**：如 **16×16×128**、**32×32×64** 等 CDNA4 披露形态。
- **`Atype` / `Btype`**：编码 **A/B 低比特格式**，常见枚举约定（与 LLVM `MFMAScaleFormats` 等一致，**以头文件为准**）：

| 值 | 含义 |
|----|------|
| 0 | E4M3（fp8） |
| 1 | E5M2（bf8） |
| 2 | E2M3（fp6） |
| 3 | E3M2（bf6） |
| 4 | E2M1（fp4） |

- **`OPSEL_A` / `OPSEL_B`**：操作数 **子字/向量 lane 选择** 等与打包相关的控制。
- **`scale_a` / `scale_b`**：与 **per-block / MXFP** 缩放因子相关的寄存器路径（与 ISA 中 scale 加载配合）。

> 提示：CDNA4 上 **OCP FP8** 与 CDNA3 **FNUZ** 的位解释不同；与 Triton/PyTorch 联用时需统一 **量化与类型**。

## FP8 HIP 类型与向量封装

存储与向量封装名称随 ROCm 演进，常见包括：

- **`__hip_fp8_storage_t`**、**`__amd_fp8_storage_t`**：8 位存储单元。
- 使用 **`vector_size`** 封装为短向量，便于在 kernel 中传递与对齐：

```cpp
using fp8x8_t = __attribute__((vector_size(8))) __hip_fp8_storage_t;
```

实际 ABI 与 intrinsic 入参宽度需与 **LLVM 内建签名** 一致；官方示例中常对 FP8 操作数使用 **显式宽化**（如 cast 到 **`long`**）以满足寄存器宽度。

## `__builtin_amdgcn_readfirstlane`（CK 等库中的常用模式）

**`__builtin_amdgcn_readfirstlane(val)`** 对应 ISA **`v_readfirstlane_b32`**：读取 **lane 0** 的标量值并在全 wavefront 广播为 **uniform** 结果。Composable Kernel（CK）等库中常用于：

- 将 **仅 lane 0 计算** 的地址、常量、表项广播到整个 wavefront；
- 减少 **VGPR 中重复标量** 与部分 **SGPR 压力**（依具体寄存器分配而定）。

典型模式：先在各 lane 算出 **`val`**，再 **`scalar = __builtin_amdgcn_readfirstlane(val);`**，后续用 **`scalar`** 参与需要 **统一偏移** 的访存或索引。注意：若 **`val`** 非 lane 0 活跃或含发散值，语义需与 **活跃掩码** 及 **WF64** 规则一致。

## 跨 Lane 操作（简表）

| Intrinsic | 用途 | ISA 提示 |
|-----------|------|----------|
| `__builtin_amdgcn_readfirstlane(val)` | lane 0 → 全 wave 广播（标量化） | `v_readfirstlane_b32` |
| `__builtin_amdgcn_readlane(val, lane)` | 读指定 lane | `v_readlane_b32` |
| `__builtin_amdgcn_ds_swizzle(val, pattern)` | Lane 重排（无 LDS 流量） | `ds_swizzle_b32` |
| `__builtin_amdgcn_mov_dpp(val, ctrl, row_mask, bank_mask, bound_ctrl)` | DPP 数据移动 | `v_mov_b32 dpp` |
| `__shfl` / `__shfl_xor` | 兼容层 shuffle | 与 wavefront 宽度 64 对齐 |

## Wavefront 归约

AMD **wavefront 宽度为 64**（非 NVIDIA 32）。**`__shfl_xor`** 归约需 **6 步**（`offset = 32,16,...,1`），而非 5 步。

```cpp
float wf_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}
```

## 内存与数学 Intrinsics（节选）

| 分类 | 示例 | 说明 |
|------|------|------|
| 非时序访存 | `__builtin_nontemporal_load` / `__builtin_nontemporal_store` | 提示绕过或弱化 cache |
| 等待计数 | `__builtin_amdgcn_s_waitcnt` | 细粒度内存/指令等待 |
| 快速数学 | `__builtin_amdgcn_rcpf`, `rsqf`, `exp2f`, `log2f` | 快速近似，精度见文档 |

## AMD vs NVIDIA（简要）

| NVIDIA | AMD 侧常见对应 |
|--------|----------------|
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)`（全 wavefront，无 mask） |
| `__syncwarp()` | wavefront 锁步，多数场景无需等价物 |
| `__ballot_sync(mask, pred)` | `__ballot(pred)`，**64 位** |

## 延伸阅读

- 仓库内更完整的 MFMA 表与峰值公式：[`isa/mfma-instructions.md`](isa/mfma-instructions.md)
- AMD ROCm Blog：*Matrix Core Programming on AMD CDNA3 and CDNA4*
- LLVM：`IntrinsicsAMDGPU.td`
