# hipcc 编译指南

## 基本命令

```bash
# 单目标
hipcc -O3 --offload-arch=gfx942 -o kernel.so -shared -fPIC kernel.cpp

# 多目标
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 -o kernel.so -shared -fPIC kernel.cpp

# 配合 PyTorch
hipcc -O3 --offload-arch=gfx942 \
  $(python3 -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")  \
  -shared -fPIC kernel.cpp -o kernel.so

# 保存中间文件（用于 ISA 检查）
hipcc -save-temps --offload-arch=gfx942 kernel.cpp
# 查找包含 ISA 的 *.s 文件
```

## 常用标志

| 标志 | 用途 |
|------|------|
| `-O3` | 完整优化（必须使用） |
| `--offload-arch=gfxNNN` | 目标 GPU 架构（必需） |
| `-shared -fPIC` | 构建共享库供 Python 加载 |
| `-save-temps` | 保留中间 .s（ISA）文件 |
| `-Rpass=inline` | 显示内联决策 |
| `-ffast-math` | 激进 FP 优化（可能影响精度） |
| `-munsafe-fp-atomics` | 更快的原子 FP 操作（极少情况损失精度） |

## 常见错误与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `error: unknown target CPU 'gfx942'` | ROCm 版本过旧 | 更新 ROCm 或用 `rocminfo` 检查正确 arch |
| `undefined reference to __hip_*` | 缺少 HIP runtime 链接 | 添加 `-lhip_hcc` 或用 `hipcc` 代替 `g++` |
| `error: use of undeclared identifier '__shfl_sync'` | CUDA API 在 HIP 中不可用 | 使用 `__builtin_amdgcn_ds_swizzle` 或 `__shfl` |
| `error: too few register available` | VGPR 过多 | 添加 `__launch_bounds__`，减少活跃变量 |
| Kernel 运行但结果错误 | `--offload-arch` 不匹配 | 验证 arch 与 `rocminfo` 输出一致 |
| 性能差，未使用 `-O3` | Debug 构建 | 始终使用 `-O3` 编译 |

## PyTorch Extension 构建

```python
from torch.utils.hip_extension import load

module = load(
    name="custom_kernel",
    sources=["kernel.cpp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--offload-arch=gfx942"],
)
```
