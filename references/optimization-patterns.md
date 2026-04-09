# 通用优化模式（早期与中期阶段）

## 内存优化

### 1. 合并 Global Memory 访问
- 确保连续线程访问连续地址
- Wavefront（64 线程）应访问连续的 256B 区域
- Stride-1 访问模式是理想的

### 2. 向量化 Load
- 使用 `float4` / `dwordx4` 进行 128 位 load（4 倍带宽效率）
- 数据对齐到 16 字节边界
- Triton：正确设置 BLOCK_SIZE 后自动处理

### 3. LDS 使用
- 用 LDS 实现 thread block 内的数据复用
- 预算：每 CU 64KB（CU 上所有 block 共享）
- 每 block 使用更多 LDS → 更少并发 block → 更低 occupancy

## 计算优化

### 4. Kernel 融合
- 将逐元素操作与前后的 GEMM/reduction 融合
- 节省 global memory 往返
- 常见融合：Linear+Activation、Norm+Scale、Attention+Softmax

### 5. Tiling
- 将大问题分解为适合 LDS 的 tile
- Tile 大小 = 数据复用（越大）与 occupancy（越小）的平衡
- CK 特定 tile 推荐见 `ck-tile-tuning.md`

### 6. 循环展开
- HIP 用 `#pragma unroll N`，Triton 自动处理
- 甜蜜点：足够填满流水线，但不至于寄存器溢出
- 展开后检查 VGPR 数量

## Launch 优化

### 7. Grid 大小设定
- MI300X：192 CU。至少需要 192 个 block 才能充分利用
- 更多 block（2-4 倍 CU 数）有助于隐藏每 block 的变异
- 非常小的 kernel：考虑批处理或 persistent kernel

### 8. Thread Block 大小
- 默认：256（每 block 4 个 wavefront）— 良好通用选择
- 最小：64（1 个 wavefront）— 寄存器重度 kernel
- 最大：1024（16 个 wavefront）— 高 occupancy 需求

## 反模式（标注硬件相关性）

| 反模式 | 影响 | 硬件说明 |
|--------|------|---------|
| Stride-N 访问 | 带宽浪费 | 所有 AMD GPU |
| LDS > 32KB/block | MI300X 上限制为 2 block/CU | MI300X: 64KB/CU |
| 发散代码中使用 `__syncthreads` | 死锁风险 | 所有 AMD GPU |
| 假设 warp=32 | 错误的 reduction、错误的 shuffle | AMD wavefront=64 |

---

## MIOpen：Find API 与 Immediate Mode

（摘自 **MIOpen** 官方 **find-and-immediate** 说明）

| 模式 | API 形态 | 特点 | 适用 |
|------|-----------|------|------|
| **Find API** | `miopenFindConvolution*` | 编译并**基准测试**所有 **solver**，结果写入磁盘缓存 | 首次部署或追求极致性能 |
| **Immediate Mode** | `miopenConvolution*Immediate` | 查询 **FindDb**，跳过在线 **find**，启动快 | 生产环境、低延迟启动 |

**Find mode**（`MIOPEN_FIND_MODE`）常见取值：`NORMAL`（完全 **find**）、`FAST`、`HYBRID`、**`DYNAMIC_HYBRID`（默认）**、`TRUST_VERIFY` 等；默认策略多为 **FindDb** 命中则用，未命中则走轻量或跳过部分动态内核。

**Immediate** 回退可选用 **AI 启发式**（`MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK`）或加权吞吐索引，在缺失缓存时猜测较优 **solver**。

典型 **Immediate** 流程：`GetSolutionCount` → `GetSolution`（按性能排序）→ 可选 `CompileSolution` → `...Immediate` 执行。

---

## MXFP4 / MXFP6 量化工作流（AMD Quark）

**MXFP**（**OCP Microscaling**）以 **32** 元素为 **block** 共享 **E8M0 scale**，元素为 **FP4（E2M1）** 或 **FP6（E2M3/E3M2）**；**MI355X** 等对 **FP4/FP6** 有原生矩阵路径，峰值相对 **FP16** 可高约 **4×**（与格式/指令有关，以白皮书为准）。

推荐流程：**Scaling** → **Clipping** → **Rounding（RNE）**（省略 **RNE** 易导致精度明显下降）。**AMD Quark** 工具链支持 **GPTQ**、**SmoothQuant**、**Quarot**、**AutoSmoothQuant** 等，产出可对接 **vLLM**、**SGLang**；可混合 **MXFP4** / **MXFP6** 以平衡精度与压缩率。公开模型示例见 **Hugging Face** 上 **amd/** 前缀的 **MXFP4** 预览权重。

---

## GEAK HIP Agent 模式（自动优化 kernel 案例）

**GEAK** 采用 **Generator → Evaluator → Reflector** 循环：由 **LLM** 生成 **HIP** 改写，编译运行测速，失败时把日志反馈再生成。

| 案例 | Agent 加速 | 对照（人工） |
|------|------------|----------------|
| **Voxelization** | **2.07×** | **1.84×** |
| **SwiGLU** | **1.68×** | **1.30×** |

**Voxelization** 侧常见模式：**shared memory** 缓存前驱坐标、合并访问、**block** 级 **tiling**、展开循环 **ILP**、`launch_bounds` 提示占用率、早退出。

**SwiGLU** 侧：**bf16x2** 配对、**uint4** **128-bit** 向量化、**16B** 对齐检测与回退、`__expf` / `__fdividef` 等快速数学、跨元素指令交错。

更一般的结论是：在特定内核上，**agent** 生成代码可超过经验工程师，但仍需完整验证与回归；**GEMM** 尺寸启发式也可做到相对默认 **~1.28×** 而无需穷举调优。
