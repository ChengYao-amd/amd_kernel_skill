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
