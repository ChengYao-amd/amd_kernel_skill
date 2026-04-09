# 爬取数据索引

本目录保存从 ROCm 官方文档和博客爬取的原始报告（103+ 页），按轮次组织。
知识库中的结构化参考文档已从这些报告中蒸馏更新。

## 内容索引

### Round 1 — ROCm 官方文档核心页面 (17 页)
**重点**：MI300X 微架构、HIP kernel 语言、低精度浮点类型、性能计数器、系统调优

关键页面：
- MI300X Workload Optimization（调优工作流、TunableOp、Triton max-autotune）
- MI300 Series Microarchitecture（XCD 结构、304 CU、峰值 TFLOPS 表）
- HIP C++ Language Extensions（函数修饰符、warp 交叉 lane 操作）
- Low Precision FP Types（FP4/FP6/FP8/FP16/BF16 完整 API）
- MI300/MI200 Performance Counters（100+ 计数器 + 延迟公式）
- MI350 Performance Counters（CDNA4 新计数器：FP6/FP4 MFMA、dual-issue）

→ 已蒸馏到：`hardware/`, `isa/`, `hip-intrinsics.md`, `rocprof-guide.md`

### Round 2 — 调优指南与优化博客 (27 页)
**重点**：FP8 GEMM 优化、GPU 分区、hipBLASLt 调优、CU masking、环境变量

关键页面：
- FP8 GEMM on CDNA4（1.15 → 2288 TFLOPS 优化递进）
- MI300X System Optimization（BIOS/GRUB/OS 调优、分区模式）
- ROCm Environment Variables（50+ 性能相关环境变量）
- hipBLASLt Online Tuning（106% baseline、31s 一次性开销）
- TensileLite Tuning（225% avg speedup）
- Speculative Decoding（最高 3x speedup）

→ 已蒸馏到：`advanced-optimization.md`, `hardware/mi300x.md`, `triton-rocm-quirks.md`, `common-mistakes.md`

### Round 3 — 深度技术内容 (14 页)
**重点**：Triton 编译器架构、CK-Tile GEMM、AITER 算子、FlyDSL、FlashInfer

关键页面：
- Triton Compiler on AMD（8 个 AMD 专用 TTGIR pass）
- CK-Tile GEMM Architecture（Pipeline 层级、WarpGEMM MFMA 示例）
- AITER Operators（DeepSeek-R1 性能：prefill -52%、decode -47%、throughput +100%）
- FlyDSL（Kimi-K2.5：TPOT -69%、throughput +162%）
- hipBLASLt Data Types（FP8 FNUZ vs OCP 按架构分）

→ 已蒸馏到：`ck-programming-model.md`, `aiter-ops-reference.md`, `triton-rocm-quirks.md`

### Round 4 — 补充爬取 (13 页)
**重点**：vLLM 优化、rocWMMA、HIP 内存/协作组、CK-Tile 实操、MI350 性能

关键页面：
- vLLM V1 Performance Optimization（AITER 开关、7 种 attention backend）
- rocWMMA API（wavefront 级矩阵编程抽象）
- CK-Tile Hands-On GEMM（从零构建 GEMM kernel）
- DeepSeek-R1 on MI300X（4x 性能提升、SGLang 集成）

→ 已蒸馏到：全部参考文档的最终版

## 数据源统计

| 来源类型 | 页数 | 占比 |
|----------|------|------|
| ROCm 官方文档 (docs.amd.com) | ~45 | 43% |
| ROCm 技术博客 (blogs.amd.com) | ~40 | 39% |
| AMD 产品页 / GPUOpen | ~10 | 10% |
| 代码库分析 (CK/AITER/rocm-examples) | ~8 | 8% |
| **总计** | **~103** | 100% |
