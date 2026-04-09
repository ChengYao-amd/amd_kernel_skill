# 爬取数据索引

本目录保存从 ROCm 官方文档和博客爬取的原始报告及代码提取（143+ 页），供深入查阅。
结构化参考文档已从这些报告蒸馏到 `references/` 的五个子目录中。

## 使用建议

- **日常优化**：使用 `references/{hardware,isa,toolchain,libraries,optimization}/` 中的结构化文档
- **深入查阅**：当结构化文档无法回答你的具体问题时，来这里搜索原始数据
- **代码参考**：`extracted_sota_patterns.md` 包含从 CK/AITER/FP8 博客提取的 909 行生产代码

## 文件索引

### SOTA 代码提取
| 文件 | 行数 | 内容 |
|------|------|------|
| `extracted_sota_patterns.md` | 909 | CK: buffer_load_lds + swizzle + sched_barrier + double buffer + MFMA layout; AITER: RMSNorm + MoE + MLA; FP8 GEMM: 7 阶段优化完整代码 |

### P0-P3 爬取报告
| 文件 | 页数 | 主题 |
|------|------|------|
| `p0_isa_mfma_lowlevel.md` | ~10 | ISA/MFMA 编码、Matrix Calculator、rocWMMA、Cooperative Groups、GPU atomics |
| `p1_gemm_attention_tuning.md` | ~10 | hipBLASLt、rocBLAS、Flash Attention 4 种实现、Stream-K、Triton 编译 |
| `p2_system_optimization.md` | ~11 | RCCL、MIOpen、MXFP4 量化、FP8 GEMM 优化、SGLang、ROCm 7.12 |
| `p3_deep_dive_cases.md` | ~11 | DeepSeek-R1/V3、Kimi-K2.5/FlyDSL、GEAK Agent、MI355X 训练 |

### Round 1-3 爬取报告（早期轮次）
| 文件 | 页数 | 主题 |
|------|------|------|
| `rocm_crawl_report_round1.md` | 17 | MI300X 微架构、HIP API、低精度类型、性能计数器、系统调优 |
| `rocm_crawl_report_round2.md` | 27 | FP8 GEMM 优化、GPU 分区、hipBLASLt 调优、CU masking、环境变量 |
| `rocm_crawl_report_round3.md` | 14 | Triton 编译器、CK-Tile GEMM、AITER 算子、FlyDSL、FlashInfer |

## 蒸馏去向

| 爬取数据 | 蒸馏到 |
|----------|--------|
| 硬件规格、XCD 布局、分区模式 | `references/hardware/` |
| MFMA 指令、寄存器映射、NOP 规则 | `references/isa/` |
| rocprof、omniperf、hipcc、Triton | `references/toolchain/` |
| CK pipeline、AITER API、hipBLASLt、RCCL | `references/libraries/` |
| 优化 pattern、SOTA 代码、常见错误 | `references/optimization/` |
