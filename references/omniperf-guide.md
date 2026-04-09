# ROCm Compute Profiler（rocprofiler-compute）指南

面向 MI 系列 GPU 的 **片上指标采集 + 报告分析** 工具 **Omniperf** 已正式更名为 **ROCm Compute Profiler**；可执行文件名为 **`rocprof-compute`**（仓库 [ROCm/rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute)）。**工作流与旧版一致**：先 **profile** 采集，再 **analyze** 查看指标；变化主要体现在 **命令名、分析入口（CLI / Web）与底层 rocprof 版本选项**。

官方文档：[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/index.html)。

## 核心特性（与官方描述对齐）

- **一次性自动化采集**：通过预置 **perfmon 输入文件** 轮询所需 **PMC**，覆盖分析报告各章节所需计数器，无需手写全套 counter 列表。见 [Profile mode — Profiling routine](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html)。
- **过滤加速**：可按 **kernel 子串**（`-k`）、**dispatch 序号**（`-d`）、**硬件报告块**（`-b`，对应报告章节/指标块）缩小采集范围。
- **独立 Roofline**：可用 `--roof-only` 只跑 roofline 相关测试；默认 profile 常包含 **roofline 基准阶段**（除非 `--no-roof`）。 empirical roofline PDF 等输出见官方 *Profile mode*。
- **分析方式**：**终端 CLI**、**Grafana 仪表盘**、**Standalone GUI**、**TUI** 等多种方式（[Analyze mode](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/mode.html)）。

## 基本工作流（新命令名）

```bash
# 1. 采集（示例：命名 workload，运行命令）
rocprof-compute profile --name workload_name -- python run_kernel.py

# 2. 命令行分析（路径指向 profile 生成的 workload 目录）
rocprof-compute analyze -p workload_name/MI300X --cli

# 3. 图形化分析（任选其一，子命令以 `rocprof-compute analyze -h` 与文档为准）
#    - Grafana 仪表盘：见官方 Analyze mode → Grafana GUI analysis
#    - Standalone GUI / TUI：见同章节
```

**说明**：

- 目标子目录名随 **SoC** 变化，常见为 **`MI200` / `MI300X` / `MI300A`** 等（见官方 [compatible accelerators](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html)）。
- 底层实际调用的 **ROCProfiler** 可能是 **rocprofv1 / rocprofv3** 等，安装包或 `--help` 中会标注 **Profiler choice**。

常用辅助命令：

```bash
rocprof-compute profile -h
rocprof-compute profile --list-metrics    # 列出报告块编号，用于 -b 过滤
rocprof-compute analyze -h
```

## Speed-of-Light（SoL）分析面板

在 **`--list-metrics`** 输出中，**System Speed-of-Light** 对应报告中的 **「光速度」类指标**：把当前 kernel 的关键吞吐（如 **VALU FLOPs、MFMA FLOPs、内存层次带宽** 等）与 **经验 roofline** 或 **峰值** 对比，用于快速判断 **算力是否吃满、是否内存墙、Matrix Core 是否被用上**。

典型子节包括（编号随版本变化，以本机 `--list-metrics` 为准）：

- **2.1 Speed-of-Light**：总览性 **GPU / 引擎** 利用与瓶颈方向。
- **2.1.0 VALU FLOPs**、**2.1.1 VALU IOPs**：非 MFMA 向量指令相关吞吐。
- **2.1.2 MFMA FLOPs (F8)** 等：低精度 Matrix Core 路径。
- 以及 **HBM/L2/LDS** 等相关带宽与利用率条目（在列表中继续编号）。

**阅读建议**：先看 **SoL** 总览，再下钻到 **MFMA / VALU / 内存层次**；与 **Roofline PDF**、`pmc_perf.csv` 交叉验证。

## 关键指标（健康范围仅供粗判）

| 面板/类别 | 指标 | 健康（示意） | 需关注（示意） |
|-----------|------|--------------|----------------|
| Speed-of-Light | GPU / 引擎有效利用 | 视 workload | 持续极低可能 launch 不足或未用上 MFMA |
| Speed-of-Light | 内存带宽利用 | 内存受限时应高 | 宣称内存限但带宽很低 → 合并度/并发问题 |
| 计算 | MFMA 利用率 | GEMM 类应高 | 过低 → tile/精度/指令未走 MFMA |
| 计算 | VALU 利用率 | 视算法 | 异常高但 MFMA 低 → 可能未用矩阵核 |
| 内存 | L2 命中率 | 越高越好（视访问模式） | 过低 → 复用差或 working set 过大 |
| 内存 | LDS bank conflict | 低 | 高 → LDS 模式或 bank 争用 |
| Occupancy | 实际 occupancy | 视寄存器/LDS | 过低 → 资源限制或 block 配置 |
| Occupancy | Scratch / spill 类指标 | 0 理想 | 非 0 → 寄存器压力过大或溢出 |

具体列名以 **analyze** 输出与 **GUI** 为准。

## 瓶颈决策树（简版）

```
GPU/SoL 显示整体利用很低？
├── 是 → launch 密度、grid 过小、同步过多、CPU 侧未喂满 GPU
└── 否 → 看内存带宽 vs 计算
    ├── 内存带宽高且接近 roofline → 内存受限：合并、向量化、tile、L2 友好
    └── 内存带宽不高 → 看 MFMA/VALU
        ├── MFMA 高 → 计算/MFMA 路径为主，可试混合精度、算法
        └── MFMA 低但 VALU/访存异常 → 指令混合、依赖、未用 MFMA，检查编译与算法映射
```

## 参考

- [Profile mode](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html)
- [Analyze mode](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/mode.html)
- [Omniperf → ROCm Compute Profiler 更名说明](https://github.com/ROCm/rocprofiler-compute/discussions/455)（讨论帖）
