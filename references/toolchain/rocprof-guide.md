# rocprof / rocprofv3 使用指南

ROCm 上 **kernel 统计、硬件计数器、时间线 trace** 的主力工具已演进为 **ROCprofiler SDK** 提供的 **`rocprofv3`**（安装路径通常为 `/opt/rocm/bin/rocprofv3`）。旧版 `rocprof`（rocprof v1）在部分发行版中仍可并存，用于兼容老脚本；新工程应优先以官方文档中的 **rocprofv3** 为准。详见 [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)。

## 与 ROCm Systems Profiler（rocprofiler-systems）的关系

**ROCm Systems Profiler**（包/命令常写作 **`rocprofiler-systems`**，曾用名含 Omnitrace）侧重 **应用级、系统级** 采集：CPU/GPU 协同、采样、可选动态插桩、系统指标等，并可通过 **Perfetto** 做交互式 trace 浏览。它与 **`rocprofv3`** 互补：后者更贴近 **GPU kernel、HIP/HSA API、PC sampling、硬件 counter** 等低开销分析。系统级瓶颈（线程、主机侧、多进程）优先 **rocprofiler-systems**；kernel 与 counter 细节优先 **rocprofv3**。文档入口：[ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)。

> 提示：部分 ROCm 版本在 Perfetto UI 版本上有兼容性说明，若 trace 显示异常可查阅当前 ROCm 发行说明中关于 Perfetto 版本的建议。

## rocprofv3：基本能力（概念）

官方文档中 **`rocprofv3`** 典型能力包括（以当前 ROCm 文档为准）：

- 多种 **输出格式**：如 **CSV、JSON、PFTrace、OTF2、rocpd（SQLite）** 等，便于脚本与可视化工具消费。
- **运行时 trace**：如 HIP runtime、kernel dispatch、内存活动、标记（markers）等（具体选项见 `--help`）。
- **PC sampling**、**硬件计数器** 等与架构相关的采集能力。
- 支持对运行中进程按 **PID** 等方式附加（见官方 *Using rocprofv3*）。

以下为**示意**命令形态；**实际 flag 名称请以本机 `rocprofv3 --help` 与当前 ROCm 文档为准**。

```bash
# 查看可用选项与输出格式
rocprofv3 --help

# 示例：对应用做 trace / counter 采集（占位：请替换为文档中的真实子命令与参数）
rocprofv3 <trace-or-counter-options> -- python run_kernel.py
```

## 时间线 Trace 与 Perfetto UI

**rocprofv3** 可输出 **PFTrace**（Perfetto 协议缓冲）等格式，用于在 **Perfetto UI**（[https://ui.perfetto.dev](https://ui.perfetto.dev)）中查看 **时间线**：kernel 启动顺序、重叠、与 API 的对应关系等。

推荐工作流：

1. 使用 **rocprofv3** 生成 **PFTrace**（或文档说明的可转换为 PFTrace 的中间格式，例如通过 `rocpd` 转换）。
2. 在浏览器打开 **Perfetto UI**，加载生成的 trace 文件。
3. 结合 **kernel 名称**、**时间戳**、**queue** 等轨道分析重叠与空闲。

若默认输出为 **rocpd（SQLite）**，可按 [Using rocpd output format](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html) 将数据库转为 **pftrace** / **otf2** 等后再用 Perfetto 或第三方工具分析。

## MI300X（MI300 系列 / gfx942）计数器示例

MI300、MI200 系列可用计数器与指标在官方文档中有系统说明：[MI300 and MI200 series performance counters and metrics](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html)。下列名称可与 **rocprof** 系工具的 **PMC 输入文件** 或 **rocprofv3** 的计数器列表配合使用（**具体语法以当前工具 `--list-counters` / 文档为准**）。

**Shader / 指令混合（SQ）——适合判断计算 vs MFMA vs 访存：**

| 计数器（示例） | 含义摘要 |
|----------------|----------|
| `SQ_INSTS` | 发射的指令总数 |
| `SQ_INSTS_VALU` | VALU 指令（含 MFMA 计入的类别以文档为准） |
| `SQ_INSTS_MFMA` | Matrix FMA（MFMA）指令条数 |
| `SQ_INSTS_VALU_MFMA_F16` / `SQ_INSTS_VALU_MFMA_BF16` / `SQ_INSTS_VALU_MFMA_F32` / `SQ_INSTS_VALU_MFMA_F64` / `SQ_INSTS_VALU_MFMA_I8` | 按精度的 MFMA 发射分类 |
| `SQ_INSTS_VMEM_RD` / `SQ_INSTS_VMEM_WR` | 向量内存读写指令 |
| `SQ_INSTS_LDS` | LDS 指令（MI300 上 flat 是否计入与 MI200 不同，见官方表注） |
| `SQ_WAVES` | 派发到 sequencer 的 wavefront 数（含恢复等，定义见文档） |
| `SQ_VALU_MFMA_BUSY_CYCLES` | Matrix FMA ALU 忙碌周期 |

**MFMA 吞吐（MOPS 单位，文档中常与 512 对齐）：**

| 计数器（示例） | 含义摘要 |
|----------------|----------|
| `SQ_INSTS_VALU_MFMA_MOPS_F16` | F16 MFMA 操作量（文档定义单位） |
| `SQ_INSTS_VALU_MFMA_MOPS_BF16` | BF16 MFMA 操作量 |
| `SQ_INSTS_VALU_MFMA_MOPS_F32` / `F64` / `I8` | 其他精度 MFMA 操作量 |

**L2 / Texture 路径（节选，MI300 上多实例带 `[n]` 后缀）：**

| 计数器（示例） | 含义摘要 |
|----------------|----------|
| `TCC_HIT` / `TCC_MISS`（及 `_sum` 聚合形式，依工具版本） | L2 命中/未命中 |
| `TA_FLAT_READ_WAVEFRONTS[n]` | Flat 读路径上 TA 处理的 wavefront 数 |
| `TA_BUFFER_READ_WAVEFRONTS[n]` | Buffer 读 wavefront 数 |

**LDS 争用：**

| 计数器（示例） | 含义摘要 |
|----------------|----------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflict 导致的 stall 周期 |

编写 PMC 文件时，旧式 **`rocprof -i counters.txt`** 风格仍常见于教程；**rocprofv3** 可能使用 **XML/YAML/CLI** 等新配置方式，请以 [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) 为准迁移。

## 输出解读（与旧版 rocprof 一致的思想）

| 计数器/指标 | 含义 | 行动方向 |
|-------------|------|----------|
| `SQ_WAVES` | Wavefront 发射量 | Grid/block 是否过小、occupancy |
| `SQ_INSTS_VALU` | VALU 活动 | 高可能偏计算或 VALU 密集 |
| `SQ_INSTS_MFMA` / MFMA MOPS | Matrix core 使用 | GEMM 类 kernel 应显著非零 |
| `SQ_INSTS_LDS` / `SQ_LDS_BANK_CONFLICT` | LDS 使用与冲突 | 检查 bank conflict、共享内存模式 |
| L2 `HIT`/`MISS` | 缓存行为 | 命中率低则优化复用与访问模式 |

## 推荐工作流

1. 用 **rocprofv3**（或 **rocprofiler-systems**，视瓶颈层级而定）做第一轮 **时间线 / 热点 kernel** 定位。
2. 对热点 kernel 配置 **硬件计数器**，区分 **MFMA / VALU / VMEM / LDS / L2**。
3. 需要 **跨 CPU–GPU** 或 **系统资源** 视图时，叠加 **rocprofiler-systems** 与 **Perfetto**。
4. 优化后重复测量；MI300X 上注意 **多 GCD** 与 **dispatch 分布**。

## 参考链接

- [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [MI300 and MI200 performance counters and metrics](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html)
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
