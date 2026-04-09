# AMD Kernel Agent Skill Pack

使现有 LLM Agent（Claude / Cursor / Qwen 等）无需额外训练即可在 AMD GPU 上完成高性能 kernel 优化的工程化 Skill + 知识库。

## 核心理念

```
不训练模型，纯工程化手段：结构化领域知识注入 + 验证闭环 + 知识沉淀
```

Agent 读取 SKILL.md 后，能够：
1. 自动检测目标硬件（MI300X / MI355X），加载对应架构知识
2. 按编程路径（Triton / HIP C++ / CK）路由到专精子 Skill
3. 参考 SOTA 代码模式，以现有最优实现为 baseline 并寻找超越方向
4. 遵循五步流程（分析→实现→验证→迭代→沉淀），每轮自动记录
5. 遇到失败坚持至少 3 次重试，查阅知识库寻找解决方案

## 项目结构

```
amd-kernel-skill/
├── SKILL.md                          # 主路由 Skill（硬件检测 → 路径分发 → 约束规则）
├── README.md                         # 本文件
│
├── skills/                           # 3 条编程路径的专精 Skill
│   ├── triton-kernel/SKILL.md        #   Triton ROCm（autotune, matrix_instr_nonkdim, AMD passes）
│   ├── hip-kernel/SKILL.md           #   HIP C++（MFMA intrinsics, LDS, profiling）
│   └── ck-kernel/SKILL.md            #   Composable Kernel（pipeline 选择, tile 配置）
│
├── references/                       # 5 层知识库（28 份结构化文档 + 爬取原始数据）
│   ├── hardware/                     #   Layer 1: 硬件架构（MI300X/MI325X/MI355X/MI350X）
│   ├── isa/                          #   Layer 2: ISA 指令级（MFMA, 内存, 寄存器, 调度, 内联汇编）
│   ├── toolchain/                    #   Layer 3: 工具链（profiling 决策树, rocprof, hipcc, Triton）
│   ├── libraries/                    #   Layer 4: 库与 API（GEMM 调优, AITER, CK, RCCL）
│   ├── optimization/                 #   Layer 5: 优化模式（SOTA recipes, 高级技术, 常见错误）
│   └── crawl-data/                   #   原始爬取数据（143+ 页, 深入查阅用）
│
├── scripts/                          # 验证脚本
│   ├── verify_correctness.py         #   正确性验证（多输入 + dtype 分级精度）
│   └── benchmark_kernel.py           #   性能测量（预热 + 中位数 + 结构化输出）
│
├── templates/                        # 代码骨架
│   ├── triton_kernel_template.py     #   Triton kernel 起点（含 AMD autotune 配置）
│   ├── hip_kernel_template.cpp       #   HIP kernel + PyTorch binding 起点
│   └── benchmark_template.py         #   通用 benchmark 脚本
│
├── agent_output/                     # Agent 迭代记录（自动生成）
│   └── <OP>/<backend>/round-N/       #   每轮：kernel + correctness.log + benchmark.log + summary.md
│
├── rocm-related-pdf/                 # 9 份 AMD 官方 PDF（ISA 规格, 白皮书, 产品手册）
├── docs/superpowers/                 # 设计文档与实施计划
│
└── vendor/                           # 参考代码库（git submodule）
    ├── composable_kernel/            #   ROCm/composable_kernel — CK pipeline/tile 源码
    ├── aiter/                        #   ROCm/aiter — AITER 优化算子
    ├── rocm-examples/                #   amd/rocm-examples — HIP 编程模式
    └── triton/                       #   ROCm/triton — Triton ROCm 后端
```

## 知识库规模

| 指标 | 数值 |
|------|------|
| 结构化参考文档 | 28 份（~3400 行） |
| SOTA Kernel Recipes | 7 个生产级代码模式（FP8 GEMM 2600T, RMSNorm, MoE, ping-pong...） |
| Profiling 决策树 | 1 份机械化 bottleneck→action 流程图 |
| 原始爬取数据 | 143+ 页 ROCm 文档/博客 |
| 已读 PDF | 9 份（CDNA3/4 ISA + 白皮书 + 产品手册） |
| 参考代码库 | 4 个 submodule（CK, AITER, rocm-examples, Triton） |
| 数据来源 | ROCm docs, ROCm blogs, GPUOpen, AMD whitepapers, CK/AITER 源码分析 |

## 快速开始

### 方法 1：安装脚本（推荐）

```bash
# 安装到 Cursor 项目
./install.sh cursor /path/to/your/project

# 安装到 Claude Code 项目
./install.sh claude /path/to/your/project

# 安装到任意 Agent 配置目录
./install.sh custom /path/to/target/skills/dir
```

### 方法 2：手动复制

```bash
# Cursor
cp -r SKILL.md skills/ references/ scripts/ templates/ /path/to/project/.cursor/skills/amd-kernel/

# Claude Code
cp -r SKILL.md skills/ references/ scripts/ templates/ /path/to/project/.claude/skills/amd-kernel/
```

### 方法 3：符号链接（开发模式，实时同步）

```bash
./install.sh cursor-link /path/to/your/project
```

## 安装后使用

直接向 Agent 下达任务即可：

```
"优化 triton/attention/attention_kernel.py，目标超过 torch.compile 10%，硬件 MI300X"

"用 HIP C++ 实现一个融合的 LayerNorm+SwiGLU kernel for MI355X"

"调优 CK 的 GEMM tile 配置，找到 BF16 GEMM M=4096 N=4096 K=8192 的最优参数"

"分析 profiling 结果，kernel 的 MFMA 利用率只有 30%，帮我找到瓶颈并优化"
```

## 支持的目标硬件

| GPU | 架构 | offload-arch | 状态 |
|-----|------|-------------|------|
| MI300X | CDNA3 | `gfx942` | 完整支持 |
| MI325X | CDNA3 | `gfx942` | 完整支持 |
| MI355X | CDNA4 | `gfx950` | 完整支持 |
| MI350X | CDNA4 | `gfx950` | 完整支持 |

## 许可

知识库内容基于 AMD 公开文档、ROCm 开源项目和官方博客整理。
vendor/ 下的 submodule 各自遵循其原始许可证。
