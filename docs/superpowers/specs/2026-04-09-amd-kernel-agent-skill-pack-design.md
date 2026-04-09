# AMD Kernel Agent Skill Pack 设计文档

**日期**: 2026-04-09
**状态**: 已确认
**目标**: 通过工程化 Agent Skill + 知识库，提升现有 LLM（Claude/Qwen3 等）在 AMD kernel 优化任务上的表现

---

## 1. 项目定位

### 1.1 问题

当前 LLM 在 AMD GPU kernel 优化上表现不佳，核心原因：
- HIP/ROCm 代码在预训练数据中占比极低（<0.001%），模型缺乏内在的 AMD 优化知识
- Agent 缺少 AMD 特有的标准化优化流程指导
- 没有正确性和性能的自动化验证闭环

### 1.2 方案

构建一个**分层 Agent Skill Pack**（不训练模型），通过结构化领域知识注入 + 验证闭环，让现有 Agent 能够在 AMD GPU 上完成 kernel 优化任务。方案借鉴：
- HuggingFace Agent Skill 模式（~550 token skill 实现 1.88x 加速）
- CUDA Agent 的 SKILL.md 标准化流程
- AVO 的知识库 $\mathcal{K}$ 设计

### 1.3 约束

- **不训练模型**：纯工程化手段
- **跨平台**：同时支持 Claude Code（`.claude/skills/`）和 Cursor（`.cursor/skills/`）
- **先内部验证（Primus-Turbo），再开源发布**
- **三条路径**：Triton + HIP C++ + Composable Kernel 都要覆盖

---

## 2. 整体架构

### 2.1 路由机制

主 Skill 作为路由器，沿**两个维度**分发：编程路径（Triton/HIP/CK）× 目标硬件（MI300X/MI355X 等）。

```
用户请求
  │
  ▼
主 SKILL.md (路由层, ~400 tokens)
  │
  ├── 维度1: 编程路径路由
  │   ├── 检测到 .py / Triton 代码 ──→ triton-kernel/SKILL.md
  │   ├── 检测到 .cpp / HIP 代码   ──→ hip-kernel/SKILL.md
  │   └── 检测到 CK 模板代码       ──→ ck-kernel/SKILL.md
  │
  └── 维度2: 硬件适配路由
      ├── 用户指定 / 检测到 gfx942  ──→ references/hardware/mi300x.md
      ├── 用户指定 / 检测到 gfx950  ──→ references/hardware/mi350x.md
      └── 用户指定 MI355X           ──→ references/hardware/mi355x.md
```

Agent 在 Step 1（分析）阶段必须先确定目标硬件，再加载对应硬件参考文档。如用户未指定，通过 `rocminfo` 或 `--offload-arch` 自动检测。

### 2.2 主 Skill 职责

主 SKILL.md 不包含具体优化知识，只负责：

1. **双维路由**：根据编程路径 + 目标硬件，引导 Agent 读取对应子 Skill 和硬件参考文档
2. **通用约束**：正确性优先、版本管理（每次有效修改 git commit）、禁止回退到 `torch.nn.functional`
3. **硬件检测规则**：未指定硬件时，运行 `rocminfo | grep gfx` 自动识别，并加载对应硬件文档

### 2.3 项目结构

```
amd-kernel-skill/
├── SKILL.md                              # 主路由 Skill (~300 tokens)
│
├── skills/
│   ├── triton-kernel/SKILL.md            # Triton ROCm 专精 (~500 tokens)
│   ├── hip-kernel/SKILL.md               # HIP C++ 专精 (~500 tokens)
│   └── ck-kernel/SKILL.md                # Composable Kernel 专精 (~500 tokens)
│
├── references/
│   ├── hardware/                         # === 硬件架构层（按 GPU 型号分文档）===
│   │   ├── mi300x.md                     # MI300X (GFX942/CDNA3): 192 CU, 192GB HBM3, 5.3TB/s, wavefront=64, LDS=64KB
│   │   ├── mi355x.md                     # MI355X (GFX950/CDNA4): 架构差异、新指令、性能特征
│   │   └── hardware-comparison.md        # 跨硬件对比表 + 移植注意事项
│   ├── isa/                              # === ISA 指令级参考（深度优化用）===
│   │   ├── isa-overview.md               # ISA 速查入口：寄存器文件、指令分类、流水线模型
│   │   ├── mfma-instructions.md          # MFMA 指令全集：精度、寄存器布局、吞吐/延迟
│   │   ├── memory-instructions.md        # 内存指令：global/LDS/buffer/fence 语义
│   │   ├── valu-salu-instructions.md     # 标量/向量 ALU 指令吞吐参考
│   │   ├── register-allocation.md        # VGPR/SGPR/AGPR 预算与 occupancy 关系
│   │   ├── scheduling-pipeline.md        # 指令调度：ILP、双发射、s_waitcnt 策略
│   │   └── inline-asm-patterns.md        # HIP 内联汇编常用优化 pattern
│   ├── amd-vs-nvidia-cheatsheet.md       # AMD vs NVIDIA 关键差异速查
│   ├── rocprof-guide.md                  # rocprof 使用与输出解读
│   ├── omniperf-guide.md                 # omniperf 工作流与关键指标
│   ├── hipcc-compilation.md              # hipcc 编译选项与常见错误（含 --offload-arch 多目标）
│   ├── triton-rocm-quirks.md             # Triton ROCm 后端已知差异
│   ├── ck-programming-model.md           # CK Tile 三级抽象模型
│   ├── ck-tile-tuning.md                 # CK tile size 调优方法与配置表（按硬件分节）
│   ├── aiter-ops-reference.md            # AITER 可用算子列表与 API
│   ├── hip-intrinsics.md                 # AMD GPU 内建函数分类速查
│   ├── optimization-patterns.md          # 通用优化模式清单（含硬件相关 pattern 标注）
│   ├── advanced-optimization.md          # 极限优化技术（瓶颈期榨干硬件性能）
│   ├── common-mistakes.md                # Agent 常犯的 AMD 特有错误
│   └── kernel-recipes.md                 # 常见算子参考实现骨架
│
├── templates/
│   ├── triton_kernel_template.py         # Triton kernel 骨架
│   ├── hip_kernel_template.cpp           # HIP kernel + PyTorch 绑定骨架
│   └── benchmark_template.py             # 通用 benchmark 脚本骨架
│
├── scripts/
│   ├── verify_correctness.py             # 正确性验证（多输入 + 精度分级）
│   └── benchmark_kernel.py               # 性能测量（预热 + 多次取中位数）
│
└── agent_output/                         # Agent 迭代记录（自动生成）
    └── <OP>/                             # 按算子名分目录
        └── <backend>/                    # 按编程路径分目录（triton/hip/ck）
            ├── round-1/                  # 每轮迭代的完整记录
            │   ├── kernel.py             #   代码快照
            │   ├── correctness.log       #   精度验证结果
            │   ├── benchmark.log         #   性能数据
            │   ├── profiling.log         #   profiling 摘要
            │   └── summary.md            #   本轮详设/调试记录
            ├── round-2/ ...
            ├── performance_trend.md      # 跨轮次性能趋势
            └── final_report.md           # 最终详设（成功）或复盘（失败）
```

---

## 3. 三条子 Skill 设计

### 3.1 通用五步流程

三条路径共享相同的高层流程，具体工具和命令不同：

| 步骤 | 目标 | 输出 |
|------|------|------|
| Step 0: 硬件识别 | 确定目标硬件（用户指定或 `rocminfo` 检测），加载对应 `references/hardware/` 文档 | 目标硬件型号 + 编译 arch + 推荐配置 |
| Step 1: 分析 | 运行 baseline benchmark + profiling 识别瓶颈 | 瓶颈类型判断（内存/计算/占用率/启动开销） |
| Step 2: 实现 | 编写 kernel + binding | 可编译的 kernel 代码 |
| Step 3: 验证 | 正确性 + 性能测量 | PASS/FAIL + speedup 数据 |
| Step 4: 迭代 | 根据反馈优化直到达标；进入瓶颈期时查阅 `references/advanced-optimization.md`，按瓶颈类型选择极限优化技术 | git commit + 性能记录 |
| Step 5: 知识沉淀 | 成功→设计文档，失败→复盘总结，提炼通用 pattern 回填 references/ | knowledge_base/ 新文档 + references/ 更新 |

### 3.2 Triton Kernel Skill

- **编译**：Triton ROCm 后端自动编译
- **Profiling**：`rocprof --stats`
- **关键知识**：
  - `BLOCK_SIZE` 推荐 64 的倍数（对齐 wavefront）
  - `tl.constexpr` 用于编译时常量
  - ROCm 后端不支持的特性列表（如 `tl.inline_asm_elementwise`）
  - `triton.autotune` 在 ROCm 上的配置空间差异
- **目标**：超过 `torch.compile` 5%+

### 3.3 HIP C++ Kernel Skill

- **编译**：`hipcc -O3 --offload-arch=<target>`（MI300X: `gfx942`, MI355X: `gfx950`；支持多目标：`--offload-arch=gfx942 --offload-arch=gfx950`）
- **Profiling**：`omniperf analyze` 微架构瓶颈分析
- **关键知识**：
  - Wavefront=64 vs warp=32 对循环展开、bank conflict 的影响
  - `__builtin_amdgcn_*` 内建函数（readfirstlane, ds_swizzle, mfma）
  - LDS 64KB/CU 的精细管理（不同硬件代际可能不同，查阅 `references/hardware/` 确认）
  - `torch.utils.hip_extension` 绑定方式
  - **多硬件适配**：通过 `#if __gfx942__` / `#if __gfx950__` 条件编译或运行时分发实现硬件特化路径
- **目标**：超过 Triton 实现和 `torch.compile`

### 3.4 CK (Composable Kernel) Skill

- **编译**：CK 编译系统
- **Profiling**：`rocprof` + CK 内置 benchmark
- **关键知识**：
  - TilePartitioner → TileScheduler → TilePipeline 三级抽象
  - Tile size 调优（block_m, block_n, block_k 组合空间）
  - CK 已有高性能模板的使用与扩展
- **目标**：找到最优 tile 配置，组装新融合 kernel

---

## 4. 知识库设计

### 4.1 五层结构

| 层次 | 内容 | 文档数 |
|------|------|--------|
| **硬件架构** | MI300X/MI355X 独立文档 + 跨硬件对比表 + AMD vs NVIDIA 差异速查 | 4 |
| **ISA 指令级** | ISA 总览、MFMA 指令、内存指令、ALU 指令、寄存器分配、指令调度、内联汇编 pattern | 7 |
| **工具链** | rocprof、omniperf、hipcc（含多目标编译）、Triton ROCm quirks | 4 |
| **库与 API** | CK 编程模型、CK tile 调优（按硬件分节）、AITER、HIP 内建函数 | 4 |
| **优化模式** | 通用优化 pattern、极限优化技术、常见错误、算子参考实现 | 4 |

### 4.2 设计原则

1. **精炼**：每个文档 500-1500 tokens，Agent 按需 grep/glob 读取
2. **可执行**：优先收录"遇到 X 瓶颈 → 用 Y 方法解决"的 pattern
3. **持续更新**：实际迭代中的新发现回填到对应文档

### 4.3 多硬件适配设计

不同 AMD GPU 代际在架构参数、指令集、最优配置上存在显著差异，Agent 必须感知目标硬件并加载对应知识。

#### 硬件参数对比（收录在 `references/hardware/hardware-comparison.md`）

| 参数 | MI300X (GFX942/CDNA3) | MI355X (GFX950/CDNA4) |
|------|----------------------|----------------------|
| CU 数量 | 192 | 待确认 |
| HBM 容量 | 192GB HBM3 | 288GB HBM3E |
| 内存带宽 | 5.3 TB/s | 待确认 |
| Wavefront | 64 | 64 |
| LDS / CU | 64KB | 待确认 |
| Matrix Core | MFMA (CDNA3) | MFMA (CDNA4, 新指令) |
| offload-arch | `gfx942` | `gfx950` |

#### 硬件感知的影响点

| 影响点 | 说明 | Agent 行为 |
|--------|------|-----------|
| **编译目标** | `--offload-arch` 必须匹配目标 GPU | 从硬件文档中读取正确的 arch 值 |
| **Tile size / Block size** | 最优配置因 CU 数量、LDS 大小、内存带宽不同而不同 | 查阅 `ck-tile-tuning.md` 中对应硬件的推荐值 |
| **新指令可用性** | CDNA4 可能引入新的 MFMA 变体或内存指令 | 查阅对应硬件文档，决定是否使用新指令 |
| **Profiling 解读** | 同一指标（如 occupancy）的合理范围因硬件不同 | 查阅对应硬件文档中的性能阈值参考 |
| **性能基线** | 同一算子在不同硬件上的绝对性能和瓶颈类型不同 | benchmark 结果必须标注目标硬件型号 |

#### 每个硬件文档的标准结构

```markdown
# MI300X (GFX942 / CDNA3) 优化指南

## 核心参数速查表
（CU、HBM、带宽、LDS、wavefront、Matrix Core 指令）

## 编译
hipcc --offload-arch=gfx942
Triton: 自动检测

## 推荐配置
- Triton BLOCK_SIZE: 通常 128 或 256
- CK tile: GEMM 推荐 block_m=256, block_n=128, block_k=64（BF16）
- HIP: thread block 推荐 256 threads

## 性能阈值参考
- Occupancy > 50% 为合理
- HBM 带宽利用率 > 60% 为内存受限算子的良好水平
- MFMA 利用率 > 70% 为计算受限算子的良好水平

## 硬件特有优化技巧
（该硬件独有的优化 pattern 或注意事项）

## 已知问题
（该硬件上遇到的特定问题和 workaround）
```

#### ISA 参考资料

深度 kernel 优化需要 Agent 理解指令级行为。类比 AVO 在 Blackwell 上的优化——无分支累加器重缩放、流水线重叠、寄存器重平衡——都依赖对 PTX/SASS 指令的理解。AMD 端对应的是 AMDGPU ISA。

ISA 知识库收录在 `references/isa/`，按用途组织（非按文档原始章节）：

```
references/isa/
├── isa-overview.md               # ISA 速查入口：寄存器文件结构、指令分类、流水线模型
├── mfma-instructions.md          # Matrix Fused Multiply-Add 指令全集
│                                 #   - 支持的精度：FP16/BF16/FP8/INT8
│                                 #   - 输入/输出寄存器布局（VGPR 映射）
│                                 #   - 吞吐量/延迟（按硬件代际分）
│                                 #   - 与 NVIDIA Tensor Core MMA 指令的差异对照
├── memory-instructions.md        # 内存指令参考
│                                 #   - Global load/store（向量化宽度、合并规则）
│                                 #   - LDS 操作（ds_read/ds_write、swizzle 模式、bank conflict 规则）
│                                 #   - Buffer 指令 vs Flat 指令的选择
│                                 #   - 内存屏障与 fence（s_waitcnt 语义、lgkmcnt/vmcnt 计数器）
├── valu-salu-instructions.md     # 标量/向量 ALU 指令参考
│                                 #   - VALU：向量算术、类型转换、比较、位操作
│                                 #   - SALU：标量算术、分支控制、常量管理
│                                 #   - 指令吞吐量表（按硬件代际分）
├── register-allocation.md        # 寄存器分配指南
│                                 #   - VGPR/SGPR 预算与 occupancy 的关系
│                                 #   - 寄存器溢出（spilling）的识别与消除
│                                 #   - AGPR（Accumulation GPR）在 MFMA 中的使用
│                                 #   - 跨 warp 寄存器压力平衡（参考 AVO v33 的优化）
├── scheduling-pipeline.md        # 指令调度与流水线
│                                 #   - CDNA3/CDNA4 流水线阶段模型
│                                 #   - 指令级并行（ILP）：隐藏内存延迟的指令交错策略
│                                 #   - 双发射（dual issue）条件与限制
│                                 #   - s_waitcnt 最优插入策略（最小化 stall 同时保证正确性）
└── inline-asm-patterns.md        # HIP 内联汇编常用 pattern
                                  #   - __builtin_amdgcn_* 内建函数 → 对应 ISA 指令映射
                                  #   - 手写内联汇编的语法和约束
                                  #   - 常见优化 pattern：
                                  #     * 手动向量化 load（buffer_load_dwordx4）
                                  #     * 手动 LDS swizzle 消除 bank conflict
                                  #     * 手动 MFMA 调度消除流水线 bubble
                                  #   - 何时使用内联汇编 vs 依赖编译器
```

**ISA 知识的分层使用策略**：

| Agent 优化阶段 | 需要的 ISA 知识深度 | 对应文档 |
|---------------|-------------------|---------|
| **早期**（融合、tiling、向量化） | 低：只需知道指令分类和大致吞吐 | `isa-overview.md` |
| **中期**（occupancy 调优、内存访问优化） | 中：需理解内存指令选择和寄存器压力 | `memory-instructions.md` + `register-allocation.md` |
| **后期**（指令级调度、消除 bubble） | 高：需理解流水线模型和指令交错 | `scheduling-pipeline.md` + `inline-asm-patterns.md` |
| **MFMA 相关**（GEMM、Attention 核心循环） | 高：需理解 MFMA 寄存器布局和调度 | `mfma-instructions.md` |

**ISA 文档的编写原则**：

1. **面向优化决策，不是指令手册翻译**：每条信息都回答"这如何影响 kernel 性能"
2. **含代码示例**：每个关键 pattern 附 HIP C++ 或内联汇编示例
3. **标注硬件差异**：GFX942 vs GFX950 的指令差异用明显标记区分
4. **从 AMD ISA 文档提炼**：源自 "AMDGPU Instruction Set Architecture" 官方文档，但重组为优化导向的结构

#### 极限优化技术文档（`references/advanced-optimization.md`）

当 Agent 完成早中期优化（融合、tiling、向量化、occupancy 调优）后进入性能平台期，需要查阅此文档中的极限优化技术来突破瓶颈。

文档结构如下：

```markdown
# 极限优化技术：突破瓶颈期的最后几个百分点

## 使用时机
当 kernel 已通过正确性验证、已超过 torch.compile baseline，但距离理论峰值
仍有差距（如带宽利用率 <70% 或 MFMA 利用率 <80%）时，按以下顺序尝试。

## 1. 软件流水线与多级缓冲
### 原理
当前迭代的计算与下一迭代的数据加载重叠执行，隐藏内存延迟。
### 技术
- 双缓冲（Double Buffering）：LDS 分两半，交替加载和计算
- 三级流水线（Prefetch-Compute-Store）：加载 N+1、计算 N、写回 N-1
- 异步拷贝：利用 DMA 引擎，数据搬运与 MFMA 完全并发
### AMD 实现要点
- 需要精细的 s_waitcnt lgkmcnt/vmcnt 控制（参见 isa/scheduling-pipeline.md）
- LDS 空间翻倍，注意 64KB/CU 上限
- [示例代码]
### 预期收益
内存受限 kernel 通常可获得 10-30% 提升

## 2. Wavefront 特化（Warp Specialization）
### 原理
同一 thread block 内不同 wavefront 承担不同角色，通过 barrier 协调。
### 技术
- 计算 wavefront：专注 MFMA 矩阵乘法
- 数据搬运 wavefront：专注 global→LDS 加载
- 归约 wavefront：专注 softmax/reduction 等辅助计算
### AMD 实现要点
- wavefront=64 意味着每个 wavefront 更重，分工收益更大
- 通过 wavefront ID（threadIdx.x / 64）分配角色
- barrier 协调用 __syncthreads() 或 LDS fence
- [FA4 风格的分工示例]
### 预期收益
复杂多阶段 kernel（如 Attention）可获得 5-15% 提升

## 3. 数据布局与 Swizzle 优化
### 原理
不改算法，只改数据在内存中的排列，消除 bank conflict 和提升合并访问率。
### 技术
- LDS padding：每行末尾加 padding 消除 bank conflict（AMD 32 bank × 4B）
- Swizzled layout：ds_swizzle 指令重排线程-数据映射
- Global memory coalescing：确保 wavefront 内连续线程访问连续 128B cacheline
- SOA vs AOS：面向向量化 load（buffer_load_dwordx4）的数据布局
### AMD 实现要点
- AMD LDS bank 规则与 NVIDIA 不同，padding 量需要重新计算
- [bank conflict 检测方法 + 修复示例]
### 预期收益
消除 bank conflict 通常可获得 5-20% 提升

## 4. Occupancy vs ILP 权衡
### 原理
反直觉：故意降低 occupancy，让每个 wavefront 获得更多 VGPR，
消除 register spilling，用 ILP 而非 TLP 隐藏延迟。
### 技术
- __launch_bounds__(threads, minBlocks) 控制编译器寄存器分配
- 检测 spilling：omniperf 中 ScratchWaveslifetimeVGPR > 0 即有溢出
- 寄存器重平衡：跨 wavefront group 重分配 VGPR 预算（参考 AVO v33）
### AMD 实现要点
- MI300X 每 CU 65536 VGPR，occupancy 与 VGPR 用量的关系表
- [occupancy 计算器使用方法]
### 预期收益
消除 spilling 通常可获得 3-10% 提升

## 5. Persistent Kernel 与 Tile 调度
### 原理
kernel 只 launch 一次，内部通过原子计数器自行分配 tile，
消除反复 launch 开销 + 实现 L2 cache 友好的遍历顺序。
### 技术
- atomicAdd 全局 tile 计数器
- Swizzled tile 遍历（L 形、Z 形、Hilbert 曲线）提升 L2 hit rate
- 跨 tile 负载均衡（对不规则形状如 causal mask 尤其有用）
### AMD 实现要点
- MI300X L2 cache 较大，tile 遍历顺序影响显著
- CK 的 TileScheduler 可作为参考实现
- [persistent GEMM 示例]
### 预期收益
多次 kernel launch 场景可获得 5-15% 提升；L2 优化通常 3-8%

## 6. 混合精度策略
### 原理
超越"全用 BF16"，在计算路径不同阶段使用不同精度。
### 技术
- 输入 FP8/BF16 → MFMA 计算 → FP32 累积 → BF16 输出
- 关键中间结果（如 softmax max/sum）保持 FP32
- 利用 MFMA 指令的混合精度能力（如 fp16 输入 fp32 输出）
### AMD 实现要点
- 不同 MFMA 变体的精度组合（参见 isa/mfma-instructions.md）
- MI355X/CDNA4 可能新增精度格式，查阅对应硬件文档
- [混合精度 GEMM 累积示例]
### 预期收益
计算受限 kernel 可获得 10-30% TFLOPS 提升（取决于精度降级幅度）

## 7. 编译器对抗与引导
### 原理
在编译器自动优化不足或过度的地方手动干预。
### 技术
- #pragma unroll N：精确控制展开（过多→寄存器溢出，过少→浪费 ILP）
- __launch_bounds__：引导寄存器分配
- volatile / __builtin_nontemporal_*：绕过 cache / 阻止重排序
- 内联汇编：编译器无法生成的指令序列（最后手段，参见 isa/inline-asm-patterns.md）
### AMD 实现要点
- hipcc -save-temps 可查看生成的 ISA，验证编译器行为
- [编译器生成代码审查流程]
### 预期收益
因情况而异，通常 2-10%

## 8. L2 Cache 全局优化
### 原理
全局层面的数据复用策略，跨 tile / 跨 kernel 最大化 L2 hit rate。
### 技术
- Tile 遍历顺序影响 L2 复用（GEMM 的 K 维度尤其敏感）
- 跨 kernel fusion 保持数据在 L2 中（避免写回 HBM 再读回）
- L2 cache residency 控制（如果硬件支持 prefetch hint）
### AMD 实现要点
- MI300X L2 大小和关联度
- rocprof 中 L2 hit/miss 指标的解读
- [tile 遍历顺序对比实验方法]
### 预期收益
内存受限 kernel 通常 3-8% 提升

## 选择指南：瓶颈类型 → 优先尝试的技术

| 瓶颈类型 | 诊断指标 | 优先技术 |
|----------|---------|---------|
| HBM 带宽受限 | 带宽利用率 >80% 但性能不达标 | 软件流水线、L2 优化、数据布局 |
| LDS 受限 | LDS bank conflict 率高 | Swizzle、padding、数据布局 |
| 计算受限 | MFMA 利用率 <70% | wavefront 特化、混合精度、ILP 优化 |
| 寄存器溢出 | ScratchWaves > 0 | occupancy 调优、寄存器重平衡 |
| Launch 开销 | 小 kernel 多次 launch | Persistent kernel |
| 编译器问题 | ISA 审查发现冗余指令 | 编译器引导、内联汇编 |
```

**与 ISA 知识库的关系**：`advanced-optimization.md` 描述"做什么"和"为什么"，`references/isa/` 描述"怎么在指令级实现"。Agent 先从前者确定优化方向，再从后者查阅具体指令用法。

**与 `optimization-patterns.md` 的分工**：`optimization-patterns.md` 覆盖早中期通用优化（融合、tiling、基本向量化），`advanced-optimization.md` 专注后期瓶颈突破。两者按优化阶段分层，不重叠。

#### 知识沉淀中的硬件标注

成功/失败总结文档中必须标注目标硬件，知识库按硬件可检索：

```markdown
# RMSNorm_triton_2026-04-15.md
## 任务描述
- 算子: RMSNorm
- 路径: Triton
- **目标硬件: MI300X (GFX942)**    ← 必填
...
```

### 4.4 获取优先级

1. 从 Primus-Turbo 现有代码提取 → `kernel-recipes.md`, `ck-tile-tuning.md`
2. 从 ROCm/CK 官方文档提炼 → 硬件架构、工具链、API
3. 从 AMDGPU ISA 官方文档提炼 → `references/isa/` 全部文档（重组为优化导向结构，而非原文档的指令手册结构）
4. 从 CUDA→HIP 差异映射整理 → `amd-vs-nvidia-cheatsheet.md`, `common-mistakes.md`
5. 从实际迭代积累 → `triton-rocm-quirks.md`, `common-mistakes.md`, `inline-asm-patterns.md`

---

## 5. 验证层设计

### 5.1 三道门控

```
Agent 修改 kernel
      │
      ▼
   Gate 1: 编译门控
   hipcc / triton compile 通过？
      │ 否 → 返回编译错误，要求修复
      ▼ 是
   Gate 2: 正确性门控
   5 个随机输入 + 2 个边界输入
   atol/rtol 按精度分级：FP32(1e-5) / BF16(1e-3) / FP8(1e-1)
      │ 否 → 返回失败输入 + 最大误差
      ▼ 是
   Gate 3: 性能门控
   10 次预热 + 100 次测量取中位数
   vs baseline（torch.compile / CK / 上一次 commit）
      │ 退化 → 警告，建议回退或换策略
      ▼ 提升 → git commit 记录版本和性能
```

### 5.2 脚本

- **`verify_correctness.py`**：自动生成随机+边界输入，按 dtype 选择 atol/rtol，输出 Agent 可解析的格式化报告
- **`benchmark_kernel.py`**：结构化性能报告（kernel_ms, baseline_ms, speedup, bandwidth_utilization, TFLOPS, 历史趋势）

### 5.3 模板

- **`triton_kernel_template.py`**：含 `@triton.jit` + `@triton.autotune` 配置骨架
- **`hip_kernel_template.cpp`**：含 HIP kernel + `torch::Tensor` 接口绑定骨架
- **`benchmark_template.py`**：含预热、计时、输出格式的通用 benchmark 骨架

---

## 6. 迭代记录与知识沉淀规则

每轮 Agent 优化迭代都必须产出结构化记录，形成**可追溯的优化历史 + 自动化经验积累闭环**。

### 6.1 每轮迭代的输出目录

Agent 的每轮迭代记录输出到 `agent_output/<OP>/<backend>/<round-N>/`，形成完整的优化轨迹：

```
agent_output/
└── rmsnorm/                          # 算子名
    └── triton/                       # 编程路径（triton / hip / ck）
        ├── round-1/                  # 第 1 轮迭代
        │   ├── kernel.py             # 本轮 kernel 代码快照
        │   ├── correctness.log       # 精度验证结果（PASS/FAIL + 误差详情）
        │   ├── benchmark.log         # 性能数据（延迟、带宽利用率、TFLOPS）
        │   ├── profiling.log         # profiling 输出摘要（瓶颈分析）
        │   └── summary.md           # 本轮详设文档（成功）或调试记录（进行中）
        ├── round-2/
        │   ├── kernel.py
        │   ├── correctness.log
        │   ├── benchmark.log
        │   ├── profiling.log
        │   └── summary.md
        ├── round-3/
        │   └── ...
        ├── performance_trend.md      # 跨轮次性能变化趋势表
        └── final_report.md           # 最终总结（详设 or 复盘）
```

#### 每轮 `summary.md` 的标准结构

```markdown
# Round N: {优化方向简述}

## 目标硬件
MI300X (GFX942)

## 本轮优化策略
具体采取了什么优化手段，为什么选择这个方向

## 关键修改
- 修改了 ...，原因是 ...
- 调整了 ...，预期效果是 ...

## 精度验证结果
- 状态: PASS / FAIL
- 最大误差: ... (atol=..., rtol=...)
- 失败的输入 shape（如有）: ...

## 性能结果
| 指标 | Round N-1 | Round N | 变化 |
|------|-----------|---------|------|
| 延迟 (ms) | 1.85 | 1.52 | -17.8% |
| vs torch.compile | 1.28x | 1.56x | +0.28x |
| 带宽利用率 | 42% | 51% | +9% |

## 分析与下一步
- 本轮有效/无效的原因分析
- 下一轮的优化方向
```

#### `performance_trend.md` 跨轮次趋势表

```markdown
# RMSNorm / Triton / MI300X 性能趋势

| Round | 优化方向 | 延迟 (ms) | vs torch.compile | 带宽利用率 | 状态 |
|-------|---------|-----------|------------------|-----------|------|
| 1 | 基础 Triton 实现 | 2.34 | 0.92x | 18% | 基线 |
| 2 | 向量化 load + BLOCK=128 | 1.85 | 1.16x | 31% | +27% |
| 3 | LDS tiling | 1.52 | 1.42x | 42% | +18% |
| 4 | 双缓冲流水线 | 1.52 | 1.42x | 42% | 无效 |
| 5 | 改用 swizzled layout | 1.38 | 1.56x | 51% | +9% |
```

### 6.2 失败时的坚持规则

**核心原则：遇到失败不要第一时间放弃，多分析和调试几轮，确认方向真的走不通后再复盘。**

#### 失败处理流程

```
尝试一个优化方向
      │
      ▼
  结果不达标（编译失败 / 正确性不通过 / 性能退化）
      │
      ▼
  ┌─ 第 1 次失败：分析错误原因，在同一 round 内修复并重试
  │
  ├─ 第 2 次失败：换一种实现方式，仍然坚持同一优化方向
  │
  ├─ 第 3 次失败：查阅 references/ 中的相关文档，寻找是否有遗漏的知识
  │                （如 common-mistakes.md, isa/, advanced-optimization.md）
  │
  └─ 第 4+ 次失败且确认方向不可行：
      ├── 记录本轮调试过程到 round-N/summary.md（含每次尝试的详细分析）
      ├── 标记该方向为"已排除"
      └── 切换到下一个优化方向，开启新 round
```

#### 具体规则

| 规则 | 说明 |
|------|------|
| **编译失败** | 至少尝试修复 3 次（分析错误信息、查阅 `hipcc-compilation.md`、检查语法），不要一遇到编译错误就换方向 |
| **正确性不通过** | 分析哪些输入 shape 失败、误差有多大、是精度问题还是逻辑错误；精度问题可调 atol/rtol 或改累积精度；逻辑错误需逐步 debug |
| **性能退化** | 运行 profiling 分析退化原因（是 occupancy 降了？register spill 增了？bank conflict？），针对性修复而非直接放弃 |
| **性能无变化** | 检查是否修改真的生效（编译缓存？），profiling 确认瓶颈是否在修改的路径上 |
| **连续 3+ 轮无进展** | 此时才可判定当前方向不可行，执行复盘，切换方向 |

#### 在 Skill 中的强制表述

```markdown
## Step 4 迭代规则
- 遇到失败时：先分析根因，在同一 round 内至少尝试修复 3 次
- 查阅 references/ 寻找是否有类似问题的解决方案
- 仅当连续 3+ 轮确认方向不可行时，才执行复盘并切换方向
- 禁止第一次失败就放弃当前优化方向
```

### 6.3 最终报告：详设（成功）或 复盘（失败）

当一个算子的优化全部完成（或确认某方向不可行）时，输出 `final_report.md`：

#### 成功 → 详设文档

```markdown
# {算子名}_{路径}_{硬件}_详设.md

## 任务描述
- 算子 / 路径 / 目标硬件

## 优化历程
引用 performance_trend.md，总结关键转折点

## 最终架构设计
- 核心优化技术（哪些有效、为什么有效）
- 关键设计决策及其理由

## 最终性能
| 指标 | Baseline | 最终版本 | 提升 |
|------|----------|---------|------|
| ... | ... | ... | ... |

## 可复用经验
提炼通用 pattern，回填到 references/
```

#### 失败 → 复盘文档

```markdown
# {算子名}_{路径}_{硬件}_复盘.md

## 任务描述
同上

## 尝试过的方向
| Round | 方向 | 结果 | 失败原因 |
|-------|------|------|---------|
| 1-3 | 向量化 + tiling | 有效 | — |
| 4-6 | 双缓冲流水线 | 无效 | register spill 抵消了流水线收益 |
| 7-9 | wavefront 特化 | 无效 | 算子太简单，分工开销 > 收益 |

## 深层原因分析
为什么这个算子/方向走不通

## 教训
简洁的规则，回填到 references/common-mistakes.md

## 替代方案建议
如果未来重新尝试，应该走什么方向
```

### 6.4 知识回填规则

无论成功还是失败，`final_report.md` 中的通用经验都必须回填：

| 经验类型 | 回填目标 |
|---------|---------|
| 新的优化 pattern | `references/optimization-patterns.md` 或 `references/advanced-optimization.md` |
| 新的 tile 配置数据 | `references/ck-tile-tuning.md` |
| 新的参考实现 | `references/kernel-recipes.md` |
| 新的常见错误 | `references/common-mistakes.md` |
| 新的 Triton ROCm 坑 | `references/triton-rocm-quirks.md` |
| 新的编译错误模式 | `references/hipcc-compilation.md` |
| 新的 ISA 级 pattern | `references/isa/inline-asm-patterns.md` |

### 6.5 完整流程图

```
Agent 开始优化算子 OP / 路径 backend
      │
      ▼
  创建 agent_output/<OP>/<backend>/
      │
      ▼
  ┌── Round 1 ──────────────────────────────┐
  │  Step 0-3: 硬件识别→分析→实现→验证      │
  │  输出: round-1/{kernel, correctness,     │
  │        benchmark, profiling, summary}    │
  │  更新: performance_trend.md              │
  │                                          │
  │  失败? → 分析+修复，至少重试3次          │
  │  仍然失败? → 记录到 summary.md           │
  └──────────────────────────────────────────┘
      │
      ▼
  ┌── Round 2 ... Round N ──────────────────┐
  │  同上，每轮记录到 round-N/               │
  │  更新 performance_trend.md               │
  │                                          │
  │  连续 3+ 轮无进展?                       │
  │  → 判定方向不可行，切换优化方向           │
  └──────────────────────────────────────────┘
      │
      ▼
  达标 or 确认不可行
      │
      ├── 成功 → 输出 final_report.md（详设）
      │          提炼经验 → 回填 references/
      │
      └── 失败 → 输出 final_report.md（复盘）
                 提炼教训 → 回填 references/
```

### 6.6 在 Skill 中的强制执行

三条子 Skill 的流程更新为：

```markdown
## Step 4: 迭代（含坚持规则）
- 每轮迭代输出到 agent_output/<OP>/<backend>/round-N/
- 记录 kernel 代码、精度验证、性能数据、profiling 摘要、本轮总结
- 更新 performance_trend.md
- 遇到失败：至少尝试修复 3 次，查阅 references/，确认不可行后才切换方向
- 进入瓶颈期时查阅 references/advanced-optimization.md

## Step 5: 知识沉淀（必须执行）
- 输出 final_report.md（成功→详设，失败→复盘）
- 提炼通用经验/教训 → 回填到 references/ 对应文档
- 不允许跳过此步骤
```

---

## 7. 安装与使用

```bash
# Claude Code
cp -r amd-kernel-skill/ /path/to/project/.claude/skills/

# Cursor
cp -r amd-kernel-skill/ /path/to/project/.cursor/skills/
```

使用方式——直接向 Agent 下达任务：
- "优化 primus_turbo/triton/attention/attention_kernel.py，目标超过 torch.compile 10%"
- "用 HIP C++ 实现一个融合的 LayerNorm+SwiGLU kernel for MI300X"
- "调优 CK 的 GEMM tile 配置，找到 BF16 GEMM M=4096 N=4096 K=8192 的最优参数"

---

## 8. 实施路径

| 阶段 | 周期 | 内容 | 验证方式 |
|------|------|------|----------|
| **Phase 1** | 1 周 | 主 Skill + Triton 子 Skill + 验证脚本 + 核心知识库（硬件架构 + 差异速查） | Primus-Turbo RMSNorm/SwiGLU 上验证 Agent 能否完成优化闭环 |
| **Phase 2** | 1 周 | HIP 子 Skill + HIP 知识库（内建函数、工具链） | 同一算子对比 Triton 和 HIP 路径的产出质量 |
| **Phase 3** | 1 周 | CK 子 Skill + CK 知识库（模板编程、tile 调优） | Agent 能否基于 CK 调优或组装新融合 kernel |
| **Phase 4** | 持续 | 回填实战经验到 common-mistakes.md / triton-rocm-quirks.md | Agent 不再重复犯同样的错误 |

---

## 9. 成功标准

1. Agent 在**无人工干预**下完成"分析→实现→验证→benchmark"完整闭环
2. 正确性验证 **100% 通过**的前提下，至少 **50% 的任务**超过 `torch.compile` baseline
3. 知识库被 Agent 实际引用（通过 grep/glob 读取）而非被忽略
