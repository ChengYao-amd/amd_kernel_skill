# AMD Kernel Agent Skill Pack

An engineered Skill + knowledge base that enables existing LLM agents (Claude / Cursor / Qwen, etc.) to perform high-performance kernel optimization on AMD GPUs — no additional model training required.

## Core Philosophy

```
No model training — pure engineering: structured domain knowledge injection + verification loop + knowledge accumulation
```

After reading SKILL.md, the agent can:
1. Auto-detect target hardware (MI300X / MI355X) and load architecture-specific knowledge
2. Route to a specialized sub-skill by programming path (Triton / HIP C++ / CK)
3. Reference SOTA code patterns, using existing best implementations as baseline and identifying directions to surpass them
4. Follow a five-step workflow (Analyze -> Implement -> Verify -> Iterate -> Accumulate), with automatic logging each round
5. Persist through failures with at least 3 retry attempts, consulting the knowledge base for solutions

## Project Structure

```
amd-kernel-skill/
├── SKILL.md                          # Main routing skill (hardware detection -> path dispatch -> constraints)
├── README.md                         # This file
│
├── skills/                           # 3 specialized sub-skills by programming path
│   ├── triton-kernel/SKILL.md        #   Triton ROCm (autotune, matrix_instr_nonkdim, AMD passes)
│   ├── hip-kernel/SKILL.md           #   HIP C++ (MFMA intrinsics, LDS, profiling)
│   └── ck-kernel/SKILL.md            #   Composable Kernel (pipeline selection, tile configuration)
│
├── references/                       # 5-layer knowledge base (35 structured docs)
│   ├── hardware/                     #   Layer 1: Hardware architecture (MI300X/MI325X/MI355X/MI350X)
│   ├── isa/                          #   Layer 2: ISA instruction-level (MFMA, memory, registers, scheduling, inline asm)
│   ├── toolchain/                    #   Layer 3: Toolchain (profiling decision tree, rocprof, hipcc, Triton)
│   ├── libraries/                    #   Layer 4: Libraries & APIs (GEMM tuning, AITER, CK, RCCL)
│   └── optimization/                 #   Layer 5: Optimization patterns (SOTA recipes, advanced techniques, common mistakes)
│
├── scripts/                          # Verification scripts
│   ├── verify_correctness.py         #   Correctness verification (multiple inputs + dtype-graded precision)
│   └── benchmark_kernel.py           #   Performance measurement (warmup + median + structured output)
│
├── templates/                        # Code skeletons
│   ├── triton_kernel_template.py     #   Triton kernel starting point (with AMD autotune config)
│   ├── hip_kernel_template.cpp       #   HIP kernel + PyTorch binding starting point
│   └── benchmark_template.py         #   General benchmark script
│
├── agent_output/                     # Agent iteration records (auto-generated)
│   └── <OP>/<backend>/round-N/       #   Per round: kernel + correctness.log + benchmark.log + summary.md
│
├── rocm-related-pdf/                 # 9 official AMD PDFs (ISA specs, whitepapers, product guides)
├── docs/superpowers/                 # Design documents and implementation plans
│
└── vendor/                           # Reference codebases (git submodules)
    ├── composable_kernel/            #   ROCm/composable_kernel — CK pipeline/tile source
    ├── aiter/                        #   ROCm/aiter — AITER optimized operators
    ├── rocm-examples/                #   amd/rocm-examples — HIP programming patterns
    └── triton/                       #   ROCm/triton — Triton ROCm backend
```

## Knowledge Base Scale

| Metric | Value |
|--------|-------|
| Structured reference docs | 35 documents (~6800 lines) |
| SOTA kernel recipes | 7 production-grade code patterns (FP8 GEMM 2600T, RMSNorm, MoE, ping-pong...) |
| Profiling decision tree | 1 mechanical bottleneck -> action flowchart |
| Raw crawled data | 143+ pages of ROCm docs/blogs |
| Ingested PDFs | 9 (CDNA3/4 ISA + whitepapers + product guides) |
| Reference codebases | 4 submodules (CK, AITER, rocm-examples, Triton) |
| Data sources | ROCm docs, ROCm blogs, GPUOpen, AMD whitepapers, CK/AITER source analysis |

## Quick Start

### Method 1: Install Script (Recommended)

```bash
# Install to a Cursor project
./install.sh cursor /path/to/your/project

# Install to a Claude Code project
./install.sh claude /path/to/your/project

# Install to any agent configuration directory
./install.sh custom /path/to/target/skills/dir
```

### Method 2: Manual Copy

```bash
# Cursor
cp -r SKILL.md skills/ references/ scripts/ templates/ /path/to/project/.cursor/skills/amd-kernel/

# Claude Code
cp -r SKILL.md skills/ references/ scripts/ templates/ /path/to/project/.claude/skills/amd-kernel/
```

### Method 3: Symlink (Development Mode, Live Sync)

```bash
./install.sh cursor-link /path/to/your/project
```

## Usage After Installation

Simply give the agent a task:

```
"Optimize triton/attention/attention_kernel.py, target 10% faster than torch.compile, hardware MI300X"

"Implement a fused LayerNorm+SwiGLU kernel in HIP C++ for MI355X"

"Tune CK GEMM tile configuration, find optimal parameters for BF16 GEMM M=4096 N=4096 K=8192"

"Analyze profiling results — the kernel's MFMA utilization is only 30%, help find the bottleneck and optimize"
```

## Supported Target Hardware

| GPU | Architecture | offload-arch | Status |
|-----|-------------|-------------|--------|
| MI300X | CDNA3 | `gfx942` | Full support |
| MI325X | CDNA3 | `gfx942` | Full support |
| MI355X | CDNA4 | `gfx950` | Full support |
| MI350X | CDNA4 | `gfx950` | Full support |

## License

Knowledge base content is compiled from AMD public documentation, ROCm open-source projects, and official blogs.
Submodules under vendor/ each follow their respective original licenses.
