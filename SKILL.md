# AMD Kernel Optimization Skill

You are an AMD GPU kernel optimization expert. Follow this routing logic to handle kernel optimization tasks.

## Step 0: Hardware Detection

1. If user specifies target hardware, use it
2. Otherwise run: `rocminfo | grep gfx` to detect GPU
3. Load the matching reference:
   - `gfx942` → read `references/hardware/mi300x.md` (MI300X: 304 CU, 5.3 TB/s, LDS 64KB)
   - `gfx950` → read `references/hardware/mi355x.md` (MI355X: 256 CU, 8 TB/s, LDS 160KB)
4. Read `references/hardware/hardware-comparison.md` if targeting multiple GPUs
5. **Critical**: CDNA3 uses FP8 FNUZ; CDNA4 uses FP8 OCP — precision format differs!

## Step 1: Route by Programming Path

Detect the programming path and read the corresponding sub-skill:

| Signal | Sub-Skill |
|--------|-----------|
| `.py` file, `triton`, `tl.` | Read `skills/triton-kernel/SKILL.md` |
| `.cpp` file, `hip`, `__global__` | Read `skills/hip-kernel/SKILL.md` |
| CK template, `ck::`, `TilePartitioner` | Read `skills/ck-kernel/SKILL.md` |
| User specifies path | Read corresponding sub-skill |
| Unclear | Ask user which path to use |

## Step 2: Study Existing Implementations as Baseline

Before writing a kernel, study existing optimized implementations to **establish baseline and identify超越方向**:

1. Read `references/libraries/aiter-ops-reference.md` — check AITER's implementation for the target op
   - If AITER has it: **benchmark as baseline**, read its source code to understand optimization techniques used
   - Study which backend it uses (ASM / CK / Triton) and why — this reveals what the experts chose
2. Read `references/libraries/gemm-tuning-guide.md` — understand the GEMM tuning landscape (hipBLASLt/rocBLAS/TensileLite)
3. Read `references/optimization/kernel-recipes.md` — check if a SOTA code pattern exists for this op type

**Goal: learn from existing implementations, then surpass them.** Possible超越方向:
- **Fusion**: AITER ops are single-op; fuse with adjacent ops to eliminate memory round-trips
- **Precision**: use newer precision (MXFP4/FP6) or mixed-precision pipeline not yet in AITER
- **Hardware-specific**: exploit CDNA4 features (160KB LDS, 64-bank, direct L1→LDS load) if AITER targets CDNA3
- **Workload-specific**: AITER uses general configs; tune tile/block sizes for your specific shapes
- **Algorithmic**: apply domain knowledge (e.g., causal mask sparsity, custom attention pattern) that generic kernels can't exploit

## Constraints (Always Apply)

1. **Correctness first**: Never sacrifice correctness for performance
2. **Version control**: `git commit` after every successful optimization round
3. **No fallback**: Never revert to `torch.nn.functional` — optimize the custom kernel
4. **Iteration records**: Every round outputs to `agent_output/<OP>/<backend>/round-N/`
5. **Persistence on failure**: At least 3 retry attempts before abandoning a direction
6. **Knowledge accumulation**: Success → design doc; Failure → postmortem; both backfill `references/`

## Knowledge Base — 5-Layer Reference Architecture

The knowledge base is organized into 5 layers, each corresponding to a stage of kernel optimization. Read layer by layer as you progress through the optimization flow.

### Layer 1 — Hardware Architecture (`references/hardware/`)

Read at **Step 0** to understand target hardware constraints.

| File | Content | When to Read |
|------|---------|-------------|
| `hardware/mi300x.md` | MI300X/MI325X: 304 CU, 8 XCD (38 active/die), 5.3 TB/s HBM3, LDS 64KB/CU, 4MB L2/XCD, partition modes (SPX/DPX/CPX), FLOPS/clock/CU table | Target is gfx942 |
| `hardware/mi355x.md` | MI355X/MI350X: 256 CU, 8 XCD (32 active/die), 8 TB/s HBM3E, **LDS 160KB/CU (64-bank)**, QPX partition, structured sparsity, FP4/FP6, MFMA throughput doubled | Target is gfx950 |
| `hardware/hardware-comparison.md` | CDNA3 vs CDNA4 full comparison: XCD layout, IOD count, Fabric speed, Matrix Core throughput, partition modes, FP8 format differences | Porting across hardware |

### Layer 2 — ISA & Instruction Level (`references/isa/`)

Read when **optimizing hot loops** and making ISA-level decisions.

| File | Content | When to Read |
|------|---------|-------------|
| `isa/mfma-instructions.md` | **Complete MFMA instruction table** (CDNA3+CDNA4), compiler intrinsics format, FLOPS/clock/CU, FP8 FNUZ vs OCP, data layout, Matrix Instruction Calculator, CBSZ/ABID/BLGP modifiers, rocWMMA fragment mapping | Writing MFMA kernel or choosing instruction variant |
| `isa/memory-instructions.md` | Global/LDS/buffer ops, **s_waitcnt semantics** (vmcnt/lgkmcnt/expcnt), LDS bank rules (32-bank CDNA3 vs 64-bank CDNA4), **DME async transfer**, **buffer_load_lds** (global→LDS direct), L1/L2 bandwidth numbers | Optimizing memory access patterns |
| `isa/register-allocation.md` | **VGPR budget vs occupancy table**, SGPR budget, AGPR for MFMA accumulators, spill detection (`-save-temps`, `.vgpr_count`), occupancy vs ILP tradeoff decision flow, MI300X/MI355X practical thresholds | Kernel has register pressure or low occupancy |
| `isa/scheduling-pipeline.md` | ILP strategy, dual-issue rules (VALU+SALU), MFMA scheduling (64-cycle overlap), **CDNA4 MFMA dependency resolution table (ISA Table 38)** — exact NOP counts for all instruction pairs, s_waitcnt strategy | Scheduling MFMA with loads, eliminating pipeline bubbles |
| `isa/valu-salu-instructions.md` | VALU/SALU throughput, FP16 packed ops, transcendental rates (4x slower, 2x faster on CDNA4), type conversion | Instruction mix optimization |
| `isa/inline-asm-patterns.md` | `__builtin_amdgcn_*` → ISA mapping, **s_setprio** (wave priority 0-3), **sched_barrier** (0x008=MFMA, 0x004=SALU), **sched_group_barrier**, **buffer_load_lds** asm pattern, architecture-adaptive waitcnt | Implementing 8-wave ping-pong, manual scheduling, extreme optimization |
| `isa/isa-overview.md` | ISA 速查入口: register files (512 KiB VGPR/CU), instruction categories, pipeline model, when to dive into which ISA doc | First time entering ISA-level optimization |

### Layer 3 — Toolchain (`references/toolchain/`)

Read when **profiling, debugging, or compiling**.

| File | Content | When to Read |
|------|---------|-------------|
| `toolchain/profiling-decision-tree.md` | **Mechanical profiling→action guide**: Speed-of-Light → memory/compute/stall classification → per-bottleneck counter checks → concrete fix actions, with Mermaid flow chart, CDNA3 vs CDNA4 thresholds, counter formulas | **Every profiling iteration** — follow this step by step |
| `toolchain/rocprof-guide.md` | rocprofv3 usage, hardware counter collection, `counters.txt` format, MI300X key counters (SQ_WAVES, SQ_INSTS_MFMA, TCC_HIT/MISS), Perfetto timeline visualization | Collecting raw counter data |
| `toolchain/omniperf-guide.md` | ROCm Compute Profiler (formerly omniperf): `rocprof-compute profile/analyze`, roofline analysis, Speed-of-Light panel, bottleneck decision tree, metric sets | High-level automated analysis |
| `toolchain/hipcc-compilation.md` | `hipcc -O3 --offload-arch=gfx942`, multi-target build, `-save-temps` for ISA inspection, PyTorch extension build, common errors & fixes, `-ffast-math`, `-munsafe-fp-atomics` | Compilation errors or inspecting generated ISA |
| `toolchain/triton-rocm-quirks.md` | AMD Triton compiler passes (AccelerateMatmul, BlockPingpong, ConvertToBufferOps, StreamPipeline), `matrix_instr_nonkdim`, `TORCHINDUCTOR_MAX_AUTOTUNE`, `TORCH_COMPILE_DEBUG=1`, FP8 type per-arch, `MLIR_ENABLE_DUMP=1` | Debugging Triton on AMD, tuning autotune config space |

### Layer 4 — Libraries & API (`references/libraries/`)

Read when **choosing or configuring library kernels**.

| File | Content | When to Read |
|------|---------|-------------|
| `libraries/gemm-tuning-guide.md` | **GEMM 三级调优**: Pre-tuned Docker → PyTorch TunableOp → hipBLASLt/rocBLAS → TensileLite custom kernel; hipBLASLt API workflow; Stream-K; online tuning; **Attention kernel 选型**: AITER asm > CK FMHA > Triton FA > FlashInfer | Any GEMM or Attention optimization task |
| `libraries/aiter-ops-reference.md` | AITER 12-category API (MHA, GEMM, PA, Norm, MoE, Quant, RoPE, etc.), **per-op speedup** (MLA 17x, MoE 3x), FP8 arch awareness (gfx950=OCP, others=FNUZ), vLLM/SGLang integration flags, disaggregated serving | Checking if AITER already has optimized version |
| `libraries/ck-programming-model.md` | CK-Tile pipeline hierarchy: `TileGemmShape → Partitioner → Traits → Pipeline → Kernel`, 7 pipeline types (MEMORY/COMPUTE_V3-V6/ASYNC/PRESHUFFLE), 3 schedulers (Default/Intrawave/Interwave), 3 partitioners (2D/1D/SpatiallyLocal+RemapXCD) | Configuring or extending CK kernels |
| `libraries/ck-tile-tuning.md` | **Real GemmConfig tables** from CK codebase: block/warp tile sizes for 10+ configurations, FMHA tuning (hdim 64/128), pipeline selection guide (workload→pipeline→scheduler), LDS/VGPR/grid check rules | Choosing tile sizes for CK |
| `libraries/hip-intrinsics.md` | **MFMA compiler intrinsics** full format, CDNA3+CDNA4 intrinsic names, **FP8 HIP types** (`__hip_fp8_storage_t`, `fp8x8_t` vectors), **block-scaled MFMA** intrinsic (Atype/Btype encoding), cross-lane ops, math intrinsics, AMD vs NVIDIA intrinsic mapping | Writing HIP kernel with MFMA or FP8 |
| `libraries/rccl-multi-gpu-guide.md` | RCCL config (12+ env vars), MSCCL++ integration, Quick Reduce (INT8/INT6/INT4 quantized all-reduce), CPX+NPS4 setup, MI300X 8-GPU xGMI topology, communication/compute overlap | Multi-GPU optimization |

### Layer 5 — Optimization Patterns & Recipes (`references/optimization/`)

Read when **iterating on kernel performance**.

| File | Content | When to Read |
|------|---------|-------------|
| `optimization/optimization-patterns.md` | **Early/mid-stage** patterns: coalesced access, vectorized loads, LDS tiling, kernel fusion, loop unrolling, grid sizing (≥304 blocks for MI300X), thread block sizing, MIOpen Find/Immediate, MXFP4/Quark quantization | Starting optimization, first 2-3 rounds |
| `optimization/advanced-optimization.md` | **Late-stage** (plateau-breaking): software pipelining + double buffer, wavefront specialization, LDS swizzle (XOR formula), occupancy vs ILP, persistent kernel, mixed precision, compiler guidance, L2 cache optimization, **FP8 GEMM 1→2600 TFLOPS progression**, structured sparsity, direct LDS load from L1 | Performance plateaus after basic optimization |
| `optimization/kernel-recipes.md` | **SOTA code examples**: FP8 GEMM with MFMA (fp8x16 vectorized, double buffer, ~2288 TFLOPS), buffer_load_lds (global→LDS direct), LDS XOR swizzle, **8-wave ping-pong** (s_setprio + sched_barrier), RMSNorm (AITER Triton persistent), Fused MoE (XCD remap + SiLU fusion), wavefront-aware reduction | Need working code pattern to reference |
| `optimization/common-mistakes.md` | AMD-specific pitfalls: MI300X has **304 CU** (not 192), CDNA4 LDS **64-bank** (not 32), FP8 **FNUZ vs OCP** mismatch, TF32 removed from CDNA4, CDNA4 Matrix FP64 halved, QPX/NPS differences, missing `-O3`, buffer ops vs flat, `s_setprio` for scheduling | Before submitting kernel, review this checklist |
| `optimization/amd-vs-nvidia-cheatsheet.md` | CUDA→HIP migration: terminology (warp→wavefront, SM→CU, shared→LDS), API mapping, behavioral differences (warp 32→64, reduction 5→6 steps, no `__syncwarp` needed), migration checklist | Porting CUDA kernel to HIP |

### Additional Reference Files (redistributed from crawl-data)

| File | Content |
|------|---------|
| `toolchain/performance-counters.md` | Complete GPU performance counter reference (MFMA, SPI, LDS, cache, CDNA4 new counters) |
| `toolchain/system-tuning.md` | BIOS settings, OS tuning, GPU partitioning, NUMA, Docker configuration |
| `toolchain/environment-variables.md` | ROCm/HIP/hipBLASLt/RCCL/Triton/vLLM environment variables reference |
| `libraries/rocwmma-reference.md` | rocWMMA API, fragment types, data type support matrix |
| `libraries/flydsl-reference.md` | FlyDSL DSL for GPU kernels: syntax, examples, integration |
| `libraries/flashinfer-reference.md` | FlashInfer on ROCm: API, performance comparison |
| `optimization/sota-code-patterns.md` | Production code patterns from CK/AITER/FP8 GEMM (buffer_load_lds, swizzle, MFMA layouts) |
| `optimization/inference-serving.md` | vLLM, SGLang, speculative decoding, prefill-decode disaggregation |
| `optimization/case-studies.md` | DeepSeek-R1/V3, Kimi-K2.5, MI355X training, MoE optimization |
